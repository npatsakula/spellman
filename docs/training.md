# spellman — training reference and guide

How data flows from raw datasets to a promoted model, how to extend the
pipeline, and what the current model was built from. Architecture and
feature-space details live in the [design document](design.md); the
shipped numbers live in the [README](../README.md).

Everything below runs from `train/` (a uv project — `uv sync` first).

## Pipeline map

```
sources/ adapters ──▶ cache/<name>-<options-hash>.jsonl   (write-through, replayed)
        │                        │
        ▼                        ▼
   spellman-mix ── dedup ── crc32 split ── augment (train/val only) ── cap/shuffle
        │                        data_mix/{train,val,test}.jsonl
        ▼
   spellman-train ── featurize (bucket_tokens_flat) ── train ── θ calibration
        │                        model/{model.json,model.safetensors,eval_*.tsv}
        ▼
   assess / spellman eval ── granularity ladder, confusion, error dumps
        │
        ├─ hygiene.py ──── judge-based row removal, rewrites caches in place
        └─ hard_negatives.py ── FW2 `_removed` windows labeled by the model
```

The loop is iterative: mix → train → assess → eyeball error dumps →
hygiene / hard negatives / new sources → mix again. Promoting a model
means replacing `model/` and verifying the mtime (a silent-promotion bug
once shipped a stale model while the README claimed new numbers).

## Data sources

### The registry

A source is one module in `train/sources/` with a `@register("name")`
dataclass subclassing `Dataset` whose fields become the
`--source name:key=value,...` knobs (values go through
`ast.literal_eval`, so `docs=3000` is an int, `streaming=False` a bool).
`samples()` yields `(lang_code, text)` pairs; everything else — option
validation, caching — is inherited:

```python
# train/sources/mycorpus.py
from dataclasses import dataclass
from typing import Iterator

from . import Dataset, register


@register("mycorpus")
@dataclass
class MyCorpus(Dataset):
    lang: str
    limit: int = 100_000
    min_chars: int = 20

    name = "mycorpus"

    def samples(self) -> Iterator[tuple[str, str]]:
        ...  # download/read once, filter, then:
        yield self.lang, " ".join(text.split())
```

Register it for the mixer by adding the module to the import line in
`mix.py` (`from sources import fineweb2, hf, …, mycorpus`), then:

```bash
uv run spellman-mix --out data_mix --source mycorpus:lang=udm --source ...
```

**Caching:** each instance's output lands in
`cache/<name>-<sha1(options)[:10]>.jsonl` with a `.options.json` sidecar
fingerprint. Re-mixing with unchanged options replays the cache
(columnar, via `polars.read_ndjson`); any option change rebuilds it.
Several instances of one adapter (e.g. multiple `hf` sources) never
collide. Caveat: a rebuild re-downloads the *upstream* data — rerun
`hygiene.py` after any rebuild (see below).

If the dataset already lives on the HuggingFace Hub you usually don't
need new code at all — the generic `hf` adapter is the escape hatch.

### Adapter reference

| source | options | normalizes by |
|---|---|---|
| `fineweb2` | `docs_per_lang`, `per_doc` | streams per-language FineWeb-2 configs, extracts **line-windows**: 1–5-line windows of ~20–200 chars + truncated prefixes — the training distribution matches short inference text, not whole documents. `mon` = `khk_Cyrl`; `eng` from FineWeb `sample-10BT` |
| `hf` | `repo`, `lang`, `config`, `column`, `docs`, `per_doc`, `streaming`, `seed`, `raw`, `min_chars`, `max_chars`, `where`, `files`, `cyr`, `drop_cjk` | any HF repo/config with a text column; line-window sampling by default. **Raw mode** (`raw=True`) yields each row as one already-short sample (tweet/post register) with row gates: `where=col=value` equality slice, `cyr=<ratio>` Cyrillic-letter gate (mixed-script corpora), `drop_cjk=True` (mojibake), `files=<glob>` repo-relative subset (one split/shard) without a named config. `docs` caps scanned rows (0 = all) |
| `tatoeba` | `train_per_lang` | training remainder of the Tatoeba dump; the frozen eval set (`tatoeba_eval.tsv`) is excluded verbatim and never rewritten; `srp`/`uzn` script-filtered to majority-Cyrillic |
| `leipzig` | `corpus`, `lang`, `limit`, `min_chars`, `cyr` | Leipzig/wortschatz tarballs (e.g. the CURL community crawls for under-resourced languages); sentences as-is; `cyr` = optional Cyrillic-letter gate for mixed-script releases |
| `opus` | `corpus`, `src`, `tgt`, `lang`, `limit`, `min_chars` | OPUS moses bitexts, latest version via the OPUS API; yields one aligned side |
| `csv` | `path`, `column`, `lang`, `min_chars` | local CSV/TSV, one text column, single language |
| `jsonl` | `path`, `min_chars` | pre-labeled `{"lang","text"}` rows — the output side of offline tools (hard negatives, hygiene exports) |
| `ukr_tweets` | `lang`, `limit`, `min_chars`, `cyr` | saganoren/ukr-twi-corpus, 1.85M raw tweets; Twitter's `lang=="uk"` self-label + Cyrillic gate; proper CSV parsing (tweets embed newlines) |
| `mn_social` | `lang`, `limit`, `min_chars` | ganaxy/diploma — 10k raw Mongolian news/FB/YouTube comments (`text_raw`) |
| `kazsandra` | `lang`, `limit`, `min_chars`, `cyr` | IS2AI/KazSAnDRA Kazakh reviews; only the canonical ib/valid/test zips, deduped on `custom_id` (the resampled `*_ros`/`*_rus` zips duplicate rows) |

The wild-UGC adapters and the `hf` raw-mode gates exist for the
social-media lane (the rusentitweet analogs); the researched candidate
list with licenses, access commands and per-dataset validation reports
lives in `train/WILD_UGC_CANDIDATES.md` and
`train/cache/raw/<slug>/VALIDATION.md`.

Shared normalization conventions: whitespace-flattened single-line
samples, `min_chars = 20`, and per-language caps applied at the source
level (`train_per_lang`, `limit`) or mix level (`--cap-per-lang`).

### Mixing, splitting, augmentation

`spellman-mix` merges any combination of sources:

```bash
uv run spellman-mix --out data_mix \
    --source fineweb2:docs_per_lang=3600,per_doc=4 \
    --source tatoeba:train_per_lang=8000 \
    --source hf:repo=cis-lmu/Glot500,config=tat_Cyrl,lang=tat,docs=3000 \
    --cap-per-lang 16000 --wild-augment 0.3 --short-augment 0.2
```

- **Dedup** on `(lang, text)`, first source wins, encounter order
  preserved — earlier sources can never lose a row to a later one.
- **Split assignment is content-addressed**: `crc32(lang \0 text) % 10` →
  8/1/1 train/val/test. Identical samples land in identical splits
  regardless of source order or rebuilds. crc32, not `hash()`, because
  Python string hashing is process-randomized.
- **`--wild-augment`** adds internet-noised copies (@mentions, short
  URLs, hashtags, emoji, RT prefixes, casing chaos, vowel elongation,
  Latin loans, digit runs); **`--short-augment`** adds random 1–3-word
  fragments. Applied to train/val **only** — test stays pristine, and
  wild robustness is measured on real wild referees instead. Each row's
  transform RNG is seeded from crc32 of its content: fully
  deterministic.
- **`--cap-per-lang`** seeded-subsamples each language per split so one
  dominant source cannot skew the val/test aggregates.
- The cap/shuffle stages intentionally stay in Python `random.Random`
  (their exact draw order *is* the output contract — Polars' RNG cannot
  reproduce it); bulk IO and dedup are Polars.

### Data hygiene

`train/hygiene.py` drops rows the current model contradicts, rewriting
caches **in place** (sidecars untouched, so the fingerprint stays valid):

- **Model judge** (default): top-1 differs from the label at
  `--conf` (default 0.995) → dropped, after a `MIN_TOKENS = 8` guard.
  Catches verbatim foreign text (503 English pages inside Cyrillic
  FineWeb-2 configs — the root cause of English-translatorese voting
  bul; Uzbek inside Glot500 tgk; Russian inside sah/kpv sources).
- **Script judge** (`--script`): a Cyrillic-labeled row that is
  majority-*Latin* is judged by fastText lid.176 instead — the exported
  model cannot judge those (it trained on the contamination).
- **Twin protection:** a confident prediction *inside* the gold
  language's close group is never dropped — Turkic
  {tat,bak,kaz,kir,tyv,chv,sah}, Permic {udm,mhr,kpv}, Balkan
  {bul,mkd,srp}, East Slavic {rus,ukr,bel}. `җ` in a bak-labeled row is
  Tatar contamination; `ҡ` is genuine Bashkir the model misjudges —
  dropping the latter would remove exactly the boundary examples
  training needs.

```bash
uv run python hygiene.py cache/<name>-*.jsonl [--conf 0.995] [--script] [--dry-run]
```

Rerun after any cache rebuild — a rebuild re-downloads the dirty
upstream data.

### Hard negatives

`train/hard_negatives.py` mines FineWeb-2's `_removed` subsets —
documents rejected from e.g. `udm_Cyrl` because the FW2 pipeline's LID
judged them *another* language, overwhelmingly Russian. Each removed
window is labeled by the current model and kept only when the model is
confident (≥ `--conf 0.98`), the label differs from the source language,
and it is **not** twin-protected. Output is a jsonl cache consumed via
`--source jsonl:path=cache/hard_negatives.jsonl`. The first run mined
1,706 rows — real but too few to move the residual udm→rus confusion;
scaling `--docs` (udm's removed subset alone is 1.4 GB) is the known
highest-leverage next step.

## Training

```bash
uv run spellman-train --data data_mix --out ../model \
    --log2-d 17 --dim 128 --epochs 6 --lr 0.05 --hash-stats
```

Flags: `--log2-d` (bucket count D = 2^log2_d), `--hash-id
{fmix32,murmur2,multiply_shift}`, `--seed`, `--dim`, `--epochs`,
`--batch-size` (256), `--k` (tokens per training sample, 256), `--lr`,
`--per-lang-cap` (50k train-side rebalance), `--hash-stats`, `--device`.

Details that matter:

- Featurization goes through the shared-contract module
  (`bucket_tokens_flat` — numpy-vectorized, asserted bit-identical to
  the scalar reference), so training and runtime see the same feature
  space by construction.
- AdamW with linear LR decay to zero across all epochs (fastText
  schedule). Embeddings zero-initialized (see design doc).
- **θ calibration:** θ = 5th percentile of validation confidence; the
  runtime flags detections below it as uncertain.
- **Hash A/B:** rerun with different `--hash-id` and compare val
  accuracy; `--hash-stats` prints the distinct-key chi²/dof.
- Export writes `model.json` + `model.safetensors` plus
  `eval_test.tsv`/`eval_val.tsv` for `assess`. `--store
  {f16,int8-row,int8-col,fp8-row,fp8-col}` picks the folded-table storage
  format: int8/fp8 add a `scales` tensor and a `quant` block in
  `model.json`, roughly halving the artifact. Every quantized store is
  gated at export against validation accuracy (`--quant-max-drop`,
  default 0.2pp) — the loader dequantizes, so the runtime graph never
  sees the difference. `uv run python quantize_eval.py --store int8-row
  --out /tmp/mi` rewrites an existing model into any format for offline
  comparison.

## Evaluation

- `assess` — the metrics CLI:
  `cargo run --release --bin assess -- --model model model/eval_test.tsv`.
  Lingua-style granularity ladder (accuracy on single words / pairs /
  triples / whole texts derived from the eval file itself), per-language
  P/R/F1, length buckets, top confusion pairs; `--dump-errors DIR
  --dump-per-lang N` writes most-confident-first error rows
  (gold/pred/conf/text) per gold language for manual auditing.
- `spellman eval` — the CLI's accuracy+throughput pass over the same
  TSVs (fp16 path, what inference actually scores).
- **Referees** (frozen, never trained on):
  - `train/tatoeba_eval.tsv` — out-of-domain sentences, 26 languages,
    2,000/lang cap, seeded selection;
  - `train/rusentitweet_eval.tsv` — 2,679 wild Russian tweets (78%
    containing Latin words, 61% @mentions, 20% URLs): the register
    clean corpora never show.
- `uv run python eval_fasttext.py ../model/eval_test.tsv
  tatoeba_eval.tsv` — fastText lid.176 baseline on the identical ladder
  (fragments of gold languages outside its label set are excluded).
- `benchmarks/lid-bench` — Rust-side baselines (whichlang, lingua) on
  identical sampled rows, accuracy + latency; see the README comparison
  table for methodology and results.

## Current model: recipe and results

Mixed-domain mix, 16k-per-language cap: FineWeb-2 line-windows (26
languages) + ~104k Tatoeba training sentences + per-language top-ups —
Tatar (Glot500, a 162k-sentence parallel corpus, Wikipedia, and
FineWeb-2 `tat_Latn` for the Latin-script side), Bashkir (Telegram-bot
parallel), Chuvash (community mono corpus), Tuvan (linguist-collected
web text), Kyrgyz (Sputnik news), Udmurt (ai-forever incl. chat logs +
zerpal news), Meadow Mari (literary parallel), Macedonian (real tweets),
Glot500 Tajik/Sakha, native Tajik/Sakha corpora (gated HF datasets),
Chechen (Leipzig community crawls 2017+2023, OPUS translatewiki UI
strings, NM 171k ce-ru parallel — Chechen went from weakest class to
F1 1.00), plus 1,706 FW2 `_removed` hard negatives. All caches passed
through hygiene (twin-protected, ~1.3k foreign rows removed). dim 128,
6 epochs, D = 2^17, fmix32, wild/short augmentation on train/val.

Results and history are tabulated in the [README](../README.md). Known
residuals: rus-attraction on short low-resource texts (wants hard
negatives at ~10× scale), the mkd/bul/sr continuum on marker-less
texts, Latin-script Tatar short sentences, tgk data thinness.

### Replaying this mix

Every `spellman-mix` run records its exact recipe into
`<out>/manifest.json` — the current model's recipe is
`train/data_mix2/manifest.json` (the standing FineWeb-2/Tatoeba/OPUS
recipe below plus the wild-UGC sources of
`train/WILD_UGC_CANDIDATES.md`). The pre-wild standing recipe,
reconstructed from the cache fingerprints (the Komi parallel corpus is
deliberately absent — rejected by the contamination audit):

```bash
# prerequisites (once):
#   - Tatoeba sentence dump at train/tatoeba/sentences.csv
#     (copy from a machine that has it, or the exports at
#     downloads.tatoeba.org)
#   - hf auth login, with the gates of muhtasham/tajik-corpus and
#     ailabykt/sakha-corpus-mono accepted in the browser
uv run spellman-mix --out data_mix \
    --source fineweb2:docs_per_lang=3600,per_doc=4 \
    --source tatoeba:train_per_lang=8000 \
    --source hf:repo=cis-lmu/Glot500,config=tat_Cyrl,lang=tat,docs=3000,per_doc=4 \
    --source hf:repo=cis-lmu/Glot500,config=tgk_Cyrl,lang=tgk,docs=12000,per_doc=2 \
    --source hf:repo=cis-lmu/Glot500,config=sah_Cyrl,lang=sah,docs=3000,per_doc=4 \
    --source hf:repo=cis-lmu/Glot500,config=udm_Cyrl,lang=udm,docs=14000,per_doc=2 \
    --source hf:repo=cis-lmu/Glot500,config=tyv_Cyrl,lang=tyv,docs=15000,per_doc=2 \
    --source hf:repo=muhtasham/tajik-corpus,lang=tgk,docs=8000,per_doc=2 \
    --source hf:repo=ailabykt/sakha-corpus-mono,lang=sah,docs=8000,per_doc=3 \
    --source hf:repo=wikimedia/wikipedia,config=20231101.tt,lang=tat,docs=2000,per_doc=4 \
    --source hf:repo=AigizK/tatar-russian-parallel-corpora,column=tat,lang=tat,docs=999999,per_doc=2,streaming=False \
    --source hf:repo=HuggingFaceFW/fineweb-2,config=tat_Latn,lang=tat,docs=10000,per_doc=3 \
    --source hf:repo=AigizK/bashkir-russian-parallel-corpora,column=ba,lang=bak,docs=30000,per_doc=2 \
    --source hf:repo=alexantonov/chuvash_mono,column=chv,lang=chv,docs=20000,per_doc=2 \
    --source hf:repo=Agisight/tyv-rus-200k,column=tyv,lang=tyv,docs=30000,per_doc=2 \
    --source hf:repo=the-cramer-project/Kyrgyz_News_Corpus,lang=kir,docs=8000,per_doc=3 \
    --source hf:repo=ai-forever/udmurt-corpora,lang=udm,docs=30000,per_doc=2 \
    --source hf:repo=udmurtNLP/zerpal,column=string,lang=udm,docs=20000,per_doc=3 \
    --source hf:repo=d0rj/ru-mhr-parallel,column=mhr,lang=mhr,docs=30000,per_doc=2 \
    --source hf:repo=mteb/MacedonianTweetSentimentClassification,lang=mkd,docs=999999,per_doc=1 \
    --source leipzig:corpus=che_community_2017,lang=che \
    --source leipzig:corpus=che_community_2023,lang=che \
    --source opus:corpus=translatewiki,src=ce,tgt=en,lang=che \
    --source hf:repo=NM-development/nmd-ce-ru-171k-v0,column=ce,lang=che,docs=999999,per_doc=1,streaming=False \
    --source csv:path=rusentitweet_train.csv,column=text,lang=rus \
    --source jsonl:path=cache/hard_negatives.jsonl \
    --cap-per-lang 16000 --wild-augment 0.3 --short-augment 0.2
```

On a fresh machine the caches rebuild by re-downloading everything, so
the full loop is: fetch a judge model
(`hf download vpermilp/spellman --local-dir ../model`), run the mix
once, then `hygiene.py cache/*.jsonl --script`, then
`hard_negatives.py --model ../model`, then run the same mix again
(warm caches replay instantly) and train. Rebuilds re-download the
upstream data — always rerun hygiene after one.

## End-to-end walkthrough

```bash
cd train && uv sync

# 1. mix (replays warm caches; first run downloads)
uv run spellman-mix --out data_mix \
    --source fineweb2:docs_per_lang=3600,per_doc=4 \
    --source tatoeba:train_per_lang=8000 \
    --cap-per-lang 16000 --wild-augment 0.3 --short-augment 0.2

# 2. train + export
uv run spellman-train --data data_mix --out ../model \
    --log2-d 17 --dim 128 --epochs 6 --lr 0.05 --hash-stats

# 3. evaluate
cd .. && cargo run --release --bin assess -- --model model model/eval_test.tsv
./target/release/spellman eval model/eval_test.tsv

# 4. audit errors, then iterate (hygiene / hard negatives / new sources)
cargo run --release --bin assess -- --model model train/tatoeba_eval.tsv \
    --dump-errors errors --dump-per-lang 100
```

The Rust↔Python feature parity must hold at all times:
`uv run spellman-gen-fixtures && cargo test` (run both sides after any
feature/hash change; every trained model is invalid if they drift).
