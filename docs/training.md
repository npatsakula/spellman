# spellman — training reference and guide

How data flows from raw datasets to a promoted, published model, how to
extend the pipeline, and what the current model was built from. Architecture
and feature-space details live in the [design document](design.md); the
shipped numbers live in the [README](../README.md).

The pipeline is one installable package with one CLI (`train/` is a uv
project — `uv sync` first):

```
spellman-train fetch          # build source caches (specs or a manifest recipe)
spellman-train clean          # model-judge hygiene over warm caches
spellman-train mix            # dataset: parquet shards + manifest.json
spellman-train train          # model: folded export -> model/
spellman-train publish        # dataset / model -> Hugging Face Hub
```

Every command also runs standalone (`python -m spellman_train.<module>`)
and has full `--help`. Supporting tools: `short-verify` (3-judge consensus
for the short lane), `hard-negatives` (FW2 `_removed` boundary mining),
`referee-short` (rebuild the frozen short referee), `eval-fasttext`
(baselines), `quantize` (storage-format rewrite), `gen-fixtures`
(Rust↔Python parity), `prepare-apertium` (mkd analyzer toolchain).

## Pipeline map

```
spellman_train.sources adapters ──▶ cache/<name>-<options-hash>.jsonl  (write-through, replayed)
        │                                      │
        ▼                                      ▼
   spellman-train mix ── dedup ── crc32 split ── augment (train/val only) ── cap/shuffle
        │                                      data/<mix>/data/{train,val,test}-*.parquet + manifest.json
        ▼
   spellman-train train ── featurize (bucket_tokens_flat) ── train ── θ calibration
        │                                      model/{model.json,model.safetensors,eval_*.tsv}
        ▼
   assess / spellman eval ── granularity ladder, confusion, error dumps
        │
        ├─ spellman-train clean ── judge-based row removal, rewrites caches in place
        └─ spellman-train hard-negatives ── FW2 `_removed` windows labeled by the model
        │
        ▼
   spellman-train publish dataset|model ── Hugging Face Hub (vpermilp/spellman, both repo types)
```

The loop is iterative: mix → train → assess → eyeball error dumps →
clean / hard negatives / new sources → mix again. Promoting a model means
replacing `model/` and verifying the mtime (a silent-promotion bug once
shipped a stale model while the README claimed new numbers), then
`publish model` + `publish dataset`.

## Data sources

### The registry

A source is one module in `spellman_train/sources/` with a
`@register("name")` dataclass subclassing `Dataset` whose fields become the
`--source name:key=value,...` knobs (values go through `ast.literal_eval`,
so `docs=3000` is an int, `streaming=False` a bool). `samples()` yields
`(lang_code, text)` pairs; everything else — option validation, caching —
is inherited. Registration is automatic: the registry imports every module
in the package on first use, so a new adapter is mixable the moment the
file exists:

```python
# spellman_train/sources/mycorpus.py
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

```bash
uv run spellman-train mix --out data_mix --source mycorpus:lang=udm --source ...
```

**Caching:** each instance's output lands in
`cache/<name>-<sha1(options)[:10]>.jsonl` with a `.options.json` sidecar
fingerprint. Re-mixing with unchanged options replays the cache
(columnar, via `polars.read_ndjson`); any option change rebuilds it.
Several instances of one adapter (e.g. multiple `hf` sources) never
collide. Caveat: a rebuild re-downloads the *upstream* data — rerun
`clean` after any rebuild (see below).

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
| `diverse` | `lang`, `pool_repo`+`pool_config`+`pool_docs` or `pool_file`, `budget`, `min_gain`, `min_df`, `expose_top`, `min_exposures`, `min_bands`, `min/max_chars`, `max_candidates`, `seed`, `algo`, `norm` (auto) | lexical-diversity generator: first-come lemma-coverage selection over a pool (any HF repo/config, materialized once under cache/, or a built cache file). Lemmas come from the language-keyed normalizer registry (`spellman_train/sources/normalize.py`: pymorphy3 rus/ukr, Stanza bel/bul/srp/kaz/kir, Apertium mkd, corpus prefix-cluster stems for the agglutinative set) and are cached in per-(pool, normalizer) sidecars. Selection (algo=6, the shipped model): seeded shuffle accepted on ≥`min_gain` new df≥2 lemmas or starved top-lemma carry, 20-char bands with per-(lemma,band) exposure spread, then a variety fill; see `docs/experiments.md` for what each variant measured |
| `ukr_tweets` | `lang`, `limit`, `min_chars`, `cyr` | saganoren/ukr-twi-corpus, 1.85M raw tweets; Twitter's `lang=="uk"` self-label + Cyrillic gate; proper CSV parsing (tweets embed newlines) |
| `mn_social` | `lang`, `limit`, `min_chars` | ganaxy/diploma — 10k raw Mongolian news/FB/YouTube comments (`text_raw`) |
| `kazsandra` | `lang`, `limit`, `min_chars`, `cyr` | IS2AI/KazSAnDRA Kazakh reviews; only the canonical ib/valid/test zips, deduped on `custom_id` (the resampled `*_ros`/`*_rus` zips duplicate rows) |

The wild-UGC adapters and the `hf` raw-mode gates exist for the
social-media lane (the rusentitweet analogs); the researched candidate
list with licenses, access commands and per-dataset validation reports
lives in `docs/wild-ugc-candidates.md` and
`train/cache/raw/<slug>/VALIDATION.md`.

Shared normalization conventions: whitespace-flattened single-line
samples, `min_chars = 20`, and per-language caps applied at the source
level (`train_per_lang`, `limit`) or mix level (`--cap-per-lang`).

### Mixing, splitting, augmentation

`spellman-train mix` merges any combination of sources:

```bash
uv run spellman-train mix --out data_mix \
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
- **Output format** is parquet by default:
  `<out>/data/{split}-00000-of-00001.parquet` (zstd, `lang`+`text`
  columns) — the HF-dataset-native layout that `publish dataset` uploads.
  `--format jsonl` restores the legacy flat `{split}.jsonl` (the trainer
  reads both).
- **Recipes are data**: every run records its exact argv, knobs and
  parsed sources into `<out>/manifest.json`. Replay one with
  `--from-manifest <old>/manifest.json --out <new>` (live `--out` /
  `--format` override; live `--source` appends). `fetch --manifest`
  prebuilds the same recipe's caches.

### Data hygiene

`spellman-train clean` drops rows the current model contradicts,
rewriting caches **in place** (sidecars untouched, so the fingerprint
stays valid):

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
uv run spellman-train clean cache/<name>-*.jsonl [--conf 0.995] [--script] [--dry-run] [--jobs 8]
```

`--jobs N` judges N caches at once (one subprocess per cache; the
per-cache work is single-threaded numpy, so a 32-core box finishes the
~100-cache pass in minutes instead of half an hour).

Rerun after any cache rebuild — a rebuild re-downloads the dirty
upstream data. The short-text lane (3–19 char rows, under the token
guard) gets `short-verify` instead: 3-judge consensus (spellman, GlotLID,
lid.176) with `--min-agree`, `--no-spellman` for eval referees.

### Hard negatives

`spellman-train hard-negatives` mines FineWeb-2's `_removed` subsets —
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
uv run spellman-train train --data data_mix --out ../model \
    --log2-d 17 --dim 128 --epochs 6 --lr 0.05 --hash-stats
```

Flags: `--log2-d` (bucket count D = 2^log2_d), `--hash-id
{fmix32,murmur2,multiply_shift}`, `--seed`, `--dim`, `--epochs`,
`--batch-size` (256), `--k` (tokens per training sample, 256), `--lr`,
`--per-lang-cap` (50k train-side rebalance), `--hash-stats`, `--device`.
`--data` reads parquet shards when present, else legacy `{split}.jsonl`.

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
  sees the difference. `uv run spellman-train quantize --store int8-row
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
- `uv run spellman-train eval-fasttext ../model/eval_test.tsv
  tatoeba_eval.tsv` — fastText lid.176 baseline on the identical ladder
  (fragments of gold languages outside its label set are excluded).
- `benchmarks/lid-bench` — Rust-side baselines (whichlang, lingua) on
  identical sampled rows, accuracy + latency; see the README comparison
  table for methodology and results.

## Publishing

Both artifacts live under `vpermilp/spellman` on the Hub — the model and
the dataset are different repo types, so the name hosts both without
collision:

```bash
uv run spellman-train publish dataset --dir data/v13c   # parquet + manifest + rendered card
uv run spellman-train publish model    --dir ../model   # model.json + model.safetensors (+ README.md)
```

`publish dataset` renders the README.md card from the mix manifest
(splits, counts, languages, recipe summary), creates the repo private if
missing, and uploads idempotently. `publish model` is transport only —
the model card stays hand-maintained in the model directory. Verify a
publish with a fresh `hf download vpermilp/spellman --repo-type dataset`.

## Current model: recipe and results

v13c = the v12 recipe (`recipes/v12.sh`, its promoted manifest at
`train/data/v12/manifest.json`) + the six crawl datasets
`vpermilp/lid-{sah,tyv,kpv,mhr,oss,udm}` as 11 raw `hf:` lanes
(`recipes/v13.sh`), **120k-per-language cap** (mix `--cap-per-lang
120000`, train `--per-lang-cap 120000`), diverse budgets ×1.5
(rus/ukr 20k, Turkic 16k), a 12k wikisource literary lane for rus, and
`--short-floor 0.40`: FineWeb-2 line-windows + ~104k Tatoeba training
sentences + per-language top-ups (Tatar Glot500/Wikipedia/parallel +
`tat_Latn`, Bashkir Telegram parallel, Chuvash community mono, Tuvan
linguist-collected, Kyrgyz Sputnik news, Udmurt ai-forever + zerpal,
Meadow Mari literary parallel, Macedonian real tweets, Glot500
Tajik/Sakha, native Tajik/Sakha corpora, Chechen Leipzig 2017+2023 +
OPUS translatewiki + NM 171k ce-ru), the wild-UGC and short-utterance
lanes of `docs/wild-ugc-candidates.md`, 12 `diverse:` lanes
(algo=6 banded lemma coverage), and ~4.8k scaled FW2 `_removed` hard
negatives. All caches passed hygiene (twin-protected). dim 128,
6 epochs, D = 2^17, fmix32, wild/short augmentation on train/val.
Splits 2,519,584 / 975,948 / 719,255. The cap is the lever that moved
the referees (32k → 80k → 120k, monotone on every referee, no language
down); FineWeb-2 top-ups of the thin tail hurt (register dilution) —
see `docs/experiments.md`, v13.

Results and history are tabulated in the [README](../README.md). Known
residuals: rus-attraction on short low-resource texts (wants hard
negatives at ~10× scale), the mkd/bul/sr continuum on marker-less
texts, Latin-script Tatar short sentences, tgk data thinness.

### Replaying this mix

Every `mix` run records its exact recipe into
`<out>/manifest.json` — the current model's recipe is
`train/data/v13c/manifest.json` (also in the published dataset). Replay
it (warm caches) into a fresh parquet mix with:

```bash
uv run spellman-train mix --from-manifest data/v13c/manifest.json --out data/v13c
```

On a fresh machine the caches rebuild by re-downloading everything
(`bash recipes/v12-pools.sh` first — the diverse lanes pin pool caches —
then `fetch --manifest data/v13c/manifest.json --jobs 4` does the rest up
front
in parallel), so the full loop is: fetch a judge model
(`hf download vpermilp/spellman --local-dir ../model`), build the
caches, run the mix once, then `clean cache/*.jsonl --script`, then
`hard-negatives --model ../model`, then re-mix (warm caches replay
instantly) and train. Rebuilds re-download the upstream data — always
rerun `clean` after one.

The pre-wild standing recipe, reconstructed from the cache fingerprints
(the Komi parallel corpus is deliberately absent — rejected by the
contamination audit):

```bash
# prerequisites (once):
#   - Tatoeba sentence dump at train/tatoeba/sentences.csv
#     (copy from a machine that has it, or the exports at
#     downloads.tatoeba.org)
#   - hf auth login, with the gates of muhtasham/tajik-corpus and
#     ailabykt/sakha-corpus-mono accepted in the browser
uv run spellman-train mix --out data_mix \
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

## End-to-end walkthrough

```bash
cd train && uv sync

# 0. (fresh machine) prebuild caches in parallel + fetch a judge model
uv run spellman-train fetch --manifest data/v13c/manifest.json --jobs 4
hf download vpermilp/spellman --local-dir ../model

# 1. mix (replays warm caches; parquet + manifest.json; --from-manifest
#    replays a recorded recipe)
uv run spellman-train mix --out data_mix \
    --source fineweb2:docs_per_lang=3600,per_doc=4 \
    --source tatoeba:train_per_lang=8000 \
    --cap-per-lang 16000 --wild-augment 0.3 --short-augment 0.2

# 2. train + export
uv run spellman-train train --data data_mix --out ../model \
    --log2-d 17 --dim 128 --epochs 6 --lr 0.05 --hash-stats

# 3. evaluate
cd .. && cargo run --release --bin assess -- --model model model/eval_test.tsv
./target/release/spellman eval model/eval_test.tsv

# 4. audit errors, then iterate (clean / hard negatives / new sources)
cargo run --release --bin assess -- --model model train/tatoeba_eval.tsv \
    --dump-errors errors --dump-per-lang 100

# 5. publish (dataset + model) once a mix/model is promoted
cd train && uv run spellman-train publish dataset --dir data_mix
uv run spellman-train publish model --dir ../model
```

The Rust↔Python feature parity must hold at all times:
`uv run spellman-train gen-fixtures && cargo test` (run both sides after
any feature/hash change; every trained model is invalid if they drift).
