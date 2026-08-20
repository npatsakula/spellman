# spellman

Cyrillic-optimized language detection: general coverage of the G10 world
languages plus ~20 Cyrillic-script languages (30 classes), built to beat
existing detectors exactly where they are weakest — closely-related
Cyrillic pairs (ru/be/uk, bg/mk/sr, kk/ky/tt/ba) on wild, short, real
internet text — at whichlang-class CPU speed.

- **[docs/design.md](docs/design.md)** — hashing, tokenization &
  canonicalization, model architecture, the algebraic fold, svod JIT
  runtime, measured optimizations.
- **[docs/training.md](docs/training.md)** — the data pipeline: adding
  sources, normalization, hygiene, hard negatives, training, evaluation.

## Results

Head-to-head against fastText lid.176 (Meta's 176-language model) on
identical eval files, including lingua-style granularity ladders
(accuracy on single words / word pairs / triples / whole texts derived
from the same eval data):

| eval | rung | spellman | GlotLID v3 | fastText lid.176 |
|---|---|---|---|---|
| held-out mix (239,042, pristine test) | text | **98.30%** | 93.58%‡ | 84.61%* |
| Tatoeba (37,051, out-of-domain) | word / pair / triple | **66.7 / 86.2 / 93.3** | 43.9 / 79.3 / 91.9‡ | 59.0 / 79.0 / 87.9 |
| Tatoeba (37,051, out-of-domain) | text | 98.42% | **99.25%**‡ | 94.90%* |
| rusentitweet (2,606 wild Russian tweets, label-audited) | text | **92.56%**† | 82.73%‡ | 90.41% |
| COSMUS Russian (2,808 wild Telegram/reviews, gold-labeled) | text | **96.79%** | 95.69%‡ | 96.65% |
| short utterances (574, orthography-certified ≤19 chars) | text | **89.90%**† | 71.25%‡ | 84.32% |

\* fastText scored on the subset of languages its label set supports
(24/30; no kpv/udm labels, and its `uz` is Latin-script Uzbek — it scores
1.15% on our Cyrillic uzn). Single words are the hard rung for everyone:
rus drops to ~45% on words (uk/be absorb them), exactly the close-pair
problem spellman is built around; it recovers to ~97% by full text.

† wild referee: real Russian tweets, 78% containing Latin words, 61%
@mentions, 20% URLs — the register clean corpora never show. The original
sentiment-era file never verified language; a GlotLID+lid.176 consensus
audit removed 73 provably mislabeled rows (13 Mongolian tweets, plus
Ukrainian/Serbian/Macedonian/Mari) → the 2,606-row v2. spellman's residual
losses are one/two-word utterances valid across Cyrillic languages ("Да!",
"шок", "Ща"). The COSMUS row is the control — manually language-labeled
wild Russian (2022–24 Telegram/reviews, never in training). The
short-utterance referee is twin rows certified by ORTHOGRAPHY (і ї є ґ
never occur in Russian, ы ъ ё э never in Ukrainian — models cannot
certify text this short: soft judge consensus leaks Ukrainian into
Russian pools because lid.176 itself misreads short Ukrainian as ru);
unmarked twins are the intrinsically-ambiguous bucket and are excluded.

‡ GlotLID v3 ([cis-lmu/GlotLID](https://huggingface.co/cis-lmu/GlotLID)),
the open-LID SOTA fastText model (2,102 labels, 1.7 GB), scored with
script-variant labels mapped to our classes (`tat_Latn` → tat — the
courtesy goes to the baseline) and full coverage of the evals' languages
(ara/cmn are absent from its label set; neither appears in these files).
By length: held-out 79.2 / 92.4 / 97.5 (≤20 / 21–100 / >100 — spellman
leads every bucket), Tatoeba 97.8 / 99.3 / 100.0 (GlotLID leads every
bucket). It predicts at ~355 µs/doc — ~100× spellman on the M1 Pro,
~300× on the AMD 395 Max. The split is the story: spellman wins the
wild, heavy-Cyrillic workload by 4.7pp and the single-word rung by ~23pp
(2,102-class label entropy is brutal on short text); GlotLID's far
larger training set still wins clean out-of-domain sentences, by
0.8pp.

### Against the Rust LID crates

[`benchmarks/`](benchmarks) (standalone crate, `lid-bench`) runs spellman,
[whichlang] and [lingua] on **identical rows** — the full eval files
below (`--rows-per-lang 0`; a seeded 500/language balanced sample is the
default for quick runs). Every tool gets the same texts and its own
language inventory as the detector: lingua is built from exactly the 17
of our languages it supports, with preloaded models — its best shot on
our workload. Rerun:
`cd benchmarks && cargo run --release -- --model ../model --rows-per-lang 0 ../model/eval_test.tsv --by-length --per-lang`.

| detector | our classes | held-out: all rows | held-out: its subset | Tatoeba: all rows | Tatoeba: its subset | µs/sample |
|---|---|---|---|---|---|---|
| spellman (bulk) | 30/30 | **98.30%** | **98.30%** | **98.42%** | **98.42%** | 5.1 |
| spellman (single) | 30/30 | **98.30%** | **98.30%** | **98.42%** | **98.42%** | 10.8 |
| whichlang 0.1 | 10/30 | 27.49% | 90.56% | 32.28% | 99.67% | **1.6** |
| lingua 1.8 (high) | 17/30 | 44.16% | 92.12% | 68.55% | 97.69% | 325 |
| lingua 1.8 (low) | 17/30 | 42.05% | 87.73% | 65.38% | 93.17% | 365 |

(239,042 / 37,051 rows; Apple M1 Pro; spellman k=1024 under BEAM=16;
µs/sample from the held-out file in this harness — the CLI eval path on
the same data reads 4.9 µs/sample. "all rows" counts gold languages
outside a tool's inventory as errors — what a 30-class Cyrillic workload
actually sees.)

**Accuracy by text length** — supported-subset accuracy per char-length
bucket (the buckets `assess` uses; for spellman the subset is all rows):

| bucket | held-out mix (n) | spellman | whichlang | lingua high |
|---|---|---|---|---|
| ≤20 chars | 17,943 | **95.6%** | 88.5% | 82.0% |
| 21–100 | 119,419 | **98.1%** | 86.7% | 91.4% |
| >100 | 101,680 | **99.1%** | 96.7% | 98.1% |

| bucket | Tatoeba (n) | spellman | whichlang | lingua high |
|---|---|---|---|---|
| ≤20 chars | 1,567 | 96.1% | **97.5%** | 92.8% |
| 21–100 | 34,674 | 98.5% | **99.7%** | 97.8% |
| >100 | 810 | 99.9% | 99.5% | **100.0%** |

(The held-out short bucket is large because the verified short-utterance
lane contributes real 3–19-char wild rows to every split.)

What the numbers say:

- **Coverage dominates a Cyrillic workload.** whichlang knows one
  Cyrillic language of our 21 (rus); lingua knows 8. For the other
  languages of the region their answer is structurally wrong, which is
  the 14–68% all-rows column.
- **Short text is lingua's advertised strength — and spellman wins it**:
  on ≤20-char rows spellman leads lingua high-accuracy by 3–14pp on both
  referees (96.1 vs 92.8 Tatoeba, 95.6 vs 82.0 held-out), and lingua's
  low-accuracy mode collapses further. Mid-length is spellman's biggest
  gap over lingua (98.1 vs 91.4 held-out); at >100 chars everyone
  converges to 98–100% and the differences are coverage, not quality.
- **On the languages they share with us, spellman wins the close pairs**
  (held-out, full file): ukr 96.8% vs lingua 87.8, mkd 97.2% vs 89.2,
  srp 98.7% vs 97.0, kaz 98.6% vs 96.1, bul 97.5% vs 95.8, eng 97.8% vs
  96.0, bel at parity (98.8 vs 99.0) — every shared language is at
  parity or ahead, with the wild-heavy classes widest.
- **whichlang's 98.0% on Russian is real — and the trade is visible:**
  its 16-class world contains no ukr/bel/kaz to confuse with Russian.
  spellman's rus (92.9%) bleeds into those close classes — and into the
  small languages whose real wild data now competes — which is precisely
  the capacity that makes the other 20 Cyrillic columns work.
- **Latency**: whichlang is the fastest per document (tiny 16-class
  model) at ~3.5× spellman bulk; lingua high-accuracy is ~64× slower
  than spellman bulk (325 vs 5.1 µs/sample, BEAM=16).

Per-language on the held-out mix: che F1 1.00 (from the campaign's
weakest class), tat 0.99, mhr/bel/bak 0.99, sah 0.98 — the residual
confusions are the genuinely hard ones (uzn P 0.87, tgk R 0.92 from data
thinness, rus-attraction on short low-resource texts).

Performance (fp16 svod JIT plans, BEAM=16, k=1024, Tatoeba eval —
37,051 documents):

| hardware | bulk | single document |
|---|---|---|
| Apple M1 Pro | 3.6 µs/sample (~280k docs/s) | 3.8 µs/doc |
| AMD AI 395 Max | 1.2 µs/sample (~830k docs/s) | 13.0 µs/doc |

Scoring is pure table lookups after the algebraic fold `P = E·W` — no
embedding gathers, no matmul. fmix32 bucket spread on real n-grams:
chi²/dof ≈ 1.006 (uniform ≈ 1.0).

## Datasets

| dataset | languages | role |
|---|---|---|
| FineWeb-2 per-language configs | 21 Cyrillic + spa/fra/por/deu | backbone, line-window sampled (short-text realism, not Wikipedia) |
| FineWeb `sample-10BT` | eng | English backbone (FineWeb-2 has no eng) |
| Tatoeba training remainder | 26 | ~104k clean sentences; eval half is a frozen referee |
| Glot500 slices | tat, tgk, sah | weak-language top-ups |
| Native gated corpora | tgk (tajik-corpus), sah (sakha-corpus-mono) | in-domain text for the two thinnest backbones |
| Parallel/community corpora | tat (162k parallel + Wikipedia + `tat_Latn`), bak (Telegram), chv (community), tyv (linguist web text), kir (Sputnik), udm (ai-forever + zerpal), mhr (literary), mkd (tweets) | per-language top-ups for the weak classes |
| Chechen stack | Leipzig community 2017+2023, OPUS translatewiki, NM 171k ce-ru parallel | che: weakest class → F1 1.00 |
| rusentitweet (train split) | rus | wild-register training rows; eval split is the frozen wild referee |
| FineWeb-2 `_removed` subsets | 12 configs | model-labeled hard negatives (twin-protected) |

All sources flow through the pluggable adapter registry
(`train/sources/`), are hygiene-audited (twin-protected judges, ~1.3k
foreign rows removed), and mixed with deterministic crc32 splits —
details in the [training guide](docs/training.md).

## Quick start

Build ([svod] comes in as a git dependency — a plain clone just works):

```bash
cargo build --release -p spellman-cli
```

Create the model from code — it comes from the Hugging Face Hub through
the standard HF cache (first call downloads, later calls replay it), the
same `from_hub` wiring svod's own models use:

```rust
use spellman_detector::{BulkDetector, SingleDetector};

// One document at a time — B=1 is baked into the plan (static kernels).
let mut single = SingleDetector::from_hub(1024)?;
let d = single.detect("Съешь ещё этих мягких французских булок")?;

// Bulk batches — constant K, rebindable batch, rayon featurization written
// zero-copy straight into the plan's input buffer. from_hub_variant picks
// a storage format: int8-col is 3.9MB instead of 7.9MB f16, same accuracy.
let mut bulk = BulkDetector::from_hub_variant("int8-col", 1024, 512)?;
let results = bulk.detect_batch(&["Привет", "Hello"])?;
```

The language inventory is re-exported by the detector — always use
`spellman_detector::Lang` rather than a separate `spellman-language`
dependency. A local model directory works too: `SingleDetector::load`
/ `BulkDetector::load` with any path.

The CLI does the same via `--model hf:<owner>/<repo>[/variant]`
(a plain path also works; default `./model`):

```bash
echo "Съешь ещё этих мягких французских булок" | ./target/release/spellman detect --model hf:vpermilp/spellman
./target/release/spellman eval --model hf:vpermilp/spellman train/tatoeba_eval.tsv  # accuracy + throughput
./target/release/spellman bench --single                                           # probes + timings
```

Device selection follows svod's convention — weights land on the default
device at load time, so call `svod_tensor::set_default_device(...)` before
loading for GPU execution.

## FAQ

### Why not just extend Whichlang with new languages?

There is nothing to extend — whichlang ships a fixed 16-language scorer
(murmur2-hashed char n-grams into a linear table) with no training
pipeline and no data engine, so adding a class means rebuilding both.
And the hard part of covering 20 more Cyrillic languages is not the
extra labels but the close pairs: they need curated corpora and hygiene
(a `җ` in a Bashkir row is Tatar contamination; a `ҡ` is genuine
Bashkir), a canonicalizing featurizer so URLs and @mentions stop
outvoting real words, and shared-embedding capacity so ru/be/uk can
coexist. whichlang's 99.2% Russian exists precisely because nothing
competes with it — it covers 10 of our 30 classes with a single
Cyrillic one, and scores 14% all-rows on our workload.

## References

Design decisions are grounded in: Weinberger et al. 2009 (feature
hashing), the fastText "Bag of Tricks" line (Joulin et al. 2016; GlotLID
/ OpenLID are the current open-LID SOTA, all fastText-style),
rurban/SMHasher quality tables (murmur2 bias vs fmix32; xxh3 low-bit
failures), the FineWeb-2 report (arXiv:2506.20920), and DSL shared-task
results showing 95%+ is achievable on bg/mk-class pairs with
discriminative n-gram models.

[whichlang]: https://github.com/quickwit-oss/whichlang
[lingua]: https://github.com/pemistahl/lingua-rs
[svod]: https://github.com/npatsakula/svod
