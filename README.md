# spellman

**Fast, tiny language detection with first-class Cyrillic coverage.**

spellman tells you what language a text is written in. It covers 30
languages — the big world languages *plus* twenty Cyrillic-script
languages most detectors have never heard of — and it is built for the
text that actually breaks language detectors: tweets, chat messages,
comments, two-word utterances, and closely-related language pairs like
Russian/Ukrainian or Bulgarian/Macedonian.

- **~99% accuracy** on out-of-domain sentences, **~95%** on real
  ≤19-character utterances — the regime where general detectors drop to
  70–85% ([full benchmarks](docs/benchmarks.md))
- **~3.3 µs per document on one core** (~1.3 µs on clean sentences),
  **~0.5 µs across all 14 cores** of an M4 Max (~2M docs/s) — two orders
  of magnitude faster than fastText-class models
- **7.9 MB model**, pure Rust, no runtime dependencies beyond the crate
- MIT licensed, and the training data is commercially clean: no
  non-commercial upstream survives the license audit

## Quick start

```bash
cargo build --release -p spellman-cli
```

The model comes from the Hugging Face Hub through the standard HF cache
(first call downloads ~9 MB, later calls replay it):

```bash
echo "Съешь ещё этих мягких французских булок" | ./target/release/spellman detect --model hf:vpermilp/spellman
printf 'Қазақша\nHello\n' | ./target/release/spellman detect --model hf:vpermilp/spellman --lines   # kaz / eng
```

From Rust:

```rust
use spellman_detector::{BulkDetector, SingleDetector};

// One document at a time.
let mut single = SingleDetector::from_hub(1024)?;
let d = single.detect("Съешь ещё этих мягких французских булок")?;

// Bulk batches.
let mut bulk = BulkDetector::from_hub(1024, 4096)?;
let results = bulk.detect_batch(&["Привет", "Hello"])?;
```

A local model directory works too (`SingleDetector::load` /
`BulkDetector::load`, CLI `--model <path>`); the language inventory is
re-exported as `spellman_detector::Lang`. For GPU execution call
`svod_tensor::set_default_device(...)` before loading.

## Languages

| group | languages |
|---|---|
| East Slavic | Russian, Ukrainian, Belarusian |
| South Slavic | Bulgarian, Macedonian, Serbian (Cyrillic) |
| Turkic | Kazakh, Kyrgyz, Uzbek (Cyrillic), Tatar, Bashkir, Chuvash, Yakut (Sakha), Tuvan |
| Other Cyrillic | Tajik, Mongolian, Ossetic, Chechen, Udmurt, Meadow Mari, Komi-Zyrian |
| Latin script | English, Spanish, French, Portuguese, German |
| Script-routed | Mandarin, Japanese, Hindi, Arabic (detected by script alone) |

The Cyrillic minority languages (Yakut, Tuvan, Udmurt, Komi, Mari,
Ossetic, Chechen…) are a deliberate focus: spellman was trained on
~900k rows of real news, library, Telegram and VK text collected for
them (published as the open [`vpermilp/lid-*`
datasets](https://huggingface.co/datasets/vpermilp/lid-sah)), so they
score F1 0.99–1.00 rather than being absent or noise.

## Accuracy

Highlights against the strongest available baselines, on identical eval
rows (v14 model; every number and the methodology in
[docs/benchmarks.md](docs/benchmarks.md)):

| eval | spellman | GlotLID v3 | fastText lid.176 | lingua (high) |
|---|---|---|---|---|
| Tatoeba, clean out-of-domain sentences | 99.0% | **99.3%** | 94.9% | 97.7%¹ |
| wild Russian tweets | **96.9%** | 82.7% | 90.4% | — |
| real short utterances (≤19 chars) | **95.0%** | 71.3% | 84.3% | — |
| single words (the hardest rung) | **72.1%** | 43.9% | 59.0% | — |
| held-out mix, 719k rows of all registers | **98.6%** | 92.6% | 81.5% | 90.3%¹ |

¹ on the subset of our languages lingua supports (17/30).

The pattern: on clean long text every good detector works, and GlotLID's
enormous training set keeps a 0.2pp lead on Tatoeba. Everywhere else —
short text, wild register, close Cyrillic pairs, minority languages —
spellman leads by 5–30 points while running ~100× faster than the
fastText-class models and ~60× faster than lingua (one thread each).

## Speed

µs per document on the held-out mix, BEAM=16:

| CPU | one thread, bulk | one thread, single document | all cores, one replica per thread | all cores, svod-threaded kernel |
|---|---|---|---|---|
| Apple M4 Max (14 cores) | 3.2–3.4 | 6.5–6.8 | **0.44–0.50** | 0.97–1.00 |
| AMD Ryzen 9 7950X3D (16C/32T) | 4.5–4.6 | 4.7 | **0.74–0.75** | 5.0 |
| AMD Ryzen AI Max+ 395 (16C/32T) | 5.2 | 11.6 | **0.44** | — |

| GPU (svod device) | bulk, featurized on one thread | bulk, featurized on all cores | single document |
|---|---|---|---|
| AMD Radeon 8060S (the AI Max+ 395's iGPU, `AMD:0`) | 3.0 | **1.5** | 20.7 |
| NVIDIA RTX 3060 (`CUDA:0`) | 14.0 | 16.7 | 34.3 |

The x86 and GPU rows score v14's own 719k-row test split plus Tatoeba
(756,306 rows); the M4 rows score the older v12-era held-out file (368,507
rows, shorter on average), so read across a row, not down a column — the
M4 row is due a rerun on the 719k split. Replicas are the fastest setup on
every CPU.

GPUs are supported (`SVOD_DEVICE=AMD:0` / `CUDA:0`) but not worth it for
language detection: featurization (~60% of the work) stays on the CPU, so
CPU replicas win on every machine measured. A unified-memory iGPU is useful
only to keep the gather off a busy CPU.

For multi-core bulk work, prepare a one-thread plan (`SVOD_THREADS=1`,
so BEAM tunes the plan for one thread) and fork one replica per worker
(`BulkDetector::replicate` under rayon `map_init`; replicas share the
weights). Methodology and the whichlang / lingua /
GlotLID comparison: [docs/benchmarks.md](docs/benchmarks.md).

Inference is pure table lookups: the trained network folds algebraically
into a single quantized lookup table (`P = E·W`), executed by the [svod]
JIT with a beam-searched schedule. No matmul, no embedding gathers. The
bulk detector compiles a fixed batch (so the kernel threads across all
cores) and keeps a small ladder of plans over the token budget, picking
the smallest one that fits each batch's longest row — a batch of single
words never pays for a 1024-token plan.

The beam-searched numbers above need svod's out-of-process search
helper (the default heuristic schedule runs ~5 µs/sample without it):

```sh
cargo install svod-tensor --bin svod-beam-worker
export SVOD_BEAM_WORKER=~/.cargo/bin/svod-beam-worker
BEAM=16 spellman ...   # or --beam 16 in the examples
```

## How it works

A fastText-style linear model, engineered for the close-pair problem:
canonicalizing tokenizer (URLs, @mentions and numbers become sentinels
so they stop outvoting real words), char 1–5-grams plus whole-word and
word-bigram features, signed feature hashing into 2^18 buckets, and a
confidence threshold θ calibrated so the `is_uncertain` flag actually
predicts errors. Details, measurements and the design rationale:
[docs/design.md](docs/design.md).

## Training data and reproducibility

The full training mix (4.2M rows, 26 languages) is published at
[vpermilp/spellman](https://huggingface.co/datasets/vpermilp/spellman)
with a byte-exact recipe in its `manifest.json` and a per-upstream
license table — rebuild it with one command from
[docs/training.md](docs/training.md). The pipeline (source adapters,
LID-hygiene with twin-language protection, deterministic splits,
training, evaluation) lives in [`train/`](train); the experiment log
with every measured decision — including the ones that failed — is
[docs/experiments.md](docs/experiments.md).

## FAQ

**Why not just extend whichlang with more languages?**
whichlang can be retrained — its repo ships the notebook that fits its
logistic regression and regenerates `weights.rs` — but its featurizer is
structurally Latin-only: character n-grams are emitted solely for ASCII
bytes (packed one byte per char into a u32), while every non-ASCII
character contributes just two coarse features — its 128-codepoint
Unicode block and a class from a hardcoded Latin-1/Japanese list. Its
one Cyrillic language, Russian, is effectively recognized as "the
Cyrillic block", which works only because it has no Cyrillic neighbors;
ru/be/uk or kk/ky/tt/ba would collapse onto the same features no matter
how you retrain. Fixing that means redesigning tokenization, hashing and
capacity — which is this project. The other hard part is the data:
close pairs need curated corpora and hygiene (a `җ` in a Bashkir row is
Tatar contamination; a `ҡ` is genuine Bashkir), and whichlang's notebook
consumes a single prepared Tatoeba dump. It covers 10 of our 30 classes
and scores 27% on this workload counting its missing languages as
errors.

**Do I need a GPU?** No — the numbers above are CPU. GPU execution
exists through svod but is unnecessary at these speeds.

**What about languages outside the 30?** Out of scope by design;
GlotLID (2,102 labels) is the right tool for broad coverage, at ~100×
the latency.

## License

MIT. Training data licenses are tabulated per-upstream in the
[dataset card](https://huggingface.co/datasets/vpermilp/spellman); no
non-commercial source remains.

## References

Weinberger et al. 2009 (feature hashing); the fastText "Bag of Tricks"
line (Joulin et al. 2016; GlotLID / OpenLID are the current open-LID
SOTA); rurban/SMHasher hash-quality tables; the FineWeb-2 report
(arXiv:2506.20920); DSL shared-task results on bg/mk-class pairs.

[whichlang]: https://github.com/quickwit-oss/whichlang
[lingua]: https://github.com/pemistahl/lingua-rs
[svod]: https://github.com/npatsakula/svod
