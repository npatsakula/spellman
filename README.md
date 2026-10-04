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
rows (v16 model; every number and the methodology in
[docs/benchmarks.md](docs/benchmarks.md)):

| eval | spellman | GlotLID v3 | fastText lid.176 | lingua (high) |
|---|---|---|---|---|
| Tatoeba, clean out-of-domain sentences | 99.1% | **99.3%** | 94.9% | 97.7%¹ |
| wild Russian tweets | **96.6%** | 82.7% | 90.4% | — |
| real short utterances (≤19 chars) | **95.8%** | 71.3% | 84.3% | — |
| single words (the hardest rung) | **72.0%** | 43.9% | 59.0% | — |
| held-out mix, 719k rows of all registers | **98.7%** | 92.9% | 82.0% | 90.3%¹² |

¹ on the subset of our languages lingua supports (17/30).
² lingua was last measured on the v14 test split (719,255 rows).

The pattern: on clean long text every good detector works, and GlotLID's
enormous training set keeps a 0.2pp lead on Tatoeba. Everywhere else —
short text, wild register, close Cyrillic pairs, minority languages —
spellman leads by 5–30 points while running ~100× faster than the
fastText-class models and ~60× faster than lingua (one thread each).

## Speed

Apple M4 Max, BEAM=16, µs per document:

| | Tatoeba sentences | held-out mix |
|---|---|---|
| one thread, bulk | 1.1–1.4 | 3.2–3.4 |
| one thread, single document | 2.6 | 6.5–6.8 |
| 14 cores, one single-thread replica per core | 0.18–0.21 (~5M docs/s) | 0.44–0.50 (~2.1M docs/s) |
| 14 cores, svod-threaded kernel, 4096-row batches | 0.75 | 0.97–1.00 |

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
