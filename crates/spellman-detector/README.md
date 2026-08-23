# spellman-detector

Cyrillic-optimized language detection: 30 classes covering the G10 world
languages plus ~20 Cyrillic-script languages, built to beat existing
detectors on closely-related Cyrillic pairs (ru/be/uk, bg/mk/sr, kk/ky/tt/ba)
in wild, short, real internet text — at whichlang-class CPU speed.

Hashed char n-grams feed a folded fastText-style linear model whose scoring
is expressed as a svod JIT plan (pure table lookups after the algebraic
fold `P = E·W` — no embedding gathers, no matmul).

```rust
use spellman_detector::{BulkDetector, SingleDetector};

// One document at a time — B=1 is baked into the plan (static kernels).
let mut single = SingleDetector::from_hub(1024)?;
let d = single.detect("Съешь ещё этих мягких французских булок")?;

// Bulk batches — constant K, rebindable batch, rayon featurization written
// zero-copy straight into the plan's input buffer.
let mut bulk = BulkDetector::from_hub_variant("int8-col", 1024, 512)?;
let results = bulk.detect_batch(&["Привет", "Hello"])?;
```

The language inventory is re-exported — always use `spellman_detector::Lang`
rather than a separate `spellman-language` dependency.

Full design, benchmarks and the training pipeline: the
[spellman repository](https://github.com/npatsakula/spellman).

> **Publishing status:** this crate is not yet on crates.io — it depends on
> [svod](https://github.com/npatsakula/svod) via git, and crates.io rejects
> git dependencies. The svod release currently on crates.io
> (0.1.0-alpha.3) predates APIs this crate uses. Once a matching svod
> release is published, the git specs only need a `version` requirement
> alongside them (cargo strips git/path specs when packaging); until then
> build this crate from the repository.
