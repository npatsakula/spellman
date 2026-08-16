# spellman — design document

How the detector works, why each piece is the way it is, and which
optimizations were measured. The companion [training guide](training.md)
covers the data pipeline that produces the shipped model.

## Scope and goals

30 classes: the G10 world languages plus ~20 Cyrillic-script languages
(exact inventory: `crates/spellman-language/src/lib.rs`, one macro row per
language). The design target is the regime where general-purpose detectors
fail:

- **Closely-related Cyrillic pairs** — ru/be/uk, bg/mk/sr, kk/ky/tt/ba,
  udm/mhr/kpv: languages sharing most of their lexicon core, separated by
  affixes, signature letters and function words.
- **Wild short text** — tweets, comments, search queries: URLs, @mentions,
  emails, digit runs, casing chaos, mixed scripts; not clean Wikipedia
  paragraphs.
- **CPU speed** — scoring must be table lookups, not matrix multiplies, so
  bulk classification is I/O-bound rather than compute-bound.

Successor to [whichlang]'s design (hashed char n-grams → linear model),
rebuilt on every layer: features, hashing, model capacity, and inference.

## Inventory and script routing

`spellman-language` is a dependency-free crate: one `languages!` macro table
defines every language — variant name (its ISO 639-3 code, also the
model-column order), ISO 639-1, NLLB-200, and Whisper codes where they
exist, English name, localized names, primary script. Everything else is
generated; adding a language is adding a row.

Routing happens before any model runs (`crates/spellman-detector/src/route.rs`):

- kana anywhere → **jpn** (Japanese mixes kanji with kana; kana is the
  unique marker even when kanji dominates the letter count);
- otherwise the majority script among Latin/Cyrillic/Han wins, with
  mixed-script ties resolved toward the rarer script (Han > Cyrillic >
  Latin — a split document usually carries the rarer script's unique words);
  Han → **cmn**; Latin/Cyrillic → the corresponding script-group columns;
- Devanagari → **hin**, Arabic → **ara** (single-language scripts);
- no supported-script letters → no language (`und` in the CLI).

Script-unique languages never reach a model, and their columns are never
the argmax target of a trained column. Model columns are contiguous per
group (Cyrillic 0..21, Latin 21..26), so the scoring loop touches only its
group's slice of the folded table.

## Feature pipeline

Implemented in `crates/spellman-detector/src/features.rs`, mirrored
bit-for-bit in `train/spellman_features.py` (see
[parity contract](#the-rustpython-parity-contract)). Pipeline per document:

```
text ─▶ whitespace split ─▶ per-word: strip leading '#', classify, lowercase
    ─▶ wrap in BOW/EOW boundary markers
    ─▶ char n-grams, n = 1..=5, over the wrapped sequence
    ─▶ packed u64 keys ─▶ (bucket, sign) via feature hashing
```

### Word classification (canonicalization)

Wild text is full of tokens whose *content* is arbitrary but whose *class*
is meaningful: URLs, emails, @mentions, digit-bearing ASCII tokens. A
25-char URL generates ~125 n-grams — one link would outweigh four real
words in the mean-pool. Each whitespace-delimited word is classified
(pure ASCII byte checks, no regex) and canonical classes pack as a single
private-use sentinel codepoint:

| class | example | sentinel |
|---|---|---|
| mention | `@user_2020` | `U+E003` |
| URL prefix | `https://t.co/x3f…`, `www.…` | `U+E001` |
| email | `user@mail.ru` | `U+E002` |
| digit-bearing ASCII | `2020`, `3.5.2`, `100px` | `U+E004` |
| bare ASCII domain | `example.com` (alphabetic TLD ≥ 2) | `U+E001` |
| everything else | real word, packed as-is | — |

A canonicalized word packs as `[BOW, sentinel, EOW]` — ~30 neutral anchor
n-grams regardless of the random tail. Evaluation order is part of the
contract: mention → URL prefix → email → digit-bearing ASCII → bare domain
→ word.

Guards, each deliberate:

- **ASCII-only rules.** Cyrillic dotted abbreviations (`т.е.`) and
  digit-bearing Cyrillic words (`миллион2020`) are real language evidence
  and always pass through as words.
- **`#` stripped from hashtags** — `#красноярск` keeps its word.
- **Domain rule requires an alphabetic TLD of length ≥ 2** — `t.co` is a
  URL, `U.S.A.` is a word (its final label is a single letter).
- **Mention/URL/email tails never reach the n-gram pool**, so the
  augmentor only needs to target what canonicalization cannot: register
  (casing, elongation, slang loans) and emoji/hashtag noise.

Measured against regex alternatives for the classifier: 29 ns/word
hand-rolled vs 67 ns/word for a 5-alternative DFA `is_match` and 83 ns with
captures (PikeVM) — and the hand classifier keeps the Rust↔Python
bit-parity contract out of regex-engine semantics.

### Casing, boundaries, packing

- Lowercasing has a fast path for exactly the scripts this detector routes
  (ASCII and Cyrillic А-Я: `+0x20`; Cyrillic supplement U+0400-U+040F:
  `+0x50`), falling back to full Unicode mapping otherwise.
- Each word is wrapped in `U+0002` (begin) / `U+0003` (end) markers —
  codepoints lowercasing never produces. Boundaries make affix n-grams
  explicit: the trigram `(BOW, с, о)` means "word starts with со" — the
  verb-ending signal that separates Russian/Belarusian/Ukrainian.
- An n-gram packs into a u64 key, 21 bits per codepoint
  (`k = ((c0 << 21 | c1) << 21 | …)`, wrapping). n ≥ 4 overflows 64 bits,
  which is fine — the encoding stays injective for n ≤ 3 (whole short
  words captured exactly) and the mixer spreads wrapped values.
- **Order tagging:** every key is XORed with `n * 0x9E3779B97F4A7C15`.
  Without it, u64 wrapping makes each 5-gram key bit-identical to its
  suffix 4-gram — the model would see amplified 4-grams instead of real
  5-grams. The XOR is bijective per order, so within an order nothing
  changes; across orders, windows no longer alias.

## Feature hashing

`crates/spellman-detector/src/hash.rs`. A packed key maps to a bucket in
`D = 2^log2_d` (17 in the shipped model) plus a sign bit:

- **Buckets from the high bits** (`h >> (32 - log2_d)`), not `h % D`.
  Every documented mixer weakness lives in the low bits (murmur2's 1.7%
  bias, xxh3low's Moment-Chi2 blowup); under `% D` those defects land
  directly in the bucket index, under high-bit extraction they don't.
- **Signed hashing** (extra hash bit → ±1): bucket collisions between
  opposing signs cancel in expectation (Weinberger et al. 2009), halving
  the effective collision damage in a mean-pooled sum.
- Mixers are u32 multiply/xor/shift only, so the identical function
  exists in numpy (training) and could run in svod tensor ops on-device.

Three mixers selectable per model (`hash` field in model metadata):

| id | function | role |
|---|---|---|
| `fmix32` | MurmurHash3-32 finalizer over the mixed key halves | **primary** |
| `murmur2` | whichlang's exact murmurhash2-on-u32, both halves | A/B parity baseline |
| `multiply_shift` | top 32 bits of `key * 0x9E3779B97F4A7C15` | pairwise-independent reference |

fmix32's key mixing: `h = lo ^ (seed · 0x85EBCA6B)`, then
`h ^= hi · 0xC2B2AE35`, then the finalizer.

**XXH3 was evaluated and rejected** on evidence: documented low-bit bias
on short keys, and secret tables are hostile to GPU/on-device replay.
Hash quality is checked empirically: a chi-square of *distinct* n-gram
keys over buckets (occurrences are Zipfian by nature — a chi-square on
them measures language, not hashing). fmix32 at D = 2^17: chi²/dof ≈ 1.006
(uniform ≈ 1.0).

## Model

fastText-style, one shared embedding table over hash buckets:

```
tokens → signed bucket ids → E [D+1, dim] → mean-pool → Linear(dim, C) → softmax
```

- dim 128, trained with softmax cross-entropy; shared embeddings give the
  model capacity to learn what close pairs share and where they differ.
- **Zero-initialized embeddings** (fastText convention). With random init,
  n-grams landing in never-updated buckets fold to arbitrary nonzero
  scores — verified failure: `"sweatshirt"` alone scored bul 1.0. With
  zero-init, untrained buckets fold to exactly-zero logits.
- Trained in f32 (PyTorch, AdamW, linear LR decay to zero across epochs);
  exported in f16.

### The algebraic fold

The head is linear, so at export `scores = mean(sign · E[bucket]) · W + b`
collapses to `(1/n) Σ P[bucket] · ±1 + b` with **`P = E·W`**, one
`[D+1, C]` table. Inference never touches an embedding table or a matrix
multiply: per token it's one table lookup, one conditional negate, C
additions. Row `D` is the padding row, exactly zero. The exported
artifact carries only `P` and `bias` — everything the runtime loads
(8 MB instead of 41 MB at D = 2^17); the unfused `E`/`W` are
training-side state.

Signed hashing folds into the table *layout* at load: the JIT gather
table is `cat([P, -P])` of shape `[2(D+1), C]`, so a token's sign selects
the row block and the runtime graph needs no multiplies.

### Export contract

`model.json` + `model.safetensors` in a model directory:

```json
{
  "format": "spellman-model",
  "version": 2,
  "canonicalize": true,
  "languages": ["rus", "…", "ara"],
  "log2_d": 17,
  "hash": "fmix32",
  "seed": 2654435769,
  "n_min": 1, "n_max": 5,
  "theta": 0.8331743478775024,
  "quant": {"dtype": "int8", "scheme": "column"}
}
```

`languages` is the column order (must equal `Lang::ALL`); `version: 2` +
`canonicalize: true` is the canonicalizing feature space — the loader
rejects anything else at load time, so a stale model can't be silently
scored with the wrong tokenizer. `theta` is the calibrated confidence
threshold (5th percentile of validation confidence): below it,
`Detection::is_uncertain` is set.

**Storage precision is decoupled from compute.** `quant` declares how `P`
is stored — `float16` (no scales), or `int8`/`fp8e4m3` with a f32
`scales` tensor per bucket row or per language column. The loader
dequantizes into the canonical table (`resolve_table` is the single place
that knows schemes), so the runtime graph and every tool are unchanged;
`spellman-train --store` gates each scheme against validation accuracy at
export. Measured on the shipped model: int8 and fp8 both land within
±0.02pp on both referees while roughly halving the artifact (3.9–4.5MB
vs 7.9MB). True int8 *compute* was prototyped and rejected on evidence:
the i8→i16→i32 cast chain that svod's type lattice forces defeats the
beam scheduler's fusion — with `BEAM=2` the int8 graph is 18% slower
than f16 at K=1024 (`examples/int8_bench.rs` measures the split).

## Runtime

All inference runs through compiled [svod] execution plans
(`crates/spellman-detector/src/jit.rs`); weights land on the default
device at load time (`svod_tensor::set_default_device` for GPU).

The graph is deliberately minimal — gather and one reduction, pure fp16
end to end (ARM/NEON native on Apple Silicon; no cast kernels in the
replayed graph):

```text
idx [b, K] i32 ──gather──> table rows [b, K, C] f16 ──sum over K──> [b, C] f16
```

Shape specialization is the core trick:

- **K (tokens per document) is a compile-time constant baked into the
  plan** — never a symbolic variable — so the scheduler specializes every
  kernel for the fixed sequence length. Documents are truncated/zero-padded
  to K; padding gathers the all-zero row, so the sum is unaffected.
- **Only the batch dimension is symbolic**, rebound per call with
  `execute_with_vars(&[("b", n)])`. `SingleDetector` additionally bakes
  B = 1 (`with_b_fixed(1)`) for fully static single-document kernels.
- **Mean-pooling, bias, softmax, argmax run host-side at read-out** (30
  floats per document): featurization already knows each document's exact
  token count, so the graph needs no count computation. The single f32
  conversion happens there, via a hand-rolled fp16→f32 widening.

**Zero-copy featurization:** the plan's input buffer is host-mapped;
rayon-parallel feature extraction writes signed bucket indices straight
into the buffer through a typed view — no staging allocation, no copy.

Measured on the shipped model (Apple Silicon): ~5.3 µs/sample bulk
(~180k docs/s), ~5.4 µs single-document with the beam scheduler
(`BEAM=2` — worth 5.8× over default heuristics on this graph).

## The Rust↔Python parity contract

The feature pipeline exists twice — Rust runtime, Python training — and
every trained model is invalid if they drift. `train/spellman_features.py`
mirrors `features.rs`/`hash.rs` bit-for-bit, including a numpy-vectorized
batch path (`bucket_tokens_flat`) asserted bit-identical to the scalar
reference. `uv run spellman-gen-fixtures` regenerates
`crates/spellman-detector/fixtures/hash_vectors.json` from the Python side;
`cargo test` verifies the Rust side against it. Both implementations were
written from the same spec; the fixture is the arbiter.

## Optimization inventory

Measured decisions, in the order they pay off:

| decision | measurement |
|---|---|
| algebraic fold P = E·W | scoring = lookup + add, no matmul (whichlang-class throughput) |
| zero-init embeddings | untrained buckets → exactly-zero logits (fixed `"sweatshirt"` → bul 1.0) |
| sentinel canonicalization | wild referee 72.49% → 92.35%; ~30 neutral n-grams vs ~125 per URL |
| precision-decoupled storage (int8/fp8 `P` + scales) | −50% artifact at ±0.02pp; int8 *compute* rejected: cast chain defeats BEAM=2 fusion (+18% at K=1024) |
| hand word classifier | 29 ns/word vs 67 ns (DFA) / 83 ns (PikeVM) |
| high-bit buckets + fmix32 | chi²/dof 1.006 at D = 2^17; XXH3 rejected |
| constant-K JIT plan | ~5.3 µs/sample bulk; BEAM=2 adds 5.8× on the B=1 graph |
| fp16 end-to-end graph | no cast kernels on the replay path; exact per-value loads, wide LID logit margins |
| zero-copy featurization | rayon writes indices directly into the plan's host-mapped buffer |
| numpy-vectorized training featurizer | batch featurization asserted bit-identical to the scalar contract |

[whichlang]: https://github.com/quickwit-oss/whichlang
[svod]: https://github.com/npatsakula/svod
