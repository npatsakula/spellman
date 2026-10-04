# Benchmarks

The full comparison record for the shipped model. The first table is
v16 (2026-10-04) on its own test split; the Rust-crate, per-language and
throughput sections below it were measured on v14 (2026-08-31) and its
719,255-row split, and say so.
Everything here is measured on identical eval rows per table; the
summary lives in the [README](../README.md#accuracy). Referee files are
frozen and, since v16, held out of the mix by construction
(`mix --holdout`); the held-out file is the mix's own content-addressed
test split (718,647 rows). Numbers marked v12-era were
measured on hardware we no longer have access to.

## Against fastText-family models (GlotLID v3, lid.176)

| eval | rung | spellman | GlotLID v3 | fastText lid.176 |
|---|---|---|---|---|
| held-out mix (718,647, pristine test) | text | **98.74%** | 92.92%‡ | 81.97%* |
| Tatoeba (37,051, out-of-domain) | word / pair / triple | **72.04 / 89.78 / 95.37** | 43.9 / 79.3 / 91.9‡ | 59.0 / 79.0 / 87.9 |
| Tatoeba (37,051, out-of-domain) | text | 99.08% | **99.25%**‡ | 94.90%* |
| rusentitweet (2,606 wild Russian tweets, label-audited) | text | **96.55%**† | 82.73%‡ | 90.41% |
| COSMUS Russian v2 (2,713 wild Telegram/reviews, gold-labeled)§ | text | 99.15% | 98.82%‡ | **99.37%** |
| short utterances (574, orthography-certified ≤19 chars) | text | **95.82%**† | 71.25%‡ | 84.32% |
| literary Russian (2,000 classic-prose sentences, held-out novel) | text | 98.40% | — | — |

§ `cosmus_rus_eval_v2.tsv`: the 2,808-row file minus 95 rows that are
Ukrainian under a "russian" manual label (found by the lexical twin
gate, read by hand). Before v16 the COSMUS Russian slice was also a
training lane — 2,172 of the 2,808 rows sat in v15's train split — so
earlier COSMUS numbers (v14: 97.40%) were not held-out and are not
comparable. Of the 49 Russian rows of the short-utterance referee, 42
came from a tweet lane v16 retired (transliteration jokes such as
"ви нид ту гоу зэр"); they were in training up to v15.

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
By length, held-out same-split: GlotLID 65.2 / 92.1 / 97.8 vs spellman
95.1 / 98.5 / 99.5 (≤20 / 21–100 / >100 — spellman leads every bucket,
by ~30pp on ≤20-char rows); Tatoeba 97.8 / 99.3 / 100.0 (GlotLID leads
every bucket). It predicts at ~355 µs/doc on CPU — two orders of
magnitude slower than spellman. The split is the story: spellman wins
the wild, heavy-Cyrillic workload by 13.8pp on Russian tweets (96.6 vs
82.7) and the single-word rung by ~28pp (2,102-class label entropy is
brutal on short text); GlotLID's far larger training set still wins
clean out-of-domain sentences, by 0.2pp.

## Against the Rust LID crates

[`benchmarks/`](../benchmarks) (standalone crate, `lid-bench`) runs
spellman, [whichlang] and [lingua] on **identical rows** — the full eval
files below (`--rows-per-lang 0`; a seeded 500/language balanced sample
is the default for quick runs). Every tool gets the same texts and its
own language inventory as the detector: lingua is built from exactly the
17 of our languages it supports, with preloaded models — its best shot
on our workload. Rerun:
`cd benchmarks && cargo run --release -- --model ../model --rows-per-lang 0 ../model/eval_test.tsv --by-length --per-lang`.

| detector | our classes | held-out: all rows | held-out: its subset | Tatoeba: all rows | Tatoeba: its subset | µs/sample |
|---|---|---|---|---|---|---|
| spellman (bulk) | 30/30 | **98.62%** | **98.62%** | **99.01%** | **99.01%** | 7.2 |
| spellman (single) | 30/30 | **98.62%** | **98.62%** | **99.01%** | **99.01%** | 15.8 |
| whichlang 0.1 | 10/30 | 27.21% | 90.15% | 32.28% | 99.67% | **1.0** |
| lingua 1.8 (high) | 17/30 | 33.01% | 90.26% | 68.55% | 97.69% | 206 |
| lingua 1.8 (low) | 17/30 | 31.20% | 85.31% | 65.38% | 93.17% | 291 |

(719,255 / 37,051 rows, v14 same-split; AMD Ryzen 9 7950X3D; spellman
k=1024 under BEAM=16; µs/sample from the held-out file in this harness —
its rows are longer than Tatoeba's, where the same path reads
3.5 µs/sample. "all rows" counts gold languages outside a tool's
inventory as errors — what a 30-class Cyrillic workload actually sees.)

**Accuracy by text length** — supported-subset accuracy per char-length
bucket (the buckets `assess` uses; for spellman the subset is all rows):

| bucket | held-out mix (n) | spellman | whichlang | lingua high |
|---|---|---|---|---|
| ≤20 chars | 60,095 | **93.79%** | 84.2% | 76.6% |
| 21–100 | 272,416 | **98.52%** | 87.8% | 91.5% |
| >100 | 386,744 | **99.45%** | 97.0% | 97.8% |

| bucket | Tatoeba (n) | spellman | whichlang | lingua high |
|---|---|---|---|---|
| ≤20 chars | 1,567 | **97.77%** | 97.5% | 92.8% |
| 21–100 | 34,674 | 99.05% | **99.7%** | 97.8% |
| >100 | 810 | 99.88% | 99.5% | **100.0%** |

(The held-out short bucket is large because the verified short-utterance
lane contributes real 3–19-char wild rows to every split.)

What the numbers say:

- **Coverage dominates a Cyrillic workload.** whichlang knows one
  Cyrillic language of our 21 (rus); lingua knows 8. For the other
  languages of the region their answer is structurally wrong, which is
  the 27–69% all-rows column.
- **Short text is lingua's advertised strength — and spellman wins it**:
  on ≤20-char rows spellman leads lingua high-accuracy by 5–17pp on both
  referees (97.8 vs 92.8 Tatoeba, 93.8 vs 76.6 held-out), and lingua's
  low-accuracy mode collapses further. Mid-length is spellman's biggest
  gap over lingua (98.5 vs 91.5 held-out); at >100 chars everyone
  converges to 98–100% and the differences are coverage, not quality.
- **On the languages they share with us, spellman wins the close pairs**
  (held-out, full 719k file, same split): ukr 95.7% vs lingua 86.1, mkd
  97.4% vs 84.2, srp 97.7% vs 96.9, kaz 98.9% vs 94.3, bul 96.7% vs
  94.6, eng 97.4% vs 91.2, bel at parity (98.5 vs 99.0) — every shared
  language is at parity or ahead, with the wild-heavy classes widest.
- **whichlang's 98.0% on Russian is real — and the trade is visible:**
  its 16-class world contains no ukr/bel/kaz to confuse with Russian.
  spellman's rus (90.3% on the wild-heavy v14 719k split) bleeds
  into those close classes — and into the small languages whose real wild
  data now competes — which is precisely the capacity that makes the
  other 20 Cyrillic columns work.
- **Latency**: whichlang is the fastest per document (tiny 16-class
  model) at ~7× spellman bulk; lingua high-accuracy is ~29× slower
  than spellman bulk (206 vs 7.2 µs/sample, BEAM=16).

Per-language on the held-out mix (v14, 719k rows): tgk/mhr/oss/deu/chv/
sah/kpv/tat F1 1.00, uzn/kir/udm/bak/kaz/mon/tyv/srp/bel 0.99 — the
residual confusions are the genuinely hard ones (rus F1 0.85 on the
short-wild-heavy slice, fra 0.94, spa 0.95, ukr 0.96, por/bul 0.97,
eng/mkd 0.98; rus-attraction on short low-resource texts).

## Throughput

On the Apple M4 Max (int8 with per-column scales, the runtime's only
store — see the design doc): `lid-bench` against
`train/tatoeba_eval.tsv` and the 368,507-row v12-era held-out mix
(`model/eval_test.tsv`), BEAM=16, timed threads on performance cores;
ranges over two runs. The one-thread and replica rows run with
`SVOD_THREADS=1` (BEAM tunes those plans for one thread), the
svod-threaded row without it:

| run | Tatoeba | held-out mix |
|---|---|---|
| bulk, svod-threaded kernel, 4096-row batches | 0.75 µs/sample | 0.97–1.00 µs/sample |
| bulk, one single-thread replica per core (14) | 0.18–0.21 µs/sample | 0.44–0.50 µs/sample |
| bulk, one thread | 1.14–1.40 µs/sample | 3.22–3.41 µs/sample |
| single document, one thread | 2.59–2.63 µs/doc | 6.52–6.84 µs/doc |
| whichlang 0.1, one thread (10/30 classes) | 0.35 µs/sample | 1.13–1.21 µs/sample |
| whichlang 0.1, 14 threads | 0.04 µs/sample | 0.12–0.16 µs/sample |
| lingua 1.8 high accuracy, one thread (17/30) | 98.5–102 µs/sample | 193–202 µs/sample |

Replicas beat the svod-threaded kernel because every threaded execute
pays a fixed launch cost (~85–90 µs on 14 threads) that single-thread
replicas never do. Called from inside a rayon worker — replicas, or the
one-thread rows — `detect_batch` also runs its host work inline (nested
`par_iter`s there let workers stack other replicas' batches on top of
their own) and picks the rung with the least gathered work, since an
inline kernel pays no launch cost. The one-thread rows are the
like-for-like comparison with whichlang and lingua, which `lid-bench`
runs on one thread.

**x86 and GPUs** (0.1.0-alpha.7, int8-col, BEAM=16, k=1024, batch
4096): `lid-bench --rows-per-lang 0` over v14's own 719k-row test split
plus Tatoeba, 756,306 rows in one pass — a different mix from the M4
table above, which scored the older v12-era held-out file (368,507 rows;
753 of them are in v14's training data), so compare within a column, not
across tables. The CPU
column is the `SVOD_THREADS=1` run except the svod-threaded row, which
comes from a run without it; GPU columns set `SVOD_DEVICE`. µs per
document:

| run | 7950X3D (16C/32T) | AI Max+ 395 CPU (16C/32T) | Radeon 8060S iGPU (`AMD:0`) | RTX 3060 (`CUDA:0`) |
|---|---|---|---|---|
| spellman bulk, one thread | 4.53–4.63 | 5.17 | 3.00 | 14.01 |
| spellman single document, one thread | 4.67–4.74 | 11.57 | 20.74 | 34.32 |
| spellman, one replica per thread (32) | 0.74–0.75 | 0.44 | 0.47 | 6.18 |
| spellman bulk, svod-threaded kernel | 4.97 | — | 1.54 | 16.72 |
| whichlang 0.1, one thread | 0.97–0.99 | 0.95 | | |
| whichlang 0.1, 32 threads | 0.07 | 0.07 | | |
| lingua 1.8 high accuracy, one thread | 193–200 | 194 | | |

Accuracy is identical on every device (98.64% over the mix). On x86 the
svod-threaded CPU kernel is no faster than one thread (7950X3D 4.97 vs
4.53 µs), unlike on the M4, so replicas are the only multi-core setup
worth using there; the AI Max+ 395's threaded CPU row is left out
because that run's thread setting is not on record. On the GPUs only
the kernel moves: featurization stays on the CPU. The 8060S shares
memory with the CPU and halves the one-thread time; the RTX 3060 is slower
than its own CPU on every row — most likely because the kernel reads the
host-mapped input over PCIe (not yet profiled).

Earlier 7950X3D measurements (v14 on svod alpha.5, threaded bulk, batch
512), other inputs: the 719k-row held-out test split
(longer, mixed-register texts) runs at 2.9 µs/sample and a 1M-row file
of single words at 0.8 µs/sample — the per-call plan ladder scores short
rows on a K=64 plan instead of padding them to 1024. The `detect_md`
example sweeps a 6.2 MB novel at ~540k sentences/s (88 MB/s) with one
replica per physical core (`--mode replicas --threads 16`); a single
detector driven from the main thread reaches ~420k/s.

Before the rework (symbolic batch axis) the same Tatoeba run took
3.5 µs/sample with BEAM and 11 µs without: svod threads a kernel over a
loop axis only when it is a constant a thread count divides, so with a
symbolic batch the only splittable axis was the 30-way class axis — a
6-thread ceiling on a 32-thread machine, visible as idle cores in `htop`.
A batch compiled at 512 splits 32-way. Without the BEAM scheduler the
default plan now runs 2.0 µs/sample. Since svod 0.1.0-alpha.5 the beam search runs in a
separate helper process: `cargo install svod-tensor --bin
svod-beam-worker` and point `SVOD_BEAM_WORKER` at the installed binary,
otherwise `BEAM=16` fails at prepare time with "BEAM helper is
unavailable" (the heuristic default needs nothing). The 2^18 table
costs nothing measurable on the 7950X3D: the v12-era 2^17 model times
identically (3.5 µs) on the same box. Scoring is pure table lookups
after the algebraic fold `P = E·W` — no embedding gathers, no matmul. fmix32 bucket spread
on real n-grams: chi²/dof ≈ 1.006 (uniform ≈ 1.0).

[whichlang]: https://github.com/quickwit-oss/whichlang
[lingua]: https://github.com/pemistahl/lingua-rs
