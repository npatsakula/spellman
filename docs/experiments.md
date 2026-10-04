# Experiments log — what worked, what didn't, and why

Every attempt on the road to the current model, in order, with the
measured outcome. Written so that (a) we never re-run a measured
negative, (b) the reason behind each design choice in the codebase is
findable, and (c) the referee deltas are read against their noise
floors (rst-v2 n=2606 → 1 row = 0.038pp; short twins n=574 → 0.17pp;
COSMUS n=2808; Tatoeba n=37,051 — anything within ~±0.2pp on the small
referees is weather, not signal).

Referee key: **held-out** = pristine test split of the training mix ·
**rst-v2** = 2,606 label-audited wild Russian tweets · **short** = 574
orthography-certified ≤19-char ukr/rus twins · **tatoeba** = 37,051
out-of-domain · **cosmus** = 2,808 gold-labeled wild Russian.

## Shipped milestones

| model | what | held-out | rst-v2 | short | tatoeba | cosmus |
|---|---|---|---|---|---|---|
| v3 | word/bigram lexical channel | 98.28 | 93.73* | — | 98.66 | — |
| v4 | wild-UGC lane (26 sources) | 98.62* | 91.17 | — | 98.57 | 94.66 |
| v5 | short lane (3–19-char wild rows, twins referee built) | — | 92.06 | 89.20 | — | — |
| v6 | + diverse lanes rus/ukr (pymorphy) | 98.30 | 92.56 | 89.90 | 98.42 | 96.79 |
| v8e | + Turkic diverse lanes, real normalizers everywhere, diverse v2 selection | 98.31 | 93.40 | 90.07 | 98.39 | 96.72 |
| **v11c (current, shipped 2026-08-22)** | + cap 32k, diverse budgets ×1.5, 12 diverse lanes (algo=6), wikisource literary lane, short-floor 0.40 | **98.56** | **94.47** | **91.29** | **98.78** | **97.15** |

\* not comparable across rows: v3–v4 measured on different mixes/referee
versions; deltas were verified within-row at the time. v11c's held-out is on
the re-baselined 32k-cap test split (352,470 rows) — not comparable to the
98.31 of the 16k-cap era. Literary referee (new in v11c): 93.90 → 97.15.

## Diverse-selection algorithm (the 2026-08-20/22 arc)

The question: how to turn a Zipfian pool into a small, maximally useful
training set, keyed by real lemmas.

1. **v1 — shuffled first-come greedy** (v6/v7): seeded shuffle, accept
   on ≥2 new lemmas, exposure floors for the top-15k. The baseline that
   worked. Weakness: no global statistics, first-come accepts eager rows.
2. **v2 — lazy-greedy weighted coverage (Minoux)**: **FAILED.** Max-first
   optimization harvests the globally richest sentences and length-biased
   the lanes to death — rus rows came out mean 149 chars with **zero**
   rows ≤40 chars (v1: mean ~95, 10–12% short) and every referee
   regressed (short −1.9pp). Lesson: coverage must not be an objective
   to maximize; it works as an acceptance *filter*.
3. **v2 + min_df=1** (singletons count): **FAILED** — ukr→rus held-out
   confusions 217→426. Singleton tail is trap mass; df≥2 weighting is
   protective.
4. **algo=4 — stratified first-come** (v8e): v1's rule + length-stratum
   budget caps + df≥2 weighting + real normalizers. **SHIPPED.**
   rst 92.94→93.40, held-out 98.31, ukr→rus back to 222.
5. **algo=5 — 20-char bands + per-(lemma,band) exposure spread**: top
   vocabulary must appear in ≥3 length regimes. Trade, not win: short
   90.07→90.77 and cosmus 96.72→96.90 (both best-ever), rst 93.40→92.71.
6. **algo=6 — fill pass**: after coverage saturates, shuffle-variety
   rows top up the budget. Born from the short-diverse finding below.

algo=5/6 are the defaults since v8e's cycle, and **v11c's 12 diverse lanes
were built with algo=6** (verified against the cache fingerprints: the
budget-20000 rus lane records `algo=6, norm=pymorphy3`).

## Short-diverse (the strongest signal we didn't ship)

Coverage-selecting the **wild short pools** (57k ukr tweets, 19k kaz)
instead of feeding them raw: the pure coverage head (7.3k rows covering
100% of df≥2 lemmas) **halved ukr→rus held-out confusions (219→100)** —
the largest single-effect measurement in the project. But it collapsed
register breadth 8× and cost the twins referee 2.3pp. The
concentration/variety dial (fill pass) is monotone at every measured
dose — no sweet spot. **Kept out of the shipped recipe.** Unlock path:
coverage head + wild-register augmentation of the tail.

## New diverse lanes that failed (measured, do not re-add blindly)

- **bel/bul/srp/mon lanes from FineWeb-2 pools** at 12k and reduced
  4–6k budgets: regress short-wild referees up to −1.9pp. Clean
  encyclopedic register for close-pair Slavic classes distorts the
  rus/ukr/bel boundary. These languages need *wild* pools.
- **mkd/oss/che/mhr/udm/kpv lanes** (wave B): short −2.4pp. Same
  mechanism: 12k clean rows for tiny classes is a class-competition
  bomb. v7's Turkic lanes worked because their pools were wild-sourced
  (kazsandra/leipzig/uz-crawl), which is the actual pattern.

## Earlier negatives (kept for the record)

- **rare_rescue tier** (scan rare-bearing sentences first): every
  referee regressed. The singleton tail of a big pool spans far more
  sentences than any budget — arbitrary singletons stay uncovered
  anyway while displacing mid-frequency coverage.
- **Close-pair specialist cascade** (prototype + 200-error audit):
  oracle ceiling ~5% capture; remaining errors are knowledge/ambiguity
  (30% content-free, 25% valid-in-both), not boundary dilution.
- **ukr_tweets 131k `lang=ru` rows** added wholesale: worse everywhere
  (Ukraine-dialect Russian displaced better rus rows under the cap).
- **Diverse for Slavic via FW2 at any budget** — see above.
- **spylls/hunspell for stems**: no stemming API (lookup/suggest only).
- **apertium-oss**: 757-entry stub dictionary. The other Apertium
  minority repos (tat/sah/bak/chv/tyv/udm/kpv) are HFST `.lexc`, not
  lttoolbox — an hfst build is the future option there.

## Tooling discoveries

- Stanza's Serbian models are **Latin-script** (Cyrillic in → POS=X);
  we transliterate on the way in — lemma script is irrelevant for
  grouping keys. bel/bul/kaz/kir models are Cyrillic-native and good.
- Stanza has Kazakh and Kyrgyz lemma models (nothing else does).
- lttoolbox builds locally via cmake in minutes (`spellman-train
  prepare-apertium`);
  mkd lemmatization quality is excellent (најубавиот→убав, including
  apertium `+`-compound heads; lt-proc eats non-tokenizable inputs like
  `km²` — identity-fill handles it).

## What's deliberately not in the model

- The church/bible corpus lane (Tatoeba counterweight where GlotLID
  still leads 99.25 vs 98.78).
- Content-free → uncertain runtime policy (URL/mention-only rows).
- bashkir-web-corpus (gated HF dataset, needs manual acceptance).

## v11c PROMOTED 2026-08-22 — current model

Published (README + HF card, verified by sha256 + fresh-download eval):
held-out 98.56 (352,470 rows, re-baselined 32k-cap split), rst-v2 94.47,
short 91.29, literary 97.15 (new referee row), tatoeba 98.78
(ladder 70.6/88.2/94.5), cosmus 97.15; ukr→rus 202. Quant gate: all four
variants exactly 98.56. Baselines on the new split: GlotLID 93.28
(buckets 77.0/92.1/97.6 — spellman leads every bucket), lid.176 82.91.

## The literary-register arc (2026-08-22, post-v8e)

detect_md over Война и мир showed 6.1% of sentences as tyv — diagnosis:
literary vocabulary is OOV for the rus class (князь in 5/16k rus training
rows, поручик/батальон in zero), so minority classes won on n-gram scraps.
Fix: `wikisource` source adapter (ru.wikisource dumps, sentence-windowed,
eval-corpus titles banned from the pool) + a literary diverse lane
(budget 6000, pymorphy keys).

- v11a (lit lane at cap 16k): lit referee 93.9→95.6 BUT rst −1.3 /
  short −1.6 — under the cap, literary rows *displaced* wild-register
  rus rows (the v8f lesson, mirrored).
- v11b (cap 32k + diverse budgets ×1.5 + lit lane, train 410k→767k):
  displacement gone — rst 94.05, cosmus 97.47, tatoeba 98.74, lit 97.70,
  all best-ever; short −1.2 remained because the 25% short floor at 32k
  wanted 8k short rows but rus only has ~5.3k (share diluted 25%→17%).
- **v11c (v11b + --short-floor 0.40): the sweep** — vs shipped v8e:
  rst 93.40→**94.47**, short 90.07→**91.29**, lit 93.90→**97.15**,
  tatoeba 98.39→**98.78**, cosmus 96.72→**97.15**, ukr→rus 222→202;
  voyna sweep tyv 6.1%→0.2%. Every referee ahead. (held-out 98.56 is on
  the re-baselined 32k-cap test split — not comparable to the 98.31
  published number.)
- Cap-raise answer (the 16k→32k question): no measurable imbalance
  damage — small-class F1s held, close-pair referees improved, because
  the diverse budgets scaled with the cap; the risk lives in register
  *composition* (short-floor), not class share.

## v12 — the commercial-clean rebuild (2026-08-23, IN FLIGHT)

License audit (agent-verified) found NC data in v11c: Leipzig community
crawls ×5 (CC-BY-NC-SA), the-cramer kir news (CC-BY-NC-4.0), muhtasham
tajik-corpus (Leipzig-derived), and NC slices inside every used Glot500
config (Leipzig_* 12–59% by config, nllb_other_til/TIL 2–19%). Replacements
(all commercial+redistribution-clean): HPLT2.0 kir 676k/tgk 1.26M docs
(CC0), alifbank Tajik sentences 1.55M (MIT — cleanest corpus ever ingested:
75 hygiene drops), MADLAD ce/ky/tg (ODC-By), wiki ce/ky/tg (CC-BY-SA),
tahrirchi uz-books-v2 cyr 21GB (MIT), Glot500 uzb_Cyrl Earthlings tweets
(GPL-3, allowlisted). averoo/kyrgyz_mono proven a HPLT+kywiki re-pack (skip
the unstated-license re-pack, use upstreams). NLLB bitext license disputed →
excluded; Tanzil non-commercial → excluded; GoURMET (CC0) dropped — host
unreachable from our network, optional re-add.

Adapter support: hf `exclude=col=v1|v2` (prefix `*`), `no_chars` on
hf/fineweb2/diverse-pools, fineweb2 `langs_exclude` (backbone split so the
ў gate can't hit Belarusian), hf `txt`→text builder.

- **v12 first pass** (recipe: `train/recipes/v12.sh`): splits
  767814/423160/363405, tgk/kir/che/uzn all at the 32k cap. Referees vs
  v11c: tatoeba 98.76/98.78, cosmus 97.26/97.15, lit 97.65/97.15 (+0.50),
  rst 94.13/94.47, short 90.24/91.29. **Same-split held-out: v12 97.73 vs
  v11c 97.37 (+0.36)** — v11c's 98.56 was on its own easier (Leipzig-clean)
  split. Seed-43 probe (same data): held-out 97.75, rst 93.90, short 90.77
  → measured seed spread rst ±0.23 / short ±0.53; the v12 deltas on frozen
  referees sit within ~1–1.5× spread (short delta = 3 rows of 574).
- **Label-noise diagnosis** (error audit): the new crawls carry mislabeled
  rows the model correctly rejects — Russian inside uzn (uz-books
  translations, CGLU tweets) and che (MADLAD ce crawl: Moscow portals,
  legal sites); genuine Sakha/Tuvan text inside HPLT kir (kir→sah/tyv
  errors are the model being RIGHT). ~770 such rows in the v12 test split
  alone explain most of the 97.73-vs-98.56 held-out gap. Hygiene missed
  them structurally: conf 0.90–0.99 < 0.995 bar; kir↔sah/tyv twin-
  protected by design; <8-token rows.
- **Orthographic gates (empirical, trusted-corpus-validated)**: tgk
  no_chars=ўы (ы: 0.08% trusted vs 3.7% MADLAD-tg; Tajik alphabet has no ы);
  kir HPLT + pool no_chars=ҕһ (Sakha-only; 0.00% trusted, 1.2% HPLT).
  uzn/che NOT orthographically separable from Russian (shared alphabets —
  ы even appears in 1.4% of genuine uz-crawl rows); lever left unpulled:
  judge pass at --conf 0.90 for che/uzn (rus not in their twin groups).
- **Status**: gates wired into the recipe; gated-cache rebuild + re-mix +
  retrain NOT yet run on the final recipe — pick up there.

## v12 completion — gated rebuild, rus-register recovery, promotion (2026-08-24)

Gated pipeline run end-to-end (fresh caches): prebuild recovered from a
race (tgk diverse lane pins the alifbank pool that a LATER recipe line
builds — prewarm alifbank first), hygiene dropped 18,445/8.79M rows
(alifbank exactly the documented 75; HPLT tgk pre-gate was 33% uzn —
38,565+7,436 judge drops; the ўы-gated sources needed ≤123), gated mix
767,995/429,114/367,762. tgk F1 1.00 on 32k test rows, kir 0.99, tgk
gone from top confusions — the gates did their job.

- **First-pass non-reproduction**: this machine's builds (gated s42/s43,
  pre-gate) all land lit 94.8–95.7 / cosmus 96.3–96.4 vs the GPU machine's
  first pass 97.65/97.26. Excluded by direct evidence: the gates (pre-gate
  reproduces identically), seed weather (±0.45 lit probe), upstream drift
  (all mutable repos' commits predate the first pass; both machines built
  caches the same week), judge identity (HF model reproduces v11c's five
  referee numbers exactly), adapter drift, source order (63 shared sources
  with v11c, one dedup-neutral inversion; crc32 splits make order unable
  to change split counts). Positive finding: the first pass's val/test
  counts imply a ~44k-smaller uzn pool — the committed recipe ≠ the
  first-pass invocation in ≥1 uzn spec. OPEN: retrieve the GPU machine's
  data/v12/manifest.json; residual suspects are CUDA-vs-MPS numerics and
  that spec gap.
- **Root symptom** (error-profile diff vs v11c): v12's cleaner, fuller
  minority classes absorb ambiguous rus (lit errors rus→che 3→18,
  udm 10→17, tyv 7→10). rus train is cap-bound at 32k in both mixes —
  v11c and v12 train the same rus rows; the competition around them
  sharpened.
- **Recovery levers, one at a time** (all on frozen referees): hard
  negatives scaled 1.7k→4.3k (docs 60k, per-lang 8k, per-doc 4, conf
  0.95; yields rus 2,030 + oss 1,176 boundary rows): lit 94.90→95.70,
  cosmus +0.28. Wikisource lit lane budget 6k→12k: short +0.87, cosmus
  +0.22, rst −0.26 (mild wild-row displacement). Hard negatives 4.3k:
  lit 96.20, short 90.24, cosmus 96.69. Final v12 referees vs v11c:
  tatoeba 98.64/98.78, cosmus 96.69/97.15, lit 96.20/97.15,
  rst 93.28/94.47, short 90.24/91.29; held-out same-split +0.62 over
  v11c (97.52 vs 96.90). GoURMET (CC0, 22,410 ky pairs, 0 hygiene drops)
  re-added to the recipe — referee-neutral, kir robustness.
- **Promoted 2026-08-24** (user-approved): model = train/model-v12hn2;
  quant gates int8-row/col fp8-row/col all 97.50–97.51 (≤0.02pp drop);
  dataset card now renders per-upstream licenses (publish.py LICENSES);
  manifest committed at train/data/v12/manifest.json. GlotLID re-scored
  same-split via a batched MPS port of its softmax scorer added to
  eval-fasttext --device (49 s for 368k rows, parity 1992/1992):
  90.47 vs spellman 97.52 — spellman leads every length bucket. Unpulled
  levers for wave 2: uzn/mon in hard-negative TARGETS, cap raise past
  32k, the conf-0.90 uzn/che judge pass.

## v13 — the six crawl datasets + the cap raise (2026-08-30)

Data: `vpermilp/lid-{sah,tyv,kpv,mhr,oss,udm}` (crawl/dataset.py: FireCrawl
news/library + Telegram + VK, ~907k rows, MIT) appended to the promoted v12
manifest as 11 `hf:` raw lanes (`recipes/v13.sh`). Built cold on the
RTX 3060 box: Tatoeba from the 2026-08 export re-wrapped as
`vpermilp/tatoeba-sentences` (downloads.tatoeba.org throttles to ~640 B/s),
OPUS translatewiki/GoURMET from the moses zips re-wrapped as
`vpermilp/opus-*` (nlpl.eu/pouta unreachable from this network), the scaled
hard negatives regenerated with the promoted judge (4,846 rows; the original
4.3k file was never committed). Hygiene 16,848/9.73M drops. All numbers
single-seed (42); seed spread from v12: rst ±0.23, short ±0.53.

| model | cap | held-out (v13a split) | tatoeba | rst v2 | cosmus | lit | short |
|---|---|---|---|---|---|---|---|
| v12 (promoted) | 32k | 97.89 | 98.64 | 93.28 | 96.69 | 96.20 | 90.24 |
| v12-local (control, this box's inputs) | 32k | 97.91 | 98.60 | 92.79 | 96.47 | 96.45 | 86.59 |
| v13a | 32k | 97.94 | 98.67 | 93.75 | 96.58 | 97.55 | 85.71 |
| v13b | 80k | 98.24 | 98.98 | 96.05 | 97.29 | 98.60 | 93.21 |
| **v13c** | **120k** | **98.34** | 98.98 | **96.70** | **97.36** | **98.65** | **94.25** |
| v13d = v13c + FW2 30k-doc top-up (thin 7 + rus) | 120k | 98.27 | **99.07** | 96.20 | 96.72 | 98.10 | 92.86 |
| v13e = v13c + FW2 top-up (thin 7 only) | 120k | 98.30 | **99.07** | 96.24 | 97.22 | 98.35 | 93.38 |

- **The six languages were never the bottleneck on the referees**: v12
  already scores F1 0.99–1.00 on the new sah/kpv/mhr/oss/udm test rows
  (tyv 0.98→0.99). The crawl text is long and orthographically loud. The
  data still matters — it fills all six to the cap (23/26 languages now
  cap-bound at 32k) — but the visible gains came from the cap.
- **v13a's short collapse is input drift, not the data**: the v12-local
  control (same recipe, this box's regenerated hard negatives + 2026
  Tatoeba dump) already sits at 86.59 with ukr→rus 43 errors. Against
  that like-for-like baseline v13a is +0.96 rst / +1.10 lit, neutral
  elsewhere.
- **The cap is the lever.** 32k→80k→120k moved every referee monotonically
  and no language lost F1 (rus 0.84→0.86, ukr 0.94→0.96, fra 0.92→0.96,
  srp 0.97→0.98 at 120k). Mechanism for the short referee: the mix holds
  200k/100k real ≤19-char ukr/rus rows and `--short-floor 0.40` admits only
  0.4×cap of them — 12.8k at 32k, 48k at 120k. Uncapped train pools: tgk
  1.42M, eng 883k, chv 845k, deu 777k, uzn 641k, kir 579k … rus 121k, oss
  122k … fra 58k, bak 44k, mkd 35k, bul 32k, mon 24k, bel 23k, srp 22k.
  Past 120k the cap stops adding rus (pool exhausted) while the big
  classes keep growing — the imbalance turns around.
- **FineWeb-2 top-ups hurt.** The thin tail is thin only because the
  backbone takes 3,600 docs/config; 30k docs (~116k rows/lang) is a
  minutes-long stream. But web register at 5× backbone volume dilutes the
  wild/literary rows inside the cap: with rus included every wild-rus
  referee dropped (v13d); tail-only (v13e) still lost lit −0.30 / short
  −0.87 and even the tail's own F1 (srp/bel/bul/mkd/fra −0.01…−0.02 on
  the non-web test split). Register composition, not class share — v12's
  lesson again. Lever left unpulled: register-matched tail sources
  (Serbian/Belarusian/Macedonian UGC, Mongolian social) instead of FW2.
- θ: 0.69 (v12) → 0.76 (v13b) → 0.78 (v13c); the confidence scale
  tightens with the bigger sample.
- **Promoted 2026-08-30 (user-approved): v13c.** On its own 719k-row
  test split 98.55 (v12: 98.10), ≤20 chars 93.38 vs 90.52; F1 up or equal
  for all 26 languages (fra 0.86→0.94, bul 0.94→0.97, mkd 0.96→0.98,
  rus 0.83→0.85). Quant gates int8-row/col, fp8-row/col all 98.34–98.35
  vs f16 98.34 on the v13a split. Dataset published as vpermilp/spellman
  (4,214,787 rows); previous promoted artifact kept at train/model-v12-promoted.
- Pipeline fixes made on the way: manifest argv recorded without the `mix`
  token (replay of data/v12/manifest.json failed with "unrecognized
  arguments: mix"), `clean --jobs N`, tatoeba/opus Hub fallbacks,
  lid-* licenses in publish.py, lid-udm card (empty `moderated` config
  broke `load_dataset` for every config).

## v14 — the architecture-review ladder (2026-08-31, PROMOTED)

A model-critique agent reviewed the pipeline; every top recommendation was
measured (train-only cycles off the v13c mix, scored on the v13a-split
held-out file + the five frozen referees):

| exp | held-out | tatoeba | rst | cosmus | lit | short |
|---|---|---|---|---|---|---|
| v13c baseline | 98.34 | 98.98 | 96.70 | 97.36 | 98.65 | 94.25 |
| weight-decay 0 | 97.26 | 98.52 | 93.67 | 98.01 | 96.90 | 92.16 |
| log2-d 18 | 98.39 | 99.09 | 96.82 | 97.40 | 98.75 | 94.77 |
| log2-d 19 | 98.41 | 99.09 | 96.89 | 97.72 | 99.00 | 92.51 |
| k 512 | 98.37 | 98.92 | 97.05 | 97.40 | 98.70 | 94.95 |
| short-floor 0.5 | 98.36 | 99.00 | 96.85 | 97.26 | 98.70 | 94.77 |
| d18+k512 | 98.40 | 99.00 | 96.97 | 97.44 | 98.80 | 94.43 |
| **d18+k512+floor0.5 (v14, s42)** | 98.41 | 99.01 | 96.89 | 97.40 | 98.90 | 94.95 |
| same, seed 43 | 98.41 | 98.94 | 96.70 | 97.33 | 98.50 | 95.47 |

- **2^18 confirmed the collision analysis** (measured ~36 distinct
  keys/signed bucket on a 24% train sample → ~100+ full-corpus): the only
  single change up on all six columns. **2^19 overshoots** — best
  long-text numbers, short −1.7 (features fragment/memorize).
- **weight-decay 0 rejected decisively** (train loss 0.125→0.058, wild
  referees collapse): torch's silently-inherited 0.01 default is genuine
  regularization against collision noise. Now an explicit
  `--weight-decay` flag documenting the choice.
- **k=512** (57.8% of rows were truncated at ~48 chars at k=256): biggest
  single short gain (+0.70) and rst +0.35.
- Combo gains don't fully stack (both attack short); the triple is the
  best overall row and its seed-43 replicate holds (short 95.47 — both
  seeds above every single-change run).
- **Zero-cycle findings**: θ by error-detection F1 beats the
  5th-percentile quantile (v14 val: F1 0.525 vs 0.484, flags 2.9% vs
  5.0% — shipped θ=0.67); logit prior adjustment inert; per-length bias
  fitted on val overfits its augmented register (rejected); train↔test
  near-dup leakage measured at **2.83%** (MinHash J≥0.5, same-lang) —
  held-out deltas <0.1pp are not trustworthy alone, frozen referees
  remain the decision metric; doc-level split keys are the eventual fix.
- **Promoted 2026-08-31 (user-approved): v14** = v13f mix (v13c manifest
  + `--short-floor 0.5`; test split byte-identical to v13c's, so
  same-split baselines carry over) trained `--log2-d 18 --k 512`.
  Own-split 98.62 (v13c 98.55, GlotLID 92.57, lid.176 81.54 same-split),
  ≤20 chars 93.79. Quant gates ≤0.01pp; **root artifact switched to the
  int8-row store** (8.9 MB vs 15.7 MB f16 at 2^18); an `f16/` subdir
  joins the store lineup. Referee lore for the future: wild-referee
  deltas of the winning row vs plain d18 are within seed spread — the
  durable wins of the whole campaign are short (+0.7–1.2) and held-out
  (+0.07 stable across seeds).

## v15 build + the Russian-coverage sweep (2026-10-03, branch exp/rus-coverage)

Scripts: `train/exp/rus_coverage/`. All arms scored through the Rust
runtime on v15's gated test split (718,479 rows) and the referees.

- **v15 built** (`recipes/v15.sh`, dense, torch 2.13.0+cu130 — the box's
  lock, not 2.14.1): 0/99 cold caches; gate dropped 8,053 rows (rus->ukr
  7,132); epochs 90-92 s; quant gate 0.9769 -> 0.9775; theta 0.819
  (5th-percentile, not yet F1-recalibrated). v14 -> v15 on the gated
  split: 98.66 -> 98.69, <=20 94.12 -> 94.38, rus 91.90 -> 92.31, ukr
  95.81 -> 96.87; rst 96.78 -> 96.55, cosmus 97.36 -> 97.93, short 94.77
  -> 97.04, tatoeba 99.03 -> 99.00.
- **The gate leaves short Ukrainian behind**: 400 of v15's 754 rus test
  errors are predicted ukr, 379 of them <=20 chars, 3 with any of і ї є ґ;
  ~70 of an 80-row sample read as Ukrainian. Rus test accuracy is
  therefore a noisy metric; rusentitweet/COSMUS are the cleaner ones.
- **Unseen words: narrow effect.** Rows whose words are *all* absent from
  train fail 4-10x more often (rus test minus rus->ukr rows: 13.9% vs
  3.75%; rst 20.4% vs 3.1%) but are ~5% of rows and hold ~1/5-1/4 of the
  errors; rows with *some* unseen words fail no more often above 20
  chars. Ukrainian shows the same shape.
- **The uniform cap does not bind for rus**: 113,250 train rows in v15.
  `--cap-override` alone is a no-op; arms add FineWeb-2 rus sentences
  (pool of 1.5M, ~186 MB streamed), hygiene-cleaned.
- **Sweep (sparse trainer, 2 train seeds per arm; s1 / s2):**

  | arm (rus train rows) | test | rus | rst | cosmus | tatoeba | others->rus |
  |---|---|---|---|---|---|---|
  | v15 base (113k) | 98.63 / 98.61 | 92.83 / 93.27 | 97.58 / 97.43 | 97.72 / 97.61 | 98.90 / 98.93 | 0.30 / 0.31 |
  | random +70k (195k) | 98.61 / 98.60 | 93.58 / 93.65 | 97.74 / 97.89 | 97.72 / 97.65 | 98.87 / 98.90 | 0.37 / 0.37 |
  | diverse 80k (194k) | 98.60 / 98.60 | 93.75 / 93.92 | 97.89 / 98.12 | 97.72 / 97.79 | 98.87 / 98.82 | 0.42 / 0.40 |
  | random +100k (228k) | 98.59 / 98.60 | 94.27 / 94.12 | 98.23 / 98.12 | 97.86 / 97.83 | 98.83 / 98.82 | 0.41 / 0.40 |
  | random +300k (458k) | 98.54 / 98.57 | 94.19 / 94.17 | 98.35 / 98.04 | 98.04 / 98.08 | 98.73 / 98.77 | 0.48 / 0.47 |
  | base + lexical dropout 0.3 | 98.62 / 98.65 | 92.79 / 92.94 | 97.35 / 97.47 | 97.26 / 97.58 | 98.92 / 98.91 | 0.30 / 0.31 |
  | 228k + lexical dropout 0.3 | 98.58 / 98.61 | 94.63 / 93.34 | 98.16 / 97.74 | 98.04 / 97.65 | 98.81 / 98.83 | 0.42 / 0.39 |

  More Russian helps Russian (+1.1 test, +0.7 rst at 228k) and saturates
  there; overall test never improves and tatoeba and the others' drift to
  rus worsen monotonically with Russian's share. Diverse selection beats
  random at equal size by ~0.2 on rus/rst, within reach of seed spread.
  Lexical dropout does nothing for Russian (test <=20 +0.2 overall).
- **Bias correction rejected** (`bias_sweep.py`, float model.pt): the
  rus-logit shift that returns the 228k arm's drift to baseline (0.5-0.75)
  drops rus to 90-91%, below the baseline's 93.1.

## Lexical twin gate (2026-10-03, branch exp/rus-coverage)

`mix --lex-gate NATS` (`lexgate.py`, lexicon `seeds/lexgate/rus-ukr.tsv.gz`:
314k words, log-odds measured on trusted lanes + 1.5M FineWeb-2 sentences
per side). On the v15 recipe at 4 nats: **dropped 11,998 rows — rus->ukr
11,353, ukr->rus 645**, after the ortho gate's 8,053.

- All 11,353 come from one lane, `ukr_tweets` read with `twitter_lang=ru`
  (40,941 rows): ortho drops 7,147, the lexical gate another 11,353, and
  the remainder is still mostly not Russian (10.7% score 1-3 towards ukr,
  29.6% carry no lexicon word, 19% lean rus). Hand audit of ~250 sampled
  rows: drops at >=3 nats read Ukrainian throughout.
- It also finds label noise in the referees' sources: 95 of COSMUS's 2,808
  "russian" rows are Ukrainian, and **2,172 of the 2,808 COSMUS referee
  rows sit in the train split** (lane 61) — COSMUS is no longer held out.
- Sparse, 2 seeds each, on the lexically gated test split (717,289 rows):

  | | test | <=20 | rus | rus<=20 | ukr | ukr<=20 | others<=20 | others->rus<=20 | rst |
  |---|---|---|---|---|---|---|---|---|---|
  | v15 mix | 98.67 / 98.65 | 94.54 / 94.39 | 94.77 / 94.74 | 88.42 / 88.59 | 97.07 / 96.74 | 94.52 / 93.91 | 94.80 / 94.64 | 0.61 / 0.68 | 97.58 / 97.43 |
  | + lex gate 4 | 98.70 / 98.70 | 94.72 / 94.75 | 93.90 / 93.60 | 85.05 / 84.46 | 98.21 / 98.21 | 96.89 / 96.85 | 95.13 / 95.19 | 0.33 / 0.33 | 97.01 / 97.08 |

  Ukrainian +1.3 (short +2.6), drift to rus halved, overall +0.04; Russian
  short recall falls (-3.7 on test, rst -0.45): the gate removed 10k of
  Russian's 37k short train rows and nothing replaced them.

## Short-Russian replacement lanes (2026-10-04, branch exp/rus-coverage)

`exp/rus_coverage/export_short_rus.py` + `mix_short_rus.py`: the noisy
`ukr_tweets … twitter_lang=ru` lane leaves the recipe; in its slot go its
2,808 clearly Russian rows (lexical score <= -3) and N rows from each of
three open chat/forum pools (3-19 chars, filtered, deduped; pools under
`cache/short-rus/`): `Den4ikAI/russian_dialogues` (279,542 kept),
`hausmer/dvach_chat` (154,200), `nyuuzyou/ruforum` first 1.5M docs
(69,963). Lexical gate at 4. Rus train rows: v15 113k (37k short), gate
only 101k (27k), N=12k 119k (42k), N=25k 161k (76k).

Sparse, s1 / s2. `test` = the lexically gated test split minus 1,930 rus
rows of the removed lane with undecidable labels (715,359 rows; identical
for every model). `okru` = 5,000 short ok.ru comments, in no mix.

| | test | <=20 | rus | ukr | ukr<=20 | others->rus<=20 | rst | okru | tatoeba |
|---|---|---|---|---|---|---|---|---|---|
| v15 mix | 98.70 / 98.68 | 94.78 / 94.61 | 96.81 / 96.69 | 97.07 / 96.74 | 94.52 / 93.91 | 0.61 / 0.68 | 97.58 / 97.43 | 90.16 / 90.32 | 98.90 / 98.93 |
| + gate | 98.74 / 98.74 | 95.11 / 95.17 | 96.92 / 96.73 | 98.21 / 98.21 | 96.89 / 96.85 | 0.33 / 0.33 | 97.01 / 97.08 | 89.94 / 89.38 | 98.99 / 98.99 |
| + gate + 12k/pool | 98.76 / 98.77 | 95.51 / 96.00 | 96.88 / 97.19 | 98.70 / 98.56 | 97.99 / 97.74 | 0.21 / 0.25 | 96.74 / 97.31 | 93.88 / 94.98 | 98.97 / 98.97 |
| + gate + 25k/pool | 98.72 / 98.74 | 95.12 / 95.42 | 97.20 / 97.38 | 98.50 / 98.45 | 97.63 / 97.55 | 0.29 / 0.30 | 97.35 / 97.51 | 95.80 / 96.10 | 98.91 / 98.90 |

- 12k/pool is the best row: test +0.07, <=20 +1.1, ukr +1.7 (short +3.6),
  out-of-source short Russian +4.2, drift to rus on short rows cut to a
  third. rst is within seed spread of the baseline on one seed and -0.8 on
  the other. 25k/pool buys +1.5 more on okru and gives back <=20 and
  bul/mkd/uzn drift.
- tatoeba others->rus <=20 rises 0.5 -> 0.67 with either size (2-3 rows
  per 1,000).
- **short_eval's Russian rows are not a referee**: 42 of 49 are rows of
  the removed lane (so in train before), mostly transliteration jokes
  ("ви нид ту гоу зэр", "пээмэска", "Ъеъ"); its rus accuracy falls
  80 -> 72 once they leave train.

## v16 candidate — dense build (2026-10-04, branch exp/rus-coverage)

`recipes/v16.sh` = the 12k/pool replacement mix (`data/v16`, train split
byte-identical to the sweep's) + v15's dense training. Epochs 91 s each;
quant gate 0.9776 -> 0.9782; theta 0.828 by quantile, **0.66 by
error-detection F1** (`exp/rus_coverage/theta_f1.py`: val F1 0.536 vs
0.485, flags 2.67% vs 4.81%), written to model.json. NOT published.

Same common test file as above; one dense run per model, three sparse
seeds per mix.

| | test | <=20 | rus | ukr | ukr<=20 | others->rus<=20 | rst | okru | short | tatoeba | lit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v14 (Hub) | 98.74 | 94.79 | 96.42 | 96.08 | 92.40 | 0.85 | 96.78 | 88.10 | 94.77 | 99.03 | 98.95 |
| v15 dense | 98.76 | 95.00 | 96.56 | 97.10 | 94.49 | 0.64 | 96.55 | 88.12 | 97.04 | 99.00 | 98.80 |
| v16 dense | 98.79 | 95.36 | 96.88 | 98.49 | 97.52 | 0.26 | 96.70 | 93.76 | 96.17 | 99.04 | 98.65 |
| v15 mix sparse s1/s2/s3 | 98.70 / 98.68 / 98.72 | 94.78 / 94.61 / 95.03 | 96.81 / 96.69 / 96.35 | 97.07 / 96.74 / 96.99 | 94.52 / 93.91 / 94.38 | 0.61 / 0.68 / 0.59 | 97.58 / 97.43 / 97.35 | 90.16 / 90.32 / 89.68 | 98.26 / 98.08 / 98.61 | 98.90 / 98.93 / 98.96 | 98.95 / 99.05 / 98.95 |
| v16 mix sparse s1/s2/s3 | 98.76 / 98.77 / 98.75 | 95.51 / 96.00 / 95.56 | 96.88 / 97.19 / 96.86 | 98.70 / 98.56 / 98.59 | 97.99 / 97.74 / 97.79 | 0.21 / 0.25 / 0.23 | 96.74 / 97.31 / 97.01 | 93.88 / 94.98 / 94.66 | 97.74 / 97.21 / 97.39 | 98.97 / 98.97 / 98.97 | 99.05 / 99.15 / 99.15 |

rst: sparse v16 is 0.4 below sparse v15 on the 3-seed mean (97.02 vs
97.45); the dense pair goes the other way (+0.15). Dense per-language
moves >= 0.15: ukr +1.39, rus +0.32, srp +0.17, mkd -0.27, bak -0.21,
che -0.17. short_eval's drop is its 42 formerly-in-train rus rows.

## v16 final build — referees held out, HF lanes (2026-10-04)

- **COSMUS fixed.** The COSMUS Russian lane *was* the referee (2,172 of
  2,808 rows in v15's train split, the rest in val/test). `recipes/v16.sh`
  retires the lane (`mix --drop-source`) and `mix --holdout FILE.tsv` keeps
  every referee row out of all splits (141 rows dropped: chv 47, ukr 42,
  rus 19, …). `cosmus_rus_eval_v2.tsv` = the referee minus its 95
  Ukrainian rows (lexical gate >= 4; all 45 lowest-scoring ones read
  Ukrainian) — 2,713 rows. **COSMUS numbers before v16 are not held-out.**
- **The short-Russian pools are plain `hf:` lanes now** (`drop_long=True`
  drops instead of truncating, `sample=N` takes a seeded random N of the
  distinct passing rows) and the tweet lane keeps its Russian rows by
  `lex_own=3` (3,108 rows) — so `spellman-train fetch --manifest
  data/v16/manifest.json` builds everything; no pre-export. New adapter
  options are `late_options`: out of the cache fingerprint at their
  default, so the 97 existing caches stayed warm (5/102 cold = the new
  lanes). ruforum needs `uv run --with zstandard` (not a dependency yet).
- Build: gates ortho 1,242 (rus->ukr 326), lex 744; rus train 115,707
  (41,582 short); epochs 90-91 s; quant gate 0.9778 -> 0.9783; theta 0.67
  by F1 (0.828 by quantile). One dense run per row, common test file:

  | | test | <=20 | rus | ukr | ukr<=20 | others->rus<=20 | rst | okru | cosmus v2 | short | tatoeba | lit |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | v14 (Hub) | 98.74 | 94.79 | 96.42 | 96.08 | 92.40 | 0.85 | 96.78 | 88.10 | 99.52* | 94.77 | 99.03 | 98.95 |
  | v15 | 98.76 | 95.00 | 96.56 | 97.10 | 94.49 | 0.64 | 96.55 | 88.12 | 99.63* | 97.04 | 99.00 | 98.80 |
  | v16 | 98.79 | 95.37 | 96.76 | 98.56 | 97.63 | 0.25 | 96.55 | 93.20 | 99.15 | 95.82 | 99.08 | 98.40 |

  \* trained on most of these rows. lit -0.40 is 8 rows of 2,000 on a
  single dense run (the first v16 build scored 98.65, its sparse seeds
  99.05-99.15).
