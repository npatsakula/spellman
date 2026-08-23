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
