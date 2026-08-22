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
| v6 (shipped 2026-08-20) | + diverse lanes rus/ukr (pymorphy) | 98.30 | 92.56 | 89.90 | 98.42 | 96.79 |
| **v8e (current)** | + Turkic diverse lanes, real normalizers everywhere, diverse v2 selection | **98.31** | **93.40** | **90.07** | 98.39 | 96.72 |

\* not comparable across rows: v3–v4 measured on different mixes/referee
versions; deltas were verified within-row at the time.

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
   In the codebase for the next cycle; not in the shipped model.
6. **algo=6 — fill pass**: after coverage saturates, shuffle-variety
   rows top up the budget. Born from the short-diverse finding below.

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
- lttoolbox builds locally via cmake in minutes (prepare_apertium.py);
  mkd lemmatization quality is excellent (најубавиот→убав, including
  apertium `+`-compound heads; lt-proc eats non-tokenizable inputs like
  `km²` — identity-fill handles it).

## What's deliberately not in the model

- The church/bible corpus lane (Tatoeba counterweight where GlotLID
  still leads 99.25 vs 98.39).
- Content-free → uncertain runtime policy (URL/mention-only rows).
- bashkir-web-corpus (gated HF dataset, needs manual acceptance).
