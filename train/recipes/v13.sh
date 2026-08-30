#!/bin/bash
# v13 recipe — v12 + the six spellman crawl datasets (vpermilp/lid-*).
# PROMOTED as v13c (2026-08-30): this file + --cap-per-lang 120000; train with
#   uv run spellman-train train --data data/v13 --out ../model --log2-d 17 --dim 128 \
#     --epochs 6 --lr 0.05 --per-lang-cap 120000 --device cuda
# (v13a = cap 32000, v13b = 80000 — the ladder is in docs/experiments.md.)
#
# Replays the PROMOTED v12 recipe byte-exact from its manifest (source order
# is part of the recipe: dedup is first-source-wins, so appending keeps every
# v12 row's claim intact and the new lanes can only add) and appends, per
# language, the two configs of vpermilp/lid-{sah,tyv,kpv,mhr,oss,udm}
# (MIT; built by crawl/dataset.py from FireCrawl news/library crawls +
# Telegram + VK, cross-source deduped, Russian-sentence stripped, PII masked,
# judged by v12 at conf 0.9 — i.e. already stricter than `clean`'s 0.995):
#
#   moderated  news / library rows (edited, published sources)   ~56-75k/lang
#   wild       tg-post · tg-comment · vk-post (community register) 35-217k/lang
#
# udm has no moderated config (udmdunne.ru is unreachable through FireCrawl;
# the card declares `wild` as its default) — one lane only.
#
# Why raw mode: rows are already single-line, sentence-group/post sized
# (p10 ≈ 30-70, median ≈ 130-280 chars); max_chars=512 is the same clamp
# the v12 raw lanes (OSCAR/MADLAD/tweets) use. Line-window mode would only
# re-slice single lines.
#
# Why --cap-per-lang stays 32000 (v12's): the point of the crawl was class
# balance — in v12 these six were the thin classes, well under the cap.
# Every one of them now overflows it (pools: sah ≈ +276k, tyv +182k,
# kpv +154k, mhr +119k, oss +93k, udm +84k rows before the crc32 split),
# so the cap's seeded subsample makes them full 32k classes like rus/ukr.
# Register composition inside the sample follows pool composition (sah:
# ~⅔ wild, mostly district-paper Telegram posts; kpv/tyv/udm: ~½ VK). To
# weight a register, split a lane with `where=register=<tg-post|tg-comment|
# vk-post>` — but note `docs=N` takes the HEAD of the file (rows are in
# source order, not shuffled), so head-capping a multi-source config is
# biased towards its first sources. Levers for wave 2, one at a time on the
# frozen referees: --cap-per-lang 48000/64000 (experiments.md lists the cap
# raise as unpulled; the train step's --per-lang-cap 50000 then binds too).
#
# Evaluation: the new rows land in val/test through the crc32 split, so the
# held-out file changes — rescore v12 on v13's eval_test.tsv (assess
# --model <v12 dir>) for the same-split comparison, plus the five frozen
# referees (tatoeba / rusentitweet v2 / cosmus / lit / short).
#
# Cold machine: uv sync && bash recipes/v12-pools.sh
#   && uv run spellman-train fetch --manifest data/v12/manifest.json --jobs 3
#   && bash recipes/v13.sh
# Warm v12 caches: just bash recipes/v13.sh (--jobs prebuilds the 11 new
# caches in parallel; ~110 MB of parquet from the Hub, under a minute each).
# Hygiene: `uv run spellman-train clean cache/hf-*.jsonl` afterwards is
# consistent with the v12 procedure but expected to be a no-op on these
# (already judged at 0.9; twin-protected either way).
set -euo pipefail
cd "$(dirname "$0")/.."
uv run spellman-train mix --from-manifest data/v12/manifest.json --out data/v13 \
  --source 'hf:repo=vpermilp/lid-sah,config=moderated,lang=sah,raw=True,docs=0,streaming=False,max_chars=512' \
  --source 'hf:repo=vpermilp/lid-sah,config=wild,lang=sah,raw=True,docs=0,streaming=False,max_chars=512' \
  --source 'hf:repo=vpermilp/lid-tyv,config=moderated,lang=tyv,raw=True,docs=0,streaming=False,max_chars=512' \
  --source 'hf:repo=vpermilp/lid-tyv,config=wild,lang=tyv,raw=True,docs=0,streaming=False,max_chars=512' \
  --source 'hf:repo=vpermilp/lid-kpv,config=moderated,lang=kpv,raw=True,docs=0,streaming=False,max_chars=512' \
  --source 'hf:repo=vpermilp/lid-kpv,config=wild,lang=kpv,raw=True,docs=0,streaming=False,max_chars=512' \
  --source 'hf:repo=vpermilp/lid-mhr,config=moderated,lang=mhr,raw=True,docs=0,streaming=False,max_chars=512' \
  --source 'hf:repo=vpermilp/lid-mhr,config=wild,lang=mhr,raw=True,docs=0,streaming=False,max_chars=512' \
  --source 'hf:repo=vpermilp/lid-oss,config=moderated,lang=oss,raw=True,docs=0,streaming=False,max_chars=512' \
  --source 'hf:repo=vpermilp/lid-oss,config=wild,lang=oss,raw=True,docs=0,streaming=False,max_chars=512' \
  --source 'hf:repo=vpermilp/lid-udm,config=wild,lang=udm,raw=True,docs=0,streaming=False,max_chars=512' \
  --cap-per-lang 120000 \
  --jobs 3
