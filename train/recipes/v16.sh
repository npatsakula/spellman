#!/bin/bash
# v16 recipe — the v15 recipe (v13f manifest + orthographic gate) with
# Russian's short-row supply replaced, the lexical twin gate on, and the
# referees held out (docs/experiments.md, 2026-10-03/04):
#   * the `ukr_tweets … twitter_lang=ru` lane (Ukrainian users' tweets that
#     Twitter tagged ru: mostly Ukrainian or undecidable) is retired; only
#     its rows whose words are Russian by >= 3 nats come back (`lex_own=3`);
#   * 12k naturally short rows (3-19 chars, dropped not truncated, random
#     sample of the distinct rows) from each of three open chat/forum
#     corpora take its place;
#   * `--lex-gate 4` drops rows whose words are a twin language's;
#   * the COSMUS Russian lane is retired — it *is* the COSMUS referee — and
#     `--holdout` keeps every referee row out of all splits.
# Training is v15's, unchanged.
#
# `--with zstandard`: nyuuzyou/ruforum streams zstd-compressed jsonl, and
# zstandard is not a declared dependency yet.
set -euo pipefail
cd "$(dirname "$0")/.."
NOT_RUSSIAN='іїєґўәғқңөұүһІЇЄҐЎӘҒҚҢӨҰҮҺ'
SHORT="lang=rus,raw=True,min_chars=3,max_chars=19,drop_long=True,cyr=0.6,no_chars=$NOT_RUSSIAN"
uv run --with zstandard spellman-train mix --from-manifest data/v15/manifest.json --out data/v16 \
  --drop-source "ukr_tweets:lang=rus,twitter_lang=ru,min_chars=3,max_chars=19,cyr=0.2,limit=100000" \
  --drop-source "hf:repo=YShynkarov/COSMUS,column=document_content,where=language_manual=russian,lang=rus,raw=True,docs=0,streaming=False,max_chars=512" \
  --source "ukr_tweets:lang=rus,twitter_lang=ru,min_chars=3,max_chars=19,cyr=0.2,limit=100000,lex_own=3" \
  --source "hf:repo=Den4ikAI/russian_dialogues,column=question,docs=0,streaming=False,sample=6000,$SHORT" \
  --source "hf:repo=Den4ikAI/russian_dialogues,column=answer,docs=0,streaming=False,sample=6000,$SHORT" \
  --source "hf:repo=hausmer/dvach_chat,docs=0,streaming=False,sample=12000,$SHORT" \
  --source "hf:repo=nyuuzyou/ruforum,docs=1500000,sample=12000,$SHORT" \
  --lex-gate 4 \
  --holdout tatoeba_eval.tsv --holdout rusentitweet_eval_v2.tsv --holdout cosmus_rus_eval.tsv \
  --holdout short_eval.tsv --holdout lit_rus_eval.tsv \
  --jobs 3
uv run spellman-train train --data data/v16 --out model-v16 \
  --log2-d 18 --k 512 --dim 128 --epochs 6 --lr 0.05 --per-lang-cap 120000 --dense \
  --device "${DEVICE:-cuda}"
