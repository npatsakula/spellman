#!/bin/bash
# v16 recipe — the v15 recipe with Russian's short-row supply replaced and
# the lexical twin gate on (docs/experiments.md, 2026-10-03/04):
#   * the `ukr_tweets … twitter_lang=ru` lane (Ukrainian users' tweets that
#     Twitter tagged ru: mostly Ukrainian or undecidable) leaves the mix,
#     except its clearly Russian rows (lexical score <= -3);
#   * 12,000 rows from each of three open short-Russian pools take its slot
#     (Telegram chat, the Dvach chat, forum messages; 3-19 chars, exported
#     once to cache/short-rus/ by export_short_rus.py);
#   * `--lex-gate 4` drops rows whose words are a twin language's.
# Training is v15's, unchanged.
#
# Not a --from-manifest replay: a lane is removed, which replay cannot do;
# exp/rus_coverage/mix_short_rus.py rebuilds the argv from data/v15's.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --with zstandard python exp/rus_coverage/export_short_rus.py 12000 25000
uv run python exp/rus_coverage/mix_short_rus.py data/v16 12000
uv run spellman-train train --data data/v16 --out model-v16 \
  --log2-d 18 --k 512 --dim 128 --epochs 6 --lr 0.05 --per-lang-cap 120000 --dense \
  --device "${DEVICE:-cuda}"
