#!/bin/bash
# v14 recipe — the v13c mix with --short-floor 0.5, trained at 2^18 buckets
# and k=512 training truncation (the 2026-08-31 architecture-review ladder:
# docs/experiments.md, v14 entry — wd=0 and 2^19 were measured and rejected).
#
# The mix replays the promoted v13c manifest (itself = v12 + the six
# vpermilp/lid-* crawl datasets at cap 120k) with the live flag overriding
# the recorded short-floor 0.40 -> 0.50.
#
# theta in the exported model.json is recalibrated afterwards by
# error-detection F1 on val (scripts in the session log / experiments.md)
# instead of train.py's 5th-percentile quantile.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run spellman-train mix --from-manifest data/v13c/manifest.json --out data/v13f --short-floor 0.5 --jobs 3
uv run spellman-train train --data data/v13f --out model-v14 \
  --log2-d 18 --k 512 --dim 128 --epochs 6 --lr 0.05 --per-lang-cap 120000 --device cuda
