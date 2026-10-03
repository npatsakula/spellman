#!/bin/bash
# v15 recipe — the v14 mix (v13f) with the orthographic twin gate
# (`--ortho-gate 0.5`, docs/training.md "Data hygiene"): drops rows whose
# spelling contradicts their twin-language label, chiefly Ukrainian tweets
# labelled rus by Twitter's lang tag (6.2% of rus train in v13f).
# Training is v14's, unchanged, so the gate is the only difference.
#
# Compare against v14 on the *gated* test split (data/v15/data/test-*), not
# on v13f's: the gate cleans test labels too, and the old split would
# count v14's correct "ukr" calls on mislabelled rows as errors.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run spellman-train mix --from-manifest data/v13f/manifest.json --out data/v15 --ortho-gate 0.5 --jobs 3
uv run spellman-train train --data data/v15 --out model-v15 \
  --log2-d 18 --k 512 --dim 128 --epochs 6 --lr 0.05 --per-lang-cap 120000 --device cuda
