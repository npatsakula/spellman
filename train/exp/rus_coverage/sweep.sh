#!/bin/bash
# Sparse-trainer sweep: v15 settings, two --train-seed values per arm, every
# model scored on v15's gated test split and the referees.
# usage: sweep.sh ARM[:extra train flags] ...   (ARM = mix dir under data/)
set -uo pipefail
cd "$(dirname "$0")/../.."
REFS="model-v15/eval_test.tsv tatoeba_eval.tsv rusentitweet_eval_v2.tsv cosmus_rus_eval.tsv short_eval.tsv lit_rus_eval.tsv"
mkdir -p data/exp data/logs
for spec in "$@"; do
  arm="${spec%%:*}"; extra=""; name="$arm"
  if [[ "$spec" == *:* ]]; then extra="${spec#*:}"; name="$arm$(echo "$extra" | tr -d ' -' | tr '=' '_')"; fi
  for seed in ${SEEDS:-1 2}; do
    out="model-exp-$name-s$seed"
    [[ -f "data/exp/$name-s$seed.json" ]] && continue
    uv run spellman-train train --data "data/$arm" --out "$out" \
      --log2-d 18 --k 512 --dim 128 --epochs 6 --lr 0.05 --per-lang-cap 120000 \
      --cap-override rus=480000 --train-seed "$seed" --device cuda $extra \
      > "data/logs/train-$name-s$seed.log" 2>&1 || { echo "FAILED $out"; continue; }
    uv run python exp/rus_coverage/score.py "$out" "data/exp/$name-s$seed.json" $REFS
  done
done
