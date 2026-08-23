#!/bin/bash
# Pool builders for recipes/v12.sh — run FIRST on a cold machine.
#
# The v12 mix's diverse lanes pin pool caches by filename. Six pools are
# caches of hf: sources with option combos that appear nowhere in the main
# recipe (they predate it), plus the wikisource literary pool. This script
# builds exactly those caches (spellman-train fetch, no mixing), with specs
# recovered verbatim from the original caches' .options.json sidecars
# (verified: option-for-option identical fingerprints).
#
# The tyv pool needs no entry here: it IS the recipe's own Glot500 tyv
# source (NC-slice-filtered, cache hf-cd00eaba11). The kaz pool likewise
# comes from the recipe's kazsandra source. The tgk pool comes from the
# recipe's alifbank source.
#
# NOTE the tyv license fix: the pre-v12 tyv pool was UNFILTERED Glot500
# tyv_Cyrl (~43% NC-licensed Leipzig/TIL slices by row count). v12 points
# the tyv lane at the filtered cache instead — that pool deliberately
# differs from the pre-transfer one.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run spellman-train fetch --jobs 3 \
  --source 'hf:repo=tahrirchi/uz-crawl,lang=uzn,docs=400000,per_doc=4,raw=True,max_chars=512,files=data/telegram_blogs*,cyr=0.6' \
  --source 'hf:repo=AigizK/tatar-russian-parallel-corpora,column=tat,lang=tat,docs=999999,per_doc=2,streaming=False' \
  --source 'hf:repo=AigizK/bashkir-russian-parallel-corpora,column=ba,lang=bak,docs=30000,per_doc=2' \
  --source 'hf:repo=alexantonov/chuvash_mono,column=chv,lang=chv,docs=500000,per_doc=4,raw=True,max_chars=512' \
  --source 'hf:repo=ailabykt/sakha-corpus-mono,lang=sah,docs=8000,per_doc=3' \
  --source 'wikisource:config=20231201.ru,lang=rus,docs=60000,per_doc=40,banned=война и мир|war and peace,min_chars=20,max_chars=200,seed=42'
