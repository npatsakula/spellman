"""The v15 recipe with the noisy short-Russian tweet lane replaced.

`ukr_tweets:lang=rus,twitter_lang=ru` (Ukrainian users' tweets Twitter
tagged ru; mostly Ukrainian or unjudgeable text) leaves the recipe. In its
slot go its clearly Russian rows (lexical score <= -3, `tweets-kept.jsonl`)
and SIZE rows from each short-Russian pool of export_short_rus.py; the
lexical gate runs at 4 nats.

usage: mix_short_rus.py OUT_DIR SIZE [extra mix flags ...]
"""
import json
import sys
from pathlib import Path

import polars as pl

from spellman_train import mix, sources
from spellman_train.lexgate import load_lexicons, score
from spellman_train.ortho import contradicting_twin
from spellman_train.paths import CACHE_DIR, TRAIN_DIR

NOISY = "ukr_tweets:lang=rus,twitter_lang=ru,min_chars=3,max_chars=19,cyr=0.2,limit=100000"
POOLS = ("dialogues", "dvach", "ruforum")
KEPT = CACHE_DIR / "short-rus" / "tweets-kept.jsonl"


def write_kept() -> int:
    if KEPT.exists():
        return sum(1 for _ in KEPT.open(encoding="utf-8"))
    lex = dict(load_lexicons()["rus"])["ukr"]
    name, opts = sources.parse_source(NOISY)
    rows = mix.read_source(sources.create(name, **opts))["text"].to_list()
    keep = [t for t in rows if contradicting_twin("rus", t, 0.5) is None and score(t, lex) <= -3]
    KEPT.parent.mkdir(parents=True, exist_ok=True)
    with KEPT.open("w", encoding="utf-8") as f:
        for t in keep:
            f.write(json.dumps({"lang": "rus", "text": t}, ensure_ascii=False) + "\n")
    return len(keep)


def main() -> None:
    out, size, extra = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    print(f"tweets-kept: {write_kept()} rows")
    recorded = json.loads((TRAIN_DIR / "data" / "v15" / "manifest.json").read_text())["argv"]
    argv: list[str] = []
    for tok in recorded:
        if tok == NOISY:
            argv.append("jsonl:path=cache/short-rus/tweets-kept.jsonl,min_chars=3")
            for pool in POOLS:
                argv += ["--source", f"jsonl:path=cache/short-rus/{pool}.{size}.jsonl,min_chars=3"]
        else:
            argv.append(tok)
    assert len(argv) == len(recorded) + 2 * len(POOLS), "noisy lane not found in the recorded recipe"
    mix.main(argv + ["--lex-gate", "4", "--jobs", "1", "--out", out, *extra])


if __name__ == "__main__":
    main()
