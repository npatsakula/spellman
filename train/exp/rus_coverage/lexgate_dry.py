"""Dry run of the lexical gate over every rus/ukr source row of a manifest:
per-source drop counts by threshold (rows the ortho gate keeps), and
random samples for a hand audit.
usage: lexgate_dry.py MANIFEST OUT.jsonl
"""
import json
import random
import sys
from pathlib import Path

import polars as pl

from spellman_train import sources
from spellman_train.lexgate import load_lexicons, score
from spellman_train.mix import read_source
from spellman_train.ortho import contradicting_twin

THR = [3, 4, 5, 6, 8]


def main() -> None:
    m = json.loads(Path(sys.argv[1]).read_text())
    lex = {"rus": dict(load_lexicons()["rus"])["ukr"], "ukr": dict(load_lexicons()["ukr"])["rus"]}
    seen: set[tuple[str, str]] = set()
    out = open(sys.argv[2], "w", encoding="utf-8")
    print("src name lang rows ortho-dropped | lex drops at T=" + "/".join(map(str, THR)) + " | short rows, short drops at T=4")
    for i, (name, opts) in enumerate(m["sources"]):
        df = read_source(sources.create(name, **opts)).filter(pl.col("lang").is_in(["rus", "ukr"]))
        for lang in ("rus", "ukr"):
            rows = [t for t in df.filter(pl.col("lang") == lang)["text"].to_list() if (lang, t) not in seen]
            if not rows:
                continue
            seen.update((lang, t) for t in rows)
            kept = [t for t in rows if contradicting_twin(lang, t, 0.5) is None]
            sc = [score(t, lex[lang]) for t in kept]
            drops = [sum(s >= T for s in sc) for T in THR]
            short = [(t, s) for t, s in zip(kept, sc) if len(t) <= 19]
            print(i, name, lang, len(rows), len(rows) - len(kept), "|", "/".join(map(str, drops)), "|",
                  len(short), sum(s >= 4 for _, s in short), flush=True)
            for t, s in zip(kept, sc):
                if s >= 1:
                    out.write(json.dumps({"src": i, "lang": lang, "score": round(s, 2), "text": t}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
