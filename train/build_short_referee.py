"""Build the frozen short-text referee (train/short_eval.tsv).

Rules — twin-aware external consensus, no spellman vote (the referee must
not be filtered through the model it scores):

  - affirm    : judge top-1 == gold (alias-mapped)
  - twin vote : judge top-1 is in the gold language's twin group
                (genuinely confusable, not a contradiction)
  - contradict: anything else
  KEEP if  (affirms == 2)                       -- both externals agree
        OR  (affirms == 1 AND contradict == 0
             AND text has >= 2 words)           -- one anchor, no dissent
  Single-word texts need both affirms (a shared word is a coin flip).

Sources are disjoint from the short training caches: the crawl tail
(training reads the head) with a belt-and-braces exact-text filter, plus
umsab validation/test splits (training used train.jsonl only).
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from eval_fasttext import GLOTLID_CANDIDATES, LID176_CANDIDATES
from short_verify import judge_text

TWINS = {
    "rus": {"rus", "ukr", "bel"}, "ukr": {"rus", "ukr", "bel"}, "bel": {"rus", "ukr", "bel"},
    "bul": {"bul", "mkd", "srp"}, "mkd": {"bul", "mkd", "srp"}, "srp": {"bul", "mkd", "srp"},
    "tat": {"tat", "bak", "kaz", "kir", "tyv", "chv", "sah"},
    "bak": {"tat", "bak", "kaz", "kir", "tyv", "chv", "sah"},
    "kaz": {"tat", "bak", "kaz", "kir", "tyv", "chv", "sah"},
    "kir": {"tat", "bak", "kaz", "kir", "tyv", "chv", "sah"},
    "tyv": {"tat", "bak", "kaz", "kir", "tyv", "chv", "sah"},
    "chv": {"tat", "bak", "kaz", "kir", "tyv", "chv", "sah"},
    "sah": {"tat", "bak", "kaz", "kir", "tyv", "chv", "sah"},
    "udm": {"udm", "mhr", "kpv"}, "mhr": {"udm", "mhr", "kpv"}, "kpv": {"udm", "mhr", "kpv"},
}
UMSAB = [("eng", "english"), ("fra", "french"), ("por", "portuguese"), ("deu", "german"), ("spa", "spanish")]


def vote(pred: str, lang: str, mapping: dict) -> str:
    """affirm | twin | contradict | abstain"""
    if pred in mapping.get(lang, []):
        return "affirm"
    for l, cands in mapping.items():
        if pred in cands:
            return "twin" if l in TWINS.get(lang, {lang}) else "contradict"
    return "abstain"


def main() -> None:
    import fasttext

    glot = fasttext.load_model("cache/raw/model.bin")
    lid = fasttext.load_model("cache/lid.176.bin")

    train_texts: set[str] = set()
    for f in Path("cache").glob("*.jsonl"):
        if f.name.startswith(("ukr_tweets", "hf-", "kazsandra", "mn_social")) and (f.stat().st_mtime > Path("mix.py").stat().st_mtime - 86400 * 3):
            pass  # precise set built below from the known short caches
    for f in [
        "cache/ukr_tweets-90bd1db874.jsonl", "cache/ukr_tweets-cda504406e.jsonl",
        "cache/hf-9822b74f8b.jsonl", "cache/hf-ae1e14ea70.jsonl", "cache/hf-bc620771c4.jsonl",
        "cache/hf-f16d89f3df.jsonl", "cache/hf-42211cf99a.jsonl",
    ]:
        for line in open(f, encoding="utf-8"):
            train_texts.add(json.loads(line)["text"])

    rows: list[tuple[str, str]] = []
    # Full-corpus scan: twin rows are certified by orthography (position-
    # independent), and exact-text exclusion below enforces disjointness
    # from the training caches.
    for chunk in pd.read_csv("cache/raw/ukr-twi-corpus/corpus.csv",
                             usecols=["text", "lang"], chunksize=200_000):
        for t, l in zip(chunk["text"].tolist(), chunk["lang"].tolist()):
            if isinstance(t, str) and l in ("uk", "ru"):
                t = " ".join(t.split())
                if 3 <= len(t) <= 19 and t not in train_texts:
                    rows.append(("ukr" if l == "uk" else "rus", t))
    for lang, name in UMSAB:
        for split in ("validation", "test"):
            p = Path(f"cache/raw/umsab/data/{name}/{split}.jsonl")
            if not p.exists():
                continue
            for line in p.open(encoding="utf-8"):
                t = " ".join(json.loads(line)["text"].split())
                if 3 <= len(t) <= 19 and t not in train_texts:
                    rows.append((lang, t))

    kept: list[tuple[str, str]] = []
    stats: Counter = Counter()
    for lang, text in rows:
        t = judge_text(text)
        (gl), = glot.predict(t)[0]
        (ll), = lid.predict(t)[0]
        v1 = vote(gl.removeprefix("__label__"), lang, GLOTLID_CANDIDATES)
        v2 = vote(ll.removeprefix("__label__"), lang, LID176_CANDIDATES)
        affirms = sum(v == "affirm" for v in (v1, v2))
        contradicts = sum(v == "contradict" for v in (v1, v2))
        words = len(t.split())
        # Twin referee rows are decided by ORTHOGRAPHY, not judges: the
        # external models cannot certify short ru/uk/be text (soft rules
        # leak Ukrainian into rus pools, unanimity keeps ~nothing), but the
        # alphabets differ deterministically — і ї є ґ never appear in
        # Russian, ы ъ ё э never in Ukrainian, ў is Belarusian's. Unmarked
        # twin rows are the intrinsically-ambiguous bucket: excluded.
        MARKERS = {"ukr": set("іїєґ"), "rus": set("ыъёэ"), "bel": set("ў")}
        if lang in MARKERS:
            chars = set(text)
            if chars & MARKERS[lang] and not any(chars & m for l2, m in MARKERS.items() if l2 != lang):
                kept.append((lang, text))
                stats[lang] += 1
            continue
        if affirms == 2 or (affirms == 1 and contradicts == 0 and words >= 2):
            kept.append((lang, text))
            stats[lang] += 1
    rng = random.Random(42)
    by: dict[str, list[str]] = {}
    for lang, text in kept:
        by.setdefault(lang, []).append(text)
    with open("short_eval.tsv", "w", encoding="utf-8") as f:
        for lang in sorted(by):
            for text in rng.sample(by[lang], min(500, len(by[lang]))):
                f.write(f"{lang}\t{text}\n")
    print(f"candidates {len(rows)} -> kept {len(kept)}: {dict(stats)}")
    print("referee:", {k: min(500, len(v)) for k, v in sorted(by.items())})


if __name__ == "__main__":
    main()
