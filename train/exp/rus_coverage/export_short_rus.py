"""Short colloquial Russian pools for the mix, from open chat/forum dumps.

The `hf` lane truncates at max_chars instead of dropping, so naturally
short rows are exported here once into `cache/short-rus/<name>.jsonl`
({"lang","text"}, shuffled with a fixed seed; the mix reads sized heads
written next to them as `<name>.<N>.jsonl`).

Kept: 3-19 characters after whitespace collapse, >= 60% Cyrillic letters,
only Russian-alphabet Cyrillic, lexical gate score towards Ukrainian < 3,
not a referee text, first occurrence case-folded. No author fields.

usage: uv run --with zstandard python export_short_rus.py [SIZE ...]
(default sizes: 12000 25000; ruforum streams zstd-compressed jsonl)
"""
import csv
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

from spellman_train.lexgate import load_lexicons, score
from spellman_train.paths import CACHE_DIR, TRAIN_DIR
from spellman_train.sources import cyrillic_ratio

OUT = CACHE_DIR / "short-rus"
RUSSIAN = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
SOURCES = {
    # name: (repo, text columns, streaming doc cap or None)
    "dialogues": ("Den4ikAI/russian_dialogues", ["question", "answer"], None),
    "dvach": ("hausmer/dvach_chat", ["text"], None),
    "ruforum": ("nyuuzyou/ruforum", ["text"], 1_500_000),
    # held out of every mix: an out-of-source probe for short Russian
    "okru-probe": ("IvanFed/russian-toxic-comments-multilabel", ["text"], None),
}


def referee_texts() -> set[str]:
    ref: set[str] = set()
    for f in ("short_eval.tsv", "rusentitweet_eval.tsv", "rusentitweet_eval_v2.tsv", "cosmus_rus_eval.tsv",
              "tatoeba_eval.tsv", "lit_rus_eval.tsv"):
        for line in (TRAIN_DIR / f).open(encoding="utf-8"):
            p = line.rstrip("\n").split("\t", 1)
            if len(p) == 2:
                ref.add(" ".join(p[1].split()).lower())
    csv.field_size_limit(10**9)
    for f in ("rusentitweet_train.csv", "rusentitweet_dev.csv", "rusentitweet_test.csv"):
        for r in csv.DictReader((TRAIN_DIR / f).open(encoding="utf-8")):
            ref.add(" ".join(r["text"].split()).lower())
    return ref


def main() -> None:
    sizes = [int(a) for a in sys.argv[1:]] or [12_000, 25_000]
    lex = dict(load_lexicons()["rus"])["ukr"]
    ref = referee_texts()
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, (repo, cols, cap) in SOURCES.items():
        if (OUT / f"{name}.jsonl").exists():
            continue  # already exported
        ds = load_dataset(repo, split="train", streaming=cap is not None)
        if cap is not None:
            ds = ds.take(cap)
        else:
            ds = ds.select_columns(cols)
        seen: set[str] = set()
        rows: list[str] = []
        c = dict(texts=0, short=0, cyr=0, alphabet=0, lex=0, referee=0, dup=0)
        for rec in ds:
            for col in cols:
                v = rec.get(col)
                if not isinstance(v, str):
                    continue
                t = " ".join(v.split())
                c["texts"] += 1
                if not 3 <= len(t) <= 19:
                    continue
                c["short"] += 1
                low = t.lower()
                if cyrillic_ratio(t) < 0.6:
                    c["cyr"] += 1
                elif any("Ѐ" <= ch <= "ӿ" and ch not in RUSSIAN for ch in low):
                    c["alphabet"] += 1
                elif score(t, lex) >= 3:
                    c["lex"] += 1
                elif low in ref:
                    c["referee"] += 1
                elif low in seen:
                    c["dup"] += 1
                else:
                    seen.add(low)
                    rows.append(t)
        random.Random(16).shuffle(rows)
        c["kept"] = len(rows)
        stats[name] = c
        print(name, c, flush=True)
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for t in rows:
                f.write(json.dumps({"lang": "rus", "text": t}, ensure_ascii=False) + "\n")
        if name.endswith("-probe"):
            with (OUT / f"{name}.tsv").open("w", encoding="utf-8") as f:
                for t in rows[:5000]:
                    f.write(f"rus\t{t}\n")
            continue
        for n in sizes:
            with (OUT / f"{name}.{n}.jsonl").open("w", encoding="utf-8") as f:
                for t in rows[:n]:
                    f.write(json.dumps({"lang": "rus", "text": t}, ensure_ascii=False) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
