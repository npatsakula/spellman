"""Unseen-word enrichment on clean Russian: the wild referees, and the
test split with the rus->ukr rows (mostly Ukrainian text under a rus label)
set aside. Also separates words that are unseen only because punctuation
is glued to them ("так?!") from lexemes absent from train.

usage: unseen_refs.py DATA_DIR MODEL
"""
import string
import subprocess
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from score import load, predict  # noqa: E402
from unseen import words  # noqa: E402

PUNCT = string.punctuation + "«»…—–“”„’"


def core(w: str) -> str:
    return w.strip(PUNCT)


def table(name: str, rows: list[dict]) -> None:
    df = pl.DataFrame(rows)
    print(f"\n== {name}: n={df.height} errors={df['err'].sum()} ({df['err'].mean() * 100:.2f}%)")
    for label, sub in (("all", df), ("<=20", df.filter(pl.col("short"))), (">20", df.filter(~pl.col("short")))):
        if not sub.height:
            continue
        line = [f"{label:5} n={sub.height:5} err={sub['err'].mean() * 100:5.2f}%"]
        for col in ("un", "un_core"):
            z = sub.filter(pl.col(col) == 0)
            a = sub.filter(pl.col(col) > 0)
            f = sub.filter(pl.col(col) == 1)
            line.append(
                f"{col}: none {z['err'].mean() * 100:5.2f}% (n={z.height}) | any {a['err'].mean() * 100:5.2f}% (n={a.height})"
                f" | all {(f['err'].mean() or 0) * 100:5.2f}% (n={f.height})"
                f" | errors with none: {z['err'].sum()}/{sub['err'].sum()}"
            )
        print("  " + "\n        ".join(line))


def main() -> None:
    data, model = Path(sys.argv[1]), sys.argv[2]
    train = pl.read_parquet(data / "data" / "train-00000-of-00001.parquet")
    vocab: set[str] = set()
    cores: set[str] = set()
    for text in train["text"].to_list():
        ws = words(text)
        vocab.update(ws)
        cores.update(core(w) for w in ws)
    for name, path, lang, skip_pred in (
        ("test rus (all)", "model-v15/eval_test.tsv", "rus", None),
        ("test rus, rus->ukr rows set aside", "model-v15/eval_test.tsv", "rus", "ukr"),
        ("test ukr (control)", "model-v15/eval_test.tsv", "ukr", None),
        ("rusentitweet v2", "rusentitweet_eval_v2.tsv", "rus", None),
        ("COSMUS rus", "cosmus_rus_eval.tsv", "rus", None),
        ("short_eval rus", "short_eval.tsv", "rus", None),
    ):
        gold, texts = load(path)
        pred = predict(model, texts)
        rows = []
        for g, t, p in zip(gold, texts, pred):
            if g != lang or (skip_pred and p == skip_pred):
                continue
            real = [w for w in words(t) if not w.startswith("\x00")]
            real_c = [c for c in map(core, real) if c]
            if not real:
                continue
            rows.append(dict(
                short=len(t) <= 20, err=p != g,
                un=sum(w not in vocab for w in real) / len(real),
                un_core=(sum(c not in cores for c in real_c) / len(real_c)) if real_c else 0.0,
            ))
        table(name, rows)


if __name__ == "__main__":
    main()
