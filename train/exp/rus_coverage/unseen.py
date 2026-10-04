"""Unseen-word enrichment: are a language's errors concentrated in rows
whose words never occur in the train split?

usage: unseen.py DATA_DIR EVAL_TSV PREDS [LANG ...]   (default: rus ukr)

Words are split and classed exactly as features.py does it (lowercase,
whitespace split, one leading '#' stripped, mention/URL/email/number words
become their class and are never "unseen"). A word is unseen when no train
row of any language contains it; `own` restricts the vocabulary to train
rows of the row's gold language. Pairs are adjacent retained words.
"""
import sys
from collections import Counter
from pathlib import Path

import polars as pl

from spellman_train.features import classify_word


def words(text: str) -> list[str]:
    out = []
    for w in text.lower().split():
        if w.startswith("#"):
            w = w[1:]
        if not w:
            continue
        s = classify_word(w)
        out.append(f"\x00{s}" if s is not None else w)
    return out


def main() -> None:
    data, tsv, preds = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    langs = sys.argv[4:] or ["rus", "ukr"]
    train = pl.read_parquet(data / "data" / "train-00000-of-00001.parquet")
    vocab: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    own: dict[str, set[str]] = {l: set() for l in langs}
    for lang, text in zip(train["lang"].to_list(), train["text"].to_list()):
        ws = words(text)
        vocab.update(ws)
        pairs.update(zip(ws, ws[1:]))
        if lang in own:
            own[lang].update(ws)
    print(f"train vocab {len(vocab)} words, {len(pairs)} pairs; "
          + ", ".join(f"{l} own {len(v)}" for l, v in own.items()))

    gold, texts = [], []
    for line in tsv.read_text(encoding="utf-8").split("\n"):
        if "\t" in line:
            g, t = line.split("\t", 1)
            gold.append(g.strip()); texts.append(t)
    pred = preds.read_text().split("\n")[: len(gold)]
    assert len(pred) == len(gold)

    for lang in langs:
        rows = []
        for g, t, p in zip(gold, texts, pred):
            if g != lang:
                continue
            ws = [w for w in words(t)]
            real = [w for w in ws if not w.startswith("\x00")]
            if not real:
                continue
            un = [w for w in real if w not in vocab]
            un_own = [w for w in real if w not in own[lang]]
            pr = list(zip(ws, ws[1:]))
            un_pr = sum(1 for q in pr if q not in pairs)
            rows.append(dict(short=len(t) <= 20, err=p != g, pred=p, n=len(real),
                             un=len(un) / len(real), un_own=len(un_own) / len(real),
                             un_pair=(un_pr / len(pr)) if pr else None, unseen=un, text=t))
        df = pl.DataFrame(rows, infer_schema_length=None)
        print(f"\n===== {lang}: {df.height} rows with words, {df['err'].sum()} errors")
        for name, sub in (("all", df), ("<=20", df.filter(pl.col("short"))), (">20", df.filter(~pl.col("short")))):
            print(f"-- {name}: n={sub.height} err={sub['err'].sum()} ({sub['err'].mean()*100:.2f}%)")
            agg = sub.group_by("err").agg(
                n=pl.len(), words=pl.col("n").mean(),
                unseen_share=pl.col("un").mean(), any_unseen=(pl.col("un") > 0).mean(),
                all_unseen=(pl.col("un") == 1).mean(), unseen_own=pl.col("un_own").mean(),
                any_unseen_own=(pl.col("un_own") > 0).mean(),
                unseen_pair=pl.col("un_pair").mean(),
            ).sort("err")
            with pl.Config(tbl_cols=20, tbl_width_chars=200, float_precision=3):
                print(agg)
            # error rate by unseen-share bin (the direction that matters)
            b = sub.with_columns(bin=pl.when(pl.col("un") == 0).then(pl.lit("0"))
                                 .when(pl.col("un") < 0.5).then(pl.lit("(0,.5)"))
                                 .when(pl.col("un") < 1).then(pl.lit("[.5,1)")).otherwise(pl.lit("1")))
            print(b.group_by("bin").agg(n=pl.len(), err_rate=pl.col("err").mean() * 100).sort("bin"))
        errs = df.filter(pl.col("err"))
        c = Counter(w for ws in errs["unseen"].to_list() for w in ws)
        print("top unseen words in errors:", ", ".join(f"{w}×{n}" for w, n in c.most_common(40)))
        cs = Counter(w for ws in errs.filter(pl.col("short"))["unseen"].to_list() for w in ws)
        print("top unseen words in <=20 errors:", ", ".join(f"{w}×{n}" for w, n in cs.most_common(40)))
        print("pred of errors:", errs["pred"].value_counts().sort("count", descending=True).head(8).to_dicts())
        print("sample <=20 errors with unseen words:")
        for r in errs.filter(pl.col("short") & (pl.col("un") > 0)).head(25).iter_rows(named=True):
            print(f"   [{r['pred']}] {r['text']}   unseen={r['unseen']}")
        print("sample <=20 errors with NO unseen words:")
        for r in errs.filter(pl.col("short") & (pl.col("un") == 0)).head(25).iter_rows(named=True):
            print(f"   [{r['pred']}] {r['text']}")


if __name__ == "__main__":
    main()
