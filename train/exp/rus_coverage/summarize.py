"""One row per scored model (data/exp/*.json from score.py).
usage: summarize.py NAME[=label] ...
"""
import json
import sys
from pathlib import Path

T, TA, RST, COS, SH = ("model-v15/eval_test.tsv", "train/tatoeba_eval.tsv", "train/rusentitweet_eval_v2.tsv",
                       "train/cosmus_rus_eval.tsv", "train/short_eval.tsv")
COLS = ["test", "test≤20", "rus", "rus≤20", "oth≤20", "oth→rus", "oth→rus≤20",
        "rst", "rst≤20", "cosmus", "short", "short rus", "short oth→rus", "tatoeba", "tat oth→rus≤20"]


def pct(v):
    return "   -  " if v is None else f"{v * 100:6.2f}"


def row(m: dict) -> list:
    def get(k):
        return m.get(k) or m[k.replace("train/", "/", 1)]
    t, ta, rst, cos, sh = get(T), get(TA), get(RST), get(COS), get(SH)
    r = t["per_lang"]["rus"]
    return [t["acc"], t["acc_short"], r["acc"], r["acc_short"], t["other_acc_short"], t["other_to_rus"],
            t["other_to_rus_short"], rst["acc"], rst["acc_short"], cos["acc"], sh["acc"],
            sh["per_lang"]["rus"]["acc"], sh["other_to_rus"], ta["acc"], ta["other_to_rus_short"]]


def main() -> None:
    print("| model | " + " | ".join(COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for spec in sys.argv[1:]:
        name, _, label = spec.partition("=")
        m = json.loads(Path(f"data/exp/{name}.json").read_text())
        print(f"| {label or name} | " + " | ".join(pct(v).strip() for v in row(m)) + " |")


if __name__ == "__main__":
    main()
