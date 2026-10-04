"""Recalibrate theta by error-detection F1 on the validation split.

The runtime flags a detection as uncertain when its confidence < theta.
Treating "the prediction is wrong" as the positive class, pick the theta
that maximizes F1 of the flag (v14's procedure; train.py's default is the
5th percentile of validation confidence). Scores through the Rust runtime.

usage: theta_f1.py MODEL_DIR [--write]     (reads MODEL_DIR/eval_val.tsv)
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

BIN = Path(__file__).resolve().parents[3] / "target" / "release" / "spellman"


def main() -> None:
    model = Path(sys.argv[1])
    gold, texts = [], []
    for line in (model / "eval_val.tsv").read_text(encoding="utf-8").split("\n"):
        if "\t" in line:
            g, t = line.split("\t", 1)
            gold.append(g)
            texts.append(t)
    out = subprocess.run([str(BIN), "detect", "--lines", "--json", "--model", str(model)],
                         input="\n".join(texts) + "\n", capture_output=True, text=True, check=True).stdout
    recs = [json.loads(l) for l in out.split("\n") if l]
    assert len(recs) == len(gold), (len(recs), len(gold))
    conf = np.array([r["confidence"] for r in recs], dtype=np.float64)
    wrong = np.array([r["lang"] != g for r, g in zip(recs, gold)])
    order = np.argsort(conf, kind="stable")
    c, w = conf[order], wrong[order]
    tp = np.cumsum(w)              # flag the i+1 least confident rows
    flagged = np.arange(1, len(c) + 1)
    f1 = 2 * tp / (flagged + w.sum())
    # a threshold must separate distinct confidences: evaluate at run ends
    ends = np.flatnonzero(np.append(c[1:] != c[:-1], True))
    best = ends[np.argmax(f1[ends])]
    theta = float((c[best] + c[min(best + 1, len(c) - 1)]) / 2)
    old = json.loads((model / "model.json").read_text())["theta"]

    def report(th: float) -> str:
        fl = conf < th
        t = int((fl & wrong).sum())
        p, r = t / max(fl.sum(), 1), t / max(wrong.sum(), 1)
        return f"theta {th:.3f}: flags {fl.mean() * 100:.2f}% P {p:.3f} R {r:.3f} F1 {2 * p * r / max(p + r, 1e-9):.3f}"

    print(f"val rows {len(gold)}, errors {int(wrong.sum())} ({wrong.mean() * 100:.2f}%)")
    print("shipped  ", report(old))
    print("F1-best  ", report(theta))
    if "--write" in sys.argv:
        meta = json.loads((model / "model.json").read_text())
        meta["theta"] = round(theta, 2)
        (model / "model.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(f"wrote theta={meta['theta']} to {model / 'model.json'}")


if __name__ == "__main__":
    main()
