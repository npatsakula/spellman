"""Per-row scoring through the Rust runtime: `spellman detect --lines`.

usage: score.py MODEL OUT.json eval.tsv [more.tsv ...]
Writes per-file metrics (overall / <=20 chars, per language, and each
language's rate of being predicted `rus`) plus a preds file next to OUT.
"""
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

BIN = Path(__file__).resolve().parent / "spellman"
if not BIN.exists():
    BIN = Path("/home/mrpink/projects/spellman/target/release/spellman")


def predict(model: str, texts: list[str]) -> list[str]:
    out = subprocess.run(
        [str(BIN), "detect", "--lines", "--model", model],
        input="\n".join(texts) + "\n", capture_output=True, text=True, check=True,
    ).stdout.split("\n")
    if out and out[-1] == "":
        out.pop()
    assert len(out) == len(texts), (len(out), len(texts))
    return out


def load(path: str) -> tuple[list[str], list[str]]:
    gold, texts = [], []
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        if "\t" not in line:
            continue
        g, t = line.split("\t", 1)
        gold.append(g.strip())
        texts.append(t.replace("\r", " "))
    return gold, texts


def metrics(gold, texts, pred) -> dict:
    tot = Counter(); ok = Counter(); to_rus = Counter()
    stot = Counter(); sok = Counter(); sto_rus = Counter()
    for g, t, p in zip(gold, texts, pred):
        short = len(t) <= 20
        tot[g] += 1; ok[g] += p == g; to_rus[g] += p == "rus"
        if short:
            stot[g] += 1; sok[g] += p == g; sto_rus[g] += p == "rus"
    n, ns = sum(tot.values()), sum(stot.values())
    per = {}
    for g in sorted(tot):
        per[g] = dict(n=tot[g], acc=ok[g] / tot[g], to_rus=to_rus[g] / tot[g],
                      n_short=stot[g], acc_short=(sok[g] / stot[g]) if stot[g] else None,
                      to_rus_short=(sto_rus[g] / stot[g]) if stot[g] else None)
    others = [g for g in tot if g != "rus"]
    no, nso = sum(tot[g] for g in others), sum(stot[g] for g in others)
    return dict(
        n=n, acc=sum(ok.values()) / n,
        n_short=ns, acc_short=(sum(sok.values()) / ns) if ns else None,
        other_n=no, other_acc=(sum(ok[g] for g in others) / no) if no else None,
        other_to_rus=(sum(to_rus[g] for g in others) / no) if no else None,
        other_n_short=nso,
        other_acc_short=(sum(sok[g] for g in others) / nso) if nso else None,
        other_to_rus_short=(sum(sto_rus[g] for g in others) / nso) if nso else None,
        per_lang=per,
    )


def main() -> None:
    model, out, files = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
    res = {}
    for f in files:
        gold, texts = load(f)
        pred = predict(model, texts)
        res[Path(f).parent.name + "/" + Path(f).name] = metrics(gold, texts, pred)
        m = res[Path(f).parent.name + "/" + Path(f).name]
        r = m["per_lang"].get("rus", {})
        print(f"{model} {f}: acc {m['acc']*100:.2f} short {((m['acc_short'] or 0)*100):.2f} "
              f"rus {r.get('acc', 0)*100:.2f} other->rus short {((m['other_to_rus_short'] or 0)*100):.3f}", flush=True)
        if "eval_test" in f:
            Path(str(out) + ".preds").write_text("\n".join(pred) + "\n")
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
