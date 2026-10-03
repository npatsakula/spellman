"""Would lowering Russian's bias at export undo the drift to `rus`?

Scores model.pt in PyTorch (float weights, no script router — close to,
not identical with, the int8 runtime), subtracts DELTA from the rus logit
and reports the trade-off on the gated test split and the referees.

usage: bias_sweep.py MODEL_DIR [MODEL_DIR ...]
"""
import sys
from pathlib import Path

import numpy as np
import torch

from spellman_train.exp_eval import load_model, read_tsv
from spellman_train.features import bucket_tokens_flat
from spellman_train.train import LANG_TO_IDX

DELTAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
FILES = ["model-v15/eval_test.tsv", "rusentitweet_eval_v2.tsv", "cosmus_rus_eval.tsv", "short_eval.tsv", "tatoeba_eval.tsv"]
RUS = LANG_TO_IDX["rus"]


@torch.no_grad()
def logits_of(net, texts: list[str], dev, chunk: int = 2048) -> np.ndarray:
    cfg = net.cfg
    out = None
    order = np.argsort([len(t) for t in texts], kind="stable")
    for t0 in range(0, len(texts), chunk):
        sel = order[t0 : t0 + chunk]
        b, ng, off = bucket_tokens_flat([texts[i] for i in sel], cfg["log2_d"], cfg["hash_id"], cfg["seed"])
        lens = np.diff(off)
        k = max(int(lens.max()), 1)
        idx = np.full((len(sel), k), 1 << cfg["log2_d"], dtype=np.int64)
        sign = np.zeros((len(sel), k), dtype=np.float32)
        mask = np.zeros((len(sel), k), dtype=np.float32)
        for j in range(len(sel)):
            m = int(lens[j])
            idx[j, :m] = b[off[j] : off[j] + m]
            sign[j, :m] = np.where(ng[off[j] : off[j] + m], -1.0, 1.0)
            mask[j, :m] = 1.0
        step = max(1, (1 << 21) // k)
        for s in range(0, len(sel), step):
            lg = net(*(torch.from_numpy(a[s : s + step]).to(dev) for a in (idx, sign, mask))).cpu().numpy()
            if out is None:
                out = np.empty((len(texts), lg.shape[1]), dtype=np.float32)
            out[sel[s : s + step]] = lg
    return out


def main() -> None:
    dev = torch.device("cuda")
    data = {f: read_tsv(Path(f)) for f in FILES}
    for model in sys.argv[1:]:
        net = load_model(Path(model) / "model.pt", dev)
        lg = {f: logits_of(net, t, dev) for f, (t, _) in data.items()}
        print(f"\n{model}\n| Δ rus bias | test | rus | rus≤20 | oth→rus | oth→rus≤20 | rst | cosmus | short | short rus | tatoeba |\n|---|---|---|---|---|---|---|---|---|---|---|")
        for d in DELTAS:
            row = []
            for f in FILES:
                texts, y = data[f]
                z = lg[f].copy()
                z[:, RUS] -= d
                p = z.argmax(1)
                short = np.array([len(t) <= 20 for t in texts])
                r, o = y == RUS, y != RUS
                acc = (p == y).mean()
                if f == FILES[0]:
                    row += [acc, (p == y)[r].mean(), (p == y)[r & short].mean(), (p == RUS)[o].mean(), (p == RUS)[o & short].mean()]
                elif f == "short_eval.tsv":
                    row += [acc, (p == y)[r].mean()]
                else:
                    row.append(acc)
            print(f"| {d:.2f} | " + " | ".join(f"{v * 100:.2f}" for v in row) + " |")


if __name__ == "__main__":
    main()
