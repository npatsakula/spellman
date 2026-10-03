"""Score `model.pt` checkpoints (linear or --head mlp) in PyTorch — the MLP
head does not fold into P, so the Rust runtime cannot run it.

Every document is scored on its full token sequence (mean over all tokens,
like the runtime's chunk-accumulated scoring), not the training k.

    uv run python -m spellman_train.exp_eval --data <mix dir> --out res.json \
        run_a/model.pt run_b/model.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from spellman_train.features import LANGUAGES, bucket_tokens_flat
from spellman_train.paths import TRAIN_DIR
from spellman_train.train import LANG_TO_IDX, SpellmanNet, load_split

REFEREES = ["tatoeba_eval.tsv", "rusentitweet_eval_v2.tsv", "cosmus_rus_eval.tsv", "short_eval.tsv"]
GROUPS = {
    "east_slavic": ["rus", "ukr", "bel"],
    "south_slavic": ["bul", "mkd", "srp"],
    "turkic": ["kaz", "kir", "tat", "bak", "uzn"],
}
BUCKETS = [("<=20", 0, 20), ("21-100", 21, 100), (">100", 101, 1 << 30)]


def load_model(path: Path, dev: torch.device) -> SpellmanNet:
    ck = torch.load(path, map_location="cpu")
    cfg = ck["config"]
    hidden = cfg["hidden"] if cfg.get("head") == "mlp" else 0
    net = SpellmanNet(1 << cfg["log2_d"], cfg["dim"], len(LANGUAGES), hidden)
    sd = {k.replace("._orig_mod", ""): v for k, v in ck["state_dict"].items()}
    net.load_state_dict(sd)
    net.cfg = cfg
    return net.to(dev).eval()


@torch.no_grad()
def predict(net: SpellmanNet, texts: list[str], dev: torch.device, chunk: int = 2048) -> np.ndarray:
    cfg = net.cfg
    preds = np.empty(len(texts), dtype=np.int64)
    order = np.argsort([len(t) for t in texts], kind="stable")  # length-sorted -> tight padding
    for t0 in range(0, len(texts), chunk):
        sel = order[t0 : t0 + chunk]
        buckets, negs, off = bucket_tokens_flat([texts[i] for i in sel], cfg["log2_d"], cfg["hash_id"], cfg["seed"])
        lens = np.diff(off)
        k = max(int(lens.max()), 1)
        idx = np.full((len(sel), k), 1 << cfg["log2_d"], dtype=np.int64)
        sign = np.zeros((len(sel), k), dtype=np.float32)
        mask = np.zeros((len(sel), k), dtype=np.float32)
        for j in range(len(sel)):
            m = int(lens[j])
            idx[j, :m] = buckets[off[j] : off[j] + m]
            sign[j, :m] = np.where(negs[off[j] : off[j] + m], -1.0, 1.0)
            mask[j, :m] = 1.0
        step = max(1, (1 << 21) // k)  # bound the [B, k, dim] gather
        for s in range(0, len(sel), step):
            logits = net(*(torch.from_numpy(a[s : s + step]).to(dev) for a in (idx, sign, mask)))
            preds[sel[s : s + step]] = logits.argmax(1).cpu().numpy()
    return preds


def read_tsv(path: Path) -> tuple[list[str], np.ndarray]:
    texts, ys = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        lang, text = line.split("\t", 1)
        texts.append(text)
        ys.append(LANG_TO_IDX[lang])
    return texts, np.array(ys)


def metrics(texts: list[str], y: np.ndarray, p: np.ndarray, detail: bool) -> dict:
    ok = p == y
    out = {"n": int(len(y)), "acc": float(ok.mean())}
    if detail:
        lens = np.array([len(t) for t in texts])
        for name, lo, hi in BUCKETS:
            m = (lens >= lo) & (lens <= hi)
            out[f"len{name}"] = {"n": int(m.sum()), "acc": float(ok[m].mean())}
        for g, langs in GROUPS.items():
            out[g] = {}
            for lang in langs:
                m = y == LANG_TO_IDX[lang]
                short = m & (lens <= 20)
                out[g][lang] = {"n": int(m.sum()), "acc": float(ok[m].mean()),
                                "n<=20": int(short.sum()), "acc<=20": float(ok[short].mean())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("models", type=Path, nargs="+")
    args = ap.parse_args()
    dev = torch.device(args.device)

    test = load_split(args.data, "test")
    sets = {"test": ([" ".join(r["text"].split()) for r in test], np.array([LANG_TO_IDX[r["lang"]] for r in test]))}
    for name in REFEREES:
        sets[name] = read_tsv(TRAIN_DIR / name)

    results = {}
    for mp in args.models:
        net = load_model(mp, dev)
        res = {}
        for name, (texts, y) in sets.items():
            p = predict(net, texts, dev)
            res[name] = metrics(texts, y, p, detail=name == "test")
            if name == "test":
                np.save(mp.parent / "test_preds.npy", p)
            print(f"{mp.parent.name} {name}: {res[name]['acc']:.4%}", flush=True)
        results[mp.parent.name] = res
    args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
