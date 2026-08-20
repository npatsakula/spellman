"""Error analysis for the specialist prototype + Q2 ablation.

Categories on val, tau=3 (the calibrated gate):
  MISSED   gated, pair matched, gold == top-2, but expert failed to fix
           (no flip, or flipped to something still wrong)
  WRONGFLIP gated, pair matched, main was right, expert flipped it wrong
  NOPAIR   gated low-margin, gold == top-2, but (top1, top2) has no
           specialist — a coverage gap of the pair list itself
For each sampled text we also compute WORD COVERAGE: the share of its
words present in data_mix4 train rows of the gold vs predicted language
(the v6 finding: failures were unseen vocabulary, not weak boundaries).

Q2 ablation: retrain the expert on LOW-MARGIN rows only (margin < 2.5)
and compare flip correctness on the same gate.
"""

from __future__ import annotations

import json
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from safetensors.numpy import load_file

from spellman_features import LANGUAGES, bucket_tokens_flat
from train import Config, featurize

TAU = 3.0
DIM = 32
EPOCHS = 4
LOW_ONLY_MARGIN = 2.5
N_SAMPLE = 200


def score_table(texts, table, bias, log2_d, chunk=4000):
    n_cls = table.shape[1]
    out = np.empty((len(texts), n_cls), dtype=np.float32)
    for t0 in range(0, len(texts), chunk):
        part = texts[t0 : t0 + chunk]
        b, negs, off = bucket_tokens_flat(part, log2_d)
        counts = np.diff(off)
        rows_i = np.repeat(np.arange(len(part)), counts)
        signs = np.where(negs, -1.0, 1.0).astype(np.float32)
        contrib = table[b] * signs[:, None]
        sums = np.stack([np.bincount(rows_i, weights=contrib[:, c], minlength=len(part)) for c in range(n_cls)], axis=1)
        out[t0 : t0 + chunk] = sums / np.maximum(counts, 1)[:, None] + bias
    return out


def main() -> None:
    w = load_file("../model/model.safetensors")
    P = w["P"].astype(np.float32); bias = w["bias"].astype(np.float32)
    log2_d = json.load(open("../model/model.json"))["log2_d"]
    D = 1 << log2_d

    def rows_of(path):
        return [tuple(json.loads(l).values()) if False else (json.loads(l)["lang"], json.loads(l)["text"])
                for l in open(path, encoding="utf-8") if l.strip()]

    train = rows_of("data_mix4/train.jsonl")
    val = rows_of("data_mix4/val.jsonl")
    val_texts = [t for _, t in val]
    val_y = np.array([LANGUAGES.index(l) for l, _ in val])

    cfg = Config()
    idx_np, sign_np, mask_np, y_np = featurize([{"lang": l, "text": t} for l, t in train], cfg)

    # train margins (main model, k=256 view)
    margins_train = np.empty(len(train), dtype=np.float32)
    with torch.no_grad():
        Pt = torch.from_numpy(P)
        for s in range(0, len(train), 8192):
            i = torch.from_numpy(idx_np[s : s + 8192]); sg = torch.from_numpy(sign_np[s : s + 8192])
            mk = torch.from_numpy(mask_np[s : s + 8192])
            lg = (Pt[i] * sg.unsqueeze(-1)).sum(1) / mk.sum(1, keepdim=True).clamp(min=1.0) + torch.from_numpy(bias)
            tv = lg.topk(2, dim=1).values
            margins_train[s : s + 8192] = (tv[:, 0] - tv[:, 1]).numpy()

    val_logits = score_table(val_texts, P, bias, log2_d)
    order = np.argsort(-val_logits, axis=1)
    t1, t2 = order[:, 0], order[:, 1]
    margin = val_logits[np.arange(len(t1)), t1] - val_logits[np.arange(len(t1)), t2]
    wrong = t1 != val_y

    # kept pairs from the prototype run (post-pruning)
    pairs = [tuple(sorted((LANGUAGES.index("tgk"), LANGUAGES.index("uzn")))),
             tuple(sorted((LANGUAGES.index("spa"), LANGUAGES.index("por")))),
             tuple(sorted((LANGUAGES.index("tat"), LANGUAGES.index("bak")))),
             tuple(sorted((LANGUAGES.index("ukr"), LANGUAGES.index("bel")))),
             tuple(sorted((LANGUAGES.index("kir"), LANGUAGES.index("tyv"))))]
    alphas = [2.0, 1.0, 0.5, 1.0, 1.0]
    pair_col = {p: i for i, p in enumerate(pairs)}

    # word-vocab per language from train rows
    vocab = {}
    for li, lang in enumerate(LANGUAGES):
        vocab[li] = set()
    for l, t in train:
        vocab[LANGUAGES.index(l)].update(t.lower().split())

    # ---------- train the expert (same recipe as prototype) ----------
    def train_expert(low_only: bool):
        net = nn.ModuleDict({"emb": nn.Embedding(D + 1, DIM), "heads": nn.Linear(DIM, len(pairs))}).to("mps")
        nn.init.zeros_(net["emb"].weight)
        opt = torch.optim.AdamW(net.parameters(), lr=0.01)
        sel_all = np.isin(y_np, [c for p in pairs for c in p])
        if low_only:
            sel_all &= margins_train < LOW_ONLY_MARGIN
        rng = np.random.default_rng(0)
        order_i = np.flatnonzero(sel_all)
        rng.shuffle(order_i)
        member = [np.isin(y_np, p) for p in pairs]
        for ep in range(EPOCHS):
            for s in range(0, len(order_i), 8192):
                sel = order_i[s : s + 8192]
                i = torch.from_numpy(idx_np[sel]).to("mps"); sg = torch.from_numpy(sign_np[sel]).to("mps")
                mk = torch.from_numpy(mask_np[sel]).to("mps")
                lg = net["heads"](((net["emb"](i) * sg.unsqueeze(-1)) * mk.unsqueeze(-1)).sum(1) / mk.sum(1, keepdim=True).clamp(min=1.0))
                B = len(sel)
                tgt = torch.zeros(B, len(pairs)); wts = torch.zeros(B, len(pairs)); col = torch.zeros(B, len(pairs))
                for pi, (a, b) in enumerate(pairs):
                    m = member[pi][sel]
                    col[:, pi] = torch.from_numpy(m)
                    tgt[:, pi] = torch.from_numpy((y_np[sel] == a).astype(np.float32))
                    na = int(((y_np[sel] == a) & m).sum()); nb = int(((y_np[sel] == b) & m).sum())
                    nmin = max(1, min(na, nb))
                    wts[:, pi] = torch.from_numpy(
                        np.where(y_np[sel] == a, nmin / max(1, na), nmin / max(1, nb)).astype(np.float32)
                        * np.where(margins_train[sel] < 1.5, 3.0, 1.0))
                loss = (nn.functional.binary_cross_entropy_with_logits(lg, tgt.to("mps"), reduction="none")
                        * wts.to("mps") * col.to("mps")).sum() / col.to("mps").sum().clamp(min=1)
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            P_sp = (net["emb"].weight.cpu().numpy() @ net["heads"].weight.cpu().numpy().T).astype(np.float16).astype(np.float32)
            bias_sp = net["heads"].bias.cpu().numpy().astype(np.float16).astype(np.float32)
        P_sp[D] = 0.0
        return P_sp, bias_sp

    P_sp, bias_sp = train_expert(low_only=False)
    val_spec = score_table(val_texts, P_sp, bias_sp, log2_d)

    # ---------- categorize ----------
    cat = {"MISSED": [], "WRONGFLIP": [], "NOPAIR": []}
    for i in range(len(t1)):
        if margin[i] >= TAU:
            continue
        key = tuple(sorted((int(t1[i]), int(t2[i]))))
        pc = pair_col.get(key)
        if wrong[i] and val_y[i] == t2[i]:
            if pc is None:
                cat["NOPAIR"].append(i)
            else:
                d = alphas[pc] * val_spec[i, pc]
                a = pairs[pc][0]
                n1 = val_logits[i, t1[i]] + (d if t1[i] == a else -d)
                n2 = val_logits[i, t2[i]] - (d if t1[i] == a else -d)
                new = t1[i] if n1 >= n2 else t2[i]
                if new != val_y[i]:
                    cat["MISSED"].append(i)
        elif (not wrong[i]) and pc is not None:
            d = alphas[pc] * val_spec[i, pc]
            a = pairs[pc][0]
            n1 = val_logits[i, t1[i]] + (d if t1[i] == a else -d)
            n2 = val_logits[i, t2[i]] - (d if t1[i] == a else -d)
            if n2 > n1:
                cat["WRONGFLIP"].append(i)

    print(f"counts: MISSED={len(cat['MISSED']):,} WRONGFLIP={len(cat['WRONGFLIP']):,} NOPAIR={len(cat['NOPAIR']):,}",
          flush=True)

    rng = random.Random(7)
    sample = (rng.sample(cat["MISSED"], min(120, len(cat["MISSED"])))
              + rng.sample(cat["WRONGFLIP"], min(40, len(cat["WRONGFLIP"])))
              + rng.sample(cat["NOPAIR"], min(40, len(cat["NOPAIR"]))))
    print(f"\n{'cat':<10} {'gold>pred':<14} {'m':>5} {'s':>6} {'vocG':>4} {'vocP':>4}  text")
    for i in sample:
        c = ("MISSED" if i in cat["MISSED"] else "WRONGFLIP" if i in cat["WRONGFLIP"] else "NOPAIR")
        words = [wd for wd in val_texts[i].lower().split() if any(ch.isalpha() for ch in wd)]
        vg = np.mean([wd in vocab[val_y[i]] for wd in words]) if words else 1.0
        vp = np.mean([wd in vocab[t1[i]] for wd in words]) if words else 1.0
        key = tuple(sorted((int(t1[i]), int(t2[i]))))
        pc = pair_col.get(key)
        s = val_spec[i, pc] if pc is not None else float("nan")
        print(f"{c:<10} {LANGUAGES[val_y[i]]}>{LANGUAGES[t1[i]]:<6} {margin[i]:5.2f} {s:6.2f} {vg:4.1f} {vp:4.1f}  {val_texts[i][:48]}")

    # ---------- Q2 ablation: low-margin-only expert ----------
    P_sp2, bias_sp2 = train_expert(low_only=True)
    val_spec2 = score_table(val_texts, P_sp2, bias_sp2, log2_d)
    for tag, vs, al in [("full-data expert", val_spec, alphas),
                        ("low-margin-only expert", val_spec2, alphas)]:
        n_flip = n_ok = 0
        for i in cat["MISSED"] + cat["WRONGFLIP"]:
            key = tuple(sorted((int(t1[i]), int(t2[i]))))
            pc = pair_col.get(key)
            if pc is None:
                continue
            d = al[pc] * vs[i, pc]
            a = pairs[pc][0]
            n1 = val_logits[i, t1[i]] + (d if t1[i] == a else -d)
            n2 = val_logits[i, t2[i]] - (d if t1[i] == a else -d)
            new = t1[i] if n1 >= n2 else t2[i]
            if new != t1[i]:
                n_flip += 1
                n_ok += int(new == val_y[i])
        print(f"\n{tag}: flips on these rows = {n_flip}, correct = {n_ok} "
              f"({100*n_ok/max(1,n_flip):.0f}%)", flush=True)


if __name__ == "__main__":
    main()
