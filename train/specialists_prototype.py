"""Prototype: close-pair specialists — cascade vs distilled-table merge.

The M1 experiment for the close-pair cascade proposal, with the
distillation-first amendment: the α·s pairwise nudge is linear in the
tokens, so it can be folded into the main table itself (columns ±α·P_sp)
and shipped as a plain version-3 artifact with zero runtime change.
This script trains the specialists once and measures BOTH merges:

  cascade : gate on margin < τ and top-2 ∈ pairs; nudge those two logits
  distilled: fold ±α·(P_sp col) into P/bias unconditionally, re-score

Efficiency design choices (vs the original plan):
  - dim 32 (binary heads, not 64)
  - margin-weighted training (specialists serve low-margin docs; train
    on that distribution) + per-pair class balance
  - per-pair α calibrated by coordinate ascent, pairs pruned at <60%
    flip correctness
  - f16-rounded tables everywhere the runtime would round

Usage: uv run python specialists_prototype.py [--device mps]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np
from safetensors.numpy import load_file

from spellman_features import LANGUAGES, bucket_tokens_flat

PAIR_ELIGIBLE_MIN_ROWS = 8000
MAX_PAIRS = 10
DIM = 32
TAUS = (0.5, 1.0, 1.5, 2.0, 3.0)
ALPHAS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
MARGIN_WEIGHT_SPLIT = 1.5   # rows below this margin get weight x3
LOW_MARGIN_STOP = 2.0       # early-stop metric slice


def load_rows(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows.append((r["lang"], r["text"]))
    return rows


def score_table(texts, table, bias, log2_d, chunk=4000):
    """Full-token logits, chunked bincount path (mirrors the runtime)."""
    n_cls = table.shape[1]
    out = np.empty((len(texts), n_cls), dtype=np.float32)
    for t0 in range(0, len(texts), chunk):
        part = texts[t0 : t0 + chunk]
        b, negs, off = bucket_tokens_flat(part, log2_d)
        counts = np.diff(off)
        rows_i = np.repeat(np.arange(len(part)), counts)
        signs = np.where(negs, -1.0, 1.0).astype(np.float32)
        contrib = table[b] * signs[:, None]
        sums = np.stack(
            [np.bincount(rows_i, weights=contrib[:, c], minlength=len(part)) for c in range(n_cls)], axis=1
        )
        out[t0 : t0 + chunk] = sums / np.maximum(counts, 1)[:, None] + bias
    return out


def top2(logits):
    order = np.argsort(-logits, axis=1)
    t1, t2 = order[:, 0], order[:, 1]
    m = logits[np.arange(len(t1)), t1] - logits[np.arange(len(t1)), t2]
    return t1, t2, m


def ladder_words(texts, golds_idx):
    """Tatoeba word rung: 1-word fragments (tokens with >=1 letter)."""
    frag_t, frag_g = [], []
    for t, g in zip(texts, golds_idx):
        toks = [w for w in t.split() if any(c.isalpha() for c in w)]
        frag_t.extend(toks)
        frag_g.extend([g] * len(toks))
    return frag_t, np.array(frag_g)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    t_start = time.time()
    w = load_file("../model/model.safetensors")
    P = w["P"].astype(np.float32)
    bias = w["bias"].astype(np.float32)
    meta = json.load(open("../model/model.json"))
    log2_d = meta["log2_d"]
    D = 1 << log2_d

    # ---- data ------------------------------------------------------------
    train = load_rows("data_mix4/train.jsonl")
    val = load_rows("data_mix4/val.jsonl")
    print(f"rows: train={len(train):,} val={len(val):,}", flush=True)

    # k=256 padded featurization for TRAINING (the main model's convention)
    import torch

    from train import Config, featurize

    cfg = Config()
    t0 = time.time()
    tr = featurize([{"lang": l, "text": t} for l, t in train], cfg)
    idx_np, sign_np, mask_np, y_np = tr
    print(f"featurized train (k={cfg.k}) in {time.time()-t0:.0f}s", flush=True)

    # ---- main-model margins on train (for weighting) ---------------------
    t0 = time.time()
    margins_train = np.empty(len(train), dtype=np.float32)
    with torch.no_grad():
        Pt = torch.from_numpy(P)
        for s in range(0, len(train), 4096):
            i = torch.from_numpy(idx_np[s : s + 4096])
            sg = torch.from_numpy(sign_np[s : s + 4096])
            mk = torch.from_numpy(mask_np[s : s + 4096])
            lg = (Pt[i] * sg.unsqueeze(-1)).sum(1) / mk.sum(1, keepdim=True).clamp(min=1.0)
            lg = lg + torch.from_numpy(bias)
            top2v = lg.topk(2, dim=1).values
            margins_train[s : s + 4096] = (top2v[:, 0] - top2v[:, 1]).numpy()
    print(f"train margins in {time.time()-t0:.0f}s", flush=True)

    # ---- pair selection from VAL confusion -------------------------------
    val_logits = score_table([t for _, t in val], P, bias, log2_d)
    val_y = np.array([LANGUAGES.index(l) for l, _ in val])
    t1, t2, m = top2(val_logits)
    train_l = np.array([l for l, _ in train])
    counts = np.array([(train_l == l).sum() for l in LANGUAGES])
    pair_mass = Counter()
    for a, b, gold in zip(t1, t2, val_y):
        pred = a if a != gold else b
        if pred != gold:
            pair_mass[tuple(sorted((gold, pred)))] += 1
    cand = [
        (mass, p) for p, mass in pair_mass.items()
        if counts[p[0]] >= PAIR_ELIGIBLE_MIN_ROWS and counts[p[1]] >= PAIR_ELIGIBLE_MIN_ROWS
    ]
    cand.sort(reverse=True)
    pairs = [p for _, p in cand[:MAX_PAIRS]]
    print("selected pairs:", [(LANGUAGES[a], LANGUAGES[b], m_) for m_, (a, b) in cand[:MAX_PAIRS]], flush=True)

    # ---- specialist net (dim 32, margin-weighted, balanced) --------------
    import torch.nn as nn

    device = torch.device(args.device)
    net = nn.ModuleDict({"emb": nn.Embedding(D + 1, DIM), "heads": nn.Linear(DIM, len(pairs))}).to(device)
    nn.init.zeros_(net["emb"].weight)
    opt = torch.optim.AdamW(net.parameters(), lr=0.01)

    member_rows = []
    for i, (a, b) in enumerate(pairs):
        member_rows.append(np.isin(y_np, (a, b)))
    # per-row: which pair columns, targets, and weights
    # weights: class balance within pair x margin weighting
    n = len(train)
    w_base = np.ones(n, dtype=np.float32)
    low = margins_train < MARGIN_WEIGHT_SPLIT
    w_base[low] = 3.0
    pair_cols = [torch.tensor([], dtype=torch.long)] * n  # not per-row; built per batch below

    y_t = torch.from_numpy(y_np)
    holdout = np.zeros(n, dtype=bool)
    rng = np.random.default_rng(42)
    for i, (a, b) in enumerate(pairs):
        sel = np.flatnonzero(member_rows[i])
        hold_idx = rng.choice(sel, size=max(1, int(0.05 * len(sel))), replace=False)
        holdout[hold_idx] = True
    train_mask = ~holdout

    def batch_iter(bs=8192):
        order = np.flatnonzero(train_mask)
        rng2 = np.random.default_rng(0)
        rng2.shuffle(order)
        for s in range(0, len(order), bs):
            yield order[s : s + bs]

    def pair_targets(sel_np):
        """columns mask [B,S] (row belongs to pair), targets [B,S] in {0,1}, weights [B,S]"""
        B = len(sel_np)
        cols = np.zeros((B, len(pairs)), dtype=bool)
        tgt = np.zeros((B, len(pairs)), dtype=np.float32)
        wts = np.zeros((B, len(pairs)), dtype=np.float32)
        for i, (a, b) in enumerate(pairs):
            mem = member_rows[i][sel_np]
            cols[:, i] = mem
            tgt[mem, i] = (y_np[sel_np][mem] == a).astype(np.float32)
            # class balance: weight = n_min / n_cls within TRAIN rows of this pair
            n_a = int(((y_np[train_mask & member_rows[i]]) == a).sum())
            n_b = int(((y_np[train_mask & member_rows[i]]) == b).sum())
            nmin = max(1, min(n_a, n_b))
            wsel = np.where(y_np[sel_np] == a, nmin / max(1, n_a), nmin / max(1, n_b)).astype(np.float32)
            wts[:, i] = wsel * w_base[sel_np]
        return cols, tgt, wts

    # holdout logits for early stop (low-margin slice per pair)
    @torch.no_grad()
    def holdout_eval():
        net.eval()
        accs = []
        sel = np.flatnonzero(holdout)
        for s in range(0, len(sel), 8192):
            sel_np = sel[s : s + 8192]
            i = torch.from_numpy(idx_np[sel_np]).to(device)
            sg = torch.from_numpy(sign_np[sel_np]).to(device)
            mk = torch.from_numpy(mask_np[sel_np]).to(device)
            lg = net["heads"](((net["emb"](i) * sg.unsqueeze(-1)) * mk.unsqueeze(-1)).sum(1) / mk.sum(1, keepdim=True).clamp(min=1.0))
            cols, tgt, _ = pair_targets(sel_np)
            lg = lg.cpu().numpy()
            for pi in range(len(pairs)):
                c = cols[:, pi]
                lm = c & (margins_train[sel_np] < LOW_MARGIN_STOP)
                if lm.sum() < 20:
                    lm = c
                pred = (lg[lm, pi] > 0).astype(np.float32)
                accs.append((pred == tgt[lm, pi]).mean())
        net.train()
        return float(np.mean(accs))

    best_state, best_acc, patience = None, -1.0, 0
    t0 = time.time()
    for epoch in range(10):
        for sel_np in batch_iter():
            i = torch.from_numpy(idx_np[sel_np]).to(device)
            sg = torch.from_numpy(sign_np[sel_np]).to(device)
            mk = torch.from_numpy(mask_np[sel_np]).to(device)
            pooled = ((net["emb"](i) * sg.unsqueeze(-1)) * mk.unsqueeze(-1)).sum(1) / mk.sum(1, keepdim=True).clamp(min=1.0)
            lg = net["heads"](pooled)
            cols, tgt, wts = pair_targets(sel_np)
            loss = (nn.functional.binary_cross_entropy_with_logits(
                lg, torch.from_numpy(tgt).to(device), reduction="none"
            ) * torch.from_numpy(wts).to(device) * torch.from_numpy(cols).to(device)).sum() / max(1.0, torch.from_numpy(cols).to(device).sum())
            opt.zero_grad(); loss.backward(); opt.step()
        acc = holdout_eval()
        print(f"epoch {epoch+1}: holdout low-margin pair acc {acc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if acc > best_acc + 1e-4:
            best_acc, patience = acc, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 2:
                break
    net.load_state_dict(best_state)

    # ---- fold + f16 round -------------------------------------------------
    with torch.no_grad():
        P_sp = (net["emb"].weight.cpu().numpy() @ net["heads"].weight.cpu().numpy().T).astype(np.float16).astype(np.float32)
        bias_sp = net["heads"].bias.cpu().numpy().astype(np.float16).astype(np.float32)
    P_sp[D] = 0.0

    # ---- specialist logits on val + referees (full tokens) ----------------
    def spec_logits(texts):
        return score_table(texts, P_sp, bias_sp, log2_d)

    referees = {
        "val": ([t for _, t in val], val_y),
        "tatoeba": None, "rusentitweet": None, "cosmus": None, "short": None,
    }
    for name, path in [("tatoeba", "tatoeba_eval.tsv"), ("rusentitweet", "rusentitweet_eval_v2.tsv"),
                       ("cosmus", "cosmus_rus_eval.tsv"), ("short", "short_eval.tsv")]:
        texts, golds = [], []
        for line in open(path, encoding="utf-8"):
            l, t = line.rstrip("\n").split("\t", 1)
            texts.append(t); golds.append(LANGUAGES.index(l))
        referees[name] = (texts, np.array(golds))

    val_spec = spec_logits(referees["val"][0])

    # ---- calibration: τ global, α per pair (coordinate ascent) ------------
    pair_col = {tuple(sorted(p)): i for i, p in enumerate(pairs)}

    def cascade_merge(logits, specs, tau, alphas):
        t1, t2, m = top2(logits)
        out = logits.copy()
        flipped = np.zeros(len(t1), dtype=bool)
        for i in range(len(t1)):
            if m[i] >= tau:
                continue
            key = tuple(sorted((int(t1[i]), int(t2[i]))))
            pc = pair_col.get(key)
            if pc is None:
                continue
            a, b = pairs[pc]
            d = alphas[pc] * specs[i, pc]
            if t1[i] == a:
                out[i, t1[i]] += d; out[i, t2[i]] -= d
            else:
                out[i, t1[i]] -= d; out[i, t2[i]] += d
            flipped[i] = True
        return out, flipped

    base_val_acc = (val_logits.argmax(1) == val_y).mean()
    best = None
    for tau in TAUS:
        alphas = [0.5] * len(pairs)
        for _ in range(3):  # coordinate ascent over pairs
            for pi in range(len(pairs)):
                best_a, best_acc_, best_flips = alphas[pi], -1, 0
                for a in ALPHAS:
                    trial = list(alphas); trial[pi] = a
                    merged, flipped = cascade_merge(val_logits, val_spec, tau, trial)
                    acc_ = (merged.argmax(1) == val_y).mean()
                    flips_ok = ((merged.argmax(1) == val_y) & flipped).sum()
                    if acc_ > best_acc_ or (acc_ == best_acc_ and flips_ok > best_flips):
                        best_a, best_acc_, best_flips = a, acc_, flips_ok
                alphas[pi] = best_a
        merged, flipped = cascade_merge(val_logits, val_spec, tau, alphas)
        acc = (merged.argmax(1) == val_y).mean()
        print(f"tau={tau}: val {100*base_val_acc:.2f} -> {100*acc:.2f} (alpha={alphas})", flush=True)
        if best is None or acc > best[0]:
            best = (acc, tau, list(alphas))
    _, tau_star, alpha_star = best

    # per-pair pruning: flip correctness on val
    keep = []
    for pi, p in enumerate(pairs):
        a_mask = np.zeros(len(val_y), dtype=bool)
        # flip correctness for pair pi alone at alpha_star
        t1, t2, m = top2(val_logits)
        n_flip = n_flip_correct = 0
        for i in range(len(t1)):
            if m[i] >= tau_star: continue
            key = tuple(sorted((int(t1[i]), int(t2[i]))))
            if pair_col.get(key) != pi: continue
            d = alpha_star[pi] * val_spec[i, pi]
            new1 = val_logits[i, t1[i]] + (d if t1[i] == pairs[pi][0] else -d)
            new2 = val_logits[i, t2[i]] - (d if t1[i] == pairs[pi][0] else -d)
            n_new = t1[i] if new1 >= new2 else t2[i]
            if n_new != t1[i]:
                n_flip += 1
                n_flip_correct += int(n_new == val_y[i])
        rate = n_flip_correct / max(1, n_flip)
        print(f"pair {LANGUAGES[p[0]]}-{LANGUAGES[p[1]]}: flips={n_flip} correct={100*rate:.0f}%", flush=True)
        if n_flip == 0 or rate >= 0.6:
            keep.append(pi)
    pairs = [pairs[i] for i in keep]
    alpha_star = [alpha_star[i] for i in keep]
    P_sp = P_sp[:, keep]
    bias_sp = bias_sp[keep]
    pair_col = {tuple(sorted(p)): i for i, p in enumerate(pairs)}
    print(f"kept {len(pairs)} pairs after pruning", flush=True)

    # ---- evaluate both merges on all referees -----------------------------
    def distilled_table(alphas):
        P2 = P.copy(); b2 = bias.copy()
        for i, (a, b) in enumerate(pairs):
            P2[:, a] += alphas[i] * P_sp[:, i]
            P2[:, b] -= alphas[i] * P_sp[:, i]
            b2[a] += alphas[i] * bias_sp[i]
            b2[b] -= alphas[i] * bias_sp[i]
        return P2, b2

    print(f"\n=== results (tau={tau_star}, alpha={alpha_star}) ===", flush=True)
    print(f"{'referee':<14} {'base':>7} {'cascade':>8} {'distill':>8} {'gated%':>7} {'flip ok':>8}")
    tot_cascade_gain = tot_distill_gain = 0.0
    for name, (texts, golds) in referees.items():
        base_lg = score_table(texts, P, bias, log2_d)
        spec_lg = spec_logits(texts)
        base_acc = (base_lg.argmax(1) == golds).mean()
        merged, flipped = cascade_merge(base_lg, spec_lg, tau_star, alpha_star)
        casc_acc = (merged.argmax(1) == golds).mean()
        flips_ok = ((merged.argmax(1) == golds) & flipped & (base_lg.argmax(1) != golds)).sum()
        P2, b2 = distilled_table(alpha_star)
        dist_lg = score_table(texts, P2, b2, log2_d)
        dist_acc = (dist_lg.argmax(1) == golds).mean()
        print(f"{name:<14} {100*base_acc:7.2f} {100*casc_acc:8.2f} {100*dist_acc:8.2f} "
              f"{100*flipped.mean():7.1f} {flips_ok:>8}", flush=True)
        if name == "val":
            tot_cascade_gain = casc_acc - base_acc
            tot_distill_gain = dist_acc - base_acc

    # tatoeba word rung
    tt, tg = ladder_words(referees["tatoeba"][0], referees["tatoeba"][1])
    base_lg = score_table(tt, P, bias, log2_d)
    spec_lg = spec_logits(tt)
    b_acc = (base_lg.argmax(1) == tg).mean()
    merged, _ = cascade_merge(base_lg, spec_lg, tau_star, alpha_star)
    c_acc = (merged.argmax(1) == tg).mean()
    P2, b2 = distilled_table(alpha_star)
    d_acc = (score_table(tt, P2, b2, log2_d).argmax(1) == tg).mean()
    print(f"tatoeba-words  {100*b_acc:7.2f} {100*c_acc:8.2f} {100*d_acc:8.2f}   (n={len(tg):,})", flush=True)

    print(f"\nwall time: {time.time()-t_start:.0f}s total", flush=True)
    print(f"val gain: cascade {100*tot_cascade_gain:+.2f}pp, distilled {100*tot_distill_gain:+.2f}pp "
          f"(oracle ceiling at tau={tau_star}: see earlier measurement)", flush=True)


if __name__ == "__main__":
    main()
