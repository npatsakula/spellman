"""Train the spellman fastText-style model and export the folded runtime model.

The trained network is exactly what inference folds:
    scores = mean_over_tokens( sign * E[bucket] ) · W + b
Because the head is linear, E·W collapses to a single [D+1, C] table `P`
(zero row D for padding) — the exported `model.json` + `model.safetensors`
that both the Rust CPU path and the svod JIT path consume.

Usage (from train/):
    uv run spellman-train --data data --out ../model --hash-id fmix32

Hash A/B: run with --hash-id {fmix32,murmur2,multiply_shift} and compare
val accuracies; --hash-stats prints a chi-square uniformity check of the
bucket occupancy on the training tokens.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from torch import nn
from tqdm import tqdm

from spellman_features import (
    DEFAULT_SEED,
    LANGUAGES,
    bucket_of,
    bucket_tokens_flat,
    token_keys,
)
from quantize import dequantize, parse_store, quantize, stats


LANG_TO_IDX = {code: i for i, code in enumerate(LANGUAGES)}


@dataclass
class Config:
    log2_d: int = 17
    hash_id: str = "fmix32"
    seed: int = DEFAULT_SEED
    dim: int = 128
    epochs: int = 3
    batch_size: int = 256
    k: int = 256  # max tokens per sample during training
    lr: float = 0.02
    per_lang_cap: int = 50_000

def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def featurize(rows: list[dict], cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (idx [N,K] i64 buckets, sign [N,K] f32, mask [N,K] f32, y [N] i64).

    Batch-vectorized (bucket_tokens_flat, bit-exact with the scalar
    reference); the K-truncation takes the first k tokens in reference
    encounter order, same as the per-row loop it replaced."""
    n, k = len(rows), cfg.k
    idx = np.zeros((n, k), dtype=np.int64)
    sign = np.zeros((n, k), dtype=np.float32)
    mask = np.zeros((n, k), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    for i, row in enumerate(rows):
        y[i] = LANG_TO_IDX[row["lang"]]

    chunk = 20_000
    for t0 in tqdm(range(0, n, chunk), desc="featurize", leave=False):
        part = rows[t0 : t0 + chunk]
        buckets, negs, offsets = bucket_tokens_flat(
            [r["text"] for r in part], cfg.log2_d, cfg.hash_id, cfg.seed
        )
        for i, _row in enumerate(part):
            lo = offsets[i]
            m = min(k, int(offsets[i + 1] - lo))
            if m == 0:
                continue
            idx[t0 + i, :m] = buckets[lo : lo + m]
            sign[t0 + i, :m] = np.where(negs[lo : lo + m], np.float32(-1.0), np.float32(1.0))
            mask[t0 + i, :m] = 1.0
    return idx, sign, mask, y


def balance_train(rows: list[dict], cap: int, rng: np.random.Generator) -> list[dict]:
    by_lang: dict[str, list[dict]] = {}
    for row in rows:
        by_lang.setdefault(row["lang"], []).append(row)
    out: list[dict] = []
    for lang, items in sorted(by_lang.items()):
        if len(items) > cap:
            items = list(rng.choice(items, size=cap, replace=False))
        out.extend(items)
    return out


class SpellmanNet(nn.Module):
    def __init__(self, d_buckets: int, dim: int, n_classes: int):
        super().__init__()
        self.emb = nn.Embedding(d_buckets + 1, dim)  # +1: padding row D, zeroed at export
        self.head = nn.Linear(dim, n_classes)
        # Zero-init (fastText convention): untrained buckets must fold to
        # exactly-zero logits through P = E·W. With random init, rare-word
        # n-grams that land in never-updated buckets contribute arbitrary
        # nonzero scores — verified failure mode: "sweatshirt" alone scored
        # bul 1.0 because its buckets were untrained noise.
        nn.init.zeros_(self.emb.weight)

    def forward(self, idx: torch.Tensor, sign: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emb = self.emb(idx) * sign.unsqueeze(-1)
        pooled = (emb * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
        return self.head(pooled)


@torch.no_grad()
def evaluate(model: SpellmanNet, tensors: tuple, batch_size: int) -> tuple[float, np.ndarray]:
    model.eval()
    idx, sign, mask, y = tensors
    correct = 0
    confs: list[np.ndarray] = []
    for start in range(0, len(y), batch_size):
        sl = slice(start, start + batch_size)
        logits = model(
            torch.from_numpy(idx[sl]),
            torch.from_numpy(sign[sl]),
            torch.from_numpy(mask[sl]),
        )
        probs = torch.softmax(logits, dim=-1).numpy()
        pred = probs.argmax(1)
        correct += int((pred == y[sl]).sum())
        confs.append(probs[np.arange(len(pred)), pred])
    model.train()
    return correct / len(y), np.concatenate(confs)


def chi_square_stats(train_rows: list[dict], cfg: Config) -> float:
    """Hash-spread check: chi-square of *distinct* n-gram keys over buckets.

    Token *occurrences* are Zipfian by nature (a chi-square on them measures
    language, not the hash); the property feature hashing needs is that
    distinct keys spread uniformly, so the statistic is computed on the set of
    unique keys.
    """
    seen_keys: set[int] = set()
    for row in train_rows[:20_000]:
        for key in token_keys(row["text"]):
            seen_keys.add(key)
    d = 1 << cfg.log2_d
    counts = np.zeros(d, dtype=np.int64)
    for key in seen_keys:
        bucket, _ = bucket_of(key, cfg.log2_d, cfg.hash_id, cfg.seed)
        counts[bucket] += 1
    total = counts.sum()
    expected = total / d
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    dof = d - 1
    # Normalized chi-square/dof: ~1.0 means uniform within sampling noise.
    print(
        f"hash={cfg.hash_id}: {total} distinct keys, chi2/dof = {chi2 / dof:.3f} "
        f"(uniform ≈ 1.0), max bucket occupancy = {counts.max()} (mean {expected:.1f})"
    )
    return chi2 / dof


def table_accuracy(p: np.ndarray, bias: np.ndarray, tensors: tuple, batch: int = 2048) -> float:
    """Accuracy of the folded-table scorer on featurized tensors — the exact
    computation the runtime performs (gather ±rows, mean over tokens, +bias).
    Used to gate quantized storage formats against the f16 fold."""
    idx, sign, mask, y = tensors
    correct = 0
    for start in range(0, len(y), batch):
        sl = slice(start, start + batch)
        n = np.maximum(mask[sl].sum(1, keepdims=True), 1.0)
        logits = (p[idx[sl]] * sign[sl][..., None]).sum(1) / n + bias
        correct += int((logits.argmax(1) == y[sl]).sum())
    return correct / len(y)


def export(model: SpellmanNet, cfg: Config, theta: float, out_dir: Path, store: str, max_drop: float, val_t: tuple) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    emb = model.emb.weight.detach().numpy()  # [D+1, dim]
    w = model.head.weight.detach().numpy()  # [C, dim]
    b = model.head.bias.detach().numpy()  # [C]

    # Train in f32, export in f16: the runtime paths are f16-native (ARM
    # NEON / GPU), the fold is rounded once here, and validation showed no
    # prediction flips from the rounding.
    p16 = (emb @ w.T).astype(np.float16)  # [D+1, C]
    p16[-1, :] = 0  # keep the padding row exactly zero after rounding

    # Storage precision is decoupled from compute: the loader dequantizes
    # any format back into the same table. The gate below refuses to ship a
    # scheme that costs more than --quant-max-drop validation accuracy.
    dtype, scheme = parse_store(store)
    stored, scales = (p16, None) if dtype == "float16" else quantize(p16, dtype, scheme)
    if dtype != "float16":
        base_acc = table_accuracy(p16.astype(np.float32), b, val_t)
        deq = dequantize(stored, scales, dtype, scheme)
        deq[-1, :] = 0
        quant_acc = table_accuracy(deq, b, val_t)
        drop_pp = 100.0 * (base_acc - quant_acc)
        print(f"quantization gate ({store}): val acc {base_acc:.4f} -> {quant_acc:.4f} ({drop_pp:+.2f}pp)")
        print("  " + stats(p16, stored, scales, dtype, scheme))
        if drop_pp > max_drop:
            raise SystemExit(
                f"quantization drop {drop_pp:+.2f}pp exceeds --quant-max-drop {max_drop:.2f}pp; pick another --store"
            )

    tensors_out = {"P": stored, "bias": b.astype(np.float16)}
    if scales is not None:
        tensors_out["scales"] = scales
    save_file(tensors_out, str(out_dir / "model.safetensors"))
    meta = {
        "format": "spellman-model",
        "version": 3,
        "canonicalize": True,
        "lexical": True,
        "languages": LANGUAGES,
        "log2_d": cfg.log2_d,
        "hash": cfg.hash_id,
        "seed": cfg.seed,
        "n_min": 1,
        "n_max": 5,
        "theta": theta,
        "quant": {"dtype": dtype, "scheme": scheme},
    }
    (out_dir / "model.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"exported model to {out_dir} (theta={theta:.3f}, store={store})")


def write_eval_tsv(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            text = " ".join(row["text"].split())
            f.write(f"{row['lang']}\t{text}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "model")
    parser.add_argument("--log2-d", type=int, default=17)
    parser.add_argument("--hash-id", choices=["fmix32", "murmur2", "multiply_shift"], default="fmix32")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--per-lang-cap", type=int, default=50_000)
    parser.add_argument("--hash-stats", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--store",
        default="f16",
        choices=["f16", "int8-row", "int8-col", "fp8-row", "fp8-col"],
        help="folded-table storage format (int8/fp8 halve the artifact; the "
        "loader dequantizes, the runtime graph is unchanged)",
    )
    parser.add_argument(
        "--quant-max-drop",
        type=float,
        default=0.2,
        help="max tolerated validation-accuracy drop (percentage points) for quantized --store",
    )
    args = parser.parse_args()

    cfg = Config(
        log2_d=args.log2_d,
        hash_id=args.hash_id,
        seed=args.seed,
        dim=args.dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        k=args.k,
        lr=args.lr,
        per_lang_cap=args.per_lang_cap,
    )

    train_rows = load_jsonl(args.data / "train.jsonl")
    val_rows = load_jsonl(args.data / "val.jsonl")
    test_rows = load_jsonl(args.data / "test.jsonl")
    print(f"loaded {len(train_rows)} train / {len(val_rows)} val / {len(test_rows)} test", flush=True)

    if args.hash_stats:
        chi_square_stats(train_rows, cfg)

    rng = np.random.default_rng(cfg.seed)
    train_rows = balance_train(train_rows, cfg.per_lang_cap, rng)

    train_t = featurize(train_rows, cfg)
    val_t = featurize(val_rows, cfg)

    device = torch.device(args.device)
    model = SpellmanNet(1 << cfg.log2_d, cfg.dim, len(LANGUAGES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    n = len(train_t[3])
    for epoch in range(cfg.epochs):
        order = rng.permutation(n)
        losses = []
        for start in tqdm(range(0, n, cfg.batch_size), desc=f"epoch {epoch + 1}/{cfg.epochs}", leave=False):
            sel = order[start : start + cfg.batch_size]
            idx = torch.from_numpy(train_t[0][sel]).to(device)
            sign = torch.from_numpy(train_t[1][sel]).to(device)
            mask = torch.from_numpy(train_t[2][sel]).to(device)
            y = torch.from_numpy(train_t[3][sel]).to(device)
            logits = model(idx, sign, mask)
            loss = nn.functional.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
            # Linear decay to zero across all epochs (fastText-style).
            frac = 1.0 - (epoch * n + min(start + cfg.batch_size, n)) / (cfg.epochs * n)
            for group in opt.param_groups:
                group["lr"] = max(cfg.lr * frac, 1e-5)
        val_acc, _ = evaluate(model, val_t, cfg.batch_size)
        print(f"epoch {epoch + 1}: loss {sum(losses) / len(losses):.4f}, val acc {val_acc:.4f}", flush=True)

    # θ calibration: 5th percentile of validation prediction confidence —
    # detections below θ are flagged uncertain by the runtime.
    _, val_confs = evaluate(model, val_t, cfg.batch_size)
    theta = float(np.percentile(val_confs, 5))

    export(model, cfg, theta, args.out, args.store, args.quant_max_drop, val_t)
    write_eval_tsv(test_rows, args.out / "eval_test.tsv")
    write_eval_tsv(val_rows, args.out / "eval_val.tsv")
    print("wrote eval_test.tsv / eval_val.tsv (feed to `cargo run --release --bin assess`)")


if __name__ == "__main__":
    main()
