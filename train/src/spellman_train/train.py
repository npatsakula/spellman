"""Train the spellman fastText-style model and export the folded runtime model.

The trained network is exactly what inference folds:
    scores = mean_over_tokens( sign * E[bucket] ) · W + b
Because the head is linear, E·W collapses to a single [D+1, C] table `P`
(zero row D for padding) — the exported `model.json` + `model.safetensors`
that both the Rust CPU path and the svod JIT path consume.

Usage:
    uv run spellman-train train --data data_mix5 --out ../model --hash-id fmix32

Hash A/B: run with --hash-id {fmix32,murmur2,multiply_shift} and compare
val accuracies; --hash-stats prints a chi-square uniformity check of the
bucket occupancy on the training tokens.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from spellman_train.features import (
    DEFAULT_SEED,
    LANGUAGES,
    bucket_of,
    bucket_tokens_flat,
    token_keys,
)
from spellman_train.paths import MODEL_DIR, TRAIN_DIR
from spellman_train.quantize import dequantize, quantize_int8_col, stats


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
    #: Decoupled weight decay. 0.01 is torch's AdamW default, which the
    #: pipeline inherited SILENTLY for its whole history (fastText, the
    #: design template, has none) — kept as the default so old runs
    #: reproduce; the v13 review flagged it: at lr 0.05 it shrinks
    #: rarely-refreshed embedding rows (rare signature n-grams) by
    #: ~e^-2.5 between touches. --weight-decay 0 is the A/B knob.
    weight_decay: float = 0.01
    per_lang_cap: int = 50_000
    #: "linear" folds into P = E·W at export; "mlp" (experiment) puts a
    #: Linear(dim, hidden) -> GELU between pool and classifier and cannot fold.
    head: str = "linear"
    hidden: int = 256
    #: Seed for the training RNGs (row balancing/order, head init). None
    #: reuses the hash seed, as every run before it did; the hash seed
    #: itself is part of the exported artifact and must not vary between
    #: replicates.
    train_seed: int | None = None
    #: Row-sparse embedding updates (LazyAdamW); False = dense torch AdamW.
    sparse: bool = True

def load_split(data_dir: Path, split: str) -> list[dict]:
    """Read one split of a mix: parquet shard if present, else legacy jsonl.

    Both carry the same {"lang","text"} schema; rows come back as dicts so
    the training code is agnostic of the on-disk format."""
    import polars as pl

    parquet = data_dir / "data" / f"{split}-00000-of-00001.parquet"
    if parquet.exists():
        return pl.read_parquet(parquet).to_dicts()
    return load_jsonl(data_dir / f"{split}.jsonl")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def featurize(rows: list[dict], cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (idx [N,K] i32 buckets, sign [N,K] i8, mask [N,K] i8, y [N] i64).

    Compact host storage (6 B/token instead of 16): the v14 budget (2.5M
    rows x k=512) would not fit a 36 GB machine as i64/f32. Values are exact
    (buckets < 2^31, sign/mask in {-1,0,1}); batches are widened on device.

    Batch-vectorized (bucket_tokens_flat, bit-exact with the scalar
    reference); the K-truncation takes the first k tokens in reference
    encounter order, same as the per-row loop it replaced."""
    n, k = len(rows), cfg.k
    idx = np.zeros((n, k), dtype=np.int32)
    sign = np.zeros((n, k), dtype=np.int8)
    mask = np.zeros((n, k), dtype=np.int8)
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
            sign[t0 + i, :m] = np.where(negs[lo : lo + m], np.int8(-1), np.int8(1))
            mask[t0 + i, :m] = 1
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


def to_device(idx: np.ndarray, sign: np.ndarray, mask: np.ndarray, dev: torch.device):
    """Widen one compact host batch (see featurize) to the net's dtypes."""
    return (
        torch.from_numpy(idx).to(dev).long(),
        torch.from_numpy(sign).to(dev).float(),
        torch.from_numpy(mask).to(dev).float(),
    )


class PooledHead(nn.Module):
    """Classifier over the pooled embedding, isolated so torch.compile
    covers exactly this subgraph: embedding inside the compiled region trips
    an unstable inductor-on-MPS bug ("Tensor device mismatch" / "Placeholder
    storage has not been allocated on MPS device").

    With hidden > 0 (the --head mlp experiment) a Linear(dim, hidden) +
    GELU sits between pool and classifier; that net no longer folds."""

    def __init__(self, dim: int, n_classes: int, hidden: int = 0):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU()) if hidden else nn.Identity()
        self.head = nn.Linear(hidden or dim, n_classes)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.head(self.mlp(pooled))


def mean_pool(table: torch.Tensor, idx: torch.Tensor, sign: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean of the signed rows ``table[idx]`` per sample.

    ``embedding_bag`` with ``sign·mask`` as per-sample weights sums the rows
    without materializing the ``[batch, k, dim]`` gather; that intermediate
    and its backward made a training step ~2.5x slower on MPS (measured,
    M4 Max: 9.6 -> 3.6 ms/step at batch 256, k 512, dim 128).
    """
    summed = F.embedding_bag(idx, table, per_sample_weights=sign * mask, mode="sum")
    return summed / mask.sum(1, keepdim=True).clamp(min=1.0)


class SpellmanNet(nn.Module):
    def __init__(self, d_buckets: int, dim: int, n_classes: int, hidden: int = 0):
        super().__init__()
        self.emb = nn.Embedding(d_buckets + 1, dim)  # +1: padding row D, zeroed at export
        self.post = PooledHead(dim, n_classes, hidden)
        # Zero-init (fastText convention): untrained buckets must fold to
        # exactly-zero logits through P = E·W. With random init, rare-word
        # n-grams that land in never-updated buckets contribute arbitrary
        # nonzero scores — verified failure mode: "sweatshirt" alone scored
        # bul 1.0 because its buckets were untrained noise.
        nn.init.zeros_(self.emb.weight)

    @property
    def head(self) -> nn.Linear:
        return self.post.head

    def forward(self, idx: torch.Tensor, sign: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.post(mean_pool(self.emb.weight, idx, sign, mask))


class LazyAdamW:
    """AdamW for the embedding table that touches only the rows a batch uses.

    Dense ``torch.optim.AdamW`` rewrites all ``(D+1)·dim`` parameters and
    both moment buffers every step, though a batch touches a few thousand
    rows. Here a step gathers the touched rows' state, updates it and
    scatters it back.

    Decoupled weight decay stays exact: the learning-rate schedule is known
    up front, so the decay a row missed while untouched is one factor from a
    prefix sum of ``log(1 - lr_t·wd)``, applied when the row is next touched
    (and to every row by :meth:`finish`). What dense AdamW does and this does
    not: keep moving untouched rows on their decaying momentum.
    """

    def __init__(self, weight: torch.Tensor, lrs: np.ndarray, wd: float,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.weight = weight
        self.m = torch.zeros_like(weight)
        self.v = torch.zeros_like(weight)
        # Steps are 1-based; last[r] = the last step row r was brought up to
        # date at (0 = never).
        self.last = torch.zeros(weight.shape[0], dtype=torch.long, device=weight.device)
        log_keep = np.concatenate([[0.0], np.cumsum(np.log1p(-lrs * wd))])
        self.log_keep = torch.tensor(log_keep, dtype=torch.float32, device=weight.device)
        self.lrs, self.wd, self.betas, self.eps = lrs, wd, betas, eps
        self.t = 0

    def _catch_up(self, rows: torch.Tensor, upto: int) -> torch.Tensor:
        """Decay factor for ``rows`` over steps ``(last, upto]``."""
        return torch.exp(self.log_keep[upto] - self.log_keep[self.last[rows]]).unsqueeze(-1)

    @torch.no_grad()
    def step(self, rows: torch.Tensor, grad: torch.Tensor) -> None:
        """One AdamW step for the unique ``rows`` with their summed ``grad``."""
        self.t += 1
        t, (b1, b2) = self.t, self.betas
        lr = float(self.lrs[t - 1])
        p = self.weight[rows] * self._catch_up(rows, t)  # decay through step t
        m = self.m[rows].mul_(b1).add_(grad, alpha=1 - b1)
        v = self.v[rows].mul_(b2).addcmul_(grad, grad, value=1 - b2)
        denom = (v / (1 - b2**t)).sqrt_().add_(self.eps)
        p.addcdiv_(m, denom, value=-lr / (1 - b1**t))
        self.weight[rows] = p
        self.m[rows] = m
        self.v[rows] = v
        self.last[rows] = t

    @torch.no_grad()
    def finish(self) -> None:
        """Bring every row's weight decay up to the last step taken."""
        self.weight.mul_(torch.exp(self.log_keep[self.t] - self.log_keep[self.last]).unsqueeze(-1))
        self.last.fill_(self.t)


def lr_schedule(cfg: Config, n: int) -> np.ndarray:
    """Per-step learning rate: linear decay to zero across all epochs
    (fastText-style), floored at 1e-5 — the value each optimizer step uses."""
    steps = []
    for epoch in range(cfg.epochs):
        for start in range(0, n, cfg.batch_size):
            frac = 1.0 - (epoch * n + min(start + cfg.batch_size, n)) / (cfg.epochs * n)
            steps.append(max(cfg.lr * frac, 1e-5))
    # Step i runs with the rate set after step i-1 (the first with cfg.lr).
    return np.array([cfg.lr] + steps[:-1])


@torch.no_grad()
def evaluate(model: SpellmanNet, tensors: tuple, batch_size: int) -> tuple[float, np.ndarray]:
    model.eval()
    idx, sign, mask, y = tensors
    correct = 0
    confs: list[np.ndarray] = []
    for start in range(0, len(y), batch_size):
        sl = slice(start, start + batch_size)
        dev = next(model.parameters()).device
        logits = model(*to_device(idx[sl], sign[sl], mask[sl], dev))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
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
        logits = (p[idx[sl]] * sign[sl][..., None].astype(np.float32)).sum(1) / n + bias
        correct += int((logits.argmax(1) == y[sl]).sum())
    return correct / len(y)


def export(model: SpellmanNet, cfg: Config, theta: float, out_dir: Path, max_drop: float, val_t: tuple) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    emb = model.emb.weight.detach().cpu().numpy()  # [D+1, dim]
    w = model.head.weight.detach().cpu().numpy()  # [C, dim]
    b = model.head.bias.detach().cpu().numpy()  # [C]

    # Train in f32, fold once and round to f16 (validation showed no
    # prediction flips from the rounding); the int8 table is quantized from
    # that fold, so the gate measures exactly what ships.
    p16 = (emb @ w.T).astype(np.float16)  # [D+1, C]
    p16[-1, :] = 0  # keep the padding row exactly zero after rounding

    # The runtime's one storage format: int8 with per-column scales. The
    # gate refuses to ship a table that costs more than --quant-max-drop
    # validation accuracy against the f16 fold.
    stored, scales = quantize_int8_col(p16)
    base_acc = table_accuracy(p16.astype(np.float32), b, val_t)
    deq = dequantize(stored, scales, "int8", "column")
    deq[-1, :] = 0
    quant_acc = table_accuracy(deq, b, val_t)
    drop_pp = 100.0 * (base_acc - quant_acc)
    print(f"quantization gate (int8-col): val acc {base_acc:.4f} -> {quant_acc:.4f} ({drop_pp:+.2f}pp)")
    print("  " + stats(p16, stored, scales))
    if drop_pp > max_drop:
        raise SystemExit(f"quantization drop {drop_pp:+.2f}pp exceeds --quant-max-drop {max_drop:.2f}pp")

    tensors_out = {"P": stored, "bias": b.astype(np.float16), "scales": scales}
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
        "quant": {"dtype": "int8", "scheme": "column"},
    }
    (out_dir / "model.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"exported model to {out_dir} (theta={theta:.3f}, int8-col)")


def write_eval_tsv(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            text = " ".join(row["text"].split())
            f.write(f"{row['lang']}\t{text}\n")


def populate(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--data", type=Path, default=TRAIN_DIR / "data")
    ap.add_argument("--out", type=Path, default=MODEL_DIR)
    ap.add_argument("--log2-d", type=int, default=17)
    ap.add_argument("--hash-id", choices=["fmix32", "murmur2", "multiply_shift"], default="fmix32")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.01,
                    help="AdamW decoupled decay (0.01 = the historically-implicit torch default)")
    ap.add_argument("--per-lang-cap", type=int, default=50_000)
    ap.add_argument("--head", choices=["linear", "mlp"], default="linear",
                    help="mlp = experiment: pool -> Linear(dim, hidden) -> GELU -> Linear; "
                    "not foldable, saves model.pt only (no runtime export)")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--train-seed", type=int, default=None,
                    help="seed for row balancing/order and head init (default: the hash --seed, "
                    "as before); vary this, not --seed, for replicates")
    ap.add_argument("--dense", action="store_true",
                    help="dense torch AdamW over the whole embedding table instead of the "
                    "row-sparse LazyAdamW (the pre-sparse behavior, for A/B)")
    ap.add_argument("--hash-stats", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile the classifier (on by default; M4 Max, sparse "
        "updates: 9.6 -> 7.9 ms/step before embedding_bag pooling; warmup ~1s, "
        "two extra recompiles for the partial eval batches)",
    )
    ap.add_argument(
        "--quant-max-drop",
        type=float,
        default=0.2,
        help="max tolerated validation-accuracy drop (percentage points) of the "
        "exported int8 table against the f16 fold",
    )


def run(args: argparse.Namespace) -> None:
    cfg = Config(
        log2_d=args.log2_d,
        hash_id=args.hash_id,
        seed=args.seed,
        dim=args.dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        k=args.k,
        lr=args.lr,
        weight_decay=args.weight_decay,
        per_lang_cap=args.per_lang_cap,
        head=args.head,
        hidden=args.hidden,
        train_seed=args.train_seed,
        sparse=not args.dense,
    )

    train_rows = load_split(args.data, "train")
    val_rows = load_split(args.data, "val")
    test_rows = load_split(args.data, "test")
    print(f"loaded {len(train_rows)} train / {len(val_rows)} val / {len(test_rows)} test", flush=True)

    if args.hash_stats:
        chi_square_stats(train_rows, cfg)

    train_seed = cfg.seed if cfg.train_seed is None else cfg.train_seed
    rng = np.random.default_rng(train_seed)
    torch.manual_seed(train_seed)
    train_rows = balance_train(train_rows, cfg.per_lang_cap, rng)

    train_t = featurize(train_rows, cfg)
    val_t = featurize(val_rows, cfg)

    device = torch.device(args.device)
    hidden = cfg.hidden if cfg.head == "mlp" else 0
    model = SpellmanNet(1 << cfg.log2_d, cfg.dim, len(LANGUAGES), hidden).to(device)
    n = len(train_t[3])
    lrs = lr_schedule(cfg, n)
    if cfg.sparse:
        # The embedding table goes to LazyAdamW; only the head is dense.
        emb_opt = LazyAdamW(model.emb.weight.data, lrs, cfg.weight_decay)
        opt = torch.optim.AdamW(model.post.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raw = model  # eval/export use the uncompiled module
    # Compile into a separate handle: assigning it back to model.post would
    # prefix the saved state_dict keys with `_orig_mod.`.
    post = torch.compile(model.post) if args.compile else model.post
    for epoch in range(cfg.epochs):
        order = rng.permutation(n)
        losses = []
        t_epoch = time.perf_counter()
        for start in tqdm(range(0, n, cfg.batch_size), desc=f"epoch {epoch + 1}/{cfg.epochs}", leave=False):
            sel = order[start : start + cfg.batch_size]
            idx, sign, mask = to_device(train_t[0][sel], train_t[1][sel], train_t[2][sel], device)
            y = torch.from_numpy(train_t[3][sel]).to(device)
            if cfg.sparse:
                # Gather each touched row once as a leaf: backward then
                # yields one summed gradient per unique row, never a dense
                # (D+1)·dim one.
                rows, inverse = torch.unique(idx, return_inverse=True)
                leaf = raw.emb.weight.detach()[rows].requires_grad_()
                logits = post(mean_pool(leaf, inverse, sign, mask))
            else:
                logits = post(mean_pool(model.emb.weight, idx, sign, mask))
            loss = nn.functional.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if cfg.sparse:
                emb_opt.step(rows, leaf.grad)
            losses.append(float(loss))
            # Linear decay to zero across all epochs (fastText-style).
            frac = 1.0 - (epoch * n + min(start + cfg.batch_size, n)) / (cfg.epochs * n)
            for group in opt.param_groups:
                group["lr"] = max(cfg.lr * frac, 1e-5)
        if cfg.sparse:
            emb_opt.finish()  # untouched rows owe decay; settle it before val/export
        t_train = time.perf_counter() - t_epoch
        val_acc, _ = evaluate(raw, val_t, cfg.batch_size)
        print(
            f"epoch {epoch + 1}: loss {sum(losses) / len(losses):.4f}, val acc {val_acc:.4f}, "
            f"train {t_train:.0f}s",
            flush=True,
        )

    # θ calibration: 5th percentile of validation prediction confidence —
    # detections below θ are flagged uncertain by the runtime.
    _, val_confs = evaluate(raw, val_t, cfg.batch_size)
    theta = float(np.percentile(val_confs, 5))

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"config": vars(cfg), "state_dict": raw.state_dict()}, args.out / "model.pt")
    if cfg.head == "mlp":
        print(f"saved {args.out / 'model.pt'} (mlp head: no folded export)")
    else:
        export(raw, cfg, theta, args.out, args.quant_max_drop, val_t)
    write_eval_tsv(test_rows, args.out / "eval_test.tsv")
    write_eval_tsv(val_rows, args.out / "eval_val.tsv")
    print("wrote eval_test.tsv / eval_val.tsv (feed to `cargo run --release --bin assess`)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="spellman-train train", description=__doc__)
    populate(ap)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
