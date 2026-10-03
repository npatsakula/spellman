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
from collections import deque
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from torch import nn
import torch.nn.functional as F
import torchdata.nodes as tn
from tqdm import tqdm

from spellman_train.features import (
    DEFAULT_SEED,
    LANGUAGES,
    bucket_of,
    bucket_tokens_flat,
    token_keys,
)
from spellman_train.mix import parse_cap_overrides
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


@dataclass
class Ragged:
    """Featurized rows, stored flat: row ``i``'s tokens are
    ``bucket[off[i]:off[i+1]]`` (first k only), signed by ``neg``.

    No padding: the v14 budget's rows average ~283 of k=512 tokens, so the
    padded [N, k] layout spent 45% of its memory and of every batch's gather
    on padding."""

    bucket: np.ndarray  # [T] int32
    neg: np.ndarray  # [T] bool
    off: np.ndarray  # [N+1] int64
    y: np.ndarray  # [N] int64
    log2_d: int
    #: [T] bool, lexical-channel tokens (word / word-pair keys); only kept
    #: when --lexical-dropout needs to tell them from n-grams
    lex: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.y)


def featurize(rows: list[dict], cfg: Config, with_lexical: bool = False) -> Ragged:
    """Featurize rows into a :class:`Ragged` set, keeping each row's first
    ``cfg.k`` tokens in reference encounter order.

    Batch-vectorized (bucket_tokens_flat, bit-exact with the scalar
    reference)."""
    y = np.array([LANG_TO_IDX[row["lang"]] for row in rows], dtype=np.int64)
    buckets, negs, lens, lexs = [], [], [], []
    chunk = 20_000
    for t0 in tqdm(range(0, len(rows), chunk), desc="featurize", leave=False):
        part = rows[t0 : t0 + chunk]
        b, ng, off, *lx = bucket_tokens_flat(
            [r["text"] for r in part], cfg.log2_d, cfg.hash_id, cfg.seed, with_lexical=with_lexical
        )
        off = np.asarray(off, dtype=np.int64)
        full = np.diff(off)
        # Position of every token inside its row; keep the first k.
        pos = np.arange(off[-1]) - np.repeat(off[:-1], full)
        keep = pos < cfg.k
        buckets.append(np.asarray(b)[keep].astype(np.int32))
        negs.append(np.asarray(ng)[keep].astype(bool))
        if with_lexical:
            lexs.append(lx[0][keep])
        lens.append(np.minimum(full, cfg.k))
    lens_all = np.concatenate(lens) if lens else np.zeros(0, dtype=np.int64)
    return Ragged(
        bucket=np.concatenate(buckets) if buckets else np.zeros(0, dtype=np.int32),
        neg=np.concatenate(negs) if negs else np.zeros(0, dtype=bool),
        off=np.concatenate([[0], np.cumsum(lens_all)]).astype(np.int64),
        y=y,
        log2_d=cfg.log2_d,
        lex=(np.concatenate(lexs) if lexs else np.zeros(0, dtype=bool)) if with_lexical else None,
    )


@dataclass
class Batch:
    """One host batch of a :class:`Ragged` set, laid out for
    ``embedding_bag``: ``ids`` are bucket ids, or with ``unique`` the
    indices into ``rows`` (the batch's distinct buckets)."""

    ids: np.ndarray  # [t] int64
    sign: np.ndarray  # [t] f32
    offsets: np.ndarray  # [b] int64, bag starts
    lens: np.ndarray  # [b] f32
    y: np.ndarray  # [b] int64
    rows: np.ndarray | None = None  # [u] int64

    def to(self, dev: torch.device) -> dict[str, torch.Tensor]:
        """Copy to ``dev`` without blocking: a blocking host-to-MPS copy
        waits for the GPU queue to drain, serializing host and GPU (3.8 ->
        2.2 ms/step, measured). The copies read this batch's arrays
        asynchronously, so the caller keeps the batch alive for a few steps.
        """
        fields = {k: v for k, v in vars(self).items() if v is not None}
        return {k: torch.from_numpy(v).to(dev, non_blocking=True) for k, v in fields.items()}


def table_rows(data: Ragged) -> int:
    """Rows of the embedding table ``data``'s buckets index (D+1)."""
    return (1 << data.log2_d) + 1


def make_batch(data: Ragged, sel: np.ndarray, unique: bool, drop: np.ndarray | None = None) -> Batch:
    """Gather rows ``sel`` of ``data`` (host side; runs in the prefetch
    threads, so the GPU never waits on data-dependent shapes). Rows flagged
    in ``drop`` (bool, aligned with ``sel``) lose their lexical-channel
    tokens (--lexical-dropout)."""
    starts = data.off[sel]
    lens = data.off[sel + 1] - starts
    ends = np.cumsum(lens)
    pos = np.repeat(starts - (ends - lens), lens) + np.arange(ends[-1] if len(ends) else 0)
    if drop is not None and drop.any():
        keep = ~(data.lex[pos] & np.repeat(drop, lens))
        kept = np.concatenate(([0], np.cumsum(keep)))
        lens = kept[ends] - kept[ends - lens]
        pos = pos[keep]
        ends = np.cumsum(lens)
    ids = data.bucket[pos].astype(np.int64)
    sign = np.where(data.neg[pos], np.float32(-1), np.float32(1))
    rows = None
    if unique:
        # Buckets are dense small ints, so a bincount over the table finds
        # the distinct rows in sorted order — np.unique's result, without
        # its sort (0.5 vs 3.2 ms per batch).
        rows = np.flatnonzero(np.bincount(ids, minlength=table_rows(data)))
        rank = np.empty(table_rows(data), dtype=np.int64)
        rank[rows] = np.arange(len(rows))
        ids = rank[ids]
    return Batch(ids=ids, sign=sign, offsets=(ends - lens).astype(np.int64),
                 lens=lens.astype(np.float32), y=data.y[sel], rows=rows)


def batches(data: Ragged, sels: list, unique: bool, workers: int = 4, ahead: int = 8):
    """Host batches for the row selections ``sels``, in order, built on a
    thread pool and kept ``ahead`` batches in front of the consumer (numpy
    releases the GIL in the gather and the unique-sort). Threads, not
    DataLoader worker processes: those would each receive a pickled copy of
    the multi-GB feature arrays."""
    node = tn.IterableWrapper(sels)
    node = tn.ParallelMapper(node, map_fn=lambda sel: make_batch(data, *sel, unique=unique) if isinstance(sel, tuple) else make_batch(data, sel, unique), num_workers=workers, method="thread")
    return tn.Loader(tn.Prefetcher(node, prefetch_factor=ahead))


def balance_train(
    rows: list[dict], cap: int, rng: np.random.Generator, overrides: dict[str, int] | None = None
) -> list[dict]:
    """Subsample each language to ``cap`` rows (``overrides[lang]`` where
    given, --cap-override)."""
    by_lang: dict[str, list[dict]] = {}
    for row in rows:
        by_lang.setdefault(row["lang"], []).append(row)
    out: list[dict] = []
    for lang, items in sorted(by_lang.items()):
        lang_cap = (overrides or {}).get(lang, cap)
        if len(items) > lang_cap:
            items = list(rng.choice(items, size=lang_cap, replace=False))
        out.extend(items)
    return out


class PooledHead(nn.Module):
    """Classifier over the pooled embedding.

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


def bag_pool(table: torch.Tensor, b: dict[str, torch.Tensor]) -> torch.Tensor:
    """:func:`mean_pool` over a device :class:`Batch` (flat ids + offsets)."""
    summed = F.embedding_bag(b["ids"], table, b["offsets"], mode="sum", per_sample_weights=b["sign"])
    return summed / b["lens"].clamp(min=1.0).unsqueeze(1)


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
def evaluate(model: SpellmanNet, data: Ragged, batch_size: int) -> tuple[float, np.ndarray]:
    model.eval()
    dev = next(model.parameters()).device
    correct = 0
    confs: list[np.ndarray] = []
    sels = [np.arange(s, min(s + batch_size, len(data))) for s in range(0, len(data), batch_size)]
    for batch in batches(data, sels, unique=False):
        # .cpu() below syncs every batch, so its async copies are done
        # before the next batch replaces it.
        logits = model.post(bag_pool(model.emb.weight, batch.to(dev)))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred = probs.argmax(1)
        correct += int((pred == batch.y).sum())
        confs.append(probs[np.arange(len(pred)), pred])
    model.train()
    return correct / len(data), np.concatenate(confs)


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
    data: Ragged = tensors
    correct = 0
    for start in range(0, len(data), batch):
        b = make_batch(data, np.arange(start, min(start + batch, len(data))), unique=False)
        sums = np.zeros((len(b.y), p.shape[1]), dtype=np.float32)
        full = b.lens > 0
        if full.any():
            # reduceat over non-empty bags only: an empty bag has no tokens,
            # so the segments between non-empty starts are exactly the bags.
            sums[full] = np.add.reduceat(p[b.ids] * b.sign[:, None], b.offsets[full], axis=0)
        logits = sums / np.maximum(b.lens, 1.0)[:, None] + bias
        correct += int((logits.argmax(1) == b.y).sum())
    return correct / len(data)


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
    ap.add_argument("--cap-override", action="append", default=[], metavar="LANG=N",
                    help="per-language train cap replacing --per-lang-cap for LANG "
                    "(repeatable, e.g. rus=240000)")
    ap.add_argument("--lexical-dropout", type=float, default=0.0, metavar="P",
                    help="per training row and epoch, with probability P drop the row's "
                    "whole-word and word-pair features so the model cannot lean on the "
                    "lexical channel alone (0 = off; inference is unchanged)")
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
        help="torch.compile the forward pass (on by default; M4 Max, ragged "
        "sparse batches: 6.9 -> 3.5 ms/step; warmup < 1s)",
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
    train_rows = balance_train(train_rows, cfg.per_lang_cap, rng, parse_cap_overrides(args.cap_override))

    train_t = featurize(train_rows, cfg, with_lexical=args.lexical_dropout > 0)
    val_t = featurize(val_rows, cfg)

    device = torch.device(args.device)
    hidden = cfg.hidden if cfg.head == "mlp" else 0
    model = SpellmanNet(1 << cfg.log2_d, cfg.dim, len(LANGUAGES), hidden).to(device)
    n = len(train_t)
    lrs = lr_schedule(cfg, n)
    if cfg.sparse:
        # The embedding table goes to LazyAdamW; only the head is dense.
        emb_opt = LazyAdamW(model.emb.weight.data, lrs, cfg.weight_decay)
        opt = torch.optim.AdamW(model.post.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raw = model  # eval/export use the uncompiled module

    def forward(table: torch.Tensor, b: dict[str, torch.Tensor]) -> torch.Tensor:
        return model.post(bag_pool(table, b))

    # The whole forward, embedding_bag included (torch 2.14.1 on MPS; the
    # inductor-on-MPS embedding bug the head-only compile once dodged is
    # gone). dynamic=True: token and unique-row counts change every batch.
    if args.compile:
        forward = torch.compile(forward, dynamic=True)
    for epoch in range(cfg.epochs):
        order = rng.permutation(n)
        loss_sum = torch.zeros((), device=device)  # read once per epoch: no per-step sync
        steps = 0
        t_epoch = time.perf_counter()
        sels = [order[s : s + cfg.batch_size] for s in range(0, n, cfg.batch_size)]
        if args.lexical_dropout > 0:
            dropped = rng.random(n) < args.lexical_dropout  # per row, redrawn every epoch
            sels = [(sel, dropped[sel]) for sel in sels]
        loader = batches(train_t, sels, unique=cfg.sparse)
        in_flight: deque = deque(maxlen=16)  # host batches their async copies may still read
        for i, host in enumerate(tqdm(loader, total=len(sels), desc=f"epoch {epoch + 1}/{cfg.epochs}", leave=False)):
            in_flight.append(host)
            b = host.to(device)
            if cfg.sparse:
                # Gather each touched row once as a leaf (the host batch
                # already deduplicated them): backward then yields one summed
                # gradient per unique row, never a dense (D+1)·dim one.
                leaf = raw.emb.weight.detach()[b["rows"]].requires_grad_()
                logits = forward(leaf, b)
            else:
                logits = forward(model.emb.weight, b)
            loss = nn.functional.cross_entropy(logits, b["y"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if cfg.sparse:
                emb_opt.step(b["rows"], leaf.grad)
            loss_sum += loss.detach()
            steps += 1
            # Linear decay to zero across all epochs (fastText-style).
            frac = 1.0 - (epoch * n + min((i + 1) * cfg.batch_size, n)) / (cfg.epochs * n)
            for group in opt.param_groups:
                group["lr"] = max(cfg.lr * frac, 1e-5)
        if cfg.sparse:
            emb_opt.finish()  # untouched rows owe decay; settle it before val/export
        t_train = time.perf_counter() - t_epoch
        val_acc, _ = evaluate(raw, val_t, cfg.batch_size)
        print(
            f"epoch {epoch + 1}: loss {float(loss_sum) / steps:.4f}, val acc {val_acc:.4f}, "
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
