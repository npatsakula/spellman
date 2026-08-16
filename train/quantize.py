"""Table quantization for the spellman folded model.

Storage precision is decoupled from compute: the Rust loader always
reconstructs the canonical f32 table from whatever is stored, so these
formats change only artifact size / download bandwidth, never the runtime
graph. Schemes:

- ``f16``                — passthrough (no ``scales`` tensor)
- ``int8-row``/``int8-col`` — symmetric int8, one f32 scale per bucket row
  (finer) or per language column (factors out of the token sum)
- ``fp8-row``/``fp8-col``   — FP8 E4M3FN bits as raw u8 (needs ml_dtypes)

Quantization always runs on the *f16-rounded* fold, so the gate in
``train.py`` measures exactly what the artifact will store.
"""

from __future__ import annotations

import numpy as np

INT8_MAX = 127.0
FP8_MAX = 448.0  # e4m3fn: no infinities, largest finite value


def parse_store(spec: str) -> tuple[str, str]:
    """``"int8-row"`` -> ``("int8", "row")``; ``"f16"`` -> ``("float16", "none")``."""
    if spec == "f16":
        return "float16", "none"
    dtype, _, scheme = spec.partition("-")
    if dtype not in ("int8", "fp8") or scheme not in ("row", "col"):
        raise SystemExit(f"bad --store {spec!r} (f16 | int8-row | int8-col | fp8-row | fp8-col)")
    return ("fp8e4m3" if dtype == "fp8" else "int8"), ("column" if scheme == "col" else "row")


def _scales(p: np.ndarray, scheme: str, qmax: float) -> np.ndarray:
    axis = 1 if scheme == "row" else 0
    s = np.abs(p).max(axis=axis, keepdims=True) / qmax
    s = np.where(s == 0, 1.0, s)  # all-zero rows (untrained buckets) stay zero
    return s.astype(np.float32)


def quantize(p: np.ndarray, dtype: str, scheme: str) -> tuple[np.ndarray, np.ndarray | None]:
    """f32 (or f16-valued) table -> (stored tensor, scales | None)."""
    p = p.astype(np.float32)
    if dtype == "float16":
        return p.astype(np.float16), None
    if dtype == "int8":
        s = _scales(p, scheme, INT8_MAX)
        q = np.clip(np.round(p / s), -INT8_MAX, INT8_MAX).astype(np.int8)
        return q, s.reshape(-1)
    if dtype == "fp8e4m3":
        import ml_dtypes  # optional dependency, only for fp8 storage

        s = _scales(p, scheme, FP8_MAX)
        q = np.clip(p / s, -FP8_MAX, FP8_MAX)
        bits = q.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
        return bits.reshape(p.shape), s.reshape(-1)
    raise SystemExit(f"unknown quant dtype {dtype!r}")


def dequantize(stored: np.ndarray, scales: np.ndarray | None, dtype: str, scheme: str) -> np.ndarray:
    """Reconstruct the f32 table exactly the way the Rust loader does."""
    if dtype == "float16":
        return stored.astype(np.float32)
    values = stored.astype(np.float32)
    if dtype == "fp8e4m3":
        import ml_dtypes

        values = stored.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    s = scales.astype(np.float32)
    if scheme == "row":
        return values * s[:, None]
    return values * s[None, :]


def stats(p: np.ndarray, stored: np.ndarray, scales: np.ndarray | None, dtype: str, scheme: str) -> str:
    """One-line human report of the quantization damage and the size win."""
    deq = dequantize(stored, scales, dtype, scheme)
    err = np.abs(deq - p.astype(np.float32))
    f16_bytes = p.size * 2
    stored_bytes = stored.nbytes + (0 if scales is None else scales.nbytes)
    zeros = "" if scales is None else f", {(np.asarray(stored) == 0).mean():.1%} entries exact zero"
    return (
        f"{dtype}/{scheme}: quant err max {err.max():.4f} mean {err.mean():.6f}{zeros}; "
        f"artifact {stored_bytes / 1e6:.2f} MB vs {f16_bytes / 1e6:.2f} MB f16"
    )
