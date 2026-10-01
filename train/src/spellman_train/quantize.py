"""Table quantization for the spellman folded model.

The runtime scores one storage format: symmetric int8 with one f32 scale per
language column. Column scales factor out of the token sum
(``Σ s_c·q = s_c·Σ q``), so the Rust runtime gathers and sums the int8
table directly and applies the scales once per document. It matched f16
accuracy on every eval (within ±0.02pp) at half the size and ran faster on
every path measured, so the other formats (f16, row-scaled int8, fp8) are
no longer written.

:func:`dequantize` still reads all of them, so ``spellman-train quantize``
can convert an older artifact.

Quantization always runs on the *f16-rounded* fold, so the gate in
``train.py`` measures exactly what the artifact will store.
"""

from __future__ import annotations

import numpy as np

INT8_MAX = 127.0


def quantize_int8_col(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """f32 (or f16-valued) ``[D+1, C]`` table -> (int8 table, f32 column scales)."""
    p = p.astype(np.float32)
    s = np.abs(p).max(axis=0, keepdims=True) / INT8_MAX
    s = np.where(s == 0, 1.0, s).astype(np.float32)  # all-zero columns stay zero
    q = np.clip(np.round(p / s), -INT8_MAX, INT8_MAX).astype(np.int8)
    return q, s.reshape(-1)


def dequantize(stored: np.ndarray, scales: np.ndarray | None, dtype: str, scheme: str) -> np.ndarray:
    """Reconstruct the f32 table from any format spellman ever exported
    (``float16``/``none``, ``int8`` or ``fp8e4m3`` with ``row``/``column``
    scales) — for the export gate and for converting older artifacts."""
    if dtype == "float16":
        return stored.astype(np.float32)
    values = stored.astype(np.float32)
    if dtype == "fp8e4m3":
        import ml_dtypes  # optional dependency, only for reading fp8 artifacts

        values = stored.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    s = scales.astype(np.float32)
    if scheme == "row":
        return values * s[:, None]
    return values * s[None, :]


def stats(p: np.ndarray, q: np.ndarray, scales: np.ndarray) -> str:
    """One-line human report of the quantization damage and the size win."""
    deq = dequantize(q, scales, "int8", "column")
    err = np.abs(deq - p.astype(np.float32))
    return (
        f"int8/column: quant err max {err.max():.4f} mean {err.mean():.6f}, "
        f"{(q == 0).mean():.1%} entries exact zero; artifact "
        f"{(q.nbytes + scales.nbytes) / 1e6:.2f} MB vs {p.size * 2 / 1e6:.2f} MB f16"
    )
