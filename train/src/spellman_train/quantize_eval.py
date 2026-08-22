"""Rewrite a model artifact in a quantized storage format and report damage.

Writes a REAL stored-quant artifact (int8/fp8 `P` + f32 `scales` + the
`quant` block in model.json) — the same thing `spellman-train train --store`
exports — so `assess` / `spellman` on the output dir exercises the actual
loader dequant path end to end:

    uv run spellman-train quantize --store int8-row --out /tmp/model-int8
    ../target/release/assess --model /tmp/model-int8 ../model/eval_test.tsv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

from spellman_train.paths import MODEL_DIR, REPO_ROOT
from spellman_train.quantize import dequantize, parse_store, quantize, stats


def populate(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", type=Path, default=MODEL_DIR)
    ap.add_argument("--store", default="int8-row", help="f16 | int8-row | int8-col | fp8-row | fp8-col")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "model-int8")


def run(args: argparse.Namespace) -> None:
    dtype, scheme = parse_store(args.store)
    weights = load_file(str(args.model / "model.safetensors"))
    p16 = weights["P"].astype(np.float32)
    meta = json.loads((args.model / "model.json").read_text())

    stored, scales = quantize(p16, dtype, scheme)
    print(stats(p16, stored, scales, dtype, scheme))
    deq = dequantize(stored, scales, dtype, scheme)
    assert (deq[-1] == 0).all(), "padding row must stay exactly zero"

    meta["quant"] = {"dtype": dtype, "scheme": scheme}
    tensors = {"P": stored, "bias": weights["bias"]}
    if scales is not None:
        tensors["scales"] = scales
    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out / "model.safetensors"))
    (args.out / "model.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {args.store} artifact -> {args.out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="spellman-train quantize", description=__doc__)
    populate(ap)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
