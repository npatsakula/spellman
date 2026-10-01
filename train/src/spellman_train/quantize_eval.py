"""Convert a model artifact to the runtime's storage format and report damage.

Reads any format spellman ever exported (f16, row-scaled int8, fp8 — the
``quant`` block in model.json says which; none means f16) and writes the
int8 table with per-column scales the runtime scores, so ``assess`` /
``spellman`` on the output dir exercise the real loader end to end:

    uv run spellman-train quantize --model /path/to/old-model --out /tmp/model
    ../target/release/assess --model /tmp/model ../model/eval_test.tsv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

from spellman_train.paths import MODEL_DIR, REPO_ROOT
from spellman_train.quantize import dequantize, quantize_int8_col, stats


def populate(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", type=Path, default=MODEL_DIR)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "model-int8")


def run(args: argparse.Namespace) -> None:
    weights = load_file(str(args.model / "model.safetensors"))
    meta = json.loads((args.model / "model.json").read_text())
    quant = meta.get("quant", {"dtype": "float16", "scheme": "none"})
    p = dequantize(weights["P"], weights.get("scales"), quant["dtype"], quant["scheme"])
    p[-1, :] = 0  # the padding row is part of the contract

    q, scales = quantize_int8_col(p)
    print(f"from {quant['dtype']}/{quant['scheme']}: " + stats(p, q, scales))
    meta["quant"] = {"dtype": "int8", "scheme": "column"}
    args.out.mkdir(parents=True, exist_ok=True)
    save_file({"P": q, "bias": weights["bias"], "scales": scales}, str(args.out / "model.safetensors"))
    (args.out / "model.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote int8-col artifact -> {args.out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="spellman-train quantize", description=__doc__)
    populate(ap)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
