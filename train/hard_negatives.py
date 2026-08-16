"""Hard negatives from FineWeb-2 `_removed` subsets.

FineWeb-2's per-language configs ship a `_removed` split: documents rejected
from e.g. udm_Cyrl because the pipeline's LID judged them ANOTHER language —
overwhelmingly Russian, the regional lingua franca. They are exactly the
boundary examples our residual confusion needs (udm->rus, sah->rus,
kpv->rus): web text that looks like it belongs to a small language's corpus
but is actually Russian.

Each removed window is labeled by the CURRENT promoted model; a window is
kept only when the model is confident (>= --conf) AND the predicted label is
outside the source language's twin group (never re-label a twin — the same
rule as hygiene) AND differs from the source language. Windows that don't
qualify are dropped: hard negatives must be unambiguous.

Output: one cache-style JSONL ({"lang","text"}) consumed via the `jsonl`
source, e.g.  --source jsonl:path=hard_negatives.jsonl

Usage: uv run python hard_negatives.py [--conf 0.98] [--per-lang 4000]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from spellman_features import LANGUAGES
from hygiene import load_judge, predict_batch
from sources.fineweb2 import windows_from_doc

# Languages whose removed subsets target our residual rus-attraction
# confusions (top confusion pairs from the last error audit).
TARGETS = [
    "udm_Cyrl", "sah_Cyrl", "kpv_Cyrl", "che_Cyrl", "oss_Cyrl", "tyv_Cyrl",
    "mhr_Cyrl", "tat_Cyrl", "bak_Cyrl", "kaz_Cyrl", "kir_Cyrl", "chv_Cyrl",
]

# Mirrors hygiene.TWIN_GROUPS (source-config -> languages we never re-label to).
TWIN_GROUPS = [
    {"tat", "bak", "kaz", "kir", "tyv", "chv", "sah"},
    {"udm", "mhr", "kpv"},
    {"bul", "mkd", "srp"},
    {"rus", "ukr", "bel"},
]


def twin_protected(gold: str, pred: str) -> bool:
    return any(gold in g and pred in g for g in TWIN_GROUPS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(__file__).parent.parent / "model")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "cache" / "hard_negatives.jsonl")
    parser.add_argument("--conf", type=float, default=0.98)
    parser.add_argument("--per-lang", type=int, default=4000, help="kept windows per source config")
    parser.add_argument("--docs", type=int, default=6000, help="removed docs to stream per config")
    parser.add_argument("--per-doc", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import load_dataset

    P, bias, log2_d, seed, hash_id = load_judge(args.model)

    kept: list[tuple[str, str]] = []
    stats: Counter = Counter()
    for cfg in TARGETS:
        src_lang = cfg.split("_")[0]
        try:
            # The removed subsets are not named configs; the generic parquet
            # builder with hf:// URLs streams them directly.
            ds = load_dataset(
                "parquet",
                data_files=f"hf://datasets/HuggingFaceFW/fineweb-2/data/{cfg}_removed/train/*.parquet",
                streaming=True,
                split="train",
            )
        except Exception as e:  # noqa: BLE001 — a missing subset must not kill the run
            print(f"  {cfg}: SKIP ({e.__class__.__name__})")
            continue

        import random

        rng = random.Random(args.seed)
        windows: list[str] = []
        for doc in ds.take(args.docs):
            windows.extend(windows_from_doc(doc["text"], rng, args.per_doc))
            if len(windows) >= args.per_lang * 2:
                break
        if not windows:
            print(f"  {cfg}: no windows")
            continue

        preds = predict_batch(windows, P, bias, log2_d, seed, hash_id)
        n = 0
        for text, (top, conf) in zip(windows, preds):
            if n >= args.per_lang:
                break
            if top < 0 or conf < args.conf:
                continue
            label = LANGUAGES[top]
            if label == src_lang or twin_protected(src_lang, label):
                continue
            kept.append((label, text))
            stats[label] += 1
            n += 1
        print(f"  {cfg}: kept {n} (labels so far: {dict(stats.most_common(3))})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for lang, text in kept:
            f.write(json.dumps({"lang": lang, "text": text}, ensure_ascii=False) + "\n")
    print(f"wrote {len(kept)} hard negatives -> {args.out}; labels: {dict(stats.most_common(8))}")


if __name__ == "__main__":
    main()
