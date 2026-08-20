"""Cache hygiene: drop rows the current model contradicts at high confidence.

Two rules:

1. Model judge (default): rows whose top-1 prediction differs from their
   label at >= --conf are dropped. Catches verbatim other-language text
   (Uzbek inside Glot500 tgk, Russian inside sah/kpv sources) — verified by
   the manual error-dump audit.

2. Script-mismatch judge (--script): a row labeled with a Cyrillic-column
   language whose text is majority-Latin is judged by fastText lid.176
   instead — the exported model CANNOT judge these (it was trained on the
   contamination; e.g. FineWeb-2's bul config ships ~700 majority-Latin
   English rows, which is how English translatorese learned to vote bul).
   fastText's blind spots are the small Cyrillic languages; on Latin-script
   text for the big-5 its top-1 is reliable. Uses the fasttext-numpy2
   build (numpy 2 ABI) directly — no overlay dance.

Caches are rewritten in place; sidecars are untouched, so the options
fingerprint stays valid and the mixer replays the cleaned rows. Caveat:
rebuilding a cache (option change) re-downloads the dirty upstream data —
rerun this tool after any rebuild.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
from safetensors.numpy import load_file

from spellman_features import LANGUAGES, bucket_tokens_flat

MIN_TOKENS = 8  # ultra-short rows carry too little signal to judge by

CYRILLIC_LANGS = set(LANGUAGES[:21])
LATIN_LANGS = {"eng", "spa", "fra", "por", "deu"}
LID176_TO_OURS = {"en": "eng", "es": "spa", "fr": "fra", "pt": "por", "de": "deu"}

# Close-language groups: a confident prediction INSIDE the gold language's
# group is too ambiguous to judge (verified by eyeball: "bak" rows predicted
# tat split into genuine Tatar contamination — җ is Tatar-only — and genuine
# Bashkir the model misjudges — ҡ is Bashkir-only). Dropping the latter
# would remove exactly the hard boundary examples training needs, so
# within-group predictions are never dropped.
TWIN_GROUPS = [
    {"tat", "bak", "kaz", "kir", "tyv", "chv", "sah"},  # Turkic
    {"udm", "mhr", "kpv"},                              # Permic
    {"bul", "mkd", "srp"},                              # Balkan Slavic
    {"rus", "ukr", "bel"},                              # East Slavic
]

def _twin_protected(gold: str, pred: str) -> bool:
    return any(gold in g and pred in g for g in TWIN_GROUPS)


def script_counts(text: str) -> tuple[int, int]:
    lat = sum(1 for c in text if c.isascii() and c.isalpha())
    cyr = sum(1 for c in text if 0x0400 <= ord(c) <= 0x04FF)
    return lat, cyr


def load_judge(model_dir: Path):
    w = load_file(str(model_dir / "model.safetensors"))
    meta = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    P = w["P"].astype(np.float32)
    bias = w["bias"].astype(np.float32)
    return P, bias, meta["log2_d"], meta["seed"], meta["hash"]


def predict_batch(texts: list[str], P, bias, log2_d, seed, hash_id, chunk: int = 4_000) -> list[tuple[int, float]]:
    """Vectorized judge: batch featurize (bit-exact) + folded-table scoring.

    Chunked to bound memory: the token gather is [tokens, C] floats, so a
    whole 360k-row cache at once would allocate many GB. Raw-mode caches
    can hold whole documents (OSCAR/MADLAD rows reach 190k chars), so each
    chunk is also truncated to JUDGE_CHARS — a language verdict needs the
    head of the text, not all 90k of its tokens."""
    JUDGE_CHARS = 1_200
    out: list[tuple[int, float]] = []
    for t0 in range(0, len(texts), chunk):
        part = [t[:JUDGE_CHARS] for t in texts[t0 : t0 + chunk]]
        out.extend(_predict_chunk(part, P, bias, log2_d, seed, hash_id))
    return out


def _predict_chunk(texts, P, bias, log2_d, seed, hash_id) -> list[tuple[int, float]]:
    buckets, negs, offsets = bucket_tokens_flat(texts, log2_d, hash_id, seed)
    counts = np.diff(offsets)
    rows = np.repeat(np.arange(len(texts), dtype=np.int64), counts)
    signs = np.where(negs, np.float32(-1.0), np.float32(1.0))
    contrib = P[buckets] * signs[:, None]  # [N tokens, C]
    # Row sums via per-class bincount (np.add.at is slow; add.reduceat
    # mishandles empty rows). C is small (30), so 30 passes are cheap.
    sums = np.stack(
        [np.bincount(rows, weights=contrib[:, c], minlength=len(texts)) for c in range(P.shape[1])], axis=1
    )
    logits = sums / np.maximum(counts, 1)[:, None] + bias
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    top = probs.argmax(axis=1)
    conf = probs[np.arange(len(texts)), top]
    top = np.where(counts >= MIN_TOKENS, top, -1)  # too short to judge
    return [(int(t), float(c)) for t, c in zip(top, conf)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("caches", nargs="+", type=Path)
    parser.add_argument("--model", type=Path, default=Path(__file__).parent.parent / "model")
    parser.add_argument("--conf", type=float, default=0.995)
    parser.add_argument("--only-lang", default=None, help="clean only rows with this gold label")
    parser.add_argument("--script", action="store_true", help="also apply the cross-script rule (see module docstring)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    P, bias, log2_d, seed, hash_id = load_judge(args.model)
    ft = None
    if args.script:
        import fasttext

        ft = fasttext.load_model(str(Path(__file__).parent / "cache" / "lid.176.bin"))

    def in_scope(lang: str) -> bool:
        return args.only_lang is None or lang == args.only_lang

    for cache in args.caches:
        df = pl.read_ndjson(cache)
        rows = list(zip(df["lang"].to_list(), df["text"].to_list()))
        preds = predict_batch([t for _, t in rows], P, bias, log2_d, seed, hash_id)

        # Script-mismatch pass: majority-Latin text under a Cyrillic-column
        # label, judged by fastText directly (numpy-2 build).
        latin_verdicts: dict[int, str] = {}
        if ft is not None:
            cand_idx = [
                i for i, (lang, text) in enumerate(rows)
                if lang in CYRILLIC_LANGS and in_scope(lang) and script_counts(text)[0] >= max(4, script_counts(text)[1] + 1)
            ]
            for i in cand_idx:
                labels, probs = ft.predict(rows[i][1].replace("\n", " "), k=1)
                if labels:
                    label = labels[0].removeprefix("__label__")
                    if LID176_TO_OURS.get(label) and probs[0] >= 0.9:
                        latin_verdicts[i] = label

        keep_mask: list[bool] = []
        dropped: Counter[str] = Counter()
        audit: list[str] = []
        for i, ((lang, text), (top, conf)) in enumerate(zip(rows, preds)):
            wrong_lang = top >= 0 and LANGUAGES[top] != lang
            reason = None
            if wrong_lang and conf >= args.conf and in_scope(lang):
                pred = LANGUAGES[top]
                reason = None if _twin_protected(lang, pred) else pred
            elif i in latin_verdicts:
                reason = f"latin:{latin_verdicts[i]}"
            if reason:
                dropped[reason] += 1
                keep_mask.append(False)
                audit.append(json.dumps(
                    {"lang": lang, "text": text, "pred": reason, "conf": round(conf, 4)},
                    ensure_ascii=False))
            else:
                keep_mask.append(True)
        status = "would drop" if args.dry_run else "dropped"
        audit_path = cache.with_suffix(cache.suffix + ".dropped")
        audit_path.write_text("\n".join(audit) + ("\n" if audit else ""), encoding="utf-8")
        print(f"{cache.name}: {status} {len(rows) - sum(keep_mask)}/{len(rows)} -> {dict(dropped.most_common(8))}", flush=True)
        if not args.dry_run and not all(keep_mask):
            tmp = cache.with_suffix(".jsonl.tmp")
            df.filter(np.asarray(keep_mask)).write_ndjson(tmp)
            tmp.replace(cache)


if __name__ == "__main__":
    main()
