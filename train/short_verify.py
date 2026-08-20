"""Consensus verification for the short-text lane (3–19 char rows).

The model-judge hygiene refuses rows under 8 tokens — too little signal
to trust a single model — so short rows get a *committee* instead: each
row's label must be confirmed by >= --min-agree of three judges:

  1. the current spellman model (folded-table scorer, no token minimum),
  2. GlotLID v3 (FLORES labels, script-variant aliases mapped),
  3. fastText lid.176 (ISO-639-1 labels mapped; abstains on languages
     its label set lacks).

Rows failing consensus move to <cache>.dropped (jsonl, with all three
votes) — never silently discarded. Caches are rewritten in place, so the
mixer replays the cleaned rows unchanged.

For EVAL referees use --min-agree 2 --no-spellman: the file a model is
scored on must not have been filtered through that model's own opinions.

Usage:
    uv run python short_verify.py cache/ukr_tweets-<hash>.jsonl [...]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from eval_fasttext import GLOTLID_CANDIDATES, LID176_CANDIDATES
from hygiene import load_judge
from spellman_features import LANGUAGES, bucket_tokens_flat

_HANDLE = re.compile(r"@\w+")
_URL = re.compile(r"https?://\S+|\bwww\.\S+")
_RT = re.compile(r"^RT\b[:\s]*")


def judge_text(text: str) -> str:
    """Strip the tokens the external judges key on but the language doesn't
    live in: @handles (a Latin handle drags GlotLID to *_Latn labels), URLs,
    RT prefixes. spellman's featurizer canonicalizes these natively."""
    return " ".join(_URL.sub(" ", _HANDLE.sub(" ", _RT.sub(" ", text))).split())


def spellman_votes(
    texts: list[str], P: np.ndarray, bias: np.ndarray, log2_d: int, seed: int, hash_id: str, chunk: int = 8_000
) -> list[int]:
    """Top-1 class per text with NO token minimum (unlike hygiene's judge,
    which abstains under 8 tokens — the whole point here). Chunked and
    length-capped for the same memory reasons as hygiene.predict_batch."""
    out: list[int] = []
    for t0 in range(0, len(texts), chunk):
        part = [t[:400] for t in texts[t0 : t0 + chunk]]
        buckets, negs, offsets = bucket_tokens_flat(part, log2_d, hash_id, seed)
        counts = np.diff(offsets)
        rows = np.repeat(np.arange(len(part), dtype=np.int64), counts)
        signs = np.where(negs, np.float32(-1.0), np.float32(1.0))
        contrib = P[buckets] * signs[:, None]
        sums = np.stack(
            [np.bincount(rows, weights=contrib[:, c], minlength=len(part)) for c in range(P.shape[1])], axis=1
        )
        logits = sums / np.maximum(counts, 1)[:, None] + bias
        top = logits.argmax(axis=1)
        top = np.where(counts > 0, top, -1)  # no letters at all -> abstain
        out.extend(int(t) for t in top)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("caches", nargs="+", type=Path)
    parser.add_argument("--model", type=Path, default=Path(__file__).parent.parent / "model")
    parser.add_argument("--min-agree", type=int, default=2)
    parser.add_argument("--no-spellman", action="store_true",
                        help="external judges only (for eval referees)")
    parser.add_argument("--glotlid", type=Path, default=Path(__file__).parent / "cache" / "raw" / "model.bin")
    parser.add_argument("--lid176", type=Path, default=Path(__file__).parent / "cache" / "lid.176.bin")
    args = parser.parse_args()

    import fasttext

    glot = fasttext.load_model(str(args.glotlid))
    lid = fasttext.load_model(str(args.lid176))
    P, bias, log2_d, seed, hash_id = load_judge(args.model)

    for cache in args.caches:
        rows = [json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()]
        texts = [r["text"] for r in rows]
        langs = [r["lang"] for r in rows]

        sv = [-1] * len(rows)
        if not args.no_spellman:
            sv = spellman_votes(texts, P, bias, log2_d, seed, hash_id)

        kept: list[str] = []
        audit_lines: list[str] = []
        dropped_by_lang: Counter[str] = Counter()
        for i, (lang, text) in enumerate(zip(langs, texts)):
            t = judge_text(text).replace("\n", " ")
            (gl), = glot.predict(t)[0]
            (ll), = lid.predict(t)[0]
            gl_ok = gl.removeprefix("__label__") in GLOTLID_CANDIDATES.get(lang, [])
            lid_ok = ll.removeprefix("__label__") in LID176_CANDIDATES.get(lang, [])
            spell_ok = sv[i] >= 0 and LANGUAGES[sv[i]] == lang
            votes = [gl_ok, lid_ok] + ([spell_ok] if not args.no_spellman else [])
            if sum(votes) >= args.min_agree:
                kept.append(json.dumps({"lang": lang, "text": text}, ensure_ascii=False))
            else:
                dropped_by_lang[lang] += 1
                audit_lines.append(json.dumps(
                    {"lang": lang, "text": text, "glotlid": gl, "lid176": ll,
                     "spellman": LANGUAGES[sv[i]] if sv[i] >= 0 else None},
                    ensure_ascii=False))
        cache.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        audit = cache.with_suffix(cache.suffix + ".dropped")
        audit.write_text("\n".join(audit_lines) + ("\n" if audit_lines else ""), encoding="utf-8")
        print(f"{cache.name}: kept {len(kept)}/{len(rows)} (min-agree {args.min_agree}"
              f"{', external-only' if args.no_spellman else ''}); dropped {dict(dropped_by_lang)}")


if __name__ == "__main__":
    main()
