"""Lexical label gate for twin languages.

The orthographic gate (ortho.py) judges a row by the letters only one twin
writes. Short rows often carry none: "Хочу жити", "дякую, отримала" are
Ukrainian without a single і ї є ґ, and Twitter's `lang=ru` tag delivered
thousands of them as Russian. Words arbitrate where letters cannot.

For a pair of twins (A, B) a lexicon holds, per word, the log-odds of the
word under B's text against A's,

    llr(w) = ln( (count_B(w) + a) / (tokens_B + a·V) )
           - ln( (count_A(w) + a) / (tokens_A + a·V) )

measured on corpora whose labels are trusted (``build``: curated,
encyclopedic and manually labelled lanes, never the lane being cleaned).
A row's score is the sum of its distinct words' llr, each clipped to ±``CLIP`` so
one word cannot decide alone unless it is itself decisive, and words
outside the lexicon count 0. A row labelled A is dropped for B when its
score >= threshold (nats: 4 is ~55:1 odds), and symmetrically.

Words are what the featurizer reads as words (mentions, URLs, emails and
digit-bearing tokens are skipped, hashtags lose their ``#``), split
further into letter runs and with elongations folded ("щооо" -> "що"),
so chat spelling meets the lexicon.

The lexicon is data, shipped under ``seeds/lexgate/`` so mixes replay
byte-identically; rebuild it only on purpose:

    uv run python -m spellman_train.lexgate build --pair rus,ukr \\
        --manifest data/v15/manifest.json --clean-b 0,1,30,60 --clean-a 0,1,18,59,61,64 \\
        --extra-a cache/diverse-pool-84b77e8960.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from spellman_train.features import classify_word
from spellman_train.paths import TRAIN_DIR

LEXICON_DIR = TRAIN_DIR / "seeds" / "lexgate"

#: per-word cap on |llr|: a single decisive word ("дякую") may carry a row,
#: a merely lopsided one may not
CLIP = 6.0

_LETTERS = re.compile(r"[^\W\d_]+(?:['’ʼ-][^\W\d_]+)*")
_ELONGATED = re.compile(r"(.)\1{2,}")


def tokens(text: str) -> list[str]:
    """Lowercased letter runs of the real words of ``text``."""
    out: list[str] = []
    for word in text.split():
        word = word.removeprefix("#")
        if not word or classify_word(word) is not None:
            continue
        for tok in _LETTERS.findall(word.lower()):
            out.append(_ELONGATED.sub(r"\1", tok).replace("’", "'").replace("ʼ", "'"))
    return out


def build_lexicon(
    texts_a: list[str], texts_b: list[str], alpha: float = 0.5, min_count: int = 5, min_llr: float = 1.0
) -> dict[str, float]:
    """word -> llr (positive = B) over two trusted corpora. Kept: words seen
    ``min_count`` times on the side they favour, with |llr| >= ``min_llr``."""
    ca: Counter[str] = Counter()
    cb: Counter[str] = Counter()
    for t in texts_a:
        ca.update(tokens(t))
    for t in texts_b:
        cb.update(tokens(t))
    vocab = len(ca.keys() | cb.keys())
    na, nb = sum(ca.values()) + alpha * vocab, sum(cb.values()) + alpha * vocab
    lex: dict[str, float] = {}
    for w in ca.keys() | cb.keys():
        llr = math.log((cb[w] + alpha) / nb) - math.log((ca[w] + alpha) / na)
        if abs(llr) >= min_llr and (cb[w] if llr > 0 else ca[w]) >= min_count:
            lex[w] = round(llr, 2)
    return lex


def lexicon_path(a: str, b: str) -> Path:
    a, b = sorted((a, b))
    return LEXICON_DIR / f"{a}-{b}.tsv.gz"


def save_lexicon(a: str, b: str, lex: dict[str, float]) -> Path:
    """Write the (sorted-pair) lexicon; llr sign always means the second
    language of the sorted pair."""
    first, second = sorted((a, b))
    sign = 1.0 if (a, b) == (first, second) else -1.0
    path = lexicon_path(a, b)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(path, "wb", mtime=0) as raw:
        for w in sorted(lex):
            raw.write(f"{w}\t{sign * lex[w]:.2f}\n".encode())
    return path


@lru_cache(maxsize=None)
def load_lexicons() -> dict[str, list[tuple[str, dict[str, float]]]]:
    """label -> [(rival, word -> llr towards the rival)] for every shipped pair."""
    out: dict[str, list[tuple[str, dict[str, float]]]] = {}
    for path in sorted(LEXICON_DIR.glob("*.tsv.gz")):
        first, second = path.name.removesuffix(".tsv.gz").split("-")
        towards_second: dict[str, float] = {}
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                w, v = line.rstrip("\n").split("\t")
                towards_second[w] = float(v)
        out.setdefault(first, []).append((second, towards_second))
        out.setdefault(second, []).append((first, {w: -v for w, v in towards_second.items()}))
    return out


def score(text: str, lex: dict[str, float]) -> float:
    """Summed clipped llr of the row's distinct words (positive = the
    lexicon's rival). Distinct, because a repeated word is one piece of
    evidence: a Russian price list saying "грн" forty times is not Ukrainian."""
    return sum(max(-CLIP, min(CLIP, lex.get(tok, 0.0))) for tok in set(tokens(text)))


def contradicting_lexicon(lang: str, text: str, threshold: float) -> str | None:
    """The twin whose vocabulary ``text`` is written in instead of its label
    ``lang`` (the strongest when several qualify), or None."""
    best, best_score = None, threshold
    for rival, lex in load_lexicons().get(lang, ()):
        s = score(text, lex)
        if s >= best_score:
            best, best_score = rival, s
    return best


def own_margin(lang: str, text: str) -> float:
    """How strongly ``text``'s words are ``lang``'s own: nats against the
    closest twin (negative when a twin fits better; 0.0 when ``lang`` has no
    lexicon or the row has no known word)."""
    lexes = load_lexicons().get(lang, ())
    return min((-score(text, lex) for _, lex in lexes), default=0.0)


def _manifest_texts(manifest: Path, indices: list[int], lang: str) -> list[str]:
    import polars as pl

    from spellman_train import sources
    from spellman_train.mix import read_source

    specs = json.loads(manifest.read_text(encoding="utf-8"))["sources"]
    out: list[str] = []
    for i in indices:
        name, opts = specs[i]
        out += read_source(sources.create(name, **opts)).filter(pl.col("lang") == lang)["text"].to_list()
    return out


def _file_texts(paths: list[Path]) -> list[str]:
    return [json.loads(line)["text"] for p in paths for line in p.open(encoding="utf-8") if line.strip()]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="spellman_train.lexgate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="measure a pair's lexicon on trusted lanes")
    b.add_argument("--pair", required=True, help="A,B")
    b.add_argument("--manifest", type=Path, required=True)
    b.add_argument("--clean-a", default="", help="manifest source indices trusted for A (comma-separated)")
    b.add_argument("--clean-b", default="", help="manifest source indices trusted for B")
    b.add_argument("--extra-a", type=Path, action="append", default=[], help='jsonl of {"text"} rows, all A')
    b.add_argument("--extra-b", type=Path, action="append", default=[], help='jsonl of {"text"} rows, all B')
    b.add_argument("--min-count", type=int, default=5)
    b.add_argument("--min-llr", type=float, default=1.0)
    args = ap.parse_args(argv)
    a, bb = args.pair.split(",")
    idx = lambda s: [int(x) for x in s.split(",") if x]  # noqa: E731
    ta = _manifest_texts(args.manifest, idx(args.clean_a), a) + _file_texts(args.extra_a)
    tb = _manifest_texts(args.manifest, idx(args.clean_b), bb) + _file_texts(args.extra_b)
    lex = build_lexicon(ta, tb, min_count=args.min_count, min_llr=args.min_llr)
    path = save_lexicon(a, bb, lex)
    pos = sum(v > 0 for v in lex.values())
    print(f"{a}: {len(ta)} rows, {bb}: {len(tb)} rows -> {len(lex)} words "
          f"({pos} towards {bb}, {len(lex) - pos} towards {a}) -> {path}")


if __name__ == "__main__":
    main()
