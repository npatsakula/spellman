"""fastText lid.176 baseline with the same granularity ladder as `assess`.

Mirrors src/bin/assess.rs fragment-for-fragment: whitespace tokens with at
least one letter, consecutive n-word windows joined by a single space,
accuracy at word / pair / triple / text level. fastText's label set does not
cover all spellman languages (no Cyrillic uzn/tyv/sah/mhr/kpv); fragments of
unsupported gold languages are excluded so the rungs are comparable to
fastText's own published methodology.

Usage (fasttext-numpy2 build — runs in the main env):
    uv run python eval_fasttext.py \
        ../model/eval_test.tsv tatoeba_eval.tsv
"""

from __future__ import annotations

import argparse
import functools
import sys
import urllib.request
from collections import Counter
from pathlib import Path

MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
DEFAULT_MODEL = Path(__file__).parent / "cache" / "lid.176.bin"

# lid.176 labels are ISO 639-1 codes; ours are 639-3. Candidate labels per
# language, first present in the model wins (languages with no candidate are
# excluded from fastText's scored subset).
LABEL_CANDIDATES = {
    "rus": ["ru"], "ukr": ["uk"], "bel": ["be"], "bul": ["bg"], "mkd": ["mk"],
    "srp": ["sr"], "kaz": ["kk"], "kir": ["ky"], "tgk": ["tg"], "uzn": ["uz"],
    "tat": ["tt"], "bak": ["ba"], "chv": ["cv"], "sah": ["sah"], "tyv": ["tyv"],
    "mon": ["mn", "khk"], "oss": ["os"], "che": ["ce"], "udm": ["udm"],
    "mhr": ["mhr"], "kpv": ["kpv"], "eng": ["en"], "spa": ["es"], "fra": ["fr"],
    "por": ["pt"], "deu": ["de"],
}


def ensure_model(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {MODEL_URL} -> {path} (one-time, ~131 MB)", file=sys.stderr)
    urllib.request.urlretrieve(MODEL_URL, path)
    return path


def word_tokens(text: str) -> list[str]:
    return [t for t in text.split() if any(c.isalpha() for c in t)]


def fragments(rows: list[tuple[str, str]], n: int) -> list[tuple[str, str]]:
    out = []
    for lang, text in rows:
        toks = word_tokens(text)
        out.extend((lang, " ".join(toks[i : i + n])) for i in range(len(toks) - n + 1))
    return out


def ladder(model, rows: list[tuple[str, str]], n: int, label_to_our: dict[str, str]) -> tuple[float, int, Counter, Counter]:
    per_lang_correct: Counter = Counter()
    per_lang_total: Counter = Counter()
    correct = 0
    total = 0
    for lang, frag in fragments(rows, n):
        labels, _probs = model.predict(frag, k=1)
        pred = label_to_our.get(labels[0].removeprefix("__label__")) if labels else None
        total += 1
        per_lang_total[lang] += 1
        if pred == lang:
            correct += 1
            per_lang_correct[lang] += 1
    acc = correct / total if total else float("nan")
    return acc, total, per_lang_correct, per_lang_total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_files", nargs="+", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    import fasttext

    model = fasttext.load_model(str(ensure_model(args.model)))

    # Which of our languages can this model express?
    labels = {l.removeprefix("__label__") for l in model.get_labels()}
    from spellman_features import LANGUAGES

    supported = {}
    for code in LANGUAGES:
        for alias in LABEL_CANDIDATES.get(code, []):
            if alias in labels:
                supported[alias] = code
                break
    print(f"fastText labels: {len(labels)}; covers {len(supported)}/{len(LANGUAGES)} of our languages")
    print("unsupported (excluded from scoring):", sorted(set(LANGUAGES) - set(supported.values())))

    rows: list[tuple[str, str]] = []
    for path in args.eval_files:
        with path.open(encoding="utf-8") as f:
            for line in f:
                code, _, text = line.rstrip("\n").partition("\t")
                if code in supported.values():
                    rows.append((code, text))
    print(f"eval samples (supported subset): {len(rows)}")

    ladder_n = functools.partial(ladder, model, rows, label_to_our=supported)
    rungs = {name: ladder_n(n) for n, name in ((1, "word"), (2, "pair"), (3, "triple"))}
    for name, (acc, total, _, _) in rungs.items():
        print(f"{name:>6}: {acc:.2%}  (n = {total})")

    # Text rung: the whole sample (not word windows).
    correct = total = 0
    per_correct: Counter = Counter()
    per_total: Counter = Counter()
    for lang, text in rows:
        labels, _probs = model.predict(text, k=1)
        pred = supported.get(labels[0].removeprefix("__label__")) if labels else None
        total += 1
        per_total[lang] += 1
        if pred == lang:
            correct += 1
            per_correct[lang] += 1
    rungs["text"] = (correct / total if total else float("nan"), total, per_correct, per_total)
    print(f"  text: {rungs['text'][0]:.2%}  (n = {rungs['text'][1]})")

    print("\nper-language (word acc / text acc, worst word first):")
    _, _, wc, wt = rungs["word"]
    _, _, tc, tt = rungs["text"]
    order = sorted(wt, key=lambda l: wc[l] / wt[l])
    for lang in order[:15]:
        print(f"  {lang:>4}  {wc[lang] / wt[lang]:.2%}  {tc[lang] / tt[lang]:.2%}  (n = {wt[lang]})")


if __name__ == "__main__":
    main()
