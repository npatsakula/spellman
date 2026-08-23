"""fastText lid.176 baseline with the same granularity ladder as `assess`.

Mirrors src/bin/assess.rs fragment-for-fragment: whitespace tokens with at
least one letter, consecutive n-word windows joined by a single space,
accuracy at word / pair / triple / text level. fastText's label set does
not cover all spellman languages (no Cyrillic uzn/tyv/sah/mhr/kpv); fragments
of unsupported gold languages are excluded so the rungs are comparable to
fastText's own published methodology.

Usage (fasttext-numpy2 build — runs in the main env):
    uv run spellman-train eval-fasttext \\
        ../model/eval_test.tsv tatoeba_eval.tsv

``--device mps``/``cuda`` scores through a batched torch scorer (the
input-row gather + output matmul on the accelerator) instead of
fastText's per-row CPU ``predict`` — required for big models (GlotLID,
2k labels) on large files. Parity-checked against ``predict`` first.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from collections import Counter
from pathlib import Path

from spellman_train.paths import CACHE_DIR

MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
DEFAULT_MODEL = CACHE_DIR / "lid.176.bin"

# lid.176 labels are ISO 639-1 codes; ours are 639-3. Candidate labels per
# language, first present in the model wins (languages with no candidate are
# excluded from fastText's scored subset).
LID176_CANDIDATES = {
    "rus": ["ru"], "ukr": ["uk"], "bel": ["be"], "bul": ["bg"], "mkd": ["mk"],
    "srp": ["sr"], "kaz": ["kk"], "kir": ["ky"], "tgk": ["tg"], "uzn": ["uz"],
    "tat": ["tt"], "bak": ["ba"], "chv": ["cv"], "sah": ["sah"], "tyv": ["tyv"],
    "mon": ["mn", "khk"], "oss": ["os"], "che": ["ce"], "udm": ["udm"],
    "mhr": ["mhr"], "kpv": ["kpv"], "eng": ["en"], "spa": ["es"], "fra": ["fr"],
    "por": ["pt"], "deu": ["de"],
}

# GlotLID labels are FLORES-200 style (`rus_Cyrl`); it covers every class
# we have, including the small Cyrillic ones lid.176 lacks. Script
# variants map to the same class (gold `tat` is correct whether GlotLID
# answers `tat_Cyrl` or `tat_Latn` — it is finer-grained than our 30
# classes, and the courtesy should go to the baseline).
GLOTLID_CANDIDATES = {
    "rus": ["rus_Cyrl"], "ukr": ["ukr_Cyrl"], "bel": ["bel_Cyrl"],
    "bul": ["bul_Cyrl"], "mkd": ["mkd_Cyrl"], "srp": ["srp_Cyrl", "srp_Latn"],
    "kaz": ["kaz_Cyrl"], "kir": ["kir_Cyrl"], "tgk": ["tgk_Cyrl"],
    "uzn": ["uzn_Cyrl", "uzn_Latn"], "tat": ["tat_Cyrl", "tat_Latn", "tat_Arab"],
    "bak": ["bak_Cyrl"], "chv": ["chv_Cyrl"], "sah": ["sah_Cyrl"],
    "tyv": ["tyv_Cyrl"], "mon": ["mon_Cyrl", "khk_Cyrl"], "oss": ["oss_Cyrl"],
    "che": ["che_Cyrl"], "udm": ["udm_Cyrl"], "mhr": ["mhr_Cyrl"],
    "kpv": ["kpv_Cyrl"], "eng": ["eng_Latn"], "spa": ["spa_Latn"],
    "fra": ["fra_Latn"], "por": ["por_Latn"], "deu": ["deu_Latn"],
    "cmn": ["cmn_Hans", "cmn_Hant", "zho_Hans"], "jpn": ["jpn_Jpan"],
    "hin": ["hin_Deva"], "ara": ["ara_Arab", "arb_Arab"],
}
LABEL_SETS = {"lid176": LID176_CANDIDATES, "glotlid": GLOTLID_CANDIDATES}


def ensure_model(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {MODEL_URL} -> {path} (one-time, ~131 MB)", file=sys.stderr)
    urllib.request.urlretrieve(MODEL_URL, path)
    return path


def word_tokens(text: str) -> list[str]:
    return [t for t in text.split() if any(c.isalpha() for c in t)]


class BatchedScorer:
    """Accelerator-batched fastText scorer for softmax-loss models.

    ``predict`` for loss=softmax is exactly argmax(W_out @ mean(input
    rows)), so the heavy part — the input-row gather and the 2k-class
    output matmul — runs batched on the device while fastText's own C++
    tokenizer (``get_line`` + ``get_subwords``) produces the input row
    ids; its hashing is never reimplemented. Aborts unless the first
    `parity_n` probe rows match ``fasttext.predict`` label-for-label
    (float32 tie-flips excepted).
    """

    def __init__(self, model, device: str, batch: int = 4096, parity_n: int = 2000):
        import torch

        loss = str(getattr(model.f.getArgs(), "loss")).rsplit(".", 1)[-1]
        if loss != "softmax":
            raise SystemExit(f"batched scorer requires softmax loss (model has {loss})")
        self.model, self.device, self.batch = model, torch.device(device), batch
        self.torch = torch
        self.inp = torch.from_numpy(model.get_input_matrix()).to(self.device)
        self.out = torch.from_numpy(model.get_output_matrix()).to(self.device)
        self._parity(parity_n)

    def _ids(self, text: str) -> list[int]:
        ids: list[int] = []
        for w in self.model.get_line(text)[0]:
            ids.extend(self.model.get_subwords(w)[1])
        return ids

    def predict_labels(self, texts: list[str]) -> list[str]:
        """Top-1 fastText labels (`__label__` prefix included), batched."""
        torch = self.torch
        labels = self.model.get_labels()
        out: list[str] = []
        for t0 in range(0, len(texts), self.batch):
            part = texts[t0 : t0 + self.batch]
            rows = [self._ids(t) for t in part]
            counts = torch.tensor([len(r) for r in rows], device=self.device)
            flat = torch.tensor([i for r in rows for i in r], dtype=torch.long,
                                device=self.device)
            seg = torch.repeat_interleave(
                torch.arange(len(part), device=self.device), counts)
            vecs = torch.zeros((len(part), self.inp.shape[1]), device=self.device)
            vecs.index_add_(0, seg, self.inp.index_select(0, flat))
            vecs /= counts.clamp(min=1).to(torch.float32).unsqueeze(1)
            idx = (vecs @ self.out.T).argmax(dim=1).tolist()
            out.extend(labels[i] for i in idx)
        return out

    def _parity(self, n: int) -> None:
        probes = [
            "Привет мир, как дела", "Да", "Қазақстан Республикасының Конституциясы",
            "Hello world this is a test", "Bonjour tout le monde", "шок",
            "Ен evening duck नमस्ते 你好 مرحبا", "…", "42 42 42",
            "Özbekçe tili dünyada", "Монгол улсын үндсэн хууль", "јужни словенски",
        ]
        texts = [p for _ in range(max(1, n // len(probes))) for p in probes]
        mine = self.predict_labels(texts)
        ref = [self.model.predict(t, k=1)[0][0] for t in texts]
        agree = sum(int(a == b) for a, b in zip(mine, ref))
        if agree < len(texts) * (1 - 1e-3):
            raise SystemExit(
                f"batched scorer failed parity vs fasttext.predict ({agree}/{len(texts)})")
        print(f"parity: {agree}/{len(texts)} vs fasttext.predict", file=sys.stderr)


def fragments(rows: list[tuple[str, str]], n: int) -> list[tuple[str, str]]:
    out = []
    for lang, text in rows:
        toks = word_tokens(text)
        out.extend((lang, " ".join(toks[i : i + n])) for i in range(len(toks) - n + 1))
    return out


def ladder(predict_labels, rows: list[tuple[str, str]], n: int,
           label_to_our: dict[str, str]) -> tuple[float, int, Counter, Counter]:
    per_lang_correct: Counter = Counter()
    per_lang_total: Counter = Counter()
    correct = 0
    frags = fragments(rows, n)
    preds = predict_labels([f for _, f in frags])
    for (lang, _frag), label in zip(frags, preds):
        pred = label_to_our.get(label.removeprefix("__label__"))
        per_lang_total[lang] += 1
        if pred == lang:
            correct += 1
            per_lang_correct[lang] += 1
    total = len(frags)
    acc = correct / total if total else float("nan")
    return acc, total, per_lang_correct, per_lang_total


def populate(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("eval_files", nargs="+", type=Path)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--labels", choices=sorted(LABEL_SETS), default="lid176",
                    help="label vocabulary of the model (lid.176 = ISO 639-1, glotlid = FLORES-200)")
    ap.add_argument("--skip-ladder", action="store_true",
                    help="text-level accuracy only (for slow models / big files)")
    ap.add_argument("--device", default="cpu",
                    help="cpu = fastText's per-row predict; mps/cuda = batched "
                         "torch scorer on the accelerator (parity-checked)")


def run(args: argparse.Namespace) -> None:
    import fasttext

    candidates = LABEL_SETS[args.labels]
    model = fasttext.load_model(str(ensure_model(args.model)))

    # Which of our languages can this model express?
    labels = {l.removeprefix("__label__") for l in model.get_labels()}
    from spellman_train.features import LANGUAGES

    supported = {}
    for code in LANGUAGES:
        for alias in candidates.get(code, []):
            if alias in labels:
                supported[alias] = code
    print(f"labels: {len(labels)}; covers {len(supported)}/{len(LANGUAGES)} of our languages")
    print("unsupported (excluded from scoring):", sorted(set(LANGUAGES) - set(supported.values())))

    rows: list[tuple[str, str]] = []
    for path in args.eval_files:
        with path.open(encoding="utf-8") as f:
            for line in f:
                code, _, text = line.rstrip("\n").partition("\t")
                if code in supported.values():
                    rows.append((code, text))
    print(f"eval samples (supported subset): {len(rows)}")

    if args.device == "cpu":
        def predict_labels(texts: list[str]) -> list[str]:
            return [model.predict(t, k=1)[0][0] for t in texts]
    else:
        scorer = BatchedScorer(model, args.device)
        predict_labels = scorer.predict_labels

    if not args.skip_ladder:
        rungs = {name: ladder(predict_labels, rows, n, supported)
                 for n, name in ((1, "word"), (2, "pair"), (3, "triple"))}
        for name, (acc, total, _, _) in rungs.items():
            print(f"{name:>6}: {acc:.2%}  (n = {total})")
    else:
        rungs = {}

    # Text rung: the whole sample (not word windows), plus length buckets
    # matching `assess` / lid-bench (chars: <=20 / 21-100 / >100).
    preds = predict_labels([t for _, t in rows])
    correct = total = 0
    per_correct: Counter = Counter()
    per_total: Counter = Counter()
    bucket = {"≤20": [0, 0], "21-100": [0, 0], ">100": [0, 0]}
    for (lang, text), label in zip(rows, preds):
        pred = supported.get(label.removeprefix("__label__"))
        total += 1
        per_total[lang] += 1
        n = len(text)
        b = bucket["≤20"] if n <= 20 else bucket["21-100"] if n <= 100 else bucket[">100"]
        b[1] += 1
        if pred == lang:
            correct += 1
            per_correct[lang] += 1
            b[0] += 1
    rungs["text"] = (correct / total if total else float("nan"), total, per_correct, per_total)
    print(f"  text: {rungs['text'][0]:.2%}  (n = {rungs['text'][1]})")
    for name, (ok, n) in bucket.items():
        acc = f"{ok / n:.2%}" if n else "—"
        print(f"    {name:>6}: {acc}  (n = {n})")

    if not args.skip_ladder:
        print("\nper-language (word acc / text acc, worst word first):")
        _, _, wc, wt = rungs["word"]
        _, _, tc, tt = rungs["text"]
        order = sorted(wt, key=lambda l: wc[l] / wt[l])
        for lang in order[:15]:
            print(f"  {lang:>4}  {wc[lang] / wt[lang]:.2%}  {tc[lang] / tt[lang]:.2%}  (n = {wt[lang]})")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="spellman-train eval-fasttext", description=__doc__)
    populate(ap)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
