"""Tatoeba source: neutral eval set + training remainder.

Two artifacts come out of the Tatoeba sentence dump:

  * the **eval set** (``tatoeba_eval.tsv``, code<TAB>text) — the frozen
    out-of-domain benchmark also used for baseline comparisons. Selection is
    seeded and must stay byte-identical across regenerations, so this file is
    only (re)written when it does not exist yet;
  * the **training remainder** — the rest of each language's eligible
    sentences (eval selection excluded verbatim), script-filtered to Cyrillic
    for Cyrillic-column languages, capped per language.

Both derivations live here so the eval/train split of this source can never
drift apart.
"""

from __future__ import annotations

import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from spellman_train.paths import TRAIN_DIR

from . import Dataset, register

# spellman code -> Tatoeba code (ISO 639-3; identical except where noted).
TATOEBA_CODES = {
    "rus": "rus", "ukr": "ukr", "bel": "bel", "bul": "bul", "mkd": "mkd",
    "srp": "srp", "kaz": "kaz", "kir": "kir", "tgk": "tgk", "uzn": "uzn",
    "tat": "tat", "bak": "bak", "chv": "chv", "sah": "sah", "tyv": "tyv",
    "mon": "mon", "oss": "oss", "che": "che", "udm": "udm", "mhr": "mhr",
    "kpv": "kpv", "eng": "eng", "spa": "spa", "fra": "fra", "por": "por",
    "deu": "deu",
}

# Languages whose Tatoeba corpora are script-mixed: keep only sentences whose
# letters are majority-Cyrillic (our coverage is the _Cyrl variety).
SCRIPT_FILTER = {"srp", "uzn"}

# Aliases Tatoeba may use instead of the ISO code.
ALIASES = {"oss": ["oss", "os"], "mon": ["mon", "khk"], "mhr": ["mhr", "mari"]}

CYRILLIC_GROUP = set(TATOEBA_CODES) - {"eng", "spa", "fra", "por", "deu"}


def is_cyrillic(text: str) -> bool:
    cyr = lat = 0
    for c in text:
        cp = ord(c)
        if 0x0400 <= cp <= 0x052F:
            cyr += 1
        elif (0x41 <= cp <= 0x5A) or (0x61 <= cp <= 0x7A):
            lat += 1
    return cyr > 0 and cyr > lat


@register("tatoeba")
@dataclass
class Tatoeba(Dataset):
    sentences: Path = TRAIN_DIR / "tatoeba" / "sentences.csv"
    eval_out: Path = TRAIN_DIR / "tatoeba_eval.tsv"
    max_per_lang: int = 2000
    min_chars: int = 20
    train_per_lang: int = 8000
    seed: int = 42

    name = "tatoeba"

    #: Hub re-wrap of the export (the same sentences.csv as zstd parquet,
    #: CC BY 2.0 FR) — materialized into `sentences` when the local dump is
    #: absent, because downloads.tatoeba.org throttles to ~1 KB/s per
    #: connection. A class attribute, not a field: the cache fingerprint
    #: (and so every warm tatoeba cache) is unchanged.
    HUB_REPO = "vpermilp/tatoeba-sentences"

    def _ensure_dump(self) -> None:
        if self.sentences.exists():
            return
        import polars as pl
        from huggingface_hub import hf_hub_download

        print(f"  tatoeba: {self.sentences} missing -> materializing from {self.HUB_REPO}", flush=True)
        parquet = hf_hub_download(self.HUB_REPO, "data/train-00000-of-00001.parquet", repo_type="dataset")
        self.sentences.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.sentences.with_suffix(".csv.part")
        pl.read_parquet(parquet).write_csv(tmp, separator="\t", include_header=False, quote_style="never")
        tmp.rename(self.sentences)

    def _eligible_by_lang(self) -> dict[str, list[str]]:
        self._ensure_dump()
        csv.field_size_limit(sys.maxsize)
        want: dict[str, set[str]] = {
            our: {tb, *ALIASES.get(our, [])} for our, tb in TATOEBA_CODES.items()
        }
        reverse = {tb: our for our, tbs in want.items() for tb in tbs}
        by_lang: dict[str, list[str]] = {code: [] for code in TATOEBA_CODES}
        with self.sentences.open(encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="\t"):
                if len(row) < 3:
                    continue
                lang, text = row[1], row[2]
                our = reverse.get(lang)
                if our is None or len(text) < self.min_chars:
                    continue
                if our in SCRIPT_FILTER and not is_cyrillic(text):
                    continue
                by_lang[our].append(text)
        return by_lang

    def _eval_selection(self, by_lang: dict[str, list[str]]) -> dict[str, list[str]]:
        """The frozen eval pick. Writes eval_out only when missing so the
        benchmark file is stable once created; the selection itself is a pure
        function of (sentences dump, seed), so recomputation matches."""
        rng = random.Random(self.seed)
        selected: dict[str, list[str]] = {}
        if not self.eval_out.exists():
            with self.eval_out.open("w", encoding="utf-8") as f:
                for code in sorted(by_lang):
                    sents = by_lang[code]
                    rng.shuffle(sents)
                    for text in sents[: self.max_per_lang]:
                        text = " ".join(text.split())
                        f.write(f"{code}\t{text}\n")
                    selected[code] = [" ".join(t.split()) for t in sents[: self.max_per_lang]]
        else:
            # Existing benchmark file: read the selection back instead of
            # rewriting it (identical by construction, and robust to dump updates).
            with self.eval_out.open(encoding="utf-8") as f:
                for line in f:
                    code, _, text = line.rstrip("\n").partition("\t")
                    selected.setdefault(code, []).append(text)
        return selected

    def samples(self) -> Iterator[tuple[str, str]]:
        by_lang = self._eligible_by_lang()
        selected = self._eval_selection(by_lang)
        excluded = {(code, t) for code, texts in selected.items() for t in texts}
        for code in sorted(by_lang):
            rest = list(by_lang[code])
            # Per-language reseed (matches the frozen tatoeba_train emission:
            # the original prep script reshuffled each language with a fresh
            # Random(seed), not one shared stream).
            random.Random(self.seed).shuffle(rest)
            taken = 0
            for text in rest:
                if taken >= self.train_per_lang:
                    break
                text = " ".join(text.split())
                if (code, text) in excluded:
                    continue
                if code in CYRILLIC_GROUP and not is_cyrillic(text):
                    continue
                yield code, text
                taken += 1
