"""FineWeb-2 / FineWeb sampling for spellman training data.

Streams per-language configs (only the head of each shard), then extracts
*line-window* samples from every document: 1-5-line windows of ~20-200 chars
plus truncated prefixes, so the training distribution matches short/wild
inference text instead of whole clean documents.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator

from . import Dataset, register

# spellman language code -> (config, repo). Mongolian is Halh Mongolian
# `khk_Cyrl` in GlotLID taxonomy; English is NOT in FineWeb-2 (the original
# FineWeb is the English side of the corpus family).
FW2_SOURCES: dict[str, tuple[str, str]] = {
    "rus": ("rus_Cyrl", "HuggingFaceFW/fineweb-2"),
    "ukr": ("ukr_Cyrl", "HuggingFaceFW/fineweb-2"),
    "bel": ("bel_Cyrl", "HuggingFaceFW/fineweb-2"),
    "bul": ("bul_Cyrl", "HuggingFaceFW/fineweb-2"),
    "mkd": ("mkd_Cyrl", "HuggingFaceFW/fineweb-2"),
    "srp": ("srp_Cyrl", "HuggingFaceFW/fineweb-2"),
    "kaz": ("kaz_Cyrl", "HuggingFaceFW/fineweb-2"),
    "kir": ("kir_Cyrl", "HuggingFaceFW/fineweb-2"),
    "tgk": ("tgk_Cyrl", "HuggingFaceFW/fineweb-2"),
    "uzn": ("uzn_Cyrl", "HuggingFaceFW/fineweb-2"),
    "tat": ("tat_Cyrl", "HuggingFaceFW/fineweb-2"),
    "bak": ("bak_Cyrl", "HuggingFaceFW/fineweb-2"),
    "chv": ("chv_Cyrl", "HuggingFaceFW/fineweb-2"),
    "sah": ("sah_Cyrl", "HuggingFaceFW/fineweb-2"),
    "tyv": ("tyv_Cyrl", "HuggingFaceFW/fineweb-2"),
    "mon": ("khk_Cyrl", "HuggingFaceFW/fineweb-2"),
    "oss": ("oss_Cyrl", "HuggingFaceFW/fineweb-2"),
    "che": ("che_Cyrl", "HuggingFaceFW/fineweb-2"),
    "udm": ("udm_Cyrl", "HuggingFaceFW/fineweb-2"),
    "mhr": ("mhr_Cyrl", "HuggingFaceFW/fineweb-2"),
    "kpv": ("kpv_Cyrl", "HuggingFaceFW/fineweb-2"),
    "eng": ("sample-10BT", "HuggingFaceFW/fineweb"),
    "spa": ("spa_Latn", "HuggingFaceFW/fineweb-2"),
    "fra": ("fra_Latn", "HuggingFaceFW/fineweb-2"),
    "por": ("por_Latn", "HuggingFaceFW/fineweb-2"),
    "deu": ("deu_Latn", "HuggingFaceFW/fineweb-2"),
}

MIN_CHARS = 20
MAX_CHARS = 200


def windows_from_doc(text: str, rng: random.Random, per_doc: int) -> list[str]:
    """Short realistic samples from one document.

    Mix: truncated prefixes (start-of-doc text — titles, lead sentences) and
    random 1-5-line windows. All are single-line-flattened and length-clamped.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    out: list[str] = []
    for _ in range(per_doc):
        if rng.random() < 0.4:
            # Truncated prefix of the whole doc (or of a random line).
            src = text if rng.random() < 0.5 else rng.choice(lines)
            src = " ".join(src.split())
        else:
            # 1-5 consecutive lines.
            n_lines = rng.randint(1, 5)
            start = rng.randint(0, max(0, len(lines) - n_lines))
            src = " ".join(lines[start : start + n_lines])
        if len(src) >= MIN_CHARS:
            out.append(src[:MAX_CHARS])
    return out


@register("fineweb2")
@dataclass
class FineWeb2(Dataset):
    docs_per_lang: int = 600
    per_doc: int = 4
    seed: int = 42
    langs: list[str] | str | None = field(default=None)
    #: Languages to skip even when ``langs`` is unset — used to split one
    #: language out of the backbone for a per-language gate (tgk's ў filter).
    langs_exclude: str = ""
    #: Drop docs containing any of these characters — the script-level
    #: pollution gate. ``no_chars=ў`` de-Uzbekifies the tgk config (ў never
    #: occurs in Tajik; measured 63% Uzbek docs in CommonCrawl-derived tg).
    no_chars: str = ""

    name = "fineweb2"

    def samples(self) -> Iterator[tuple[str, str]]:
        from tqdm import tqdm

        rng = random.Random(self.seed)
        if self.langs is None:
            targets = list(FW2_SOURCES)
        elif isinstance(self.langs, str):
            targets = [l for l in self.langs.split("+") if l]
        else:
            targets = list(self.langs)
        if self.langs_exclude:
            skip = {l for l in self.langs_exclude.split("+") if l}
            targets = [l for l in targets if l not in skip]
        no_chars = self.no_chars or ""
        for code in targets:
            config, repo = FW2_SOURCES[code]
            try:
                from datasets import load_dataset

                ds = load_dataset(repo, config, streaming=True, split="train")
            except Exception as e:  # noqa: BLE001 — a missing config must not kill the run
                print(f"  {code}: SKIP ({e.__class__.__name__}: {e})")
                continue
            n = 0
            for doc in tqdm(ds.take(self.docs_per_lang), total=self.docs_per_lang, desc=code, leave=False):
                if no_chars and any(ch in doc["text"] for ch in no_chars):
                    continue
                for window in windows_from_doc(doc["text"], rng, self.per_doc):
                    yield code, window
                    n += 1
            print(f"  {code}: {n} samples")
