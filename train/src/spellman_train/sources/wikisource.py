"""Wikimedia Wikisource dumps, sentence-windowed — the literary register.

The web-windowed training pools barely contain literary vocabulary
(measured on the shipped v8e: "князь" in 5 of 16,000 rus training rows,
"поручик"/"батальон" in zero), so classical-prose document sweeps
collapse the rus score and minority classes win on n-gram scraps (6% of
Война и ми�р sentences classified tyv). ru.wikisource is public-domain
classics at scale, shapeable through the diverse provider's
lemma-coverage selection — the v6 "лечит словарь" treatment at register
scale.

`banned` is a regex over page titles: works whose vocabulary IS an eval
referee (Война и мир) are excluded — same-register contemporaries are
the point, the eval text itself is not. Books whose *content* the eval
was derived from must stay out of the pool.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Iterator

from . import Dataset, register

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
#: wikisource boilerplate that survives plain-text extraction (the \| is
#: a literal table-cell pipe — NOT an empty regex alternative, which
#: would match every line and drop the whole pool; measured bug)
_SKIP = re.compile(r"^(навигация|поиск|\||редактировать|источник|←|→)")


@register("wikisource")
@dataclass
class Wikisource(Dataset):
    #: dump config, e.g. 20231201.ru (see the wikimedia/wikisource repo)
    config: str = "20231201.ru"
    lang: str = "rus"
    #: pages to stream (docs are small; ~150k pages ≈ a rich pool)
    docs: int = 150_000
    #: sentences kept per page (a fat novel must not dominate the pool)
    per_doc: int = 40
    #: title regex of works excluded as eval referees
    banned: str = "война и мир|war and peace"
    min_chars: int = 20
    max_chars: int = 200
    seed: int = 42

    name = "wikisource"

    def samples(self) -> Iterator[tuple[str, str]]:
        from datasets import load_dataset
        from tqdm import tqdm

        banned_re = re.compile(self.banned, re.I)
        ds = load_dataset("wikimedia/wikisource", self.config, streaming=True, split="train")
        rng = random.Random(self.seed)
        n = 0
        pages = 0
        for page in tqdm(ds.take(self.docs), total=self.docs, desc=f"wikisource[{self.config}]", leave=False):
            pages += 1
            title = page.get("title") or ""
            if banned_re.search(title):
                continue
            text = page.get("text") or ""
            sents = []
            for s in _SENT_SPLIT.split(text):
                s = " ".join(s.split())
                if self.min_chars <= len(s) <= self.max_chars and not _SKIP.match(s.lower()):
                    sents.append(s)
            if not sents:
                continue
            rng.shuffle(sents)
            for s in sents[: self.per_doc]:
                yield self.lang, s
                n += 1
        print(f"  wikisource[{self.config}]: {n:,} sentences from {pages:,} pages", flush=True)
