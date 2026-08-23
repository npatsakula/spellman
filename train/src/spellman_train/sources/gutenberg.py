"""Project Gutenberg public-domain literature, sentence-windowed.

The web-windowed training pools (FineWeb-2 & co.) barely contain
literary register — measured: "князь" in 5 of 16,000 rus training rows,
"поручик"/"батальон" in zero — so classical prose sweeps (detect_md over
Война и мир) collapse the rus score and minority classes win on n-gram
scraps (6% of sentences went tyv). This adapter yields 20-200-char
sentences from public-domain books (Gutendex metadata + plain-text
downloads), giving the diverse provider a literary pool for exactly the
v6 лечит-словарь treatment at register scale.

Books are fetched once per options fingerprint into the normal cache;
the Gutenberg header/footer is stripped, and each book contributes its
share of sentence windows.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass
from typing import Iterator

from . import Dataset, register

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)
_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "spellman-data/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def _fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "spellman-data/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode("utf-8", errors="replace")


def _plain_url(formats: dict[str, str]) -> str | None:
    # prefer the utf-8 plain-text rendering; fall back to any text/plain
    for key in sorted(formats):
        if key.startswith("text/plain"):
            if "charset=utf-8" in key:
                return formats[key]
    for key in sorted(formats):
        if key.startswith("text/plain"):
            return formats[key]
    return None


@register("gutenberg")
@dataclass
class Gutenberg(Dataset):
    #: Gutenberg language code(s) — 'ru', 'fr', 'de', 'en'...
    langs: str = "ru"
    #: how many plain-text books to take (gutendex popularity order)
    max_books: int = 80
    #: sentences per book kept (spread over the book; books vary 50k-1M chars)
    per_book: int = 4000
    min_chars: int = 20
    max_chars: int = 200
    seed: int = 42

    name = "gutenberg"

    def samples(self) -> Iterator[tuple[str, str]]:
        import random

        lang_map = {"ru": "rus", "fr": "fra", "de": "deu", "en": "eng"}
        lang = lang_map.get(self.langs)
        if lang is None:
            raise SystemExit(f"gutenberg: no spellman mapping for {self.langs!r} (known: {sorted(lang_map)})")

        # exclude works whose vocabulary IS an eval referee (Война и мир);
        # same-register contemporaries are the point, this exact text is not
        banned = ("война и мир", "war and peace", "guerre et paix")

        url = f"https://gutendex.com/books?languages={self.langs}&mime_type=text%2Fplain&sort=popular"
        kept_books = 0
        while url and kept_books < self.max_books:
            data = _fetch_json(url)
            for book in data.get("results", []):
                if kept_books >= self.max_books:
                    break
                title = (book.get("title") or "").lower()
                if any(b in title for b in banned):
                    continue
                src = _plain_url(book.get("formats", {}))
                if not src:
                    continue
                for attempt in range(3):
                    try:
                        text = _fetch_text(src)
                        break
                    except Exception:  # noqa: BLE001 — flaky mirrors; retry politely
                        if attempt == 2:
                            text = ""
                        else:
                            time.sleep(2.0 * (attempt + 1))
                if not text:
                    continue
                m = _START.search(text)
                if m:
                    text = text[m.end():]
                m = _END.search(text)
                if m:
                    text = text[: m.start()]
                rng = random.Random(self.seed + book.get("id", 0))
                sents = []
                for s in _SENT_SPLIT.split(text):
                    s = " ".join(s.split())
                    if self.min_chars <= len(s) <= self.max_chars:
                        sents.append(s)
                rng.shuffle(sents)
                n = 0
                for s in sents[: self.per_book]:
                    yield lang, s
                    n += 1
                kept_books += 1
                print(f"  gutenberg[{self.langs}] #{book.get('id')} {book.get('title', '?')[:48]!r}: {n} sentences", flush=True)
                time.sleep(0.4)  # be a good citizen of gutenberg.org
            url = data.get("next")
