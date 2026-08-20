"""Local CSV/TSV file source (one text column, single language).

For locally-held corpora that have no HF home, e.g. the rusentitweet train
split:

    --source csv:path=rusentitweet_train.csv,column=text,lang=rus
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from . import Dataset, register


@register("csv")
@dataclass
class LocalCsv(Dataset):
    path: Path
    column: str
    lang: str
    min_chars: int = 20
    max_chars: int = 0

    name = "csv"

    def __post_init__(self) -> None:
        # CLI source options arrive as plain strings.
        if isinstance(self.path, str):
            self.path = Path(self.path)

    def samples(self) -> Iterator[tuple[str, str]]:
        csv.field_size_limit(sys.maxsize)
        n = 0
        with self.path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                text = " ".join((row.get(self.column) or "").split())
                if len(text) >= self.min_chars and not (self.max_chars and len(text) > self.max_chars):
                    yield self.lang, text
                    n += 1
        print(f"  csv/{self.path.name} [{self.column}] -> {self.lang}: {n} rows")


@register("jsonl")
@dataclass
class LocalJsonl(Dataset):
    """A pre-labeled cache-style file ({"lang","text"} rows) — the output
    side of offline tools like hard_negatives.py that carry their own labels:

        --source jsonl:path=cache/hard_negatives.jsonl
    """

    path: Path
    min_chars: int = 20

    name = "jsonl"

    def __post_init__(self) -> None:
        if isinstance(self.path, str):
            self.path = Path(self.path)

    def samples(self) -> Iterator[tuple[str, str]]:
        n = 0
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if len(row["text"]) >= self.min_chars:
                    yield row["lang"], row["text"]
                    n += 1
        print(f"  jsonl/{self.path.name}: {n} rows")
