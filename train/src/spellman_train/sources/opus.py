"""OPUS corpus source (opus.nlpl.eu), moses plain-text bitexts.

Resolves the latest version through the OPUS API and yields one side of the
aligned bitext as single-line samples:

    --source opus:corpus=translatewiki,src=ce,tgt=en,lang=che
    --source opus:corpus=wikimedia,src=ce,tgt=ru,lang=che
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Iterator

from . import CACHE_DIR, Dataset, download_with_retries, register

OPUS_API = (
    "https://opus.nlpl.eu/opusapi/?source={src}&target={tgt}"
    "&preprocessing=moses&corpus={corpus}&version=latest"
)


@register("opus")
@dataclass
class Opus(Dataset):
    corpus: str
    src: str
    tgt: str
    lang: str
    limit: int = 999_999
    min_chars: int = 20

    name = "opus"

    def samples(self) -> Iterator[tuple[str, str]]:
        raw_dir = CACHE_DIR / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        zip_path = raw_dir / f"opus-{self.corpus}-{self.src}-{self.tgt}.zip"
        if not zip_path.exists():
            api = OPUS_API.format(src=self.src, tgt=self.tgt, corpus=self.corpus)
            corpora = json.load(urllib.request.urlopen(api))["corpora"]
            url = next(c["url"] for c in corpora if c.get("url", "").endswith(".zip"))
            print(f"  downloading {self.corpus} {self.src}-{self.tgt} from OPUS...", flush=True)
            download_with_retries(url, zip_path)

        side = f"{self.corpus}.{self.src}-{self.tgt}.{self.src}"
        n = 0
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(side) as f:
                for raw in io.TextIOWrapper(f, encoding="utf-8"):
                    sent = " ".join(raw.split())
                    if len(sent) < self.min_chars:
                        continue
                    yield self.lang, sent
                    n += 1
                    if n >= self.limit:
                        break
        print(f"  opus/{self.corpus} {self.src}: {n} sentences")
