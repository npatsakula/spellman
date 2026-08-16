"""Leipzig Corpora Collection source (wortschatz-leipzig.de).

Works for any Leipzig corpus name, e.g. the CURL community crawls of
under-resourced languages:

    --source leipzig:corpus=che_community_2023,lang=che
    --source leipzig:corpus=che_community_2017,lang=che

Sentences are yielded as-is (single-line samples ≥ min_chars); the tarball is
downloaded once into cache/raw/.
"""

from __future__ import annotations

import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from . import CACHE_DIR, Dataset, register

LEIPZIG_URL = "https://downloads.wortschatz.leipzig.de/corpora/{corpus}.tar.gz"


@register("leipzig")
@dataclass
class Leipzig(Dataset):
    corpus: str
    lang: str
    limit: int = 999_999
    min_chars: int = 20

    name = "leipzig"

    def samples(self) -> Iterator[tuple[str, str]]:
        raw_dir = CACHE_DIR / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        tgz = raw_dir / f"{self.corpus}.tar.gz"
        if not tgz.exists():
            print(f"  downloading {self.corpus} from Leipzig...", flush=True)
            urllib.request.urlretrieve(LEIPZIG_URL.format(corpus=self.corpus), tgz)

        n = 0
        with tarfile.open(tgz, "r:gz") as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith("-sentences.txt"))
            with tar.extractfile(member) as f:
                assert f is not None
                for line in f:
                    # Leipzig sentence files are id<TAB>sentence TSV.
                    _, _, sent = line.decode("utf-8").rstrip("\n").partition("\t")
                    sent = " ".join(sent.split())
                    if len(sent) < self.min_chars:
                        continue
                    yield self.lang, sent
                    n += 1
                    if n >= self.limit:
                        break
        print(f"  leipzig/{self.corpus}: {n} sentences")
