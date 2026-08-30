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

    #: Hub re-wraps of the bitexts the recipes use (one parquet, one column
    #: per side, the OPUS README/LICENSE alongside) — tried when the moses
    #: zip is not in cache/raw, BEFORE the OPUS API: opus.nlpl.eu and
    #: object.pouta.csc.fi are unreachable from some build networks (v13a
    #: build, 2026-08-30). Class attribute: the cache fingerprint is unchanged.
    HUB_REPOS = {
        ("translatewiki", "ce", "en"): "vpermilp/opus-translatewiki-ce-en",
        ("GoURMET", "ky", "ru"): "vpermilp/opus-gourmet-ky-ru",
    }

    def _source_lines(self) -> Iterator[str]:
        """Raw source-side lines: cache/raw zip -> Hub re-wrap -> OPUS API."""
        raw_dir = CACHE_DIR / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        zip_path = raw_dir / f"opus-{self.corpus}-{self.src}-{self.tgt}.zip"
        repo = self.HUB_REPOS.get((self.corpus, self.src, self.tgt))
        if not zip_path.exists() and repo:
            import polars as pl
            from huggingface_hub import hf_hub_download

            print(f"  opus/{self.corpus} {self.src}-{self.tgt}: reading the Hub re-wrap {repo}", flush=True)
            parquet = hf_hub_download(repo, "data/train-00000-of-00001.parquet", repo_type="dataset")
            yield from pl.read_parquet(parquet, columns=[self.src])[self.src].to_list()
            return
        if not zip_path.exists():
            api = OPUS_API.format(src=self.src, tgt=self.tgt, corpus=self.corpus)
            corpora = json.load(urllib.request.urlopen(api))["corpora"]
            url = next(c["url"] for c in corpora if c.get("url", "").endswith(".zip"))
            print(f"  downloading {self.corpus} {self.src}-{self.tgt} from OPUS...", flush=True)
            download_with_retries(url, zip_path)
        side = f"{self.corpus}.{self.src}-{self.tgt}.{self.src}"
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(side) as f:
                yield from io.TextIOWrapper(f, encoding="utf-8")

    def samples(self) -> Iterator[tuple[str, str]]:
        n = 0
        for raw in self._source_lines():
            sent = " ".join(raw.split())
            if len(sent) < self.min_chars:
                continue
            yield self.lang, sent
            n += 1
            if n >= self.limit:
                break
        print(f"  opus/{self.corpus} {self.src}: {n} sentences")
