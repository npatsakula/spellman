"""Wild-register UGC corpora held as raw files (no usable HF parquet view).

Three purpose-built adapters for the wild-social-media lane (the
rusentitweet analogs found in the 2026-08 dataset sweep; see
train/WILD_UGC_CANDIDATES.md). Each fetches its origin artifact into
``cache/raw/<slug>/`` on first use and reuses whatever is already there —
including a validation-agent download — so a warmed cache never re-fetches.

    --source ukr_tweets:limit=400000            # 1.85M raw Ukrainian tweets
    --source mn_social                          # 10k Mongolian social comments
    --source kazsandra                          # 175k Kazakh reviews (deduped)
"""

from __future__ import annotations

import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from . import CACHE_DIR, Dataset, cyrillic_ratio, download_with_retries, register

RAW = CACHE_DIR / "raw"

UKR_TAR_URL = "https://github.com/saganoren/ukr-twi-corpus/raw/master/corpus.tar.xz"
MN_CSV_URL = (
    "https://raw.githubusercontent.com/ganaxy/diploma/master/"
    "sample%20scores/relabeled_v7_corrected.csv"
)
KAZSANDRA_REPO = "https://github.com/IS2AI/KazSAnDRA.git"
# The resampled *_ros/*_rus zips duplicate rows; the canonical set is
# ib-train + valid + test of both tasks, deduplicated on custom_id.
KAZSANDRA_ZIPS = [
    "01_pc_train_ib.zip", "04_pc_valid.zip", "05_pc_test.zip",
    "06_sc_train_ib.zip", "09_sc_valid.zip", "10_sc_test.zip",
]


@register("ukr_tweets")
@dataclass
class UkrTweets(Dataset):
    """saganoren/ukr-twi-corpus: 1.85M raw Ukrainian tweets (2018-19).

    Pre-filtered to Twitter's own ``lang == "uk"`` self-label (84% of rows)
    plus a Cyrillic-ratio gate — the corpus still carries a real surzhyk/
    Russian tail, which the pipeline's LID hygiene is for. The CSV embeds
    newlines inside quoted tweets, so it must be parsed with a real CSV
    reader (pandas), never line-split.
    """

    lang: str = "ukr"
    limit: int = 400_000
    min_chars: int = 20
    cyr: float = 0.5

    name = "ukr_tweets"

    def samples(self) -> Iterator[tuple[str, str]]:
        import pandas as pd

        slug = RAW / "ukr-twi-corpus"
        slug.mkdir(parents=True, exist_ok=True)
        csv = slug / "corpus.csv"
        if not csv.exists():
            tgz = slug / "corpus.tar.xz"
            if not tgz.exists():
                print(f"  downloading ukr-twi-corpus ({UKR_TAR_URL})...", flush=True)
                download_with_retries(UKR_TAR_URL, tgz)
            print("  extracting corpus.tar.xz...", flush=True)
            with tarfile.open(tgz, "r:xz") as tar:
                member = next(m for m in tar.getmembers() if m.name.endswith("corpus.csv"))
                tar.extract(member, slug, filter="data")  # noqa: S202 — trusted source
                csv = Path(slug / member.name)

        n = 0
        for chunk in pd.read_csv(csv, usecols=["text", "lang"], chunksize=200_000):
            for t, l in zip(chunk["text"].tolist(), chunk["lang"].tolist()):
                if l != "uk" or not isinstance(t, str):
                    continue
                t = " ".join(t.split())
                if len(t) < self.min_chars or cyrillic_ratio(t) < self.cyr:
                    continue
                yield self.lang, t
                n += 1
                if n >= self.limit:
                    break
            if n >= self.limit:
                break
        print(f"  ukr_tweets: {n} tweets")


@register("mn_social")
@dataclass
class MnSocial(Dataset):
    """ganaxy/diploma: 10k raw Mongolian news/FB/YouTube comments
    (``text_raw`` column, hand-annotated for a thesis — labels ignored)."""

    lang: str = "mon"
    limit: int = 999_999
    min_chars: int = 20

    name = "mn_social"

    def samples(self) -> Iterator[tuple[str, str]]:
        import pandas as pd

        slug = RAW / "mn-social-comments"
        slug.mkdir(parents=True, exist_ok=True)
        csv = slug / "relabeled_v7_corrected.csv"
        if not csv.exists():
            print(f"  downloading mn-social-comments ({MN_CSV_URL})...", flush=True)
            download_with_retries(MN_CSV_URL, csv)

        n = 0
        df = pd.read_csv(csv)
        for t in df["text_raw"].tolist():
            if not isinstance(t, str):
                continue
            t = " ".join(t.split())
            if len(t) < self.min_chars:
                continue
            yield self.lang, t
            n += 1
            if n >= self.limit:
                break
        print(f"  mn_social: {n} comments")


@register("kazsandra")
@dataclass
class KazSandra(Dataset):
    """IS2AI/KazSAnDRA: ~175k unique Kazakh app-store/market reviews.

    Reads only the canonical zips (imbalanced-train + valid + test of both
    tasks) and dedups on ``custom_id`` — the ``*_ros``/``*_rus`` resampled
    zips duplicate rows and are skipped on purpose. Reviews are colloquial
    but cleaned (no URLs/@mentions)."""

    lang: str = "kaz"
    limit: int = 999_999
    min_chars: int = 20
    cyr: float = 0.5

    name = "kazsandra"

    def samples(self) -> Iterator[tuple[str, str]]:
        import pandas as pd

        slug = RAW / "kazsandra"
        if not slug.exists():
            import subprocess

            print(f"  cloning KazSAnDRA ({KAZSANDRA_REPO})...", flush=True)
            subprocess.run(["git", "clone", "--depth", "1", KAZSANDRA_REPO, str(slug)], check=True)

        n = 0
        seen_ids: set[str] = set()
        for name in KAZSANDRA_ZIPS:
            path = slug / "dataset" / name
            if not path.exists():
                raise SystemExit(f"kazsandra: missing {path}")
            with zipfile.ZipFile(path) as zf:
                inner = zf.namelist()[0]
                with zf.open(inner) as f:
                    df = pd.read_csv(f)
            for cid, t in zip(df["custom_id"].tolist(), df["text"].tolist()):
                if cid in seen_ids or not isinstance(t, str):
                    continue
                seen_ids.add(cid)
                t = " ".join(t.split())
                if len(t) < self.min_chars or cyrillic_ratio(t) < self.cyr:
                    continue
                yield self.lang, t
                n += 1
                if n >= self.limit:
                    break
            if n >= self.limit:
                break
        print(f"  kazsandra: {n} reviews (deduped)")
