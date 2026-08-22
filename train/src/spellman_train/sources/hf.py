"""Generic HuggingFace Hub corpus source: any repo/config, one language.

The escape hatch that makes new datasets a CLI knob instead of new code —
point it at a dataset with a text column and it joins the mix with the same
line-window sampling as the FineWeb-2 source:

    --source hf:repo=cis-lmu/Glot500,config=tat_Cyrl,lang=tat,docs=3000
    --source hf:repo=AigizK/tatar-russian-parallel-corpora,column=tat,lang=tat

For multi-language datasets, declare one source per language/config.

**Raw mode** (``raw=True``) is for tweet/post-shaped corpora where
line-window chopping is wrong — each row is one already-short text and is
yielded as-is (``min_chars``/``max_chars`` clamps only). Raw mode also
unlocks the row gates:

    # telegram posts, keep only Cyrillic-dominant rows (Uzbek is mixed-script)
    --source hf:repo=tahrirchi/uz-crawl,files=data/telegram_blogs*,lang=uzn,raw=True,cyr=0.6
    # 829k German-politician tweets, self-labeled: keep language=="de"
    --source hf:repo=NLP-UniBW/tweets_about_german_politicians_jan_feb_2025,where=language=de,lang=deu,raw=True
    # labeled ua/ru/surzhyk corpus: slice the gold-Ukrainian rows
    --source hf:repo=YShynkarov/COSMUS,column=document_content,where=language_manual=ukrainian,lang=ukr,raw=True

``docs`` caps the number of rows *scanned* (0 = everything); gates filter
within that budget. ``files`` takes a repo-relative glob (e.g.
``data/telegram_blogs*``) passed as ``data_files`` — it selects a subset of
a repo without a named config, and jsonl.gz/parquet/CSV layouts all work.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

from . import Dataset, cyrillic_ratio, register
from .fineweb2 import windows_from_doc


@register("hf")
@dataclass
class HfCorpus(Dataset):
    repo: str
    lang: str
    config: str | None = None
    column: str = "text"
    #: Split to read; most repos use ``train``, single-file dumps sometimes
    #: name it differently (sentiment140's one CSV is split ``complete``).
    split: str = "train"
    docs: int = 600
    per_doc: int = 4
    seed: int = 42
    #: Streaming keeps memory flat for huge corpora but iterates the
    #: auto-converted parquet of small non-native-parquet repos very slowly;
    #: set False to download once and sample locally.
    streaming: bool = True
    #: Yield each row as one sample (tweet/post register) instead of
    #: extracting line windows from documents (article register).
    raw: bool = False
    min_chars: int = 20
    max_chars: int = 0
    #: Row equality filter ``column=value`` (Twitter-style self-labels, gold
    #: language slices). Compared as strings.
    where: str | None = None
    #: Hub-side file glob (e.g. ``data/telegram_blogs*``) selecting a subset
    #: of a repo without a named config. Routed through hf://datasets/.
    files: str | None = None
    #: Keep only rows whose letters are ≥ this share Cyrillic (0 = off) —
    #: the script gate for mixed-script corpora.
    cyr: float = 0.0
    #: Drop rows containing CJK characters (UTF-8→GBK mojibake damage seen
    #: in told-br; CJK is never legitimate in our languages).
    drop_cjk: bool = False

    name = "hf"

    def samples(self) -> Iterator[tuple[str, str]]:
        from datasets import load_dataset
        from tqdm import tqdm

        repo: str = self.repo
        config: str | None = self.config
        kwargs: dict = {}
        if self.files:
            # A repo-relative glob selects a subset (one telegram split, one
            # parquet shard, one jsonl.gz) without a named config. When the
            # file type is identifiable we go through the packaged builder
            # with an hf:// data_files URI — that also bypasses legacy repo
            # loading scripts, which modern `datasets` refuses to run (e.g.
            # the cardiffnlp tweet corpora).
            prefix = self.files.split("*")[0]
            ext = prefix.rsplit(".", 1)[-1].lower() if "." in prefix else ""
            builder = {"jsonl": "json", "json": "json", "gz": "json", "parquet": "parquet", "csv": "csv"}.get(ext)
            if builder:
                repo = builder
                config = None
                kwargs["data_files"] = f"hf://datasets/{self.repo}/{self.files}"
            else:
                kwargs["data_files"] = self.files
        ds = load_dataset(repo, config, split=self.split, streaming=self.streaming, **kwargs)

        where_col = where_val = None
        if self.where:
            where_col, _, where_val = self.where.partition("=")
            if not where_col or not where_val:
                raise SystemExit(f"bad where {self.where!r} (expected column=value)")

        desc = f"{self.repo}:{self.config or self.files or ''}->{self.lang}"
        rng = random.Random(self.seed)
        n = 0
        if self.streaming:
            it = ds.take(self.docs) if self.docs > 0 else ds
        else:
            rows = list(ds)
            it = rows[: self.docs] if self.docs > 0 else rows
        for doc in tqdm(it, total=self.docs or None, desc=desc, leave=False):
            if self.where is not None and str(doc.get(where_col)) != where_val:
                continue
            text = doc.get(self.column)
            if not isinstance(text, str):
                continue
            if self.raw:
                text = " ".join(text.split())
                if len(text) < self.min_chars:
                    continue
                if self.max_chars:
                    text = text[: self.max_chars]
                if self.cyr > 0.0 and cyrillic_ratio(text) < self.cyr:
                    continue
                if self.drop_cjk and any("\u4e00" <= ch <= "\u9fff" for ch in text):
                    continue
                yield self.lang, text
                n += 1
            else:
                for window in windows_from_doc(text, rng, self.per_doc):
                    yield self.lang, window
                    n += 1
        print(f"  {desc}: {n} samples")
