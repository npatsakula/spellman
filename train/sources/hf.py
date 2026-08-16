"""Generic HuggingFace Hub corpus source: any repo/config, one language.

The escape hatch that makes new datasets a CLI knob instead of new code —
point it at a dataset with a text column and it joins the mix with the same
line-window sampling as the FineWeb-2 source:

    --source hf:repo=cis-lmu/Glot500,config=tat_Cyrl,lang=tat,docs=3000
    --source hf:repo=AigizK/tatar-russian-parallel-corpora,column=tat,lang=tat

For multi-language datasets, declare one source per language/config.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

from . import Dataset, register
from .fineweb2 import windows_from_doc


@register("hf")
@dataclass
class HfCorpus(Dataset):
    repo: str
    lang: str
    config: str | None = None
    column: str = "text"
    docs: int = 600
    per_doc: int = 4
    seed: int = 42
    #: Streaming keeps memory flat for huge corpora but iterates the
    #: auto-converted parquet of small non-native-parquet repos very slowly;
    #: set False to download once and sample locally.
    streaming: bool = True

    name = "hf"

    def samples(self) -> Iterator[tuple[str, str]]:
        from datasets import load_dataset
        from tqdm import tqdm

        ds = load_dataset(self.repo, self.config, streaming=self.streaming, split="train")
        rng = random.Random(self.seed)
        desc = f"{self.repo}:{self.config or ''}->{self.lang}"
        if self.streaming:
            docs = ds.take(self.docs)
        else:
            docs = list(ds)[: self.docs]
        n = 0
        for doc in tqdm(docs, total=self.docs, desc=desc, leave=False):
            text = doc.get(self.column)
            if not isinstance(text, str):
                continue
            for window in windows_from_doc(text, rng, self.per_doc):
                yield self.lang, window
                n += 1
        print(f"  {desc}: {n} samples")
