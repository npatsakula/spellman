"""Lexical-diversity source: greedy lemma-coverage selection over a pool.

The bulk pools (FineWeb-2 & co.) are Zipfian — random rows re-cover the
same frequent words while everyday vocabulary ("молодцы", "обалденно")
never shows up in a small sample; we measured exactly this failure mode
on short wild Russian. This adapter turns a big pool into a *small but
maximally diverse* training set: sentences are selected greedily by the
number of NEW lemmas they contribute, with a frequency floor so the top
of the vocabulary keeps repeated exposures (coverage gives a dictionary;
mass gives a distribution — the model needs both).

Pool comes from either an HF repo/config (streamed) or any already-built
cache file, so every provider in this package can feed it:

    # from FineWeb-2 directly (any language with a config):
    --source 'diverse:lang=rus,pool_repo=HuggingFaceFW/fineweb-2,pool_config=rus_Cyrl,pool_docs=300000,budget=14000'
    # from an existing cache (leipzig/opus/csv/hf pools alike):
    --source 'diverse:lang=kaz,pool_file=cache/kazsandra-eb6571e000.jsonl,budget=8000'

Lemmatization is a language property, resolved through
sources/normalize.py (pymorphy3 for Russian — morphology makes surface
counting lie, "молодцы/молодец/молодца" are one lemma; identity fallback
elsewhere until dicts are added to the registry).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from . import Dataset, register
from .normalize import normalizer_for

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)

def _sentences(text: str, min_chars: int, max_chars: int) -> list[str]:
    out = []
    for s in _SENT_SPLIT.split(text):
        s = " ".join(s.split())
        if min_chars <= len(s) <= max_chars:
            out.append(s)
    return out


@register("diverse")
@dataclass
class Diverse(Dataset):
    lang: str
    #: pool A: an HF repo (streamed), like the hf adapter's repo/config
    pool_repo: str | None = None
    pool_config: str | None = None
    pool_docs: int = 300_000
    #: pool B: any built cache file ({"lang","text"} rows) — rows are used
    #: as-is (they are already unit-sized); overrides pool_repo
    pool_file: Path | None = None
    #: training rows to emit
    budget: int = 14_000
    #: accept a sentence only if it contributes >= this many new lemmas
    min_gain: int = 2
    #: top-N lemmas (by pool frequency) each get >= min_exposures picks
    expose_top: int = 15_000
    min_exposures: int = 3
    min_chars: int = 20
    max_chars: int = 200
    #: cap on candidate sentences held in memory
    max_candidates: int = 1_500_000
    seed: int = 42

    name = "diverse"

    def __post_init__(self) -> None:
        if isinstance(self.pool_file, str):
            self.pool_file = Path(self.pool_file)
        if not self.pool_file and not self.pool_repo:
            raise SystemExit("diverse: needs pool_file or pool_repo")

    def _pool_sentences(self) -> list[str]:
        if self.pool_file is not None:
            rows = []
            for line in self.pool_file.open(encoding="utf-8"):
                if line.strip():
                    t = json.loads(line)["text"]
                    if self.min_chars <= len(t) <= self.max_chars:
                        rows.append(t)
            return rows[: self.max_candidates]
        from datasets import load_dataset

        ds = load_dataset(self.pool_repo, self.pool_config, streaming=True, split="train").take(self.pool_docs)
        out: list[str] = []
        for doc in ds:
            for s in _sentences(doc.get("text") or "", self.min_chars, self.max_chars):
                out.append(s)
                if len(out) >= self.max_candidates:
                    return out
        return out

    def samples(self) -> Iterator[tuple[str, str]]:
        sents = None  # populated in pass 1 below
        import random

        sents = self._pool_sentences()
        # Shuffle before greedy: sequential scanning accepts the first
        # qualifying sentences, so pool order (crawl order!) would decide
        # the selection head. Seeded -> deterministic mixes.
        random.Random(self.seed).shuffle(sents)
        print(f"  diverse[{self.lang}]: pool {len(sents):,} sentences", flush=True)

        # Pass 1: lemma frequency. Counting surfaces is cheap but WRONG
        # for the exposure floor — it keys lemmas ("молодец") against
        # inflected surfaces ("молодцов") and never fires. Lemmatizing
        # only the K most frequent surfaces costs seconds and gives an
        # honest lemma ranking.
        freq: dict[str, int] = {}
        for s in sents:
            for t in _TOKEN.findall(s.lower()):
                freq[t] = freq.get(t, 0) + 1
        norm = normalizer_for(self.lang)
        lemma_freq: dict[str, int] = {}
        for t in sorted(freq, key=freq.get, reverse=True)[:200_000]:
            for l in {norm([t])[0]}:
                lemma_freq[l] = lemma_freq.get(l, 0) + freq[t]
        top = set(sorted(lemma_freq, key=lemma_freq.get, reverse=True)[: self.expose_top])
        exposures: dict[str, int] = {}

        # Pass 2: greedy. A sentence is taken if it adds >= min_gain new
        # lemmas, or if it carries a top-lemma still under its exposure
        # floor (frequency mass interleaves with coverage in one pass).
        seen: set[str] = set()
        taken = 0
        for s in sents:
            if taken >= self.budget:
                break
            tokens = _TOKEN.findall(s.lower())
            if len(tokens) > 40 or not tokens:
                continue
            lemmas = norm(tokens)
            new = [l for l in lemmas if l not in seen]
            starved = [l for l in lemmas if l in top and exposures.get(l, 0) < self.min_exposures]
            if len(new) >= self.min_gain or (starved and len(new) >= 1):
                for l in lemmas:
                    if l in top:
                        exposures[l] = exposures.get(l, 0) + 1
                seen.update(new)
                yield self.lang, s
                taken += 1
        covered = len(seen & top)
        print(
            f"  diverse[{self.lang}]: kept {taken:,}/{len(sents):,} sentences, "
            f"{len(seen):,} lemmas seen, top-{self.expose_top:,} covered {covered:,} "
            f"({100 * covered / max(1, len(top)):.0f}%)",
            flush=True,
        )
