"""Lexical-diversity source: lemma-coverage selection over a pool.

The bulk pools (FineWeb-2 & co.) are Zipfian — random rows re-cover the
same frequent words while everyday vocabulary ("молодцы", "обалденно")
never shows up in a small sample; we measured exactly this failure mode
on short wild Russian. This adapter turns a big pool into a *small but
maximally diverse* training set.

Pipeline (v2), decoupled so each stage is paid for once:

    pool -> normalize (sidecar cache) -> weighted lazy-greedy selection

* **Pool** — an HF repo (streamed, split into `min_chars..max_chars`
  sentences, materialized once under cache/) or any existing cache file.
  Pools are content-immobile on disk, so everything downstream can key
  off a content hash.
* **Normalization sidecar** — lemmas for every pool sentence are computed
  once per (pool content, normalizer) pair and cached next to the pool.
  A/B-ing selection knobs or upgrading a language's registry entry then
  re-selects without re-parsing anything.
* **Selection** — lazy-greedy weighted maximum coverage (Minoux): a
  max-heap of stale marginal gains, rescoring only the head after each
  pick. A sentence's marginal gain sums, per unique lemma,

      +1  new lemma with pool document-frequency >= min_df
      +1  top-K lemma still under its exposure floor (repeat it)

  and sentences are taken while the best gain >= min_gain (tiny pools
  get a tail pass at gain >= 1 so budget ~= pool still saturates
  coverage). Two measured lessons are baked into the weights: singleton
  lemmas (df < min_df) carry ZERO weight — the rare_rescue experiment
  showed chasing the pool's singleton tail displaces mid-frequency
  coverage and regressed every referee — while the exposure floor keeps
  frequent lemmas repeated: coverage gives a dictionary, mass gives a
  distribution, and the model needs both.

Lemmatization is a language property, resolved through
sources/normalize.py (pymorphy3 for rus/ukr, Stanza for the Slavic lanes
and kaz/kir, hunspell stems for mon, identity elsewhere until more
languages enter the registry).
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from . import CACHE_DIR, Dataset, register
from .normalize import normalizer_for, normalizer_id

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)

#: sentences longer than this many tokens are skipped: their n-gram mass
#: is redundant and they crowd out shorter, wilder rows
_MAX_TOKENS = 40


def _sentences(text: str, min_chars: int, max_chars: int) -> list[str]:
    out = []
    for s in _SENT_SPLIT.split(text):
        s = " ".join(s.split())
        if min_chars <= len(s) <= max_chars:
            out.append(s)
    return out


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


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
    #: accept a sentence only while its marginal gain >= this
    min_gain: int = 2
    #: lemmas appearing in fewer pool sentences carry zero selection
    #: weight (singleton proper nouns are weak language evidence —
    #: measured: rare-chasing regressed every referee)
    min_df: int = 2
    #: top-N lemmas (by pool sentence-frequency) each get >= min_exposures picks
    expose_top: int = 15_000
    min_exposures: int = 3
    min_chars: int = 20
    max_chars: int = 200
    #: cap on candidate sentences held in memory
    max_candidates: int = 1_500_000
    seed: int = 42
    #: effective normalizer (set from the registry) — folded into the
    #: cache fingerprint so re-keying a language rebuilds its selection
    norm: str = ""
    #: selection algorithm version — bump to bust every diverse cache
    #: (2 = lazy greedy: length-biased, regressed; 3 = + strata;
    #:  4 = stratified first-come, v1 rule + strata + df weights)
    algo: int = 4

    name = "diverse"

    def __post_init__(self) -> None:
        if isinstance(self.pool_file, str):
            self.pool_file = Path(self.pool_file)
        if not self.pool_file and not self.pool_repo:
            raise SystemExit("diverse: needs pool_file or pool_repo")
        self.norm = normalizer_id(self.lang)

    # ------------------------------------------------------------- pool

    def _pool_path(self) -> Path:
        """Local pool file, materializing a streamed repo on first use."""
        if self.pool_file is not None:
            return self.pool_file
        fp = {
            "repo": self.pool_repo,
            "config": self.pool_config,
            "docs": self.pool_docs,
            "min_chars": self.min_chars,
            "max_chars": self.max_chars,
            "max_candidates": self.max_candidates,
            "v": 2,
        }
        slug = hashlib.sha1(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:10]
        path = CACHE_DIR / f"diverse-pool-{slug}.jsonl"
        if not path.exists():
            from datasets import load_dataset
            from tqdm import tqdm

            ds = load_dataset(
                self.pool_repo, self.pool_config, streaming=True, split="train"
            ).take(self.pool_docs)
            tmp = path.with_suffix(".jsonl.part")
            n = 0
            with tmp.open("w", encoding="utf-8") as out:
                for doc in tqdm(ds, total=self.pool_docs, desc=f"pool[{self.lang}]", leave=False):
                    for s in _sentences(doc.get("text") or "", self.min_chars, self.max_chars):
                        out.write(json.dumps({"text": s}, ensure_ascii=False) + "\n")
                        n += 1
                        if n >= self.max_candidates:
                            break
                    if n >= self.max_candidates:
                        break
            tmp.rename(path)
            print(f"  diverse[{self.lang}]: materialized pool {n:,} sentences -> {path.name}", flush=True)
        return path

    def _pool_texts(self, pool: Path) -> list[str]:
        rows = []
        for line in pool.open(encoding="utf-8"):
            if line.strip():
                t = json.loads(line)["text"]
                if self.min_chars <= len(t) <= self.max_chars:
                    rows.append(t)
        return rows[: self.max_candidates]

    # ------------------------------------------------------ normalization

    def _lemmatized(self, pool: Path, texts: list[str]) -> list[list[str]]:
        """Lemmas per pool sentence, cached in a sidecar keyed by pool
        content + normalizer identity."""
        fp = {"lang": self.lang, "norm": self.norm, "pool": _sha1_file(pool), "v": 2}
        slug = hashlib.sha1(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:10]
        side = pool.with_name(f"{pool.stem}.norm-{slug}.jsonl")
        if side.exists():
            out = []
            with side.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.append([sys.intern(l) for l in json.loads(line)["l"]])
            if len(out) == len(texts):
                return out
            print(f"  diverse[{self.lang}]: sidecar stale ({len(out):,} != {len(texts):,}), rebuilding", flush=True)

        t0 = time.time()
        norm = normalizer_for(self.lang)
        bulk = getattr(norm, "bulk", None)
        out: list[list[str]] = []
        if bulk is not None:
            uniq: dict[str, None] = {}
            for t in texts:
                uniq.update(dict.fromkeys(_TOKEN.findall(t.lower())))
            mapping = bulk(list(uniq))
            for t in texts:
                out.append([mapping[tok] for tok in _TOKEN.findall(t.lower())])
        else:
            for t in texts:
                out.append(norm(_TOKEN.findall(t.lower())))
        tmp = side.with_suffix(".jsonl.part")
        with tmp.open("w", encoding="utf-8") as f:
            for lemmas in out:
                f.write(json.dumps({"l": lemmas}, ensure_ascii=False) + "\n")
        tmp.rename(side)
        print(
            f"  diverse[{self.lang}]: normalized {len(out):,} sentences "
            f"[{self.norm}] in {time.time() - t0:.0f}s -> {side.name}",
            flush=True,
        )
        return out

    # -------------------------------------------------------- selection

    #: length strata (char counts): selection runs WITHIN each stratum,
    #: budget shared by the stratum's pool share. An unstratified
    #: optimizer — even per-pick lazy greedy — prefers long sentences
    #: (more new lemmas per pick): measured on v8, the rus lane came out
    #: mean 149 chars with ZERO rows <=40, every referee regressed.
    #: Stratifying keeps the lane's length profile at the pool's natural
    #: shape (the 20-200 window was chosen to match wild inference).
    _STRATA = (60, 120)

    def _select(self, texts: list[str], lemma_lists: list[list[str]]) -> list[int]:
        """Stratified first-come coverage selection; accepted indices in
        acceptance order.

        Measured history (held-out ukr->rus / rst-v2 / short):
          v1 shuffle first-come ..... 217 / 92.94 / 90.24  (the baseline)
          v2 lazy greedy max-first .. 347 / 92.02 / 91.11
          v2 + min_df=1 ............. 426 / 91.86 / 89.55
        Max-first optimization harvests the globally richest sentences
        first and distorts register/topic mix — coverage works better as
        an acceptance FILTER over a random shuffle than as an objective
        to maximize. So: seeded shuffle (v1's rule) within length strata
        (v2's fix), gains weighted by df >= min_df (v2's fix), exposure
        floors for the top of the vocabulary (v1's rule).
        """
        t0 = time.time()
        df: Counter[str] = Counter()
        for lemmas in lemma_lists:
            df.update(set(lemmas))
        top = set(sorted(df, key=df.get, reverse=True)[: self.expose_top])
        covered: set[str] = set()
        exposures: dict[str, int] = {}

        cands = [i for i, ls in enumerate(lemma_lists) if ls and len(ls) <= _MAX_TOKENS]
        rng = random.Random(self.seed)
        rng.shuffle(cands)
        strata: list[list[int]] = [[], [], []]
        for i in cands:
            n = len(texts[i])
            strata[0 if n <= self._STRATA[0] else 1 if n <= self._STRATA[1] else 2].append(i)

        total = sum(len(s) for s in strata)
        caps = [round(self.budget * len(s) / max(1, total)) for s in strata]
        taken = [0, 0, 0]
        order: list[int] = []
        accepted: set[int] = set()

        def accept(i: int) -> None:
            order.append(i)
            accepted.add(i)
            for lemma in set(lemma_lists[i]):
                if df[lemma] >= self.min_df:
                    covered.add(lemma)
                if lemma in top:
                    exposures[lemma] = exposures.get(lemma, 0) + 1

        def try_accept(i: int, si: int, others_done: bool) -> bool:
            new = 0
            starved = False
            for lemma in set(lemma_lists[i]):
                if df[lemma] >= self.min_df and lemma not in covered:
                    new += 1
                if lemma in top and exposures.get(lemma, 0) < self.min_exposures:
                    starved = True
            if not (new >= self.min_gain or (starved and new >= 1)):
                return False
            if taken[si] >= caps[si] and not others_done:
                return False  # stratum full; revisit on the drain pass
            accept(i)
            taken[si] += 1
            return True

        # pass 1: one walk over the shuffled candidates, stratum caps enforced
        for si, stratum in enumerate(strata):
            for i in stratum:
                if len(order) >= self.budget:
                    break
                try_accept(i, si, others_done=False)
        # pass 2: budget left on the table (some stratum exhausted early)
        # -> drain every stratum, caps lifted
        if len(order) < self.budget:
            for si, stratum in enumerate(strata):
                for i in stratum:
                    if len(order) >= self.budget:
                        break
                    if i not in accepted:
                        try_accept(i, si, others_done=True)

        objective = sum(1 for lemma in df if df[lemma] >= self.min_df)
        short = sum(1 for i in order if len(texts[i]) <= self._STRATA[0])
        print(
            f"  diverse[{self.lang}]: kept {len(order):,}/{len(texts):,} sentences "
            f"({100 * short / max(1, len(order)):.0f}% <= {self._STRATA[0]} chars); "
            f"lemmas covered {len(covered):,}/{objective:,} df>={self.min_df} "
            f"({100 * len(covered) / max(1, objective):.0f}%); top-{self.expose_top:,} "
            f"exposed {len(covered & top):,}; select {time.time() - t0:.0f}s",
            flush=True,
        )
        return order

        objective = sum(1 for lemma in df if df[lemma] >= self.min_df)
        short = sum(1 for i in order if len(texts[i]) <= self._STRATA[0])
        print(
            f"  diverse[{self.lang}]: kept {len(order):,}/{len(texts):,} sentences "
            f"({100 * short / max(1, len(order)):.0f}% <= {self._STRATA[0]} chars); "
            f"lemmas covered {len(covered):,}/{objective:,} df>={self.min_df} "
            f"({100 * len(covered) / max(1, objective):.0f}%); top-{self.expose_top:,} "
            f"exposed {len(covered & top):,}; select {time.time() - t0:.0f}s",
            flush=True,
        )
        return order

    def samples(self) -> Iterator[tuple[str, str]]:
        pool = self._pool_path()
        texts = self._pool_texts(pool)
        print(f"  diverse[{self.lang}]: pool {len(texts):,} sentences ({pool.name})", flush=True)
        lemma_lists = self._lemmatized(pool, texts)
        order = self._select(texts, lemma_lists)
        for i in order:
            yield self.lang, texts[i]
