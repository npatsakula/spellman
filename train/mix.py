"""Mix training data from any combination of pluggable dataset sources.

Each ``--source name:key=value,...`` is resolved through the source registry
(``train/sources/``), replayed from cache when options are unchanged, then
merged, split, augmented, capped and written. The pipeline splits its labor
between Polars (columnar bulk work) and Python's ``random`` (the stages whose
exact RNG streams are part of the data contract):

1. **Ingest** — each source's rows are materialized as a Polars frame. A warm,
   fingerprint-validated cache is parsed columnar with ``polars.read_ndjson``
   (``sources.cache_paths`` resolves the file); a cold cache falls back to
   ``Dataset.samples_cached()`` (download/rebuild, write-through).
2. **Dedup** — source frames are concatenated in ``--source`` order, tagged
   with their source index, and deduplicated with
   ``unique(subset=["lang","text"], keep="first", maintain_order=True)``:
   first source wins, encounter order preserved.
3. **Split** — content-addressed by crc32 of (lang, text): identical samples
   land in identical splits regardless of source order or how often the mix
   is rebuilt.
4. **Augment** — wild/shortened copies (deterministic per-row RNG streams
   seeded from crc32) appended in the original row's split, deduplicated
   against everything seen; test stays pristine.
5. **Cap + shuffle + write** — ``--cap-per-lang`` subsampling and the row
   shuffle reproduce the historical ``random.Random(seed)`` consumption
   order exactly (fresh RNG per capped language, one shared RNG across
   train/val/test shuffles), then the ordered rows go out through
   ``DataFrame.write_ndjson`` (UTF-8, non-ASCII unescaped).

Usage (from train/):
    uv run spellman-mix --out data_mix \\
        --source fineweb2:docs_per_lang=600,per_doc=4 \\
        --source tatoeba:train_per_lang=8000

List registered sources:
    uv run spellman-mix --list
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import zlib
from pathlib import Path

import polars as pl

import sources
from sources import parse_source, registered


def split_of(lang: str, text: str) -> str:
    # Deterministic split assignment (crc32, not Python's per-process-randomized
    # hash) so identical samples land in the same split across runs and sources.
    h = zlib.crc32(f"{lang}\x00{text}".encode("utf-8"))
    return "train" if h % 10 < 8 else "val" if h % 10 == 8 else "test"


# ---------------------------------------------------------------------------
# Wild augmentation: synthetic internet-noise transforms applied to clean
# samples. The featurizer canonicalizes URLs/emails/mentions/numbers, so the
# augmentor targets what canonicalization cannot: register (casing chaos,
# vowel elongation, slang loans) and residual noise classes (emoji, hashtags).
# ---------------------------------------------------------------------------

_NICKS = [
    "daria_karapet", "user_2020", "kek_lord", "nya_shkolnik", "xX_pro_Xx",
    "мама_варя", "донбасс_информ", "sadgirl22", "toha_99", "lil_bit",
]
_EMOJI = ["🤙", "👍", "😂", "🔥", "❤️", "😮", "😍", "🤣", "😢", "👌", "💀", "🥲"]
_LOANS = ["lol", "wow", "ok", "best", "top", "go", "new", "home", "love", "life", "funny", "dead", "cringe"]
_DIGITS = ["2020", "2021", "100", "24/7", "2k20", "99+", "10/10"]
_URL_ALNUM = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_VOWELS = set("аеёиоуыэюяaeiou")


def wildify(text: str, rng: random.Random) -> str:
    out = text.split()
    if not out:
        return text

    def insert(tok: str) -> None:
        out.insert(rng.randrange(len(out) + 1), tok)

    if rng.random() < 0.5:
        insert("@" + rng.choice(_NICKS))
    if rng.random() < 0.4:
        tail = "".join(rng.choice(_URL_ALNUM) for _ in range(rng.randint(6, 10)))
        insert(rng.choice(("https://t.co/", "http://bit.ly/", "https://example.com/")) + tail)
    if rng.random() < 0.2:
        bare = rng.choice(out)
        out.append("#" + bare.strip(",.!?\"'«»()") or "#news")
    if rng.random() < 0.25:
        out.append(rng.choice(_EMOJI))
    if rng.random() < 0.15:
        out.insert(0, f"RT @{rng.choice(_NICKS)}:")
    if rng.random() < 0.3:
        for i in rng.sample(range(len(out)), min(3, len(out))):
            out[i] = out[i].upper() if rng.random() < 0.5 else out[i].lower()
    if rng.random() < 0.2:
        i = rng.randrange(len(out))
        word = out[i]
        vowel_positions = [j for j, c in enumerate(word) if c.lower() in _VOWELS]
        if vowel_positions:
            j = rng.choice(vowel_positions)
            word = word[:j] + word[j] * rng.randint(2, 4) + word[j + 1 :]
            out[i] = word
    if rng.random() < 0.2:
        insert(rng.choice(_LOANS))
    if rng.random() < 0.2:
        insert(rng.choice(_DIGITS))
    return " ".join(out)


def shorten(text: str, rng: random.Random) -> str | None:
    """Random 1-3-word window — trains the short-fragment regime."""
    words = text.split()
    if len(words) < 2:
        return None
    n = min(rng.randint(1, 3), len(words))
    start = rng.randrange(len(words) - n + 1)
    frag = " ".join(words[start : start + n])
    return frag if any(c.isalpha() for c in frag) else None


def augment(
    splits: dict[str, list[tuple[str, str]]],
    seen: set[tuple[str, str]],
    wild_frac: float,
    short_frac: float,
    seed: int,
) -> dict[str, int]:
    """Add noised/shortened copies of existing rows, in the SAME split as the
    original (no train/test leakage) and deduplicated against everything.
    Deterministic: each row's transform stream is seeded from crc32 of its
    content (str hashing is process-randomized and must not be used)."""
    added = {"wild": 0, "short": 0}
    for split in list(splits):
        if split == "test":
            # Test stays pristine: synthetic noise in test measures the
            # training distribution, not reality. Wild robustness is measured
            # by the real-wild referees (rusentitweet et al.); val keeps the
            # augmented rows so θ calibration sees the wild confidence spread.
            continue
        extra: list[tuple[str, str]] = []
        for lang, text in splits[split]:
            rw = random.Random(zlib.crc32(f"{seed}\x00w\x00{lang}\x00{text}".encode("utf-8")))
            if rw.random() < wild_frac:
                key = (lang, wildify(text, rw))
                if key not in seen:
                    seen.add(key)
                    extra.append(key)
                    added["wild"] += 1
            rs = random.Random(zlib.crc32(f"{seed}\x00s\x00{lang}\x00{text}".encode("utf-8")))
            if rs.random() < short_frac:
                frag = shorten(text, rs)
                if frag is not None:
                    key = (lang, frag)
                    if key not in seen:
                        seen.add(key)
                        extra.append(key)
                        added["short"] += 1
        splits[split].extend(extra)
    return added


def read_source(ds: sources.Dataset) -> pl.DataFrame:
    """Materialize one source's samples as a {"lang","text"} DataFrame.

    Fast path: a warm cache whose sidecar matches the options fingerprint is
    parsed columnar (``read_ndjson`` beats per-line ``json.loads`` on large
    caches). Anything else — cold cache, corrupt file, unreadable sidecar —
    falls back to ``samples_cached()`` for the download/rebuild/write-through
    behavior; its (lang, text) iterator is simply drained into columns."""
    path, meta_path, fingerprint = sources.cache_paths(ds)
    if path.exists() and meta_path.exists():
        try:
            if json.loads(meta_path.read_text()) == fingerprint:
                return pl.read_ndjson(path, schema={"lang": pl.String, "text": pl.String})
        except (OSError, ValueError, pl.exceptions.PolarsError):
            pass  # unreadable/corrupt cache: rebuild through samples_cached()
    langs: list[str] = []
    texts: list[str] = []
    for lang, text in ds.samples_cached():
        langs.append(lang)
        texts.append(text)
    return pl.DataFrame(
        {"lang": pl.Series(langs, dtype=pl.String), "text": pl.Series(texts, dtype=pl.String)}
    )


def cap_stratified(
    rows: list[tuple[str, str]], cap: int, short_floor: float, seed: int
) -> list[tuple[str, str]]:
    """Cap with a guaranteed share of genuinely-short rows (<= 19 chars).

    The short lane exists because fragments of long text cannot stand in
    for real short utterances; without this, a dominant long source wins
    the plain sample and the short rows vanish under --cap-per-lang.
    Best effort: languages without enough short rows backfill with long
    ones. short_floor = 0 reduces to the historical plain sample (same
    RNG stream), so old mixes replay unchanged."""
    if short_floor <= 0:
        return random.Random(seed).sample(rows, cap)
    short = [r for r in rows if len(r[1]) <= 19]
    long_ = [r for r in rows if len(r[1]) > 19]
    rng = random.Random(seed)
    want_short = min(len(short), int(cap * short_floor))
    want_long = cap - want_short
    if want_long > len(long_):  # backfill from short
        want_short += want_long - len(long_)
        want_long = len(long_)
    out = rng.sample(short, want_short) if want_short else []
    out += rng.sample(long_, want_long) if want_long else []
    return out


def _prewarm_worker(spec: str) -> tuple[str, int]:
    """Build one source's cache to completion (runs in a worker process).

    Each spec's cache is content-addressed and independent, so cold
    caches build concurrently without write races; the mixer then
    replays them warm through its normal single-process path, keeping
    dedup order and RNG streams byte-identical to a sequential build.
    """
    from sources import diverse, fineweb2, hf, leipzig, local, opus, tatoeba, ugc  # noqa: F401,E401,F403

    name, opts = parse_source(spec)
    ds = sources.create(name, **opts)
    n = sum(1 for _ in ds.samples_cached())
    return spec, n


def prebuild_colds(specs: list[str], jobs: int) -> None:
    """Build every cold cache in `specs` with a process pool.

    Bounded by RAM, not cores: the big diverse pools hold ~1-3GB each
    during normalize+select, so 3-4 jobs is the practical ceiling on a
    16GB machine regardless of core count.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from sources import diverse, fineweb2, hf, leipzig, local, opus, tatoeba, ugc  # noqa: F401,E401,F403

    cold: list[str] = []
    for spec in specs:
        name, opts = parse_source(spec)
        if not sources.cache_is_warm(sources.create(name, **opts)):
            cold.append(spec)
    if not cold:
        print("prebuild: all caches warm")
        return
    print(f"prebuild: {len(cold)}/{len(specs)} cold caches, {jobs} jobs", flush=True)
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futures = {ex.submit(_prewarm_worker, s): s for s in cold}
        try:
            for fut in as_completed(futures):
                spec, n = fut.result()
                print(f"prebuilt {spec}: {n:,} samples", flush=True)
        except Exception:
            for f in futures:
                f.cancel()
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data_mix")
    parser.add_argument("--source", action="append", default=[], metavar="NAME[:K=V,...]")
    parser.add_argument("--seed", type=int, default=42, help="row shuffle seed (splits are content-deterministic)")
    parser.add_argument(
        "--cap-per-lang",
        type=int,
        default=None,
        help="cap each language's rows per split (seeded subsample) so one "
        "dominant source cannot skew the val/test aggregates",
    )
    parser.add_argument(
        "--wild-augment",
        type=float,
        default=0.0,
        help="probability per row of also emitting an internet-noised copy "
        "(URLs, mentions, emoji, casing chaos, elongation, loans) in the same split",
    )
    parser.add_argument(
        "--short-floor",
        type=float,
        default=0.0,
        help="when capping, guarantee at least this fraction of each "
        "language's sample is genuinely-short rows (<=19 chars, the wild "
        "short lane); languages without enough short rows backfill with long",
    )
    parser.add_argument(
        "--short-augment",
        type=float,
        default=0.0,
        help="probability per row of also emitting a random 1-3-word fragment "
        "in the same split (short-text regime training)",
    )
    parser.add_argument("--jobs", type=int, default=1,
                        help="build cold source caches in N parallel processes before mixing "
                             "(warm caches are skipped; RAM-bound: 3-4 is the ceiling on 16GB "
                             "because big diverse pools hold GBs each)")
    parser.add_argument("--list", action="store_true", help="list registered sources and exit")
    args = parser.parse_args()

    # Import adapters for their registration side effects.
    from sources import diverse, fineweb2, hf, leipzig, local, opus, tatoeba, ugc  # noqa: F401,E401

    if args.list:
        for name in registered():
            print(name)
        return
    if not args.source:
        raise SystemExit("no --source given (see --list)")

    if args.jobs > 1:
        prebuild_colds(args.source, args.jobs)

    # Ingest + cross-source dedup, first source wins. Deduplicating
    # incrementally keeps the per-source "-> N unique samples" counts and
    # console interleaving identical to the streaming mixer; later sources
    # can never steal a row an earlier source already claimed.
    kept: pl.DataFrame | None = None
    for src_idx, spec in enumerate(args.source):
        name, opts = parse_source(spec)
        ds = sources.create(name, **opts)
        print(f"source {spec}: {opts or 'defaults'}")
        frame = read_source(ds).with_columns(_src=pl.lit(src_idx, dtype=pl.UInt32))
        before = 0 if kept is None else kept.height
        kept = frame if kept is None else pl.concat([kept, frame], how="vertical")
        kept = kept.unique(subset=["lang", "text"], keep="first", maintain_order=True)
        print(f"  -> {kept.height - before} unique samples")
    assert kept is not None

    # Content-addressed split assignment. crc32 has no Polars expression, and
    # the augmentation/cap/shuffle stages below need Python str rows anyway,
    # so the split is computed in one Python pass over the deduped columns.
    langs = kept["lang"].to_list()
    texts = kept["text"].to_list()
    splits: dict[str, list[tuple[str, str]]] = {"train": [], "val": [], "test": []}
    seen: set[tuple[str, str]] = set()
    for lang, text in zip(langs, texts):
        key = (lang, text)
        seen.add(key)
        splits[split_of(lang, text)].append(key)

    if args.wild_augment > 0 or args.short_augment > 0:
        added = augment(splits, seen, args.wild_augment, args.short_augment, args.seed)
        print(f"augmentation: +{added['wild']} wild, +{added['short']} short fragments")

    # Cap and shuffle stay in Python on purpose: both consume random.Random
    # streams whose exact draw order defines the output (a fresh Random(seed)
    # per capped language — Polars' sample/shuffle use their own RNG and
    # cannot reproduce random.Random.sample / .shuffle). The ordered rows are
    # handed back to Polars for the write.
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        if args.cap_per_lang is not None:
            by_lang: dict[str, list[tuple[str, str]]] = {}
            for row in rows:
                by_lang.setdefault(row[0], []).append(row)
            rows = [
                row
                for lang in sorted(by_lang)
                for row in (
                    by_lang[lang]
                    if len(by_lang[lang]) <= args.cap_per_lang
                    else cap_stratified(by_lang[lang], args.cap_per_lang, args.short_floor, args.seed)
                )
            ]
        rng.shuffle(rows)
        path = args.out / f"{split}.jsonl"
        pl.DataFrame(
            {"lang": [r[0] for r in rows], "text": [r[1] for r in rows]},
            schema={"lang": pl.String, "text": pl.String},
        ).write_ndjson(path)
        print(f"{split}: {len(rows)} -> {path}")

    # The mix recipe is data: record it so a model directory can always be
    # traced back to its exact sources and knobs (source order matters —
    # dedup is first-source-wins).
    manifest = {
        "argv": sys.argv[1:],
        "seed": args.seed,
        "cap_per_lang": args.cap_per_lang,
        "wild_augment": args.wild_augment,
        "short_augment": args.short_augment,
        "short_floor": args.short_floor,
        "sources": [list(parse_source(spec)) for spec in args.source],
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
