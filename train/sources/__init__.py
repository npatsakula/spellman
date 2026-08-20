"""Pluggable dataset adapters for spellman training data.

Adding a new dataset = one module in this package with a class that:
  1. subclasses :class:`Dataset`,
  2. declares its options as dataclass fields (they become the
     `--source name:key=value` knobs of `spellman-mix`), and
  3. implements :meth:`Dataset.samples` yielding `(lang_code, text)` pairs.

Register it with ``@register("name")`` and it is immediately mixable:

    uv run spellman-mix --out data_mix \\
        --source fineweb2:docs_per_lang=600,per_doc=4 \\
        --source tatoeba:train_per_lang=8000 \\
        --source my_new_dataset

Each adapter's output is cached under ``cache/<name>-<options-hash>.jsonl``
with an options-fingerprint sidecar: re-mixing with unchanged options replays
the cache (no re-download), while any option change rebuilds it.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Callable, Iterator, Type

TRAIN_DIR = Path(__file__).parent.parent
CACHE_DIR = TRAIN_DIR / "cache"

_REGISTRY: dict[str, type[Dataset]] = {}


def register(name: str) -> Callable[[type[Dataset]], type[Dataset]]:
    def wrap(cls: type[Dataset]) -> type[Dataset]:
        _REGISTRY[name] = cls
        return cls

    return wrap


def create(name: str, **opts) -> Dataset:
    """Instantiate a registered adapter from mix-source options."""
    cls = _REGISTRY.get(name)
    if cls is None:
        known = ", ".join(sorted(_REGISTRY))
        raise SystemExit(f"unknown dataset {name!r} (registered: {known})")
    known = {f.name for f in fields(cls)} if is_dataclass(cls) else set()
    bad = set(opts) - known
    if bad:
        raise SystemExit(f"dataset {name!r}: unknown options {sorted(bad)} (known: {sorted(known)})")
    return cls(**opts)


def registered() -> list[str]:
    return sorted(_REGISTRY)


def _coerce(value: str) -> object:
    import ast

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def download_with_retries(url: str, dest: Path, attempts: int = 3, timeout: float = 60.0) -> None:
    """Fetch `url` to `dest` with backoff. DNS hiccups and connection
    resets on corpus hosts are routine; one failed attempt must not kill
    a mix that has been downloading for an hour."""
    import time
    import urllib.request

    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=timeout) as resp, tmp.open("wb") as out:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
            tmp.rename(dest)
            return
        except Exception as err:  # noqa: BLE001 — retry anything network-shaped
            last = err
            tmp.unlink(missing_ok=True)
            if attempt < attempts:
                delay = 5.0 * attempt
                print(f"  download attempt {attempt}/{attempts} failed ({err}); retrying in {delay:.0f}s", flush=True)
                time.sleep(delay)
    raise SystemExit(f"could not fetch {url} after {attempts} attempts: {last}")


def parse_source(spec: str) -> tuple[str, dict[str, object]]:
    """``name:key=value,key=value`` -> (name, {key: value})."""
    name, _, opts_str = spec.partition(":")
    opts = {}
    if opts_str:
        for part in opts_str.split(","):
            key, eq, value = part.partition("=")
            if not eq:
                raise SystemExit(f"bad option {part!r} in source {name!r} (expected key=value)")
            opts[key.strip()] = _coerce(value.strip())
    return name.strip(), opts


def cyrillic_ratio(text: str) -> float:
    """Share of a text's Unicode letters that are Cyrillic.

    The script gate for mixed-script corpora (modern Uzbek web data is ~half
    Latin; Serbian UGC is Latin-dominant): keep rows whose letters are
    predominantly Cyrillic. Texts with no letters at all score 0.0.
    """
    letters = 0
    cyr = 0
    for ch in text:
        if ch.isalpha():
            letters += 1
            if "\u0400" <= ch <= "\u04FF":
                cyr += 1
    return cyr / letters if letters else 0.0


def _fingerprint(ds: Dataset) -> dict:
    """Option values normalized to JSON-comparable primitives (Path -> str)."""
    if not is_dataclass(ds):
        return {}
    return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(ds).items()}


def cache_paths(ds: Dataset, cache_dir: Path = CACHE_DIR) -> tuple[Path, Path, dict]:
    """Resolve a dataset instance's (cache_file, sidecar, fingerprint).

    Shared by :meth:`Dataset.samples_cached` and callers that want to read a
    warm cache directly (e.g. the mixer's columnar fast path via
    ``polars.read_ndjson``). The sidecar validates that the cache was built
    with exactly these options; the sha1 slug keeps several instances of one
    adapter (e.g. multiple ``hf`` sources) from colliding.
    """
    fingerprint = {"dataset": type(ds).__name__, "options": _fingerprint(ds)}
    slug = hashlib.sha1(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:10]
    path = cache_dir / f"{ds.name}-{slug}.jsonl"
    return path, path.with_suffix(".options.json"), fingerprint


def cache_is_warm(ds: Dataset, cache_dir: Path = CACHE_DIR) -> bool:
    """True when the source's cache exists and its sidecar matches the
    current options fingerprint (i.e. ``samples_cached`` would replay it
    without rebuilding). Lets callers prebuild only the cold caches."""
    path, meta_path, fingerprint = cache_paths(ds, cache_dir)
    if not (path.exists() and meta_path.exists()):
        return False
    try:
        return json.loads(meta_path.read_text()) == fingerprint
    except json.JSONDecodeError:
        return False


class Dataset(ABC):
    """A training-data source: acquisition + conversion to (lang, text)."""

    #: populated by subclasses via @register
    name: str = "?"

    def samples_cached(self, cache_dir: Path = CACHE_DIR) -> Iterator[tuple[str, str]]:
        """Yield samples through the on-disk cache (write-through on first run).

        The cache file is keyed by a short hash of the options fingerprint so
        several instances of one adapter (e.g. multiple `hf` sources pointing
        at different repos) never collide.
        """
        cache_dir.mkdir(parents=True, exist_ok=True)
        path, meta_path, fingerprint = cache_paths(self, cache_dir)
        if path.exists() and meta_path.exists():
            try:
                if json.loads(meta_path.read_text()) == fingerprint:
                    with path.open(encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                row = json.loads(line)
                                yield row["lang"], row["text"]
                    return
            except (json.JSONDecodeError, KeyError):
                pass  # corrupt cache: rebuild
        n = 0
        with path.open("w", encoding="utf-8") as f:
            for lang, text in self.samples():
                f.write(json.dumps({"lang": lang, "text": text}, ensure_ascii=False) + "\n")
                n += 1
                yield lang, text
        meta_path.write_text(json.dumps(fingerprint, indent=1), encoding="utf-8")
        print(f"  [{self.name}] cached {n} samples -> {path}")

    @abstractmethod
    def samples(self) -> Iterator[tuple[str, str]]:
        """Yield (lang_code, text) training samples for this source."""
        raise NotImplementedError
