"""Token normalization (lemmatization) — a property of the language.

Which normalizer a language uses is decided HERE, once, and every
consumer (the `diverse` adapter today) asks the registry via
``normalizer_for(lang)``. No per-source knobs: two sources of the same
language must never normalize differently, and adding a language's
lemmatizer is one registry entry — every consumer upgrades instantly.

Registry values are builder names; builders import their machinery
lazily so pipelines for other languages never pay for (or require) it.
The fallback is identity over lowercase tokens — coverage selection on
surface forms is still a large win over random sampling, just less exact
under rich morphology.

Adding a language: add a builder + one registry entry, e.g. a Kazakh
stemmer when one enters the dependency set.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

#: lang -> builder name (the single extension point)
REGISTRY: dict[str, str] = {
    "rus": "pymorphy3",
    # Ukrainian: the pymorphy2-format dicts from pymorphy2-dicts-uk,
    # loaded by path into pymorphy3's analyzer
    "ukr": "pymorphy3-uk",
}


def _build_pymorphy3() -> Callable[[list[str]], list[str]]:
    import pymorphy3

    morph = pymorphy3.MorphAnalyzer()
    cache: dict[str, str] = {}

    def lemmas(tokens: list[str]) -> list[str]:
        c = cache
        for t in tokens:
            if t not in c:
                p = morph.parse(t)
                c[t] = p[0].normal_form if p else t
        return [c[t] for t in tokens]

    return lemmas


def _build_pymorphy3_uk() -> Callable[[list[str]], list[str]]:
    import pathlib

    import pymorphy2_dicts_uk
    import pymorphy3
    from pymorphy3.units.by_lookup import DictionaryAnalyzer

    dict_path = pathlib.Path(pymorphy2_dicts_uk.__file__).parent / "data"
    # Dictionary-only: the uk dict's prediction-suffixes DAWGs predate the
    # installed dawg-python format and crash on OOV lookups (struct.error).
    # Without prediction units, OOV parses return [] and the caller falls
    # back to the surface token — the same policy as unregistered langs.
    morph = pymorphy3.MorphAnalyzer(path=str(dict_path), units=[DictionaryAnalyzer()])
    cache: dict[str, str] = {}

    def lemmas(tokens: list[str]) -> list[str]:
        c = cache
        for t in tokens:
            if t not in c:
                p = morph.parse(t)
                c[t] = p[0].normal_form if p else t
        return [c[t] for t in tokens]

    return lemmas


_BUILDERS: dict[str, Callable[[], Callable[[list[str]], list[str]]]] = {
    "pymorphy3": _build_pymorphy3,
    "pymorphy3-uk": _build_pymorphy3_uk,
}


@lru_cache(maxsize=None)
def normalizer_for(lang: str) -> Callable[[list[str]], list[str]]:
    """Token->lemma function for a language (identity fallback).

    One instance per language per process, shared by every source — the
    per-token cache survives across sources of the same language."""
    builder = _BUILDERS.get(REGISTRY.get(lang, ""))
    if builder is None:
        return lambda tokens: tokens
    return builder()
