"""Token normalization (lemmatization) — a property of the language.

Which normalizer a language uses is decided HERE, once, and every
consumer (the `diverse` adapter today) asks the registry via
``normalizer_for(lang)``. No per-source knobs: two sources of the same
language must never normalize differently, and adding a language's
lemmatizer is one registry entry — every consumer upgrades instantly.

Registry values are builder names; builders import their machinery
lazily so pipelines for other languages never pay for (or require) it.
The fallback is identity over lowercase tokens — coverage selection on
surface forms is still a large win over random sampling, just less
exact under rich morphology.

Every builder returns a ``lemmas(tokens) -> list[str]`` callable that
also carries a ``bulk`` attribute (``bulk(tokens) -> {token: lemma}``)
for whole-vocabulary passes: the diverse sidecar uses it so neural /
FST lookups run once per unique token, batched.

``normalizer_id(lang)`` exposes the registry string so caches can
fingerprint normalization *behavior*, not just options — re-keying a
language here must rebuild everything keyed on its lemmas.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

from spellman_train.paths import TRAIN_DIR

#: lang -> builder name (the single extension point). The string is part
#: of the diverse cache fingerprint: bump/rename a value whenever its
#: builder's behavior changes.
REGISTRY: dict[str, str] = {
    "rus": "pymorphy3",
    # Ukrainian: the pymorphy2-format dicts from pymorphy2-dicts-uk,
    # loaded by path into pymorphy3's analyzer
    "ukr": "pymorphy3-uk",
    # Stanza neural lemma models (tokenize_pretokenized, pos+lemma, no
    # charlm) — the Slavic lanes plus the two biggest Turkic pools.
    "bel": "stanza",
    "bul": "stanza",
    "srp": "stanza",
    "kaz": "stanza",
    "kir": "stanza",
    # No lemmatizer/stemmer exists for these (checked: no Stanza model,
    # no UDPipe 2 model, no usable hunspell stems — spylls only exposes
    # spell-check lookup, not stemming). Prefix-cluster stems below.
    "mon": "stem-corpus",
    "tgk": "stem-corpus",
    "uzn": "stem-corpus",
    "tat": "stem-corpus",
    "bak": "stem-corpus",
    "chv": "stem-corpus",
    "sah": "stem-corpus",
    "tyv": "stem-corpus",
    # lttoolbox FST analyzer (spellman-train prepare-apertium) — Macedonian is the
    # one language only Apertium covers (the oss repo is a 757-entry
    # stub, everything else is HFST .lexc, not lttoolbox):
    "mkd": "apertium:mkd",
    # no normalizer tooling at all; prefix-cluster stems when lanes appear
    "oss": "stem-corpus",
    "che": "stem-corpus",
    "mhr": "stem-corpus",
    "udm": "stem-corpus",
    "kpv": "stem-corpus",
}

_STANZA_CODES = {"bel": "be", "bul": "bg", "srp": "sr", "kaz": "kk", "kir": "ky"}


def normalizer_id(lang: str) -> str:
    """Registry string for a language ('identity' when unregistered)."""
    return REGISTRY.get(lang, "identity")


def _cached(one: Callable[[str], str]) -> Callable[[list[str]], list[str]]:
    """Wrap a 1:1 token->lemma function with a process-wide cache + bulk."""
    cache: dict[str, str] = {}

    def lemmas(tokens: list[str]) -> list[str]:
        miss = [t for t in tokens if t not in cache]
        for t in miss:
            cache[t] = one(t)
        return [cache[t] for t in tokens]

    def bulk(tokens) -> dict[str, str]:
        lemmas(list(tokens))
        return {t: cache[t] for t in tokens}

    lemmas.bulk = bulk  # type: ignore[attr-defined]
    return lemmas


def _build_pymorphy3(lang: str) -> Callable[[list[str]], list[str]]:
    import pymorphy3

    if REGISTRY[lang] == "pymorphy3-uk":
        import pathlib

        import pymorphy2_dicts_uk
        from pymorphy3.units.by_lookup import DictionaryAnalyzer

        dict_path = pathlib.Path(pymorphy2_dicts_uk.__file__).parent / "data"
        # Dictionary-only: the uk dict's prediction-suffixes DAWGs predate
        # the installed dawg-python format and crash on OOV lookups
        # (struct.error). Without prediction units, OOV parses return []
        # and the caller falls back to the surface token.
        morph = pymorphy3.MorphAnalyzer(path=str(dict_path), units=[DictionaryAnalyzer()])
    else:
        morph = pymorphy3.MorphAnalyzer()

    def one(token: str) -> str:
        parses = morph.parse(token)
        return parses[0].normal_form if parses else token

    return _cached(one)


#: Stanza's Serbian models (SET treebank) are Latin-script: Cyrillic in
#: comes back POS=X / lemma=identity. Serbian Cyrillic->Latin is a
#: per-character bijection, so transliterate tokens on the way in and
#: keep the Latin lemma as the grouping key (script is irrelevant for
#: coverage selection).
_SR_CYR2LAT = str.maketrans(
    {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ђ": "đ", "е": "e",
        "ж": "ž", "з": "z", "и": "i", "ј": "j", "к": "k", "л": "l", "љ": "lj",
        "м": "m", "н": "n", "њ": "nj", "о": "o", "п": "p", "р": "r", "с": "s",
        "т": "t", "ћ": "ć", "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "č",
        "џ": "dž", "ш": "š", "ё": "e", "ы": "y", "э": "e", "ю": "ju", "я": "ja",
    }
)

#: tokens per Stanza pipeline call — pseudo-sentences of pretokenized
#: unique tokens; small enough to batch comfortably, large enough to
#: amortize the pipeline overhead.
_STANZA_CHUNK = 256


def _build_stanza(lang: str) -> Callable[[list[str]], list[str]]:
    import stanza

    sl = _STANZA_CODES[lang]
    try:
        nlp = stanza.Pipeline(
            lang=sl,
            processors="tokenize,pos,lemma",
            tokenize_pretokenized=True,
            verbose=False,
            download_method=None,
        )
    except Exception:  # noqa: BLE001 — first use: fetch resources, retry once
        stanza.download(sl, verbose=False)
        nlp = stanza.Pipeline(
            lang=sl,
            processors="tokenize,pos,lemma",
            tokenize_pretokenized=True,
            verbose=False,
            download_method=None,
        )

    cache: dict[str, str] = {}

    def run(chunk: list[str]) -> None:
        source = chunk
        if lang == "srp":
            chunk = [t.translate(_SR_CYR2LAT) for t in chunk]
        doc = nlp([chunk])
        words = doc.sentences[0].words
        for token, word in zip(source, words):
            # None lemma (OOV for the seq2seq head) -> surface token
            cache[token] = word.lemma or token

    def lemmas(tokens: list[str]) -> list[str]:
        miss = [t for t in dict.fromkeys(tokens) if t not in cache]
        for i in range(0, len(miss), _STANZA_CHUNK):
            run(miss[i : i + _STANZA_CHUNK])
        return [cache[t] for t in tokens]

    def bulk(tokens) -> dict[str, str]:
        lemmas(list(tokens))
        return {t: cache[t] for t in tokens}

    lemmas.bulk = bulk  # type: ignore[attr-defined]
    return lemmas


def _common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _build_stem_corpus(lang: str) -> Callable[[list[str]], list[str]]:
    """Vocabulary-internal prefix-cluster stems — the no-dictionary
    fallback for agglutinative languages.

    The vocabulary is sorted; tokens adjacent in sorted order that share
    a long-enough prefix (>=3 chars, and the shorter token is within 2
    characters of being a prefix of the longer, length gap <= 6) chain
    into one cluster keyed by its first member — inflection is almost
    entirely suffixal, so surface forms of one lemma sort together.
    Chaining makes this transitive: бала -> баласы -> баласында ->
    баласындағы one cluster even across length gaps.

    The fit is a pure function of the token SET handed to ``bulk`` (the
    diverse sidecar passes the whole pool vocabulary); ad-hoc small
    ``lemmas`` calls before any bulk fall back to identity.
    """
    state: dict[str, dict[str, str] | None] = {"fit": None}

    def fit(tokens) -> dict[str, str]:
        mapping: dict[str, str] = {}
        head: str | None = None
        prev: str | None = None
        for tok in sorted(set(tokens)):
            if prev is not None:
                cp = _common_prefix(prev, tok)
                if (
                    cp >= 3
                    and cp >= min(len(prev), len(tok)) - 2
                    and abs(len(prev) - len(tok)) <= 6
                ):
                    mapping[tok] = head  # type: ignore[index]
                    prev = tok
                    continue
            head = prev = tok
            mapping[tok] = tok
        return mapping

    def lemmas(tokens: list[str]) -> list[str]:
        mapping = state["fit"]
        if mapping is None:
            return list(tokens)
        return [mapping.get(t, t) for t in tokens]

    def bulk(tokens) -> dict[str, str]:
        if state["fit"] is None:
            state["fit"] = fit(tokens)
        mapping = state["fit"]
        return {t: mapping.get(t, t) for t in tokens}

    lemmas.bulk = bulk  # type: ignore[attr-defined]
    return lemmas


def _build_apertium(lang: str) -> Callable[[list[str]], list[str]]:
    """lttoolbox FST analyzer (see `spellman-train prepare-apertium`): one lt-proc
    subprocess per bulk pass, unique tokens on stdin, first analysis's
    lemma (text before the first '<') as the key. Unknown words come
    back as *surface and fall back to identity — the same policy as the
    pymorphy/ukr dictionary-only path."""
    import os
    import subprocess as sp

    root = TRAIN_DIR / "cache" / "apertium"
    bin_path = root / f"{lang}.automorf.bin"
    ltt = root / "lttoolbox"
    if not bin_path.exists():
        raise SystemExit(
            f"apertium analyzer for {lang} missing ({bin_path}); "
            "run: uv run spellman-train prepare-apertium"
        )
    env = {**os.environ, "DYLD_LIBRARY_PATH": str(ltt / "lib")}
    cache: dict[str, str] = {}

    def run_ltproc(chunk: list[str]) -> None:
        proc = sp.run(
            [str(ltt / "bin" / "lt-proc"), str(bin_path)],
            input="\n".join(chunk) + "\n",
            capture_output=True, text=True, env=env, check=True,
        )
        for line in proc.stdout.splitlines():
            # ^surface/lemma<tags>/lemma<tags>$ | unknown: ^surface/*surface$
            if not (line.startswith("^") and line.endswith("$")):
                continue
            body = line[1:-1]
            surface, _, analyses = body.partition("/")
            first = analyses.split("/", 1)[0]
            if first.startswith("*"):
                cache[surface] = surface
            else:
                # 'adj<pref><sup>+убав<adj>...' — apertium marks joins with
                # '+', so the headword is the segment after the last one
                cache[surface] = first.rsplit("+", 1)[-1].split("<", 1)[0]
        # lt-proc eats tokens it can't even tokenize ('km²' & co): anything
        # it didn't answer falls back to identity so bulk() stays total
        for token in chunk:
            cache.setdefault(token, token)

    def lemmas(tokens: list[str]) -> list[str]:
        bulk(tokens)
        return [cache[t] for t in tokens]

    def bulk(tokens) -> dict[str, str]:
        miss = [t for t in dict.fromkeys(tokens) if t not in cache]
        for i in range(0, len(miss), 50_000):
            run_ltproc(miss[i : i + 50_000])
        return {t: cache[t] for t in tokens}

    lemmas.bulk = bulk  # type: ignore[attr-defined]
    return lemmas


_BUILDERS: dict[str, Callable[[str], Callable[[list[str]], list[str]]]] = {
    "pymorphy3": _build_pymorphy3,
    "pymorphy3-uk": _build_pymorphy3,
    "stanza": _build_stanza,
    "stem-corpus": _build_stem_corpus,
}


@lru_cache(maxsize=None)
def normalizer_for(lang: str) -> Callable[[list[str]], list[str]]:
    """Token->lemma function for a language (identity fallback).

    One instance per language per process, shared by every source — the
    per-token cache survives across sources of the same language."""
    spec = REGISTRY.get(lang, "")
    if spec.startswith("apertium:"):
        return _build_apertium(spec.partition(":")[2])
    builder = _BUILDERS.get(spec)
    if builder is None:
        return lambda tokens: tokens
    return builder(lang)
