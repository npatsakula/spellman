"""Orthographic label gate for twin languages.

The model-judge hygiene (``spellman-train clean``) never drops a row whose
confident prediction stays inside the gold language's twin group — the
model is too unreliable there to arbitrate. Spelling is not: within a group
each language has letters its twins never write (і ї є ґ are impossible in
Russian, ы э ё ъ in Ukrainian). This gate uses them to drop rows whose label
is contradicted by their own text, e.g. Ukrainian tweets that Twitter's
`lang=ru` tag delivered as Russian.

It must survive *small* infestations — a Russian document quoting one
Ukrainian sentence is a good Russian row. So it never fires on mere presence
of a rival letter; it estimates how much of the row is written in each
language from that language's exclusive letters:

    share(X) = count(letters X has and its twin lacks)
               / (typical rate of those letters in X text × Cyrillic letters in the row)

A row labelled L is dropped for rival R when ``share(R) >= threshold`` and
``share(L) < threshold``: most of the text reads as R, and too little of it
reads as L. When L has no letters R lacks (Kyrgyz vs Kazakh), ``share(L)``
is unmeasurable and counts as 0. Rows without Cyrillic letters (Latin-script
Tatar, Serbian Latin) carry no evidence and always pass.

Only the words the detector itself reads as words are counted: mentions,
URLs, emails and digit-bearing tokens become service tokens at detection
time (:func:`spellman_train.features.classify_word`, the Rust/Python
contract), so a Russian tweet tagging ``@київ_новини`` is judged by its
text, not by the handle. Hashtags count without their ``#``, as in the
featurizer. Dirty rows are never dropped for being dirty.

``RATES`` are per-row means over the v13f training split, rows with at least
50 Cyrillic letters (:func:`measure_rates` regenerates them from any mix),
so a row written entirely in X scores ``share(X) ≈ 1``.
"""

from __future__ import annotations

from spellman_train.features import classify_word

_RUSSIAN = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"

#: Lowercase Cyrillic alphabet of every language in a twin group.
ALPHABETS: dict[str, str] = {
    "rus": _RUSSIAN,
    "ukr": "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя",
    "bel": "абвгдеёжзійклмнопрстуўфхцчшыьэюя",
    "bul": "абвгдежзийклмнопрстуфхцчшщъьюя",
    "mkd": "абвгдѓежзѕијклљмнњопрстќуфхцчџш",
    "srp": "абвгдђежзијклљмнњопрстћуфхцчџш",
    "kaz": _RUSSIAN + "әғқңөұүһі",
    "kir": _RUSSIAN + "ңөү",
    "tat": _RUSSIAN + "әөүҗңһ",
    "bak": _RUSSIAN + "әөүғҡңҙҫһ",
    "tyv": _RUSSIAN + "ңөү",
    "chv": _RUSSIAN + "ӑӗӳҫ",
    "sah": _RUSSIAN + "ҕңөһү",  # ҥ folds into ң in this group, see _FOLD
    "udm": _RUSSIAN + "ӝӟӥӧӵ",
    "mhr": _RUSSIAN + "ҥӧӱ",
    "kpv": _RUSSIAN + "іӧ",
}

#: Close-language groups: languages a model cannot reliably tell apart, so
#: hygiene protects them from each other and this gate arbitrates by spelling.
TWIN_GROUPS: list[set[str]] = [
    {"tat", "bak", "kaz", "kir", "tyv", "chv", "sah"},  # Turkic
    {"udm", "mhr", "kpv"},                              # Permic
    {"bul", "mkd", "srp"},                              # Balkan Slavic
    {"rus", "ukr", "bel"},                              # East Slavic
]

#: Mean per-letter frequency (fraction of Cyrillic letters) of every letter
#: that some twin lacks, in that language's own text.
RATES: dict[str, dict[str, float]] = {
    "bak": {"ғ": 0.0164, "ҙ": 0.0207, "ҡ": 0.0247, "ң": 0.0102, "ҫ": 0.0032, "ү": 0.0128, "һ": 0.0163, "ә": 0.0671, "ө": 0.0177},
    "bel": {"ы": 0.0437, "э": 0.0155, "ё": 0.0040, "і": 0.0528, "ў": 0.0211},
    "bul": {"й": 0.0058, "щ": 0.0052, "ъ": 0.0155, "ь": 0.0004, "ю": 0.0017, "я": 0.0170},
    "chv": {"ҫ": 0.0310, "ӑ": 0.0556, "ӗ": 0.0513, "ӳ": 0.0033},
    "kaz": {"і": 0.0418, "ғ": 0.0144, "қ": 0.0268, "ң": 0.0105, "ү": 0.0065, "ұ": 0.0096, "һ": 0.0003, "ә": 0.0054, "ө": 0.0111},
    "kir": {"ң": 0.0034, "ү": 0.0255, "ө": 0.0202},
    "kpv": {"і": 0.0113},
    "mhr": {"ҥ": 0.0058, "ӱ": 0.0103},
    "mkd": {"ѓ": 0.0012, "ѕ": 0.0001, "ј": 0.0176, "љ": 0.0004, "њ": 0.0031, "ќ": 0.0037, "џ": 0.0005},
    "rus": {"и": 0.0761, "щ": 0.0036, "ъ": 0.0005, "ы": 0.0196, "э": 0.0040, "ё": 0.0010},
    "sah": {"ҕ": 0.0097, "ң": 0.0068, "ү": 0.0270, "һ": 0.0141, "ө": 0.0145},
    "srp": {"ђ": 0.0023, "ј": 0.0331, "љ": 0.0046, "њ": 0.0072, "ћ": 0.0058, "џ": 0.0003},
    "tat": {"җ": 0.0039, "ң": 0.0100, "ү": 0.0118, "һ": 0.0044, "ә": 0.0602, "ө": 0.0086},
    "tyv": {"ң": 0.0116, "ү": 0.0122, "ө": 0.0071},
    "udm": {"ӝ": 0.0008, "ӟ": 0.0031, "ӥ": 0.0101, "ӵ": 0.0023},
    "ukr": {"и": 0.0606, "щ": 0.0056, "є": 0.0065, "і": 0.0567, "ї": 0.0060, "ґ": 0.0001},
}

_CYRILLIC = frozenset("".join(ALPHABETS.values()))

#: Variant spellings writers use interchangeably inside a group, folded
#: before counting. Turkic: Sakha text is often typed with ң for ҥ and ђ for
#: ҕ ("Бүгүңңү", "армияђа"), Tuvan with ҥ for ң ("ХҮННҮҤ") — unfolded, those
#: genuine rows read as Kyrgyz. Group-local on purpose: Mari writes ҥ and
#: Serbian ђ as letters of their own.
_TURKIC_FOLD = str.maketrans("ҥђ", "ңҕ")
_FOLD: dict[str, dict[int, str]] = dict.fromkeys(("tat", "bak", "kaz", "kir", "tyv", "chv", "sah"), _TURKIC_FOLD)


def _exclusive(owner: str, other: str) -> tuple[frozenset[str], float]:
    """Letters ``owner`` writes and ``other`` never does, with their summed
    typical rate in ``owner`` text."""
    letters = frozenset(ALPHABETS[owner]) - frozenset(ALPHABETS[other])
    return letters, sum(RATES[owner].get(c, 0.0) for c in letters)


#: label -> [(rival, rival's letters the label lacks + rate,
#:            label's letters the rival lacks + rate)]
_PAIRS: dict[str, list[tuple[str, tuple[frozenset[str], float], tuple[frozenset[str], float]]]] = {
    lang: [
        (rival, _exclusive(rival, lang), _exclusive(lang, rival))
        for group in TWIN_GROUPS
        if lang in group
        for rival in sorted(group - {lang})
        if _exclusive(rival, lang)[1] > 0
    ]
    for lang in ALPHABETS
}


def _word_text(lang: str, text: str) -> str:
    """The lowercased, group-folded words of ``text`` the featurizer reads
    as real words."""
    words = []
    for word in text.split():
        word = word.removeprefix("#")
        if word and classify_word(word) is None:
            words.append(word)
    return " ".join(words).lower().translate(_FOLD.get(lang, {}))


def contradicting_twin(lang: str, text: str, threshold: float = 0.5) -> str | None:
    """The twin language that ``text`` is mostly written in instead of its
    label ``lang`` (the strongest one when several qualify), or None when
    the label holds or cannot be judged."""
    pairs = _PAIRS.get(lang)
    if not pairs:
        return None
    lower = _word_text(lang, text)
    letters = sum(ch in _CYRILLIC for ch in lower)
    if letters == 0:
        return None
    best, best_share = None, threshold
    for rival, (theirs, their_rate), (ours, our_rate) in pairs:
        rival_share = sum(ch in theirs for ch in lower) / (their_rate * letters)
        if rival_share < best_share:
            continue
        own_share = sum(ch in ours for ch in lower) / (our_rate * letters) if our_rate > 0 else 0.0
        if own_share < threshold:
            best, best_share = rival, rival_share
    return best


def measure_rates(rows: list[tuple[str, str]], min_letters: int = 50) -> dict[str, dict[str, float]]:
    """Recompute :data:`RATES` from ``(lang, text)`` rows: for every language,
    the mean per-row frequency of each of its letters some twin lacks, over
    rows with at least ``min_letters`` Cyrillic letters."""
    needed = {
        lang: frozenset().union(*(_exclusive(lang, rival)[0] for g in TWIN_GROUPS if lang in g for rival in g - {lang}))
        for lang in ALPHABETS
    }
    sums: dict[str, dict[str, float]] = {lang: dict.fromkeys(needed[lang], 0.0) for lang in ALPHABETS}
    counts = dict.fromkeys(ALPHABETS, 0)
    for lang, text in rows:
        if lang not in needed:
            continue
        lower = _word_text(lang, text)
        letters = sum(ch in _CYRILLIC for ch in lower)
        if letters < min_letters:
            continue
        counts[lang] += 1
        for ch in needed[lang]:
            sums[lang][ch] += lower.count(ch) / letters
    return {
        lang: {ch: round(total / counts[lang], 4) for ch, total in sorted(sums[lang].items())}
        for lang in ALPHABETS
        if counts[lang]
    }
