"""Bit-exact Python mirror of spellman's Rust feature pipeline.

This module is the shared contract between training (Python) and inference
(Rust). Any change to the packing, hashing, or bucket extraction must be made
in BOTH implementations and validated against ``fixtures/hash_vectors.json``
(``uv run spellman-gen-fixtures`` regenerates it; ``cargo test`` verifies it).

Parity notes:
- Lowercasing uses ``str.lower()``; Rust uses per-``char`` ``to_lowercase()``.
  These agree on all Latin/Cyrillic text. (They differ on Greek final sigma,
  which none of the target languages use.)
- Word splitting: Rust splits on ``char::is_whitespace`` (Unicode White_Space);
  ``str.split()`` splits on ``str.isspace()`` which additionally accepts a few
  C0 controls. No target text contains those.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
U32_ONE = 1

BOW = 0x02  # \u{2} word-begin boundary marker
EOW = 0x03  # \u{3} word-end boundary marker
CP_BITS = 21
CP_MASK = (1 << CP_BITS) - 1

# Token-class canonicalization sentinels (private-use codepoints; a
# canonicalized word packs as [BOW, sentinel, EOW]). Must match
# src/features.rs.
SENTINEL_URL = 0xE001
SENTINEL_EMAIL = 0xE002
SENTINEL_MENTION = 0xE003
SENTINEL_NUM = 0xE004

_EMAIL_LOCAL = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._%+-")
_DOMAIN_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")


def _is_ascii_domain(s: str) -> bool:
    """ASCII alnum/hyphen/dot labels, non-empty, alphabetic TLD >= 2 chars."""
    if not s.isascii() or "." not in s:
        return False
    if not all(c in _DOMAIN_CHARS for c in s):
        return False
    labels = s.split(".")
    if any(not l for l in labels):
        return False
    tld = labels[-1]
    return len(tld) >= 2 and tld.isalpha()


def classify_word(word: str) -> int | None:
    """Sentinel codepoint for a canonicalizable word, else None (real word).

    Order matters and is part of the Rust/Python contract:
    mention -> URL prefix -> email -> digit-bearing ASCII -> bare domain.
    All rules are ASCII-only so Cyrillic dotted abbreviations and
    digit-bearing Cyrillic words always pass through as real words.
    """
    if len(word) > 1 and word.startswith("@"):
        return SENTINEL_MENTION
    lw = word.lower()
    if lw.startswith(("http://", "https://", "www.")):
        return SENTINEL_URL
    if "@" in word:
        parts = word.split("@")
        if len(parts) == 2 and parts[0] and all(c in _EMAIL_LOCAL for c in parts[0]) and _is_ascii_domain(parts[1]):
            return SENTINEL_EMAIL
    if word.isascii() and any(c in "0123456789" for c in word):
        return SENTINEL_NUM
    if _is_ascii_domain(word):
        return SENTINEL_URL
    return None

DEFAULT_SEED = 0x9E3779B9
MS_CONSTANT = 0x9E3779B97F4A7C15


@dataclass(frozen=True)
class FeatureConfig:
    n_min: int = 1
    n_max: int = 5


# ---------------------------------------------------------------------------
# Hash mixers (u32 arithmetic only, so the same math can run in numpy/torch)
# ---------------------------------------------------------------------------


def fmix32(h: int) -> int:
    """MurmurHash3-32 finalizer."""
    h &= MASK32
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & MASK32
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & MASK32
    h ^= h >> 16
    return h


def murmurhash2(k: int, seed: int) -> int:
    """whichlang's murmurhash2 on a u32 (bit-for-bit parity baseline)."""
    m = 0x5BD1E995
    k &= MASK32
    h = seed & MASK32
    k = (k * m) & MASK32
    k ^= k >> 24
    k = (k * m) & MASK32
    h = (h * m) & MASK32
    h ^= k
    h ^= h >> 13
    h = (h * m) & MASK32
    return h ^ (h >> 15)


def _rotl32(x: int, n: int) -> int:
    x &= MASK32
    return ((x << n) | (x >> (32 - n))) & MASK32


def hash_u64(key: int, hash_id: str, seed: int = DEFAULT_SEED) -> int:
    lo = key & MASK32
    hi = (key >> 32) & MASK32
    if hash_id == "fmix32":
        h = lo ^ ((seed * 0x85EBCA6B) & MASK32)
        h ^= (hi * 0xC2B2AE35) & MASK32
        return fmix32(h)
    if hash_id == "murmur2":
        seed_hi = _rotl32(seed, 16)
        return murmurhash2(lo ^ seed, seed) ^ murmurhash2(hi ^ seed_hi, seed_hi)
    if hash_id == "multiply_shift":
        top32 = ((key * MS_CONSTANT) & MASK64) >> 32
        return (top32 ^ seed) & MASK32
    raise ValueError(f"unknown hash id: {hash_id}")


def bucket_of(key: int, log2_d: int, hash_id: str, seed: int = DEFAULT_SEED) -> tuple[int, bool]:
    """(bucket, neg): bucket from the high bits, sign from bit 0 (disjoint)."""
    h = hash_u64(key, hash_id, seed)
    return h >> (32 - log2_d), bool(h & 1)


# ---------------------------------------------------------------------------
# N-gram extraction (must match Rust token_keys / bucket_tokens exactly)
# ---------------------------------------------------------------------------


def pack_ngram(chars: list[int]) -> int:
    """Reference per-window packing (kept for tests/training clarity).

    The rolling implementation in [`token_keys`] produces the identical
    multiset of keys; only emission order differs (position-major there,
    n-major here).
    """
    key = 0
    for c in chars:
        key = ((key << CP_BITS) | (c & CP_MASK)) & MASK64
    return key


_MASKS = {1: (1 << 21) - 1, 2: (1 << 42) - 1, 3: (1 << 63) - 1}
# Per-n domain-separation salt, XORed into every key: key ^ (n * ODD).
# Without it, u64 wrapping makes every 5-gram key identical to its suffix
# 4-gram key; the salt keeps the orders distinct. Must match Rust's N_TAG.
_N_TAG = [((n * 0x9E3779B97F4A7C15) & MASK64) for n in range(6)]


def token_keys(text: str, cfg: FeatureConfig = FeatureConfig()) -> list[int]:
    """Rolling position-major n-gram extraction (must match Rust exactly).

    One u64 register per word, shifted 21 bits per character; the n-gram
    keys ending at each character are masks of the register (n <= 3) or the
    full register (n >= 4, where windows wrap past 64 bits), XORed with the
    per-n salt. Words are classified first: URL/email/mention/number words
    pack as their class sentinel instead of their characters.
    """
    keys: list[int] = []
    for word in text.lower().split():
        if word.startswith("#"):
            word = word[1:]  # hashtag -> inner word (kept when it has letters)
        if not word:
            continue
        sentinel = classify_word(word)
        if sentinel is not None:
            chars = [BOW, sentinel, EOW]
        else:
            chars = [BOW] + [ord(ch) for ch in word] + [EOW]
        r = 0
        ln = 0
        for c in chars:
            r = ((r << CP_BITS) | (c & CP_MASK)) & MASK64
            ln += 1
            for n in range(cfg.n_min, min(cfg.n_max, ln) + 1):
                window = r & _MASKS[n] if n <= 3 else r
                keys.append(window ^ _N_TAG[n])
    return keys


def bucket_tokens(
    text: str,
    log2_d: int,
    hash_id: str = "fmix32",
    seed: int = DEFAULT_SEED,
    cfg: FeatureConfig = FeatureConfig(),
) -> list[tuple[int, bool]]:
    return [bucket_of(k, log2_d, hash_id, seed) for k in token_keys(text, cfg)]


# ---------------------------------------------------------------------------
# Vectorized batch extraction (bit-exact with token_keys/bucket_tokens)
#
# The rolling packer is sequential per word, but the register at position p
# is a fixed expression: r(p) = c[p] | c[p-1]<<21 | c[p-2]<<42 | c[p-3]<<63
# (uint64 wrapping drops everything shifted past 64 bits), so all five
# n-orders fall out of ONE shifted-OR over the whole stream. Word isolation
# is enforced with a word-id mask instead of a register reset. Word
# classification runs vectorized for the common cases; the rare
# @/./digit-bearing words fall back to the scalar classify_word (the
# reference path), keeping bit-exactness by construction.
# ---------------------------------------------------------------------------

import numpy as np

# Unicode White_Space property (matches Rust char::is_whitespace; the
# reference notes str.isspace additionally accepts 0x1C-0x1F, which no
# target text contains).
_WS_ARRAY = np.array(
    [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B),
     0x2028, 0x2029, 0x202F, 0x205F, 0x3000],
    dtype=np.uint32,
)


def _hash_batch_u64(keys: np.ndarray, hash_id: str, seed: int) -> np.ndarray:
    """Vectorized mirror of hash_u64 (u32/u64 wrapping arithmetic)."""
    k = keys.astype(np.uint64)
    lo = (k & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    hi = (k >> np.uint64(32)).astype(np.uint32)
    if hash_id == "fmix32":
        h = lo ^ np.uint32(seed * 0x85EBCA6B & 0xFFFFFFFF) ^ (hi * np.uint32(0xC2B2AE35))
        h ^= h >> np.uint32(16)
        h *= np.uint32(0x85EBCA6B)
        h ^= h >> np.uint32(13)
        h *= np.uint32(0xC2B2AE35)
        h ^= h >> np.uint32(16)
        return h
    if hash_id == "murmur2":
        def mm(kk: np.ndarray, sd: np.uint32) -> np.ndarray:
            m = np.uint32(0x5BD1E995)
            x = kk * m
            x ^= x >> np.uint32(24)
            x *= m
            h = sd * m
            h ^= x
            h ^= h >> np.uint32(13)
            h *= m
            return h ^ (h >> np.uint32(15))

        rot = np.uint32(seed << 16 & 0xFFFFFFFF | seed >> 16)
        return mm(lo ^ np.uint32(seed), np.uint32(seed)) ^ mm(hi ^ rot, rot)
    if hash_id == "multiply_shift":
        prod = (k * np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
        return ((prod >> np.uint64(32)) ^ np.uint64(seed)).astype(np.uint32)
    raise ValueError(f"unknown hash id: {hash_id}")


def bucket_tokens_flat(
    texts: list[str],
    log2_d: int,
    hash_id: str = "fmix32",
    seed: int = DEFAULT_SEED,
    cfg: FeatureConfig = FeatureConfig(),
    chunk_texts: int = 20_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch extraction of signed bucket tokens for many texts.

    Returns (buckets uint32[N], negs bool[N], offsets int64[len+1]) — the
    tokens of text i are buckets[offsets[i]:offsets[i+1]], in the exact
    encounter order of the reference bucket_tokens (position-major, n
    ascending within position).

    numpy >= 2 only: numpy 1.26 hits an IndexError in the stream assembly on
    large chunks (verified: identical intermediate arrays, divergent
    behavior). The pipeline standardizes on numpy 2; the only numpy<2
    corner left is the fasttext overlay, which uses the scalar path.
    """
    if np.lib.NumpyVersion(np.__version__) < "2.0.0":
        raise RuntimeError("bucket_tokens_flat requires numpy >= 2 (1.26 breaks on large chunks)")
    all_buckets: list[np.ndarray] = []
    all_negs: list[np.ndarray] = []
    offsets = np.zeros(len(texts) + 1, dtype=np.int64)
    n_min, n_max = cfg.n_min, min(cfg.n_max, 5)

    for t0 in range(0, len(texts), chunk_texts):
        chunk = texts[t0 : t0 + chunk_texts]

        # -- 1. Codepoint stream per text (utf-32-le = direct codepoints).
        # Lowercase up front exactly like the reference (text.lower() before
        # split) so classification and packing see the same codepoints,
        # including rare case-folding expansions. --
        streams, row_lens = [], []
        for t in chunk:
            b = t.lower().encode("utf-32-le", "surrogatepass")
            # Trailing separator: rows are concatenated into one stream and
            # would otherwise fuse their edge words into a single token.
            streams.append(np.concatenate((np.frombuffer(b, dtype=np.uint32), np.array([0x20], dtype=np.uint32))))
            row_lens.append(len(b) // 4 + 1)
        if not streams:
            continue
        cps = np.concatenate(streams)
        row_id_chars = np.repeat(np.arange(len(chunk), dtype=np.int64), row_lens)

        # -- 2. Word spans: maximal non-whitespace runs, one '#' stripped. --
        is_ws = np.isin(cps, _WS_ARRAY)
        idx = np.flatnonzero(~is_ws)
        if idx.size == 0:
            continue
        brk = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate(([idx[0]], idx[brk + 1]))
        ends = np.concatenate((idx[brk], [idx[-1]])) + 1  # exclusive
        hashed = cps[starts] == 0x23
        starts = starts + hashed
        lens = ends - starts
        keep = lens > 0
        starts, lens = starts[keep], lens[keep]
        word_row = row_id_chars[starts]

        # -- 3. Classification (vectorized; scalar fallback for candidates).--
        first = cps[starts]
        mention = (first == 0x40) & (lens > 1)
        has_at = np.add.reduceat(cps == 0x40, starts)
        has_digit = np.add.reduceat((cps >= 0x30) & (cps <= 0x39), starts)
        non_ascii = np.add.reduceat(cps > 0x7F, starts)

        # URL prefix, case-insensitive, on the first 8 codepoints.
        take = np.minimum(lens, 8)
        w8 = np.zeros((len(starts), 8), dtype=np.uint32)
        nz = take > 0
        seg = np.concatenate([np.arange(t, dtype=np.int64) for t in take[nz]]) if nz.any() else np.empty(0, np.int64)
        w8[np.repeat(np.flatnonzero(nz), take[nz]), seg] = cps[np.repeat(starts[nz], take[nz])]
        f8 = w8.copy()
        f8[(f8 >= 0x41) & (f8 <= 0x5A)] += np.uint32(0x20)

        def pref(pat: list[int]) -> np.ndarray:
            p = np.array(pat, dtype=np.uint32)
            return np.all(f8[:, : len(p)] == p, axis=1) & (take >= len(p))

        is_url = (
            pref([0x68, 0x74, 0x74, 0x70, 0x3A, 0x2F, 0x2F])       # http://
            | pref([0x68, 0x74, 0x74, 0x70, 0x73, 0x3A, 0x2F, 0x2F])  # https://
            | pref([0x77, 0x77, 0x77, 0x2E])                        # www.
        )
        is_num = has_digit & ~non_ascii

        flagged = has_at | np.add.reduceat(cps == 0x2E, starts) | has_digit | mention | is_url
        sentinels = np.zeros(len(starts), dtype=np.uint32)
        eff_len = lens.copy()
        for i in np.flatnonzero(flagged):
            word_str = "".join(map(chr, cps[starts[i] : starts[i] + lens[i]]))
            s = classify_word(word_str)
            if s is not None:
                sentinels[i] = s
                eff_len[i] = 1

        # -- 4. Wrapped stream: [BOW] + (sentinel | chars) + [EOW] per word. --
        wrapped_len = eff_len + 2
        word_base = np.concatenate(([0], np.cumsum(wrapped_len)[:-1]))
        out = np.empty(int(wrapped_len.sum()), dtype=np.uint32)
        wid = np.repeat(np.arange(len(starts), dtype=np.int64), wrapped_len)
        out[word_base] = BOW
        out[word_base + eff_len + 1] = EOW
        total_chars = int(eff_len.sum())
        seg_starts = np.concatenate(([0], np.cumsum(eff_len)[:-1]))
        within = np.arange(total_chars, dtype=np.int64) - np.repeat(seg_starts, eff_len)
        vals = cps[np.repeat(starts, eff_len) + within].astype(np.uint32)
        can = sentinels != 0
        if can.any():
            # Canonical words are always length-1: their interior slot in
            # the flattened vals array is exactly their segment start.
            vals[seg_starts[can]] = sentinels[can]
        out[np.repeat(word_base + 1, eff_len) + within] = vals
        stream_row = word_row[wid]

        # -- 5. Register + keys over the whole stream (vectorized). --
        c = out.astype(np.uint64)
        c1 = np.concatenate((np.zeros(1, np.uint64), c[:-1]))
        c2 = np.concatenate((np.zeros(2, np.uint64), c[:-2]))
        c3 = np.concatenate((np.zeros(3, np.uint64), c[:-3]))
        r = c | (c1 << np.uint64(21)) | (c2 << np.uint64(42)) | (c3 << np.uint64(63))
        pos = np.arange(len(out), dtype=np.int64)

        keys: list[np.ndarray] = []
        ord_p: list[np.ndarray] = []
        ord_n: list[np.ndarray] = []
        ord_row: list[np.ndarray] = []
        for n in range(n_min, n_max + 1):
            if n > 1:
                shifted = np.concatenate((np.full(n - 1, -1, dtype=wid.dtype), wid[: -(n - 1)]))
                valid = (wid == shifted) & (pos >= n - 1)
            else:
                valid = np.ones(len(out), dtype=bool)
            if n == 1:
                window = r & np.uint64(0x1FFFFF)
            elif n == 2:
                window = r & np.uint64(0x3FFFFFFFFFF)
            elif n == 3:
                window = r & np.uint64(0x7FFFFFFFFFFFFFFF)
            else:
                window = r  # n >= 4 windows wrap: key is the full register
            keys.append(window[valid] ^ np.uint64(_N_TAG[n]))
            ord_p.append(pos[valid])
            ord_n.append(np.full(int(valid.sum()), n, dtype=np.int8))
            ord_row.append(stream_row[valid])

        keys_all = np.concatenate(keys)
        p_all = np.concatenate(ord_p)
        n_all = np.concatenate(ord_n)
        row_all = np.concatenate(ord_row)
        # Reference order: (row, position, n).
        ord_idx = np.lexsort((n_all, p_all, row_all))

        h = _hash_batch_u64(keys_all[ord_idx], hash_id, seed)
        all_buckets.append((h >> np.uint32(32 - log2_d)).astype(np.uint32))
        all_negs.append((h & np.uint32(1)).astype(bool))
        per_row = np.zeros(len(chunk), dtype=np.int64)
        np.add.at(per_row, row_all, 1)
        offsets[t0 + 1 : t0 + 1 + len(chunk)] = offsets[t0] + np.cumsum(per_row)

    if all_buckets:
        return np.concatenate(all_buckets), np.concatenate(all_negs), offsets
    return np.empty(0, dtype=np.uint32), np.empty(0, dtype=bool), offsets


def assert_batch_parity(texts: list[str], log2_d: int = 17) -> None:
    """Self-test: bucket_tokens_flat must reproduce bucket_tokens exactly,
    including encounter order, for every text given."""
    buckets, negs, offsets = bucket_tokens_flat(texts, log2_d)
    for i, t in enumerate(texts):
        ref = bucket_tokens(t, log2_d)
        sl = slice(offsets[i], offsets[i + 1])
        assert list(map(int, buckets[sl])) == [b for b, _ in ref], f"bucket mismatch on {t!r}"
        assert list(map(bool, negs[sl])) == [n for _, n in ref], f"neg mismatch on {t!r}"


def to_signed_index(bucket: int, neg: bool, d: int) -> int:
    """Fold the sign into the index: [0, D] is +P, [D+1, 2D+1] is -P."""
    return bucket if not neg else d + 1 + bucket


# ---------------------------------------------------------------------------
# Shared language inventory (must match src/lang.rs column order)
# ---------------------------------------------------------------------------

LANGUAGES: list[str] = [
    # Cyrillic group
    "rus", "ukr", "bel", "bul", "mkd", "srp", "kaz", "kir", "tgk", "uzn",
    "tat", "bak", "chv", "sah", "tyv", "mon", "oss", "che", "udm", "mhr", "kpv",
    # Latin group
    "eng", "spa", "fra", "por", "deu",
    # Direct script mapping
    "cmn", "jpn", "hin", "ara",
]


def load_lang_examples() -> dict[str, str]:
    """Multilingual snippets used for the Rust/Python parity fixture."""
    return {
        "rus": "Съешь ещё этих мягких французских булок, да выпей чаю",
        "ukr": "Швидкість світла у вакуумі є фундаментальною фізичною константою",
        "bel": "У Іўі худы жвавы чорт у зялёнай камізэльцы пабег пад'есці фасоляці",
        "bul": "Под южното дърво, цъфтящо в синьо, пееше топъл глас",
        "mkd": "Ѓулапче со ќефче на шамче рој на гнездо кај што жеже",
        "srp": "Љубазни фењерџија чађавог лица хоће да ми покаже штосу",
        "kaz": "Абдыгапардың әжейі өңірлі ғажайып үй құдықын шолып жүр",
        "kir": "Атаман Домукайдын уулу Күнкүжүр жөнүндө эмне айтылбайт",
        "tgk": "Сафар Ҷангалов ғайратнок ҳимматписару ношиноса шуда истодааст",
        "uzn": "Аёл кўкракда ёдгорлик белгилари тўғрисида маълумот беради",
        "eng": "The quick brown fox jumps over the lazy dog; pack my box!",
        "spa": "El veloz murciélago hindú comía feliz cardillo y kiwi",
        "fra": "Portez ce vieux whisky au juge blond qui fume",
        "deu": "Victor jagt zwölf Boxkämpfer quer über den großen Sylter Deich",
        "mixed": "Привет world мир 123 !",
        "short": "да",
        "punct": "!!! ... ???",
        # Canonicalization cases (token-class sentinels).
        "wild_url": "все мы помним https://t.co/3Kr7yzeYLC и www.grozny-inform.ru",
        "wild_email": "напиши на test.mail+tag@example.com или a@b.co",
        "wild_mention": "@daria_karapet привет @nick123 пока",
        "wild_num": "в 2020 году 3.5.2 релиз 100px COVID19 обновление",
        "wild_hashtag": "#красноярск #2020 #URGENT пушка",
        "wild_abbrev": "т.е. и т.д. и U.S.A. и миллион2020",
        "wild_HTTP": "HTTP://EXAMPLE.COM/PATH заглавные",
        "wild_mixed": "@user RT #новости https://t.co/xYz123 тест@mail.ru 100%",
    }
