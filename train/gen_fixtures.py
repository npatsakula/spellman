"""Regenerate the Rust↔Python parity fixture (hash_vectors.json).

Run:  cd train && uv run spellman-gen-fixtures
Then: cargo test  (tests/hash_vectors.rs verifies the Rust implementation
against the regenerated fixture).
"""

from __future__ import annotations

import json
from pathlib import Path

from spellman_features import (
    DEFAULT_SEED,
    FeatureConfig,
    LANGUAGES,
    bucket_tokens,
    load_lang_examples,
    token_keys,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "crates"
    / "spellman-detector"
    / "fixtures"
    / "hash_vectors.json"
)

HASH_IDS = ["fmix32", "murmur2", "multiply_shift"]
LOG2_D_VALUES = [12, 17]


def main() -> None:
    cases = []
    examples = load_lang_examples()
    texts = [examples[code] for code in LANGUAGES if code in examples]
    texts += [examples[k] for k in ("mixed", "short", "punct")]
    texts += [examples[k] for k in ("wild_url", "wild_email", "wild_mention", "wild_num", "wild_hashtag", "wild_abbrev", "wild_HTTP", "wild_mixed")]

    for text in texts:
        keys = token_keys(text, FeatureConfig())
        for hash_id in HASH_IDS:
            for log2_d in LOG2_D_VALUES:
                toks = bucket_tokens(text, log2_d, hash_id, DEFAULT_SEED)
                cases.append(
                    {
                        "text": text,
                        "hash_id": hash_id,
                        "seed": DEFAULT_SEED,
                        "n_min": 1,
                        "n_max": 5,
                        "log2_d": log2_d,
                        # u64 keys exceed 2^53 — stored as decimal strings.
                        "keys": [str(k) for k in keys],
                        "buckets": [t[0] for t in toks],
                        "negs": [int(t[1]) for t in toks],
                    }
                )

    # One non-default seed case to pin the seed handling.
    text = examples["rus"]
    keys = token_keys(text, FeatureConfig())
    toks = bucket_tokens(text, 17, "fmix32", 12345)
    cases.append(
        {
            "text": text,
            "hash_id": "fmix32",
            "seed": 12345,
            "n_min": 1,
            "n_max": 5,
            "log2_d": 17,
            "keys": [str(k) for k in keys],
            "buckets": [t[0] for t in toks],
            "negs": [int(t[1]) for t in toks],
        }
    )

    fixture = {
        "note": "Rust↔Python feature parity contract; regenerate with uv run spellman-gen-fixtures",
        "cases": cases,
    }
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(fixture, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(cases)} cases to {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
