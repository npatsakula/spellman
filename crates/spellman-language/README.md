# spellman-language

The language inventory shared by the [spellman](https://github.com/npatsakula/spellman)
stack — a dependency-free (std-only) crate generated from a single macro table.

One macro table (`languages!`) defines every language: the variant (named by
its ISO 639-3 code), ISO 639-1 / NLLB-200 codes where they exist,
the English name, the primary script, localized names, and a flag emoji.
`Lang`, `Lang::ALL` and every accessor are generated from that table; adding a
language is adding one row. Variant order is load-bearing: it is the
model-column order of the detector's folded score table. Whisper codes are
deliberately not stored — for every language of this inventory that Whisper
supports, its code IS the ISO 639-1 code.

```rust
use std::str::FromStr;
use spellman_language::{char_script, Lang, Script};

let rus = Lang::from_str("rus").unwrap(); // parse by ISO 639-3 code
assert_eq!(rus.iso_639_1(), Some("ru"));
assert_eq!(rus.nllb(), Some("rus_Cyrl"));
assert_eq!(rus.name(), "Russian");
assert_eq!(rus.script(), Script::Cyrillic);
assert_eq!(char_script('я'), Some(Script::Cyrillic));

// Reverse lookups across code systems (case-insensitive):
assert_eq!(Lang::from_iso_639_1("ru"), Some(rus));
assert_eq!(Lang::from_nllb("rus_Cyrl"), Some(rus));

assert_eq!(Lang::ALL.len(), spellman_language::NUM_LANGS);
```

The detector crate re-exports this inventory — users of `spellman-detector`
should not depend on `spellman-language` directly.
