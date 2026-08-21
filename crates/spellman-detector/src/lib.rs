//! spellman — Cyrillic-optimized language detection.
//!
//! General detector for the G10 world languages plus ~20 Cyrillic-script
//! languages, built to beat existing detectors exactly where they are weakest
//! (closely-related Cyrillic pairs: ru/be/uk, bg/mk/sr, kk/ky/tt/ba).
//!
//! The language inventory itself lives in the `spellman-language` crate:
//! one macro table holds the ISO 639-1/639-3, NLLB-200 and Whisper codes,
//! names and scripts, and this crate re-exports [`Lang`] from it.
//!
//! Architecture:
//! - lowercase char n-grams (1..=5, order-tagged so 4- and 5-gram windows
//!   stay distinct despite the u64 wrap) hashed with signed feature hashing
//!   (fmix32 primary) into `2^log2_d` buckets;
//! - a fastText-style model (averaged n-gram embeddings + linear head) whose
//!   algebraic fold `P = E·W` turns scoring into table lookups — see
//!   [`model`];
//! - script routing (see [`route`]): unique-script languages (jpn/cmn/hin/ara)
//!   never touch a model; Latin and Cyrillic text goes to the merged
//!   30-class table;
//! - all inference runs through compiled svod execution plans:
//!   [`jit::SingleDetector`] (B=1 baked into the plan — fully static
//!   kernels) and [`jit::BulkDetector`] (constant K, rebindable batch,
//!   zero-copy featurization straight into the plan's input buffer).
//!   The graph is pure fp16 end to end; f32 conversion happens once at
//!   host read-out.

pub mod features;
pub mod hash;
pub mod hub;
pub mod jit;
pub mod model;
pub mod route;

pub use jit::{BulkDetector, SingleDetector};
pub use route::{Route, ScriptGroup};

// The language inventory is re-exported wholesale so callers never need a
// direct `spellman-language` dependency: two versions of that crate in one
// dependency graph would make `Lang` distinct, incompatible types across
// the boundary. Every language type the detector's API leaks —
// `Detection.lang`, `Lang::script`, `FromStr::Err`, … — resolves through
// these aliases, and `spellman_detector::Lang` is the one type to use.
/// The table macro, re-exported so inventory definitions also stay
/// single-sourced (see the `spellman-language` crate).
pub use spellman_language::languages;
pub use spellman_language::{Lang, NUM_LANGS, Script, UnknownLang, char_script};

/// A single detection result.
#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    /// Best language; `None` when the router found no letters from a supported
    /// script (digits/punctuation-only input).
    pub lang: Option<Lang>,
    /// Softmax confidence of the winning class (1.0 for script-routed
    /// languages, 0.0 when `lang` is `None`).
    pub confidence: f32,
    /// True when `confidence` is below the model's calibrated threshold θ —
    /// the caller should treat the detection as uncertain.
    pub is_uncertain: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The re-export surface must cover everything the public API leaks;
    /// this test resolves each item through `spellman_detector::` paths
    /// only, so removing one fails here before it fails downstream.
    #[test]
    fn language_inventory_reexports_are_complete() {
        let lang: Lang = "kaz".parse().unwrap();
        assert_eq!(lang.name_in(Lang::Rus), "Казахский");
        assert_eq!(lang.script(), Script::Cyrillic);
        assert_eq!(char_script('қ'), Some(Script::Cyrillic));
        assert_eq!(Lang::ALL.len(), NUM_LANGS);
        let Err(err) = "nope".parse::<Lang>() else {
            panic!("unknown code parsed")
        };
        assert!(matches!(err, UnknownLang(_)));
    }
}
