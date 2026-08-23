//! spellman-language — the language inventory shared by the spellman stack.
//!
//! One macro table ([`languages!`]) defines every language: the variant
//! (named by its ISO 639-3 code), ISO 639-1 / NLLB-200 / Whisper codes where
//! they exist, the English name, the primary script, localized names, and a
//! flag emoji.
//! `Lang`, [`Lang::ALL`] and every accessor are generated from that table;
//! adding a language is adding one row. Variant order is load-bearing: it is
//! the model-column order of the detector's folded score table.
//!
//! The crate is dependency-free (std only), so the detector, training-side
//! tooling and labeling utilities can all share one inventory.

use std::fmt;
use std::str::FromStr;

/// Writing system of a language (or of a single character via
/// [`char_script`]). `Kana` covers hiragana and katakana — the script unique
/// to Japanese among the supported languages.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum Script {
    Latin,
    Cyrillic,
    Devanagari,
    Arabic,
    Han,
    Kana,
}

/// Script of a single codepoint, restricted to the scripts this inventory
/// cares about; `None` for anything else (digits, punctuation, other
/// scripts).
pub const fn char_script(c: char) -> Option<Script> {
    let script = match c as u32 {
        0x41..=0x5A | 0x61..=0x7A | 0xC0..=0x24F | 0x1E00..=0x1EFF => Script::Latin,
        0x0400..=0x052F | 0x2DE0..=0x2DFF | 0xA640..=0xA69F => Script::Cyrillic,
        0x0900..=0x097F => Script::Devanagari,
        0x0600..=0x06FF | 0x0750..=0x077F | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF => Script::Arabic,
        0x3040..=0x309F | 0x30A0..=0x30FF | 0x31F0..=0x31FF | 0xFF66..=0xFF9D => Script::Kana,
        0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF => Script::Han,
        _ => return None,
    };
    Some(script)
}

/// Define a language inventory from one row per language:
///
/// ```ignore
/// languages! {
///     Rus {
///         iso1:    Some("ru"),
///         iso3:    "rus",
///         nllb:    Some("rus_Cyrl"),
///         whisper: Some("russian"),
///         name:    "Russian",
///         script:  Script::Cyrillic,
///         flag:    "🇷🇺",
///         names:   { Rus: "Русский" },
///     },
/// }
/// ```
///
/// `iso1` / `nllb` / `whisper` are `Option<&'static str>` — languages the
/// standard doesn't cover carry `None`. `names` maps a display language to a
/// localized name (empty `{}` when none); unknown display languages fall
/// back to the English `name`. `flag` is the flag emoji of the primary
/// speaker country; languages that exist only as republic languages within
/// Russia conventionally take 🇷🇺.
///
/// Generates the `Lang` enum (variants in table order = model-column
/// order), `Lang::ALL`, `NUM_LANGS`, and the accessors. `code()` is the
/// ISO 639-3 code — also the variant name lowercased.
#[macro_export]
macro_rules! languages {
    (
        $(
            $(#[$doc:meta])*
            $variant:ident {
                iso1:    $iso1:expr,
                iso3:    $iso3:expr,
                nllb:    $nllb:expr,
                whisper: $whisper:expr,
                name:    $name:expr,
                script:  $script:expr,
                flag:    $flag:expr,
                names:   { $( $display:ident : $local:literal ),* $(,)? },
            }
        ),* $(,)?
    ) => {
        /// Supported language. Variant order is the detector's model-column
        /// order; the variant name is the ISO 639-3 code.
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[allow(missing_docs)]
        pub enum Lang {
            $(
                $(#[$doc])*
                $variant,
            )*
        }

        /// Number of languages in the inventory.
        pub const NUM_LANGS: usize = 0 $( + { let _ = stringify!($variant); 1 })*;

        #[allow(missing_docs)]
        impl Lang {
            /// All languages, in model-column order.
            pub const ALL: [Lang; NUM_LANGS] = [$(Lang::$variant),*];

            /// ISO 639-3 code.
            pub const fn code(self) -> &'static str {
                match self { $(Lang::$variant => $iso3,)* }
            }

            /// ISO 639-3 code (alias of [`Lang::code`]).
            pub const fn iso_639_3(self) -> &'static str {
                self.code()
            }

            /// ISO 639-1 code, when the language has one.
            pub const fn iso_639_1(self) -> Option<&'static str> {
                match self { $(Lang::$variant => $iso1,)* }
            }

            /// NLLB-200 / FLORES-200 code when the language is covered —
            /// otherwise a FLORES-style script-qualified code for the
            /// variety (see the table rows).
            pub const fn nllb(self) -> Option<&'static str> {
                match self { $(Lang::$variant => $nllb,)* }
            }

            /// Whisper language token, when Whisper supports the language.
            pub const fn whisper(self) -> Option<&'static str> {
                match self { $(Lang::$variant => $whisper,)* }
            }

            /// English name.
            pub const fn name(self) -> &'static str {
                match self { $(Lang::$variant => $name,)* }
            }

            /// Primary script of the language.
            pub const fn script(self) -> Script {
                match self { $(Lang::$variant => $script,)* }
            }

            /// Flag emoji of the primary speaker country; republic
            /// languages within Russia conventionally take 🇷🇺.
            pub const fn flag(self) -> &'static str {
                match self { $(Lang::$variant => $flag,)* }
            }

            /// Position in [`Lang::ALL`] (the model-column index).
            pub const fn index(self) -> usize {
                self as usize
            }

            /// Localized name for a display language, falling back to the
            /// English name when no localization is recorded.
            pub const fn name_in(self, display: Lang) -> &'static str {
                match (self, display) {
                    $(
                        $((Lang::$variant, Lang::$display) => $local,)*
                    )*
                    _ => self.name(),
                }
            }
        }
    };
}

crate::languages! {
    // Cyrillic script group (model columns 0..21)
    Rus {
        iso1:    Some("ru"),
        iso3:    "rus",
        nllb:    Some("rus_Cyrl"),
        whisper: Some("russian"),
        name:    "Russian",
        script:  Script::Cyrillic,
        flag:    "🇷🇺",
        names:   { Rus: "Русский" },
    },
    Ukr {
        iso1:    Some("uk"),
        iso3:    "ukr",
        nllb:    Some("ukr_Cyrl"),
        whisper: Some("ukrainian"),
        name:    "Ukrainian",
        script:  Script::Cyrillic,
        flag:    "🇺🇦",
        names:   { Rus: "Украинский" },
    },
    Bel {
        iso1:    Some("be"),
        iso3:    "bel",
        nllb:    Some("bel_Cyrl"),
        whisper: Some("belarusian"),
        name:    "Belarusian",
        script:  Script::Cyrillic,
        flag:    "🇧🇾",
        names:   { Rus: "Белорусский" },
    },
    Bul {
        iso1:    Some("bg"),
        iso3:    "bul",
        nllb:    Some("bul_Cyrl"),
        whisper: Some("bulgarian"),
        name:    "Bulgarian",
        script:  Script::Cyrillic,
        flag:    "🇧🇬",
        names:   { Rus: "Болгарский" },
    },
    Mkd {
        iso1:    Some("mk"),
        iso3:    "mkd",
        nllb:    Some("mkd_Cyrl"),
        whisper: Some("macedonian"),
        name:    "Macedonian",
        script:  Script::Cyrillic,
        flag:    "🇲🇰",
        names:   { Rus: "Македонский" },
    },
    Srp {
        iso1:    Some("sr"),
        iso3:    "srp",
        nllb:    Some("srp_Cyrl"),
        whisper: Some("serbian"),
        name:    "Serbian",
        script:  Script::Cyrillic,
        flag:    "🇷🇸",
        names:   { Rus: "Сербский" },
    },
    Kaz {
        iso1:    Some("kk"),
        iso3:    "kaz",
        nllb:    Some("kaz_Cyrl"),
        whisper: Some("kazakh"),
        name:    "Kazakh",
        script:  Script::Cyrillic,
        flag:    "🇰🇿",
        names:   { Rus: "Казахский" },
    },
    Kir {
        iso1:    Some("ky"),
        iso3:    "kir",
        nllb:    Some("kir_Cyrl"),
        // Whisper's 99-language list has no Kyrgyz.
        whisper: None,
        name:    "Kyrgyz",
        script:  Script::Cyrillic,
        flag:    "🇰🇬",
        names:   { Rus: "Киргизский" },
    },
    Tgk {
        iso1:    Some("tg"),
        iso3:    "tgk",
        nllb:    Some("tgk_Cyrl"),
        whisper: Some("tajik"),
        name:    "Tajik",
        script:  Script::Cyrillic,
        flag:    "🇹🇯",
        names:   { Rus: "Таджикский" },
    },
    Uzn {
        iso1:    Some("uz"),
        iso3:    "uzn",
        // FLORES-200 ships Uzbek only as uzn_Latn; this row is the Cyrillic
        // variety, so the code stays script-qualified.
        nllb:    Some("uzn_Cyrl"),
        whisper: Some("uzbek"),
        name:    "Uzbek",
        script:  Script::Cyrillic,
        flag:    "🇺🇿",
        names:   { Rus: "Узбекский" },
    },
    Tat {
        iso1:    Some("tt"),
        iso3:    "tat",
        nllb:    Some("tat_Cyrl"),
        whisper: Some("tatar"),
        name:    "Tatar",
        script:  Script::Cyrillic,
        // Tatarstan — republic within Russia.
        flag:    "🇷🇺",
        names:   { Rus: "Татарский" },
    },
    Bak {
        iso1:    Some("ba"),
        iso3:    "bak",
        nllb:    Some("bak_Cyrl"),
        whisper: Some("bashkir"),
        name:    "Bashkir",
        script:  Script::Cyrillic,
        // Bashkortostan — republic within Russia.
        flag:    "🇷🇺",
        names:   { Rus: "Башкирский" },
    },
    Chv {
        iso1:    Some("cv"),
        iso3:    "chv",
        nllb:    Some("chv_Cyrl"),
        whisper: None,
        name:    "Chuvash",
        script:  Script::Cyrillic,
        // Chuvashia — republic within Russia.
        flag:    "🇷🇺",
        names:   { Rus: "Чувашский" },
    },
    Sah {
        iso1:    None, // no ISO 639-1 code; "sah" is the 639-3 subtag
        iso3:    "sah",
        nllb:    Some("sah_Cyrl"),
        whisper: None,
        name:    "Sakha",
        script:  Script::Cyrillic,
        // Sakha (Yakutia) — republic within Russia.
        flag:    "🇷🇺",
        names:   { Rus: "Якутский" },
    },
    Tyv {
        iso1:    None, // no ISO 639-1 code
        iso3:    "tyv",
        nllb:    Some("tyv_Cyrl"),
        whisper: None,
        name:    "Tuvan",
        script:  Script::Cyrillic,
        // Tuva — republic within Russia.
        flag:    "🇷🇺",
        names:   { Rus: "Тувинский" },
    },
    Mon {
        // NLLB follows GlotLID: Halh Mongolian `khk_Cyrl`.
        iso1:    Some("mn"),
        iso3:    "mon",
        nllb:    Some("khk_Cyrl"),
        whisper: Some("mongolian"),
        name:    "Mongolian",
        script:  Script::Cyrillic,
        flag:    "🇲🇳",
        names:   { Rus: "Монгольский" },
    },
    Oss {
        iso1:    Some("os"),
        iso3:    "oss",
        nllb:    Some("oss_Cyrl"),
        whisper: None,
        name:    "Ossetian",
        script:  Script::Cyrillic,
        // Most speakers live in North Ossetia–Alania (Russia), not Georgia.
        flag:    "🇷🇺",
        names:   { Rus: "Осетинский" },
    },
    Che {
        iso1:    Some("ce"),
        iso3:    "che",
        nllb:    Some("che_Cyrl"),
        whisper: None,
        name:    "Chechen",
        script:  Script::Cyrillic,
        // Chechnya — republic within Russia.
        flag:    "🇷🇺",
        names:   { Rus: "Чеченский" },
    },
    Udm {
        iso1:    None, // no ISO 639-1 code
        iso3:    "udm",
        nllb:    Some("udm_Cyrl"),
        whisper: None,
        name:    "Udmurt",
        script:  Script::Cyrillic,
        // Udmurtia — republic within Russia.
        flag:    "🇷🇺",
        names:   { Rus: "Удмуртский" },
    },
    Mhr {
        iso1:    None, // no ISO 639-1 code
        iso3:    "mhr",
        // Not covered by NLLB-200/FLORES-200; mari_Cyrl is the FLORES-style
        // code for Mari, not mhr_Cyrl.
        nllb:    Some("mari_Cyrl"),
        whisper: None,
        name:    "Mari",
        script:  Script::Cyrillic,
        // Mari El — republic within Russia.
        flag:    "🇷🇺",
        names:   { Rus: "Марийский" },
    },
    Kpv {
        iso1:    None, // Komi-Zyrian proper; "kv" is the Komi macrolanguage
        iso3:    "kpv",
        nllb:    Some("kpv_Cyrl"),
        whisper: None,
        name:    "Komi",
        script:  Script::Cyrillic,
        // Komi Republic — republic within Russia.
        flag:    "🇷🇺",
        names:   { Rus: "Коми" },
    },
    // Latin script group (model columns 21..26)
    Eng {
        iso1:    Some("en"),
        iso3:    "eng",
        nllb:    Some("eng_Latn"),
        whisper: Some("english"),
        name:    "English",
        script:  Script::Latin,
        flag:    "🇬🇧",
        names:   { Rus: "Английский" },
    },
    Spa {
        iso1:    Some("es"),
        iso3:    "spa",
        nllb:    Some("spa_Latn"),
        whisper: Some("spanish"),
        name:    "Spanish",
        script:  Script::Latin,
        flag:    "🇪🇸",
        names:   { Rus: "Испанский" },
    },
    Fra {
        iso1:    Some("fr"),
        iso3:    "fra",
        nllb:    Some("fra_Latn"),
        whisper: Some("french"),
        name:    "French",
        script:  Script::Latin,
        flag:    "🇫🇷",
        names:   { Rus: "Французский" },
    },
    Por {
        iso1:    Some("pt"),
        iso3:    "por",
        nllb:    Some("por_Latn"),
        whisper: Some("portuguese"),
        name:    "Portuguese",
        script:  Script::Latin,
        flag:    "🇵🇹",
        names:   { Rus: "Португальский" },
    },
    Deu {
        iso1:    Some("de"),
        iso3:    "deu",
        nllb:    Some("deu_Latn"),
        whisper: Some("german"),
        name:    "German",
        script:  Script::Latin,
        flag:    "🇩🇪",
        names:   { Rus: "Немецкий" },
    },
    // Direct-script languages (model columns 26..30; resolved by the router,
    // never a trained argmax target)
    Cmn {
        iso1:    Some("zh"),
        iso3:    "cmn",
        // FLORES-200 uses zho_Hans; cmn_Hans is the individual-language
        // (Mandarin) equivalent.
        nllb:    Some("cmn_Hans"),
        whisper: Some("chinese"),
        name:    "Mandarin",
        script:  Script::Han,
        flag:    "🇨🇳",
        names:   { Rus: "Китайский" },
    },
    Jpn {
        iso1:    Some("ja"),
        iso3:    "jpn",
        nllb:    Some("jpn_Jpan"),
        whisper: Some("japanese"),
        name:    "Japanese",
        script:  Script::Kana, // kana presence is the Japanese differentiator
        flag:    "🇯🇵",
        names:   { Rus: "Японский" },
    },
    Hin {
        iso1:    Some("hi"),
        iso3:    "hin",
        nllb:    Some("hin_Deva"),
        whisper: Some("hindi"),
        name:    "Hindi",
        script:  Script::Devanagari,
        flag:    "🇮🇳",
        names:   { Rus: "Хинди" },
    },
    Ara {
        iso1:    Some("ar"),
        iso3:    "ara",
        // FLORES-200 uses arb_Arab (Standard Arabic); ara_Arab is the
        // macrolanguage-code equivalent.
        nllb:    Some("ara_Arab"),
        whisper: Some("arabic"),
        name:    "Arabic",
        script:  Script::Arabic,
        flag:    "🇦🇪",
        names:   { Rus: "Арабский" },
    },
}

/// Failed to parse a [`Lang`] from a string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownLang(pub String);

impl fmt::Display for UnknownLang {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown language code: {}", self.0)
    }
}

impl std::error::Error for UnknownLang {}

impl FromStr for Lang {
    type Err = UnknownLang;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Lang::ALL
            .iter()
            .find(|lang| lang.code().eq_ignore_ascii_case(s))
            .copied()
            .ok_or_else(|| UnknownLang(s.to_owned()))
    }
}

impl fmt::Display for Lang {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_round_trip() {
        for lang in Lang::ALL {
            let parsed: Lang = lang.code().parse().unwrap();
            assert_eq!(parsed, lang);
            assert_eq!(lang.to_string(), lang.code());
        }
        assert_eq!(NUM_LANGS, Lang::ALL.len());
        assert_eq!(Lang::Rus.index(), 0);
        assert_eq!(Lang::Eng.index(), 21);
        assert_eq!(Lang::Ara.index(), 29);
    }

    #[test]
    fn code_tables() {
        assert_eq!(Lang::Rus.iso_639_1(), Some("ru"));
        assert_eq!(Lang::Kaz.iso_639_1(), Some("kk"));
        // Languages without an ISO 639-1 code carry None, not a 639-3 echo.
        for lang in [Lang::Sah, Lang::Tyv, Lang::Udm, Lang::Mhr, Lang::Kpv] {
            assert_eq!(lang.iso_639_1(), None, "{}", lang.code());
        }
        assert_eq!(Lang::Rus.nllb(), Some("rus_Cyrl"));
        assert_eq!(Lang::Mon.nllb(), Some("khk_Cyrl"));
        assert_eq!(Lang::Mhr.nllb(), Some("mari_Cyrl"));
        assert_eq!(Lang::Cmn.nllb(), Some("cmn_Hans"));
        // Whisper covers the big Turkic languages — Kyrgyz excepted —
        // and none of the smaller ones.
        assert_eq!(Lang::Bak.whisper(), Some("bashkir"));
        assert_eq!(Lang::Tgk.whisper(), Some("tajik"));
        assert_eq!(Lang::Kir.whisper(), None);
        assert_eq!(Lang::Chv.whisper(), None);
        assert_eq!(Lang::Sah.whisper(), None);
    }

    #[test]
    fn flags_follow_the_primary_country() {
        assert_eq!(Lang::Rus.flag(), "🇷🇺");
        assert_eq!(Lang::Ukr.flag(), "🇺🇦");
        assert_eq!(Lang::Uzn.flag(), "🇺🇿");
        assert_eq!(Lang::Cmn.flag(), "🇨🇳");
        // Republic languages within Russia take the Russian flag.
        assert_eq!(Lang::Tat.flag(), "🇷🇺");
        assert_eq!(Lang::Oss.flag(), "🇷🇺");
        // Every flag is exactly one regional-indicator pair.
        for lang in Lang::ALL {
            assert_eq!(lang.flag().chars().count(), 2, "{}", lang.code());
        }
    }

    #[test]
    fn localized_names_fall_back_to_english() {
        assert_eq!(Lang::Kaz.name_in(Lang::Rus), "Казахский");
        assert_eq!(Lang::Kaz.name_in(Lang::Eng), "Kazakh");
        // No Russian localization recorded for a hypothetical display lang:
        // any unrecorded display language falls back to the English name.
        assert_eq!(Lang::Rus.name_in(Lang::Srp), "Russian");
        assert_eq!(Lang::Rus.name_in(Lang::Rus), "Русский");
    }

    #[test]
    fn scripts_partition_the_inventory() {
        for lang in Lang::ALL[..21].iter() {
            assert_eq!(lang.script(), Script::Cyrillic, "{}", lang.code());
        }
        for lang in Lang::ALL[21..26].iter() {
            assert_eq!(lang.script(), Script::Latin, "{}", lang.code());
        }
        assert_eq!(Lang::Cmn.script(), Script::Han);
        assert_eq!(Lang::Jpn.script(), Script::Kana);
        assert_eq!(Lang::Hin.script(), Script::Devanagari);
        assert_eq!(Lang::Ara.script(), Script::Arabic);
    }

    #[test]
    fn char_script_ranges() {
        assert_eq!(char_script('а'), Some(Script::Cyrillic));
        assert_eq!(char_script('ѣ'), Some(Script::Cyrillic)); // 0x0463
        assert_eq!(char_script('z'), Some(Script::Latin));
        assert_eq!(char_script('ö'), Some(Script::Latin));
        assert_eq!(char_script('ひ'), Some(Script::Kana));
        assert_eq!(char_script('ア'), Some(Script::Kana));
        assert_eq!(char_script('北'), Some(Script::Han));
        assert_eq!(char_script('क'), Some(Script::Devanagari));
        assert_eq!(char_script('م'), Some(Script::Arabic));
        assert_eq!(char_script('7'), None);
        assert_eq!(char_script('!'), None);
    }
}
