//! Script routing: which model (if any) scores a document.
//!
//! Script classification itself lives in `spellman-language`
//! (`char_script`); this module maps scripts onto the detector's model
//! layout. Scripts with exactly one supported language (kana→jpn, Han→cmn,
//! Devanagari→hin, Arabic→ara) resolve directly and never touch a model;
//! Latin and Cyrillic text goes to the corresponding script-group columns
//! of the folded score table.

use spellman_language::{Lang, Script, NUM_LANGS};

/// Which model columns score a script's text.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum ScriptGroup {
    Cyrillic,
    Latin,
    /// Scripts with exactly one supported language; never reaches a model.
    Direct,
}

impl ScriptGroup {
    /// Languages whose columns the group model owns, in column order.
    pub fn languages(self) -> &'static [Lang] {
        match self {
            ScriptGroup::Cyrillic => &Lang::ALL[..21],
            ScriptGroup::Latin => &Lang::ALL[21..26],
            ScriptGroup::Direct => &Lang::ALL[26..NUM_LANGS],
        }
    }

    /// Contiguous model-column range owned by this group. The folded table's
    /// group rows are `row[column_range()]` — the scoring loop touches only
    /// these columns.
    pub const fn column_range(self) -> std::ops::Range<usize> {
        match self {
            ScriptGroup::Cyrillic => 0..21,
            ScriptGroup::Latin => 21..26,
            ScriptGroup::Direct => 26..NUM_LANGS,
        }
    }
}

/// Group owning a language's model columns, derived from its script.
pub const fn group_of(lang: Lang) -> ScriptGroup {
    match lang.script() {
        Script::Cyrillic => ScriptGroup::Cyrillic,
        Script::Latin => ScriptGroup::Latin,
        _ => ScriptGroup::Direct,
    }
}

/// Where the router sends a piece of text.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Route {
    /// Run the script-group model.
    Group(ScriptGroup),
    /// Script uniquely identifies the language.
    Direct(Lang),
    /// No letters from a supported script were found.
    Unknown,
}

/// Route text by dominant script. Kana is decisive for Japanese even when
/// kanji dominates the letter count; otherwise the script with the most
/// letters wins. Mixed-script ties go to the later candidate in declaration
/// order (Han > Cyrillic > Latin): a document split between scripts more
/// often carries the rarer script's unique words.
pub fn route(text: &str) -> Route {
    let mut counts: [usize; 6] = [0; 6];
    let script_index = |s: Script| match s {
        Script::Latin => 0,
        Script::Cyrillic => 1,
        Script::Devanagari => 2,
        Script::Arabic => 3,
        Script::Kana => 4,
        Script::Han => 5,
    };

    for c in text.chars() {
        if let Some(script) = spellman_language::char_script(c) {
            counts[script_index(script)] += 1;
        }
    }

    let [latin, cyrillic, devanagari, arabic, kana, han] = counts;

    if kana > 0 {
        return Route::Direct(Lang::Jpn);
    }
    let candidates =
        [(Script::Latin, latin), (Script::Cyrillic, cyrillic), (Script::Han, han)];
    if let Some((best, _)) = candidates.iter().copied().max_by_key(|(_, n)| *n).filter(|(_, n)| *n > 0) {
        return match best {
            Script::Latin => Route::Group(ScriptGroup::Latin),
            Script::Cyrillic => Route::Group(ScriptGroup::Cyrillic),
            Script::Han => Route::Direct(Lang::Cmn),
            _ => unreachable!("only Latin/Cyrillic/Han are candidates"),
        };
    }
    if devanagari > 0 {
        return Route::Direct(Lang::Hin);
    }
    if arabic > 0 {
        return Route::Direct(Lang::Ara);
    }
    Route::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_by_script() {
        assert_eq!(route("Привет, как дела?"), Route::Group(ScriptGroup::Cyrillic));
        assert_eq!(route("Hello world"), Route::Group(ScriptGroup::Latin));
        assert_eq!(route("¿Dónde está el baño?"), Route::Group(ScriptGroup::Latin));
        assert_eq!(route("Привіт hello мир"), Route::Group(ScriptGroup::Cyrillic));
        assert_eq!(route("こんにちは世界"), Route::Direct(Lang::Jpn));
        assert_eq!(route("北京是中国的首都"), Route::Direct(Lang::Cmn));
        assert_eq!(route("नमस्ते दुनिया"), Route::Direct(Lang::Hin));
        assert_eq!(route("مرحبا بالعالم"), Route::Direct(Lang::Ara));
        assert_eq!(route("12345 !!!"), Route::Unknown);
    }

    #[test]
    fn kana_beats_kanji_count() {
        // Mostly kanji with a single kana char: still Japanese.
        assert_eq!(route("東京駅へ行くのが好きですか"), Route::Direct(Lang::Jpn));
    }

    #[test]
    fn groups_are_contiguous_and_script_derived() {
        assert_eq!(ScriptGroup::Cyrillic.languages().len(), 21);
        assert_eq!(ScriptGroup::Latin.languages().len(), 5);
        assert_eq!(ScriptGroup::Direct.languages().len(), 4);
        assert_eq!(Lang::ALL.len(), NUM_LANGS);
        // Every language sits in the group its script implies, and each
        // group's languages are contiguous in Lang::ALL.
        for (i, lang) in Lang::ALL.iter().enumerate() {
            let group = group_of(*lang);
            assert!(group.languages().contains(lang), "{} misplaced", lang.code());
            assert!(group.column_range().contains(&i), "{} column {}", lang.code(), i);
        }
    }
}
