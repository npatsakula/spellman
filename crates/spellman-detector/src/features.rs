//! Character n-gram feature extraction.
//!
//! Text is split on whitespace and each word is **classified** before
//! packing: URLs, emails, @mentions and digit-bearing ASCII tokens are
//! canonicalized to fixed sentinel codepoints ([`SENTINEL_URL`] etc.) so
//! their arbitrary random tails never pollute the n-gram pool, while real
//! words are lowercased and wrapped in boundary markers
//! (`\u{2}` begin-of-word, `\u{3}` end-of-word — never produced by
//! lowercasing real text). Every n-gram with `n_min <= n <= n_max` over the
//! wrapped sequence is emitted as a packed `u64` key:
//! `k = ((...((c0 << 21) | c1) << 21 | ...) << 21 | c_{n-1}`
//! (wrapping; n >= 4 overflows 64 bits, which is fine — the mixer spreads the
//! wrapped value and the encoding stays injective for n <= 3, capturing whole
//! short words exactly).
//!
//! Boundaries make prefix/suffix n-grams explicit: the trigram
//! `(\u{2}, с, о)` is "word starts with со" — the affix signal that separates
//! Russian/Belarusian/Ukrainian verb endings.
//!
//! Canonicalization guards: a leading `#` is stripped (hashtags keep their
//! word — `#красноярск` is language evidence), and every class rule is
//! ASCII-only so Cyrillic dotted abbreviations (`т.е.`, `т.д.`) and
//! digit-bearing Cyrillic words (`миллион2020`) always pass through as real
//! words.
//!
//! The output of [`token_keys`] is stable across versions and identical to the
//! Python training implementation (verified against
//! `fixtures/hash_vectors.json`).

/// Word-begin boundary marker.
pub const BOW: char = '\u{2}';
/// Word-end boundary marker.
pub const EOW: char = '\u{3}';

/// Token-class canonicalization sentinels: private-use codepoints that never
/// occur in real text. A canonicalized word packs as `[BOW, sentinel, EOW]`,
/// so every URL/email/mention/number contributes the same handful of neutral
/// anchor n-grams regardless of its random tail — and weighs ~30 n-grams in
/// the mean-pool instead of the ~125 a 25-char URL would generate.
pub const SENTINEL_URL: char = '\u{E001}';
/// `user@domain.tld` → this sentinel.
pub const SENTINEL_EMAIL: char = '\u{E002}';
/// `@nickname` → this sentinel (nicknames are arbitrary Latin, not language
/// evidence).
pub const SENTINEL_MENTION: char = '\u{E003}';
/// Digit-bearing ASCII token (`2020`, `3.5.2`, `COVID19`, `100px`) → this
/// sentinel.
pub const SENTINEL_NUM: char = '\u{E004}';

/// Bits per codepoint in the packed key. Unicode codepoints fit in 21 bits.
pub const CP_BITS: u32 = 21;
const CP_MASK: u64 = (1 << CP_BITS) - 1;

/// N-gram window configuration. Serialized in model metadata.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FeatureConfig {
    pub n_min: u8,
    pub n_max: u8,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        FeatureConfig { n_min: 1, n_max: 5 }
    }
}

/// Pack a character slice into a `u64` key (wrapping shift-or chain).
///
/// Kept for reference/tests; the rolling packer in [`token_keys`] produces
/// the identical multiset of keys incrementally.
#[inline]
pub fn pack_ngram(chars: &[char]) -> u64 {
    let mut key: u64 = 0;
    for &c in chars {
        key = (key << CP_BITS) | ((c as u64) & CP_MASK);
    }
    key
}

const M1: u64 = (1 << CP_BITS) - 1;
const M2: u64 = (1 << (2 * CP_BITS)) - 1;
const M3: u64 = (1 << (3 * CP_BITS)) - 1;

/// Per-n domain-separation salt, XORed into every key: `key ^ (n * ODD)`.
/// Without it, u64 wrapping makes every 5-gram key bit-identical to its
/// suffix 4-gram key (and 4-grams collide across first chars sharing low
/// bits) — the model then sees amplified 4-grams instead of real 5-grams.
/// The XOR is bijective for fixed n, so within an order keys are unchanged;
/// across orders, windows no longer alias.
pub const N_TAG: [u64; 6] = {
    let odd = 0x9E37_79B9_7F4A_7C15u64;
    let mut tags = [0u64; 6];
    let mut n = 0;
    while n < 6 {
        tags[n] = (n as u64).wrapping_mul(odd);
        n += 1;
    }
    tags
};

/// Lowercase fast path for the scripts this detector routes: ASCII and the
/// main Cyrillic block А-Я are simple +0x20 offsets; the Cyrillic supplement
/// block U+0400-U+040F (Ё І Ї Є …) is a uniform +0x50. Both match
/// `char::to_lowercase` and Python's `str.lower()` exactly for these
/// ranges. Returns `None` for everything else (caller falls back to the
/// full Unicode mapping, which may expand to multiple codepoints).
#[inline]
fn fast_lower(c: char) -> Option<char> {
    match c {
        'A'..='Z' | '\u{0410}'..='\u{042F}' => char::from_u32(c as u32 + 0x20),
        '\u{0400}'..='\u{040F}' => char::from_u32(c as u32 + 0x50),
        'a'..='z' | '\u{0430}'..='\u{044F}' | '\u{0450}'..='\u{045F}' => Some(c),
        _ => None,
    }
}

/// Structural class of a whitespace-delimited word (after a leading `#` has
/// been stripped). Pure ASCII byte checks — no Unicode tables, so the Python
/// mirror is bit-exact by construction.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WordClass {
    /// Ordinary word: packed as-is.
    Word,
    /// `http(s)://…`, `www.…` or a bare ASCII domain → [`SENTINEL_URL`].
    Url,
    /// `user@domain.tld` → [`SENTINEL_EMAIL`].
    Email,
    /// `@nickname` → [`SENTINEL_MENTION`].
    Mention,
    /// Digit-bearing ASCII token → [`SENTINEL_NUM`].
    Num,
}

fn starts_with_ignore_case(word: &str, prefix: &str) -> bool {
    word.len() >= prefix.len()
        && word.as_bytes()[..prefix.len()].eq_ignore_ascii_case(prefix.as_bytes())
}

/// `letters/digits/hyphens/dots`, non-empty labels, alphabetic TLD of
/// length ≥ 2 (`t.co` yes, `a.b` no — protects `U.S.A.`-style abbreviations,
/// whose final label is a single letter).
fn is_ascii_domain(s: &str) -> bool {
    if !s.is_ascii() || !s.contains('.') {
        return false;
    }
    if !s.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-')) {
        return false;
    }
    let labels: Vec<&str> = s.split('.').collect();
    if labels.iter().any(|l| l.is_empty()) {
        return false;
    }
    let tld = labels[labels.len() - 1];
    tld.len() >= 2 && tld.bytes().all(|b| b.is_ascii_alphabetic())
}

/// Classify one word. Evaluation order matters and is part of the contract:
/// mention → URL prefix → email → digit-bearing ASCII → bare domain → word.
pub fn classify_word(word: &str) -> WordClass {
    if word.len() > 1 && word.starts_with('@') {
        return WordClass::Mention;
    }
    if starts_with_ignore_case(word, "http://")
        || starts_with_ignore_case(word, "https://")
        || starts_with_ignore_case(word, "www.")
    {
        return WordClass::Url;
    }
    if let Some((local, domain)) = word.split_once('@') {
        if !domain.contains('@')
            && !local.is_empty()
            && local.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'%' | b'+' | b'-'))
            && is_ascii_domain(domain)
        {
            return WordClass::Email;
        }
    }
    if word.is_ascii() && word.bytes().any(|b| b.is_ascii_digit()) {
        return WordClass::Num;
    }
    if is_ascii_domain(word) {
        return WordClass::Url;
    }
    WordClass::Word
}

/// Extract packed n-gram keys from `text`, in encounter order.
///
/// Convenience wrapper around the streaming [`for_each_key`]; allocates one
/// output vector (reserved from the byte length).
pub fn token_keys(text: &str, cfg: &FeatureConfig) -> Vec<u64> {
    // ~5 tokens per char (n_max ≤ 5); byte length over-reserves for
    // multi-byte scripts, which is cheaper than growing.
    let mut keys: Vec<u64> = Vec::with_capacity(text.len() * 3);
    for_each_key(text, cfg, |key| keys.push(key));
    keys
}

/// Streaming n-gram extraction: invokes `f` with every packed n-gram key in
/// encounter order. Zero-allocation core of the feature pipeline — the
/// rolling packer produces keys one char at a time, so consumers can score
/// or pack without materializing a token vector.
///
/// Rolling packer: one `u64` register per word, shifted by 21 bits per
/// character. The n-gram keys ending at each character are masks of the
/// register — n ≤ 3 windows fit under 64 bits exactly; n ≥ 4 windows wrap,
/// so the key is the full register (the wrap is what makes 5-gram keys
/// equal their suffix 4-gram keys; the multiset matches the reference
/// `pack_ngram` implementation bit-for-bit, only the emission order differs:
/// position-major here, n-major in `pack_ngram`).
///
/// Emission order affects nothing observable — the model consumes the token
/// multiset (float summation order shifts within rounding noise).
pub fn for_each_key<F: FnMut(u64)>(text: &str, cfg: &FeatureConfig, mut f: F) {
    let n_max = cfg.n_max as usize;
    let mut r: u64 = 0;
    let mut len: usize = 0;

    fn feed<F: FnMut(u64)>(
        r: &mut u64,
        len: &mut usize,
        c: u64,
        cfg: &FeatureConfig,
        n_max: usize,
        f: &mut F,
    ) {
        *r = (*r << CP_BITS) | (c & CP_MASK);
        *len += 1;
        let upper = n_max.min(*len);
        for n in cfg.n_min as usize..=upper {
            let window = match n {
                1 => *r & M1,
                2 => *r & M2,
                3 => *r & M3,
                _ => *r, // 4- and 5-gram windows wrap; window is the full register
            };
            f(window ^ N_TAG[n]);
        }
    }

    // Word-wise state machine: split_whitespace yields word slices with no
    // buffering, each word is classified whole (URL/email/mention/number
    // detection needs the full word), then either its lowercased chars or
    // one class sentinel feed the rolling packer between BOW/EOW markers.
    // Pass-through words emit exactly the same key sequence as the previous
    // char-streaming implementation.
    for word in text.split_whitespace() {
        // Hashtag: strip one '#' and classify the inner word (#красноярск is
        // real Cyrillic evidence; #COVID2020 becomes a Num sentinel).
        let word = word.strip_prefix('#').unwrap_or(word);
        if word.is_empty() {
            continue;
        }
        let sentinel = match classify_word(word) {
            WordClass::Word => None,
            WordClass::Url => Some(SENTINEL_URL),
            WordClass::Email => Some(SENTINEL_EMAIL),
            WordClass::Mention => Some(SENTINEL_MENTION),
            WordClass::Num => Some(SENTINEL_NUM),
        };
        feed(&mut r, &mut len, BOW as u64, cfg, n_max, &mut f);
        match sentinel {
            Some(s) => feed(&mut r, &mut len, s as u64, cfg, n_max, &mut f),
            None => {
                for c in word.chars() {
                    match fast_lower(c) {
                        Some(lc) => feed(&mut r, &mut len, lc as u64, cfg, n_max, &mut f),
                        None => {
                            for lc in c.to_lowercase() {
                                feed(&mut r, &mut len, lc as u64, cfg, n_max, &mut f);
                            }
                        }
                    }
                }
            }
        }
        feed(&mut r, &mut len, EOW as u64, cfg, n_max, &mut f);
        r = 0;
        len = 0;
    }
}

/// A token reduced to its signed bucket: `bucket` indexes the folded score
/// table, `neg` flips the row's contribution (signed hashing).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BucketToken {
    pub bucket: u32,
    pub neg: bool,
}

/// Extract signed bucket tokens from `text`.
///
/// This is the full Rust-side feature pipeline; the Python training pipeline
/// produces the identical sequence for the same `(text, cfg, hasher, log2_d)`.
pub fn bucket_tokens(
    text: &str,
    cfg: &FeatureConfig,
    hasher: &crate::hash::FeatureHasher,
    log2_d: u32,
) -> Vec<BucketToken> {
    let mut tokens = Vec::new();
    bucket_tokens_into(text, cfg, hasher, log2_d, &mut tokens);
    tokens
}

/// Scratch-buffer variant of [`bucket_tokens`]: clears `out`, then fills it
/// with the signed bucket tokens of `text`. Reusing one `Vec` across calls
/// removes the per-call allocation from batch paths.
pub fn bucket_tokens_into(
    text: &str,
    cfg: &FeatureConfig,
    hasher: &crate::hash::FeatureHasher,
    log2_d: u32,
    out: &mut Vec<BucketToken>,
) {
    out.clear();
    for_each_bucket(text, cfg, hasher, log2_d, |bucket, neg| {
        out.push(BucketToken { bucket, neg });
    });
}

/// Streaming signed-bucket extraction: invokes `f(bucket, neg)` for every
/// n-gram token in encounter order — zero-allocation, single pass.
pub fn for_each_bucket<F: FnMut(u32, bool)>(
    text: &str,
    cfg: &FeatureConfig,
    hasher: &crate::hash::FeatureHasher,
    log2_d: u32,
    mut f: F,
) {
    for_each_key(text, cfg, |key| {
        let (bucket, neg) = hasher.bucket(key, log2_d);
        f(bucket, neg);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_is_injective_for_short_ngrams() {
        let a = pack_ngram(&['а', 'б']);
        let b = pack_ngram(&['б', 'а']);
        assert_ne!(a, b);
        // Codepoints below 2^21 pack without collision for n <= 3.
        let c = pack_ngram(&['x', 'y', 'z']);
        assert_eq!(c, (('x' as u64) << 42) | (('y' as u64) << 21) | 'z' as u64);
    }

    #[test]
    fn single_word_ngrams_position_major() {
        let cfg = FeatureConfig { n_min: 1, n_max: 3 };
        let keys = token_keys("ab", &cfg);
        // seq = [BOW, a, b, EOW]; rolling packer emits, per position, all
        // n-grams ENDING there: 1 + 2 + 3 + 3 = 9 keys.
        assert_eq!(keys.len(), 9);
        // Position 1 (BOW): 1-gram.
        assert_eq!(keys[0], BOW as u64 ^ N_TAG[1]);
        // Position 2 (a): 1-, 2-gram.
        assert_eq!(keys[1], 'a' as u64 ^ N_TAG[1]);
        assert_eq!(keys[2], pack_ngram(&[BOW, 'a']) ^ N_TAG[2]);
        // Position 3 (b): 1-, 2-, 3-gram.
        assert_eq!(keys[3], 'b' as u64 ^ N_TAG[1]);
        assert_eq!(keys[4], pack_ngram(&['a', 'b']) ^ N_TAG[2]);
        assert_eq!(keys[5], pack_ngram(&[BOW, 'a', 'b']) ^ N_TAG[3]);
        // Position 4 (EOW): 1-, 2-, 3-gram.
        assert_eq!(keys[6], EOW as u64 ^ N_TAG[1]);
        assert_eq!(keys[7], pack_ngram(&['b', EOW]) ^ N_TAG[2]);
        assert_eq!(keys[8], pack_ngram(&['a', 'b', EOW]) ^ N_TAG[3]);
    }

    #[test]
    fn rolling_matches_reference_multiset() {
        // The rolling packer must produce exactly the multiset of the
        // reference per-window packing (order differs, values must not).
        let cfg = FeatureConfig { n_min: 1, n_max: 5 };
        for text in ["ab", "abc", "привет мир", "Съешь ещё этих", "гнійеже ґава їжак"] {
            let rolling = token_keys(text, &cfg);
            let mut reference = Vec::new();
            for word in text.to_lowercase().split_whitespace() {
                let seq: Vec<char> = std::iter::once(BOW).chain(word.chars()).chain(std::iter::once(EOW)).collect();
                for n in cfg.n_min as usize..=cfg.n_max as usize {
                    if n > seq.len() {
                        break;
                    }
                    for start in 0..=seq.len() - n {
                        reference.push(pack_ngram(&seq[start..start + n]) ^ N_TAG[n]);
                    }
                }
            }
            let mut a = rolling.clone();
            let mut b = reference;
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "multiset mismatch for {text:?}");
        }
    }

    #[test]
    fn words_are_isolated_by_whitespace() {
        let cfg = FeatureConfig { n_min: 2, n_max: 2 };
        let keys = token_keys("a b", &cfg);
        // Two words: [BOW,a,EOW] and [BOW,b,EOW], bigram at positions 2 and 3.
        assert_eq!(keys.len(), 4);
        assert_eq!(keys[0], pack_ngram(&[BOW, 'a']) ^ N_TAG[2]);
        assert_eq!(keys[1], pack_ngram(&['a', EOW]) ^ N_TAG[2]);
        assert_eq!(keys[2], pack_ngram(&[BOW, 'b']) ^ N_TAG[2]);
        assert_eq!(keys[3], pack_ngram(&['b', EOW]) ^ N_TAG[2]);
    }

    #[test]
    fn lowercases_before_packing() {
        let cfg = FeatureConfig { n_min: 1, n_max: 1 };
        let upper = token_keys("AБ", &cfg);
        let lower = token_keys("aб", &cfg);
        assert_eq!(upper, lower);
    }

    #[test]
    fn n_max_capped_by_word_length() {
        let cfg = FeatureConfig { n_min: 1, n_max: 5 };
        // Word "ab" wrapped is 4 chars: 4+3+2+1 = 10 n-grams, no 5-grams.
        assert_eq!(token_keys("ab", &cfg).len(), 10);
        // Word "abc" wrapped is 5 chars: 5+4+3+2+1 = 15.
        assert_eq!(token_keys("abc", &cfg).len(), 15);
    }

    #[test]
    fn empty_text_yields_nothing() {
        assert!(token_keys("", &FeatureConfig::default()).is_empty());
        assert!(token_keys("   \n\t ", &FeatureConfig::default()).is_empty());
    }

    #[test]
    fn bucket_tokens_cover_full_range() {
        let cfg = FeatureConfig::default();
        let hasher = crate::hash::FeatureHasher::default();
        let toks = bucket_tokens("Привет, hello мир!", &cfg, &hasher, 17);
        assert!(!toks.is_empty());
        for t in &toks {
            assert!(t.bucket < (1 << 17));
        }
    }

    #[test]
    fn word_classification() {
        use WordClass::*;
        // URLs: schemes, www, bare ASCII domains.
        assert_eq!(classify_word("https://t.co/3Kr7yzeYLC"), Url);
        assert_eq!(classify_word("HTTP://EXAMPLE.COM/PATH"), Url);
        assert_eq!(classify_word("www.grozny-inform.ru"), Url);
        assert_eq!(classify_word("t.co"), Url);
        assert_eq!(classify_word("example.com"), Url);
        // Emails.
        assert_eq!(classify_word("test@mail.ru"), Email);
        assert_eq!(classify_word("a.b+c%tag@sub.domain.org"), Email);
        assert_eq!(classify_word("a@b"), Word); // no dot in domain
        // Mentions.
        assert_eq!(classify_word("@daria_karapet"), Mention);
        assert_eq!(classify_word("@"), Word); // bare @ is not a mention
        // Digit-bearing ASCII.
        assert_eq!(classify_word("2020"), Num);
        assert_eq!(classify_word("3.5.2"), Num);
        assert_eq!(classify_word("100px"), Num);
        assert_eq!(classify_word("COVID19"), Num);
        // Cyrillic immunity: dotted abbreviations and digit-bearing words
        // stay real words.
        assert_eq!(classify_word("т.е."), Word);
        assert_eq!(classify_word("т.д."), Word);
        assert_eq!(classify_word("миллион2020"), Word);
        assert_eq!(classify_word("U.S.A."), Word); // single-letter TLD
        assert_eq!(classify_word("Привет"), Word);
    }

    #[test]
    fn canonicalized_words_pack_as_sentinels() {
        let cfg = FeatureConfig::default();
        // A whole URL canonicalizes to [BOW, SENTINEL_URL, EOW] — 6 keys,
        // identical for every URL regardless of tail.
        let a = token_keys("https://t.co/3Kr7yzeYLC", &cfg);
        let b = token_keys("https://example.com/other/tail", &cfg);
        assert_eq!(a, b);
        assert_eq!(a.len(), 6);

        // Distinct per class.
        let url = token_keys("https://x.io", &cfg);
        let email = token_keys("a@b.co", &cfg);
        let mention = token_keys("@nick", &cfg);
        let num = token_keys("2020", &cfg);
        assert_ne!(url, email);
        assert_ne!(url, mention);
        assert_ne!(url, num);

        // Hashtags keep their word (real language evidence).
        let cfg1 = FeatureConfig { n_min: 1, n_max: 1 };
        assert_eq!(token_keys("#красноярск", &cfg1), token_keys("красноярск", &cfg1));
        // ...but a digit-bearing hashtag canonicalizes.
        assert_eq!(token_keys("#2020", &cfg1), token_keys("2020", &cfg1));

        // Mixed text: sentinels interleave with real words in encounter order.
        let mixed = token_keys("Привет @nick пока", &cfg1);
        let plain = token_keys("Привет @x пока", &cfg1);
        assert_eq!(mixed, plain);
    }

    #[test]
    fn bare_hashtag_marker_yields_nothing() {
        let cfg = FeatureConfig::default();
        assert!(token_keys("#", &cfg).is_empty());
        assert!(token_keys("# #", &cfg).is_empty());
    }
}
