//! Language histogram of a markdown book under lingua 1.8 — the
//! baseline counterpart of spellman's `detect_md` sweep, on IDENTICAL
//! sentences: the same example-local splitter (included by path) and
//! the same min-chars/letter filters, so distribution differences are
//! purely the detector's.
//!
//! lingua is built from exactly the 17 languages it shares with
//! spellman's inventory (as in `lid-bench`), high-accuracy mode,
//! preloaded models — its best shot.
//!
//! Usage (from benchmarks/):
//!   cargo run --release --bin voyna_lingua -- ../tests/voyna-i-mir.md

use std::collections::HashMap;
use std::time::Instant;

#[path = "../../../crates/spellman-detector/examples/detect_md/sent.rs"]
mod sent;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../tests/voyna-i-mir.md".to_string());
    let md = std::fs::read_to_string(&path).expect("read markdown");
    let min_chars = 20usize;

    let sentences: Vec<String> = md
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .flat_map(|line| {
            sent::glue_short(
                sent::split_line(line).iter().map(|s| s.to_string()).collect(),
                min_chars,
            )
        })
        .filter(|s| s.chars().count() >= min_chars && s.chars().any(char::is_alphabetic))
        .collect();
    println!("md={} ({} KB)", path, md.len() / 1024);
    println!("sentences: {} (min {min_chars} chars, letters required)", sentences.len());

    use lingua::Language as L;
    let pairs: Vec<(L, &'static str)> = vec![
        (L::Russian, "rus"),
        (L::Ukrainian, "ukr"),
        (L::Belarusian, "bel"),
        (L::Bulgarian, "bul"),
        (L::Macedonian, "mkd"),
        (L::Serbian, "srp"),
        (L::Kazakh, "kaz"),
        (L::Mongolian, "mon"),
        (L::English, "eng"),
        (L::Spanish, "spa"),
        (L::French, "fra"),
        (L::Portuguese, "por"),
        (L::German, "deu"),
        (L::Chinese, "cmn"),
        (L::Japanese, "jpn"),
        (L::Hindi, "hin"),
        (L::Arabic, "ara"),
    ];
    let langs: Vec<L> = pairs.iter().map(|(l, _)| *l).collect();
    let t = Instant::now();
    let mut builder = lingua::LanguageDetectorBuilder::from_languages(&langs);
    builder.with_preloaded_language_models();
    let detector = builder.build();
    println!("models loaded in {:.2?}", t.elapsed());

    let mut hist: HashMap<&'static str, u64> = HashMap::new();
    let mut none = 0u64;
    let t = Instant::now();
    for s in &sentences {
        match detector.detect_language_of(s) {
            Some(l) => {
                let code = pairs
                    .iter()
                    .find(|(x, _)| *x == l)
                    .map(|(_, c)| *c)
                    .unwrap_or("other");
                *hist.entry(code).or_default() += 1;
            }
            None => none += 1,
        }
    }
    let dt = t.elapsed();
    let total: u64 = hist.values().sum::<u64>() + none;
    println!(
        "detected {total} sentences in {:.3?}: {:.0} sentences/s",
        dt,
        total as f64 / dt.as_secs_f64()
    );
    println!();
    println!("language  count    share");
    let mut rows: Vec<_> = hist.into_iter().collect();
    rows.sort_by_key(|a| std::cmp::Reverse(a.1));
    for (code, n) in rows {
        println!(
            "  {:>4}  {:>7}  {:>5.1}%",
            code,
            n,
            100.0 * n as f64 / total as f64
        );
    }
    println!("no prediction: {none} ({:.2}%)", 100.0 * none as f64 / total as f64);
}
