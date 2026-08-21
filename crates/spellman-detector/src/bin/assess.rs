//! Accuracy assessment CLI over TSV eval files (`lang_code<TAB>text`).
//!
//! Reports, lingua-style, accuracy at four input granularities — single
//! words, word pairs, word triples, and full texts — plus per-language
//! precision/recall/F1, length buckets and confusion pairs at text level.
//! The granularity ladder measures how much context the detector needs:
//! single words are the hardest (many Cyrillic languages share most of their
//! lexicon core), and the pair/triple rungs show where it converges.
//!
//! Fragments are derived from the eval texts themselves: whitespace tokens
//! containing at least one letter, and consecutive n-windows of that token
//! sequence joined by a single space.
//!
//! Usage:
//!   cargo run --release --bin assess -- --model model/ eval.tsv [more.tsv ...]

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use spellman_detector::{BulkDetector, Lang};

const K: usize = 1024;
const MAX_BATCH: usize = 512;

struct Args {
    model: PathBuf,
    eval_files: Vec<PathBuf>,
    /// Optional directory to dump up to `--dump-per-lang` misclassified
    /// samples per gold language as `<gold>.tsv` (gold<TAB>pred<TAB>conf<TAB>text).
    dump_errors: Option<PathBuf>,
    dump_per_lang: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut model = None;
    let mut eval_files = Vec::new();
    let mut dump_errors = None;
    let mut dump_per_lang = 100usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => {
                model = Some(args.next().ok_or("--model requires a path")?.into());
            }
            "--dump-errors" => {
                dump_errors = Some(args.next().ok_or("--dump-errors requires a path")?.into());
            }
            "--dump-per-lang" => {
                dump_per_lang = args
                    .next()
                    .ok_or("--dump-per-lang requires a number")?
                    .parse()
                    .map_err(|_| "bad --dump-per-lang")?;
            }
            other if other.starts_with("--") => return Err(format!("unknown flag: {other}")),
            other => eval_files.push(other.into()),
        }
    }
    Ok(Args {
        model: model.ok_or("missing --model DIR")?,
        eval_files: if eval_files.is_empty() {
            Err("no eval files given")?
        } else {
            eval_files
        },
        dump_errors,
        dump_per_lang,
    })
}

#[derive(Default)]
struct Confusion {
    counts: HashMap<(Lang, Lang), usize>, // (gold, predicted)
}

/// Accumulated (correct, total) counts per language plus the overall tally.
#[derive(Default)]
struct Tally {
    overall: (usize, usize),
    per_lang: HashMap<Lang, (usize, usize)>,
}

impl Tally {
    fn record(&mut self, gold: Lang, correct: bool) {
        self.overall.0 += usize::from(correct);
        self.overall.1 += 1;
        let e = self.per_lang.entry(gold).or_default();
        e.0 += usize::from(correct);
        e.1 += 1;
    }

    fn accuracy(&self) -> f64 {
        if self.overall.1 == 0 {
            f64::NAN
        } else {
            self.overall.0 as f64 / self.overall.1 as f64 * 100.0
        }
    }

    fn lang_accuracy(&self, lang: Lang) -> f64 {
        match self.per_lang.get(&lang) {
            Some((c, t)) if *t > 0 => *c as f64 / *t as f64 * 100.0,
            _ => f64::NAN,
        }
    }
}

/// Letter-bearing whitespace tokens of a text (the fragment building blocks;
/// punctuation/number-only tokens carry no language signal).
fn word_tokens(text: &str) -> Vec<&str> {
    text.split_whitespace()
        .filter(|t| t.chars().any(char::is_alphabetic))
        .collect()
}

/// Consecutive n-word fragments joined by a single space, labeled with the
/// source text's gold language.
fn fragments(rows: &[(Lang, String)], n: usize) -> Vec<(Lang, String)> {
    let mut out = Vec::new();
    for (gold, text) in rows {
        let tokens = word_tokens(text);
        for window in tokens.windows(n) {
            out.push((*gold, window.join(" ")));
        }
    }
    out
}

/// Batch-detect `items` and tally per-gold-language accuracy.
fn tally_fragments(
    detector: &mut BulkDetector,
    items: &[(Lang, String)],
) -> Result<Tally, Box<dyn std::error::Error>> {
    let mut tally = Tally::default();
    for chunk in items.chunks(MAX_BATCH) {
        let texts: Vec<&str> = chunk.iter().map(|(_, t)| t.as_str()).collect();
        for ((gold, _), det) in chunk.iter().zip(detector.detect_batch(&texts)?) {
            tally.record(*gold, det.lang == Some(*gold));
        }
    }
    Ok(tally)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let mut detector = BulkDetector::load(&args.model, K, MAX_BATCH)?;

    let mut rows: Vec<(Lang, String)> = Vec::new();
    for path in &args.eval_files {
        let content = fs::read_to_string(path)?;
        for line in content.lines() {
            let Some((code, text)) = line.split_once('\t') else {
                continue;
            };
            match code.trim().parse::<Lang>() {
                Ok(gold) => rows.push((gold, text.to_string())),
                Err(_) => eprintln!("skipping unknown language code: {code}"),
            }
        }
    }
    if rows.is_empty() {
        return Err("no eval samples read".into());
    }
    println!("samples: {}", rows.len());

    // ---- Text level: confusion, length buckets, per-language P/R/F1 ----
    let mut confusion = Confusion::default();
    let mut length_buckets: [(usize, usize); 3] = [(0, 0); 3]; // (correct, total) per bucket
    let mut n_unknown = 0usize;
    // gold -> capped list of (pred, conf, uncertain, text)
    let mut errors: HashMap<Lang, Vec<(Lang, f32, bool, String)>> = HashMap::new();
    for chunk in rows.chunks(MAX_BATCH) {
        let texts: Vec<&str> = chunk.iter().map(|(_, t)| t.as_str()).collect();
        for ((gold, text), det) in chunk.iter().zip(detector.detect_batch(&texts)?) {
            let bucket = match text.chars().count() {
                0..=20 => 0,
                21..=100 => 1,
                _ => 2,
            };
            length_buckets[bucket].0 += usize::from(det.lang == Some(*gold));
            length_buckets[bucket].1 += 1;
            match det.lang {
                Some(pred) => {
                    *confusion.counts.entry((*gold, pred)).or_default() += 1;
                    if pred != *gold {
                        let errs = errors.entry(*gold).or_default();
                        if errs.len() < args.dump_per_lang {
                            errs.push((pred, det.confidence, det.is_uncertain, text.clone()));
                        }
                    }
                }
                None => n_unknown += 1,
            }
        }
    }

    let mut text_tally = Tally::default();
    let mut correct: HashMap<Lang, usize> = HashMap::new();
    let mut gold_total: HashMap<Lang, usize> = HashMap::new();
    let mut pred_total: HashMap<Lang, usize> = HashMap::new();
    for (&(gold, pred), &count) in &confusion.counts {
        *gold_total.entry(gold).or_default() += count;
        *pred_total.entry(pred).or_default() += count;
        if gold == pred {
            correct.insert(gold, count);
            text_tally.overall.0 += count;
        }
    }
    text_tally.overall.1 = rows.len();
    for (&gold, &count) in &gold_total {
        text_tally
            .per_lang
            .insert(gold, (correct.get(&gold).copied().unwrap_or(0), count));
    }

    println!("text accuracy: {:.2}%", text_tally.accuracy());
    if n_unknown > 0 {
        println!("no-script (routed to None): {n_unknown}");
    }
    println!();
    println!("by length:        ≤20    21-100   >100");
    let accs: Vec<f64> = length_buckets
        .iter()
        .map(|(c, t)| {
            if *t == 0 {
                f64::NAN
            } else {
                *c as f64 / *t as f64 * 100.0
            }
        })
        .collect();
    println!(
        "  accuracy:    {:>6.2}%  {:>6.2}%  {:>6.2}%   (n = {} / {} / {})",
        accs[0], accs[1], accs[2], length_buckets[0].1, length_buckets[1].1, length_buckets[2].1
    );

    // ---- Granularity ladder (lingua-style) ----
    let word_tally = tally_fragments(&mut detector, &fragments(&rows, 1))?;
    let pair_tally = tally_fragments(&mut detector, &fragments(&rows, 2))?;
    let triple_tally = tally_fragments(&mut detector, &fragments(&rows, 3))?;
    println!();
    println!("granularity ladder (accuracy on fragments derived from the eval texts):");
    println!("  lang   word   pair  triple   text");
    println!(
        "  ALL  {:>6.2} {:>6.2} {:>6.2} {:>6.2}   (n = {} / {} / {} / {})",
        word_tally.accuracy(),
        pair_tally.accuracy(),
        triple_tally.accuracy(),
        text_tally.accuracy(),
        word_tally.overall.1,
        pair_tally.overall.1,
        triple_tally.overall.1,
        text_tally.overall.1
    );

    // Per-language rows, worst single-word accuracy first.
    let mut langs: Vec<Lang> = word_tally.per_lang.keys().copied().collect();
    langs.sort_by(|a, b| {
        word_tally
            .lang_accuracy(*a)
            .total_cmp(&word_tally.lang_accuracy(*b))
    });
    for lang in langs.iter().take(15) {
        println!(
            "  {:>4}  {:>6.2} {:>6.2} {:>6.2} {:>6.2}",
            lang,
            word_tally.lang_accuracy(*lang),
            pair_tally.lang_accuracy(*lang),
            triple_tally.lang_accuracy(*lang),
            text_tally.lang_accuracy(*lang),
        );
    }
    if langs.len() > 15 {
        println!(
            "  (+{} languages below the 15 shown, all ≥ {:.2}% on words)",
            langs.len() - 15,
            word_tally.lang_accuracy(langs[15])
        );
    }

    // ---- Per-language P/R/F1 at text level ----
    println!();
    println!("per-language (sorted by F1, worst first):");
    println!("  lang  n      P      R      F1");
    let mut f1_langs: Vec<Lang> = gold_total.keys().copied().collect();
    f1_langs.sort_by_key(|l| {
        f1(*l, &correct, &gold_total, &pred_total)
            .total_cmp(&0.0)
            .reverse()
    });
    for lang in f1_langs
        .iter()
        .filter(|l| f1(**l, &correct, &gold_total, &pred_total) < 0.999)
    {
        let p = precision(*lang, &correct, &pred_total);
        let r = recall(*lang, &correct, &gold_total);
        println!(
            "  {:>4}  {:>5}  {:.2}  {:.2}  {:.2}",
            lang,
            gold_total[lang],
            p,
            r,
            f1(*lang, &correct, &gold_total, &pred_total)
        );
    }
    let perfect = f1_langs
        .iter()
        .filter(|l| f1(**l, &correct, &gold_total, &pred_total) >= 0.999)
        .count();
    println!("  (+{perfect} languages with F1 ≥ 99.9%)");
    println!();
    println!("top confusions (gold -> predicted):");
    let mut pairs: Vec<((Lang, Lang), usize)> = confusion
        .counts
        .iter()
        .filter(|((g, p), _)| g != p)
        .map(|(k, v)| (*k, *v))
        .collect();
    pairs.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    for ((gold, pred), count) in pairs.iter().take(12) {
        println!("  {gold} -> {pred}: {count}");
    }

    if let Some(dir) = &args.dump_errors {
        std::fs::create_dir_all(dir)?;
        let mut langs: Vec<Lang> = errors.keys().copied().collect();
        langs.sort();
        for gold in langs {
            let mut errs = errors.remove(&gold).unwrap_or_default();
            errs.sort_by(|a, b| b.1.total_cmp(&a.1)); // most confident mistakes first
            let path = dir.join(format!("{gold}.tsv"));
            let mut out = String::new();
            for (pred, conf, uncertain, text) in &errs {
                out.push_str(&format!(
                    "{gold}\t{pred}\t{conf:.3}\t{}\t{}\n",
                    if *uncertain { "uncertain" } else { "" },
                    text.replace(['\t', '\n'], " ")
                ));
            }
            fs::write(&path, out)?;
            println!("dumped {} errors -> {}", errs.len(), path.display());
        }
    }

    Ok(())
}

fn precision(lang: Lang, correct: &HashMap<Lang, usize>, pred_total: &HashMap<Lang, usize>) -> f64 {
    let c = *correct.get(&lang).unwrap_or(&0) as f64;
    let t = *pred_total.get(&lang).unwrap_or(&0) as f64;
    if t == 0.0 { f64::NAN } else { c / t }
}

fn recall(lang: Lang, correct: &HashMap<Lang, usize>, gold_total: &HashMap<Lang, usize>) -> f64 {
    let c = *correct.get(&lang).unwrap_or(&0) as f64;
    let t = *gold_total.get(&lang).unwrap_or(&0) as f64;
    if t == 0.0 { f64::NAN } else { c / t }
}

fn f1(
    lang: Lang,
    correct: &HashMap<Lang, usize>,
    gold_total: &HashMap<Lang, usize>,
    pred_total: &HashMap<Lang, usize>,
) -> f64 {
    let p = precision(lang, correct, pred_total);
    let r = recall(lang, correct, gold_total);
    if p + r == 0.0 {
        0.0
    } else {
        2.0 * p * r / (p + r)
    }
}
