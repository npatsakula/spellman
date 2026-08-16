//! lid-bench — accuracy + latency of spellman vs third-party detectors on
//! identical eval rows.
//!
//! Methodology: eval TSVs (`code<TAB>text`, spellman codes) are sampled
//! deterministically (seeded, balanced per language) and every tool scores
//! exactly the same rows. Two accuracy figures per tool:
//!
//! - **supported-acc** — accuracy on the rows whose gold language the tool
//!   can even express (its label inventory ∩ our 30 classes): the tool's
//!   best case, matching how the tools benchmark themselves.
//! - **all-rows-acc** — accuracy on the whole sample counting
//!   unsupported-gold rows as errors: what a 30-class Cyrillic-heavy
//!   workload actually sees.
//!
//! lingua is built from exactly the languages of ours it supports (17/30)
//! with preloaded models — its best shot on our inventory. whichlang knows
//! 16 languages total (10 of ours, only `rus` among the Cyrillic classes).
//!
//! Run from this directory:
//!   cargo run --release -- --model ../model ../model/eval_test.tsv

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use spellman_detector::{BulkDetector, SingleDetector};
use spellman_language::Lang;

/// Our ISO 639-3 code -> lingua's `Language`, where supported.
fn lingua_languages() -> Vec<(lingua::Language, &'static str)> {
    use lingua::Language as L;
    vec![
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
    ]
}

fn lingua_code(lang: lingua::Language) -> Option<&'static str> {
    lingua_languages().into_iter().find(|(l, _)| *l == lang).map(|(_, c)| c)
}

/// Our ISO 639-3 code -> whichlang's `Lang`, where supported.
fn whichlang_support(code: &str) -> Option<whichlang::Lang> {
    use whichlang::Lang as W;
    Some(match code {
        "rus" => W::Rus,
        "cmn" => W::Cmn,
        "deu" => W::Deu,
        "eng" => W::Eng,
        "fra" => W::Fra,
        "hin" => W::Hin,
        "jpn" => W::Jpn,
        "por" => W::Por,
        "spa" => W::Spa,
        "ara" => W::Ara,
        _ => return None,
    })
}

fn supports_all(_: &str) -> bool {
    true
}

fn supports_whichlang(code: &str) -> bool {
    whichlang_support(code).is_some()
}

fn supports_lingua(code: &str) -> bool {
    lingua_languages().iter().any(|(_, c)| *c == code)
}

/// One tool's complete run over the sample.
struct RunResult {
    name: String,
    supported: usize,
    load_secs: f64,
    elapsed_secs: f64,
    /// Per-row prediction as our ISO 639-3 code (None = no answer).
    preds: Vec<Option<String>>,
    supports: fn(&str) -> bool,
}

#[derive(Parser)]
#[command(
    name = "lid-bench",
    about = "spellman vs whichlang vs lingua on identical eval rows"
)]
struct Cli {
    /// Model directory for spellman.
    #[arg(long, default_value = "../model")]
    model: PathBuf,

    /// Eval TSV files (code<TAB>text).
    #[arg(required = true)]
    eval_files: Vec<PathBuf>,

    /// Rows sampled per gold language (0 = all rows).
    #[arg(long, default_value_t = 500)]
    rows_per_lang: usize,

    #[arg(long, default_value_t = 42)]
    seed: u32,

    /// spellman per-document token budget.
    #[arg(long, default_value_t = 1024)]
    k: usize,

    /// spellman batch size.
    #[arg(long, default_value_t = 1024)]
    max_batch: usize,

    /// Print a per-gold-language accuracy matrix.
    #[arg(long)]
    per_lang: bool,

    /// Print accuracy per text-length bucket (chars: <=20 / 21-100 / >100,
    /// the same buckets `assess` uses).
    #[arg(long)]
    by_length: bool,
}

/// Char-length bucket, matching `assess`: 0 = ≤20 chars, 1 = 21–100, 2 = >100.
fn len_bucket(text: &str) -> usize {
    let n = text.chars().count();
    if n <= 20 {
        0
    } else if n <= 100 {
        1
    } else {
        2
    }
}

fn main() {
    let cli = Cli::parse();
    let rows = load_and_sample(&cli.eval_files, cli.rows_per_lang, cli.seed);
    let langs: BTreeSet<Lang> = rows.iter().map(|(l, _)| *l).collect();
    let files: Vec<String> = cli.eval_files.iter().map(|p| p.display().to_string()).collect();
    println!(
        "eval: {files:?} — {} rows, {} languages, {} per-language cap, seed {}",
        rows.len(),
        langs.len(),
        if cli.rows_per_lang == 0 { "none".to_string() } else { cli.rows_per_lang.to_string() },
        cli.seed,
    );
    let texts: Vec<String> = rows.iter().map(|(_, t)| t.clone()).collect();
    let golds: Vec<&str> = rows.iter().map(|(l, _)| l.code()).collect();

    let runs = vec![
        run_spellman_bulk(&cli, &texts),
        run_spellman_single(&cli, &texts),
        run_whichlang(&texts, &golds),
        run_lingua(&texts, &golds, false),
        run_lingua(&texts, &golds, true),
    ];

    println!(
        "\n{:<20} {:>10} {:>14} {:>13} {:>10} {:>8}",
        "tool", "supported", "supported-acc", "all-rows-acc", "µs/sample", "load"
    );
    for run in &runs {
        let (sup_rows, sup_ok, all_ok) = score(run, &golds);
        let sup_acc = 100.0 * sup_ok as f64 / sup_rows.max(1) as f64;
        let all_acc = 100.0 * all_ok as f64 / run.preds.len().max(1) as f64;
        let us = run.elapsed_secs * 1e6 / run.preds.len().max(1) as f64;
        println!(
            "{:<20} {:>7}/30 {:>13.2}% {:>12.2}% {:>10.1} {:>7.1}s",
            run.name,
            run.supported,
            sup_acc,
            all_acc,
            us,
            run.load_secs,
        );
    }

    if cli.by_length {
        println!("\naccuracy by text length (chars): supported-acc / all-rows-acc");
        let buckets: Vec<usize> = texts.iter().map(|t| len_bucket(t)).collect();
        for (b, name) in ["≤20", "21-100", ">100"].iter().enumerate() {
            let idxs: Vec<usize> = (0..golds.len()).filter(|i| buckets[*i] == b).collect();
            if idxs.is_empty() {
                println!("  {name}: (no rows)");
                continue;
            }
            println!("  {name} ({} rows):", idxs.len());
            for run in &runs {
                let sup = idxs.iter().filter(|i| (run.supports)(golds[**i]));
                let sup_rows = sup.clone().count();
                let sup_ok = sup.filter(|i| run.preds[**i].as_deref() == Some(golds[**i])).count();
                let all_ok =
                    idxs.iter().filter(|i| run.preds[**i].as_deref() == Some(golds[**i])).count();
                println!(
                    "    {:<20} {:>6.2}% / {:>6.2}%",
                    run.name,
                    100.0 * sup_ok as f64 / sup_rows.max(1) as f64,
                    100.0 * all_ok as f64 / idxs.len() as f64,
                );
            }
        }
    }

    if cli.per_lang {
        println!("\nper-gold-language accuracy (— = gold language not in the tool's inventory):");
        print!("{:>6}", "lang");
        for run in &runs {
            print!(" {:>20}", run.name);
        }
        println!();
        for lang in langs {
            print!("{:>6}", lang.code());
            for run in &runs {
                let idxs: Vec<usize> = golds
                    .iter()
                    .enumerate()
                    .filter(|(_, g)| **g == lang.code())
                    .map(|(i, _)| i)
                    .collect();
                let cell = if idxs.is_empty() || !(run.supports)(lang.code()) {
                    "—".to_string()
                } else {
                    let ok = idxs.iter().filter(|i| run.preds[**i].as_deref() == Some(lang.code())).count();
                    format!("{:.1}% ({})", 100.0 * ok as f64 / idxs.len() as f64, idxs.len())
                };
                print!(" {:>20}", cell);
            }
            println!();
        }
    }
}

/// (supported-gold rows, correct on those, correct on all)
fn score(run: &RunResult, golds: &[&str]) -> (usize, usize, usize) {
    let mut sup_rows = 0;
    let mut sup_ok = 0;
    let mut all_ok = 0;
    for (gold, pred) in golds.iter().zip(&run.preds) {
        let ok = pred.as_deref() == Some(*gold);
        all_ok += ok as usize;
        if (run.supports)(gold) {
            sup_rows += 1;
            sup_ok += ok as usize;
        }
    }
    (sup_rows, sup_ok, all_ok)
}

fn run_spellman_bulk(cli: &Cli, texts: &[String]) -> RunResult {
    let t = Instant::now();
    let mut det = BulkDetector::load(&cli.model, cli.k, cli.max_batch).expect("load spellman model");
    let load = t.elapsed().as_secs_f64();

    // Warm the compiled plan with one batch, then time the full pass.
    let head: Vec<&str> = texts.iter().take(cli.max_batch.min(texts.len())).map(String::as_str).collect();
    let _ = det.detect_batch(&head).expect("warmup");

    let t = Instant::now();
    let mut preds = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(cli.max_batch) {
        let refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
        for d in det.detect_batch(&refs).expect("spellman bulk detect") {
            preds.push(d.lang.map(|l| l.code().to_string()));
        }
    }
    RunResult {
        name: "spellman (bulk)".into(),
        supported: 30,
        load_secs: load,
        elapsed_secs: t.elapsed().as_secs_f64(),
        preds,
        supports: supports_all,
    }
}

fn run_spellman_single(cli: &Cli, texts: &[String]) -> RunResult {
    let t = Instant::now();
    let mut det = SingleDetector::load(&cli.model, cli.k).expect("load spellman model");
    let load = t.elapsed().as_secs_f64();

    for text in texts.iter().take(64) {
        let _ = det.detect(text).expect("warmup");
    }

    let t = Instant::now();
    let preds = texts
        .iter()
        .map(|text| {
            det.detect(std::hint::black_box(text))
                .expect("spellman single detect")
                .lang
                .map(|l| l.code().to_string())
        })
        .collect();
    RunResult {
        name: "spellman (single)".into(),
        supported: 30,
        load_secs: load,
        elapsed_secs: t.elapsed().as_secs_f64(),
        preds,
        supports: supports_all,
    }
}

/// Warm a detector that loads language models lazily: run one detection
/// per distinct gold language (in tool-supported set) so the timed pass
/// never pays a model-load cost.
fn warmup_by_gold(texts: &[String], golds: &[&str], supports: fn(&str) -> bool, mut detect: impl FnMut(&str)) {
    let mut warmed: std::collections::BTreeSet<&str> = Default::default();
    for (text, gold) in texts.iter().zip(golds) {
        if warmed.contains(gold) || !supports(gold) {
            continue;
        }
        detect(text);
        warmed.insert(gold);
    }
}

fn run_whichlang(texts: &[String], golds: &[&str]) -> RunResult {
    warmup_by_gold(texts, golds, supports_whichlang, |text| {
        std::hint::black_box(whichlang::detect_language(text));
    });
    let t = Instant::now();
    let preds = texts
        .iter()
        .map(|text| {
            let lang = whichlang::detect_language(std::hint::black_box(text));
            Some(lang.three_letter_code().to_string())
        })
        .collect();
    RunResult {
        name: "whichlang 0.1".into(),
        supported: 30 - Lang::ALL.iter().filter(|l| whichlang_support(l.code()).is_none()).count(),
        load_secs: 0.0,
        elapsed_secs: t.elapsed().as_secs_f64(),
        preds,
        supports: supports_whichlang,
    }
}

fn run_lingua(texts: &[String], golds: &[&str], low: bool) -> RunResult {
    let langs: Vec<lingua::Language> = lingua_languages().iter().map(|(l, _)| *l).collect();
    let t = Instant::now();
    let mut builder = lingua::LanguageDetectorBuilder::from_languages(&langs);
    if low {
        builder.with_low_accuracy_mode();
    }
    builder.with_preloaded_language_models();
    let detector = builder.build();
    let load = t.elapsed().as_secs_f64();

    warmup_by_gold(texts, golds, supports_lingua, |text| {
        std::hint::black_box(detector.detect_language_of(text));
    });
    let t = Instant::now();
    let preds = texts
        .iter()
        .map(|text| {
            detector
                .detect_language_of(std::hint::black_box(text))
                .and_then(lingua_code)
                .map(str::to_string)
        })
        .collect();
    RunResult {
        name: format!("lingua 1.8 ({})", if low { "low acc" } else { "high acc" }),
        supported: lingua_languages().len(),
        load_secs: load,
        elapsed_secs: t.elapsed().as_secs_f64(),
        preds,
        supports: supports_lingua,
    }
}

fn load_and_sample(files: &[PathBuf], per_lang: usize, seed: u32) -> Vec<(Lang, String)> {
    let mut by_lang: HashMap<String, Vec<String>> = HashMap::new();
    for path in files {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        for line in text.lines() {
            let Some((code, text)) = line.split_once('\t') else { continue };
            if let Ok(lang) = code.trim().parse::<Lang>() {
                by_lang.entry(lang.code().to_string()).or_default().push(text.to_string());
            }
        }
    }
    let mut rng = simple_rng(seed);
    let mut rows: Vec<(Lang, String)> = Vec::new();
    for lang in Lang::ALL {
        let Some(texts) = by_lang.get(lang.code()) else { continue };
        let mut texts = texts.clone();
        shuffle(&mut texts, &mut rng);
        let take = if per_lang == 0 { texts.len() } else { per_lang.min(texts.len()) };
        rows.extend(texts[..take].iter().map(|t| (lang, t.clone())));
    }
    shuffle(&mut rows, &mut rng);
    rows
}

/// xorshift64 — tiny deterministic RNG; the sample must be identical across
/// runs so results are comparable.
fn simple_rng(seed: u32) -> u64 {
    (u64::from(seed)).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1
}

fn shuffle<T>(items: &mut [T], rng: &mut u64) {
    for i in (1..items.len()).rev() {
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;
        let j = (*rng % (i as u64 + 1)) as usize;
        items.swap(i, j);
    }
}
