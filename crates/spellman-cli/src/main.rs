//! `spellman` — command-line interface for the detector.
//!
//! Subcommands:
//! - `detect` — read stdin, write ISO 639-3 codes: the whole input as one
//!   document by default, one document per line with `--lines`, `und` when
//!   no supported-script letters are found;
//! - `eval`   — accuracy + throughput over eval TSV files (`code<TAB>text`);
//! - `bench`  — probe detections + steady-state timing of the bulk plan
//!   (add `--single` for the B=1 plan).
//!
//! The language inventory is used exclusively through spellman-detector's
//! re-exports — this crate has no direct `spellman-language` dependency.

use std::error::Error;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Args, Parser, Subcommand};
use spellman_detector::{BulkDetector, Detection, Lang, SingleDetector};

/// Model-backed runtime knobs shared by the subcommands.
#[derive(Args)]
struct PlanArgs {
    /// Per-document token budget K; longer documents are truncated.
    #[arg(long, default_value_t = 1024)]
    k: usize,
    /// Batch size; documents are scored in chunks of this many.
    #[arg(long, default_value_t = 1024)]
    max_batch: usize,
}

impl PlanArgs {
    /// Batch size clamped to something usable.
    fn batch(&self) -> usize {
        self.max_batch.max(1)
    }
}

#[derive(Parser)]
#[command(
    name = "spellman",
    version,
    about = "Cyrillic-optimized language detection (30 classes, folded fastText-style model, svod JIT)"
)]
struct Cli {
    /// Model directory (model.json + model.safetensors), or
    /// `hf:<owner>/<repo>[/variant]` to fetch it from the Hugging Face Hub.
    #[arg(long, global = true, default_value = "model", env = "SPELLMAN_MODEL")]
    model: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Detect the language of stdin; prints one ISO 639-3 code (`und` when
    /// the text contains no supported-script letters).
    Detect {
        /// Treat each input line as its own document (one code per line).
        #[arg(long)]
        lines: bool,
        /// Print JSON records (lang/name/confidence/uncertain) instead of bare codes.
        #[arg(long)]
        json: bool,
        #[command(flatten)]
        plan: PlanArgs,
    },
    /// Accuracy + throughput over eval TSV files (`code<TAB>text` per line).
    Eval {
        /// Eval TSV files to score.
        #[arg(required = true)]
        files: Vec<PathBuf>,
        #[command(flatten)]
        plan: PlanArgs,
    },
    /// Probe texts, detections, and steady-state timing of the bulk plan.
    Bench {
        #[command(flatten)]
        plan: PlanArgs,
        /// Timed batch repeats.
        #[arg(long, default_value_t = 20)]
        repeats: usize,
        /// Also time the B=1 single-document plan (2000 documents).
        #[arg(long)]
        single: bool,
    },
}

/// Resolve `--model` to a local model directory: `hf:<owner>/<repo>[/variant]`
/// goes through the detector crate's Hub support (standard HF cache, replayed
/// when warm); anything else is a local path, used as-is.
fn resolve_model(spec: &str) -> Result<PathBuf, Box<dyn Error>> {
    match spellman_detector::hub::parse_hub_ref(spec) {
        Some((repo, variant)) => {
            Ok(spellman_detector::hub::download_model(&repo, variant.as_deref())?)
        }
        None => Ok(spec.into()),
    }
}

fn main() {
    let cli = Cli::parse();
    let model_dir = match resolve_model(&cli.model.to_string_lossy()) {
        Ok(dir) => dir,
        Err(err) => {
            eprintln!("spellman: {err}");
            std::process::exit(1);
        }
    };
    let result = match &cli.command {
        Command::Detect { lines, json, plan } => detect(&model_dir, *lines, *json, plan),
        Command::Eval { files, plan } => eval(&model_dir, files, plan),
        Command::Bench { plan, repeats, single } => bench(&model_dir, plan, *repeats, *single),
    };
    if let Err(err) = result {
        eprintln!("spellman: {err}");
        std::process::exit(1);
    }
}

/// ISO 639-3 code of a detection; `und` (undetermined) when the router
/// found no letters from a supported script.
fn code_of(d: &Detection) -> &str {
    d.lang.map(Lang::code).unwrap_or("und")
}

fn write_detection(out: &mut impl Write, d: &Detection, json: bool) -> io::Result<()> {
    if json {
        let record = serde_json::json!({
            "lang": code_of(d),
            "name": d.lang.map(Lang::name).unwrap_or("Undetermined"),
            "confidence": d.confidence,
            "uncertain": d.is_uncertain,
        });
        writeln!(out, "{record}")
    } else {
        writeln!(out, "{}", code_of(d))
    }
}

fn detect(model_dir: &Path, lines: bool, json: bool, plan: &PlanArgs) -> Result<(), Box<dyn Error>> {
    let mut input = String::new();
    let mut stdin = io::stdin().lock();
    stdin.read_to_string(&mut input)?;

    let mut detector = BulkDetector::load(model_dir, plan.k, plan.batch())?;
    let stdout = io::stdout();
    let mut out = stdout.lock();

    if lines {
        let docs: Vec<&str> = input.lines().collect();
        for chunk in docs.chunks(plan.batch()) {
            for d in detector.detect_batch(chunk)? {
                write_detection(&mut out, &d, json)?;
            }
        }
    } else {
        let text = input.trim();
        let d = &detector.detect_batch(&[text])?[0];
        write_detection(&mut out, d, json)?;
    }
    out.flush()?;
    Ok(())
}

fn eval(model_dir: &Path, files: &[PathBuf], plan: &PlanArgs) -> Result<(), Box<dyn Error>> {
    let mut rows: Vec<(Lang, String)> = Vec::new();
    let mut skipped = 0usize;
    for path in files {
        let text = std::fs::read_to_string(path)?;
        for line in text.lines() {
            let Some((code, text)) = line.split_once('\t') else {
                skipped += 1;
                continue;
            };
            match code.trim().parse::<Lang>() {
                Ok(lang) => rows.push((lang, text.to_string())),
                Err(_) => skipped += 1,
            }
        }
    }
    if rows.is_empty() {
        return Err(format!("no usable rows (skipped {skipped}) in {files:?}").into());
    }
    let note = if skipped > 0 { format!(" ({skipped} rows skipped)") } else { String::new() };
    println!("eval: {} samples from {} file(s){note}", rows.len(), files.len());

    let mut detector = BulkDetector::load(model_dir, plan.k, plan.batch())?;
    let refs: Vec<&str> = rows.iter().map(|(_, t)| t.as_str()).collect();

    let t = Instant::now();
    let mut ok = 0usize;
    let mut idx = 0usize;
    for chunk in refs.chunks(plan.batch()) {
        for d in detector.detect_batch(chunk)? {
            if d.lang == Some(rows[idx].0) {
                ok += 1;
            }
            idx += 1;
        }
    }
    let dt = t.elapsed();
    println!("accuracy:   {}/{} ({:.2}%)", ok, rows.len(), 100.0 * ok as f64 / rows.len() as f64);
    println!("throughput: {dt:?} total, {:.1} µs/sample", dt.as_micros() as f64 / rows.len() as f64);
    Ok(())
}

/// Probe texts: one per hard pair plus the script-routed and no-script
/// cases (indices 8/9/10 are asserted below — keep them last).
const PROBES: [&str; 11] = [
    "Съешь ещё этих мягких французских булок, да выпей чаю",
    "The quick brown fox jumps over the lazy dog",
    "Швидкість світла у вакуумі є фундаментальною фізичною константою",
    "Ґей, хлопці, не вспію на вербі гнізде, а на світанку повені",
    "Абдыгапардың әжейі өңірлі ғажайып үй құдықын шолып жүр",
    "El veloz murciélago hindú comía feliz cardillo y kiwi",
    "Portez ce vieux whisky au juge blond qui fume",
    "Victor jagt zwölf Boxkämpfer quer über den großen Sylter Deich",
    "快速棕色狐狸跳过了懒狗", // cmn — script-routed, never hits the plan
    "こんにちは世界",         // jpn — script-routed
    "12345 !!!",              // no supported-script letters
];

fn bench(model_dir: &Path, plan: &PlanArgs, repeats: usize, single: bool) -> Result<(), Box<dyn Error>> {
    let batch = plan.batch().max(PROBES.len());
    let t = Instant::now();
    let mut bulk = BulkDetector::load(model_dir, plan.k, batch)?;
    let prepare = t.elapsed();

    let results = bulk.detect_batch(&PROBES)?;
    // Script routing must bypass the plan entirely for the last three
    // probes: Han → cmn, kana → jpn, no letters → no language.
    assert_eq!(results[8].lang, Some(Lang::Cmn));
    assert_eq!(results[9].lang, Some(Lang::Jpn));
    assert_eq!(results[10].lang, None);

    println!("prepare (load + schedule + codegen): {prepare:?}");
    println!("probes (k={}, batch={batch}):", plan.k);
    for (text, d) in PROBES.iter().zip(&results) {
        let flag = if d.is_uncertain { "  (uncertain)" } else { "" };
        println!("  {:>40.40}  {:?}  conf={:.3}{flag}", text, d.lang, d.confidence);
    }

    let mut times = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let t = Instant::now();
        bulk.detect_batch(&PROBES)?;
        times.push(t.elapsed());
    }
    times.sort();
    println!("bulk steady state, {repeats} batches of {}:", PROBES.len());
    println!("  min {:?}  median {:?}", times[0], times[times.len() / 2]);

    if single {
        let mut sd = SingleDetector::load(model_dir, plan.k)?;
        for text in PROBES.iter().cycle().take(64) {
            sd.detect(text)?;
        }
        let iters = 2000usize;
        let t = Instant::now();
        for i in 0..iters {
            sd.detect(PROBES[i % PROBES.len()])?;
        }
        let ns = t.elapsed().as_nanos() as f64 / iters as f64;
        println!("single (B=1 plan): {ns:.0} ns/doc over {iters} docs");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detection(lang: Option<Lang>, confidence: f32, uncertain: bool) -> Detection {
        Detection { lang, confidence, is_uncertain: uncertain }
    }

    #[test]
    fn bare_code_output() {
        let mut buf = Vec::new();
        write_detection(&mut buf, &detection(Some(Lang::Kaz), 0.99, false), false).unwrap();
        assert_eq!(String::from_utf8(buf).unwrap(), "kaz\n");
    }

    #[test]
    fn undetermined_and_json_output() {
        let mut buf = Vec::new();
        write_detection(&mut buf, &detection(None, 0.0, true), false).unwrap();
        assert_eq!(String::from_utf8(buf).unwrap(), "und\n");

        let mut buf = Vec::new();
        write_detection(&mut buf, &detection(Some(Lang::Kaz), 0.99, false), true).unwrap();
        let record: serde_json::Value =
            serde_json::from_str(String::from_utf8(buf).unwrap().trim()).unwrap();
        assert_eq!(record["lang"], "kaz");
        assert_eq!(record["name"], "Kazakh");
        assert_eq!(record["uncertain"], false);
    }
}
