//! Bulk-detection sweep over a markdown book: end-to-end throughput and the
//! language histogram of every sentence.
//!
//! One plan is prepared up front on the main thread (warming svod's
//! schedule/opt/kernel caches), then replicated per rayon worker — svod
//! plans are not cloneable, but the replicas only pay planning and buffer
//! allocation. Each worker owns its plan and input buffer, so batches never
//! contend; the optimizer strategy is passed programmatically via
//! `PrepareConfig` (beam vs heuristic), never through the BEAM env var.
//! Sentences are split from the markdown with our own rule-based
//! splitter (`spellman_detector::sent` — terminator runs, «closers»,
//! initials, decimals, dialogue-dash rules), with heading lines dropped
//! first; see the module docs for why not UAX #29.
//!
//! Memory note: every replica holds its own weights and buffers (~16 MB
//! each at the default k/batch).
//!
//! Usage:
//!   cargo run --release --example detect_md -- \
//!       --model ../../model --md ../../tests/voyna-i-mir.md \
//!       --threads 16 --batch 512 --k 1024 --beam 16

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use clap::Parser;
use rayon::prelude::*;
use svod_schedule::{HeuristicsConfig, OptStrategy, OptimizerConfig};
use svod_tensor::PrepareConfig;

use spellman_detector::{BulkDetector, Detection};

#[derive(Parser)]
struct Args {
    /// Model directory (model.json + model.safetensors); omitted = the
    /// default Hugging Face Hub repo (downloaded into the HF cache).
    #[arg(long)]
    model: Option<PathBuf>,
    /// Markdown file to sweep.
    #[arg(long, default_value = "../../tests/voyna-i-mir.md")]
    md: PathBuf,
    /// Rayon worker threads.
    #[arg(long, default_value_t = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))]
    threads: usize,
    /// Sentences per detect_batch call.
    #[arg(long, default_value_t = 512)]
    batch: usize,
    /// Per-document token budget (see BulkDetector::load).
    #[arg(long, default_value_t = 1024)]
    k: usize,
    /// Beam-optimizer width for plan compilation; 0 = heuristic optimizer.
    #[arg(long, default_value_t = 16)]
    beam: usize,
    /// Drop sentences shorter than this many chars.
    #[arg(long, default_value_t = 20)]
    min_chars: usize,
}

/// Split markdown text into sentences: heading lines are skipped, each
/// remaining paragraph line is segmented with `sent::split_line`
/// (terminator runs + closers + initials + decimals + dialogue dashes —
/// see the module docs). Letter/min-length filtering happens in `main`;
/// the fragments it drops cost nothing.
fn split_sentences(md: &str) -> Vec<String> {
    md.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .flat_map(spellman_detector::sent::split_line)
        .map(str::to_string)
        .collect()
}

/// Worker-local histogram: language -> sentence count, plus the uncertain
/// (below θ) tally.
#[derive(Default)]
struct Tally {
    langs: HashMap<&'static str, u64>,
    uncertain: u64,
}

impl Tally {
    fn record(&mut self, d: &Detection) {
        let code = d.lang.map(|l| l.code()).unwrap_or("und");
        *self.langs.entry(code).or_default() += 1;
        self.uncertain += u64::from(d.is_uncertain);
    }

    fn merge(mut self, other: Tally) -> Tally {
        for (code, n) in other.langs {
            *self.langs.entry(code).or_default() += n;
        }
        self.uncertain += other.uncertain;
        self
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let config: PrepareConfig = OptimizerConfig::builder()
        .heuristics(HeuristicsConfig {
            thread_count: args.threads,
            ..Default::default()
        })
        .strategy(if args.beam > 0 {
            OptStrategy::Beam { width: args.beam }
        } else {
            OptStrategy::Heuristic
        })
        .build()
        .into();

    let model_dir: PathBuf = match &args.model {
        Some(p) => p.clone(),
        None => {
            spellman_detector::hub::download_model(spellman_detector::hub::DEFAULT_HUB_REPO, None)?
        }
    };

    let md = std::fs::read_to_string(&args.md)?;
    let sentences: Vec<String> = split_sentences(&md)
        .into_iter()
        .filter(|s| s.chars().count() >= args.min_chars && s.chars().any(char::is_alphabetic))
        .collect();
    println!(
        "model={} md={} ({} KB)",
        model_dir.display(),
        args.md.display(),
        md.len() / 1024
    );
    println!("threads={} batch={} k={}", args.threads, args.batch, args.k);
    println!(
        "sentences: {} (min {} chars, letters required)",
        sentences.len(),
        args.min_chars
    );

    // Warm-up plan on the main thread: compiles (or replays the disk cache
    // for) the kernels once, so the replicas below only pay planning +
    // buffer allocation.
    let t0 = Instant::now();
    let mut plans: Vec<BulkDetector> = vec![BulkDetector::load_with_prepare_config(
        &model_dir, args.k, args.batch, &config,
    )?];
    println!("warm-up plan prepared in {:.2?}", t0.elapsed());

    let t1 = Instant::now();
    while plans.len() < args.threads {
        plans.push(BulkDetector::load_with_prepare_config(
            &model_dir, args.k, args.batch, &config,
        )?);
    }
    println!(
        "replicated to {} plans in {:.2?}",
        plans.len(),
        t1.elapsed()
    );

    let plans = Mutex::new(plans);
    let shard = sentences.len().div_ceil(args.threads).max(args.batch);
    let rp = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()?;

    let t = Instant::now();
    let tally = rp.install(|| {
        sentences
            .par_chunks(shard)
            .map(|shard| {
                let mut det = plans.lock().unwrap().pop().expect("more shards than plans");
                let mut tally = Tally::default();
                for chunk in shard.chunks(args.batch) {
                    let texts: Vec<&str> = chunk.iter().map(String::as_str).collect();
                    for d in det.detect_batch(&texts).expect("detect_batch") {
                        tally.record(&d);
                    }
                }
                tally
            })
            .reduce(Tally::default, Tally::merge)
    });
    let dt = t.elapsed();

    let total: u64 = tally.langs.values().sum();
    println!(
        "detected {total} sentences in {:.3?}: {:.0} sentences/s, {:.1} MB/s",
        dt,
        total as f64 / dt.as_secs_f64(),
        md.len() as f64 / 1024.0 / 1024.0 / dt.as_secs_f64()
    );
    println!();
    println!("language  count    share");
    let mut rows: Vec<_> = tally.langs.into_iter().collect();
    rows.sort_by_key(|a| std::cmp::Reverse(a.1));
    for (code, n) in rows {
        println!(
            "  {:>4}  {:>7}  {:>5.1}%",
            code,
            n,
            100.0 * n as f64 / total as f64
        );
    }
    println!(
        "uncertain (below θ): {} ({:.2}%)",
        tally.uncertain,
        100.0 * tally.uncertain as f64 / total as f64
    );
    Ok(())
}
