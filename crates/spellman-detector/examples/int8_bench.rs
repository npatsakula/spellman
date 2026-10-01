//! int8 vs f16 bulk-path performance probe (same gather+sum graph shape).
//!
//! The accuracy side is settled (the `spellman-train quantize` rewrite): per-column
//! symmetric int8 is lossless to −0.01pp, and per-column scales factor out
//! of the K-sum (`logit_c = s_c · Σ ±q`), so the int8 graph stays a pure
//! gather + widen + sum, with the 30 scale multiplies at host read-out.
//! Two widenings are measured:
//! - `int8→i32`: a plain i8 sum, which svod (alpha.7+) accumulates in i32 —
//!   exact;
//! - `int8→f16`: one cast(i8→f16) + sum, the same path the f16 graph takes
//!   (svod accumulates in f32 and casts back to f16). `|Σq| ≤ K·127` stays
//!   under the f16 max (65504) for K ≤ 512; above that a decisive row could
//!   overflow on the cast back.
//!
//! Every plan is compiled with the batch fixed (`with_b_fixed`), exactly as
//! `BulkDetector` ships, so svod threads the gather over the batch axis.
//!
//! Measures, for the same signed-bucket batch:
//! - featurize-only (single-threaded reference; the real path rayon-izes it),
//! - execute-only f16 plan vs both int8 plans,
//! - end-to-end `BulkDetector::detect_batch` for context,
//!
//! and cross-checks the int8 plans' logits against the f16 sums.
//!
//! The index batch is uniform-random buckets by default (worst case for the
//! cache); pass a `lang<TAB>text` TSV to fill it with real featurized rows
//! instead (Zipf-distributed n-grams keep hot rows cached).
//!
//! Usage:
//!   BEAM=16 cargo run --release --example int8_bench -- model/ [k] [batch] [reps] [rows.tsv]

// svod's tensor Result crosses this probe's helper API.
#![allow(clippy::result_large_err)]

use std::path::PathBuf;
use std::time::Instant;

use svod_ir::SInt;
use svod_macros::jit_wrapper;
use svod_model::jit::InputSpec;
use svod_tensor::{BoundVariable, Tensor};

use spellman_detector::BulkDetector;
use spellman_detector::features::{FeatureConfig, bucket_tokens, fill_signed_indices};
use spellman_detector::hash::FeatureHasher;
use spellman_detector::jit::{SpellmanJit, SpellmanModel, f16_to_f32};
use spellman_detector::model::Model;

/// K-sum accumulator of an int8 plan.
#[derive(Copy, Clone)]
enum Accum {
    /// cast(i8→i16→i32), exact i32 sum.
    I32,
    /// cast(i8→f16), f16 sum.
    F16,
}

/// Int8 weight model: `q` block then `-q` block, `[2*(D+1), C]`, per-column
/// scales kept host-side (applied at read-out).
struct Int8Model {
    table: Tensor,
    scales: Vec<f32>,
    accum: Accum,
}

impl Int8Model {
    fn from_table(table: &[f32], d: usize, cols: usize, accum: Accum) -> Int8Model {
        let mut scales = vec![0f32; cols];
        for r in 0..=d {
            for c in 0..cols {
                scales[c] = scales[c].max(table[r * cols + c].abs());
            }
        }
        for s in &mut scales {
            *s /= 127.0;
        }
        let mut q = vec![0i8; (d + 1) * cols];
        for (i, v) in table.iter().enumerate() {
            let s = scales[i % cols];
            q[i] = (v / s).round().clamp(-127.0, 127.0) as i8;
        }
        // ±q blocks concatenated, mirroring the f16 cat([P, -P]) layout.
        let mut both = Vec::with_capacity(2 * q.len());
        both.extend_from_slice(&q);
        both.extend(q.iter().map(|v| -v));
        let table = Tensor::from_slice(&both)
            .try_reshape([2 * (d + 1) as isize, cols as isize])
            .unwrap();
        Int8Model {
            table,
            scales,
            accum,
        }
    }

    fn forward_batch(
        &self,
        idx: &Tensor,
        b: &BoundVariable,
    ) -> Result<Tensor, svod_tensor::error::Error> {
        let bv = b.as_sint();
        let idx = idx.try_shrink([Some((SInt::Const(0), bv.clone())), None])?;
        // i32 indices: an i64 cast defeats svod's one-hot collapse (see jit.rs).
        let rows = self.table.embedding(&idx)?; // [b, K, C] i8
        match self.accum {
            // `sum` promotes an i8 accumulator to i32 on its own (svod
            // alpha.7) — no explicit widening chain.
            Accum::I32 => rows.sum(1),
            Accum::F16 => rows.cast(svod_dtype::DType::Float16).sum(1),
        }
    }
}

jit_wrapper! {
    Int8Jit(Int8Model) {
        idx: Tensor,

        vars {
            b: (1, 4096),
        }

        build(idx, b) {
            model.forward_batch(idx, &b)
        }
    }
}

const TEXTS: [&str; 4] = [
    "Съешь ещё этих мягких французских булок, да выпей чаю. Быстрая бурая лиса прыгает через ленивую собаку.",
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
    "Швидкість світла у вакуумі є фундаментальною фізичною константою, виміряною з високою точністю.",
    "Абдыгапардың әжейі өңірлі ғажайып үй құдықын шолып жүр. Абдыгапардың әжейі өңірлі ғажайып.",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir: PathBuf = args.next().unwrap_or_else(|| "model".into()).into();
    let k: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(128);
    let batch: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(256);
    let reps: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(50);
    let tsv: Option<PathBuf> = args.next().map(PathBuf::from);
    let beam = std::env::var("BEAM").unwrap_or_default();
    println!(
        "model={} k={k} batch={batch} reps={reps} BEAM={beam:?} idx={}",
        model_dir.display(),
        tsv.as_ref()
            .map_or("uniform-random".to_string(), |p| p.display().to_string())
    );

    let model = Model::load(&model_dir)?;
    let d = model.num_buckets() as usize;
    let cols = spellman_language::NUM_LANGS;
    println!(
        "table: 2^{} buckets — f16 ±P {:.1} MB, int8 ±q {:.1} MB",
        model.log2_d,
        (2 * (d + 1) * cols * 2) as f64 / 1e6,
        (2 * (d + 1) * cols) as f64 / 1e6
    );

    // ---- plans: batch fixed at compile time, as BulkDetector ships -------
    let mut f16_plan =
        SpellmanJit::new(SpellmanModel::from_table(&model.table, cols)?).with_b_fixed(batch);
    f16_plan.prepare(InputSpec::i32(&[batch, k]))?;

    let int8 = Int8Model::from_table(&model.table, d, cols, Accum::I32);
    let i8_scales = int8.scales.clone();
    let mut i32_plan = Int8Jit::new(int8).with_b_fixed(batch);
    i32_plan.prepare(InputSpec::i32(&[batch, k]))?;

    let mut i8f16_plan =
        Int8Jit::new(Int8Model::from_table(&model.table, d, cols, Accum::F16)).with_b_fixed(batch);
    i8f16_plan.prepare(InputSpec::i32(&[batch, k]))?;

    // Signed bucket batch, the same values into every plan: real featurized
    // rows (truncated / padded to K) when a TSV is given, else a
    // deterministic LCG mix of uniform buckets with alternating signs.
    let mut idx = vec![d as i32; batch * k];
    match &tsv {
        Some(path) => {
            let text = std::fs::read_to_string(path)?;
            let mut docs = text
                .lines()
                .filter_map(|l| l.split_once('\t').map(|(_, t)| t))
                .cycle();
            for row in idx.chunks_mut(k) {
                fill_signed_indices(
                    docs.next().ok_or("empty TSV")?,
                    &model.features,
                    &model.hasher,
                    model.log2_d,
                    k,
                    row,
                );
            }
        }
        None => {
            let mut state = 0x9E37_79B9u64;
            for (i, slot) in idx.iter_mut().enumerate() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bucket = ((state >> 33) as usize) % (d + 1);
                *slot = if (i / k + i).is_multiple_of(2) {
                    bucket as i32
                } else {
                    (d + 1 + bucket) as i32
                };
            }
        }
    }
    {
        let mut view = f16_plan.idx_mut()?.as_array_mut::<i32>()?;
        view.as_slice_mut().unwrap().copy_from_slice(&idx);
    }
    {
        let mut view = i32_plan.idx_mut()?.as_array_mut::<i32>()?;
        view.as_slice_mut().unwrap().copy_from_slice(&idx);
    }
    {
        let mut view = i8f16_plan.idx_mut()?.as_array_mut::<i32>()?;
        view.as_slice_mut().unwrap().copy_from_slice(&idx);
    }

    // ---- correctness cross-check (every row) ------------------------------
    {
        f16_plan.execute()?;
        let mut f16_sums = vec![0u16; batch * cols];
        f16_plan
            .output()?
            .copyout_prefix(bytemuck::cast_slice_mut(&mut f16_sums))?;
        i32_plan.execute()?;
        let mut i32_sums = vec![0i32; batch * cols];
        i32_plan
            .output()?
            .copyout_prefix(bytemuck::cast_slice_mut(&mut i32_sums))?;
        i8f16_plan.execute()?;
        let mut i8f16_sums = vec![0u16; batch * cols];
        i8f16_plan
            .output()?
            .copyout_prefix(bytemuck::cast_slice_mut(&mut i8f16_sums))?;

        let reference: Vec<f32> = f16_sums.iter().map(|&s| f16_to_f32(s)).collect();
        let check = |label: &str, logit: &dyn Fn(usize) -> f32| {
            let mut worst_abs = 0f32;
            let mut worst_rel = 0f32;
            let mut argmax_flips = 0usize;
            for row in 0..batch {
                let (mut best_a, mut best_b) = (0usize, 0usize);
                for c in 0..cols {
                    let a = reference[row * cols + c];
                    let b = logit(row * cols + c);
                    worst_abs = worst_abs.max((a - b).abs());
                    // Relative error only means something on decisive
                    // logits: random-sign buckets produce near-zero sums.
                    if a.abs() > 50.0 {
                        worst_rel = worst_rel.max((a - b).abs() / a.abs());
                    }
                    if a > reference[row * cols + best_a] {
                        best_a = c;
                    }
                    if b > logit(row * cols + best_b) {
                        best_b = c;
                    }
                }
                argmax_flips += usize::from(best_a != best_b);
            }
            println!(
                "cross-check {label} vs f16 ({batch} rows): max abs diff {worst_abs:.2}, \
                 max rel diff on |logit|>50: {worst_rel:.4}, argmax flips {argmax_flips}"
            );
        };
        check("int8→i32", &|i| i32_sums[i] as f32 * i8_scales[i % cols]);
        check("int8→f16", &|i| {
            f16_to_f32(i8f16_sums[i]) * i8_scales[i % cols]
        });
    }

    // ---- timings -----------------------------------------------------------
    fn time_exec<F: FnMut() -> Result<(), Box<dyn std::error::Error>>>(
        label: &str,
        reps: usize,
        batch: usize,
        mut run: F,
    ) {
        for _ in 0..5 {
            run().unwrap();
        }
        let t = Instant::now();
        for _ in 0..reps {
            run().unwrap();
        }
        let dt = t.elapsed();
        println!(
            "  {:<24} {:>10.1} µs/exec  {:>7.2} µs/sample",
            label,
            dt.as_micros() as f64 / reps as f64,
            dt.as_micros() as f64 / (reps * batch) as f64
        );
    }

    println!("execute-only (plan, synthetic idx):");
    // The fence matters: on async devices (AMD/GPU) execute_with_vars only
    // queues work — without reading a byte of the output the timed loop
    // measures submission latency (~1 µs/exec), not execution.
    {
        let plan = &mut f16_plan;
        let mut sink = vec![0u8; 8];
        time_exec("f16 gather+sum", reps, batch, || {
            plan.execute()?;
            plan.output()?.copyout_prefix(&mut sink)?;
            Ok(())
        });
    }
    for (label, plan) in [
        ("int8→i32 gather+sum", &mut i32_plan),
        ("int8→f16 gather+sum", &mut i8f16_plan),
    ] {
        let mut sink = vec![0u8; 8];
        time_exec(label, reps, batch, || {
            plan.execute()?;
            plan.output()?.copyout_prefix(&mut sink)?;
            Ok(())
        });
    }

    // Featurize-only, single-threaded (the real path rayon-izes across rows).
    {
        let cfg = FeatureConfig::default();
        let hasher = FeatureHasher {
            id: model.hasher.id,
            seed: model.hasher.seed,
        };
        let docs: Vec<&str> = (0..batch).map(|i| TEXTS[i % TEXTS.len()]).collect();
        for d_ in docs.iter().take(4) {
            std::hint::black_box(bucket_tokens(d_, &cfg, &hasher, model.log2_d).len());
        }
        let t = Instant::now();
        for d_ in &docs {
            std::hint::black_box(bucket_tokens(d_, &cfg, &hasher, model.log2_d).len());
        }
        let dt = t.elapsed();
        println!(
            "  {:<24} {:>10.1} µs total {:>7.2} µs/doc (1 thread)",
            "featurize-only",
            dt.as_micros() as f64,
            dt.as_micros() as f64 / batch as f64
        );
    }

    // End-to-end reference through the shipped bulk path.
    {
        let mut det = BulkDetector::load(&model_dir, k, batch)?;
        let docs: Vec<String> = (0..batch)
            .map(|i| TEXTS[i % TEXTS.len()].to_string())
            .collect();
        let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        det.detect_batch(&refs)?;
        let t = Instant::now();
        for _ in 0..reps {
            det.detect_batch(&refs)?;
        }
        let dt = t.elapsed();
        println!(
            "  {:<24} {:>10.1} µs/exec  {:>7.2} µs/sample",
            "end-to-end detect_batch",
            dt.as_micros() as f64 / reps as f64,
            dt.as_micros() as f64 / (reps * batch) as f64
        );
    }
    Ok(())
}
