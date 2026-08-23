//! int8 vs f16 bulk-path performance probe (same gather+sum graph shape).
//!
//! The accuracy side is settled (the `spellman-train quantize` rewrite): per-column
//! symmetric int8 is lossless to −0.01pp, and per-column scales factor out
//! of the K-sum (`logit_c = s_c · Σ ±q`), so the int8 graph stays a pure
//! gather + cast(i8→i16→i32, the lattice allows no direct i8→i32) + sum,
//! with the 30 scale multiplies at host read-out.
//!
//! Measures, for the same synthetic signed-bucket batch:
//! - featurize-only (single-threaded reference; the real path rayon-izes it),
//! - execute-only f16 plan vs execute-only int8 plan,
//! - end-to-end `BulkDetector::detect_batch` for context,
//!
//! and cross-checks the two plans' logits (i32 sum × column scale vs f16 sum).
//!
//! Usage:
//!   BEAM=2 cargo run --release --example int8_bench -- model/ [k] [batch] [reps]

// svod's tensor Result crosses this probe's helper API.
#![allow(clippy::result_large_err)]

use std::path::PathBuf;
use std::time::Instant;

use svod_ir::SInt;
use svod_macros::jit_wrapper;
use svod_model::jit::InputSpec;
use svod_tensor::{BoundVariable, Tensor};

use spellman_detector::BulkDetector;
use spellman_detector::features::{FeatureConfig, bucket_tokens};
use spellman_detector::hash::FeatureHasher;
use spellman_detector::jit::{SpellmanJit, SpellmanModel, f16_to_f32};
use spellman_detector::model::Model;

/// Int8 weight model: `q` block then `-q` block, `[2*(D+1), C]`, per-column
/// scales kept host-side (applied at read-out).
struct Int8Model {
    table: Tensor,
    scales: Vec<f32>,
}

impl Int8Model {
    fn from_table(table: &[f32], d: usize, cols: usize) -> Int8Model {
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
        Int8Model { table, scales }
    }

    fn forward_batch(
        &self,
        idx: &Tensor,
        b: &BoundVariable,
    ) -> Result<Tensor, svod_tensor::error::Error> {
        let bv = b.as_sint();
        let idx = idx.try_shrink([Some((SInt::Const(0), bv.clone())), None])?;
        let idx = idx.cast(svod_dtype::DType::Int64)?;
        let rows = self.table.embedding(&idx)?; // [b, K, C] i8
        // Int8 casts only to Int16 on the lattice; widen twice before the
        // K-sum (i16 would overflow past K=259 at |q|=127).
        rows.cast(svod_dtype::DType::Int16)?
            .cast(svod_dtype::DType::Int32)?
            .sum(1)
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
    let beam = std::env::var("BEAM").unwrap_or_default();
    println!(
        "model={} k={k} batch={batch} reps={reps} BEAM={beam:?}",
        model_dir.display()
    );

    let model = Model::load(&model_dir)?;
    let d = model.num_buckets() as usize;
    let cols = spellman_language::NUM_LANGS;

    // ---- plans ------------------------------------------------------------
    let mut f16_plan = SpellmanJit::new(SpellmanModel::from_table(&model.table, cols)?);
    f16_plan.prepare(InputSpec::i32(&[batch, k]))?;

    let int8 = Int8Model::from_table(&model.table, d, cols);
    let i8_scales = int8.scales.clone();
    let mut i8_plan = Int8Jit::new(int8);
    i8_plan.prepare(InputSpec::i32(&[batch, k]))?;

    // Synthetic signed bucket batch: deterministic LCG mix of buckets,
    // signs alternating — same values into both plans.
    let fill = |flat: &mut [i32]| {
        let mut state = 0x9E37_79B9u64;
        for (i, slot) in flat.iter_mut().enumerate() {
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
    };
    {
        let mut view = f16_plan.idx_mut()?.as_array_mut::<i32>()?;
        fill(view.as_slice_mut().unwrap());
    }
    {
        let mut view = i8_plan.idx_mut()?.as_array_mut::<i32>()?;
        fill(view.as_slice_mut().unwrap());
    }

    // ---- correctness cross-check (first 8 rows) ---------------------------
    {
        f16_plan.execute_with_vars(&[("b", batch as i64)])?;
        let mut f16_sums = vec![0u16; batch * cols];
        f16_plan
            .output()?
            .copyout_prefix(bytemuck::cast_slice_mut(&mut f16_sums))?;
        let mut i32_sums = vec![0i32; batch * cols];
        i8_plan.execute_with_vars(&[("b", batch as i64)])?;
        i8_plan
            .output()?
            .copyout_prefix(bytemuck::cast_slice_mut(&mut i32_sums))?;

        let mut worst_abs = 0f32;
        let mut worst_rel = 0f32;
        for row in 0..8 {
            for c in 0..cols {
                let a = f16_to_f32(f16_sums[row * cols + c]);
                let b = i32_sums[row * cols + c] as f32 * i8_scales[c];
                worst_abs = worst_abs.max((a - b).abs());
                // Relative error only means something on decisive logits:
                // random-sign synthetic buckets produce near-zero sums.
                if a.abs() > 50.0 {
                    worst_rel = worst_rel.max((a - b).abs() / a.abs());
                }
            }
        }
        println!(
            "cross-check (8 rows): max abs diff {worst_abs:.2}, max rel diff on |logit|>50: {worst_rel:.4}"
        );
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
            plan.execute_with_vars(&[("b", batch as i64)])?;
            plan.output()?.copyout_prefix(&mut sink)?;
            Ok(())
        });
    }
    {
        let plan = &mut i8_plan;
        let mut sink = vec![0u8; 8];
        time_exec("int8 gather+cast+sum", reps, batch, || {
            plan.execute_with_vars(&[("b", batch as i64)])?;
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
