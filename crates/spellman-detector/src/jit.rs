//! Bulk batched detection as a compiled svod execution plan.
//!
//! The graph is deliberately minimal — gather and one reduction, pure fp16
//! end to end (ARM/NEON native on Apple Silicon; no cast kernels in the
//! replayed graph; f32 conversion happens once at host read-out):
//!
//! ```text
//! idx [b, K] i32 ──gather──> table rows [b, K, C] f16 ──sum over K──> [b, C] f16
//! ```
//!
//! Mean-pooling (÷ token count) and the bias add run host-side at read-out:
//! featurization already knows each document's exact token count, so the
//! graph needs no count computation at all.
//!
//! `K` (tokens per document, truncated/zero-padded) is baked into the plan as
//! a compile-time constant so the scheduler can specialize every kernel for
//! the fixed sequence length; only the batch dim stays symbolic and is
//! rebound per call with `execute_with_vars(&[("b", n)])`.
//!
//! Signed hashing is folded into the table layout: the gather table is
//! `[2*(D+1), C]` with rows `0..=D` equal to `P` and rows `D+1..=2D+1` equal
//! to `-P`, so a token's sign selects the row block and the graph needs no
//! multiplies. The padding row lives at index `D` (all-zero) in both blocks.

use svod_ir::SInt;
use svod_macros::jit_wrapper;
use svod_model::jit::InputSpec;
use svod_tensor::{BoundVariable, Tensor};

use crate::Detection;
use spellman_language::{Lang, NUM_LANGS};

use crate::model::Model;

/// Weight tensors for the JIT graph (device-resident, lazily computed).
pub struct SpellmanModel {
    /// `[2*(D+1), NUM_LANGS]` — `P` block then `-P` block, fp16.
    table: Tensor,
}

impl SpellmanModel {
    /// Build from the canonical (dequantized) host table `[D+1, NUM_LANGS]`
    /// f32 — the single representation the loader resolves from any storage
    /// precision. The f16 cast and the ±P doubling (`cat`) stay in the
    /// graph, so the doubled table is fused into the JIT plan's constant
    /// realization instead of staging through host memory twice.
    pub fn from_table(table: &[f32], num_langs: usize) -> Result<SpellmanModel, svod_tensor::error::Error> {
        let rows = table.len() / num_langs;
        let p = Tensor::from_slice(table)
            .try_reshape(&[rows as isize, num_langs as isize])?
            .cast(svod_dtype::DType::Float16)?;
        let neg = p.try_neg()?;
        let jit_table = Tensor::cat(&[&p, &neg], 0)?;
        Ok(SpellmanModel { table: jit_table })
    }

    /// Build the gather-sum graph over a `[b, K]` bucket-index batch.
    /// Returns raw fp16 row-sums `[b, C]`; mean-pooling, the bias add, the
    /// softmax, and the argmax all run host-side at read-out (30 floats per
    /// document, with the token counts featurization already computed).
    ///
    /// The whole graph is fp16 (no cast kernels on the replay path — ARM
    /// NEON fp16 native). Precision is safe here: each gathered value is one
    /// exact fp16 load, the K-term sums stay far from fp16 range, and LID
    /// logit margins are wide. Padding tokens gather the all-zero row, so
    /// the sum is unaffected by padding.
    pub fn forward_batch(
        &self,
        idx: &Tensor,
        b: &BoundVariable,
    ) -> Result<Tensor, svod_tensor::error::Error> {
        let bv = b.as_sint();
        // The prepare-time placeholder is allocated at max batch; shrink to
        // the symbolic batch for kernel specialization at bind time.
        let idx = idx.try_shrink([Some((SInt::Const(0), bv.clone())), None])?;
        let idx = idx.cast(svod_dtype::DType::Int64)?;
        // Row-gather the ±P table: [b, K] -> [b, K, C]. `embedding` needs a
        // concrete index shape, which is exactly why K stays a JIT constant.
        let rows = self.table.embedding(&idx)?;
        Ok(rows.sum(1)?)
    }
}

jit_wrapper! {
    SpellmanJit(SpellmanModel) {
        idx: Tensor,

        vars {
            b: (1, 4096),
        }

        build(idx, b) {
            model.forward_batch(idx, &b)
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BulkError {
    #[error("jit: {0}")]
    Jit(#[from] svod_model::jit::JitError),
    #[error("tensor: {0}")]
    Tensor(#[from] svod_tensor::error::Error),
    #[error("device: {0}")]
    Device(#[from] svod_device::error::Error),
    #[error("model: {0}")]
    Model(#[from] crate::model::ModelError),
    #[error("state: {0}")]
    State(#[from] svod_model::state::Error),
    #[error("buffer view: {0}")]
    View(String),
    #[error("batch of {len} exceeds max_batch {max}")]
    BatchTooLarge { len: usize, max: usize },
}

/// Batched detector over a compiled svod plan.
///
/// Documents are truncated/padded to `k` bucket tokens on the CPU (cheap
/// sequential byte work), then scored in one fused kernel launch per batch.
/// Script-unique languages (jpn/cmn/hin/ara) are routed on the CPU and never
/// reach the plan.
///
/// Device placement follows svod's loading convention: weights live on the
/// default device at load time, so call `svod_tensor::set_default_device`
/// before constructing if the plan should run on a GPU.
pub struct BulkDetector {
    jit: SpellmanJit,
    model: Model,
    k: usize,
    max_batch: usize,
}

impl BulkDetector {
    /// Load from a model directory and compile the plan. `k` is the
    /// per-document token budget (try 1024+ for paragraph text); `max_batch`
    /// sizes the input buffer and the symbolic bound on `b`. The state dict
    /// is loaded once and feeds both the host-side feature config and the
    /// device-resident fp16 weight tensors.
    ///
    /// The input buffer is host-mapped (not device-local): featurization
    /// writes bucket indices directly into the plan's buffer through a
    /// zero-copy typed view, skipping a staging allocation and a copy.
    /// AMD's host-visible VRAM mapping supports the same path; a CUDA
    /// device-local buffer would need `copyin` staging instead.
    pub fn load(dir: &std::path::Path, k: usize, max_batch: usize) -> Result<BulkDetector, BulkError> {
        let metadata = crate::model::read_metadata(dir)?;
        let sd = svod_model::state::load_safetensors_dir(dir)?;
        let model = Model::from_state_dict(&sd, metadata)?;
        let inner = SpellmanModel::from_table(&model.table, NUM_LANGS)?;
        let mut jit = SpellmanJit::new(inner);
        jit.prepare(InputSpec::i32(&[max_batch, k]))?;
        Ok(BulkDetector { jit, model, k, max_batch })
    }

    pub fn detect_batch(&mut self, texts: &[&str]) -> Result<Vec<Detection>, BulkError> {
        if texts.len() > self.max_batch {
            return Err(BulkError::BatchTooLarge { len: texts.len(), max: self.max_batch });
        }
        let d = self.model.num_buckets();
        let mut routed: Vec<(usize, ())> = Vec::new(); // model-scored slots
        // Texts for the model-scored slots; None for slots that were routed
        // on the CPU (direct script / no script).
        let mut row_texts: Vec<Option<&str>> = Vec::with_capacity(texts.len());

        let mut results: Vec<Detection> = Vec::with_capacity(texts.len());
        for (slot, text) in texts.iter().enumerate() {
            match crate::route::route(text) {
                crate::route::Route::Direct(lang) => {
                    results.push(Detection { lang: Some(lang), confidence: 1.0, is_uncertain: false });
                    row_texts.push(None);
                }
                crate::route::Route::Unknown => {
                    results.push(Detection { lang: None, confidence: 0.0, is_uncertain: true });
                    row_texts.push(None);
                }
                crate::route::Route::Group(_) => {
                    routed.push((slot, ()));
                    row_texts.push(Some(text));
                    // Placeholder; overwritten after execution.
                    results.push(Detection { lang: None, confidence: 0.0, is_uncertain: true });
                }
            }
        }

        if routed.is_empty() {
            return Ok(results);
        }

        // Zero-copy: write signed bucket indices straight into the plan's
        // host-mapped input buffer. Feature extraction is the bulk-path
        // bottleneck (sequential byte work); it parallelizes cleanly across
        // batch rows. Each task also records the document's token count for
        // the host-side mean-pool at read-out.
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicU32, Ordering};
        let jit = &mut self.jit;
        let model = &self.model;
        let k = self.k;
        let mut view = jit.idx_mut()?.as_array_mut::<i32>().map_err(BulkError::Device)?;
        let flat: &mut [i32] = view
            .as_slice_mut()
            .ok_or_else(|| BulkError::View("input buffer not contiguous".into()))?;
        let counts: Vec<AtomicU32> = (0..flat.len() / k).map(|_| AtomicU32::new(0)).collect();
        flat.chunks_mut(k)
            .zip(&row_texts)
            .zip(&counts)
            .par_bridge()
            .for_each(|((row, text), count)| {
                let Some(text) = text else { return };
                // Write token indices into the row head, then pad the tail.
                let pad = d as i32;
                let mut out = 0usize;
                for token in crate::features::bucket_tokens(text, &model.features, &model.hasher, model.log2_d) {
                    if out == k {
                        break;
                    }
                    row[out] = if token.neg { d as i32 + 1 + token.bucket as i32 } else { token.bucket as i32 };
                    out += 1;
                }
                for dst in &mut row[out..] {
                    *dst = pad;
                }
                count.store(out as u32, Ordering::Relaxed);
            });
        let n = texts.len() as i64;
        self.jit.execute_with_vars(&[("b", n)])?;

        // Output buffer holds fp16 row-sums, sized for the variable's max
        // bound; read only the active `b = n` rows. Mean-pool, bias, softmax,
        // and argmax run host-side with the featurizer's exact counts.
        let mut sums_f16 = vec![0u16; (n as usize) * NUM_LANGS];
        self.jit.output()?.copyout_prefix(bytemuck::cast_slice_mut(&mut sums_f16))?;

        for slot in routed.iter().map(|(slot, _)| *slot) {
            let row = &sums_f16[slot * NUM_LANGS..][..NUM_LANGS];
            results[slot] = logits_to_detection(
                row,
                counts[slot].load(Ordering::Relaxed),
                &self.model.bias,
                self.model.metadata.theta,
            );
        }
        Ok(results)
    }
}

/// Widen an IEEE 754 binary16 value to f32. The JIT plan emits fp16 logits;
/// this is the only dtype conversion on the runtime path (once per output
/// element, after execution).
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = u32::from((bits >> 10) & 0x1F);
    let frac = u32::from(bits & 0x03FF);
    let out = match (exp, frac) {
        (0, 0) => sign,                               // ±0
        (0, _) => {
            // Subnormal: normalize the mantissa, re-bias the exponent.
            let mut shifts = 0u32;
            let mut f = frac;
            while f & 0x0400 == 0 {
                f <<= 1;
                shifts += 1;
            }
            f &= 0x03FF;
            sign | (((127 - 24 + 10 - shifts) as u32) << 23) | (f << 13)
        }
        (0x1F, _) => sign | 0x7F80_0000 | (frac << 13), // inf / nan
        _ => sign | ((exp + 112) << 23) | (frac << 13), // normal
    };
    f32::from_bits(out)
}

/// Host-side finisher shared by the JIT paths: mean-pool (÷ the token count
/// featurization already computed), bias add, softmax + argmax over the
/// class axis, and the θ uncertainty flag.
fn logits_to_detection(sums_f16: &[u16], count: u32, bias: &[f32], theta: f32) -> Detection {
    let inv = if count > 0 { 1.0 / count as f32 } else { 0.0 };
    let logits: Vec<f32> =
        sums_f16.iter().zip(bias).map(|(&s, &b)| f16_to_f32(s) * inv + b).collect();
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let id = exps
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .expect("NUM_LANGS > 0");
    let confidence = exps[id] / sum;
    Detection {
        lang: Lang::ALL.get(id).copied(),
        confidence,
        is_uncertain: confidence < theta,
    }
}

/// Single-document svod detector: the same graph as [`BulkDetector`] with
/// **B = 1 baked in at compile time** (`with_b_fixed(1)`), so every kernel
/// is specialized for fully static shapes — no symbolic batch rebinding on
/// execution. Weights stay resident in the plan; the document's bucket
/// indices are written straight into the host-mapped input buffer and the
/// fp16 logits are read back through the plan's output buffer.
///
/// Counterpart to [`BulkDetector`] for one-shot use, with fully static
/// shapes; see the README for the measured latency trade-offs.
pub struct SingleDetector {
    jit: SpellmanJit,
    model: Model,
    k: usize,
}

impl SingleDetector {
    /// Compile the B=1 plan. `k` is the per-document token budget
    /// (K ≥ 1024 for paragraph text — see [`BulkDetector`]).
    pub fn load(dir: &std::path::Path, k: usize) -> Result<SingleDetector, BulkError> {
        let metadata = crate::model::read_metadata(dir)?;
        let sd = svod_model::state::load_safetensors_dir(dir)?;
        let model = Model::from_state_dict(&sd, metadata)?;
        let inner = SpellmanModel::from_table(&model.table, NUM_LANGS)?;
        let mut jit = SpellmanJit::new(inner).with_b_fixed(1);
        jit.prepare(InputSpec::i32(&[1, k]))?;
        Ok(SingleDetector { jit, model, k })
    }

    /// Detect the language of one document.
    pub fn detect(&mut self, text: &str) -> Result<Detection, BulkError> {
        match crate::route::route(text) {
            crate::route::Route::Direct(lang) => {
                Ok(Detection { lang: Some(lang), confidence: 1.0, is_uncertain: false })
            }
            crate::route::Route::Unknown => {
                Ok(Detection { lang: None, confidence: 0.0, is_uncertain: true })
            }
            crate::route::Route::Group(_) => {
                let d = self.model.num_buckets();
                let k = self.k;
                let model = &self.model;
                let count;
                {
                    let mut view = self.jit.idx_mut()?.as_array_mut::<i32>()?;
                    let flat: &mut [i32] = view
                        .as_slice_mut()
                        .ok_or_else(|| BulkError::View("input buffer not contiguous".into()))?;
                    let row = &mut flat[..k];
                    let pad = d as i32;
                    let mut out = 0usize;
                    for token in crate::features::bucket_tokens(text, &model.features, &model.hasher, model.log2_d) {
                        if out == k {
                            break;
                        }
                        row[out] = if token.neg { d as i32 + 1 + token.bucket as i32 } else { token.bucket as i32 };
                        out += 1;
                    }
                    for dst in &mut row[out..] {
                        *dst = pad;
                    }
                    count = out as u32;
                }
                self.jit.execute()?;
                let mut sums_f16 = vec![0u16; NUM_LANGS];
                self.jit
                    .output()?
                    .copyout_prefix(bytemuck::cast_slice_mut(&mut sums_f16))
                    .map_err(BulkError::Device)?;
                Ok(logits_to_detection(&sums_f16, count, &self.model.bias, self.model.metadata.theta))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_to_f32_reference_values() {
        // IEEE 754 binary16 reference patterns.
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xBC00), -1.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x3555), 0.333251953125); // ~1/3
        assert_eq!(f16_to_f32(0x7BFF), 65504.0);        // max normal
        assert_eq!(f16_to_f32(0x0001), 5.960464477539063e-8); // min subnormal
        assert_eq!(f16_to_f32(0x03FF), 6.097555160522461e-5); // max subnormal
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0xFC00).is_infinite());
        assert!(f16_to_f32(0x7E00).is_nan());
        // Monotonicity over all positive finite values.
        let mut prev = f32::NEG_INFINITY;
        for bits in 0x0000u16..0x7C00 {
            let v = f16_to_f32(bits);
            assert!(v >= prev, "not monotone at {bits:#06x}");
            prev = v;
        }
    }

    #[test]
    fn bulk_smoke_end_to_end() {
        let tmp = tempfile::tempdir().unwrap();
        crate::model::test_support::write_test_model(tmp.path());
        let mut det = BulkDetector::load(tmp.path(), 16, 8).expect("plan compiles and prepares");

        let texts = ["Привет, как дела?", "Hello world", "こんにちは", "12345"];
        let res = det.detect_batch(&texts).expect("batch executes");

        // Script routing outside the plan.
        assert_eq!(res[2].lang, Some(Lang::Jpn));
        assert_eq!(res[3].lang, None);
        assert!(res[3].is_uncertain);
        // Model-scored slots come back with a language and a finite confidence.
        for r in &res[..2] {
            let lang = r.lang.expect("group-routed detection has a language");
            assert!(r.confidence.is_finite() && (0.0..=1.0).contains(&r.confidence));
            assert!(Lang::ALL.contains(&lang));
        }
        // A smaller batch rebinds `b` on the same plan.
        let res2 = det.detect_batch(&["ещё раз"]).unwrap();
        assert!(res2[0].lang.is_some());
    }
}
