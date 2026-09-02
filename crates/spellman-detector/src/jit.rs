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
//! `K` (tokens per document, zero-padded) and the batch size are both baked
//! into a plan as compile-time constants: the scheduler specializes every
//! kernel for the fixed shape, and a constant batch axis is what lets svod
//! thread the gather across cores (see [`BulkDetector`]). `BulkDetector`
//! keeps a small ladder of plans over `K` and picks one per call by the
//! longest row; [`SingleDetector`] is the `B = 1` special case.
//!
//! Signed hashing is folded into the table layout: the gather table is
//! `[2*(D+1), C]` with rows `0..=D` equal to `P` and rows `D+1..=2D+1` equal
//! to `-P`, so a token's sign selects the row block and the graph needs no
//! multiplies. The padding row lives at index `D` (all-zero) in both blocks.

// svod's tensor/jit `Result` types cross this module's API (from_table,
// forward_batch, the jit_wrapper build closure); they are svod-owned and
// boxed as soon as BulkError takes over.
#![allow(clippy::result_large_err)]

use svod_ir::SInt;
use svod_macros::jit_wrapper;
use svod_model::jit::InputSpec;
use svod_tensor::{BoundVariable, Tensor};

use crate::Detection;
use snafu::prelude::*;
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
    pub fn from_table(
        table: &[f32],
        num_langs: usize,
    ) -> Result<SpellmanModel, svod_tensor::error::Error> {
        let rows = table.len() / num_langs;
        // The constant buffer must be born 2-D: a reshape op between the
        // buffer and the cast breaks the embedding fusion (~2.4× on the
        // BEAM-scheduled graph, measured), and explicit boundaries are
        // worse still — an eager realize() blocks inlining, a contiguous()
        // marker lands on the execution path (60× single-doc). buffer →
        // cast → neg → cat is the state-dict load idiom, fully lazy for
        // the plan to fold.
        let p = Tensor::from_raw_bytes(
            bytemuck::cast_slice(table),
            &[rows, num_langs],
            svod_dtype::DType::Float32,
        )?
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
        rows.sum(1)
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

#[derive(Debug, snafu::Snafu)]
pub enum BulkError {
    #[snafu(display("jit: {source}"))]
    Jit {
        #[snafu(source(from(svod_model::jit::JitError, Box::new)))]
        source: Box<svod_model::jit::JitError>,
    },
    #[snafu(display("tensor: {source}"))]
    Tensor {
        #[snafu(source(from(svod_tensor::error::Error, Box::new)))]
        source: Box<svod_tensor::error::Error>,
    },
    #[snafu(display("device: {source}"))]
    Device {
        #[snafu(source(from(svod_device::error::Error, Box::new)))]
        source: Box<svod_device::error::Error>,
    },
    #[snafu(display("model: {source}"))]
    Model {
        #[snafu(source(from(crate::model::ModelError, Box::new)))]
        source: Box<crate::model::ModelError>,
    },
    #[snafu(display("state: {source}"))]
    State {
        #[snafu(source(from(svod_model::state::Error, Box::new)))]
        source: Box<svod_model::state::Error>,
    },
    #[snafu(display("hub: {source}"))]
    Hub {
        #[snafu(source(from(crate::hub::HubError, Box::new)))]
        source: Box<crate::hub::HubError>,
    },
    #[snafu(display("buffer view: {message}"))]
    View { message: String },
    #[snafu(display("batch of {len} exceeds max_batch {max}"))]
    BatchTooLarge { len: usize, max: usize },
}

/// Per-call plan ladder: every call scores all of its rows on the smallest
/// plan whose `K` covers the longest row (the caller's `k` is the top rung;
/// rungs at or above it are dropped). The gather kernel does `B × K` work
/// per execute whatever the rows hold, so a batch of single words on a
/// K=1024 plan is ~98% padding — the ladder cuts that without changing a
/// result (padding gathers the all-zero row). Rows longer than the top
/// rung are chunk-accumulated exactly as before.
const K_LADDER: [usize; 2] = [64, 256];

/// One prepared plan of the ladder.
struct Plan {
    jit: SpellmanJit,
    k: usize,
    /// Rows `[0, dirty_rows)` of the input buffer hold indices from an
    /// earlier call; everything past them is already the pad index. Each
    /// execute re-pads only that window instead of the whole buffer.
    dirty_rows: usize,
}

/// Batched detector over compiled svod plans.
///
/// Documents are featurized on the CPU (cheap sequential byte work, one
/// rayon task per row), then scored in one fused kernel launch per batch.
/// Script-unique languages (jpn/cmn/hin/ara) are routed on the CPU and never
/// reach the plan.
///
/// The batch dimension is **fixed at `max_batch` at compile time**
/// (`with_b_fixed`), not symbolic: svod threads a kernel over a loop axis
/// only when that axis is a constant a thread count divides, and with a
/// symbolic batch the only splittable axis was the 30-way class axis (a
/// 6-thread ceiling, measured). A constant batch of 512 splits 32-way and
/// the kernel scales with the box; a partial batch pays for the padded rows
/// (all-zero gathers), which is why bulk callers should fill their batches.
///
/// Device placement follows svod's loading convention: weights live on the
/// default device at load time, so call `svod_tensor::set_default_device`
/// before constructing if the plan should run on a GPU.
pub struct BulkDetector {
    /// Ascending K; the last rung is the caller's `k`.
    plans: Vec<Plan>,
    model: Model,
    max_batch: usize,
}

impl BulkDetector {
    /// Load from a model directory and compile the plan ladder. `k` is the
    /// per-document token budget of the top rung (try 1024+ for paragraph
    /// text; longer documents are chunk-accumulated, never truncated);
    /// `max_batch` is the compiled batch size — `detect_batch` accepts up
    /// to that many rows per call, and a full batch is the efficient one.
    /// The state dict is loaded once and feeds both the host-side feature
    /// config and the device-resident fp16 weight table, which the ladder's
    /// plans share.
    ///
    /// The input buffers are host-mapped (not device-local): featurization
    /// results are copied straight into the plan's buffer through a
    /// zero-copy typed view. AMD's host-visible VRAM mapping supports the
    /// same path; a CUDA device-local buffer would need `copyin` staging
    /// instead.
    pub fn load(
        dir: &std::path::Path,
        k: usize,
        max_batch: usize,
    ) -> Result<BulkDetector, BulkError> {
        Self::load_with_prepare_config(dir, k, max_batch, &svod_tensor::PrepareConfig::from_env())
    }

    /// [`Self::load`] with an explicit prepare configuration (optimizer
    /// strategy, beam width) instead of the environment-derived default —
    /// the programmatic route svod's own benches take. Parallel callers
    /// prepare one detector and fork it per worker with
    /// [`Self::replicate`] — the fork shares the sealed weight storage and
    /// pays buffer allocation only.
    pub fn load_with_prepare_config(
        dir: &std::path::Path,
        k: usize,
        max_batch: usize,
        config: &svod_tensor::PrepareConfig,
    ) -> Result<BulkDetector, BulkError> {
        let metadata = crate::model::read_metadata(dir).context(ModelSnafu)?;
        let sd = svod_model::state::load_safetensors_dir(dir).context(StateSnafu)?;
        let model = Model::from_state_dict(&sd, metadata).context(ModelSnafu)?;
        let max_batch = max_batch.max(1);
        let k = k.max(1);
        // One realized ±P table shared by every rung: realizing it here
        // (rather than letting each plan fold its own copy) keeps the
        // ladder at one table's worth of memory; the plans gather from the
        // shared buffer.
        let inner = SpellmanModel::from_table(&model.table, NUM_LANGS).context(TensorSnafu)?;
        let mut table = inner.table;
        table.realize().context(TensorSnafu)?;
        let rungs = K_LADDER
            .iter()
            .copied()
            .filter(|&rung| rung < k)
            .chain(std::iter::once(k));
        let mut plans = Vec::new();
        for rung in rungs {
            let mut jit = SpellmanJit::new(SpellmanModel {
                table: table.clone(),
            })
            .with_b_fixed(max_batch);
            jit.prepare_with_config(InputSpec::i32(&[max_batch, rung]), config)
                .context(JitSnafu)?;
            plans.push(Plan {
                jit,
                k: rung,
                // Fresh buffer contents are unspecified: treat every row as
                // dirty so the first execute pads the whole buffer.
                dirty_rows: max_batch,
            });
        }
        Ok(BulkDetector {
            plans,
            model,
            max_batch,
        })
    }

    /// Load the default model from the Hugging Face Hub
    /// ([`hub::DEFAULT_HUB_REPO`], f16) — svod's `from_hub` wiring: the
    /// first call downloads into the HF cache, later calls replay it.
    pub fn from_hub(k: usize, max_batch: usize) -> Result<BulkDetector, BulkError> {
        let dir =
            crate::hub::download_model(crate::hub::DEFAULT_HUB_REPO, None).context(HubSnafu)?;
        Self::load(&dir, k, max_batch)
    }

    /// Load a storage-format variant (`"int8-col"`, `"fp8-col"`, …) of the
    /// default Hub model.
    pub fn from_hub_variant(
        variant: &str,
        k: usize,
        max_batch: usize,
    ) -> Result<BulkDetector, BulkError> {
        let dir = crate::hub::download_model(crate::hub::DEFAULT_HUB_REPO, Some(variant))
            .context(HubSnafu)?;
        Self::load(&dir, k, max_batch)
    }

    /// Load any Hub repo (optionally under a variant subdirectory).
    pub fn from_hub_repo(
        repo_id: &str,
        variant: Option<&str>,
        k: usize,
        max_batch: usize,
    ) -> Result<BulkDetector, BulkError> {
        let dir = crate::hub::download_model(repo_id, variant).context(HubSnafu)?;
        Self::load(&dir, k, max_batch)
    }

    /// Fork this detector for another worker thread: every plan of the
    /// ladder is replicated (fresh input/output buffers over the shared
    /// sealed weight storage), so the replica pays buffer allocation only —
    /// no planning, no kernel compilation, no second weight upload.
    /// Host-side state (feature config, bias, θ) is cloned.
    ///
    /// Note that svod runs a kernel single-threaded when `detect_batch` is
    /// called from inside a rayon worker (its nested-parallelism policy),
    /// so replicas suit one-replica-per-core designs; a single detector
    /// driven from a non-rayon thread already threads its kernel across
    /// the machine.
    pub fn replicate(&self) -> Result<BulkDetector, BulkError> {
        let mut plans = Vec::with_capacity(self.plans.len());
        for plan in &self.plans {
            plans.push(Plan {
                jit: plan.jit.replicate().context(JitSnafu)?,
                k: plan.k,
                dirty_rows: self.max_batch,
            });
        }
        Ok(BulkDetector {
            plans,
            model: self.model.clone(),
            max_batch: self.max_batch,
        })
    }

    /// Compiled batch size: the most rows one `detect_batch` call accepts.
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    /// Token budget of the top rung: rows with more tokens are
    /// chunk-accumulated over several executes.
    pub fn k(&self) -> usize {
        self.plans.last().map(|p| p.k).unwrap_or(0)
    }

    pub fn detect_batch(&mut self, texts: &[&str]) -> Result<Vec<Detection>, BulkError> {
        if texts.len() > self.max_batch {
            return Err(BulkError::BatchTooLarge {
                len: texts.len(),
                max: self.max_batch,
            });
        }
        let pad = self.model.num_buckets() as i32;

        // CPU routing: script-unique languages and letterless text never
        // reach a plan. `rows` are the (slot, text) pairs the model scores.
        let mut results: Vec<Detection> = Vec::with_capacity(texts.len());
        let mut rows: Vec<(usize, &str)> = Vec::new();
        for (slot, text) in texts.iter().enumerate() {
            match crate::route::route(text) {
                crate::route::Route::Direct(lang) => results.push(Detection {
                    lang: Some(lang),
                    confidence: 1.0,
                    is_uncertain: false,
                }),
                crate::route::Route::Unknown => results.push(Detection {
                    lang: None,
                    confidence: 0.0,
                    is_uncertain: true,
                }),
                crate::route::Route::Group(_) => {
                    rows.push((slot, text));
                    // Placeholder; overwritten after execution.
                    results.push(Detection {
                        lang: None,
                        confidence: 0.0,
                        is_uncertain: true,
                    });
                }
            }
        }
        if rows.is_empty() {
            return Ok(results);
        }

        // Featurize every row in full (no truncation) — one indexed rayon
        // task per row, so the pool splits the batch without the mutex +
        // yield spin of a bridged iterator. The exact token counts pick the
        // plan rung and drive the host-side mean-pool.
        use rayon::prelude::*;
        let model = &self.model;
        let ids: Vec<Vec<i32>> = rows
            .par_iter()
            .map(|(_, text)| {
                let mut out = Vec::with_capacity(text.len() / 2 + 8);
                crate::features::push_signed_indices(
                    text,
                    &model.features,
                    &model.hasher,
                    model.log2_d,
                    &mut out,
                );
                out
            })
            .collect();

        // Smallest rung that holds the longest row; the top rung otherwise
        // (its overflow is chunk-accumulated below).
        let longest = ids.iter().map(Vec::len).max().unwrap_or(0);
        let plan_idx = self
            .plans
            .iter()
            .position(|p| p.k >= longest)
            .unwrap_or(self.plans.len() - 1);
        let plan = &mut self.plans[plan_idx];
        let k = plan.k;

        // Every (row, chunk) pair to score: the first chunk of each row in
        // row order — one round for a batch nothing overflows — then the
        // remaining chunks of the long rows. The folded score is additive
        // over feature ids, so summing the per-chunk outputs gives the
        // exact untruncated document score: no truncation, no first-K
        // position bias (a French opening over a Russian body reads all
        // the way down).
        let mut chunks: Vec<(usize, usize)> = (0..ids.len()).map(|r| (r, 0)).collect();
        for (r, row_ids) in ids.iter().enumerate() {
            let mut start = k;
            while start < row_ids.len() {
                chunks.push((r, start));
                start += k;
            }
        }

        let mut sums: Vec<[f32; NUM_LANGS]> = vec![[0.0; NUM_LANGS]; ids.len()];
        let mut out_f16 = vec![0u16; self.max_batch * NUM_LANGS];
        for group in chunks.chunks(self.max_batch) {
            {
                let mut view = plan
                    .jit
                    .idx_mut()
                    .context(JitSnafu)?
                    .as_array_mut::<i32>()
                    .context(DeviceSnafu)?;
                let flat: &mut [i32] = view.as_slice_mut().ok_or_else(|| BulkError::View {
                    message: "input buffer not contiguous".into(),
                })?;
                // Zero-copy: chunk ids land straight in the host-mapped
                // plan buffer, row tails padded; rows past this group that
                // an earlier call wrote are re-padded.
                flat[..group.len() * k]
                    .par_chunks_mut(k)
                    .zip(group.par_iter())
                    .for_each(|(row, &(r, start))| {
                        let src = &ids[r][start..(start + k).min(ids[r].len())];
                        row[..src.len()].copy_from_slice(src);
                        row[src.len()..].fill(pad);
                    });
                if plan.dirty_rows > group.len() {
                    flat[group.len() * k..plan.dirty_rows * k].fill(pad);
                }
                plan.dirty_rows = group.len();
            }
            plan.jit.execute().context(JitSnafu)?;
            // Output buffer holds fp16 row-sums for the full compiled
            // batch; read only the active rows.
            let active = &mut out_f16[..group.len() * NUM_LANGS];
            plan.jit
                .output()
                .context(JitSnafu)?
                .copyout_prefix(bytemuck::cast_slice_mut(active))
                .context(DeviceSnafu)?;
            for (i, &(r, _)) in group.iter().enumerate() {
                let acc = &mut sums[r];
                for (x, &s) in acc.iter_mut().zip(&active[i * NUM_LANGS..][..NUM_LANGS]) {
                    *x += f16_to_f32(s);
                }
            }
        }

        // Mean-pool (÷ the exact token count), bias, softmax, argmax, θ.
        for (r, &(slot, _)) in rows.iter().enumerate() {
            results[slot] = pooled_to_detection(
                &sums[r],
                ids[r].len() as u32,
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
        (0, 0) => sign, // ±0
        (0, _) => {
            // Subnormal: normalize the mantissa, re-bias the exponent.
            let mut shifts = 0u32;
            let mut f = frac;
            while f & 0x0400 == 0 {
                f <<= 1;
                shifts += 1;
            }
            f &= 0x03FF;
            sign | ((127 - 24 + 10 - shifts) << 23) | (f << 13)
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
    let sums: Vec<f32> = sums_f16.iter().map(|&s| f16_to_f32(s)).collect();
    pooled_to_detection(&sums, count, bias, theta)
}

/// As [`logits_to_detection`] for f32-accumulated sums (the chunked
/// long-document path adds per-chunk fp16 plan outputs in f32).
fn pooled_to_detection(sums: &[f32], count: u32, bias: &[f32], theta: f32) -> Detection {
    let inv = if count > 0 { 1.0 / count as f32 } else { 0.0 };
    let logits: Vec<f32> = sums.iter().zip(bias).map(|(&s, &b)| s * inv + b).collect();
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
        let metadata = crate::model::read_metadata(dir).context(ModelSnafu)?;
        let sd = svod_model::state::load_safetensors_dir(dir).context(StateSnafu)?;
        let model = Model::from_state_dict(&sd, metadata).context(ModelSnafu)?;
        let inner = SpellmanModel::from_table(&model.table, NUM_LANGS).context(TensorSnafu)?;
        let mut jit = SpellmanJit::new(inner).with_b_fixed(1);
        jit.prepare(InputSpec::i32(&[1, k])).context(JitSnafu)?;
        Ok(SingleDetector { jit, model, k })
    }

    /// Load the default model from the Hugging Face Hub (f16 variant);
    /// see [`BulkDetector::from_hub`] for the caching behavior.
    pub fn from_hub(k: usize) -> Result<SingleDetector, BulkError> {
        let dir =
            crate::hub::download_model(crate::hub::DEFAULT_HUB_REPO, None).context(HubSnafu)?;
        Self::load(&dir, k)
    }

    /// Load a storage-format variant of the default Hub model.
    pub fn from_hub_variant(variant: &str, k: usize) -> Result<SingleDetector, BulkError> {
        let dir = crate::hub::download_model(crate::hub::DEFAULT_HUB_REPO, Some(variant))
            .context(HubSnafu)?;
        Self::load(&dir, k)
    }

    /// Detect the language of one document of ANY size.
    ///
    /// Documents up to the compile-time token budget K take the fast
    /// path (one plan execute, unchanged). Longer documents are scored
    /// in full — no truncation, no position bias — by laying the
    /// feature ids into K-sized chunks and summing the per-chunk plan
    /// outputs. The folded model's score is additive over ids, so the
    /// chunked sum IS the exact untruncated document score (a French
    /// opening over a Russian body reads all the way down).
    pub fn detect(&mut self, text: &str) -> Result<Detection, BulkError> {
        match crate::route::route(text) {
            crate::route::Route::Direct(lang) => Ok(Detection {
                lang: Some(lang),
                confidence: 1.0,
                is_uncertain: false,
            }),
            crate::route::Route::Unknown => Ok(Detection {
                lang: None,
                confidence: 0.0,
                is_uncertain: true,
            }),
            crate::route::Route::Group(_) => {
                let d = self.model.num_buckets();
                let k = self.k;
                let model = &self.model;
                let count;
                {
                    let mut view = self
                        .jit
                        .idx_mut()
                        .context(JitSnafu)?
                        .as_array_mut::<i32>()
                        .context(DeviceSnafu)?;
                    let flat: &mut [i32] = view.as_slice_mut().ok_or_else(|| BulkError::View {
                        message: "input buffer not contiguous".into(),
                    })?;
                    let row = &mut flat[..k];
                    let pad = d as i32;
                    let out = crate::features::fill_signed_indices(
                        text,
                        &model.features,
                        &model.hasher,
                        model.log2_d,
                        k,
                        row,
                    );
                    for dst in &mut row[out..] {
                        *dst = pad;
                    }
                    count = out as u32;
                }
                if (count as usize) < k {
                    // fast path: the whole document fit in one execute
                    self.jit.execute().context(JitSnafu)?;
                    let mut sums_f16 = vec![0u16; NUM_LANGS];
                    self.jit
                        .output()
                        .context(JitSnafu)?
                        .copyout_prefix(bytemuck::cast_slice_mut(&mut sums_f16))
                        .context(DeviceSnafu)?;
                    return Ok(logits_to_detection(
                        &sums_f16,
                        count,
                        &self.model.bias,
                        self.model.metadata.theta,
                    ));
                }
                // the row filled to K: the document may continue past the
                // budget — featurize it in full (the k-truncation above
                // dropped nothing a re-scan won't re-emit) and accumulate
                // exact chunk sums
                let mut ids: Vec<i32> = Vec::with_capacity(text.len() / 2 + 8);
                crate::features::for_each_key(text, &model.features, |key| {
                    ids.push(model.hasher.signed_index(key, model.log2_d));
                });
                let total = ids.len() as u32;
                let mut acc = vec![0f32; NUM_LANGS];
                for chunk in ids.chunks(k) {
                    {
                        let mut view = self
                            .jit
                            .idx_mut()
                            .context(JitSnafu)?
                            .as_array_mut::<i32>()
                            .context(DeviceSnafu)?;
                        let flat: &mut [i32] =
                            view.as_slice_mut().ok_or_else(|| BulkError::View {
                                message: "input buffer not contiguous".into(),
                            })?;
                        let row = &mut flat[..k];
                        row[..chunk.len()].copy_from_slice(chunk);
                        for dst in &mut row[chunk.len()..] {
                            *dst = d as i32;
                        }
                    }
                    self.jit.execute().context(JitSnafu)?;
                    let mut sums_f16 = vec![0u16; NUM_LANGS];
                    self.jit
                        .output()
                        .context(JitSnafu)?
                        .copyout_prefix(bytemuck::cast_slice_mut(&mut sums_f16))
                        .context(DeviceSnafu)?;
                    for (a, &s) in acc.iter_mut().zip(&sums_f16) {
                        *a += f16_to_f32(s);
                    }
                }
                Ok(pooled_to_detection(
                    &acc,
                    total,
                    &self.model.bias,
                    self.model.metadata.theta,
                ))
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
        assert_eq!(f16_to_f32(0x3555), 1365.0 / 4096.0); // ~1/3
        assert_eq!(f16_to_f32(0x7BFF), 65504.0); // max normal
        assert_eq!(f16_to_f32(0x0001), 1.0 / 16777216.0); // min subnormal (2^-24)
        assert_eq!(f16_to_f32(0x03FF), 1023.0 / 16777216.0); // max subnormal (1023·2^-24)
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

    #[test]
    fn single_long_document_scores_exactly() {
        // The folded score is additive over feature ids, so chunked
        // scoring at a tiny K must reproduce the untruncated K=8192
        // detection: same language, same confidence (up to fp16 sum
        // ordering), same uncertainty. This is the property that makes
        // detect() size-safe: no truncation, no first-K position bias.
        let tmp = tempfile::tempdir().unwrap();
        crate::model::test_support::write_test_model(tmp.path());
        let long = "Привет, как дела? Это длинный документ для проверки накопления. ".repeat(120);

        let mut chunked = SingleDetector::load(tmp.path(), 16).unwrap();
        let mut whole = SingleDetector::load(tmp.path(), 8192).unwrap();
        let a = chunked.detect(&long).unwrap();
        let b = whole.detect(&long).unwrap();

        assert_eq!(a.lang, b.lang);
        assert!(
            (a.confidence - b.confidence).abs() < 1e-3,
            "chunked {} vs whole {}",
            a.confidence,
            b.confidence
        );
        assert_eq!(a.is_uncertain, b.is_uncertain);

        // a document that fits in K keeps the one-execute fast path
        let mut short_k = SingleDetector::load(tmp.path(), 8192).unwrap();
        let c = short_k.detect("Привет, как дела?").unwrap();
        assert!(c.lang.is_some());
    }

    #[test]
    fn bulk_long_documents_score_exactly() {
        // Batch of long documents with a tiny K and a tiny max_batch:
        // the remainder-chunk row stream spans multiple execute rounds,
        // and every result must still equal the untruncated score.
        let tmp = tempfile::tempdir().unwrap();
        crate::model::test_support::write_test_model(tmp.path());
        let long = "Привет, как дела? Это длинный документ для проверки накопления. ".repeat(120);

        let mut det = BulkDetector::load(tmp.path(), 16, 8).unwrap();
        let texts = [long.as_str(), long.as_str(), "короткий текст"];
        let res = det.detect_batch(&texts).unwrap();

        let mut whole = SingleDetector::load(tmp.path(), 8192).unwrap();
        let reference = whole.detect(&long).unwrap();
        for r in &res[..2] {
            assert_eq!(r.lang, reference.lang);
            assert!(
                (r.confidence - reference.confidence).abs() < 1e-3,
                "chunked {} vs whole {}",
                r.confidence,
                reference.confidence
            );
        }
        assert!(res[2].lang.is_some());
    }
}
