//! Bulk batched detection as a compiled svod execution plan.
//!
//! The graph is deliberately minimal — an int8 gather and one reduction,
//! accumulated exactly in i32; the per-column scales, mean-pooling (÷ token
//! count) and the bias add run host-side at read-out:
//!
//! ```text
//! idx [b, K] i32 ──gather──> table rows [b, K, C] i8 ──sum over K──> [b, C] i32
//! ```
//!
//! Featurization already knows each document's exact token count, so the
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
//! `[2*(D+1), C]` with rows `0..=D` equal to `q` and rows `D+1..=2D+1` equal
//! to `-q`, so a token's sign selects the row block and the graph needs no
//! multiplies. The padding row lives at index `D` (all-zero) in both blocks.

// svod's tensor/jit `Result` types cross this module's API (from_model,
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

/// Columns of the gather table: `NUM_LANGS` rounded up to 32, the trailing
/// columns all-zero. A 30-wide int8 row straddles two 64-byte cache lines
/// at 14 of 16 offsets; a 32-wide one never does — ~10-16% faster
/// single-thread kernel (M4 Max). The plan's row-sums keep this width;
/// read-out drops the padding.
pub const TABLE_COLS: usize = NUM_LANGS.next_multiple_of(32);

/// Copy `[rows, NUM_LANGS]` row-major values into `[rows, TABLE_COLS]`,
/// zero-padding each row.
fn pad_columns<T: Copy + Default>(values: &[T]) -> Vec<T> {
    let mut padded = vec![T::default(); values.len() / NUM_LANGS * TABLE_COLS];
    for (dst, src) in padded
        .as_chunks_mut::<TABLE_COLS>()
        .0
        .iter_mut()
        .zip(values.as_chunks::<NUM_LANGS>().0)
    {
        dst[..NUM_LANGS].copy_from_slice(src);
    }
    padded
}

/// Weight tensors for the JIT graph (device-resident, lazily computed).
/// Cloning shares the table.
#[derive(Clone)]
pub struct SpellmanModel {
    /// `[2*(D+1), TABLE_COLS]` int8 — the `q` block then the `-q` block.
    table: Tensor,
    /// Row count of `table`, `2*(D+1)`: the bound the gather indices are
    /// clamped to (see [`Self::forward_batch`]).
    rows: i32,
    /// How the plan's row-sums turn back into f32 logit sums.
    readout: Readout,
}

/// Widens the plan's `[b, TABLE_COLS]` i32 row-sums to f32 logit sums: the
/// i8 sum is exact in i32, and each language column's scale factors out of
/// it (`Σ s_c·q = s_c·Σ q`).
#[derive(Clone, Debug)]
pub struct Readout {
    scales: Vec<f32>,
}

impl Readout {
    /// Copy the first `rows` row-sums out of `jit`'s output buffer into
    /// `out` as f32 logit sums (`rows × NUM_LANGS` values; the padding
    /// columns are dropped). `scratch` holds the raw bytes between calls.
    fn read_sums(
        &self,
        jit: &SpellmanJit,
        rows: usize,
        scratch: &mut Vec<u8>,
        out: &mut [f32],
    ) -> Result<(), BulkError> {
        scratch.resize(rows * TABLE_COLS * size_of::<i32>(), 0);
        jit.output()
            .context(JitSnafu)?
            .copyout_prefix(scratch)
            .context(DeviceSnafu)?;
        let out = out[..rows * NUM_LANGS].as_chunks_mut::<NUM_LANGS>().0;
        let raw = scratch.as_chunks::<4>().0.as_chunks::<TABLE_COLS>().0;
        for (dst, src) in out.iter_mut().zip(raw) {
            for ((o, b), scale) in dst.iter_mut().zip(src).zip(&self.scales) {
                *o = i32::from_ne_bytes(*b) as f32 * scale;
            }
        }
        Ok(())
    }
}

impl SpellmanModel {
    /// The JIT weights for a loaded model: its int8 table, padded to
    /// `TABLE_COLS` and doubled into the ±q blocks, with the per-column
    /// scales kept for read-out.
    pub fn from_model(model: &Model) -> Result<SpellmanModel, svod_tensor::error::Error> {
        let q = pad_columns(&model.table.q);
        let rows = q.len() / TABLE_COLS;
        // The constant buffer must be born 2-D: a reshape op between the
        // buffer and the graph breaks the embedding fusion (~2.4× on the
        // BEAM-scheduled graph, measured), and explicit boundaries are
        // worse still — an eager realize() blocks inlining, a contiguous()
        // marker lands on the execution path (60× single-doc). buffer →
        // neg → cat stays lazy for the plan to fold.
        let p = Tensor::from_raw_bytes(
            bytemuck::cast_slice(&q),
            &[rows, TABLE_COLS],
            svod_dtype::DType::Int8,
        )?;
        let neg = -&p;
        let table = Tensor::cat(&[&p, &neg], 0)?;
        Ok(SpellmanModel {
            table,
            rows: (2 * rows) as i32,
            readout: Readout {
                scales: model.table.scales.clone(),
            },
        })
    }

    /// [`Self::from_model`] with the ±q table realized once, so every plan
    /// of a ladder gathers from one shared buffer instead of folding its
    /// own copy.
    fn realized(model: &Model) -> Result<SpellmanModel, BulkError> {
        let inner = SpellmanModel::from_model(model).context(TensorSnafu)?;
        inner.table.realize().context(TensorSnafu)?;
        Ok(inner)
    }

    /// Build the gather-sum graph over a `[b, K]` bucket-index batch.
    /// Returns raw i32 row-sums `[b, TABLE_COLS]` (see [`Readout`]); the
    /// scales, mean-pooling, the bias add, the softmax, and the argmax all
    /// run host-side at read-out (30 floats per document, with the token
    /// counts featurization already computed). Padding tokens gather the
    /// all-zero row, so the sum is unaffected by padding.
    pub fn forward_batch(
        &self,
        idx: &Tensor,
        b: &BoundVariable,
    ) -> Result<Tensor, svod_tensor::error::Error> {
        let bv = b.as_sint();
        // The prepare-time placeholder is allocated at max batch; shrink to
        // the symbolic batch for kernel specialization at bind time.
        let idx = idx.try_shrink([Some((SInt::Const(0), bv.clone())), None])?;
        // Row-gather the ±q table: [b, K] -> [b, K, C]. `embedding` needs a
        // concrete index shape, which is exactly why K stays a JIT constant.
        //
        // The indices stay i32 on purpose (the largest table offset,
        // 2·(2^18+1)·32 ≈ 16.8M, fits). `embedding` builds a one-hot
        // `where(idx == arange, table, 0)` reduce that the scheduler must
        // collapse into a direct row load; svod's collapse strips a single
        // cast off the range side, and since alpha.7 `arange` is i32, so an
        // i64 index made that side `cast(i64, cast(i32, range))` — the
        // collapse missed and every token scanned all 2·(D+1) table rows.
        // Do not cast `idx` to i64 here.
        //
        // The clamp is a no-op on every index featurization emits (all are
        // < 2·(D+1)), but it is the bound the backend cannot see otherwise:
        // the collapsed gather keeps a `0 <= idx < rows` gate whose
        // else-branch is 0, so without it LLVM must zero the lanes on every
        // token and cannot fuse the widen into the add (`saddw` on NEON).
        // Clamped, the gate folds away — 79 → 50 instructions per token,
        // ~20-27% faster single-thread kernel (M4 Max, 32 columns).
        let idx = idx.maximum(0i32)?.minimum(self.rows - 1)?;
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

/// The host-side part of a loaded model the detectors keep after their
/// plans are built: featurization config and the read-out constants. The
/// int8 table only seeds the svod weights at load time and is dropped
/// there, so [`BulkDetector::replicate`] copies none of it (svod's own
/// `replicate` shares the weights).
#[derive(Clone, Debug)]
struct HostModel {
    features: crate::features::FeatureConfig,
    hasher: crate::hash::FeatureHasher,
    log2_d: u32,
    bias: Vec<f32>,
    theta: f32,
}

impl HostModel {
    fn new(model: &Model) -> HostModel {
        HostModel {
            features: model.features,
            hasher: model.hasher,
            log2_d: model.log2_d,
            bias: model.bias.clone(),
            theta: model.metadata.theta,
        }
    }

    /// Number of buckets `D`; the padding index.
    fn num_buckets(&self) -> u32 {
        1u32 << self.log2_d
    }
}

/// Per-call plan ladder: every call scores all of its rows on the smallest
/// plan whose `K` covers the longest row (the caller's `k` is the top rung;
/// rungs at or above it are dropped). The gather kernel does `B × K` work
/// per execute whatever the rows hold, so a batch of single words on a
/// K=1024 plan is ~98% padding — the ladder cuts that without changing a
/// result (padding gathers the all-zero row). Rows longer than the top
/// rung are chunk-accumulated exactly as before.
const K_LADDER: [usize; 2] = [64, 256];

/// The rungs for a top-rung budget `k`, ascending: the ladder below `k`,
/// then `k` itself (at least 1).
fn ladder(k: usize) -> impl Iterator<Item = usize> {
    let k = k.max(1);
    K_LADDER
        .into_iter()
        .filter(move |&rung| rung < k)
        .chain(std::iter::once(k))
}

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
/// Larger batches are faster per document: every execute pays a fixed
/// launch cost (~85-90 µs on a 14-thread M4 Max) that only a large batch
/// amortizes — held-out throughput went 1.9 → 1.6 → 1.3 → 1.2 µs/sample at
/// 512 / 1024 / 2048 / 4096 rows. 4096 is the compiled limit.
///
/// Device placement follows svod's loading convention: weights live on the
/// default device at load time, so call `svod_tensor::set_default_device`
/// before constructing if the plan should run on a GPU.
pub struct BulkDetector {
    /// Ascending K; the last rung is the caller's `k`.
    plans: Vec<Plan>,
    model: HostModel,
    max_batch: usize,
    /// Shared by every rung: they gather from the same table.
    readout: Readout,
}

impl BulkDetector {
    /// Load from a model directory and compile the plan ladder. `k` is the
    /// per-document token budget of the top rung (try 1024+ for paragraph
    /// text; longer documents are chunk-accumulated, never truncated);
    /// `max_batch` is the compiled batch size — `detect_batch` accepts up
    /// to that many rows per call, and a full batch is the efficient one.
    /// The state dict is loaded once and feeds both the host-side feature
    /// config and the device-resident int8 weight table, which the ladder's
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
        let model = Model::load(dir).context(ModelSnafu)?;
        let max_batch = max_batch.max(1);
        let inner = SpellmanModel::realized(&model)?;
        let readout = inner.readout.clone();
        let mut plans = Vec::new();
        for rung in ladder(k) {
            let mut jit = SpellmanJit::new(inner.clone()).with_b_fixed(max_batch);
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
            model: HostModel::new(&model),
            max_batch,
            readout,
        })
    }

    /// Load the default model from the Hugging Face Hub
    /// ([`crate::hub::DEFAULT_HUB_REPO`]) — svod's `from_hub` wiring: the first
    /// call downloads into the HF cache, later calls replay it.
    pub fn from_hub(k: usize, max_batch: usize) -> Result<BulkDetector, BulkError> {
        let dir =
            crate::hub::download_model(crate::hub::DEFAULT_HUB_REPO, None).context(HubSnafu)?;
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
            readout: self.readout.clone(),
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

        // Called from inside a rayon worker — one replica per worker, the
        // shape `replicate` exists for — the caller already owns the
        // parallelism: svod runs the kernel inline, and the host work below
        // runs sequentially too. Nested `par_iter`s there made a worker
        // blocked on its own subtask steal other replicas' batches and stack
        // them on top of its own (14 replicas kept ~4 cores busy).
        let inline = rayon::current_thread_index().is_some();

        // Route and featurize every document in one walk over its text:
        // script-unique languages and letterless text resolve on the spot
        // and never reach a plan; group-routed rows are featurized in full
        // (no truncation), and their exact token counts pick the plan rung
        // and drive the host-side mean-pool.
        use rayon::prelude::*;
        let model = &self.model;
        let route_one = |text: &&str| {
            let mut out = Vec::with_capacity(text.len() / 2 + 8);
            match crate::features::push_signed_indices_routed(
                text,
                &model.features,
                &model.hasher,
                model.log2_d,
                &mut out,
            ) {
                crate::route::Route::Group(_) => Ok(out),
                route => Err(direct_detection(route)),
            }
        };
        let routed: Vec<Result<Vec<i32>, Detection>> = if inline {
            texts.iter().map(route_one).collect()
        } else {
            texts.par_iter().map(route_one).collect()
        };
        // `rows[r]` is the result slot of `ids[r]`; every other slot is
        // already final.
        let mut results: Vec<Detection> = Vec::with_capacity(texts.len());
        let mut rows: Vec<usize> = Vec::new();
        let mut ids: Vec<Vec<i32>> = Vec::new();
        for (slot, routed) in routed.into_iter().enumerate() {
            match routed {
                Ok(row) => {
                    rows.push(slot);
                    ids.push(row);
                    // Placeholder; overwritten after execution.
                    results.push(Detection {
                        lang: None,
                        confidence: 0.0,
                        is_uncertain: true,
                    });
                }
                Err(detection) => results.push(detection),
            }
        }
        if rows.is_empty() {
            return Ok(results);
        }

        // The rung. A threaded kernel pays a fixed launch cost per execute
        // (~85-90 µs on a 14-thread M4 Max) while padding only gathers the
        // cached all-zero row, so fewer, fuller executes win: the smallest
        // rung that holds the longest row, the top rung otherwise (a
        // padding-minimizing choice ran ~6x more executes, 3.8 vs 2.1
        // µs/sample). An inline kernel pays no launch cost, so there the
        // rung with the least gathered work wins — rows past its K are
        // chunk-accumulated below, exactly.
        let plan_idx = if inline {
            let work = |k: usize| {
                let chunks: usize = ids.iter().map(|row| row.len().div_ceil(k).max(1)).sum();
                chunks.div_ceil(self.max_batch) * self.max_batch * k
            };
            (0..self.plans.len())
                .min_by_key(|&i| work(self.plans[i].k))
                .expect("the ladder has a top rung")
        } else {
            let longest = ids.iter().map(Vec::len).max().unwrap_or(0);
            self.plans
                .iter()
                .position(|p| p.k >= longest)
                .unwrap_or(self.plans.len() - 1)
        };
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
        let mut out = vec![0f32; self.max_batch * NUM_LANGS];
        let mut scratch = Vec::new();
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
                let fill = |(row, &(r, start)): (&mut [i32], &(usize, usize))| {
                    let src = &ids[r][start..(start + k).min(ids[r].len())];
                    row[..src.len()].copy_from_slice(src);
                    row[src.len()..].fill(pad);
                };
                if inline {
                    flat[..group.len() * k]
                        .chunks_mut(k)
                        .zip(group)
                        .for_each(fill);
                } else {
                    flat[..group.len() * k]
                        .par_chunks_mut(k)
                        .zip(group.par_iter())
                        .for_each(fill);
                }
                if plan.dirty_rows > group.len() {
                    flat[group.len() * k..plan.dirty_rows * k].fill(pad);
                }
                plan.dirty_rows = group.len();
            }
            plan.jit.execute().context(JitSnafu)?;
            // Output buffer holds row-sums for the full compiled batch;
            // read only the active rows.
            self.readout
                .read_sums(&plan.jit, group.len(), &mut scratch, &mut out)?;
            for (i, &(r, _)) in group.iter().enumerate() {
                let acc = &mut sums[r];
                for (x, &s) in acc.iter_mut().zip(&out[i * NUM_LANGS..][..NUM_LANGS]) {
                    *x += s;
                }
            }
        }

        // Mean-pool (÷ the exact token count), bias, softmax, argmax, θ.
        for (r, &slot) in rows.iter().enumerate() {
            results[slot] = pooled_to_detection(
                &sums[r],
                ids[r].len() as u32,
                &self.model.bias,
                self.model.theta,
            );
        }
        Ok(results)
    }
}

/// The detection of a text routed past the model: its script-unique
/// language, or none when it has no letters of a supported script.
fn direct_detection(route: crate::route::Route) -> Detection {
    match route {
        crate::route::Route::Direct(lang) => Detection {
            lang: Some(lang),
            confidence: 1.0,
            is_uncertain: false,
        },
        _ => Detection {
            lang: None,
            confidence: 0.0,
            is_uncertain: true,
        },
    }
}

/// Host-side finisher shared by the JIT paths, over f32 logit sums (see
/// [`Readout`]; chunked long documents add per-chunk sums in f32):
/// mean-pool (÷ the token count featurization already computed), bias add,
/// softmax + argmax over the class axis, and the θ uncertainty flag.
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
/// execution. Weights stay resident in the plans; the document's bucket
/// indices are copied into the host-mapped input buffer and the row-sums
/// are read back through the plan's output buffer.
///
/// Like [`BulkDetector`] it keeps a ladder of plans over `K` (64 / 256 /
/// the caller's `k`, sharing one realized table) and scores each document
/// on the smallest rung that holds it: the gather does `K` work whatever
/// the document holds, so a short query on a K=1024 plan was ~98% padding
/// (≤20-character held-out rows: 21.7 → 3.9 µs/doc at K=64, M4 Max).
pub struct SingleDetector {
    /// Ascending K; the last rung is the caller's `k`.
    plans: Vec<SinglePlan>,
    model: HostModel,
    readout: Readout,
    /// The document's signed bucket ids, reused across calls.
    ids: Vec<i32>,
    /// Raw output bytes, reused across calls.
    scratch: Vec<u8>,
}

/// One B=1 plan of the [`SingleDetector`] ladder.
struct SinglePlan {
    jit: SpellmanJit,
    k: usize,
}

impl SingleDetector {
    /// Compile the B=1 plan ladder. `k` is the per-document token budget of
    /// the top rung (K ≥ 1024 for paragraph text — see [`BulkDetector`]);
    /// longer documents are chunk-accumulated, never truncated.
    pub fn load(dir: &std::path::Path, k: usize) -> Result<SingleDetector, BulkError> {
        Self::load_with_prepare_config(dir, k, &svod_tensor::PrepareConfig::from_env())
    }

    /// [`Self::load`] with an explicit prepare configuration (optimizer
    /// strategy, beam width, thread count) instead of the
    /// environment-derived default — e.g. a one-thread plan for
    /// single-core measurements, as [`BulkDetector::load_with_prepare_config`].
    pub fn load_with_prepare_config(
        dir: &std::path::Path,
        k: usize,
        config: &svod_tensor::PrepareConfig,
    ) -> Result<SingleDetector, BulkError> {
        let model = Model::load(dir).context(ModelSnafu)?;
        let inner = SpellmanModel::realized(&model)?;
        let readout = inner.readout.clone();
        let mut plans = Vec::new();
        for rung in ladder(k) {
            let mut jit = SpellmanJit::new(inner.clone()).with_b_fixed(1);
            jit.prepare_with_config(InputSpec::i32(&[1, rung]), config)
                .context(JitSnafu)?;
            plans.push(SinglePlan { jit, k: rung });
        }
        Ok(SingleDetector {
            plans,
            model: HostModel::new(&model),
            readout,
            ids: Vec::new(),
            scratch: Vec::new(),
        })
    }

    /// Load the default model from the Hugging Face Hub; see
    /// [`BulkDetector::from_hub`] for the caching behavior.
    pub fn from_hub(k: usize) -> Result<SingleDetector, BulkError> {
        let dir =
            crate::hub::download_model(crate::hub::DEFAULT_HUB_REPO, None).context(HubSnafu)?;
        Self::load(&dir, k)
    }

    /// Detect the language of one document of ANY size.
    ///
    /// A document that fits a rung takes one execute on the smallest such
    /// plan. Longer documents are scored in full — no truncation, no
    /// position bias — by laying the feature ids into top-rung chunks and
    /// summing the per-chunk plan outputs. The folded model's score is
    /// additive over ids, so the chunked sum IS the exact untruncated
    /// document score (a French opening over a Russian body reads all the
    /// way down).
    pub fn detect(&mut self, text: &str) -> Result<Detection, BulkError> {
        let model = &self.model;
        self.ids.clear();
        match crate::features::push_signed_indices_routed(
            text,
            &model.features,
            &model.hasher,
            model.log2_d,
            &mut self.ids,
        ) {
            route @ (crate::route::Route::Direct(_) | crate::route::Route::Unknown) => {
                Ok(direct_detection(route))
            }
            crate::route::Route::Group(_) => {
                let pad = model.num_buckets() as i32;
                // Smallest rung that holds the document; the top rung
                // otherwise, chunk-accumulated.
                let rung = self
                    .plans
                    .iter()
                    .position(|plan| plan.k >= self.ids.len())
                    .unwrap_or(self.plans.len() - 1);
                let plan = &mut self.plans[rung];
                let mut acc = [0f32; NUM_LANGS];
                for chunk in self.ids.chunks(plan.k) {
                    {
                        let mut view = plan
                            .jit
                            .idx_mut()
                            .context(JitSnafu)?
                            .as_array_mut::<i32>()
                            .context(DeviceSnafu)?;
                        let row: &mut [i32] =
                            view.as_slice_mut().ok_or_else(|| BulkError::View {
                                message: "input buffer not contiguous".into(),
                            })?;
                        row[..chunk.len()].copy_from_slice(chunk);
                        row[chunk.len()..].fill(pad);
                    }
                    plan.jit.execute().context(JitSnafu)?;
                    let mut sums = [0f32; NUM_LANGS];
                    self.readout
                        .read_sums(&plan.jit, 1, &mut self.scratch, &mut sums)?;
                    for (a, &s) in acc.iter_mut().zip(&sums) {
                        *a += s;
                    }
                }
                Ok(pooled_to_detection(
                    &acc,
                    self.ids.len() as u32,
                    &self.model.bias,
                    self.model.theta,
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // A partial batch runs on the same fixed-batch plan.
        let res2 = det.detect_batch(&["ещё раз"]).unwrap();
        assert!(res2[0].lang.is_some());
    }

    #[test]
    fn single_ladder_rungs_score_like_one_plan() {
        // Padding gathers the all-zero row and long documents are
        // chunk-summed, so whichever rung scores a document — K=64, 256,
        // the top K=1024, or top-rung chunks past it — the detection must
        // match a single-rung detector that only ever pads or chunks.
        let tmp = tempfile::tempdir().unwrap();
        crate::model::test_support::write_test_model(tmp.path());
        let mut ladder = SingleDetector::load(tmp.path(), 1024).unwrap();
        let mut single_rung = SingleDetector::load(tmp.path(), 16).unwrap();
        assert_eq!(
            ladder.plans.iter().map(|p| p.k).collect::<Vec<_>>(),
            [64, 256, 1024]
        );
        let sentence = "Привет, как дела? ";
        for repeats in [1, 10, 40, 120] {
            let text = sentence.repeat(repeats);
            let a = ladder.detect(&text).unwrap();
            let b = single_rung.detect(&text).unwrap();
            assert_eq!(a.lang, b.lang, "{repeats} repeats");
            assert!(
                (a.confidence - b.confidence).abs() < 1e-3,
                "{repeats} repeats: {} vs {}",
                a.confidence,
                b.confidence
            );
        }
    }

    #[test]
    fn single_long_document_scores_exactly() {
        // The folded score is additive over feature ids, so chunked
        // scoring at a tiny K must reproduce the untruncated K=8192
        // detection: same language, same confidence (up to f32 sum
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
    }

    #[test]
    fn bulk_inside_a_rayon_worker_scores_like_outside() {
        // Inside a rayon worker detect_batch runs its host work inline and
        // picks the least-work rung (chunk-accumulating rows past its K);
        // outside it fans out and picks the longest-row rung. Both must give
        // the same detections, in order.
        let tmp = tempfile::tempdir().unwrap();
        crate::model::test_support::write_test_model(tmp.path());
        let short = "Привет, как дела?";
        let long = "Привет, как дела? Это длинный документ. ".repeat(40);
        let texts: Vec<&str> = [short, long.as_str(), "Hello world", "こんにちは", "12345"]
            .into_iter()
            .cycle()
            .take(13)
            .collect();
        let mut outside = BulkDetector::load(tmp.path(), 1024, 16).unwrap();
        let expect = outside.detect_batch(&texts).unwrap();
        let mut replica = outside.replicate().unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let got = pool.install(|| {
            assert!(rayon::current_thread_index().is_some());
            replica.detect_batch(&texts).unwrap()
        });
        assert_eq!(got.len(), expect.len());
        for (a, b) in expect.iter().zip(&got) {
            assert_eq!(a.lang, b.lang);
            assert!((a.confidence - b.confidence).abs() < 1e-3);
            assert_eq!(a.is_uncertain, b.is_uncertain);
        }
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
