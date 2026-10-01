//! Model artifacts: safetensors weights + JSON metadata.
//!
//! A model directory contains:
//! - `model.json` — the runtime contract: language inventory (column order),
//!   bucket count, hash id/seed, n-gram config, confidence threshold, and
//!   the storage-quantization spec;
//! - `model.safetensors` — `P` (folded score table, `[D+1, NUM_LANGS]`) as
//!   int8 with one f32 scale per language column (`scales`, `[NUM_LANGS]`),
//!   and `bias` (`[NUM_LANGS]`). The runtime scores the int8 table
//!   directly: column scales factor out of the token sum
//!   (`Σ s_c·q = s_c·Σ q`), so it is applied once per document at
//!   read-out. These are everything the runtime computes with; the unfused
//!   `E` / `W` are training-side state and are not shipped.
//!
//! int8 with per-column scales is the only supported storage: it matched
//! f16 accuracy on every eval (within ±0.02pp) at half the size and ran
//! faster on every path measured, so f16, row-scaled int8 and fp8 artifacts
//! — which could only be scored through a slower dequantized f16 path —
//! were dropped. They are rejected at load with a re-export hint.
//!
//! `P` is the algebraic fold of the trained model: scores are
//! `mean(E[token]) · W`, and because the head is linear this equals
//! `(1/n) Σ P[token]` — so inference never touches an embedding table.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use snafu::prelude::*;

use crate::features::FeatureConfig;
use crate::hash::{FeatureHasher, HashId};
use spellman_language::{Lang, NUM_LANGS};

/// Storage quantization of `P` as recorded in `model.json`. The runtime
/// accepts only `int8` + `column`; the defaults describe the unquantized
/// artifacts older exports wrote without a `quant` block, so those load as
/// a clear rejection rather than a parse error.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuantSpec {
    /// Element type of the stored `P`; must be `int8`.
    #[serde(default = "default_dtype")]
    pub dtype: String,
    /// Scale granularity; must be `column` (one scale per language).
    #[serde(default = "default_scheme")]
    pub scheme: String,
}

fn default_dtype() -> String {
    "float16".into()
}

fn default_scheme() -> String {
    "none".into()
}

impl Default for QuantSpec {
    fn default() -> Self {
        QuantSpec {
            dtype: default_dtype(),
            scheme: default_scheme(),
        }
    }
}

/// Metadata JSON (model.json).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Format tag; must be `spellman-model`.
    pub format: String,
    /// Metadata schema version.
    pub version: u32,
    /// Language codes in model-column order (must equal [`Lang::ALL`] order).
    pub languages: Vec<String>,
    /// log2 of the bucket count.
    pub log2_d: u32,
    /// Feature hash id.
    pub hash: String,
    /// Feature hash seed.
    pub seed: u32,
    /// N-gram window.
    pub n_min: u8,
    pub n_max: u8,
    /// Token-class canonicalization flag (version 2 feature space): URLs,
    /// emails, @mentions and digit-bearing ASCII words pack as class
    /// sentinels instead of their characters. Models trained with and
    /// without this are incompatible.
    #[serde(default)]
    pub canonicalize: bool,
    /// Lexical channel flag (version 3 feature space): word-unigram and
    /// word-bigram keys hashed into the same bucket space as the char
    /// n-grams. Models with and without this are incompatible.
    #[serde(default)]
    pub lexical: bool,
    /// Calibrated confidence threshold: below this, treat detection as
    /// uncertain (GlotLID-style θ).
    #[serde(default)]
    pub theta: f32,
    /// Storage quantization of `P`; absent in older artifacts = f16, none.
    #[serde(default)]
    pub quant: QuantSpec,
}

/// A loaded, inference-ready model.
#[derive(Clone, Debug)]
pub struct Model {
    pub metadata: ModelMetadata,
    /// The folded table as stored: int8 values and per-column scales.
    pub table: ColumnInt8,
    /// Per-language bias.
    pub bias: Vec<f32>,
    /// Derived: number of buckets `D = 2^log2_d`; the padding row sits at
    /// index `D`.
    pub log2_d: u32,
    pub features: FeatureConfig,
    pub hasher: FeatureHasher,
}

/// An int8 folded table with per-language-column scales.
#[derive(Clone, Debug)]
pub struct ColumnInt8 {
    /// `[D+1][NUM_LANGS]`, row-major; the padding row `D` is all-zero.
    pub q: Vec<i8>,
    /// Per-column scale: `P[r][c] = q[r][c] · scales[c]`.
    pub scales: Vec<f32>,
}

#[derive(Debug, snafu::Snafu)]
pub enum ModelError {
    #[snafu(display("io: {source}"))]
    Io { source: std::io::Error },
    #[snafu(display("state: {source}"))]
    State {
        #[snafu(source(from(svod_model::state::Error, Box::new)))]
        source: Box<svod_model::state::Error>,
    },
    #[snafu(display("bad metadata: {message}"))]
    Metadata { message: String },
    #[snafu(display("tensor: {source}"))]
    Tensor {
        #[snafu(source(from(svod_tensor::error::Error, Box::new)))]
        source: Box<svod_tensor::error::Error>,
    },
    #[snafu(display("missing tensor: {name}"))]
    MissingTensor { name: String },
    #[snafu(display(
        "tensor shape mismatch for {name}: expected {expected} elements, got {actual}"
    ))]
    ShapeMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },
}

impl Model {
    /// Number of buckets `D`.
    pub fn num_buckets(&self) -> u32 {
        1u32 << self.log2_d
    }

    /// Padding index: the all-zero row at `D`.
    pub fn pad_index(&self) -> u32 {
        self.num_buckets()
    }

    /// Load from a directory containing `model.json` + safetensors weights
    /// (single `model.safetensors` or sharded
    /// `model-00001-of-0000N.safetensors` + index).
    ///
    /// Weights are placed on the **default device at load time** — call
    /// `svod_tensor::set_default_device` first if the model should live on a
    /// GPU. The JIT table is built from it via
    /// [`crate::jit::SpellmanModel::from_model`].
    pub fn load(dir: &Path) -> Result<Model, ModelError> {
        let metadata = read_metadata(dir)?;
        let sd = svod_model::state::load_safetensors_dir(dir).context(StateSnafu)?;
        Self::from_state_dict(&sd, metadata)
    }

    /// Build the host-side (CPU fast path) model from a preloaded state dict.
    pub fn from_state_dict(
        sd: &svod_model::state::StateDict,
        metadata: ModelMetadata,
    ) -> Result<Model, ModelError> {
        Self::validate(&metadata)?;
        let table = read_table(sd, metadata.log2_d)?;
        let bias = read_cast_f32(sd, "bias")?;
        if bias.len() != NUM_LANGS {
            return Err(ModelError::ShapeMismatch {
                name: "bias".into(),
                expected: NUM_LANGS,
                actual: bias.len(),
            });
        }

        let hasher = FeatureHasher {
            id: HashId::from_id(&metadata.hash).ok_or_else(|| ModelError::Metadata {
                message: format!("unknown hash id: {}", metadata.hash),
            })?,
            seed: metadata.seed,
        };
        let features = FeatureConfig {
            n_min: metadata.n_min,
            n_max: metadata.n_max,
        };

        let log2_d = metadata.log2_d;
        Ok(Model {
            metadata,
            table,
            bias,
            log2_d,
            features,
            hasher,
        })
    }

    fn validate(metadata: &ModelMetadata) -> Result<(), ModelError> {
        if metadata.format != "spellman-model" {
            return Err(ModelError::Metadata {
                message: format!("unexpected format: {}", metadata.format),
            });
        }
        if metadata.version != 3 {
            return Err(ModelError::Metadata {
                message: format!(
                    "unsupported version: {} (this runtime speaks version 3, the \
                     canonicalizing + lexical feature space)",
                    metadata.version
                ),
            });
        }
        if !metadata.canonicalize {
            return Err(ModelError::Metadata {
                message: "model predates token-class canonicalization; retrain with the \
                          current train/ pipeline"
                    .into(),
            });
        }
        if !metadata.lexical {
            return Err(ModelError::Metadata {
                message: "model predates the lexical word-ngram channel; retrain with the \
                          current train/ pipeline"
                    .into(),
            });
        }
        let expected: Vec<&str> = Lang::ALL.iter().map(|l| l.code()).collect();
        let actual: Vec<&str> = metadata.languages.iter().map(String::as_str).collect();
        if expected != actual {
            return Err(ModelError::Metadata {
                message: format!(
                    "language inventory mismatch: model was trained for {actual:?}, runtime expects {expected:?}"
                ),
            });
        }
        if !(1..31).contains(&metadata.log2_d) {
            return Err(ModelError::Metadata {
                message: format!("log2_d out of range: {}", metadata.log2_d),
            });
        }
        if (
            metadata.quant.dtype.as_str(),
            metadata.quant.scheme.as_str(),
        ) != ("int8", "column")
        {
            return Err(ModelError::Metadata {
                message: format!(
                    "unsupported weight storage {}/{}: this runtime scores int8 with \
                     per-column scales (int8-col); convert the artifact with \
                     `spellman-train quantize --model <dir> --out <dir>`",
                    metadata.quant.dtype, metadata.quant.scheme
                ),
            });
        }
        Ok(())
    }
}

/// Read the stored int8 table and its per-column scales, checking both
/// shapes. The padding row `D` is part of the contract (all-zero) and is
/// enforced here whatever the export wrote.
fn read_table(sd: &svod_model::state::StateDict, log2_d: u32) -> Result<ColumnInt8, ModelError> {
    let d = 1usize << log2_d;
    let mut q = read_i8(sd, "P")?;
    if q.len() != (d + 1) * NUM_LANGS {
        return Err(ModelError::ShapeMismatch {
            name: "P".into(),
            expected: (d + 1) * NUM_LANGS,
            actual: q.len(),
        });
    }
    q[d * NUM_LANGS..].fill(0);
    let scales = read_cast_f32(sd, "scales")?;
    if scales.len() != NUM_LANGS {
        return Err(ModelError::ShapeMismatch {
            name: "scales".into(),
            expected: NUM_LANGS,
            actual: scales.len(),
        });
    }
    Ok(ColumnInt8 { q, scales })
}

/// Read a tensor as host f32 values (`bias` and `scales`; f16 and f32
/// storage are both legal on the cast lattice). The cast must happen before
/// `realize()` — casting a realized tensor yields a lazy child with no
/// buffer.
fn read_cast_f32(sd: &svod_model::state::StateDict, name: &str) -> Result<Vec<f32>, ModelError> {
    let tensor = sd
        .get(name)
        .cloned()
        .ok_or_else(|| ModelError::MissingTensor {
            name: name.to_owned(),
        })?;
    let cast = tensor.cast(svod_dtype::DType::Float32);
    cast.realize().context(TensorSnafu)?;
    cast.as_vec::<f32>().context(TensorSnafu)
}

fn read_i8(sd: &svod_model::state::StateDict, name: &str) -> Result<Vec<i8>, ModelError> {
    let tensor = sd
        .get(name)
        .cloned()
        .ok_or_else(|| ModelError::MissingTensor {
            name: name.to_owned(),
        })?;
    tensor.realize().context(TensorSnafu)?;
    tensor.as_vec::<i8>().context(TensorSnafu)
}

/// Read and validate `model.json` from a model directory.
pub fn read_metadata(dir: &Path) -> Result<ModelMetadata, ModelError> {
    let meta_path = dir.join("model.json");
    let metadata: ModelMetadata =
        serde_json::from_str(&fs::read_to_string(&meta_path).context(IoSnafu)?).map_err(|e| {
            ModelError::Metadata {
                message: format!("{meta_path:?}: {e}"),
            }
        })?;
    if metadata.format != "spellman-model" {
        return Err(ModelError::Metadata {
            message: format!("unexpected format: {}", metadata.format),
        });
    }
    Ok(metadata)
}

/// Shared test fixture writer: a tiny synthetic model (D = 2^12) with two
/// nonzero weights so loading/validation paths have something real to chew on.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    /// The fixture's folded table before quantization: Russian (column 0)
    /// and English (column 21) told apart by bucket 0.
    fn fixture_table() -> Vec<f32> {
        let d = 1usize << 12;
        let mut table = vec![0.0f32; (d + 1) * NUM_LANGS];
        table[0] = 5.0; // rus
        table[21] = -5.0; // eng
        table
    }

    /// The metadata half of the fixture.
    pub fn fixture_metadata() -> ModelMetadata {
        ModelMetadata {
            format: "spellman-model".into(),
            version: 3,
            canonicalize: true,
            lexical: true,
            languages: Lang::ALL.iter().map(|l| l.code().to_string()).collect(),
            log2_d: 12,
            hash: "fmix32".into(),
            seed: 0x9E37_79B9,
            n_min: 1,
            n_max: 3,
            theta: 0.3,
            quant: QuantSpec {
                dtype: "int8".into(),
                scheme: "column".into(),
            },
        }
    }

    /// Write the fixture as the runtime's storage format: i8 `P` with f32
    /// per-column `scales` (symmetric, max-abs / 127).
    pub fn write_test_model(dir: &std::path::Path) {
        let table = fixture_table();
        let mut scales = vec![0.0f32; NUM_LANGS];
        for (i, v) in table.iter().enumerate() {
            let s = &mut scales[i % NUM_LANGS];
            *s = s.max(v.abs());
        }
        for s in &mut scales {
            *s = if *s == 0.0 { 1.0 } else { *s / 127.0 };
        }
        let q: Vec<i8> = table
            .iter()
            .enumerate()
            .map(|(i, v)| (v / scales[i % NUM_LANGS]).round().clamp(-127.0, 127.0) as i8)
            .collect();
        let bias = vec![0.0f32; NUM_LANGS];
        std::fs::write(
            dir.join("model.json"),
            serde_json::to_string(&fixture_metadata()).unwrap(),
        )
        .unwrap();
        fn view<'a>(
            name: &'a str,
            dtype: safetensors::Dtype,
            shape: Vec<usize>,
            bytes: &'a [u8],
        ) -> (&'a str, safetensors::tensor::TensorView<'a>) {
            (
                name,
                safetensors::tensor::TensorView::new(dtype, shape, bytes).unwrap(),
            )
        }
        safetensors::serialize_to_file(
            vec![
                view(
                    "P",
                    safetensors::Dtype::I8,
                    vec![table.len() / NUM_LANGS, NUM_LANGS],
                    bytemuck::cast_slice(&q),
                ),
                view(
                    "bias",
                    safetensors::Dtype::F32,
                    vec![NUM_LANGS],
                    bytemuck::cast_slice(&bias),
                ),
                view(
                    "scales",
                    safetensors::Dtype::F32,
                    vec![NUM_LANGS],
                    bytemuck::cast_slice(&scales),
                ),
            ],
            None,
            &dir.join("model.safetensors"),
        )
        .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rewrite_metadata(dir: &std::path::Path, edit: impl FnOnce(&mut ModelMetadata)) {
        let meta_path = dir.join("model.json");
        let mut meta: ModelMetadata =
            serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
        edit(&mut meta);
        std::fs::write(&meta_path, serde_json::to_string(&meta).unwrap()).unwrap();
    }

    #[test]
    fn loads_and_validates() {
        let tmp = tempfile::tempdir().unwrap();
        test_support::write_test_model(tmp.path());
        let model = Model::load(tmp.path()).unwrap();
        assert_eq!(model.num_buckets(), 4096);
        assert_eq!(model.pad_index(), 4096);
        assert_eq!(model.hasher.id, HashId::Fmix32);
        assert_eq!(model.features.n_max, 3);
        assert_eq!(model.table.q.len(), 4097 * NUM_LANGS);
        // The two nonzero cells are their own column maxima: ±127 × 5/127.
        assert_eq!((model.table.q[0], model.table.q[21]), (127, -127));
        assert!((f32::from(model.table.q[0]) * model.table.scales[0] - 5.0).abs() < 1e-5);
        assert!((f32::from(model.table.q[21]) * model.table.scales[21] + 5.0).abs() < 1e-5);
        // The padding row is all-zero.
        assert!(model.table.q[4096 * NUM_LANGS..].iter().all(|&v| v == 0));
    }

    #[test]
    fn rejects_wrong_inventory() {
        let tmp = tempfile::tempdir().unwrap();
        test_support::write_test_model(tmp.path());
        rewrite_metadata(tmp.path(), |meta| meta.languages.reverse());
        assert!(Model::load(tmp.path()).is_err());
    }

    #[test]
    fn rejects_every_storage_but_int8_column() {
        // f16, row-scaled int8 and fp8 artifacts — and older exports with no
        // `quant` block at all — fail with the re-export hint, before any
        // tensor is read.
        for (dtype, scheme) in [
            ("float16", "none"),
            ("int8", "row"),
            ("fp8e4m3", "row"),
            ("fp8e4m3", "column"),
            ("int8", "banana"),
        ] {
            let tmp = tempfile::tempdir().unwrap();
            test_support::write_test_model(tmp.path());
            rewrite_metadata(tmp.path(), |meta| {
                meta.quant = QuantSpec {
                    dtype: dtype.into(),
                    scheme: scheme.into(),
                };
            });
            let err = Model::load(tmp.path()).unwrap_err().to_string();
            assert!(
                err.contains("spellman-train quantize"),
                "{dtype}/{scheme}: {err}"
            );
        }
        let tmp = tempfile::tempdir().unwrap();
        test_support::write_test_model(tmp.path());
        let meta_path = tmp.path().join("model.json");
        let mut json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
        json.as_object_mut().unwrap().remove("quant");
        std::fs::write(&meta_path, json.to_string()).unwrap();
        let err = Model::load(tmp.path()).unwrap_err().to_string();
        assert!(err.contains("float16/none"), "no quant block: {err}");
    }

    #[test]
    fn rejects_version_2_model() {
        let tmp = tempfile::tempdir().unwrap();
        test_support::write_test_model(tmp.path());
        rewrite_metadata(tmp.path(), |meta| {
            meta.version = 2;
            meta.lexical = false;
        });
        let err = Model::load(tmp.path()).unwrap_err();
        assert!(err.to_string().contains("version 3"), "got: {err}");
        // A v3 artifact without the lexical flag is equally rejected.
        rewrite_metadata(tmp.path(), |meta| meta.version = 3);
        assert!(Model::load(tmp.path()).is_err());
    }
}
