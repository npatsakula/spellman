//! Model artifacts: safetensors weights + JSON metadata.
//!
//! A model directory contains:
//! - `model.json` — the runtime contract: language inventory (column order),
//!   bucket count, hash id/seed, n-gram config, confidence threshold, and
//!   the storage-quantization spec;
//! - `model.safetensors` — `P` (folded score table, `[D+1, NUM_LANGS]`) and
//!   `bias` (`[NUM_LANGS]`), plus a `scales` tensor when `P` is stored
//!   quantized. Storage precision (`float16` | `int8` | `fp8e4m3`, scaled
//!   per row or per column) is decoupled from compute: the loader always
//!   reconstructs the full table, so everything downstream sees one
//!   canonical representation. These two are everything the runtime
//!   computes with; the unfused `E` / `W` are training-side state and are
//!   not shipped.
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

/// Storage quantization of `P`: how the table (and its `scales` tensor) are
/// encoded in `model.safetensors. `float16` + `none` is the unquantized
/// artifact (no `scales`); `row` scales are per bucket row, `column` scales
/// are per language column.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuantSpec {
    /// `float16` | `int8` | `fp8e4m3`.
    #[serde(default = "default_dtype")]
    pub dtype: String,
    /// `none` | `row` | `column`.
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
    /// Folded table `[D+1][NUM_LANGS]`, row-major.
    pub table: Vec<f32>,
    /// Per-language bias.
    pub bias: Vec<f32>,
    /// Derived: number of buckets `D = 2^log2_d`; the padding row sits at
    /// index `D`.
    pub log2_d: u32,
    pub features: FeatureConfig,
    pub hasher: FeatureHasher,
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
    /// GPU. The JIT table is built from the same resolved host table via
    /// [`crate::jit::SpellmanModel::from_table`].
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
        let (p, bias) = resolve_table(sd, &metadata)?;

        let d = 1usize << metadata.log2_d;
        if p.len() != (d + 1) * NUM_LANGS {
            return Err(ModelError::ShapeMismatch {
                name: "P".into(),
                expected: (d + 1) * NUM_LANGS,
                actual: p.len(),
            });
        }
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
            table: p,
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
        let legal_quant = matches!(
            (
                metadata.quant.dtype.as_str(),
                metadata.quant.scheme.as_str()
            ),
            ("float16", "none")
                | ("int8", "row")
                | ("int8", "column")
                | ("fp8e4m3", "row")
                | ("fp8e4m3", "column")
        );
        if !legal_quant {
            return Err(ModelError::Metadata {
                message: format!(
                    "unsupported quant spec: {}/{}",
                    metadata.quant.dtype, metadata.quant.scheme
                ),
            });
        }
        Ok(())
    }
}

/// Reconstruct the canonical f32 folded table and bias from a state dict,
/// whatever the storage precision. This is the single place that knows
/// about quantization; every downstream consumer (host scorer, JIT table)
/// sees only the resolved representation.
fn resolve_table(
    sd: &svod_model::state::StateDict,
    metadata: &ModelMetadata,
) -> Result<(Vec<f32>, Vec<f32>), ModelError> {
    let d = 1usize << metadata.log2_d;
    let bias = read_cast_f32(sd, "bias")?;

    let mut table = match (
        metadata.quant.dtype.as_str(),
        metadata.quant.scheme.as_str(),
    ) {
        ("float16", "none") => read_cast_f32(sd, "P")?,
        (dtype @ ("int8" | "fp8e4m3"), scheme @ ("row" | "column")) => {
            let per_row = scheme == "row";
            let expected_scales = if per_row { d + 1 } else { NUM_LANGS };
            let scales = read_cast_f32(sd, "scales")?;
            if scales.len() != expected_scales {
                return Err(ModelError::ShapeMismatch {
                    name: "scales".into(),
                    expected: expected_scales,
                    actual: scales.len(),
                });
            }
            let values: Vec<f32> = if dtype == "int8" {
                read_i8(sd, "P")?
                    .into_iter()
                    .map(i32::from)
                    .map(|v| v as f32)
                    .collect()
            } else {
                read_u8(sd, "P")?.iter().map(|&b| e4m3_to_f32(b)).collect()
            };
            values
                .into_iter()
                .enumerate()
                .map(|(i, v)| {
                    v * scales[if per_row {
                        i / NUM_LANGS
                    } else {
                        i % NUM_LANGS
                    }]
                })
                .collect()
        }
        _ => unreachable!("validate() rejects every other combination"),
    };

    // The padding row D is part of the contract (all-zero); enforce it
    // regardless of what rounding did.
    for c in 0..NUM_LANGS {
        table[d * NUM_LANGS + c] = 0.0;
    }
    Ok((table, bias))
}

/// Read a tensor as host f32 values (f16 and f32 storage both legal on the
/// cast lattice; quantized artifacts store `bias`/`scales` in f32 anyway).
/// The cast must happen before `realize()` — casting a realized tensor
/// yields a lazy child with no buffer. Host readout genuinely must
/// materialize (these values never enter a graph); the JIT table path uses
/// `.contiguous()` boundaries instead, see `jit::SpellmanModel::from_table`.
fn read_cast_f32(sd: &svod_model::state::StateDict, name: &str) -> Result<Vec<f32>, ModelError> {
    let tensor = sd
        .get(name)
        .cloned()
        .ok_or_else(|| ModelError::MissingTensor {
            name: name.to_owned(),
        })?;
    let mut cast = tensor
        .cast(svod_dtype::DType::Float32)
        .context(TensorSnafu)?;
    cast.realize().context(TensorSnafu)?;
    cast.as_vec::<f32>().context(TensorSnafu)
}

fn read_i8(sd: &svod_model::state::StateDict, name: &str) -> Result<Vec<i8>, ModelError> {
    let mut tensor = sd
        .get(name)
        .cloned()
        .ok_or_else(|| ModelError::MissingTensor {
            name: name.to_owned(),
        })?;
    tensor.realize().context(TensorSnafu)?;
    tensor.as_vec::<i8>().context(TensorSnafu)
}

fn read_u8(sd: &svod_model::state::StateDict, name: &str) -> Result<Vec<u8>, ModelError> {
    let mut tensor = sd
        .get(name)
        .cloned()
        .ok_or_else(|| ModelError::MissingTensor {
            name: name.to_owned(),
        })?;
    tensor.realize().context(TensorSnafu)?;
    tensor.as_vec::<u8>().context(TensorSnafu)
}

/// Decode an FP8 E4M3FN value (OCP flavor: 4-bit exponent, bias 7, 3-bit
/// mantissa, no infinities, max finite 448, `S.1111.111` = NaN) to f32.
/// The `fp8e4m3` storage format packs these bits as raw u8.
pub fn e4m3_to_f32(bits: u8) -> f32 {
    let sign = u32::from(bits >> 7) << 31;
    let exp = u32::from((bits >> 3) & 0xF);
    let man = u32::from(bits & 0x7);
    match (exp, man) {
        (0, 0) => f32::from_bits(sign), // ±0
        (0, _) => {
            // Subnormal: mantissa × 2^-9.
            let mag = (man as f32) * f32::from_bits(0x3B00_0000); // 2^-9
            if sign != 0 { -mag } else { mag }
        }
        (0xF, 0x7) => f32::NAN,
        _ => f32::from_bits(sign | ((exp + 120) << 23) | (man << 20)),
    }
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

    pub fn write_test_model(dir: &std::path::Path) {
        let d = 1usize << 12;
        let mut table = vec![0.0f32; (d + 1) * NUM_LANGS];
        // Distinguish Russian (column 0) from English (column 21) via bucket 0.
        table[0] = 5.0; // rus
        table[21] = -5.0; // eng
        let bias = vec![0.0f32; NUM_LANGS];
        let meta = fixture_metadata();
        std::fs::write(
            dir.join("model.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();
        safetensors::serialize_to_file(
            vec![
                (
                    "P",
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F32,
                        vec![d + 1, NUM_LANGS],
                        bytemuck::cast_slice(&table),
                    )
                    .unwrap(),
                ),
                (
                    "bias",
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F32,
                        vec![NUM_LANGS],
                        bytemuck::cast_slice(&bias),
                    )
                    .unwrap(),
                ),
            ],
            None,
            &dir.join("model.safetensors"),
        )
        .unwrap();
    }

    /// The metadata half of the fixture (shared by the f16 and int8 writers).
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
            quant: QuantSpec::default(),
        }
    }

    /// int8 variant of the fixture: quantizes the same synthetic table with
    /// the given scheme and stores i8 `P` + f32 `scales` + the quant spec.
    pub fn write_int8_model(dir: &std::path::Path, scheme: &str) {
        let d = 1usize << 12;
        let mut table = vec![0.0f32; (d + 1) * NUM_LANGS];
        table[0] = 5.0; // rus
        table[21] = -5.0; // eng
        let per_row = scheme == "row";
        let scale_index = |i: usize| {
            if per_row {
                i / NUM_LANGS
            } else {
                i % NUM_LANGS
            }
        };
        let n_scales = if per_row { d + 1 } else { NUM_LANGS };

        let mut scales = vec![0.0f32; n_scales];
        for (i, v) in table.iter().enumerate() {
            let s = &mut scales[scale_index(i)];
            *s = s.max(v.abs());
        }
        for s in &mut scales {
            *s = if *s == 0.0 { 1.0 } else { *s / 127.0 };
        }
        let q: Vec<i8> = table
            .iter()
            .enumerate()
            .map(|(i, v)| (v / scales[scale_index(i)]).round().clamp(-127.0, 127.0) as i8)
            .collect();
        let bias = vec![0.0f32; NUM_LANGS];
        let meta = ModelMetadata {
            quant: QuantSpec {
                dtype: "int8".into(),
                scheme: scheme.into(),
            },
            ..fixture_metadata()
        };

        std::fs::write(
            dir.join("model.json"),
            serde_json::to_string(&meta).unwrap(),
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
                    vec![d + 1, NUM_LANGS],
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
                    vec![n_scales],
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

    #[test]
    fn loads_and_validates() {
        let tmp = tempfile::tempdir().unwrap();
        test_support::write_test_model(tmp.path());
        let model = Model::load(tmp.path()).unwrap();
        assert_eq!(model.num_buckets(), 4096);
        assert_eq!(model.pad_index(), 4096);
        assert_eq!(model.hasher.id, HashId::Fmix32);
        assert_eq!(model.features.n_max, 3);
        assert_eq!(model.table.len(), 4097 * NUM_LANGS);
        assert_eq!(model.table[0], 5.0);
        assert_eq!(model.table[21], -5.0);
    }

    #[test]
    fn rejects_wrong_inventory() {
        let tmp = tempfile::tempdir().unwrap();
        test_support::write_test_model(tmp.path());
        let meta_path = tmp.path().join("model.json");
        let mut meta: ModelMetadata =
            serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
        meta.languages.reverse();
        std::fs::write(&meta_path, serde_json::to_string(&meta).unwrap()).unwrap();
        assert!(Model::load(tmp.path()).is_err());
    }

    #[test]
    fn loads_int8_quantized_artifacts() {
        for scheme in ["row", "column"] {
            let tmp = tempfile::tempdir().unwrap();
            test_support::write_int8_model(tmp.path(), scheme);
            let model = Model::load(tmp.path()).expect("int8/{scheme} artifact loads");
            let d = model.num_buckets() as usize;
            // The two nonzero cells are their own column/row maxima, so they
            // quantize to ±127 and come back to ±5 up to f32 rounding.
            assert!((model.table[0] - 5.0).abs() < 1e-4);
            assert!((model.table[21] + 5.0).abs() < 1e-4);
            // Padding row must be exactly zero through any storage format.
            assert!(model.table[d * NUM_LANGS..].iter().all(|v| *v == 0.0));
        }
    }

    #[test]
    fn rejects_version_2_model() {
        let tmp = tempfile::tempdir().unwrap();
        test_support::write_test_model(tmp.path());
        let meta_path = tmp.path().join("model.json");
        let mut meta: ModelMetadata =
            serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
        meta.version = 2;
        meta.lexical = false;
        std::fs::write(&meta_path, serde_json::to_string(&meta).unwrap()).unwrap();
        let err = Model::load(tmp.path()).unwrap_err();
        assert!(err.to_string().contains("version 3"), "got: {err}");
        // A v3 artifact without the lexical flag is equally rejected.
        meta.version = 3;
        std::fs::write(&meta_path, serde_json::to_string(&meta).unwrap()).unwrap();
        assert!(Model::load(tmp.path()).is_err());
    }

    #[test]
    fn rejects_unknown_quant_spec() {
        let tmp = tempfile::tempdir().unwrap();
        test_support::write_int8_model(tmp.path(), "row");
        let meta_path = tmp.path().join("model.json");
        let mut meta: ModelMetadata =
            serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
        meta.quant.scheme = "banana".into();
        std::fs::write(&meta_path, serde_json::to_string(&meta).unwrap()).unwrap();
        assert!(Model::load(tmp.path()).is_err());
    }

    #[test]
    fn e4m3_decode_reference_values() {
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(e4m3_to_f32(0x80), -0.0);
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(e4m3_to_f32(0xB8), -1.0);
        assert_eq!(e4m3_to_f32(0x3C), 1.5);
        assert_eq!(e4m3_to_f32(0x40), 2.0);
        assert_eq!(e4m3_to_f32(0x70), 128.0);
        assert_eq!(e4m3_to_f32(0x7E), 448.0); // max finite (e4m3fn has no inf)
        assert_eq!(e4m3_to_f32(0x01), 0.001953125); // subnormal: 1 × 2^-9
        assert_eq!(e4m3_to_f32(0x03), 0.005859375); // 3 × 2^-9
        assert_eq!(e4m3_to_f32(0x08), 0.015625); // smallest normal: 2^-6
        assert!(e4m3_to_f32(0x7F).is_nan());
        // Monotone across the positive normal range.
        let mut prev = 0.0f32;
        let mut bits = 0x08u8; // smallest normal, 2^-6
        while bits < 0x7E {
            let v = e4m3_to_f32(bits);
            assert!(v >= prev, "not monotone at {bits:#04x}");
            prev = v;
            bits += 1;
        }
    }
}
