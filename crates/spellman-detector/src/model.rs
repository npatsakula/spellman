//! Model artifacts: safetensors weights + JSON metadata.
//!
//! A model directory contains:
//! - `model.json` — the runtime contract: language inventory (column order),
//!   bucket count, hash id/seed, n-gram config, confidence threshold;
//! - `model.safetensors` — `P` (folded score table, `[D+1, NUM_LANGS]` f32,
//!   row `D` all-zero for padding) and `bias` (`[NUM_LANGS]` f32). The
//!   training pipeline additionally stores the unfused `E` / `W` tensors for
//!   reference; the loader ignores them.
//!
//! `P` is the algebraic fold of the trained model: scores are
//! `mean(E[token]) · W`, and because the head is linear this equals
//! `(1/n) Σ P[token]` — so inference never touches an embedding table.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::features::FeatureConfig;
use crate::hash::{FeatureHasher, HashId};
use spellman_language::{Lang, NUM_LANGS};

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
    /// Calibrated confidence threshold: below this, treat detection as
    /// uncertain (GlotLID-style θ).
    #[serde(default)]
    pub theta: f32,
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

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("state: {0}")]
    State(#[from] svod_model::state::Error),
    #[error("bad metadata: {0}")]
    Metadata(String),
    #[error("tensor: {0}")]
    Tensor(#[from] svod_tensor::error::Error),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("tensor shape mismatch for {name}: expected {expected} elements, got {actual}")]
    ShapeMismatch { name: String, expected: usize, actual: usize },
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
    /// GPU. The CPU fast path copies to host vectors regardless; for
    /// device-resident weights use [`crate::jit::SpellmanModel::from_state_dict`]
    /// on the same [`svod_model::state::StateDict`] to avoid the host
    /// round-trip.
    pub fn load(dir: &Path) -> Result<Model, ModelError> {
        let metadata = read_metadata(dir)?;
        let sd = svod_model::state::load_safetensors_dir(dir)?;
        Self::from_state_dict(&sd, metadata)
    }

    /// Build the host-side (CPU fast path) model from a preloaded state dict.
    pub fn from_state_dict(sd: &svod_model::state::StateDict, metadata: ModelMetadata) -> Result<Model, ModelError> {
        Self::validate(&metadata)?;
        let sd = svod_model::state::cast_all(sd, svod_dtype::DType::Float32);
        let p = read_f32(&sd, "P")?;
        let bias = read_f32(&sd, "bias")?;

        let d = 1usize << metadata.log2_d;
        if p.len() != (d + 1) * NUM_LANGS {
            return Err(ModelError::ShapeMismatch {
                name: "P".into(),
                expected: (d + 1) * NUM_LANGS,
                actual: p.len(),
            });
        }
        if bias.len() != NUM_LANGS {
            return Err(ModelError::ShapeMismatch { name: "bias".into(), expected: NUM_LANGS, actual: bias.len() });
        }

        let hasher = FeatureHasher {
            id: HashId::from_id(&metadata.hash)
                .ok_or_else(|| ModelError::Metadata(format!("unknown hash id: {}", metadata.hash)))?,
            seed: metadata.seed,
        };
        let features = FeatureConfig { n_min: metadata.n_min, n_max: metadata.n_max };

        let log2_d = metadata.log2_d;
        Ok(Model { metadata, table: p, bias, log2_d, features, hasher })
    }

    fn validate(metadata: &ModelMetadata) -> Result<(), ModelError> {
        if metadata.format != "spellman-model" {
            return Err(ModelError::Metadata(format!("unexpected format: {}", metadata.format)));
        }
        if metadata.version != 2 {
            return Err(ModelError::Metadata(format!(
                "unsupported version: {} (this runtime speaks version 2, the \
                 canonicalizing feature space)",
                metadata.version
            )));
        }
        if !metadata.canonicalize {
            return Err(ModelError::Metadata(
                "model predates token-class canonicalization; retrain with the \
                 current train/ pipeline"
                    .into(),
            ));
        }
        let expected: Vec<&str> = Lang::ALL.iter().map(|l| l.code()).collect();
        let actual: Vec<&str> = metadata.languages.iter().map(String::as_str).collect();
        if expected != actual {
            return Err(ModelError::Metadata(format!(
                "language inventory mismatch: model was trained for {actual:?}, runtime expects {expected:?}"
            )));
        }
        if !(1..31).contains(&metadata.log2_d) {
            return Err(ModelError::Metadata(format!("log2_d out of range: {}", metadata.log2_d)));
        }
        Ok(())
    }
}

/// Read and validate `model.json` from a model directory.
pub fn read_metadata(dir: &Path) -> Result<ModelMetadata, ModelError> {
    let meta_path = dir.join("model.json");
    let metadata: ModelMetadata = serde_json::from_str(&fs::read_to_string(&meta_path)?)
        .map_err(|e| ModelError::Metadata(format!("{meta_path:?}: {e}")))?;
    if metadata.format != "spellman-model" {
        return Err(ModelError::Metadata(format!("unexpected format: {}", metadata.format)));
    }
    Ok(metadata)
}

fn read_f32(sd: &svod_model::state::StateDict, name: &str) -> Result<Vec<f32>, ModelError> {
    let mut tensor = sd.get(name).cloned().ok_or_else(|| ModelError::MissingTensor(name.to_owned()))?;
    tensor.realize()?;
    Ok(tensor.as_vec::<f32>()?)
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
        let meta = ModelMetadata {
            format: "spellman-model".into(),
            version: 2,
            canonicalize: true,
            languages: Lang::ALL.iter().map(|l| l.code().to_string()).collect(),
            log2_d: 12,
            hash: "fmix32".into(),
            seed: 0x9E37_79B9,
            n_min: 1,
            n_max: 3,
            theta: 0.3,
        };
        std::fs::write(dir.join("model.json"), serde_json::to_string(&meta).unwrap()).unwrap();
        safetensors::serialize_to_file(
            vec![
                (
                    "P",
                    safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![d + 1, NUM_LANGS], bytemuck::cast_slice(&table)).unwrap(),
                ),
                (
                    "bias",
                    safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![NUM_LANGS], bytemuck::cast_slice(&bias)).unwrap(),
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
        let mut meta: ModelMetadata = serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
        meta.languages.reverse();
        std::fs::write(&meta_path, serde_json::to_string(&meta).unwrap()).unwrap();
        assert!(Model::load(tmp.path()).is_err());
    }
}
