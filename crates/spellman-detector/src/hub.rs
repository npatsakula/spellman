//! Hugging Face Hub model sources, wired the same way as svod's own
//! `from_hub` loaders: fetch through the standard HF cache (`hf-hub`
//! sync API), then load from the snapshot directory. A warmed cache means
//! the second call is a pure local load.
//!
//! The default repo ships the same model in several storage formats
//! (see the design doc's precision section): the f16 table at the repo
//! root and quantized variants in subdirectories —
//! `from_hub_variant("int8-col", …)` fetches the 3.9MB int8 artifact.

use std::path::PathBuf;

/// Default Hub repo: f16 at the root, quantized variants in subdirs.
pub const DEFAULT_HUB_REPO: &str = "vpermilp/spellman";

#[derive(Debug, snafu::Snafu)]
pub enum HubError {
    #[snafu(display("hugging face hub: {source}"))]
    Api {
        #[snafu(source(from(hf_hub::api::sync::ApiError, Box::new)))]
        source: Box<hf_hub::api::sync::ApiError>,
    },
    #[snafu(display("bad hub ref {spec:?} (expected hf:<owner>/<repo>[/variant])"))]
    BadRef { spec: String },
    #[snafu(display("hub snapshot path has no parent directory"))]
    NoParent,
}

/// Parse `hf:<owner>/<repo>[/variant]` into `(repo id, variant)`;
/// `None` when the string is not a hub ref (i.e. a local path).
pub fn parse_hub_ref(spec: &str) -> Option<(String, Option<String>)> {
    let rest = spec.strip_prefix("hf:")?;
    let mut parts = rest.split('/');
    let owner = parts.next()?;
    let name = parts.next()?;
    let variant = parts.next();
    if owner.is_empty() || name.is_empty() || parts.next().is_some() || variant == Some("") {
        return None;
    }
    Some((format!("{owner}/{name}"), variant.map(str::to_string)))
}

/// Download (or replay from cache) `model.json` + `model.safetensors`
/// from `repo_id`, optionally under a `variant` subdirectory, and return
/// the snapshot directory — a drop-in for the local model directories
/// the loaders take.
pub fn download_model(repo_id: &str, variant: Option<&str>) -> Result<PathBuf, HubError> {
    use snafu::prelude::*;

    let api = hf_hub::api::sync::Api::new().context(ApiSnafu)?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        repo_id.to_string(),
        hf_hub::RepoType::Model,
        "main".into(),
    ));
    let prefix = variant.map(|v| format!("{v}/")).unwrap_or_default();
    repo.get(&format!("{prefix}model.json")).context(ApiSnafu)?;
    let weights = repo
        .get(&format!("{prefix}model.safetensors"))
        .context(ApiSnafu)?;
    Ok(weights.parent().ok_or(HubError::NoParent)?.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hub_refs() {
        assert_eq!(parse_hub_ref("model"), None);
        assert_eq!(parse_hub_ref("/opt/models/x"), None);
        assert_eq!(
            parse_hub_ref("hf:vpermilp/spellman"),
            Some(("vpermilp/spellman".into(), None))
        );
        assert_eq!(
            parse_hub_ref("hf:vpermilp/spellman/int8-col"),
            Some(("vpermilp/spellman".into(), Some("int8-col".into())))
        );
        assert_eq!(parse_hub_ref("hf:only-owner"), None);
        assert_eq!(parse_hub_ref("hf:a/b/c/d"), None);
        assert_eq!(parse_hub_ref("hf:a/b/"), None);
    }

    #[test]
    #[ignore = "hits the network / HF cache"]
    fn downloads_default_repo() {
        let dir = download_model(DEFAULT_HUB_REPO, None).unwrap();
        assert!(dir.join("model.json").is_file());
        assert!(dir.join("model.safetensors").is_file());
    }
}
