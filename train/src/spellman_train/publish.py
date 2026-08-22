"""Publish mixes and model artifacts to the Hugging Face Hub.

Two targets, one repo name (the model and the dataset are different repo
types, so ``vpermilp/spellman`` hosts both without collision):

  dataset — uploads a mix directory (``data/*.parquet`` shards +
            ``manifest.json``) plus a README.md dataset card rendered from
            the manifest (splits, counts, languages, recipe summary).
            Idempotent: re-running overwrites the same paths.
  model   — transport only: uploads ``model.json`` / ``model.safetensors``
            and ``README.md`` if present. The card stays hand-maintained.

Usage:
    uv run spellman-train publish dataset --dir data/v11c
    uv run spellman-train publish model --dir ../model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_REPO = "vpermilp/spellman"


def _shards(mix_dir: Path) -> dict[str, list[Path]]:
    data = mix_dir / "data"
    out: dict[str, list[Path]] = {}
    for split in ("train", "val", "test"):
        shards = sorted(data.glob(f"{split}-*-of-*.parquet"))
        if shards:
            out[split] = shards
    return out


def _split_sizes(mix_dir: Path, manifest: dict, shards: dict[str, list[Path]]) -> dict[str, list[int]]:
    """Per-split [num_examples, num_bytes]. Counts come from the manifest
    when the new mixer wrote them; otherwise a metadata-only parquet scan."""
    import polars as pl

    recorded = manifest.get("splits", {})
    out: dict[str, list[int]] = {}
    for split, paths in shards.items():
        if split in recorded:
            n = int(recorded[split])
        else:
            n = int(pl.scan_parquet([str(p) for p in paths]).select(pl.len()).collect().item())
        out[split] = [n, sum(p.stat().st_size for p in paths)]
    return out


def dataset_card(manifest: dict, shards: dict[str, list[Path]], sizes: dict[str, list[int]]) -> str:
    """Render the README.md dataset card (YAML frontmatter + recipe summary)."""
    langs = "\n".join(f"  - {code}" for code in _language_codes(manifest))
    data_files = "\n".join(
        f"      - split: {split}\n        path: data/{split}-*.parquet" for split in shards
    )
    splits = "\n".join(
        f"    - name: {split}\n      num_examples: {sizes[split][0]}\n      num_bytes: {sizes[split][1]}"
        for split in shards
    )
    source_rows = "\n".join(
        f"| `{name}` | {', '.join(f'{k}={v!r}' for k, v in opts.items()) or 'defaults'} |"
        for name, opts in manifest["sources"]
    )
    return f"""---
language:
{langs}
license: other
license_name: mixed-upstream-per-source-in-manifest
task_categories:
  - text-classification
tags:
  - language-identification
  - cyrillic
configs:
  - config_name: default
    data_files:
{data_files}
dataset_info:
  features:
    - name: lang
      dtype: string
    - name: text
      dtype: string
  splits:
{splits}
---

# spellman — Cyrillic language-ID training mix

Training dataset for the [spellman](https://huggingface.co/{DEFAULT_REPO}) detector:
30 classes (25 Cyrillic-column + 5 Latin big-5), one row per sample
(`lang`, `text`), content-addressed train/val/test split.

License: mixed — each row inherits its upstream corpus's license; the
per-source recipe (and therefore attribution) is recorded in
`manifest.json`.

Recipe (also recorded, byte-exact, in `manifest.json`):

- seed `{manifest.get('seed')}`, `cap_per_lang={manifest.get('cap_per_lang')}`
- augmentation: wild `{manifest.get('wild_augment')}`, short `{manifest.get('short_augment')}` (train/val only; test pristine)
- short floor `{manifest.get('short_floor')}`

Sources in mix order (dedup is first-source-wins):

| source | options |
|---|---|
{source_rows}

The exact spec strings, including option quoting, are in `manifest.json` (`argv`).
"""


def _language_codes(manifest: dict) -> list[str]:
    """The card's language tags: the model's class inventory — sources carry
    adapter-level names, but the dataset exists to train exactly these 30
    classes."""
    from spellman_train.features import LANGUAGES

    return list(LANGUAGES)


def run_dataset(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi, create_repo

    mix_dir: Path = args.dir
    manifest_path = mix_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"no manifest.json in {mix_dir} — not a mix directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = _shards(mix_dir)
    if not shards:
        raise SystemExit(
            f"no data/*.parquet shards in {mix_dir} — rebuild the mix with "
            "`spellman-train mix --format parquet` (default)"
        )
    sizes = _split_sizes(mix_dir, manifest, shards)
    total = sum(n for n, _ in sizes.values())
    print(f"dataset: {len(shards)} splits, {total:,} rows, "
          f"{sum(b for _, b in sizes.values()) / 1e6:.1f} MB parquet")

    card = dataset_card(manifest, shards, sizes)
    (mix_dir / "README.md").write_text(card, encoding="utf-8")

    create_repo(args.repo, repo_type="dataset", private=not args.public, exist_ok=True)
    api = HfApi()
    commit = api.upload_folder(
        repo_id=args.repo,
        repo_type="dataset",
        revision=args.revision,
        folder_path=mix_dir,
        allow_patterns=["data/*.parquet", "manifest.json", "README.md"],
        commit_message=args.message or f"upload mix: {total:,} rows",
    )
    print(f"uploaded -> https://huggingface.co/datasets/{args.repo} (commit {commit.oid[:10]})")


def run_model(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    model_dir: Path = args.dir
    missing = [n for n in ("model.json", "model.safetensors") if not (model_dir / n).exists()]
    if missing:
        raise SystemExit(f"{model_dir} is missing {missing} — not a model artifact")
    extra = [n for n in ("README.md",) if (model_dir / n).exists()]
    print(f"model: uploading model.json, model.safetensors{', README.md' if extra else ''}")

    api = HfApi()
    commit = api.upload_folder(
        repo_id=args.repo,
        repo_type="model",
        revision=args.revision,
        folder_path=model_dir,
        allow_patterns=["model.json", "model.safetensors", "README.md"],
        commit_message=args.message or "update model artifacts",
    )
    print(f"uploaded -> https://huggingface.co/{args.repo} (commit {commit.oid[:10]})")


def populate(ap: argparse.ArgumentParser) -> None:
    sub = ap.add_subparsers(dest="target", required=True)
    ds = sub.add_parser("dataset", help="upload a mix directory (parquet + manifest + rendered card)")
    ds.add_argument("--dir", type=Path, required=True, help="mix directory (from spellman-train mix)")
    ds.add_argument("--repo", default=DEFAULT_REPO)
    ds.add_argument("--revision", default="main")
    ds.add_argument("--public", action="store_true",
                    help="create the repo public (default: private)")
    ds.add_argument("--message", default=None, help="commit message override")
    md = sub.add_parser("model", help="upload model.json + model.safetensors (+ README.md if present)")
    md.add_argument("--dir", type=Path, required=True, help="model artifact directory")
    md.add_argument("--repo", default=DEFAULT_REPO)
    md.add_argument("--revision", default="main")
    md.add_argument("--message", default=None, help="commit message override")


def run(args: argparse.Namespace) -> None:
    if args.target == "dataset":
        run_dataset(args)
    else:
        run_model(args)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="spellman-train publish", description=__doc__)
    populate(ap)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
