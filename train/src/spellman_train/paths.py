"""Filesystem anchors for the spellman training pipeline.

The package code lives at ``train/src/spellman_train`` but its data — caches,
mixes, referee TSVs, the exported model — stays rooted at the ``train/``
project directory, so the multi-GB warm cache survives package restructurings
untouched. ``uv run`` installs this project editable, so ``__file__`` is the
on-disk source file; set ``SPELLMAN_TRAIN_HOME`` to relocate the data root
(e.g. for a non-editable install or a shared cache).
"""

from __future__ import annotations

import os
from pathlib import Path

TRAIN_DIR = Path(
    os.environ.get("SPELLMAN_TRAIN_HOME", Path(__file__).resolve().parents[2])
).resolve()
#: repository root (crates/, docs/, the exported model/)
REPO_ROOT = TRAIN_DIR.parent
CACHE_DIR = TRAIN_DIR / "cache"
MODEL_DIR = REPO_ROOT / "model"
