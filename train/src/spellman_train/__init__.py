"""Training / evaluation pipeline for spellman.

One installable package and one CLI (``spellman-train``) covering the whole
flow: source fetching and caching, hygiene cleaning, dataset mixing (parquet),
model training and folded export, and publishing to the Hugging Face Hub.
See ``spellman-train --help`` and ``docs/training.md``.
"""

__version__ = "0.2.0"
