"""Prebuild source caches — the download step of the pipeline.

Resolves the same ``--source name:key=value`` specs the mixer would, then
builds every cold cache up front (optionally in parallel) so the subsequent
mix runs entirely from warm caches. Accepts specs directly or replays the
source list recorded in a previous mix's ``manifest.json`` — the recorded
argv holds the exact spec strings, including option quoting.

Usage:
    uv run spellman-train fetch --source fineweb2:docs_per_lang=600 --jobs 4
    uv run spellman-train fetch --manifest data_mix5/manifest.json
    uv run spellman-train fetch --list
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spellman_train.mix import prebuild_colds
from spellman_train.sources import registered


def specs_from_manifest(path: Path) -> list[str]:
    """Extract the verbatim ``--source`` spec strings from a mix manifest.

    The parsed ``sources`` array in the manifest has already lost the exact
    option spelling (values went through ast.literal_eval); the argv keeps
    it, so replay from there."""
    argv = json.loads(path.read_text(encoding="utf-8"))["argv"]
    specs: list[str] = []
    take_value = False
    for tok in argv:
        if take_value:
            specs.append(tok)
            take_value = False
            continue
        if tok == "--source":
            take_value = True
            continue
        if tok.startswith("--source="):
            specs.append(tok.split("=", 1)[1])
    return specs


def populate(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--source", action="append", default=[], metavar="NAME[:K=V,...]",
                    help="source spec to prebuild (repeatable)")
    ap.add_argument("--manifest", type=Path, default=None, metavar="PATH",
                    help="replay the --source list recorded in this mix manifest.json")
    ap.add_argument("--jobs", type=int, default=4,
                    help="parallel cache builds (RAM-bound: 3-4 is the ceiling on 16GB "
                         "because big diverse pools hold GBs each)")
    ap.add_argument("--list", action="store_true", help="list registered sources and exit")


def run(args: argparse.Namespace) -> None:
    if args.list:
        for name in registered():
            print(name)
        return
    specs = list(args.source)
    if args.manifest is not None:
        replayed = specs_from_manifest(args.manifest)
        print(f"replaying {len(replayed)} sources from {args.manifest}")
        specs += replayed
    if not specs:
        raise SystemExit("no --source or --manifest given (see --list)")
    prebuild_colds(specs, args.jobs)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="spellman-train fetch", description=__doc__)
    populate(ap)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
