"""spellman-train — one CLI for the whole training pipeline.

The flow, in pipeline order (``docs/training.md`` has the full walkthrough):

    spellman-train fetch          # build source caches (download step)
    spellman-train clean          # model-judge hygiene over warm caches
    spellman-train mix            # dataset: parquet shards + manifest.json
    spellman-train train          # model: folded export -> model/
    spellman-train publish ...    # dataset / model -> Hugging Face Hub

Plus the supporting tools (short-lane verification, hard negatives, referee
rebuilds, baselines, quantization, the Rust parity fixture, and the Apertium
toolchain). Every command is also runnable standalone as
``python -m spellman_train.<module>``.
"""

from __future__ import annotations

import argparse

from spellman_train import (
    build_short_referee,
    eval_fasttext,
    fetch,
    gen_fixtures,
    hard_negatives,
    hygiene,
    mix,
    prepare_apertium,
    publish,
    quantize_eval,
    short_verify,
    train,
)

#: (name, module, help) — pipeline order.
COMMANDS = [
    ("fetch", fetch, "build source caches (download step; specs or a manifest recipe)"),
    ("clean", hygiene, "model-judge cache hygiene (rewrites caches in place)"),
    ("short-verify", short_verify, "3-judge consensus for the short-text lane"),
    ("hard-negatives", hard_negatives, "mine FineWeb-2 _removed boundary examples"),
    ("referee-short", build_short_referee, "rebuild the frozen short-text referee"),
    ("mix", mix, "build the training dataset (parquet shards + manifest.json)"),
    ("train", train, "train the model and export folded artifacts"),
    ("eval-fasttext", eval_fasttext, "fastText/GlotLID baseline over eval TSVs"),
    ("quantize", quantize_eval, "rewrite a model artifact in a quantized store"),
    ("gen-fixtures", gen_fixtures, "regenerate the Rust<->Python parity fixture"),
    ("prepare-apertium", prepare_apertium, "build lttoolbox analyzers (mkd)"),
    ("publish", publish, "upload dataset/model to the Hugging Face Hub"),
]

_BY_NAME = dict((name, module) for name, module, _ in COMMANDS)


def main(argv: list[str] | None = None) -> None:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    # Modules may need to rewrite their tokens before argparse sees them
    # (e.g. mix --from-manifest replays a recorded recipe into --source
    # specs). The command is always the first token.
    if argv:
        module = _BY_NAME.get(argv[0])
        expander = getattr(module, "expand_argv", None)
        if expander:
            argv = [argv[0]] + expander(argv[1:])
    ap = argparse.ArgumentParser(
        prog="spellman-train",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)
    for name, module, help_text in COMMANDS:
        p = sub.add_parser(
            name,
            help=help_text,
            description=(module.__doc__ or help_text).strip(),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        module.populate(p)
        p.set_defaults(func=module.run)
    args = ap.parse_args(argv)
    # Modules that record their invocation (mix -> manifest.json) must see
    # argv WITHOUT the subcommand token, or a later --from-manifest replay
    # feeds `mix` back to argparse as a positional ("unrecognized arguments").
    args._argv = argv[1:]
    args.func(args)


if __name__ == "__main__":
    main()
