"""Build one source cache in a throwaway subprocess.

pyarrow (under ``datasets``) can deadlock at interpreter shutdown: its C++
thread pools unwind inside static destructors and ``ThreadPool::Shutdown``
ends up waiting on a condition variable that never fires. A worker that
hits this has finished its work but can never exit, wedging whatever pool
owns it (observed killing a full ``mix`` prebuild). So this module never
returns into interpreter teardown: after the build it flushes stdio and
terminates via ``os._exit``.

Usage (one spec per process, argv only — never a shell):
    python -m spellman_train.prewarm '<name:key=value,...>'
"""

from __future__ import annotations

import os
import sys
import traceback


def build(spec: str) -> tuple[str, int]:
    """Build one source's cache to completion.

    Each spec's cache is content-addressed and independent, so cold caches
    build concurrently without write races; the mixer then replays them
    warm through its normal single-process path, keeping dedup order and
    RNG streams byte-identical to a sequential build.
    """
    from spellman_train import sources

    name, opts = sources.parse_source(spec)
    ds = sources.create(name, **opts)
    n = sum(1 for _ in ds.samples_cached())
    return spec, n


def main() -> None:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: python -m spellman_train.prewarm <source-spec>\n")
        os._exit(2)
    spec = sys.argv[1]
    code = 0
    try:
        _, n = build(spec)
        print(f"prebuilt {spec}: {n:,} samples", flush=True)
    except BaseException:  # noqa: BLE001 — every failure path must reach os._exit
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
