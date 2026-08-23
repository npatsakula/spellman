"""Build Apertium lttoolbox analyzer binaries for spellman normalization.

Macedonian and Ossetian have no Stanza model, no UDPipe 2 model and no
usable hunspell stems — but Apertium ships full lttoolbox FST
dictionaries for both (apertium-mkd, apertium-oss). This script makes
them usable offline:

1. builds lttoolbox (small C++ lib + CLI) into ``cache/apertium/lttoolbox``
   if missing — one-time brew deps: autoconf automake libtool pkg-config
   utf8cpp icu4c cmake;
2. shallow-clones each language repo under ``cache/apertium/src/`` and
   compiles its monodix to ``cache/apertium/<lang>.automorf.bin``
   (``lt-comp lr``: surface -> lemma+morph-tags transducer).

Everything lands under gitignored ``cache/`` so re-running this script
reproduces the analyzers from source. Registered languages live in
``spellman_train/sources/normalize.py`` as ``apertium:<lang>``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from spellman_train.paths import CACHE_DIR

APERTIUM_DIR = CACHE_DIR / "apertium"

#: lang -> (repo, monodix, acx or None). Only mkd: the other Apertium
# minority-language repos are either HFST .lexc (not lttoolbox) or stubs
# (apertium-oss: 757 entries).
LANGS: dict[str, tuple[str, str, str | None]] = {
    "mkd": ("apertium-mkd", "apertium-mkd.mkd.dix", "apertium-mkd.mkd.acx"),
}


def sh(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def ensure_lttoolbox() -> Path:
    """Build cache/apertium/lttoolbox if the binaries are missing."""
    prefix = APERTIUM_DIR / "lttoolbox"
    if (prefix / "bin" / "lt-comp").exists():
        return prefix

    def brew(pkg: str) -> str:
        return subprocess.run(  # brew --prefix is cheap
            ["brew", "--prefix", pkg], capture_output=True, text=True, check=True
        ).stdout.strip()

    src = APERTIUM_DIR / "src" / "lttoolbox"
    src.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        sh(["git", "clone", "--depth", "1",
             "https://github.com/apertium/lttoolbox", str(src)])
    env = {
        **os.environ,
        "ICU_ROOT": brew("icu4c"),
        "CMAKE_PREFIX_PATH": f"{brew('utf8cpp')};{brew('icu4c')}",
    }
    build = src / "build"
    sh(["cmake", "-B", str(build), "-S", str(src),
         "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_INSTALL_PREFIX={prefix}"],
        env=env)
    sh(["cmake", "--build", str(build), "--parallel", "6"], env=env)
    sh(["cmake", "--install", str(build)], env=env)
    return prefix


def build_lang(lang: str, prefix: Path) -> Path:
    repo, dix, acx = LANGS[lang]
    src = APERTIUM_DIR / "src" / repo
    if not src.exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        sh(["git", "clone", "--depth", "1",
             f"https://github.com/apertium/{repo}", str(src)])
    out = APERTIUM_DIR / f"{lang}.automorf.bin"
    cmd = [str(prefix / "bin" / "lt-comp"), "-j", "lr", str(src / dix), str(out)]
    if acx and (src / acx).exists():
        cmd.append(str(src / acx))
    sh(cmd, env={**os.environ, "DYLD_LIBRARY_PATH": str(prefix / "lib")})
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         cwd=src, capture_output=True, text=True).stdout.strip()
    print(f"  {lang}: {out.name} ({out.stat().st_size // 1024}KB, {repo}@{rev})", flush=True)
    return out


def populate(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--lang", action="append", default=[],
                    help="build only these languages (default: all)")


def run(args: argparse.Namespace) -> None:
    targets = args.lang or list(LANGS)
    bad = set(targets) - set(LANGS)
    if bad:
        raise SystemExit(f"unknown languages {sorted(bad)} (known: {sorted(LANGS)})")
    prefix = ensure_lttoolbox()
    for lang in targets:
        build_lang(lang, prefix)
    print("done")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="spellman-train prepare-apertium", description=__doc__)
    populate(ap)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
