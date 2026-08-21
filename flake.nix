{
  description = "spellman — Cyrillic-optimized language detection with svod JIT (C backend)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    flake-utils.url = "github:numtide/flake-utils";

    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    crane.url = "github:ipetkov/crane";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      rust-overlay,
      crane,
    }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
        };

        rustStable = pkgs.rust-bin.stable.latest.default;

        craneLib = (crane.mkLib pkgs).overrideToolchain rustStable;

        src = pkgs.lib.cleanSourceWith {
          src = ./.;
          filter =
            path: type:
            (craneLib.filterCargoSources path type)
            # .cargo/config.toml carries the -C target-cpu=native build policy
            || (pkgs.lib.hasSuffix "/.cargo/config.toml" path)
            # tests/hash_vectors.rs reads the Python-generated parity fixture
            || (pkgs.lib.hasSuffix ".json" path);
        };

        # svod's default backend is the C one: plan preparation shells out to
        # `clang` (`clang -c` → object loaded by svod's in-process ELF
        # loader). The wrapped nixpkgs clang carries its own libc headers, so
        # that spawned compile works inside the sandbox. hardeningDisable =
        # "all" keeps JIT compiles identical to a distro clang: the cc
        # wrapper's hardening flags would otherwise add -fstack-protector
        # (undefined __stack_chk_fail in the -c object, which svod's custom
        # ELF loader does not resolve) and -D_FORTIFY_SOURCE to a compile
        # that may not pass -O.
        #
        # bindgen runs in svod-device's build.rs (headers are vendored
        # upstream, only libclang itself is needed); openssl is linked by
        # hf-hub through native-tls.
        nativeBuildInputs = with pkgs; [
          clang
          pkg-config
          openssl.dev
        ];

        commonArgs = {
          inherit src nativeBuildInputs;
          LIBCLANG_PATH = "${pkgs.libclang.lib}/lib";
          hardeningDisable = [ "all" ];
        };

        cargoArtifacts = craneLib.buildDepsOnly commonArgs;
      in
      {
        packages = rec {
          spellman = craneLib.buildPackage (
            commonArgs
            // {
              inherit cargoArtifacts;
              # the root manifest is virtual; give the store path a real name
              pname = "spellman";
              version = "0.1.0";
            }
          );
          default = spellman;
        };

        apps.default = {
          type = "app";
          program = pkgs.lib.getExe self.packages.${system}.spellman;
        };

        checks = {
          rustfmt = craneLib.cargoFmt { inherit src; };

          clippy = craneLib.cargoClippy (
            commonArgs
            // {
              inherit cargoArtifacts;
              cargoClippyExtraArgs = "--all-targets -- --deny warnings";
            }
          );

          test = craneLib.cargoNextest (
            commonArgs
            // {
              inherit cargoArtifacts;
            }
          );
        };

        # commonArgs already puts clang/pkg-config/openssl.dev on the shell's
        # PATH via nativeBuildInputs (and exports LIBCLANG_PATH for bindgen).
        devShells.default = pkgs.mkShell (
          commonArgs
          // {
            packages = [
              rustStable
              pkgs.rust-analyzer
            ];
          }
        );
      }
    );
}
