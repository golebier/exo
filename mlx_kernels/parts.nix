# SPDX-License-Identifier: Apache-2.0
# Nix build skeleton for EXO's vendored oMLX GLM-5.2 native Metal kernels.
#
# STATUS: skeleton, not yet wired into the default `exo` package.
#
# Darwin-only and opt-in: exposed as the `exo-mlx-kernels` package attribute,
# NOT composed into the default `exo` venv, so `nix build` / `nix flake check`
# stay light (no Xcode / Metal toolchain needed for the default build).
#
# What this derivation does (verified by inspection, not yet by `nix build`
# on CI — see mlx_kernels/README.md §"Verification status"):
#   * compiles the vendored
#     `src/exo/worker/engines/mlx/vendor/omlx_custom_kernels/glm_moe_dsa/csrc/`
#     (the same CMakeLists.txt oMLX ships) into a nanobind `_ext` extension
#     plus `omlx_glm_kernels.metallib`, patching `xcrun -sdk macosx metal` →
#     `metal` (mirrors nix/darwin-build-fixes.patch for the mlx build).
#
# What is NOT done here (the remaining integration step):
#   * overlay the built `_ext.so` + metallib into the runtime exo venv's
#     package dir (mirroring the `exo-rs` overlay in python/parts.nix) so a
#     single `nix build .#exo` produces a binary with the extension
#     importable. The extension must link against the SAME mlx ABI the
#     runtime venv uses; wiring that correctly needs the `pythonSet` mlx
#     as a build input, which is a follow-up.
#
# Until that overlay lands, build the kernels with the verified standalone
# path: `just mlx-kernels` or `EXO_BUILD_MLX_KERNELS=1 ./BUILD.sh`, which
# install `_ext.so` + metallib directly into the source tree (and are then
# bundled by PyInstaller).
{ inputs, ... }:
{
  perSystem =
    { self', pkgs, lib, ... }:
    let
      inherit (pkgs.stdenv.hostPlatform) isDarwin;

      repoRoot = inputs.self;
      kernelSrc = repoRoot + /src/exo/worker/engines/mlx/vendor/omlx_custom_kernels/glm_moe_dsa;

      # nanobind must match the ABI of the mlx the extension loads against.
      # EXO's nix mlx build (python/parts.nix) vendors nanobind v2.10.2; oMLX's
      # upstream pin for the PyPI mlx wheel is 2.13.0. fast.py's `abi_probe`
      # disables native on mismatch instead of crashing, so either is safe —
      # but matching the nix mlx build's nanobind is the correct choice here.
      nanobindSrc = pkgs.fetchFromGitHub {
        owner = "wjakob";
        repo = "nanobind";
        rev = "v2.10.2";
        hash = "sha256-io44YhN+VpfHFWyvvLWSanRgbzA0whK8WlDNRi3hahU=";
        fetchSubmodules = true;
      };

      # Build-time python with nanobind importable (for CMake's
      # `python -m nanobind --cmake_dir` probe). mlx's CMake package is
      # resolved via MLX_DIR / FETCHCONTENT in the runtime venv; for the
      # standalone skeleton we point PYTHONPATH at the fetched nanobind src.
      buildPython = pkgs.python313.withPackages (ps: [ ps.nanobind ]);
    in
    {
      packages = lib.optionalAttrs isDarwin {
        exo-mlx-kernels = pkgs.stdenv.mkDerivation {
          pname = "exo-mlx-kernels";
          version = "0.1.0";
          src = kernelSrc;

          nativeBuildInputs = [
            pkgs.cmake
            pkgs.ninja
            self'.packages.metal-toolchain
            pkgs.apple-sdk_26
            buildPython
          ];

          # Patch `xcrun -sdk macosx metal` → `metal` (nix has no Xcode).
          # Mirrors nix/darwin-build-fixes.patch for the mlx build.
          postPatch = ''
            substituteInPlace csrc/CMakeLists.txt \
              --replace-fail "xcrun -sdk macosx metal" "metal -fmodules-cache-path=$NIX_BUILD_TOP/metal-cache"
          '';

          cmakeFlags = [
            "-DCMAKE_BUILD_TYPE=Release"
            "-DBUILD_SHARED_LIBS=ON"
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$out"
            "-DPython_EXECUTABLE=${buildPython}/bin/python3.13"
            "-DPython3_EXECUTABLE=${buildPython}/bin/python3.13"
            "-DCMAKE_OSX_DEPLOYMENT_TARGET=${pkgs.apple-sdk_26.version}"
            "-DCMAKE_OSX_SYSROOT=${pkgs.apple-sdk_26.passthru.sdkroot}"
          ];

          dontFixup = true; # metal artifacts must not be stripped/patchelf'd

          installPhase = ''
            runHook preInstall
            # CMake wrote _ext.so + omlx_glm_kernels.metallib to
            # CMAKE_LIBRARY_OUTPUT_DIRECTORY ($out) during the build phase.
            test -f "$out"/_ext*.so || { echo "missing _ext.so" >&2; exit 1; }
            test -f "$out"/omlx_glm_kernels.metallib || { echo "missing metallib" >&2; exit 1; }
            runHook postInstall
          '';

          meta = {
            description = "EXO vendored oMLX GLM-5.2 native Metal kernels (skeleton; see mlx_kernels/README.md)";
            platforms = [ "aarch64-darwin" ];
            license = lib.licenses.asl20;
          };
        };
      };
    };
}