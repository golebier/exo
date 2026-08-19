# SPDX-License-Identifier: Apache-2.0
"""Build EXO's vendored oMLX native Metal kernels into a nanobind ``_ext``.

This is the standalone (non-nix) build path for the GLM-5.2 native Metal
kernels vendored under
``src/exo/worker/engines/mlx/vendor/omlx_custom_kernels/glm_moe_dsa/csrc/``.
It invokes CMake (the same ``CMakeLists.txt`` oMLX ships) to compile the
nanobind ``_ext`` extension and the ``omlx_glm_kernels.metallib``, then
installs both next to ``fast.py`` so the runtime ``from . import _ext``
import resolves and the C++ ``current_binary_dir()`` lookup finds the
metallib sibling.

Requirements (Darwin only):
  * Full Xcode (not just Command Line Tools) — ``xcrun -sdk macosx metal``
    must be available. oMLX is explicit: CLT alone lacks the ``metal``
    utility.
  * ``mlx`` and ``nanobind`` importable in the active interpreter. EXO's
    ``build`` extra carries nanobind, so run via::

        uv run --extra build python mlx_kernels/build_kernels.py

    (or ``just mlx-kernels`` which does the same).

The build is opt-in and gated by ``EXO_BUILD_MLX_KERNELS=1``: default EXO
builds ship no native extension and the GLM-5.2 model transparently falls
back to the standard attention path (see
``docs/omlx-porting/02-native-metal-kernels.md``).

Mirrors ``mlx.extension.CMakeBuild`` (the setuptools command oMLX's
``setup.py`` uses) but standalone — no setuptools, no ``build_ext``, just a
direct CMake configure/build against the active interpreter.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ── Layout ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = (
    REPO_ROOT / "src/exo/worker/engines/mlx/vendor/omlx_custom_kernels/glm_moe_dsa"
)
CSRC_DIR = PKG_DIR / "csrc"
BUILD_DIR = REPO_ROOT / "mlx_kernels" / "build"

DEFAULT_DEPLOYMENT_TARGET = "15.0"
TRUTHY = {"1", "true", "yes", "on"}


def _die(msg: str) -> None:
    print(f"mlx_kernels: error: {msg}", file=sys.stderr)
    sys.exit(1)


def _check_platform() -> None:
    if sys.platform != "darwin":
        _die(
            "native Metal kernels are Darwin-only (got "
            f"sys.platform={sys.platform!r}); the GLM-5.2 model falls back "
            "to the standard attention path on this platform."
        )


def _check_xcode_metal() -> None:
    """Verify ``xcrun -sdk macosx metal`` is available (full Xcode)."""
    try:
        subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", "--version"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        _die(
            "`xcrun -sdk macosx metal` is unavailable — install full Xcode "
            "(not just Command Line Tools). oMLX: 'Command Line Tools alone "
            f"do not provide the metal utility.' ({exc})"
        )


def _check_build_deps() -> None:
    """Verify mlx + nanobind are importable in the active interpreter."""
    missing: list[str] = []
    try:
        import mlx  # noqa: F401
    except ImportError:
        missing.append("mlx")
    try:
        import nanobind  # noqa: F401
    except ImportError:
        missing.append("nanobind")
    if missing:
        _die(
            f"missing build dependency: {', '.join(missing)}. Install with: "
            "`uv sync --extra build` (or `uv run --extra build python "
            "mlx_kernels/build_kernels.py`)."
        )


def _deployment_target() -> str:
    return (
        os.environ.get("OMLX_CUSTOM_KERNEL_DEPLOYMENT_TARGET")
        or os.environ.get("MACOSX_DEPLOYMENT_TARGET")
        or DEFAULT_DEPLOYMENT_TARGET
    )


def _cmake_args() -> list[str]:
    target = _deployment_target()
    # CMake otherwise chooses the first framework Python on PATH, which can
    # differ from the interpreter running this script (and lack nanobind/MLX).
    args = [
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={PKG_DIR}{os.sep}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=ON",
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-DCMAKE_OSX_DEPLOYMENT_TARGET={target}",
    ]
    # Honour CMAKE_ARGS (e.g. nix sets it) the same way mlx.extension does.
    extra = os.environ.get("CMAKE_ARGS", "").strip()
    if extra:
        args += [tok for tok in extra.split(" ") if tok]
    # Respect ARCHFLAGS (conda-forge / cross-compile) like mlx.extension.
    archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
    if archs:
        args.append(f"-DCMAKE_OSX_ARCHITECTURES={';'.join(archs)}")
    return args


def _is_opted_in() -> bool:
    return os.environ.get("EXO_BUILD_MLX_KERNELS", "").strip().lower() in TRUTHY or (
        "--force" in sys.argv
    )


def _verify_artifacts() -> None:
    exts = list(PKG_DIR.glob("_ext*.so"))
    metallib = PKG_DIR / "omlx_glm_kernels.metallib"
    if not exts:
        _die("build finished but no _ext*.so was produced in " + str(PKG_DIR))
    if not metallib.exists():
        _die(
            "build finished but omlx_glm_kernels.metallib is missing in " + str(PKG_DIR)
        )
    print(f"mlx_kernels: installed {exts[0].name} + {metallib.name}")


def _runtime_check() -> None:
    """Confirm the freshly built extension imports and reports available."""
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    try:
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from exo.worker.engines.mlx.vendor.glm_moe_dsa.kernels "
                    "import fast; "
                    "print('native_available=', fast.native_available())"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        print(f"mlx_kernels: runtime check {out.stdout.strip()}")
    except subprocess.CalledProcessError as exc:
        _die(
            "runtime check failed — the extension built but does not import: "
            f"{exc.stderr.strip() or exc.stdout.strip()}"
        )


def main() -> None:
    if not _is_opted_in():
        _die(
            "set EXO_BUILD_MLX_KERNELS=1 (or pass --force) to build the "
            "native Metal kernels. Default EXO builds ship no native "
            "extension and fall back to the standard attention path."
        )
    _check_platform()
    _check_xcode_metal()
    _check_build_deps()

    if not CSRC_DIR.is_dir():
        _die(f"CMake source dir not found: {CSRC_DIR}")
    if shutil.which("cmake") is None:
        _die("cmake not found on PATH (install CMake >= 3.27).")

    # Remove stale _ext*.so / metallib BEFORE the build so a failed rebuild
    # can't mask itself behind the previous build's artifacts.
    for stale in list(PKG_DIR.glob("_ext*.so")) + list(
        PKG_DIR.glob("omlx_glm_kernels.metallib")
    ):
        stale.unlink()

    print(
        f"mlx_kernels: configuring {CSRC_DIR.relative_to(REPO_ROOT)} "
        f"(deployment target {_deployment_target()})"
    )
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cmake", "-S", str(CSRC_DIR), "-B", str(BUILD_DIR), *_cmake_args()],
        check=True,
    )
    parallel = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL")
    build_cmd = ["cmake", "--build", str(BUILD_DIR)]
    if parallel:
        build_cmd += ["-j", parallel]
    elif os.cpu_count():
        build_cmd += [f"-j{os.cpu_count()}"]
    print(f"mlx_kernels: building ({' '.join(build_cmd)})")
    subprocess.run(build_cmd, check=True)

    # CMake wrote _ext.so + metallib to CMAKE_LIBRARY_OUTPUT_DIRECTORY
    # (PKG_DIR) via nanobind_add_module + the metallib custom target.
    _verify_artifacts()
    _runtime_check()
    print("mlx_kernels: done — GLM-5.2 native Metal kernels are active.")


if __name__ == "__main__":
    main()
