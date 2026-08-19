# -*- mode: python ; coding: utf-8 -*-

import sys
import importlib.util
import shutil
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

PROJECT_ROOT = Path.cwd()
SOURCE_ROOT = PROJECT_ROOT / "src"
ENTRYPOINT = SOURCE_ROOT / "exo" / "__main__.py"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard" / "build"
RESOURCES_DIR = PROJECT_ROOT / "resources"
EXO_SHARED_MODELS_DIR = SOURCE_ROOT / "exo" / "shared" / "models"

if not ENTRYPOINT.is_file():
    raise SystemExit(f"Unable to locate Exo entrypoint: {ENTRYPOINT}")

if not DASHBOARD_DIR.is_dir():
    raise SystemExit(f"Dashboard assets are missing: {DASHBOARD_DIR}")

if not RESOURCES_DIR.is_dir():
    raise SystemExit(f"Resource assets are missing: {RESOURCES_DIR}")

if not EXO_SHARED_MODELS_DIR.is_dir():
    raise SystemExit(f"Shared model assets are missing: {EXO_SHARED_MODELS_DIR}")

block_cipher = None


def _module_directory(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise SystemExit(
            f"Module '{module_name}' is not available in the current environment."
        )
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve()
    if spec.origin:
        return Path(spec.origin).resolve().parent
    raise SystemExit(f"Unable to determine installation directory for '{module_name}'.")


MLX_PACKAGE_DIR = _module_directory("mlx")
MLX_LIB_DIR = MLX_PACKAGE_DIR / "lib"
if not MLX_LIB_DIR.is_dir():
    raise SystemExit(f"mlx Metal libraries are missing: {MLX_LIB_DIR}")

# Vendored oMLX GLM-5.2 native Metal kernels. Built opt-in via
# EXO_BUILD_MLX_KERNELS=1 ./BUILD.sh (or `just mlx-kernels`), which install
# _ext*.so + omlx_glm_kernels.metallib next to fast.py. When absent the GLM-5.2
# model falls back to the standard attention path, so bundling is conditional.
GLM_KERNEL_PKG_DIR = (
    SOURCE_ROOT
    / "exo"
    / "worker"
    / "engines"
    / "mlx"
    / "vendor"
    / "omlx_custom_kernels"
    / "glm_moe_dsa"
)
GLM_KERNEL_BINARIES: list[tuple[str, str]] = []
if GLM_KERNEL_PKG_DIR.is_dir():
    for so in GLM_KERNEL_PKG_DIR.glob("_ext*.so"):
        GLM_KERNEL_BINARIES.append(
            (str(so), str(GLM_KERNEL_PKG_DIR.relative_to(SOURCE_ROOT)))
        )
    # _ext.so links against libomlx_glm_kernel_ops.dylib (sibling); bundle it too.
    for dylib in GLM_KERNEL_PKG_DIR.glob("*.dylib"):
        GLM_KERNEL_BINARIES.append(
            (str(dylib), str(GLM_KERNEL_PKG_DIR.relative_to(SOURCE_ROOT)))
        )
    metallib = GLM_KERNEL_PKG_DIR / "omlx_glm_kernels.metallib"
    if metallib.is_file():
        metallib_dest = str(GLM_KERNEL_PKG_DIR.relative_to(SOURCE_ROOT))
        GLM_KERNEL_BINARIES.append((str(metallib), metallib_dest))
        # ALSO place the metallib at the _internal/ root (dest "."). The native
        # kernels locate it via current_binary_dir(), which uses dladdr() on a
        # symbol inside _ext.so. In a PyInstaller --onedir bundle dladdr()
        # resolves that symbol to the bootloader/core image under _internal/
        # rather than to _ext.so's own directory, so the metallib must be
        # discoverable at _internal/omlx_glm_kernels.metallib at runtime.
        # Keep the package-dir copy too so the lookup also succeeds when the
        # extension is imported from its source location (dev runs/tests).
        GLM_KERNEL_BINARIES.append((str(metallib), "."))
    if GLM_KERNEL_BINARIES:
        print(
            f"[exo.spec] bundling {len(GLM_KERNEL_BINARIES)} GLM native kernel artifact(s)"
        )


def _safe_collect(package_name: str) -> list[str]:
    try:
        return collect_submodules(package_name)
    except ImportError:
        return []


HIDDEN_IMPORTS = sorted(
    set(
        collect_submodules("exo")
        + collect_submodules("mlx")
        + _safe_collect("mlx_lm")
        + _safe_collect("mlx_vlm")
        + _safe_collect("transformers")
    ),
)

DATAS: list[tuple[str, str]] = [
    (str(DASHBOARD_DIR), "dashboard"),
    (str(RESOURCES_DIR), "resources"),
    (str(MLX_LIB_DIR), "mlx/lib"),
    (str(EXO_SHARED_MODELS_DIR), "exo/shared/models"),
]

if sys.platform == "darwin":
    MACMON_PATH = shutil.which("macmon")
    if MACMON_PATH is None:
        raise SystemExit(
            "macmon binary not found in PATH. "
            "Install the pinned fork used by exo via: "
            "cargo install --git https://github.com/vladkens/macmon "
            "--rev a1cd06b6cc0d5e61db24fd8832e74cd992097a7d macmon --force"
        )

BINARIES: list[tuple[str, str]] = (
    [
        (MACMON_PATH, "."),
    ]
    if sys.platform == "darwin"
    else []
)

# Add the built GLM native kernel artifacts (if present) so PyInstaller bundles
# them next to fast.py and the runtime `from . import _ext` import resolves.
BINARIES += GLM_KERNEL_BINARIES

a = Analysis(
    [str(ENTRYPOINT)],
    pathex=[str(SOURCE_ROOT)],
    binaries=BINARIES,
    datas=DATAS,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="exo",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="exo",
)
