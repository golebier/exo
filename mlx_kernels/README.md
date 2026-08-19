# EXO MLX Native Metal Kernels

Vendored oMLX native Metal kernels for GLM-5.2 (and, later, MiniMax M3 /
Qwen3.5 / Bonsai). Built opt-in; default EXO builds ship no native extension
and the GLM-5.2 model transparently falls back to the standard attention path.

See `docs/omlx-porting/02-native-metal-kernels.md` for the full design
rationale. This README covers the build/packaging mechanics.

## Why

EXO already ships a faithful port of oMLX's GLM-5.2 model code, but
deliberately ships no native Metal kernels. The fast dispatch
(`src/exo/worker/engines/mlx/vendor/glm_moe_dsa/kernels.py`) already calls
`omlx.custom_kernels.glm_moe_dsa.fast` if importable — so enabling the native
path is primarily a packaging/build task, not an architecture change. oMLX
reports ~30× faster prefill with native kernels (845 vs ~29 tok/s on an M3
Ultra).

## Layout

```
mlx_kernels/
├── build_kernels.py          # Standalone (non-nix) build: CMake → _ext.so + metallib
├── parts.nix                 # Darwin-only nix derivation (skeleton; see below)
├── OMLX_COMMIT.txt           # Pinned oMLX commit vendored from
└── README.md                 # This file

src/exo/worker/engines/mlx/vendor/omlx_custom_kernels/
├── __init__.py               # native_kernel_status() — runtime availability report
├── glm_moe_dsa/
│   ├── __init__.py
│   ├── fast.py               # Vendored verbatim from oMLX; loads _ext if built
│   └── csrc/                 # C++/Metal sources (build inputs)
│       ├── CMakeLists.txt     # oMLX's CMake (nanobind + metallib)
│       ├── bindings.cpp
│       ├── dsa_indexer.{cpp,h,metal}
│       ├── sparse_mla.{cpp,h,metal}
│       ├── exact_block_attention.{cpp,h,metal}
│       └── ... (see csrc/ for the full list)
└── common/csrc/              # Shared headers (steel attention, quantized MoE)
```

## Build paths

### Option A — Standalone (verified working path)

```bash
# One-time: ensure nanobind is available (EXO's `build` extra carries it)
uv sync --extra build

# Build + install _ext.so + omlx_glm_kernels.metallib next to fast.py
just mlx-kernels
# equivalently:
EXO_BUILD_MLX_KERNELS=1 uv run --extra build python mlx_kernels/build_kernels.py --force
```

Requirements (Darwin only):
- Full Xcode (not just Command Line Tools) — `xcrun -sdk macosx metal` must
  be available. oMLX is explicit: CLT alone lacks the `metal` utility.
- `mlx` and `nanobind` importable in the active interpreter.
- CMake ≥ 3.27.

The build is gated by `EXO_BUILD_MLX_KERNELS=1` (or `--force`): default
builds produce a clean error rather than silently requiring Xcode.

### Option B — BUILD.sh (app + DMG bundle)

```bash
EXO_BUILD_MLX_KERNELS=1 ./BUILD.sh
```

This runs Option A between the Rust rebuild and the PyInstaller bundle, so
the built `_ext.so` + metallib land in the source tree and are then bundled
by `packaging/pyinstaller/exo.spec` (which conditionally adds them as
binaries when present).

### Option C — nix (skeleton)

```bash
nix build .#exo-mlx-kernels
```

Darwin-only, opt-in (not wired into the default `exo` package). **Status:
skeleton.** `parts.nix` compiles the extension and metallib but does not yet
overlay them into the runtime exo venv (the remaining integration step —
see `parts.nix` header). Use Option A or B for verified-working builds.

## Runtime

Nothing changes at runtime when unbuilt: `kernels.fast.native_available()`
returns `False`, every `glm_fast.has("glm_dsa_*")` check returns `False`
(EXO's `mx.fast` ships none of the GLM symbols), and the model falls back to
the standard attention path with the sparse top-k mask — identical to EXO's
pre-kernel behavior (verified causally safe in
`docs/gra/GLM-5.2-RESEARCH-RESULTS.md` §5).

When built, the `from . import _ext` import in `fast.py` succeeds,
`is_native_available()` returns `True`, and the GLM-5.2 sparse MLA + exact
block attention fast paths activate. Startup logs confirm either way:

```
INFO ... GLM-5.2 native Metal kernels available; all required fast symbols
       resolved (sparse MLA + exact block attention fast path active)
```

or

```
INFO ... GLM-5.2 native Metal kernels not available (dsa_indexer_scores, ...);
       falling back to standard attention path with sparse top-k mask.
       Build with EXO_BUILD_MLX_KERNELS=1 (Darwin + Metal toolchain) to enable.
```

Programmatic availability:

```python
from exo.worker.engines.mlx.vendor.omlx_custom_kernels import native_kernel_status
print(native_kernel_status())
# {'glm_moe_dsa': {'available': True, 'import_error': None}}
```

## Verification status

| Path | Status |
|------|--------|
| Vendored sources (csrc/ + fast.py + __init__.py) | ✅ byte-identical to oMLX commit `1f1aff3` |
| `kernels.py` dispatch → vendored `fast` | ✅ verified: unbuilt → `native_available()=False`; built → `True` |
| `build_kernels.py` gating + dep checks | ✅ verified: errors cleanly without `EXO_BUILD_MLX_KERNELS=1` / nanobind |
| `just mlx-kernels` / `BUILD.sh` full compile | ✅ verified: `_ext.so` + `omlx_glm_kernels.metallib` built, `native_available()=True`, smoke test passes |
| `exo.spec` conditional bundling | ✅ syntax-verified; bundles artifacts when present |
| nix `.#exo-mlx-kernels` | ⏳ skeleton; needs venv overlay (see `parts.nix` header) |
| Native-vs-fallback numerical equivalence | ⏳ follow-up: needs full GLM-5.2 Indexer + real weights (see tests) |

## Syncing from upstream oMLX

The vendored `csrc/` and `fast.py` are kept byte-identical to oMLX to make
syncs trivial. To update:

```bash
# From a fresh oMLX checkout at the desired commit:
OMLX=/path/to/omlx
DEST=src/exo/worker/engines/mlx/vendor/omlx_custom_kernels

cp -R "$OMLX/omlx/custom_kernels/glm_moe_dsa/csrc/." "$DEST/glm_moe_dsa/csrc/"
cp -R "$OMLX/omlx/custom_kernels/common/csrc/."   "$DEST/common/csrc/"
# fast.py: re-prepend the EXO attribution header (see the file's top comment)
# Update mlx_kernels/OMLX_COMMIT.txt with the new commit SHA.
```

Then run `just mlx-kernels` and the native-equivalence test to confirm the
new kernels build and match the fallback within tolerance.

## Licensing

Apache-2.0 (oMLX's license). The vendored sources carry oMLX's headers; EXO
adds an attribution header to `fast.py` and `__init__.py` only.