# 02 — Native Metal Custom Kernels (GLM-5.2, MiniMax M3, Qwen3.5)

**Tier:** ⭐⭐⭐ (Tier 1)
**Effort:** Medium (build/packaging-heavy, low code change)
**Impact:** Very high (~30× prefill on GLM-5.2 per oMLX)
**oMLX source:** `omlx/custom_kernels/{glm_moe_dsa, minimax_m3, qwen35_prefill, bonsai}/`
**EXO target:** `src/exo/worker/engines/mlx/vendor/glm_moe_dsa/kernels.py` (already wired), plus build/packaging

---

## Why this is the highest-leverage finish line

EXO has already invested heavily in GLM-5.2:
- Vendored model code: `src/exo/worker/engines/mlx/vendor/glm_moe_dsa/{glm_moe_dsa_model,deepseek_v32,sparse_mla,kernels,switch_layers}.py`
- Patch registration: `src/exo/worker/engines/mlx/patches/glm_moe_dsa/__init__.py`
- Bootstrap wiring: `src/exo/worker/runner/bootstrap.py:75-77` calls `apply_mlx_patches()` before `mlx_lm.load()`
- Tool parser: `src/exo/worker/engines/mlx/tool_parsers/glm52.py`
- Verified correct as a port (see `../gra/GLM-5.2-RESEARCH-RESULTS.md`)

**But EXO deliberately ships no native kernels.** From
`vendor/glm_moe_dsa/kernels.py`:

```python
# EXO does not ship oMLX's native GLM kernels. The fast dispatch falls back
# to mx.fast for the symbols it provides, and returns None for the GLM-specific
# native kernels (sparse MLA, exact block attention, etc.) so the model code
# falls through to the standard attention path.
try:
    from omlx.custom_kernels.glm_moe_dsa import fast as _native_fast
except Exception:
    _native_fast = None
```

The `_FastDispatch` already calls `omlx.custom_kernels.glm_moe_dsa.fast` if it's
importable — so **this is primarily a packaging/build task, not an architecture
change**. The fallback path works (verified causally safe in
`../gra/GLM-5.2-RESEARCH-RESULTS.md` §5) but is ~30× slower.

oMLX reports (README): fused DSA prefill is **~30× faster** with native kernels
(845 vs ~29 tok/s on an M3 Ultra) and the fallback also uses more memory (#2137).

---

## oMLX kernel families

### `glm_moe_dsa/` — GLM-5.2 fused kernels
```
csrc/
├── bindings.cpp                      # PyO3 bindings
├── CMakeLists.txt
├── dsa_indexer.{cpp,h,metal}         # sparse DSA indexer
├── sparse_mla.{cpp,h,metal}          # sparse MLA attention
├── exact_block_attention.{cpp,h,metal}
├── deepseek_v4_sparse_attention.{cpp,h,metal}
├── deepseek_moe.metal                # fused MoE
├── fused_moe.{cpp,h,metal}
├── dspark_gemm.{cpp,h,metal}
├── dspark_qmv.{cpp,h,metal}
└── kernels/
    ├── steel_deepseek_v4_sparse_attention.h
    ├── steel_dsa_indexer_score.h
    └── steel_sparse_mla.h
```
Python: `fast.py` (the `fast` module EXO's dispatch imports), `__init__.py`
with `native_kernel_status()`.

### `minimax_m3/` — MiniMax M3 MSA kernels
```
csrc/
├── bindings.cpp
├── CMakeLists.txt
├── minimax_msa.{cpp,h,metal}
```

### `qwen35_prefill/` — Qwen3.5 fused prefill + ANE path
```
csrc/
├── bindings.cpp
├── CMakeLists.txt
├── qwen35_ane.{h,metal,mm}           # Apple Neural Engine path (.mm = Obj-C++)
├── qwen35_attention.metal
├── qwen35_prefill.{cpp,h}
├── qwen35_qmm.metal                  # quantized matmul
└── qwen35_qmm_nax.metal
```
Python: `fast.py`, `gdn.py`.

### `bonsai/` — Bonsai quantized + speculative decode kernels
```
csrc/
├── bindings.cpp
├── bonsai_kernels.{cpp,h}
├── bonsai_quantized.metal
├── quantized.h
├── spec_decode.metal
└── CMakeLists.txt
```
Plus shared: `common/csrc/kernels/{quantized_moe.h, steel_attention_block_token.h}`
and `common/csrc/mlx/backend/metal/kernels/steel/attn/params.h`.

---

## EXO current state

`vendor/glm_moe_dsa/kernels.py` provides `_FastDispatch`:
- `__getattr__(name)`: native if `_native_fast.has_symbol(name)`, else `mx.fast`.
- `has(name)`, `missing(required)`, `native_available()`, `native_import_error()`.

So model code calls `fast.something(...)` and transparently gets native-or-fallback.
**The only missing piece is that `_native_fast` is always `None` in EXO because
the extension is never built/installed.**

---

## What EXO needs to do

### Option A — Build oMLX's extension and import it (recommended, fastest)
1. Vendor `omlx/custom_kernels/glm_moe_dsa/csrc/` (and `common/`) into EXO under
   `rust/` or a new `mlx_kernels/` tree.
2. Add a build step (CMake → `.metallib` + PyO3 extension) to `BUILD.sh` /
   `flake.nix`. oMLX's `setup.py` + `CMakeLists.txt` are the reference.
3. Gate behind `OMLX_WITH_CUSTOM_KERNEL=1`-style env var (oMLX pattern) so default
   builds stay light.
4. At runtime, `kernels.py`'s existing `try: from omlx.custom_kernels...` import
   succeeds → `_FastDispatch` uses native symbols automatically.
5. Wire `native_kernel_status()` into EXO's startup log so users can verify.

**Build requirement:** full Xcode (not just CLT) for `xcrun metal`. oMLX README
is explicit: *"Command Line Tools alone do not provide `xcrun: error: unable to
find utility "metal"`: install full Xcode."* EXO's nix flake must declare this
dependency for Darwin.

### Option B — Vendor `fast.py` only, build metallibs, load via `mx.fast`
If importing the full PyO3 extension is too heavy for EXO's packaging, an
intermediate path: precompile `.metallib` files and register kernels via
`mx.fast`'s custom-kernel API. More fragile; only consider if Option A's
PyO3 build conflicts with EXO's Rust toolchain.

### Option C — Ship prebuilt metallibs in releases (like oMLX's DMG)
For binary releases, precompile kernels in CI (Apple Silicon runner) and bundle.
Source builds fall back. This matches oMLX's DMG approach.

---

## Phased plan

### Phase 1 — GLM-5.2 kernels only (the model EXO already supports)
- Vendor `glm_moe_dsa/csrc/` + `common/`.
- Add CMake build to `flake.nix` (Darwin-only, gated on `full Xcode` availability).
- Add `EXO_BUILD_MLX_KERNELS=1` flag to `BUILD.sh`.
- Verify: `python -c "from exo.worker.engines.mlx.vendor.glm_moe_dsa.kernels import fast; print(fast.native_available())"` → `True`.
- Benchmark GLM-5.2 prefill before/after; record tok/s.
- **Tests:** extend `src/exo/worker/tests/unittests/test_mlx/test_glm_moe_dsa_indexer.py`
  to assert native path is taken when built; add a numerical-equivalence test
  (native vs fallback output within tol) so the build-gated path is trustworthy.

### Phase 2 — MiniMax M3 + Qwen3.5 kernels
- Same pattern for `minimax_m3/` and `qwen35_prefill/` once EXO serves those
  families. Lower priority until EXO has model support for them.

### Phase 3 — Dashboard surface
- Show kernel availability (`native_available()`) in the EXO dashboard model
  card, like oMLX's admin panel.

---

## Risks & open questions

- **Xcode dependency:** heavy for CI. Gate the build so default `nix build` /
  `uv run exo` doesn't require it; only opt-in kernel builds do.
- **Metal toolchain vs Rust toolchain coexistence:** EXO's `rust/` uses Cargo.
  CMake-built PyO3 extension is a separate build system; ensure `flake.nix`
  composes them (likely two derivations).
- **Apple Silicon only:** kernels are `.metal`. On non-Darwin EXO builds (Linux
  workers exist in the cluster), the import must gracefully fail → fallback.
  EXO's `try/except` already handles this.
- **Version drift:** pin the oMLX kernel commit EXO vendors from, and add a test
  that diffs vendored `csrc/` against upstream on a schedule.
- **Licensing:** Apache-2.0; carry oMLX's headers. EXO already does this for the
  model code.

---

## Definition of done

- [ ] `EXO_BUILD_MLX_KERNELS=1 nix build` produces a binary with importable
      `exo` kernel extension on Apple Silicon.
- [ ] `fast.native_available()` returns `True` after build; `False` gracefully
      on non-Darwin / no-Xcode builds.
- [ ] GLM-5.2 prefill tok/s ≥ 5× the fallback (target oMLX's ~30×; floor 5× to
      account for different hardware).
- [ ] Numerical-equivalence test (native vs fallback) passes within fp16 tol.
- [ ] `basedpyright` + `ruff` + `nix fmt` + `pytest` all clean.