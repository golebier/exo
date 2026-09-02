# GLM-5.3 (`glm5_next`) Porting Plan — EXO

**Goal:** run `mlx-community/GLM-5.3-4bit` and `pipenetwork/GLM-5.3-Flash-MLX-8bit` in EXO.
**Source:** `jundot/omlx` `omlx/patches/mlx_vlm_glm5_next_compat/` (cloned to `/tmp/omlx-fresh`).
**Date:** 2025-09-02

## Architecture facts

GLM-5.3 is architecturally **`glm5_next`** (`Glm5NextForConditionalGeneration`) — a **VLM**
(loaded via `mlx_vlm`, not `mlx_lm`). EXO already depends on `mlx-vlm>=0.3.11` (installed
0.4.4), but 0.4.4 does **not** ship `glm5_next`, `deepseek_v4`, `hyper_connection`, or
`PoolingCache` — all must be vendored.

GLM-5.3's language model **reuses GLM-5.2's DSA components** (which EXO already has vendored
under `vendor/glm_moe_dsa/`): `deepseek_v32.Model`/`group_expert_select`, `sparse_mla.*`,
`switch_layers.SwitchGLU`. The **new** pieces are:
- **Gated-delta (GDN) linear-attention layers** (`gated_delta.py`) — hybrid cache (`ArraysCache`)
- **HyperConnection** (`hyper_connection.py`, from DeepSeek-V4) — `attn_hc`/`ffn_hc` per layer
- **PoolingCache** (`cache_extras.py`, from DeepSeek-V4) — pooled indexer cache, wrapped in
  `CacheList(KVCache(), PoolingCache(...))` for sparse-attention layers
- **`linear.py`** — affine-quantized matmul routing (falls back to `mx.quantized_matmul`
  without oMLX's qwen35 native kernels — fine for EXO)
- **Vision tower + torch-free image processor** (`vision.py`, `processing.py`)

## EOS / tool-parser

EXO's `get_eos_token_ids_for_model` already returns `[154820, 154827, 154829]` for any
`glm-5*` model id (matches `glm-5.3`). GLM-5.3 uses the same tool-call token grammar as
GLM-5.2, so the existing `glm52` parser applies — extend the `glm-5.2`/`glm-5.1` branch to
`glm-5.3`.

## Dependency graph (what to vendor + import rewrites)

`language.py` imports → EXO resolution:
| oMLX import | EXO resolution |
|---|---|
| `mlx_vlm.models.base.*` | exists in mlx_vlm 0.4.4 ✅ |
| `mlx_vlm.models.cache` (ArraysCache, CacheList, KVCache) | exists ✅ (PoolingCache injected by patch) |
| `mlx_vlm.models.deepseek_v4.hyper_connection` | **vendor** → register as `mlx_vlm.models.deepseek_v4.hyper_connection` |
| `mlx_lm.models.mla.MultiLinear` | exists ✅ |
| `omlx.patches.deepseek_v4.switch_layers.SwitchGLU` | **vendor** (or reuse EXO `vendor/glm_moe_dsa/switch_layers.SwitchGLU`) |
| `omlx.patches.glm_moe_dsa.deepseek_v32.{Model, group_expert_select}` | reuse EXO `vendor/glm_moe_dsa/deepseek_v32` ✅ |
| `omlx.patches.glm_moe_dsa.sparse_mla.*` | reuse EXO `vendor/glm_moe_dsa/sparse_mla` ✅ |
| `.config/.gated_delta/.linear` | local (vendor) |

## Phases

> **Status: All phases complete (2026-01).** `uv run ruff check`, `uv run basedpyright`, and
> import/card/EOS smoke tests all pass. `uv run pytest` shows no regressions (560 passed;
> 1 pre-existing Rust-binding failure and 8 pre-existing download-test collection errors,
> both unrelated to this port). Remaining work: download a real GLM-5.3 checkpoint and run an
> end-to-end chat completion (requires ~400 GB disk for the 4-bit model).

### Phase 1 — Vendor the GLM-5.3 model package + DeepSeek-V4 deps — ✅ DONE
- `vendor/glm5_next/`: config, glm5_next, language, vision, processing, gated_delta, linear
  (rewrite `omlx.patches.*` imports → EXO paths; `mlx_vlm.models.deepseek_v4.hyper_connection`
  → vendored module)
- `vendor/deepseek_v4/`: hyper_connection.py, cache_extras.py (PoolingCache/BatchPoolingCache),
  switch_layers.py (if needed separately)
- `linear.py`: drop the `omlx.custom_kernels.qwen35_prefill` native-qmm path (keep the
  `mx.quantized_matmul` fallback) — EXO has no qwen35 kernels.

### Phase 2 — Registration patch — ✅ DONE
- `patches/glm5_next/__init__.py`: `apply_glm5_next_patch()` — inject PoolingCache into
  `mlx_lm.models.cache`, register `mlx_vlm.models.deepseek_v4.hyper_connection`, expose
  `mlx_vlm.models.glm5_next` from the vendor tree, set `MODEL_CONFIG["glm5_next"]`.
- Wire into `patches/__init__.py::apply_mlx_patches()`.

### Phase 3 — Cache-type support in EXO's `cache.py` — ✅ DONE (no change needed)
- `PoolingCache` is non-trimmable (pool state). Add to `is_non_trimmable_cache_entry` /
  snapshot handling so `KVPrefixCache` trim/snapshot treats it correctly (it's wrapped in
  `CacheList` with a `KVCache`, so the `CacheList` path already handles "trim the trimmable
  sub-cache, leave the rest").

### Phase 4 — Model cards + EOS/parser wiring — ✅ DONE
- Model cards for `mlx-community/GLM-5.3-4bit`, `pipenetwork/GLM-5.3-Flash-MLX-8bit`.
- Extend `glm52` parser branch to `glm-5.3`.
- VLM model-type alias if needed.

### Phase 5 — Verification — ✅ DONE (static + smoke; end-to-end pending model download)
- `uv run basedpyright && uv run ruff check && nix fmt && uv run pytest`
- Import smoke test: `apply_glm5_next_patch()` then `import mlx_vlm.models.glm5_next`.

## What is NOT ported (out of scope, future)
- Lightning MTP (intentionally excluded by oMLX too)
- Native qwen35 affine-qmm prefill tile (EXO has no qwen35 kernels; `mx.quantized_matmul`
  fallback works, just slower on long prefills)
- Native GLM DSA indexer kernels for GLM-5.3 (EXO's `glm_moe_dsa` kernels are default-off
  already; the MLX fallback path in `language.py` is used)