# GLM-5.2 Deep Investigation Summary — Findings & New Plan

## Executive Summary

The `GLM-5.2-INVESTIGATION-SUMMARY.md` concluded that the "most likely remaining
cause" of the `produces 0s` issue is the **missing
`install_local_sharded_load_fallback()`** from oMLX. This investigation
**disproves that conclusion** for EXO's code path and reframes the problem.

## What the previous investigation got right

1. Thinking-token detection is correctly wired (verified with real tokenizer).
2. Tool parser detection is correct for GLM-5.2's `XCTS` format.
3. EOS token handling is correct (`[154820, 154827, 154829]`).
4. The indexer-sharing patch is structurally correct (8 unit tests pass).
5. EXO's `shard_and_load()` calls `load_model()` directly, not `sharded_load`.

## What the previous investigation got WRONG

### 1. The `install_local_sharded_load_fallback()` is a RED HERRING for EXO

The previous summary claimed the missing fallback is "likely real fix". The
oMLX `apply_glm_moe_dsa_patch()` calls two additional sub-patches that EXO
doesn't:

- `apply_glm_moe_dsa_generate_patch()` — adaptive prefill chunking
  (performance only, not correctness).
- `install_local_sharded_load_fallback()` — wraps `mlx_lm.utils.sharded_load`
  and `mlx_lm.server.sharded_load` to fall back to local loading when the
  sharded_load weight-index gate rejects fused parameter names.

**The fallback only matters if code calls `sharded_load` directly.** EXO does
not:

- `grep -rn "sharded_load" src/exo/` → **zero hits** in EXO source.
- EXO's `shard_and_load()` in `utils_mlx.py:232` calls `load_model()` directly,
  then manually shards via `tensor_auto_parallel` / `pipeline_auto_parallel`.
- The sharded_load gate (lines 586-645 in pinned `mlx_lm/utils.py`) is never
  reached on EXO's path.

So porting `install_local_sharded_load_fallback()` into EXO would be a no-op
for EXO's load path. It would only help if EXO called `mlx_lm.utils.sharded_load`
directly, which it doesn't.

### 2. The wk/weights_proj fusion does NOT happen for this model

The `avlp12/GLM-5.2-Alis-MLX-Dynamic-2.3bpw` model has **NO indexer
quantization entries** in its `quantization` config:

```python
# HuggingFace config.json → quantization dict has 0 indexer keys
# model.layers.0.self_attn.indexer.wk → MISSING
# model.layers.0.self_attn.indexer.weights_proj → MISSING
# model.layers.0.self_attn.indexer.wk_weights_proj → MISSING
```

The vendored `_use_load_fused_wk_weights_proj()` requires indexer Q8
quantization (`bits=8, group_size=64, mode=affine`) to fuse `wk` +
`weights_proj` → `wk_weights_proj`. For this model, that function returns
**False**. So:

- The fusion in `deepseek_v32.py:sanitize()` (lines 872-903) is **not
  triggered**.
- The fused parameter name `wk_weights_proj` is **never constructed**.
- The `sharded_load` weight-index gate (which rejects fused names) is **not
  relevant** even if it were called.

The "Alis dynamic-2.3bpw" quantization is a standard 2-4-6-bit affine scheme
without per-indexer Q8 fusion. The oMLX fallback was written for models that
DO have indexer Q8 fusion (e.g., the `mlx-community/GLM-5.2-*` Q8 variants),
not for this dynamic-quantization variant.

### 3. The forward path is NOT producing zeros

Two integration tests (built tiny GLM models with random weights, ran actual
forward passes):

- **Below index_topk** (5 tokens, index_topk=8): output non-zero, non-NaN.
- **Above index_topk** (10 tokens, triggering sparse top-k path): output
  non-zero, non-NaN.

This means the vendored GLM model code — including indexer sharing, sparse
top-k mask, fallback SDPA path — is **structurally correct** and does not
produce zeros in isolation. The bug is **elsewhere**.

## What the actual bug likely IS

Given that the forward path works and the previous fix is structurally correct,
the remaining causes must be in one of these areas (none fully investigated
yet):

### A. Pipeline/tensor parallel sharding path (HIGH SUSPECT)

`DeepSeekShardingStrategy.shard_model()` in `auto_parallel.py:698` shards the
MLA projections (`q_b_proj`, `o_proj`, `embed_q`, `unembed_out`) and the MoE,
but **does NOT shard the indexer weights** for full layers.

The indexer has `wq_b`, `wk`/`weights_proj` (or fused `wk_weights_proj`),
`k_norm` — these are **not sharded** by the strategy. On a multi-node tensor
parallel run, every rank would load the full indexer weights, leading to:

- Duplicate indexer computation across ranks.
- Potential memory overflow.
- Incorrect top-k indices if the indexer is not sharded correctly.

The `GlmMoeDsaModel` is in the sharding strategy list (line 516), but the
strategy is the generic `DeepSeekShardingStrategy` which doesn't know about
GLM's indexer. oMLX likely has a GLM-specific sharding strategy that EXO is
missing.

### B. The actual model load / weight mapping

The checkpoint has 46 safetensors shards. The model has 78 layers + 1 MTP
layer. The `load_model(strict=False)` path may be silently ignoring missing
weights or mis-mapping fused/unfused weights. Need to verify the actual
parameter tree after load matches the checkpoint exactly.

### C. Generation path / sampling / EOS interaction

The investigation said EOS is correct, but didn't verify the actual generation
loop. The `stream_generate` call and the `ban_token_ids` / stop_tokens
interaction may have a bug specific to GLM-5.2's thinking tokens.

### D. Model card / download path

The model card may have incorrect fields (e.g., `n_layers`, `hidden_size`)
that cause the wrong model class to be instantiated, or the download to miss
files. Need to verify the card matches `config.json` exactly.

## New Investigation Plan (avoiding the previous loop)

### Step 1: Verify the sharding strategy handles GLM indexer (HIGH PRIORITY)

Check whether `DeepSeekShardingStrategy` is correct for GLM-5.2, or whether
oMLX has a GLM-specific sharding strategy that EXO is missing. Fetch the oMLX
sharding code and compare.

```bash
curl -sL "https://raw.githubusercontent.com/jundot/omlx/main/omlx/patches/glm_moe_dsa/__init__.py"
# Look for sharding strategy code
```

### Step 2: Verify the actual parameter tree after load

Build the GLM model from the real config, inspect the parameter tree, and
compare against `model.safetensors.index.json`. Check for missing/mis-mapped
weights, especially the indexer weights for full layers.

### Step 3: Verify the generation path with a real tokenizer

Load the GLM-5.2 tokenizer, run a short prompt through the generation path,
and check the output tokens. Verify EOS detection, thinking tokens, and
sampling.

### Step 4: Verify the model card fields

Compare `avlp12--GLM-5.2-Alis-MLX-Dynamic-2.3bpw.toml` against the actual
`config.json` and `model.safetensors.index.json`. Check `n_layers`,
`hidden_size`, `num_key_value_heads`, `context_length`, `storage_size`.

### Step 5: If all above pass, re-examine the forward pass with REAL weights

The tiny model test uses random weights. The real model has specific weights
that may trigger NaN/inf in the sparse path. Need to load the real weights
(or a subset) and run a forward pass with context > 2048 to check for NaN.

## What NOT to do (avoiding the loop)

- **Do NOT port `install_local_sharded_load_fallback()`** — it's a red herring
  for EXO's path. EXO doesn't call `sharded_load`, and this model doesn't fuse
  weights.
- **Do NOT re-verify thinking-token detection** — already verified correct.
- **Do NOT re-verify EOS handling** — already verified correct.
- **Do NOT re-run the 8 unit tests** — they pass but don't test generation.

## Key Files

- `src/exo/worker/engines/mlx/patches/glm_moe_dsa/__init__.py` — patch
  registration (correct).
- `src/exo/worker/engines/mlx/vendor/glm_moe_dsa/glm_moe_dsa_model.py` —
  vendored GLM model with indexer sharing (structurally correct).
- `src/exo/worker/engines/mlx/vendor/glm_moe_dsa/deepseek_v32.py` — vendored
  DeepSeek-V3.2 with wk/weights_proj fusion (fusion not triggered for this
  model).
- `src/exo/worker/engines/mlx/vendor/glm_moe_dsa/kernels.py` — fast kernel
  dispatch, falls back to `mx.fast` (EXO doesn't ship native kernels).
- `src/exo/worker/engines/mlx/auto_parallel.py:698` —
  `DeepSeekShardingStrategy.shard_model` — **does not shard indexer weights**.
- `src/exo/worker/engines/mlx/utils_mlx.py:232` — `shard_and_load` calls
  `load_model` directly, not `sharded_load`.
- `resources/inference_model_cards/avlp12--GLM-5.2-Alis-MLX-Dynamic-2.3bpw.toml`
  — model card.

## Environment

- EXO working dir: `<repo root>` (the exo git checkout)
- mlx-lm: 0.31.3 (git `rltakashige/mlx-lm`, branch `leo/deepseek-v4`)
- Model: `avlp12/GLM-5.2-Alis-MLX-Dynamic-2.3bpw` (204GB, not downloaded
  locally)
- oMLX: `jundot/omlx` (not cloned locally; fetched key files via curl)