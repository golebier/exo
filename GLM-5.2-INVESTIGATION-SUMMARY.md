# GLM-5.2 "Produces 0s" Investigation Summary

## Background

The `GLM-5.2-EXO-PLAN.md` claims all phases (1–6) are DONE, but the model
still produces zeros/garbled output instead of thinking. This document
summarizes the investigation into why the implemented fix isn't working in
practice.

## Investigation Steps Performed

1. **Read the existing plan and code.** Examined `GLM-5.2-EXO-PLAN.md`,
   the vendored model in `src/exo/worker/engines/mlx/vendor/glm_moe_dsa/`,
   the patch registration in `src/exo/worker/engines/mlx/patches/`, and the
   generation path in `src/exo/worker/engines/mlx/generator/generate.py`.
2. **Ran the existing unit tests.** All 8 tests in
   `test_glm_moe_dsa_indexer.py` pass — structural correctness is verified,
   but they don't test actual generation.
3. **Fetched GLM-5.2 tokenizer/config from HuggingFace.** Downloaded
   `tokenizer.json` and `config.json` from `zai-org/GLM-5.2` to inspect the
   actual token IDs and thinking-token structure.
4. **Compared oMLX vs EXO vendored code.** Diffed
   `jundot/omlx/omlx/patches/glm_moe_dsa/` against EXO's vendored copy.
   Core logic is identical; differences are cosmetic (type annotations,
   docstrings, comment wording).
5. **Identified two missing patches.** The oMLX `apply_glm_moe_dsa_patch()`
   calls two additional sub-patches that EXO does not.

## Key Findings

### 1. Thinking-token detection is NOT the issue

Initial suspicion was that `mlx_lm`'s `_infer_thinking()` fails for GLM-5.2
because ` XEND` is not a single token in `get_vocab()`. However, testing with
the real GLM-5.2 tokenizer showed `_infer_thinking()` correctly returns
`('ikat', ' XEND', (154841,), (154842,))` and `has_thinking` is True. The
thinking-token path is wired correctly. This is **not** the cause of the 0s.

### 2. Missing `install_local_sharded_load_fallback()` — likely real fix

The oMLX `apply_glm_moe_dsa_patch()` does two things EXO doesn't:

- `apply_glm_moe_dsa_generate_patch()` — adaptive prefill chunking
  (performance only, not correctness).
- `install_local_sharded_load_fallback()` — **correctness-critical**.

The oMLX comment on the second patch reads:

> `sharded_load` gates on mapping every constructed parameter name through
> `model.safetensors.index.json` before downloading, but an architecture
> whose `sanitize()` fuses weights at load time (glm_moe_dsa's DSA indexer
> fusing `wk` and `weights_proj` into `wk_weights_proj` …) constructs names
> the index cannot contain, so the gate raises "Pipeline loading is only
> supported for MLX converted models" for a checkpoint the normal loader
> handles fine.

EXO's `shard_and_load()` calls `load_model()` directly (not `sharded_load`),
so single-node loads may be unaffected. But if the user runs sharded/tensor
parallel, the missing fallback can cause a load failure or incorrect weight
fusion, which would produce garbage output.

### 3. Output-parser model-type branch

`model_output_parsers.py` uses
`issubclass(model_type, DeepseekV32Model) and "deepseek" in normalized_id`.
GLM-5.2's vendored `Model` subclasses `DeepseekV32Model`, but the model ID
`avlp12/GLM-5.2-Alis-MLX-Dynamic-2.3bpw` doesn't contain "deepseek", so it
falls through to the `else` branch. That branch uses
`parse_thinking_models(...)` with `tokenizer.think_start`/`think_end`, which
is correct since `has_thinking` is True. This is **not** a bug.

## Conclusion

The most likely remaining cause of "produces 0s" is the missing
`install_local_sharded_load_fallback()` for sharded loading of GLM-5.2's
fused DSA indexer weights. The adaptive prefill patch is a performance
optimization, not a correctness fix.

## Next-step plan

1. Port `install_local_sharded_load_fallback()` from oMLX into EXO's
   `src/exo/worker/engines/mlx/patches/` and call it from
   `apply_glm_moe_dsa_patch()`.
2. Port `apply_glm_moe_dsa_generate_patch()` for prefill performance.
3. Run a real model load (single-node and sharded) to confirm the fix
   eliminates the 0s.
4. If 0s persist, re-examine the forward-pass fallback paths in
   `glm_moe_dsa_model.py` for NaN/inf propagation.