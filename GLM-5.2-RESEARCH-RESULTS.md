# GLM-5.2 "Produces 0s" — Research Results (oMLX + EXO + GitHub)

Method: cached all heavy network output to `/tmp/glm-research/` and grepped.
No loops repeated; every conclusion below is backed by a cited source.

## What was actually verified this round (NEW evidence)

### 1. EXO's vendored GLM code is a FAITHFUL port of oMLX
Diffs of all 4 vendored files vs `jundot/omlx` are **cosmetic only**
(type annotations, docstrings, comment wording). The forward/indexer/attention
LOGIC is identical. So the bug is NOT a porting error in the model math.

- `diff omlx .../glm_moe_dsa_model.py` → only formatting/comments
- `diff omlx .../deepseek_v32.py`     → only type annotations
- `diff omlx .../sparse_mla.py`       → only `hasattr(glm_fast,X)` →
  `glm_fast.has(X)` (semantically equivalent) + formatting
- `diff omlx .../kernels.py`          → only docstring + formatting

### 2. Patch is correctly applied BEFORE `mlx_lm.load()`
`src/exo/worker/runner/bootstrap.py:75-77` calls `apply_mlx_patches()` (which
calls `apply_glm_moe_dsa_patch()`) before importing `MlxBuilder`. The vendored
module is registered via `sys.modules["mlx_lm.models.glm_moe_dsa"]`. So the thin
shim is NOT used → the 57 shared layers do get `indexer = None`. ✅

### 3. Indexer sharing is correctly configured for the REAL avlp12 checkpoint
Fetched the real `config.json` + `model.safetensors.index.json`:
- `indexer_types`: explicit list[78] = **21 full + 57 shared**
  pattern `FFFsss Fsss Fsss ...` (layers 0,1,2 full; then 1 full + 3 shared)
- Checkpoint has **5 indexer weights for full layers**, **0 for shared layers** ✅
- `ModelArgs.__post_init__` keeps the explicit list (derivation only runs when
  `indexer_types is None`). So `skip_topk`/`sanitize`/`make_cache` are correct.

### 4. Issue #2208 (OPEN) confirms the ORIGINAL root cause + fix
Title: "Long-context requests ... produce garbled output". Comment thread:
- Reproduced **single-node world_size=1, no RDMA**.
- "corruption onset lines up exactly with `bypass` flipping to `False`" (>2048).
- Root cause: 57 shared layers get **random indexer weights** → garbage top-k.
- Workaround posted: `Indexer.__call__` → always `return None` (force dense).
- "Real fix": indexer sharing = **exactly EXO Phase 2**.

So EXO Phase 2 fixes the **"garbled"** symptom (random shared-layer weights).
It does NOT necessarily fix a **"0s / NaN"** symptom from a different cause.

### 5. ❌ REFUTED HYPOTHESIS: argpartition-fallback → all-future top-k → NaN/0s

Initial suspicion: without native kernels, the argpartition fallback could
select causally-future top-k positions → all-False mask row → NaN → 0s.

**Tested directly with the REAL avlp12 config** (index_topk=2048, explicit
indexer_types, real ModelArgs). Ran prefill forwards at cache offsets
N_prev ∈ {3000, 6000, 12000} with L ∈ {16, 64, 256} (sparse path active,
K_total up to 12256 >> 2048):

```
N=12000 L=256 K=12256 | NaN=0 Inf=0 zero_rows=0/256 | all_future_rows=0/256 any_future_rows=0/256
```

**Zero NaN, zero zero-rows, zero all-future rows across ALL configs.**

WHY it's safe: the Indexer's prefill fallback explicitly masks scores to
`-inf` for causally-future positions BEFORE `argpartition`:
```python
# deepseek_v32.py Indexer.__call__, prefill fallback (no native kernels):
head_scores = q @ k.swapaxes(-1, -2)
scores = mx.maximum(head_scores, 0) * weights
scores = scores.sum(axis=1, keepdims=True)
fuse_causal_mask = False
if mask is not None and not fuse_causal_mask:
    scores = mx.where(mask, scores, -float("inf"))   # <-- future -> -inf
return select_topk(scores, prefix_topk_rows)          # argpartition never picks -inf
```
So argpartition can NEVER select future positions. The oMLX comment about
"argpartition fallback can select future rows" refers to the `L<=8` decode
path (which separately clamps via `valid = idx <= row_pos`), not the prefill
path. **The no-native prefill path is causally safe.**

### 6. `generate_patch.py` is NOT the issue
oMLX's adaptive prefill triggers only when `prefill_step_size == 2048`.
EXO uses `prefill_step_size = 4096` (generate.py:334, batch_generate.py:108),
so the patch would be inert anyway. And 4096 chunking still crosses the 2048
sparse boundary at chunk 2. So porting generate_patch won't help.

### 7. The deep-summary's "forward path works" test is INVALID
It used a tiny model with (a) the DERIVED indexer_types (wrong: 19 full at
indices {2,6,10,...}, not 21+57), (b) random weights, (c) index_topk=8 not
2048. It only proved "no NaN crash", not correctness. Its conclusion
("bug is elsewhere / sharding") is therefore unfounded.

## Concrete next-step plan (few checks, no loops)

### Check A — DONE ✅ (refuted the NaN hypothesis)
Built `GlmMoeDsaAttention` from the REAL avlp12 `ModelArgs` (explicit
indexer_types, index_topk=2048) and ran prefill forwards at cache offsets up
to 12000 with L up to 256. **No NaN, no zero-rows, no all-future top-k.**
The prefill fallback masks scores to `-inf` for future positions before
`argpartition`, so it is causally safe. (See section 5.)

### Additional verification DONE ✅
- `Model` MRO: `Model → Model(vendored DSV32) → Module` — base is EXO's own
  vendored deepseek_v32, not the fork's shim. ✓
- Dry-run sanitize against the REAL `model.safetensors.index.json` (3406
  weights): checkpoint has exactly **105 indexer weights (21 full × 5)** and
  **0 for shared layers** — perfectly matches `indexer_types`. GLM sanitize
  removes 0 (correct no-op). Full-layer indexer param names (`wq_b`, `wk`,
  `weights_proj`, `k_norm.weight/bias`) match the checkpoint. ✓

## VERDICT

EXO's Phase 2 implementation (vendored GLM-5.2 model with indexer sharing)
is **correct and complete** as a port of oMLX. Every layer I could verify
without the 204GB download checks out: patch ordering, indexer sharing
config, weight mapping, sanitize, and the no-native-kernel forward path.

The original issue #2208 root cause (57 shared layers with random indexer
weights → garbled output >2048 tokens) is exactly what Phase 2 fixes. There
is **no remaining code-level defect** I can find by static analysis + the
synthetic forward test.

## Remaining plan (the ONLY thing not yet done: real end-to-end smoke)

Since static + synthetic verification all pass, the "produces 0s" report is
most likely either (a) stale — never re-tested after Phase 2 — or (b) an
environment/build issue (old build, stale `.venv`, patch not in the running
binary). The decisive next step:

### Check B — Real end-to-end smoke (decisive)
1. Rebuild from clean: `rm -rf dashboard/build .venv 2>/dev/null; uv sync; cd
   dashboard && npm install && npm run build && cd ..`
2. Start exo, load the model with a prompt that crosses 2048 tokens:
   `uv run exo` then POST `/v1/chat/completions` with a ~3000-token prompt.
3. If output is real text → Phase 2 already fixed it; the prior "0s" was
   stale. Close the loop.
4. If output is still 0s/garbled → capture the runner log and check for the
   `Registered ... from EXO vendored module` line + `GLM-5.2 native kernels
   not available` warning. If the registration line is MISSING, the patch
   isn't applied in the running process → fix `apply_mlx_patches` call site.
   If present but still 0s → the bug is in a path only exercised by the real
   weights (e.g. a dtype/quantization interaction in the base `sanitize`
   FP8/dequant path) → report and move to Check C.

### Check C (only if B shows 0s with patch applied) — real-weight isolated forward
Load just the first ~6 layers of the real avlp12 checkpoint (a few GB, not
204GB), build the vendored model slice, run a prefill >2048 tokens, check
output for NaN. This isolates whether real weights + quantization trigger a
NaN that random weights didn't.

## Files / sources cited
- EXO vendored: `src/exo/worker/engines/mlx/vendor/glm_moe_dsa/{glm_moe_dsa_model,deepseek_v32,sparse_mla,kernels}.py`
- EXO patch: `src/exo/worker/engines/mlx/patches/glm_moe_dsa/__init__.py`
- EXO bootstrap: `src/exo/worker/runner/bootstrap.py:75-77`
- EXO generate: `src/exo/worker/engines/mlx/generator/generate.py:334`
- oMLX (fetched): `jundot/omlx` `omlx/patches/glm_moe_dsa/{__init__,generate_patch,kernels,glm_moe_dsa_model,deepseek_v32,sparse_mla}.py`
- oMLX native kernels: `jundot/omlx` `omlx/custom_kernels/glm_moe_dsa/csrc/{dsa_indexer,sparse_mla,exact_block_attention}.metal`
- EXO issue #2208 (OPEN): single-node repro, root cause = shared-layer random weights
- Real model config: `avlp12/GLM-5.2-Alis-MLX-Dynamic-2.3bpw` config.json + model.safetensors.index.json (cached `/tmp/glm-research/`)