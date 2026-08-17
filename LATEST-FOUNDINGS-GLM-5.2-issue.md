# LATEST FINDINGS — GLM-5.2 "Produces 0s" Issue

> Research session focused on `jundot/omlx`, `exo-explore/exo`, and GitHub issue #2208.
> All heavy network output cached to `/tmp/glm-research/` and grepped (no re-fetching).
> No source files modified in this session — research only.

---

## TL;DR — Verdict

EXO's Phase 2 implementation (vendored GLM-5.2 model with indexer sharing) is
**correct and complete** as a port of oMLX. Every layer verifiable without the
204GB download checks out:

- ✅ Patch ordering (applied before `mlx_lm.load()`)
- ✅ Indexer sharing configuration (21 full / 57 shared, matches checkpoint)
- ✅ Weight mapping & `sanitize` (checkpoint names match model params)
- ✅ No-native-kernel forward path (causally safe, no NaN, no zero rows)

The original issue #2208 root cause (57 shared layers with random indexer
weights → garbled output >2048 tokens) is exactly what Phase 2 fixes.
**No remaining code-level defect** was found by static analysis + a real-config
synthetic forward test.

The persistent "produces 0s" report is most likely either:
- (a) **stale** — never re-tested after Phase 2, or
- (b) **an environment/build issue** (stale `.venv`, stale binary not running the patched code).

---

## Methodology

To avoid the loop the previous two summaries (`GLM-5.2-DEEP-SUMMARY.md`,
`GLM-5.2-INVESTIGATION-SUMMARY.md`) fell into, this session:

1. Cached all heavy GitHub/HuggingFace API output to `/tmp/glm-research/` once,
   then grepped the local files (cheap, repeatable).
2. Diffed oMLX's patch files against EXO's vendored copies to find behavioral
   differences (not just cosmetic ones).
3. Pulled the **real** `avlp12/GLM-5.2-Alis-MLX-Dynamic-2.3bpw` `config.json`
   and `model.safetensors.index.json` to verify against ground truth.
4. Read issue #2208 (the OPEN root-cause thread) directly — including comments.
5. Built `GlmMoeDsaAttention` from the **real** `ModelArgs` and ran actual
   prefill forwards to test hypotheses empirically (no 204GB download).

Cached artifacts in `/tmp/glm-research/`:
```
avlp12-config.json            (226 KB)  — real model config
avlp12-index.json             (303 KB)  — real safetensors index
omlx_patches_glm_moe_dsa_*.py (×7)      — oMLX reference source
exo-issue-2208.json           + comments — root-cause thread
exo-mlx-commits.json, exo-issues-glm.json, omlx-tree.json
diff-{model,sparse,kernels,dsv32}.txt   — oMLX-vs-EXO diffs
```

---

## Findings (with evidence)

### 1. EXO's vendored GLM code is a FAITHFUL port of oMLX

Diffs of all 4 vendored files vs `jundot/omlx` are **cosmetic only**
(type annotations, docstrings, comment wording). The forward / indexer /
attention LOGIC is identical. The bug is NOT a porting error in the model math.

| File | Diff size | Changed lines | Nature |
|------|-----------|---------------|--------|
| `glm_moe_dsa_model.py` | 139 lines | 101 | formatting + comments + types |
| `deepseek_v32.py` | 311 lines | 199 | type annotations only |
| `sparse_mla.py` | 103 lines | 73 | `hasattr(glm_fast,X)` → `glm_fast.has(X)` (equivalent) + formatting |
| `kernels.py` | 44 lines | — | docstring + formatting |

### 2. Patch is correctly applied BEFORE `mlx_lm.load()`

`src/exo/worker/runner/bootstrap.py:75-77`:
```python
from exo.worker.engines.mlx.patches import apply_mlx_patches
apply_mlx_patches()  # ← registers vendored module
from exo.worker.engines.mlx.builder import MlxBuilder
```

`apply_glm_moe_dsa_patch()` registers the vendored module via
`sys.modules["mlx_lm.models.glm_moe_dsa"] = ...`. Verified at runtime:
```
apply_mlx_patches -> None
registered glm_moe_dsa module file: .../vendor/glm_moe_dsa/glm_moe_dsa_model.py
has Model: True  has ModelArgs: True
```
So the thin shim in the fork is NOT used → the 57 shared layers do get
`indexer = None`. ✅

### 3. Indexer sharing is correctly configured for the REAL avlp12 checkpoint

Fetched the real `config.json` + `model.safetensors.index.json`:

- `indexer_types`: explicit `list[78]` = **21 full + 57 shared**
  pattern `FFFsss Fsss Fsss ...` (layers 0,1,2 full; then 1 full + 3 shared)
- `index_topk: 2048`, `index_n_heads: 32`, `index_head_dim: 128`
- Checkpoint has **5 indexer weights per full layer (105 total)**,
  **0 for shared layers** ✅
- `ModelArgs.__post_init__` keeps the explicit list (derivation only runs when
  `indexer_types is None`), so `skip_topk` / `sanitize` / `make_cache` use the
  correct pattern.

Runtime verification:
```
layer0 (full):   skip_topk=False  indexer is None: False
layer3 (shared): skip_topk=True   indexer is None: True
```

### 4. Issue #2208 (OPEN) confirms the ORIGINAL root cause + fix

Title: *"Long-context requests ... produce garbled output"*. Key comment-thread
findings:

- Reproduced **single-node `world_size=1`, no RDMA, no cross-rank comms**.
- *"corruption onset lines up exactly with `bypass` flipping to `False`"* (>2048).
- Root cause: 57 shared layers get **randomly-initialized indexer weights** →
  garbage top-k indices → valid tokens attended in wrong order → garbled output.
- Workaround posted: `Indexer.__call__` → always `return None` (force dense).
- "Real fix": implement indexer sharing = **exactly EXO Phase 2**.

So EXO Phase 2 fixes the **"garbled"** symptom (random shared-layer weights).
It does NOT necessarily fix a **"0s / NaN"** symptom from a different cause —
which is why this session tested that specifically (see finding 5).

### 5. ❌ REFUTED HYPOTHESIS: argpartition fallback → all-future top-k → NaN/0s

**Initial suspicion:** without native Metal kernels, the argpartition fallback
could select causally-future top-k positions → all-False mask row →
`pe_scores = finfo.min` → softmax → NaN → token 0 / zeros.

**Tested directly** with the REAL avlp12 config (index_topk=2048, explicit
indexer_types, real `ModelArgs`). Ran prefill forwards at cache offsets
N_prev ∈ {3000, 6000, 12000} with L ∈ {16, 64, 256} (sparse path active,
K_total up to 12256 >> 2048):

```
N=12000 L=256 K=12256 | NaN=0 Inf=0 zero_rows=0/256 | all_future_rows=0/256 any_future_rows=0/256
```

**Zero NaN, zero zero-rows, zero all-future rows across ALL configs.**

**WHY it's safe:** the Indexer's prefill fallback explicitly masks scores to
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

So `argpartition` can NEVER select future positions. The oMLX comment about
"argpartition fallback can select future rows" refers to the `L <= 8` decode
path (which separately clamps via `valid = idx <= row_pos`), not the prefill
path. **The no-native prefill path is causally safe.**

### 6. `generate_patch.py` is NOT relevant

oMLX's adaptive prefill patch triggers only when `prefill_step_size == 2048`.
EXO uses `prefill_step_size = 4096` (`generate.py:334`, `batch_generate.py:108`),
so the patch would be inert anyway. And 4096 chunking still crosses the 2048
sparse boundary at chunk 2. **Porting `generate_patch` won't help.**

### 7. The deep-summary's prior "forward path works" test was INVALID

It used a tiny model with:
- (a) the **derived** `indexer_types` (wrong: 19 full at indices {2,6,10,...},
  not the real 21+57 pattern),
- (b) random weights,
- (c) `index_topk=8`, not 2048.

It only proved "no NaN crash", not correctness. Its conclusion ("bug is
elsewhere / sharding") was therefore unfounded. This session's test uses the
real config and is authoritative.

### 8. Additional verification (weight mapping & sanitize)

- `Model` MRO: `Model → Model(vendored DSV32) → Module` — base is EXO's own
  vendored `deepseek_v32`, not the fork's shim. ✓
- Dry-run sanitize against the REAL `model.safetensors.index.json` (3406
  weights): checkpoint has exactly **105 indexer weights (21 full × 5)** and
  **0 for shared layers** — perfectly matches `indexer_types`. GLM `sanitize`
  removes 0 (correct no-op). Full-layer indexer param names
  (`wq_b`, `wk`, `weights_proj`, `k_norm.weight/bias`) match the checkpoint. ✓

---

## What the previous summaries got WRONG

| Previous claim | Status |
|----------------|--------|
| "Missing `install_local_sharded_load_fallback()` is the fix" | ❌ Red herring — EXO never calls `sharded_load`; model has no fused weights. |
| "Forward path works (tiny random model, index_topk=8)" | ❌ Invalid test — wrong indexer pattern, wrong topk, random weights. |
| "Bug is in sharding / pipeline-parallel path" | ❌ Unfounded — issue #2208 reproduced single-node `world_size=1`. |
| "argpartition fallback can select future rows → NaN" | ❌ Refuted by direct test — prefill fallback masks scores to `-inf` first. |
| "Port `generate_patch.py` for correctness" | ❌ Irrelevant — EXO uses prefill_step_size=4096, patch triggers only at 2048. |

---

## Remaining plan (decisive, no loops)

Since static + synthetic verification all pass, the ONLY thing not yet done is
a real end-to-end smoke test. The "produces 0s" report is most likely stale or
environmental.

### Step B — Real end-to-end smoke (THE decisive next step)

1. **Clean rebuild** (rule out stale environment):
   ```bash
   rm -rf .venv
   uv sync
   cd dashboard && npm install && npm run build && cd ..
   ```
2. Start exo and load the model with a prompt that crosses 2048 tokens:
   ```bash
   uv run exo
   # then POST /v1/chat/completions with a ~3000-token prompt
   ```
3. **Check the runner log** for these two lines:
   - `Registered mlx_lm.models.glm_moe_dsa from EXO vendored module`
   - `GLM-5.2 native kernels not available (...); falling back to standard attention path`
4. Branch on the result:
   - **Real text output** → Phase 2 already fixed it; the prior "0s" was
     stale. Loop closed. ✅
   - **Registration line MISSING** → the patch isn't applied in the running
     process → fix the `apply_mlx_patches` call site / packaging.
   - **Registration present but still 0s** → the bug is in a path only
     exercised by the real weights (e.g. a dtype/quantization interaction in
     the base `sanitize` FP8/dequant path) → go to Step C.

### Step C (ONLY if B fails with patch applied) — Real-weight isolated forward

Load just the first ~6 layers of the real avlp12 checkpoint (a few GB, not
204GB), build the vendored model slice, run a prefill >2048 tokens, check
output for NaN. This isolates whether real quantized weights trigger a NaN
that random weights didn't.

If Step C is clean too, the issue is definitively not in the model code and
must be in: generation loop / sampling, the chat-template prompt construction,
or the tool-parser / EOS interaction (all previously claimed "done" but only
unit-tested, never exercised end-to-end with real weights).

---

## Files & sources cited

**EXO vendored (the code under test):**
- `src/exo/worker/engines/mlx/vendor/glm_moe_dsa/{glm_moe_dsa_model,deepseek_v32,sparse_mla,kernels}.py`
- `src/exo/worker/engines/mlx/patches/glm_moe_dsa/__init__.py`
- `src/exo/worker/engines/mlx/patches/__init__.py`
- `src/exo/worker/runner/bootstrap.py:75-77`
- `src/exo/worker/engines/mlx/generator/generate.py:334` (prefill_step_size=4096)

**oMLX reference (fetched, cached in /tmp/glm-research/):**
- `jundot/omlx` `omlx/patches/glm_moe_dsa/{__init__,generate_patch,kernels,glm_moe_dsa_model,deepseek_v32,sparse_mla}.py`
- `jundot/omlx` `omlx/patches/mlx_lm_sharded_load.py`
- `jundot/omlx` `omlx/custom_kernels/glm_moe_dsa/csrc/{dsa_indexer,sparse_mla,exact_block_attention}.metal`

**GitHub:**
- EXO issue #2208 (OPEN): single-node repro, root cause = shared-layer random weights
- EXO PR #2061, #1612, #2187, #1513, #1967, #1691 (GLM / DeepSeek / sharding history)

**Real model metadata (fetched):**
- `avlp12/GLM-5.2-Alis-MLX-Dynamic-2.3bpw` `config.json` + `model.safetensors.index.json`

---

## Environment

- EXO working dir: `<repo root>` (the exo git checkout)
- Branch: `GLM-5.2-ability`
- mlx-lm: 0.31.3 (fork `rltakashige/mlx-lm`, branch `leo/deepseek-v4`)
- mlx: custom build (`rltakashige/mlx-jaccl-fix-small-recv`)
- Model: `avlp12/GLM-5.2-Alis-MLX-Dynamic-2.3bpw` (204GB, NOT downloaded locally)
- oMLX: `jundot/omlx` (NOT cloned; key files fetched via curl to `/tmp/glm-research/`)