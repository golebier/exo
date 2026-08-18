# 06 — GLM Adaptive Prefill Patch

**Tier:** ⭐⭐ (Tier 2)
**Effort:** Low
**Impact:** Medium (unlocks native-kernel fast path; fixes >2048 boundary)
**oMLX source:** `omlx/patches/glm_moe_dsa/generate_patch.py`
**EXO target:** `src/exo/worker/engines/mlx/patches/glm_moe_dsa/` (new), `generator/generate.py`, `generator/batch_generate.py`

---

## What it is

The **one GLM-5.2 patch EXO deliberately did not port.** oMLX's
`generate_patch.py` adapts `mlx_lm.generate`'s prefill chunking specifically for
GLM's DSA (Dynamic Sparse Attention) so the sparse path engages correctly at
the 2048-token boundary.

EXO's `../gra/GLM-5.2-RESEARCH-RESULTS.md` §6 notes:

> oMLX's adaptive prefill triggers only when `prefill_step_size == 2048`. EXO
> uses `prefill_step_size = 4096` (`generate.py:334`, `batch_generate.py:108`),
> so the patch would be inert anyway. And 4096 chunking still crosses the 2048
> sparse boundary at chunk 2. So porting generate_patch won't help.

**That conclusion was correct at the time**, but it's worth revisiting now that:
1. EXO's GLM port is verified correct (Phase 2 indexer sharing).
2. Doc 02 (native kernels) would make the 2048 sparse path fast instead of slow.
3. The >2048-token garbled-output issue (#2208) is exactly the boundary this
   patch targets.

If EXO ships native kernels (doc 02), switching to `prefill_step_size=2048` +
this patch **unlocks the fused DSA prefill** and aligns with oMLX's validated
configuration.

---

## oMLX design

`omlx/patches/glm_moe_dsa/generate_patch.py`:

```python
@dataclass(frozen=True)
class _AdaptivePrefillConfig:
    step_size: int
    after: int
    min_remaining: int

def _glm_dsa_adaptive_prefill_config(
    model: Any, prefill_step_size: int
) -> _AdaptivePrefillConfig | None:
    model_type = getattr(model, "model_type", None) or getattr(
        getattr(model, "args", None), "model_type", None
    )
    if (
        model_type != "glm_moe_dsa"
        or prefill_step_size != 2048
        or os.environ.get("MLX_LM_GLM_DSA_ADAPTIVE_PREFILL_STEP", "1") != "1"
    ):
        return None
    ...
```

Key points:
- **Gated** to `model_type == "glm_moe_dsa"` AND `prefill_step_size == 2048`.
- Env override `MLX_LM_GLM_DSA_ADAPTIVE_PREFILL_STEP` (default enabled).
- Returns a `_AdaptivePrefillConfig(step_size, after, min_remaining)` that
  changes chunking behavior around the sparse boundary.
- Ports "the small GLM-specific prefill chunking change from the optimized
  mlx-lm snapshot without replacing the whole `mlx_lm.generate` module" —
  intentionally surgical, **inert for non-GLM models**.
- `_APPLIED` idempotency flag; uses `_sync_and_clear_cache` from
  `omlx/utils/metal_sync`.

The patch is a **runtime monkey-patch on `mlx_lm.generate`**, applied alongside
the model code patch.

---

## EXO current state

- EXO's GLM patch: `src/exo/worker/engines/mlx/patches/glm_moe_dsa/__init__.py`
  registers the vendored model via `sys.modules["mlx_lm.models.glm_moe_dsa"]`.
- EXO's `kernels.py` falls back to `mx.fast` (no native kernels).
- `generate.py:334` and `batch_generate.py:108` use `prefill_step_size = 4096`.
- No `generate_patch.py` equivalent exists in EXO's `patches/glm_moe_dsa/`.

---

## Integration seam in EXO

- **Patch site:** new
  `src/exo/worker/engines/mlx/patches/glm_moe_dsa/generate_patch.py`, applied
  from `patches/glm_moe_dsa/__init__.py` (extend `apply_glm_moe_dsa_patch()`).
- Respect bootstrap ordering (`bootstrap.py:75-77`).
- **Prefill step size:** make `prefill_step_size` configurable per model family
  (currently hard-coded 4096). For `glm_moe_dsa` with native kernels available,
  default to 2048; without native kernels, keep 4096 (the patch is inert at 4096
  anyway, so this is safe).
- **Env override:** port `MLX_LM_GLM_DSA_ADAPTIVE_PREFILL_STEP` →
  `EXO_GLM_DSA_ADAPTIVE_PREFILL_STEP`.

---

## Phased plan

### Phase 1 — Port the patch (inert until step size changes)
- Port `generate_patch.py` verbatim (cosmetic adaptation to EXO style).
- Wire into `apply_glm_moe_dsa_patch()`.
- At EXO's current `prefill_step_size=4096`, the patch is inert (gate returns
  `None`) — so this is safe to land without behavior change.
- **Tests:** unit test that the config function returns `None` at 4096 and a
  config at 2048 for `glm_moe_dsa`; idempotency test (`_APPLIED`).

### Phase 2 — Enable 2048 step size when native kernels are present
- Gate `prefill_step_size` on `fast.native_available()` (doc 02): 2048 if
  native, else 4096.
- This activates the adaptive prefill patch.
- **Tests:** end-to-end GLM-5.2 prefill >2048 tokens produces correct (non-
  garbled, non-zero) output; benchmark TTFT vs 4096 fallback.
- This is the decisive "Check B" from `../gra/GLM-5.2-RESEARCH-RESULTS.md` — the real
  end-to-end smoke that was never completed.

### Phase 3 — Validate against issue #2208
- Reproduce the original >2048 garbled-output scenario from #2208.
- Confirm Phase 2 fixes it (indexer sharing + adaptive prefill + native kernels
  all engaged).

---

## Risks & open questions

- **Depends on doc 02 (native kernels):** without native kernels, switching to
  2048 makes the sparse fallback path slower and may not help. Land Phase 1
  (inert) anytime; land Phase 2 only after doc 02 Phase 1.
- **Correctness at the boundary:** EXO's research (§5) proved the *no-native*
  prefill path is causally safe (no NaN/0s) via the `-inf` mask before
  `argpartition`. The adaptive patch changes *chunking*, not the mask logic, so
  safety should carry over — but re-verify with the synthetic forward test from
  §5 at `prefill_step_size=2048`.
- **`metal_sync` dependency:** oMLX's patch imports `_sync_and_clear_cache` from
  `omlx.utils.metal_sync`. EXO needs an equivalent (likely already has via
  `mx.eval` + `mx.clear_cache` patterns in `cache.py`).
- **Interaction with EXO's remote prefill:** if a prompt is prefilled remotely
  (`remote_prefill.py`), does the adaptive chunking apply on the remote node
  too? Ensure the patch is applied on both ends, or remote prefill stays on the
  standard path.

---

## Definition of done

- [ ] Phase 1: `generate_patch.py` ported; inert at 4096; unit tests green.
- [ ] Phase 2: with native kernels, 2048 step size + adaptive patch produces
      correct GLM-5.2 output >2048 tokens (the #2208 repro is fixed).
- [ ] TTFT benchmark: 2048+native ≥ 4096+fallback on GLM-5.2 prefill.
- [ ] Synthetic forward test (§5) re-run at 2048: zero NaN, zero zero-rows.
- [ ] `basedpyright` + `ruff` + `nix fmt` + `pytest` clean.