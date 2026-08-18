# 11 — Memory Headroom Before Prefill + Near-Limit Placement

**Tier:** ⭐ (Tier 3)
**Effort:** Low
**Impact:** Medium (OOM avoidance; reliability)
**Upstream evidence:**
- PR #2251 `fix: reserve memory headroom before prefill`
- #1709 `[BUG] can't place models close to memory limits`
- PR #2240 `Add configurable context-window with memory-safety validation`
- PR #2241 `Add Max Context Length control to instance launch Advanced Options`

---

## What it is

Two related memory-safety gaps:

### A. Prefill OOM from prefix-cache starvation (#2251)
> Persistent prefix-cache allocations can leave too little room for temporary prefill activations. Existing eviction only runs while adding a completed cache entry, **after** the dangerous allocation has already happened.
>
> Changes:
> - evict LRU prefix entries **before** cache lookup and local/remote prefill
> - introduce `EXO_PREFILL_MEMORY_THRESHOLD`, defaulting to 10 points below `EXO_MEMORY_THRESHOLD`
> - validate the prefill watermark against the normal cache watermark
> - preserve distributed max-pressure checks and reclaim MLX allocations after eviction
> - regression coverage for prefill-specific LRU eviction

This is a focused, well-specified fix — eviction happens too late, so a prefill activation allocation OOMs.

### B. Placement near memory limits (#1709)
EXO places models up to the memory ceiling, leaving no room for activations → models fail to load or crash after load. #2240/#2241 add configurable context-window with memory-safety validation.

---

## Why it matters

- **Reliability:** OOM crashes during prefill are a top crash category (interacts with #2208 hangs, #2067 load crashes).
- **Small, safe, high-leverage:** #2251 is a tight fix with tests already written. Low risk, immediate reliability win.
- **Composes with TurboQuant (doc 07):** KV quantization reduces the persistent-cache pressure that causes #2251's starvation.
- **Composes with placement (doc 05):** the memory-feasibility constraint in placement must include activation headroom, not just weights + KV.

---

## Upstream PR landscape

| PR/Issue | Scope |
|----------|-------|
| #2251 | Prefill headroom (`EXO_PREFILL_MEMORY_THRESHOLD`, evict before prefill) — **the fix, ready** |
| #1709 | Can't place models close to memory limits (the placement side) |
| #2240 | Configurable context-window with memory-safety validation |
| #2241 | Max Context Length control in instance launch Advanced Options (dashboard) |

#2251 fixes the runtime side; #2240/#2241 fix the placement/config side; #1709 is the underlying complaint.

---

## EXO current state (local fork)

`src/exo/worker/engines/mlx/cache.py`:
- `KVPrefixCache._evict_if_needed()` — evicts LRU **while adding a completed cache entry** (the "too late" path #2251 flags).
- `_MEMORY_THRESHOLD` (0.70–0.85 by RAM).
- `get_memory_used_percentage()` aggregates across `mx.distributed.Group`.
- No `EXO_PREFILL_MEMORY_THRESHOLD`.

So the local fork has the **exact bug #2251 describes** — eviction runs after the dangerous allocation.

---

## Integration seam

- **#2251:** add `EXO_PREFILL_MEMORY_THRESHOLD` (default = `EXO_MEMORY_THRESHOLD` − 0.10); evict LRU **before** `get_kv_cache` + prefill, not only on `add_kv_cache`. Port the PR's regression tests.
- **#2240/#2241:** memory-safety validation in placement — refuse a context-window that won't leave activation headroom. Dashboard control for max context length.
- **#1709:** placement's memory-feasibility check must reserve activation headroom (weights + KV + activations + margin ≤ memory).

---

## Phased plan

### Phase 1 — Port #2251 (prefill headroom)
- Add `EXO_PREFILL_MEMORY_THRESHOLD`; evict before prefill.
- Port the regression tests.
- **Tests:** prefix-cache starves memory → prefill still succeeds (LRU evicts first); distributed max-pressure preserved.

### Phase 2 — Placement activation headroom (#1709, #2240)
- Placement feasibility: weights + KV(context) + activations + margin ≤ memory.
- Configurable context-window with validation.
- **Tests:** near-limit placement rejected with clear error; validated context-window fits.

### Phase 3 — Dashboard control (#2241)
- Max Context Length in instance launch Advanced Options.
- **Tests:** dashboard sets context length; validation surfaces in UI.

---

## Risks & open questions

- **Threshold tuning:** `EXO_PREFILL_MEMORY_THRESHOLD` 10 points below `EXO_MEMORY_THRESHOLD` is the PR's default; validate on small-memory nodes (32GB) where headroom is tight.
- **Eviction churn:** evicting before every prefill could thrash the cache. The PR evicts only when above threshold — verify it doesn't over-evict.
- **Distributed asymmetry:** a remote node may be above threshold while local is fine. The PR preserves distributed max-pressure checks — verify cross-node behavior.
- **Activation size estimation:** #2240's validation needs an activation-size model (per-architecture). Start conservative (overestimate).

---

## Definition of done

- [ ] Phase 1: #2251 ported; prefill no longer OOMs under prefix-cache pressure; regression tests green.
- [ ] Phase 2: placement rejects near-limit configs; context-window validated.
- [ ] Phase 3: dashboard max-context-length control.
- [ ] `basedpyright` + `ruff` + `nix fmt` + `pytest` clean.