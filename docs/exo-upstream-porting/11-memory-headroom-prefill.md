# 11 — Memory Headroom Before Prefill + Near-Limit Placement

**Tier:** ⭐ (Tier 3)
**Effort:** Low–Medium
**Impact:** Medium–High (OOM avoidance; reliability; near-limit placement)
**Upstream evidence:**
- PR #2251 `fix: reserve memory headroom before prefill`
- #1709 `[BUG] can't place models close to memory limits`
- PR #2240 `Add configurable context-window with memory-safety validation`
- PR #2241 `Add Max Context Length control to instance launch Advanced Options`

**oMLX evidence (combined):**
- `omlx/cluster/prefill_guard.py` — `RankPrefillGuard.check_collective`: every rank
  measures its slice, exchanges a one-hot rejection vote, and raises before the
  collective starts. Admission is one rank-agreed decision.
- `omlx/cluster/memory_guard.py` — `ceiling_breakdown`: static (RAM − tier reserve),
  dynamic (reclaimable now), metal_cap (GPU cap); `hard_limit = min`. The "ceiling"
  oMLX admits against.
- `omlx/memory_monitor.py::raise_if_prefill_exceeds` — the shared front-door guard:
  `current + estimate_prefill_peak_bytes(...) ≤ hard_limit` else reject (HTTP 400).
- `omlx/memory_monitor.py::estimate_prefill_peak_bytes` — peak = new KV bytes
  (resident, exact per-layer shapes, window-capped for rotating layers) + SDPA
  attention activation peak for the last chunk (`eff_chunk = min(step, new_tokens)`,
  `full_kv_len = new + cached`).
- `omlx/scheduler.py::_preflight_memory_check` / `_raise_prefill_eviction_if_available`
  — when a preflight would fail, **first** evict idle LRU models / reclaim pooled
  MLX buffers (`engine_pool._evict_idle_lru_for_prefill`), then re-admit. Only if
  eviction can't help does it reject.
- `omlx/scheduler.py::_guard_prefill_chunk` — per-chunk guard: clamp/abort each
  prefill chunk so its predicted peak can't reach the Metal cap (the uncatchable
  async OOM). Uses an EWMA of measured per-chunk transients
  (`PrefillTransientTracker`) + a static estimator fallback.
- `omlx/prefill_transient_tracker.py` — EWMA of bytes-per-prefill-token, updated
  post-chunk from `phys_footprint` deltas; outlier rejection so one noisy sample
  can't poison admission.

---

## What it is

Two related memory-safety gaps:

### A. Prefill OOM from prefix-cache starvation (#2251)
> Persistent prefix-cache allocations can leave too little room for temporary
> prefill activations. Existing eviction only runs while adding a completed
> cache entry, **after** the dangerous allocation has already happened.

EXO's `KVPrefixCache._evict_if_needed()` is called from `add_kv_cache` only —
i.e. after prefill already allocated its transient activations and the new KV.
That is the exact "too late" path #2251 flags.

### B. Placement near memory limits (#1709)
EXO places models up to the memory ceiling (`ram_available >= storage_size`,
weights only), leaving no room for activations → models fail to load or crash
after load. #2240/#2241 add configurable context-window with memory-safety
validation.

---

## Why it matters

- **Reliability:** OOM crashes during prefill are a top crash category
  (interacts with #2208 hangs, #2067 load crashes).
- **Small, safe, high-leverage:** #2251 is a tight fix. oMLX adds the
  preflight *estimate* that turns "evict and hope" into "evict, measure, and
  refuse if it still won't fit" — the difference between avoiding an OOM and
  rejecting a request cleanly.
- **Composes with TurboQuant (doc 07):** KV quantization reduces the
  persistent-cache pressure that causes #2251's starvation.
- **Composes with placement (doc 05):** the memory-feasibility constraint in
  placement must include activation headroom, not just weights + KV.
- **Composes with the oMLX tiered cache (docs/omlx-porting/01):** once a
  paged/SSD cache lands, eviction-before-prefill + a preflight estimate are
  the admission gate that keeps the hot tier from starving prefill.

---

## Upstream PR / oMLX landscape

| Source | Scope |
|--------|-------|
| #2251 | Prefill headroom (`EXO_PREFILL_MEMORY_THRESHOLD`, evict before prefill) — **the fix, ready** |
| #1709 | Can't place models close to memory limits (the placement side) |
| #2240 | Configurable context-window with memory-safety validation |
| #2241 | Max Context Length control in instance launch Advanced Options (dashboard) |
| oMLX `prefill_guard` | Collective preflight admission (rank-agreed reject before model execution) |
| oMLX `memory_monitor` | `estimate_prefill_peak_bytes` (KV + SDPA) + `raise_if_prefill_exceeds` |
| oMLX `scheduler` | Evict-idle-LRU-before-prefill + per-chunk guard + EWMA transient tracker |

#2251 fixes the runtime eviction ordering; oMLX adds the preflight estimate and
the evict-then-re-admit loop; #2240/#2241 fix the placement/config side; #1709
is the underlying complaint.

---

## EXO current state (local fork)

`src/exo/worker/engines/mlx/cache.py`:
- `KVPrefixCache._evict_if_needed()` — evicts LRU **while adding a completed
  cache entry** (the "too late" path #2251 flags). Called only from
  `add_kv_cache`.
- `_MEMORY_THRESHOLD` (0.70–0.85 by RAM), env `EXO_MEMORY_THRESHOLD`.
- `get_memory_used_percentage()` aggregates across `mx.distributed.Group`
  (max-pressure) — **keep this**; it's EXO's equivalent of oMLX's collective
  ceiling.
- No `EXO_PREFILL_MEMORY_THRESHOLD`. No prefill-peak estimation. No preflight
  admission. No per-chunk guard.
- After eviction: `gc.collect()` + `mx.clear_cache()` — **keep** (oMLX's
  `_reclaim_pooled_buffers_for_prefill` analogue).

`src/exo/worker/engines/mlx/generator/generate.py` / `batch_generate.py`:
- Prefill is a single `stream_generate(..., prefill_step_size=4096)` call (or
  `pipeline_parallel_prefill`). Chunks are not interceptable from EXO's side
  without a monkey-patch, so a per-chunk guard is **Phase 2**.
- `get_kv_cache` is called immediately before prefill; `add_kv_cache` /
  `update_kv_cache` immediately after. The preflight + evict-before-prefill
  seam is **between** those two.

`src/exo/master/placement_utils.py`:
- `filter_cycles_by_memory`: `total_mem >= required_memory` (weights only).
- `_allocate_and_validate_layers`: `required_memory = storage_size *
  node_layers / total_layers`; compares to `ram_available`. **No activation
  headroom, no KV(context) term.**

---

## Integration seam

### Phase 1 — Prefill headroom + preflight admission (the #2251 fix + oMLX estimate)

- Add `EXO_PREFILL_MEMORY_THRESHOLD` (default = `EXO_MEMORY_THRESHOLD − 0.10`,
  clamped to ≥ 0). This is the watermark prefill must leave headroom under.
- **Evict before prefill:** add `KVPrefixCache.evict_for_prefill_headroom()`
  that evicts LRU until `get_memory_used_percentage() ≤
  EXO_PREFILL_MEMORY_THRESHOLD` (mirrors oMLX's "free headroom before
  prefill"). Call it from the generators **before** `get_kv_cache` + prefill,
  not only from `add_kv_cache`.
- **Preflight admission:** add a prefill-peak estimator
  (`estimate_prefill_peak_bytes`) ported from oMLX: peak = new KV bytes/token ×
  `new_tokens` + SDPA activation for the last chunk
  (`eff_chunk = min(prefill_step_size, new_tokens)`, `full_kv_len = new +
  cached`). Compare `current + peak ≤ prefill_ceiling_bytes`; if not, evict
  once more and re-check; if still over, raise `PrefillMemoryExceededError`
  (HTTP 400, maps to a clear API error instead of an OOM crash).
- Preserve distributed max-pressure: `get_memory_used_percentage()` already
  takes the cluster max; the preflight uses the same aggregated pressure so a
  remote node above threshold is respected (oMLX's `check_collective` intent).
- Reclaim MLX allocations after eviction (already `gc.collect()` +
  `mx.clear_cache()`).

### Phase 2 — Per-chunk prefill guard (oMLX `_guard_prefill_chunk`)

- Intercept prefill chunks to clamp each chunk's predicted peak under the
  Metal cap. Requires either a monkey-patch on `mlx_lm.generate` (oMLX's
  approach) or EXO's own chunked-prefill loop. **Deferred** — EXO's prefill is
  a single `stream_generate` call today.
- Add a `PrefillTransientTracker` EWMA once chunks are observable.

### Phase 3 — Placement activation headroom (#1709, #2240)

- Placement feasibility: `weights + KV(context) + activations + margin ≤
  memory`. Use the same `estimate_prefill_peak_bytes` for the activation term
  and a KV(context) estimate from the model card's `num_key_value_heads`,
  `head_dim`, `n_layers`, `context_length`.
- Configurable context-window with validation; dashboard max-context-length
  control (#2241).

---

## Phased plan

### Phase 1 — Port #2251 + oMLX preflight (prefill headroom)
- Add `EXO_PREFILL_MEMORY_THRESHOLD`; `evict_for_prefill_headroom()` before
  prefill.
- Port `estimate_prefill_peak_bytes` + `raise_if_prefill_exceeds`
  (single-process; EXO has no cluster admission layer like oMLX's
  `RankPrefillGuard`, but the distributed max-pressure in
  `get_memory_used_percentage` stands in).
- Add `PrefillMemoryExceededError` → API 400.
- Wire into `generate.py` and `batch_generate.py` between `get_kv_cache` and
  prefill.
- Port the PR's regression tests + oMLX-style preflight tests.
- **Tests:** prefix-cache starves memory → prefill still succeeds (LRU evicts
  first); preflight rejects an impossible prompt with a clear error (no OOM);
  distributed max-pressure preserved; eviction reclaims MLX allocations.

### Phase 2 — Per-chunk guard (done)
- Hybrid approach: keep `stream_generate` for prefill (no correctness risk),
  measure per-chunk transient via the progress callback, feed an EWMA
  (`PrefillTransientTracker`), and abort *before* the next chunk when the
  EWMA-predicted peak would cross the abort cap. EWMA refines admission
  upward via `_admission_transient_bound` (first prefill deferred to static
  estimate, later prefills refined by measurement).

### Phase 3 — Placement activation headroom (#1709, #2240, #2241) (done)
- Placement feasibility: `weights + KV(effective_context) + activation margin`
  per node (`src/exo/master/placement_memory.py`).
- `EXO_PLACEMENT_CONTEXT_TOKENS` overrides the model's `context_length` so a
  1M-context model can be placed on a 16 GB node by reserving less KV — the
  #1709 near-limit lever.
- `num_attention_heads` added to the model card for precise `head_dim`; falls
  back to the conservative `hidden_size / num_key_value_heads` bound when
  absent (over-estimate for GQA/MQA/MLA — safe).
- Activation margin defaults to 0 (runtime guards are the precise gate);
  `EXO_PLACEMENT_ACTIVATION_MARGIN` enables a placement-time buffer.
- Dashboard max-context-length control (#2241) deferred — backend lever in
  place.

---

## Risks & open questions

- **Threshold tuning:** `EXO_PREFILL_MEMORY_THRESHOLD` 10 points below
  `EXO_MEMORY_THRESHOLD` is the PR's default; validate on small-memory nodes
  (32GB) where headroom is tight.
- **Eviction churn:** evicting before every prefill could thrash the cache.
  The guard evicts only when above the prefill threshold — verify it doesn't
  over-evict. oMLX caps retries (`_MAX_PREFILL_EVICTION_RETRIES`); EXO should
  too.
- **Distributed asymmetry:** a remote node may be above threshold while local
  is fine. EXO's `get_memory_used_percentage()` already takes the cluster max
  — the preflight uses it, so a remote above-threshold node triggers eviction
  locally too (conservative; oMLX's `check_collective` votes per-rank).
- **Activation size estimation:** the SDPA term depends on head dims and
  whether MLX fuses. oMLX charges the unfused fp32 score matrix as a safe
  upper bound for unsupported head dims; EXO does the same (over-estimate).
- **Per-chunk guard not in Phase 1:** a single prefill chunk that's fine at
  admission can still OOM mid-prefill if the prompt is far larger than
  `prefill_step_size` and memory climbs. Phase 1's preflight charges the *last
  chunk's* SDPA peak + the *full* new KV, which is the dominant term; the
  per-chunk guard (Phase 2) tightens the mid-prefill case.

---

## Definition of done

- [x] Phase 1: #2251 ported + oMLX preflight estimate; prefill no longer OOMs
      under prefix-cache pressure; impossible prompts rejected with
      `PrefillMemoryExceededError`; regression tests green.
- [x] Phase 2: per-chunk guard clamps mid-prefill peaks via EWMA transient
      tracking (`PrefillTransientTracker` + `guard_prefill_chunk_or_raise`),
      wired through `prefill()`'s progress callback. EWMA-refined admission
      via `_admission_transient_bound`.
- [x] Phase 3: placement reserves `weights + KV(context) + activation margin`
      per node (`placement_memory.py`); `EXO_PLACEMENT_CONTEXT_TOKENS` is the
      near-limit lever (#1709); `num_attention_heads` added to the model card
      for precise `head_dim`. Dashboard max-context-length control (#2241)
      deferred — backend lever is in place.
- [x] `basedpyright` + `ruff` + `ruff format` + `pytest` clean (402 passed;
      3 model-download skips; pre-existing `download/tests` collection error
      unrelated).