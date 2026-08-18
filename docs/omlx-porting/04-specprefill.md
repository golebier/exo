# 04 — SpecPrefill: Sparse Prefill via Draft-Model Token Importance

**Tier:** ⭐⭐ (Tier 2)
**Effort:** Medium
**Impact:** Medium (TTFT reduction on long prompts)
**oMLX source:** `omlx/patches/specprefill.py`, `omlx/specprefill/`
**EXO target:** `src/exo/worker/engines/mlx/generator/generate.py`, `batch_generate.py`, new `src/exo/worker/engines/mlx/patches/specprefill.py`

---

## What it is

SpecPrefill (arXiv 2502.02789; also waybarrios/vllm-mlx PR #180) reduces **Time
To First Token (TTFT)** on long prompts by:

1. **Score** — a small draft model scores token importance via attention.
2. **Select** — chunk-based top-K% selection + a mandatory tail window.
3. **Sparse prefill** — the target model prefills only the selected tokens while
   preserving original positional encoding via manual RoPE.
4. **Cleanup** — restore original RoPE after generation.

The key insight: RoPE is *relative* (`Q_m @ K_p^T` depends only on `m - p`), so
selected keys stored contiguously in cache with correct RoPE angles produce
correct attention during decode, even though they were prefilled out of order.

---

## Why it fits EXO

EXO already has a **remote prefill** fast path
(`src/exo/worker/engines/mlx/generator/remote_prefill.py`,
`src/exo/worker/engines/mlx/disaggregated/`) that offloads prefill to another
node. SpecPrefill is a **local** prefill accelerator that *complements* it:

- Long prompt, single node → SpecPrefill (no network round-trip).
- Long prompt, cluster available → remote prefill (already implemented).
- Could even compose: remote node does sparse prefill, ships only the selected
  KV blocks over the wire (fewer bytes than full prefill).

For the agentic use case (Claude Code / Pi re-sending long context), cutting
local TTFT is high-value.

---

## oMLX design

### `omlx/patches/specprefill.py` — pipeline

```python
# 1. score_tokens()  — draft model scores token importance via attention
# 2. select_chunks() — chunk-based top-K% selection + mandatory tail window
# 3. sparse_prefill() — target prefill with manual RoPE at original positions
# 4. cleanup_rope()  — restore original RoPE after generation
```

Design notes (verbatim from the oMLX docstring):
- RoPE is relative: `Q_m @ K_p^T` depends only on `(m - p)`. Selected keys
  stored contiguously in cache with correct RoPE angles produce correct attention
  during decode.
- After sparse prefill of N tokens from M total, `cache.offset = N` but decode
  needs position M. `_OffsetAdjustedRoPE` adds `(M - N)` to each offset.

### `omlx/specprefill/draft.py` — draft scoring
- Uses a small draft model (separate from target).
- Reports `keep_percent = selected_token_count / n_to_score * 100`.

---

## EXO current state

- `generate.py::prefill(...)` does full sequential prefill in
  `prefill_step_size` chunks (default 4096).
- `batch_generate.py` same.
- `remote_prefill.py` fetches KV from a remote node.
- EXO has no draft-model concept and no sparse prefill.
- EXO's `KVPrefixCache` already manipulates `cache.offset` and trims caches
  (`trim_cache`), so the offset bookkeeping SpecPrefill needs is in EXO's
  vocabulary.

---

## Integration seam in EXO

- **Patch site:** new `src/exo/worker/engines/mlx/patches/specprefill.py`,
  applied via `patches/__init__.py` like the GLM patch.
- **Trigger:** in `generate.py::prefill`, before the standard chunk loop, if
  `len(prompt_tokens) > EXO_SPECPREFILL_MIN_TOKENS` and a draft model is
  configured, run the sparse path.
- **Draft model source:** either a user-configured small model, or an
  auto-selected tiny model from the same family. Start with explicit config
  (`EXO_SPECPREFILL_DRAFT=<model_id>`) to avoid surprise downloads.
- **RoPE patching:** `_OffsetAdjustedRoPE` must wrap EXO's model's RoPE. EXO
  already has `patches/standard_yarn_rope.py` — follow that pattern for
  applying/undoing a RoPE patch.
- **Cache:** EXO's `KVPrefixCache.add_kv_cache` stores the prompt tokens; after
  sparse prefill the stored prompt must still be the *full* prompt (for future
  prefix matching) while the cache holds only selected positions. Document this
  invariant clearly.

---

## Phased plan

### Phase 1 — Single-stream sparse prefill (no draft model yet)
**Goal:** Prove the RoPE-offset mechanism with a trivial selector (e.g. uniform
stride + tail window), no draft model.

- Port `_OffsetAdjustedRoPE` and `cleanup_rope`.
- Implement `select_chunks` with a naive selector.
- Wire into `generate.py::prefill` behind `EXO_SPECPREFILL_ENABLED=1`.
- **Tests:** decode-after-sparse-prefill produces correct tokens vs full prefill
  (greedy equality on a fixed prompt); offset bookkeeping correctness.

### Phase 2 — Draft-model scoring
- Port `score_tokens` using a configured draft model.
- Add `EXO_SPECPREFILL_DRAFT` config + draft-model load (reuse EXO's model
  download/loading).
- **Tests:** quality regression test — output similarity vs full prefill on a
  benchmark set; TTFT benchmark.

### Phase 3 — Batched path + remote composition
- Extend to `batch_generate.py` (per-sequence sparse prefill before batching).
- Optional: compose with `remote_prefill` — remote node returns only selected KV
  blocks (protocol change in `src/exo/worker/disaggregated/protocol.py`).
- **Tests:** batched correctness; remote-sparse composition round-trip.

---

## Risks & open questions

- **Quality vs speed trade-off:** sparse prefill is approximate. Must measure
  output quality (not just TTFT) on real workloads. Set a conservative default
  `keep_percent` (e.g. 30–50%) and let users tune.
- **Draft model cost:** loading a second model costs memory. On memory-constrained
  nodes this may negate the TTFT win. Consider a shared draft model across
  requests (ties into doc 05's engine pool).
- **RoPE correctness across model families:** EXO supports many model types;
  `_OffsetAdjustedRoPE` must handle each RoPE variant (standard, YaRN, GLM, …).
  EXO's `standard_yarn_rope.py` patch is a reference for the variants.
- **Prefix cache interaction:** after sparse prefill, a subsequent request with
  the same prompt must not falsely "hit" a cache that only has selected
  positions. The stored prompt tokens are full, but `cache_length` is N, not M.
  Ensure `get_kv_cache`'s `best_length` logic accounts for this (likely: sparse
  entries flagged so they're only reused for exact-match, not partial-prefix).
- **Mandated tail window:** the tail window must always be fully prefilled to
  preserve local context correctness — don't let aggressive selection drop it.

---

## Definition of done

- [ ] Phase 1: sparse prefill with naive selector produces greedy-identical
      output to full prefill on test prompts.
- [ ] Phase 2: draft-model path; TTFT on a 16k-token prompt drops by ≥30% with
      ≤2% quality regression on a benchmark.
- [ ] Prefix cache correctness: no false hits on sparse-prefilled entries.
- [ ] `basedpyright` + `ruff` + `nix fmt` + `pytest` clean.