# Missing-Column Work — DONE log (omlx 01 SSD tier · exo 07 hardening · exo 11 HTTP 400)

**Date:** 2025-08-24
**Scope:** The "What's still missing" column of the
`docs/porting-status-and-synthesis.md` summary table, for the three tasks whose
shipped layer was already in place but whose heavy plumbing / hardening / error
mapping was pending. omlx 02's JACCL race was already fixed (see its DONE log);
its remaining `Fence::wait` decode race is in the mlx fork, outside this repo.

---

## 0. What was done (summary)

Three pieces of the "still missing" column landed, each behind its existing
default-off flag so a fresh install behaves exactly as before:

1. **omlx 01 — Tiered KV cache, Phases 2–3** (the doc's single highest-leverage
   item). A cold SSD tier (`ssd_cache.py`) wired into `KVPrefixCache` so evicted
   RAM entries spill to disk and a future exact-match prefix lookup that misses
   RAM is restored from SSD instead of recomputed — even after a restart. This
   is oMLX's stated raison d'être (skip re-prefilling the re-sent agentic system
   prompt). Built as an adjunct to the existing entry-based cache (not a
   paged-block rewrite), default-off behind `EXO_TIERED_KV_CACHE`.
2. **exo 07 — TurboQuant correctness hardening (#1990, #2261).** Skip KV
   quantization in single-node BatchGenerator mode (batched trim/extend
   desync); force a clean prefill after chained prefix-cache extensions when KV
   quantization is active (stale-quantized-state guard). Both are latent bugs in
   the existing quant path.
3. **exo 11 — HTTP 400 mapping for `PrefillMemoryExceededError`.** A too-large
   prompt rejected by the prefill admission guard now surfaces as a 400
   (client-recoverable) across the OpenAI chat (stream / non-stream / bench) and
   Claude `/v1/messages` (stream / non-stream) paths, instead of a blanket 500
   / truncated 200. The Claude streaming path also gained a proper
   `event: error` (it previously swallowed errors into an empty message).

### Verification bar (AGENTS.md)

`basedpyright` 0 errors · `ruff check` clean · `ruff format` applied (`nix fmt`
equivalent — nix unavailable in the build env) · **151 new/affected tests
pass** (14 SSD-tier + 8 TurboQuant-hardening + 9 error-mapping + existing
cache/turboquant/prefill/placement suites) · full suite 604 passed (the 1
`rust/exo_rs` failure and 8 `download/tests` collection errors are pre-existing
on `main`, unrelated to this change).

---

## 1. omlx 01 — SSD cold tier (Phases 2–3)

### Design decision: adjunct, not paged-block rewrite

The design doc's Phase 1 (paged-block manager, O(1) LRU, chain-hash, COW) is an
*optimization* of the existing entry-based `KVPrefixCache`. The **value** of
the tiered cache — persistence across restart/LRU so re-sent contexts skip
re-prefill — does not require it. So the SSD tier is built as an **adjunct** to
the existing working `KVPrefixCache` (lower risk than a wholesale rewrite of
the hot path): eviction spills, exact-match lookup restores. The paged-block
manager remains a future optimization.

### Serialization: mlx-lm's `save_prompt_cache` / `load_prompt_cache`

Rather than hand-rolling safetensors serialization, the tier uses mlx-lm's own
`save_prompt_cache` / `load_prompt_cache` (safetensors, with per-layer
`state` / `meta_state` / class names). This gives a **byte-exact** round-trip
for the standard cache classes (`KVCache`, `QuantizedKVCache`,
`RotatingKVCache`, `ArraysCache`, `CacheList`) for free.

### SSD-eligibility (oMLX `type_handlers` analogue)

`load_prompt_cache` reconstructs via `globals()[class_name].from_state` in
`mlx_lm.models.cache`, so a layer whose class isn't there (or lacks
`from_state` — e.g. `DeepseekV4Cache`) can't be restored. Entries containing
such a layer are **SSD-ineligible**: they're never spilled and degrade
gracefully to today's RAM-only behaviour. This mirrors oMLX's
block-slice-eligibility concept — not every cache class is SSD-eligible.

### Cache signature (model/quant swap guard)

A signature (model id + per-layer cache class names + quant bits/group-size +
effective KV bits) is stored in the safetensors metadata and re-checked on
restore, so a stale SSD block left by a different model / quant config is
**refused** (and removed) rather than restoring incompatible state — the
design doc's `cache_signature_for` / `_cache_compat_signature` guard.

### Restart recovery (Phase 3)

On construction the store scans the SSD dir's hash-prefix subdirs,
`tree_unflatten`s each file's metadata, and rebuilds the in-RAM index. Files
missing the EXO metadata keys (older/different build) are removed — they
can't be signature-validated. LRU access order is seeded from file mtime.

### Files added / changed

| File | Status | Purpose |
|------|--------|---------|
| `src/exo/worker/engines/mlx/ssd_cache.py` | new | `SSDKVCacheStore`: spill/restore, eligibility, signature guard, restart-recovery scan, LRU size cap, observability |
| `src/exo/worker/engines/mlx/cache.py` | modified | `KVPrefixCache.set_ssd_store` / `set_model_id`; spill on `_evict_until_under`; exact-match SSD restore in `get_kv_cache`; `clear()` wipes SSD too |
| `src/exo/worker/engines/mlx/builder.py` | modified | Wires the store from the resolved tiered-cache settings + model id |
| `src/exo/worker/tests/unittests/test_mlx/test_ssd_cache.py` | new | 14 tests: round-trip byte-exactness, signature refusal (model + quant swap), eligibility, restart recovery, legacy-file removal, LRU cap + touch order, KVPrefixCache evict→spill→restore, model-swap refusal, disabled-tier no-op |

### What is NOT yet ported

- **Phase 1 — paged-block manager** (O(1) LRU, chain-hash, COW). The adjunct
  SSD tier captures the persistence value without it; it remains a future
  optimization.
- **Prefix SSD restore** (restore the longest common prefix, not just exact
  match). oMLX restores the longest prefix held by every rank; EXO's prefix
  match is still served from RAM, and SSD is the restart/eviction-recovery
  path. A prefix-SSD restore is a later refinement.
- **Phase 4 — boundary snapshot SSD offload** (doc 08).

---

## 2. exo 07 — TurboQuant correctness hardening (#1990, #2261)

### #1990 — skip KV quantization in single-node BatchGenerator mode

mlx-lm's `BatchGenerator` does multi-sequence batched trim/extend on a single
node, where `QuantizedKVCache`'s state can desync across the batched
sequences. `make_kv_cache` gained a `force_plain` flag; `ExoBatchGenerator`
passes `force_plain=(self.group is None)` so single-node batched mode builds
plain `KVCache` regardless of TurboQuant / `KV_CACHE_BITS`. Distributed mode
(each rank holds one sequence shard) and hybrid-cache models (`make_cache`)
are unaffected.

### #2261 — force a clean prefill after chained prefix-cache extensions

When KV quantization is active, a partial hit that would *extend* an entry
which was itself produced by an extension (a "chained" entry) reuses quantized
KV state accumulated across two extensions; the quantization boundaries can
desync and corrupt output. `KVPrefixCache` now tracks a per-entry `_chained`
flag (set by `update_kv_cache`, the extension path); `get_kv_cache` refuses to
reuse a chained+quantized entry for a further partial extension and returns a
fresh cache (clean prefill). The guard is gated on quantization — plain caches
are always safe to chain — and only applies to *partial* (non-exact) hits
(exact hits don't extend).

### Files changed

| File | Status | Purpose |
|------|--------|---------|
| `src/exo/worker/engines/mlx/cache.py` | modified | `make_kv_cache(force_plain=)`; `_cache_is_quantized`; `_chained` tracking + `_ensure_chained_parallel`; #2261 guard in `get_kv_cache` |
| `src/exo/worker/engines/mlx/generator/batch_generate.py` | modified | `force_plain=single_node` on the batched cache construction |
| `src/exo/worker/tests/unittests/test_mlx/test_turboquant_hardening.py` | new | 8 tests: force_plain overrides TurboQuant + legacy bits; quantized chained partial hit → clean prefill; exact hit / plain / non-chained still reuse; `update_kv_cache` sets chained flag |

### What is NOT yet ported

- **Phase 3 — Apple Silicon fast path** (tuned quantized KV kernel, oMLX
  `turboquant_attention.py`). The integer `QuantizedKVCache` fallback remains.
- **Phase 4 — PDD cache handoff** for quantized KV in disaggregated prefill.

---

## 3. exo 11 — HTTP 400 mapping for `PrefillMemoryExceededError`

### The error-flow problem

`PrefillMemoryExceededError` is raised in the worker's cache layer, caught by
the runner's `_send_error`, serialized to an `ErrorChunk`, and sent over zenoh
to the API process — so the API can't `isinstance`-check the exception class.
The fix carries an HTTP status + OpenAI-style `type` **on the chunk** across
the boundary.

### Changes

- `ErrorChunk` gained `error_code: int = 500` and `error_type: str =
  "InternalServerError"` (defaults preserve historic behaviour).
- `exceptions.py` gained `http_error_status_for` / `http_error_type_for`
  (`PrefillMemoryExceededError` → 400 / `PrefillMemoryExceeded`); both
  `_send_error` methods (Sequential + BatchGenerator) populate them.
- **OpenAI `/v1/chat/completions`:** streaming error event carries the chunk's
  code/type; the non-stream path now returns the response object directly
  (not a `StreamingResponse`) so an `ErrorChunk` raises `HTTPException(code)`
  *before* headers are committed (a `StreamingResponse` sends status 200 before
  iterating its body, so an error raised inside couldn't change the status —
  the old `raise ValueError` produced a truncated 200). The bench path's
  `HTTPException` uses `chunk.error_code or 500`.
- **Claude `/v1/messages`:** streaming now emits a proper Anthropic
  `event: error` (it previously `break`ed and emitted an empty message,
  hiding the error from Claude Code entirely); non-stream returns the object
  directly with clean `HTTPException(code)`. Added `ClaudeErrorEvent` /
  `ClaudeErrorBody` types and a status→Anthropic-error-type mapper
  (`invalid_request_error` for 4xx).

### Files changed

| File | Status | Purpose |
|------|--------|---------|
| `src/exo/shared/types/chunks.py` | modified | `ErrorChunk.error_code` / `error_type` |
| `src/exo/worker/engines/mlx/exceptions.py` | modified | `http_error_status_for` / `http_error_type_for` |
| `src/exo/worker/runner/llm_inference/batch_generator.py` | modified | both `_send_error` populate code/type |
| `src/exo/api/adapters/chat_completions.py` | modified | streaming error event uses chunk code/type; `build_chat_completion_response` raises `HTTPException(code)` for non-stream |
| `src/exo/api/adapters/claude.py` | modified | `build_claude_response`; streaming `event: error`; status→Anthropic-type mapper |
| `src/exo/api/types/claude_api.py` | modified | `ClaudeErrorEvent` / `ClaudeErrorBody` |
| `src/exo/api/main.py` | modified | non-stream chat + Claude routes return objects directly; bench path `error_code or 500` |
| `src/exo/api/tests/test_chat_completions_stream.py` | modified | streaming 400/500 mapping + non-stream `HTTPException` tests |
| `src/exo/worker/tests/unittests/test_mlx/test_error_mapping.py` | new | exception→code/type mapping, `ErrorChunk` defaults, `_send_error` population |

### What remains for exo 11

- ⚠️ Dashboard max-context-length control (#2241): backend lever
  (`EXO_PLACEMENT_CONTEXT_TOKENS`) is in place; UI wiring is separate.
- ⚠️ MLA-precise KV estimation (currently over-counts MLA; safe but may
  false-reject placement).
- ⚠️ Cluster validation of the reclaim ceiling with
  `EXO_MEMORY_GUARD_TIER=aggressive`.

---

## 4. Test/CI record

- `uv run basedpyright` → 0 errors, 0 warnings, 0 notes
- `uv run ruff check` → clean
- `uv run ruff format` → applied (4 files reformatted; `nix fmt` equivalent —
  nix unavailable in this build env, treefmt-nix runs `ruff-format` per
  `flake.nix`)
- `uv run pytest` (excluding the pre-existing broken `download/tests`
  collection) → 604 passed, 5 skipped, 1 pre-existing failure
  (`rust/exo_rs/tests/test_python.py::test_sleep_on_multiple_items`, a
  `NetworkingHandle.new()` signature mismatch that fails on `main` without
  these changes)