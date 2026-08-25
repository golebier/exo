# 01 — Tiered KV Cache: Hot (RAM) + Cold (SSD) with Persistence

**Tier:** ⭐⭐⭐ (Tier 1)
**Effort:** High (~2–4 weeks)
**Impact:** Very high
**oMLX source:** `omlx/cache/{paged_cache.py, paged_ssd_cache.py, prefix_cache.py, boundary_snapshot_store.py, type_handlers.py, type_registry.py, hybrid_cache.py, factory.py, stats.py, interface.py, recovery.py, observability.py}`
**EXO target:** `src/exo/worker/engines/mlx/cache.py` (extend `KVPrefixCache`)

> **Status (partial — shipped):** Phases 2–3 (SSD spill/restore + restart
> recovery) and the **prefix-SSD restore** refinement are implemented in
> `src/exo/worker/engines/mlx/ssd_cache.py` + `cache.py`, behind
> `EXO_TIERED_KV_CACHE=1` (default-off). The SSD store now restores the
> **longest common prefix** across SSD entries (not just exact match), so the
> dominant agentic re-send-context workload skips re-prefilling the shared
> prefix after a restart/LRU eviction. Phase 1's paged-block RAM manager
> (O(1) LRU + chain-hash) remains future work — the persistence value
> doesn't require it, and the existing list-of-entries hot path is untouched.

---

## Why this is #1

EXO's `KVPrefixCache` is **RAM-only**, evicted by LRU when memory crosses a
threshold (`_MEMORY_THRESHOLD`), and **fully lost on every process restart**.

For agentic/coding workloads (Claude Code, Pi, Codex) the same long system prompt
+ tool history is re-sent across requests. Recomputing that prefill every restart
— or every time LRU evicts it — is the dominant latency and the explicit reason
oMLX exists ("makes local LLMs practical for real coding work").

oMLX's cache stack splits KV state across two tiers:

- **Hot tier (RAM):** frequently accessed blocks, fast access.
- **Cold tier (SSD):** when hot fills, blocks spill to disk in safetensors
  format. On the next request with a matching prefix, they're **restored from
  disk instead of recomputed** — even after a server restart.

This is the single highest-value architectural addition for EXO.

---

## oMLX design

### Architecture (from oMLX README)

```
Cache Stack
├── PagedCacheManager      (GPU/RAM, block-based, CoW, prefix sharing)
├── Hot Cache              (in-memory tier, write-back)
└── PagedSSDCacheManager   (SSD cold tier, safetensors format)
```

### Key files (line counts at time of analysis)

| File | Lines | Role |
|------|-------|------|
| `paged_cache.py` | 1793 | vLLM-v1-style block pool: `KVCacheBlock`, `FreeKVCacheBlockQueue` (O(1) LRU), `BlockHashToBlockMap`, `PagedCacheManager` with COW + refcounting + chain-hash prefix caching |
| `paged_ssd_cache.py` | 5062 | SSD block store: safetensors serialization, hash-based subdir layout, LRU size mgmt, startup scan to reuse existing files |
| `prefix_cache.py` | 4798 | Prefix-cache orchestration layer |
| `boundary_snapshot_store.py` | 2000 | Offload non-sliceable layers (ArraysCache/SSM) to SSD *during* prefill (see doc 08) |
| `type_handlers.py` | 1451 | Per-cache-class block-slice eligibility (KVCache vs ArraysCache vs CacheList vs DeepseekV4Cache vs …) |
| `type_registry.py` | 269 | Registry mapping cache class → handler |
| `hybrid_cache.py` | 338 | Config for mixed cache types per layer (e.g. Qwen3-Next) |
| `factory.py` | 247 | Cache construction |
| `stats.py` | 267 | Hit/miss/eviction metrics |
| `recovery.py` | 128 | Restart recovery |
| `interface.py` | 118 | `CacheManager` ABC |
| `observability.py` | 199 | Metrics export |

### Core concepts

1. **Block-based allocation** — configurable tokens-per-block. A prefix is a
   chain of blocks.
2. **Chain hashing** — a block's hash depends on its parent's hash + its token
   content, so prefix matches are O(1) hash lookups (no token-by-token compare
   at lookup time; EXO currently does `get_prefix_length` with `mx.cumprod`).
3. **Copy-on-Write (COW)** — shared blocks are refcounted; a write to a shared
   block forks it instead of mutating.
4. **LRU eviction (O(1))** — doubly-linked-list free queue, not EXO's
   `self._last_used.index(min(...))` (which is O(n)).
5. **SSD spill** — when a block is evicted from hot, it's serialized to
   `~/.omlx/cache/<hash-prefix>/<block-hash>.safetensors` and indexed. On a
   future prefix hit, restore bytes → `mx.array` → repopulate cache.
6. **Restart recovery** — on startup, scan the SSD cache dir, rebuild the
   `PagedSSDCacheIndex`, and reuse existing files.
7. **Type handlers** — not every cache class is block-sliceable. `ArraysCache`
   (SSM) and `DeepseekV4Cache` branches need special handling; oMLX's
   `type_handlers.py` is the reference.

### Key oMLX API surfaces

`PagedCacheManager` (`paged_cache.py:484`):
- `allocate_block()`, `get_new_blocks(n)`, `free_block(id)`, `free_blocks(ids)`
- `acquire_cached_block(hash)`, `get_cached_block(hash)`, `cache_full_blocks(...)`
- `find_shared_prefix(...)`, `fork_block_table(...)`, `_cow_copy_block(...)`
- `evict_lru_blocks(n)`, `handle_memory_pressure(requested)`, `mark_block_cold(id)`
- `restore_block(...)` — brings a block back from SSD
- `set_paged_ssd_cache_manager(ssd_mgr)` — wires the cold tier

`PagedSSDCacheManager` (`paged_ssd_cache.py`):
- `PagedSSDBlockMetadata` (dataclass: block_hash, size, last_access, …)
- `PagedSSDCacheIndex` (in-RAM index: `add`, `get`, `remove`, `touch`, `evict_until_size`)
- `_write_safetensors_no_mx(...)` — serialize without holding MLX graph refs
- `_extract_tensor_bytes(arr)` / `_restore_tensor_from_bytes(...)` — Metal-safe byte round-trip
- Startup scan to reuse existing cache files

---

## EXO current state

`src/exo/worker/engines/mlx/cache.py`:

- `KVPrefixCache` — list-of-entries, RAM-only.
  - `add_kv_cache(...)`, `update_kv_cache(...)`, `get_kv_cache(model, prompt, media_regions)`
  - Prefix match via `get_prefix_length` (token-by-token `mx.cumprod`).
  - LRU via `_last_used` list + `_evict_if_needed()` → `self._last_used.index(min(...))` (O(n)).
  - `_validate_media_match` — truncates match into a media region whose content hash differs. **Keep this**; oMLX has nothing equivalent.
- `snapshot_ssm_states` / `copy_snapshot_entry` — deep-copy non-trimmable layers (ArraysCache, RotatingKVCache, CacheList, DeepseekV4Cache). **This is exactly the machinery the boundary snapshot store (doc 08) extends to SSD.**
- `make_kv_cache(model, max_kv_size, keep)` — constructs `KVCache` / `QuantizedKVCache` / `RotatingKVCache` lists, or defers to `model.make_cache()`.
- Memory threshold: `_default_memory_threshold()` scales with total RAM (0.70–0.85).

**Strengths to preserve:**
- Media-region-aware prefix matching (oMLX lacks this).
- `DeepseekV4Cache` branch copy (`_copy_compressor_branch`) — oMLX handles V4 via type handlers; EXO's explicit copy is fine.
- Distributed memory pressure via `mx.distributed.all_gather` in `get_memory_used_percentage()` — keep for cluster mode.

---

## Integration seam in EXO

`KVPrefixCache` is consumed in two places:
- `src/exo/worker/engines/mlx/generator/batch_generate.py` (continuous batching path)
- `src/exo/worker/engines/mlx/generator/generate.py` (single-stream path)

Both call `get_kv_cache(model, prompt_tokens, media_regions)` and
`add_kv_cache(...)`. **The public method signatures should stay identical** so
the generators don't change; only the internals get a paged + SSD backing.

---

## Phased plan

### Phase 1 — Paged block manager (RAM only), parity with today
**Goal:** Replace the list-of-entries + O(n) LRU with a block pool, no SSD yet.
Behavior identical to today but O(1) eviction and chain-hash prefix lookup.

- Port `KVCacheBlock`, `FreeKVCacheBlockQueue`, `BlockHashToBlockMap`,
  `BlockTable` from `paged_cache.py` (≈400 lines, self-contained).
- Port `compute_block_hash` + chain hashing.
- Port `type_handlers.py` block-slice eligibility for EXO's cache classes
  (`KVCache`, `QuantizedKVCache`, `RotatingKVCache`, `ArraysCache`, `CacheList`,
  `DeepseekV4Cache`). EXO already knows which are non-trimmable
  (`is_non_trimmable_cache_entry`); generalize into a registry.
- Reimplement `KVPrefixCache.get_kv_cache` / `add_kv_cache` on top of
  `PagedCacheManager`, preserving the existing return contract.
- Preserve `_validate_media_match` by folding content-hash into the block
  extra-key (oMLX's `resolve_block_extra_keys` already supports `extra_key_ranges`
  for segmented VLM keying — use it).
- **Tests:** extend `src/exo/worker/tests/unittests/test_mlx/test_kv_prefix_cache.py`
  with prefix-hit, COW-fork, and LRU-eviction cases; add a block-hash determinism test.

### Phase 2 — SSD cold tier (spill + restore, no restart recovery yet)
**Goal:** Evicted hot blocks spill to SSD; a later prefix hit can restore them.

- Port `PagedSSDBlockMetadata`, `PagedSSDCacheIndex`, `_write_safetensors_no_mx`,
  `_extract_tensor_bytes`, `_restore_tensor_from_bytes` from `paged_ssd_cache.py`.
- Add `set_paged_ssd_cache_manager(...)` wiring + `restore_block(...)` in
  `PagedCacheManager`.
- Add an EXO config: `EXO_SSD_CACHE_DIR` (default under XDG cache dir — EXO
  already has `src/exo/shared/tests/test_xdg_paths.py`, so follow that convention).
- `mark_block_cold(id)` on hot eviction → serialize to SSD; `restore_block` on
  prefix hit → deserialize + repopulate.
- **Tests:** spill-then-restore round-trip; byte-exactness vs in-RAM path;
  concurrent spill/restore under batched load.

### Phase 3 — Restart recovery
**Goal:** On exo restart, scan the SSD dir and reuse cached blocks.

- Port the startup scan from `paged_ssd_cache.py` + `recovery.py`.
- Validate restored blocks against the *current* model load (cache signature —
  oMLX's `cache_signature_for` / `_cache_compat_signature` guards against model
  swaps; port this).
- **Tests:** kill -9 mid-request → restart → prefix hit serves from SSD without
  recompute; model swap invalidates stale SSD blocks.

### Phase 4 — Boundary snapshot SSD offload (decoupled; see doc 08)
Offload ArraysCache/SSM snapshots to SSD *during* prefill to flatten memory
spikes. This is doc [08](08-boundary-snapshot-offload.md) and can land after
Phase 3 independently.

---

## Risks & open questions

- **Cache signature / invalidation:** oMLX computes a signature from cache
  types, turboquant bits, cachelist subtypes, etc. EXO must define its own
  (simpler) signature covering `KV_CACHE_BITS`, `CACHE_GROUP_SIZE`,
  `max_kv_size`, model id, and quant config. Mismatched signatures must not
  restore.
- **Distributed mode:** EXO's `get_memory_used_percentage()` already aggregates
  across `mx.distributed.Group`. Decide whether SSD cache is per-node (yes) or
  shared (no — keep per-node; the cluster already has disaggregated prefill for
  cross-node sharing).
- **Disk pressure:** add an SSD cache size cap (`EXO_SSD_CACHE_MAX_SIZE`,
  oMLX's `--memory-guard-gb` analogue) and LRU eviction on the SSD index
  (`PagedSSDCacheIndex.evict_until_size`).
- **Corruption:** safetensors is self-describing; still add a checksum field in
  `PagedSSDBlockMetadata` and refuse to restore on mismatch.
- **Security:** SSD cache dir must be created with restrictive perms (0700) so
  KV state (which can leak prompt content) isn't world-readable.

---

## Definition of done

- [ ] Phase 1: paged RAM manager passes existing `test_kv_prefix_cache.py` + new
      block/COW/LRU tests; `basedpyright` + `ruff` clean. *(Not yet done — the
      persistence value ships without it; see status note above.)*
- [x] Phase 2: spill + restore works; round-trip tests green
      (`test_ssd_cache.py::TestSpillRestoreRoundTrip`).
- [x] **Prefix-SSD restore** (Phase 2 refinement): longest-common-prefix
      restore across SSD entries — `test_ssd_cache.py::TestPrefixRestore` +
      `TestPrefixRestoreIntegration` green.
- [x] Phase 3: restart-recovery test green
      (`test_ssd_cache.py::TestRestartRecovery` +
      `test_prefix_restore_works_after_restart_recovery`).
- [ ] Benchmark: TTFT on a 8k-token repeat prompt after restart drops from
      "full recompute" to "SSD restore" — measure and record tok/s.
- [ ] Dashboard surface (optional): hot/cold block counts + hit rate in the
      EXO dashboard, mirroring oMLX's cache observability.