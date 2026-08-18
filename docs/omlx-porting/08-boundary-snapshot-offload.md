# 08 — Boundary Snapshot SSD Offload During Prefill

**Tier:** ⭐ (Tier 3)
**Effort:** Medium
**Impact:** Medium (flattens prefill memory spikes for SSM models)
**oMLX source:** `omlx/cache/boundary_snapshot_store.py` (2000 lines)
**EXO target:** `src/exo/worker/engines/mlx/cache.py` (extend `snapshot_ssm_states` path)

---

## What it is

oMLX's `boundary_snapshot_store.py` stores **non-sliceable cache layer
snapshots** (e.g. `ArraysCache` for SSM/GatedDeltaNet layers) **to SSD during
prefill**, freeing GPU memory immediately. At request completion, snapshots are
loaded back one block at a time for final SSD cache storage.

This is **decoupled from the full paged-SSD cache** (doc 01) — it targets a
specific memory-spike problem: during prefill of long prompts, SSM-layer
snapshots accumulate in RAM and can OOM. Offloading them to SSD mid-prefill
flattens the spike.

From the oMLX docstring:

> Stores non-sliceable cache layer snapshots (e.g. ArraysCache) to SSD during
> prefill, freeing GPU memory immediately. At request completion the snapshots
> are loaded back one block at a time for final SSD cache storage.
>
> Uses the same async-write pattern as PagedSSDCacheManager: tensors are
> serialized to raw bytes on the inference thread (Metal-safe), buffered in
> `_pending_writes` for instant read-back, and flushed to disk by a background
> writer thread via `_write_safetensors_no_mx`.

---

## Why it fits EXO

EXO already has the **in-RAM** version of this machinery:

- `cache.py::snapshot_ssm_states` — deep-copies `ArraysCache`/`RotatingKVCache`/
  `CacheList`/`DeepseekV4Cache` entries into `CacheSnapshot(states, token_count)`.
- `cache.py::copy_snapshot_entry` + `copy_rotating_kv_cache` /
  `_copy_arrays_cache` / `_copy_cache_list` / `_copy_v4_cache` — the per-class
  deep-copy logic (the "boundary" snapshots).
- `cache.py::_find_nearest_snapshot` + `KVPrefixCache._get_snapshot` — restore
  the nearest snapshot at/before a target position.
- `is_non_trimmable_cache_entry` / `has_non_kv_caches` — the eligibility
  classification (exactly what oMLX's `type_handlers.py` generalizes).

EXO keeps these snapshots **in RAM** (`KVPrefixCache._snapshots`). For long
prefills with many SSM layers, that's a memory spike. Porting oMLX's SSD
offload pattern turns those snapshots into an SSD-backed store with async
write-back and instant `_pending_writes` read-back — flattening the spike.

This is the **narrowest, most self-contained** of the cache improvements and can
land **before** the full paged-SSD tier (doc 01).

---

## oMLX design

### `boundary_snapshot_store.py` key elements
- Imports from `paged_ssd_cache.py`: `HAS_MLX`, `_encode_shape`,
  `_extract_tensor_bytes`, `_has_zero_dim`, `_restore_tensor_from_bytes`,
  `_write_safetensors_no_mx` — reuses the same Metal-safe byte serialization.
- `compact_pooling_cache_snapshot` from `pooling_delta.py` — for pooling caches.
- **Async-write pattern:** tensors serialized to raw bytes on the inference
  thread (Metal-safe — doesn't hold MLX graph refs), buffered in
  `_pending_writes` for instant read-back, flushed to disk by a **background
  writer thread** via `_write_safetensors_no_mx`.
- **Block-at-a-time restore:** at request completion, snapshots load back one
  block at a time (not all at once) for final SSD cache storage — bounded peak
  memory.

### Why "Metal-safe" matters
`mx.array` holds shared_ptr to Metal buffers + graph inputs. Naive
serialization can pin memory or race with the GPU. oMLX's
`_extract_tensor_bytes` + `_write_safetensors_no_mx` extract raw bytes without
holding MLX graph refs, then a background thread writes them. EXO's
`_detached_copy` (in `cache.py`) already does the analogous np-array detach to
break shared_ptr — the same principle applies to SSD serialization.

---

## EXO current state

- `snapshot_ssm_states` returns a `CacheSnapshot(states=[...], token_count=N)`
  where each state is a deep-copied cache object **in RAM**.
- `KVPrefixCache._snapshots: list[list[CacheSnapshot] | None]` — all in RAM.
- `trim_cache` restores from a snapshot via `copy_snapshot_entry`.
- EXO's `_detached_copy` already detaches Metal shared_ptr (np round-trip).

---

## Integration seam in EXO

- **New class:** `BoundarySnapshotStore` in
  `src/exo/worker/engines/mlx/cache.py` (or a new
  `src/exo/worker/engines/mlx/snapshot_store.py`).
- **Storage:** SSD dir under XDG cache (EXO convention; see
  `src/exo/shared/tests/test_xdg_paths.py`). Default
  `EXO_SNAPSHOT_SSD_DIR` → `$XDG_CACHE_HOME/exo/snapshots`.
- **Replace** the RAM `_snapshots` list with a store-backed list. Snapshot
  creation (`snapshot_ssm_states`) writes to SSD + `_pending_writes`; snapshot
  restore (`_find_nearest_snapshot` → `copy_snapshot_entry`) reads from
  `_pending_writes` first (instant), else disk.
- **Background writer thread:** flush `_pending_writes` to disk. Port oMLX's
  thread + queue pattern.
- **Bounded restore:** load snapshots block-at-a-time at completion, not all at
  once.
- **Lifecycle:** clear store entries when the corresponding `KVPrefixCache`
  entry is evicted (LRU) — don't leak SSD.

---

## Phased plan

### Phase 1 — SSD-backed snapshot store (RAM read-back cache)
- Port the async-write + `_pending_writes` read-back pattern.
- `BoundarySnapshotStore.save(snapshot)` → bytes + async disk write;
  `BoundarySnapshotStore.load(token_count)` → read-back from pending or disk.
- Wire into `KVPrefixCache` so `snapshot_ssm_states` results spill to SSD.
- **Tests:** save/load round-trip byte-exactness; `_pending_writes` hit before
  disk flush; disk-flushed load; concurrency (save while loading).

### Phase 2 — Block-at-a-time restore + eviction hygiene
- Port the block-at-a-time completion restore.
- Clear store entries on LRU eviction; add an SSD size cap +
  `evict_until_size` (port from `PagedSSDCacheIndex`).
- **Tests:** eviction frees SSD; size cap enforced; restore memory peak
  bounded (measure peak RSS during a long-prefill restore).

### Phase 3 — Compose with paged-SSD cache (doc 01)
- If doc 01 lands, the boundary snapshot store feeds into the final SSD cache
  storage at request completion (oMLX's design). Integrate.

---

## Risks & open questions

- **Correctness of restore:** a snapshot restored from SSD must be byte-identical
  to the RAM version. EXO's `copy_rotating_kv_cache` reconstructs metadata
  (`offset`, `_idx`, `keep`, `max_size`) — the SSD path must round-trip all of
  that. Add a serialization schema (oMLX's `_store_nstate_elements_flat` /
  `_load_nstate_flat` handle non-trivial metadata).
- **`DeepseekV4Cache` branches:** `_copy_compressor_branch` copies
  `buffer_kv`, `buffer_gate`, `prev_kv`, `prev_gate`, `pool`, lengths, counts.
  SSD serialization must cover all fields. This is the hardest cache class.
- **Async thread safety:** the inference thread (MLX executor) and the writer
  thread must not race on `mx.array`. oMLX's pattern (serialize to bytes on
  inference thread, write bytes on writer thread) is safe; preserve it.
- **Disk latency on read-back:** if `_pending_writes` is flushed and a restore
  needs disk, latency spikes. Mitigate by keeping recently-written snapshots in
  `_pending_writes` longer (small RAM buffer).
- **Independent of doc 01?** Yes — this can land first. It only needs the
  byte-serialization helpers from `paged_ssd_cache.py` (`_extract_tensor_bytes`,
  `_write_safetensors_no_mx`, etc.), which are self-contained.

---

## Definition of done

- [ ] Phase 1: snapshot save/load round-trip byte-exact; `_pending_writes`
      fast-path works.
- [ ] Phase 2: peak RSS during long-prefill restore is bounded (measure &
      record); SSD size cap enforced; eviction frees SSD.
- [ ] No regression on `test_kv_prefix_cache.py` (snapshot restore still
      correct).
- [ ] `basedpyright` + `ruff` + `nix fmt` + `pytest` clean.