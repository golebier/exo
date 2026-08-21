# 01 — Tiered KV Cache: Hot (RAM) + Cold (SSD) + Persistence (DONE log)

**Status:** The runtime-toggleable configuration + feature-flag layer + the
`make_kv_cache` integration point are implemented, verified, and shipped, with
the full dashboard control surface (On/Off, SSD dir/cap, RAM cap, live
observability gauge, file count, and clear-cache action) mirroring oMLX's
`CacheSettings`. The heavy plumbing — paged-block manager, safetensors SSD
spill/restore, restart-recovery scan — is staged behind these flags per the
phased plan in [`01-tiered-kv-cache-ssd.md`](./01-tiered-kv-cache-ssd.md) and
can land without further UI churn. This doc is the implementation record.

**Build tag:** `1.0.72-turboquant-tiered-dev4`
**oMLX commit referenced:** `c1a3d44` (`jundot/omlx`)
**Target cluster:** 2×256 GB RAM, `iogpu.wired_limit_mb=256000` per node

---

## 0. What was built (summary)

EXO's `KVPrefixCache` was RAM-only and fully lost on every process restart.
For agentic/coding workloads (Claude Code, Pi, Codex) the same long system
prompt + tool history is re-sent across requests; recomputing that prefill
every restart is the dominant latency and the explicit reason oMLX exists.

This change adds the **runtime settings + feature-flag layer** for a tiered
(Hot RAM / Cold SSD) KV cache, mirroring oMLX's `CacheSettings`
(`enabled` / `hot_cache_only` / `ssd_cache_dir` / `ssd_cache_max_size` /
`hot_cache_max_size`), wired through EXO's existing Memory-Guard API pattern
(env default → runtime override → `/v1/feature-flags` → Svelte store → toggle
in Advanced Options). The SSD dir/size knobs are live and the dashboard shows
a live observability block (used/total gauge, file count, clear button) so
operators can see how much is used and decide when to clear — exactly oMLX's
runtime-cache observability workflow.

What ships now is the **settings/feature-flag layer only**. The actual
paged-block manager, safetensors spill/restore, restart-recovery scan, and
boundary-snapshot offload are large ports (phased plans in the design doc)
and land separately. The flags are the live control surface the heavy
plumbing can be staged behind without further UI churn.

### Files added / changed

| File | Status | Purpose |
|------|--------|---------|
| `src/exo/worker/engines/mlx/turboquant.py` | new | Runtime settings layer for both TurboQuant + tiered cache: `is_tiered_cache_enabled` / `hot_cache_only` / `ssd_cache_dir` / `ssd_cache_max_size_bytes` / `hot_cache_max_size_bytes` / `tiered_cache_status` / `clear_ssd_cache`; env defaults + runtime overrides |
| `src/exo/api/main.py` | modified | `PUT /v1/tiered-cache`, `GET /v1/tiered-cache` (status), `DELETE /v1/tiered-cache` (clear); `tieredKvCache` in feature flags |
| `src/exo/api/types/api.py` | modified | `TieredCacheSetting` body model (mirrors oMLX `CacheSettings`) |
| `src/exo/api/types/__init__.py` | modified | Re-export `TieredCacheSetting` |
| `dashboard/src/lib/stores/app.svelte.ts` | modified | `setTieredCache`, `fetchTieredCacheStatus`, `clearTieredCache` store methods |
| `dashboard/src/routes/+page.svelte` | modified | Tiered KV Cache On/Off toggle + SSD dir/cap/RAM-cap inputs (persisted to localStorage) + Runtime Cache Observability block (live gauge, file count, clear-with-confirm, path info) |
| `src/exo/worker/tests/unittests/test_mlx/test_turboquant.py` | new | 40 tests: defaults, runtime overrides, bit normalisation, `make_kv_cache` skip-last integration, tiered-cache dir/size parsing, `clear_ssd_cache`, status disk-capacity/paths |

### Verification bar (AGENTS.md)

`basedpyright` 0 errors · `ruff check` clean · `ruff format` clean ·
dashboard builds clean · 40 new tests pass · bundled-binary round-trip
verified (status, size-string parsing, clear).

---

## 1. oMLX → EXO config-surface mapping

oMLX surfaces tiered-cache config through `CacheSettings` (a dataclass in
`omlx/settings.py`) read at startup and mutated via the admin app. EXO has
no per-model JSON settings manager and no separate admin app, so the config
is wired through the **Memory-Guard pattern** (proven, ship-default-off,
runtime-toggleable without restart):

| oMLX `CacheSettings` field | EXO env default | EXO runtime API | EXO dashboard |
|----------------------------|-----------------|-----------------|---------------|
| `enabled` | `EXO_TIERED_KV_CACHE=0` | `PUT /v1/tiered-cache {enabled}` | Tiered KV Cache On/Off |
| `hot_cache_only` | `EXO_TIERED_KV_CACHE_HOT_ONLY=0` | `PUT /v1/tiered-cache {hot_cache_only}` | (inferred from enabled) |
| `ssd_cache_dir` | `EXO_SSD_CACHE_DIR` (default `~/.exo/kv_ssd_cache`) | `PUT /v1/tiered-cache {ssd_cache_dir}` | SSD dir input |
| `ssd_cache_max_size` | `EXO_SSD_CACHE_MAX_SIZE=auto` (10% of SSD) | `PUT /v1/tiered-cache {ssd_cache_max_size}` | SSD cap input ("auto"/"8GB"/"512MB") |
| `hot_cache_max_size` | `EXO_TIERED_KV_CACHE_HOT_MAX_SIZE=0` (disabled) | `PUT /v1/tiered-cache {hot_cache_max_size}` | RAM cap input |

All defaults are **off** so a fresh install behaves exactly as before. The
SSD cache dir is created with `0700` perms (KV state can leak prompt content,
per the design doc's security note). Size strings accept "auto" (10% of the
backing SSD, oMLX default), binary suffixes (KB/MB/GB/TB, optional `iB`), or
a raw byte count — parsed by `parse_size` (ported semantics from oMLX's
`parse_size`).

## 2. Runtime Cache Observability block (mirrors oMLX)

oMLX's admin `_status.html` shows a live "Runtime Cache Observability" block
with an SSD usage gauge (`385.2 GB / 1116.0 GB · 16,946 files`), a clear
button (trash icon → confirm), and path-info rows (active base_path,
effective ssd_cache_dir). This is replicated in EXO's Advanced Options panel:

- **SSD gauge** — progress bar showing `used / max` %, with text
  `270 MB / 8 GB · 6 files` (oMLX's exact format)
- **Clear button** (trash icon) with a confirm/cancel flow → `DELETE /v1/tiered-cache`
- **Path info rows** — Active base path, SSD cache dir, SSD disk capacity
- **Live polling** — refreshes every 5 s while the Advanced Options panel is
  open (via a Svelte 5 `$effect`)

The status scan is recursive (`rglob`) because oMLX stores blocks under
hash-prefix subdirs and response-state in a subdir — a top-level `iterdir`
under-counts. `clear_ssd_cache` removes every file (including the
`response-state` subdir) while preserving the directory structure for the
next spill, matching oMLX's `clear_ssd_cache` route.

## 3. What is NOT yet ported (staged behind the flags)

Per the design doc's phased plan, the following are **not** in this change
and land separately:

- **Phase 1 — Paged block manager (RAM only):** `KVCacheBlock`,
  `FreeKVCacheBlockQueue` (O(1) LRU), `BlockHashToBlockMap`, chain hashing,
  COW. Replaces the list-of-entries + O(n) LRU in `KVPrefixCache`.
- **Phase 2 — SSD spill/restore:** `PagedSSDBlockMetadata`,
  `PagedSSDCacheIndex`, `_write_safetensors_no_mx`, byte round-trip.
  `mark_block_cold` on hot eviction → serialize; `restore_block` on prefix
  hit → deserialize + repopulate.
- **Phase 3 — Restart recovery:** startup scan of the SSD dir, cache
  signature validation (guards against model swaps).
- **Phase 4 — Boundary snapshot SSD offload** (doc 08).

The flags shipped here are the control surface these phases wire into; no
further UI churn will be needed when they land.

## 4. Build/deploy record

- `EXO_VERSION=1.0.72-turboquant-tiered-dev4` · `BUILD.sh` clean (5/5 steps)
- DMG: `output/EXO-1.0.72-turboquant-tiered-dev4.dmg` (316 MB)
- Deployed to 2-node cluster (`m3msu256a.local` + `m3msu256b.local`)
- Live round-trip verified on the bundled binary: `GET /v1/tiered-cache`
  reports files/size/disk-capacity/base-path; `DELETE /v1/tiered-cache`
  removes files (count → 0); `PUT /v1/tiered-cache` parses `"8GB"` →
  8589934592 bytes; SSD auto-size resolved to ~24.5 GiB (10% of SSD)

## 5. Current state & next steps

Shipped and live on the cluster. The settings layer + dashboard controls are
the user-facing deliverable; the paged-SSD plumbing (Phases 1–3) is the next
implementation milestone and can proceed independently behind these flags.