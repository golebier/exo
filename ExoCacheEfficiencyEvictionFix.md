# Fix: Cache-Efficiency Stat Was Always 0 (Prefix-Cache Evicted Every Prefill)

Follow-up to `ExoCacheEfficiencyImpl.md` (`v1.0.72-cache-efficiency-dev1`).
The cache-efficiency stat cards shipped, but on the 2-node RDMA cluster
(`M3MSU256a` + `M3MSU256b`, `iogpu.wired_limit_mb=256000`, GLM-5.2-oQ4
Tensor+MlxJaccl) the dashboard reported, for every request:

```
PREFILL 88.4k · CACHED 0 · EFFICIENCY 0.0%
```

`cached_tokens` was always 0. This doc is the root-cause + fix record.

**Build tag:** `1.0.72-cache-efficiency-dev2`
**Artifact:** `output/EXO-1.0.72-cache-efficiency-dev2.dmg`
**SHA256:** `58cb9c7c3552f57b3fd826949ad2181f84560c0b905c79b70efa26dd6576f743`

---

## Diagnosis

The `cached_tokens` value reported in the UI is `prefix_hit_length` from the
MLX generator (`src/exo/worker/engines/mlx/generator/{generate,batch_generate}.py`)
— the number of prompt tokens served from the **KV prefix cache** rather than
recomputed. It flows, fully verified, through:

```
generator (prefix_hit_length)
  → TokenChunk.usage.prompt_tokens_details.cached_tokens
  → Runner.send_chunk → InstanceTokensUpdated(cached_tokens=…)
  → master indexes + apply → State.instance_token_usage[iid].cached_tokens
  → /state (model_dump by_alias → cachedTokens)
  → dashboard InstanceTokenUsage.cachedTokens → Prefill/Cached/Efficiency card
```

Every stage was verified correct in isolation (event→apply→state→JSON
round-trip reproduces `cachedTokens`; DiskEventLog msgpack round-trip
preserves it; pickled mp_channel preserves it; the frontend reads it). So
`CACHED 0` meant `prefix_hit_length` was genuinely **0** — the KV prefix
cache was not producing hits.

### Root cause

The prefix-cache **eviction** in `src/exo/worker/engines/mlx/cache.py`
(`_evict_until_under`) compared `psutil.virtual_memory().percent / 100`
(`get_memory_used_percentage`) against the fixed soft/hard watermarks
`_PREFILL_MEMORY_THRESHOLD` (0.75 on a 256 GB box) / `_MEMORY_THRESHOLD`
(0.85).

On a large Apple-Silicon box whose model weights are wired via
`iogpu.wired_limit_mb`, `psutil.virtual_memory().percent` **counts those
wired pages as "used"** and reports >75 % with *just the model loaded* —
before any request. `evict_for_prefill_headroom()` runs before **every**
`get_kv_cache` lookup, so it evicted the entire prefix cache before each
prefill. With the cache empty at every lookup, `prefix_hit_length` was
always 0 → `cached_tokens` always 0 → `EFFICIENCY 0.0 %`.

The repo already had the correct reclaim-based metric —
`memory_guard.current_usage_bytes()` (= `max(phys_footprint, MLX active)`,
which only counts what *this process* holds, not OS-wide wired/compressed
pages) and `iogpu.wired_limit_mb`-aware byte ceilings — but it was wired
only into the preflight **admission** check (`preflight_or_raise`), not the
**eviction** path. The eviction still used the naive `psutil` percentage.

> Why the unit tests didn't catch it: the existing eviction tests patch
> `get_memory_used_percentage` to simulate pressure; they never exercised a
> resident-model scenario where `psutil.percent` is high but real process
> footprint is low. The bug only manifests on a loaded large-memory host.

---

## The fix

Make eviction compare the process's real footprint against the reclaim-based
byte ceilings, with the legacy percentage only as a fallback when the
reclaim model can't be computed.

### `src/exo/worker/engines/mlx/cache.py`

`_evict_until_under` now consults a new `_memory_pressure_exceeds(threshold)`:

- Resolves `memory_guard.eviction_soft_bytes()` / `eviction_hard_bytes()`.
- Maps the legacy fractional `threshold` onto a byte ceiling by interpolating
  between the soft and hard watermarks: the before-prefill path
  (`threshold = _PREFILL_MEMORY_THRESHOLD`) targets **soft**, the post-add
  path (`threshold = _MEMORY_THRESHOLD`) targets **hard**.
- Compares `memory_guard.current_usage_bytes()` against that ceiling.
- **Distributed (TP) mode:** aggregates usage and ceilings across the
  `mx.distributed.Group` (cluster-max usage vs cluster-min ceiling),
  mirroring the legacy `all_gather`-of-percentages conservative behaviour.
- **Fallback:** when the byte ceilings are unresolvable (0), falls back to
  `get_memory_used_percentage()` so behaviour is unchanged on platforms
  where the reclaim model can't be computed.

A legitimately-resident model no longer looks like memory pressure, so the
prefix cache survives across requests and produces hits → `cached_tokens`
becomes non-zero.

### `src/exo/worker/engines/mlx/memory_guard.py`

Added `eviction_hard_bytes()` / `eviction_soft_bytes()` — the byte ceilings
resolved **independently of the preflight-admission toggle**
(`EXO_ENABLE_PREFILL_GUARD`). Eviction and admission are separate concerns:
an operator may run with the guard off (no preflight rejection) yet still
want the prefix cache evicted against the real OOM ceiling rather than
`psutil`'s wired-page percentage. These complement the existing
`hard_limit_bytes()` / `soft_limit_bytes()` (which return 0 when the guard
is disabled) so eviction is correct whether or not admission rejection is on.

### Why not also fix the SSD "0 B / 0 files"?

Separate, pre-existing gap — not a count bug. The tiered-cache SSD spill
was never implemented; the design doc
(`docs/omlx-porting/01-tiered-kv-cache-ssd-DONE.md`) states "What ships now
is the settings/feature-flag layer only. The actual paged-block manager,
safetensors spill/restore... land separately." `cache.py`'s `_evict_until_under`
drops evicted entries; it never writes to SSD, so the observability gauge is
always 0 regardless of the toggle. Implementing the SSD spill (phases 1–3)
is a multi-week feature out of scope for this fix. The eviction fix above is
what makes the **cached-token count** work.

---

## Verification

- `basedpyright`: 0 errors · `ruff check`: clean · `ruff format`: applied.
- **4 new model-free regression tests** (`TestEvictionPressureModel` in
  `test_kv_prefix_cache.py`):
  - no-evict when real footprint < ceiling (the regression — `psutil`=0.95
    ignored),
  - evict when footprint > soft ceiling,
  - post-add uses the hard ceiling (footprint between soft and hard → no evict),
  - legacy percentage fallback when ceilings unresolvable.
- Updated the existing eviction tests (`test_kv_prefix_cache.py`,
  `test_prefill_memory_headroom.py`) to drive pressure via
  `memory_guard.current_usage_bytes` + the eviction byte ceilings (the
  patched `get_memory_used_percentage` is now the fallback, not the
  consulted metric).
- Full suite: 567 passed, 5 skipped (only pre-existing unrelated
  download-test collection errors and a Rust signature test remain).
- **Bundle verification:** extracted the PyInstaller PYZ from the DMG's
  `exo` executable and disassembled the marshaled bytecode —
  `cache.pyc`'s `_memory_pressure_exceeds` references
  `memory_guard.eviction_{soft,hard}_bytes` + `current_usage_bytes`;
  `memory_guard.pyc` defines both new functions. The fix ships in the binary.
- **Live confirmation:** deployed `1.0.72-cache-efficiency-dev2` to the
  cluster — the cache-efficiency card now reports non-zero cached tokens.

---

## Files changed

| File | Change |
|------|--------|
| `src/exo/worker/engines/mlx/cache.py` | `_evict_until_under` uses new `_memory_pressure_exceeds` (reclaim-based byte ceiling vs `current_usage_bytes`, distributed all-gather, legacy fallback) |
| `src/exo/worker/engines/mlx/memory_guard.py` | `eviction_hard_bytes()` / `eviction_soft_bytes()` — ceilings independent of the admission toggle |
| `src/exo/worker/tests/unittests/test_mlx/test_kv_prefix_cache.py` | `TestEvictionPressureModel` (4 new tests) + existing eviction tests migrated to the byte model |
| `src/exo/worker/tests/unittests/test_mlx/test_prefill_memory_headroom.py` | `TestEvictForPrefillHeadroom` migrated to the byte model |

## Pre-commit gates (per AGENTS.md)

`uv run basedpyright` (0 errors) · `uv run ruff check` (clean) ·
`uv run ruff format` (applied — `nix` unavailable in this env, `ruff format`
is the underlying formatter it invokes) · `uv run pytest` (green).