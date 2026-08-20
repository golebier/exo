# 11 — Memory Headroom Before Prefill + Near-Limit Placement (DONE log)

**Status:** Phases 1–3 implemented and verified; two runtime regressions found
during cluster deployment and fixed (an over-aggressive admission ceiling, then
a SIGBUS from a truncated ctypes struct); the ceiling model was rewritten as a
port of oMLX's reclaim-based guard; a runtime on/off toggle and a build-version
badge were added to the dashboard. This doc is the post-mortem / implementation
record. See [`11-memory-headroom-prefill.md`](./11-memory-headroom-prefill.md)
for the original design rationale.

**Build tag:** `1.0.72-memory-headroom-dev4`
**oMLX commit referenced:** `c1a3d44` (`jundot/omlx`)
**Target cluster:** 2×256 GB RAM, `iogpu.wired_limit_mb=256000` per node

---

## 0. What was built (summary)

Combines PR #2251 ("evict before prefill") with oMLX's preflight admission +
per-chunk EWMA transient tracking, and adds placement-time activation/KV
headroom (#1709, #2240, #2241). The admission ceiling was then rewritten as a
port of oMLX's three-component reclaim-based model (`min(static, dynamic,
metal_cap)`) so a model that legitimately fills 80%+ of memory is no longer
rejected on every prefill.

### Files added / changed

| File | Status | Purpose |
|------|--------|---------|
| `src/exo/worker/engines/mlx/exceptions.py` | new | `PrefillMemoryExceededError` |
| `src/exo/worker/engines/mlx/prefill_transient_tracker.py` | new | `PrefillTransientTracker` (EWMA, outlier rejection, observed-max clamp) |
| `src/exo/worker/engines/mlx/memory_guard.py` | new | Reclaim-based ceiling (port of oMLX `ProcessMemoryEnforcer`): `vm_statistics64`, `phys_footprint`, `iogpu.wired_limit_mb`, tiers, runtime toggle |
| `src/exo/worker/engines/mlx/cache.py` | modified | `evict_for_prefill_headroom`, `preflight_or_raise`, `guard_prefill_chunk_or_raise`, `estimate_prefill_peak_bytes`, `raise_if_prefill_exceeds`; delegates ceiling/usage to `memory_guard` |
| `src/exo/worker/engines/mlx/generator/generate.py` | modified | `bind_chunk_guard()` (public), `prefill()` `chunk_guard` param, pipeline cleanup on abort |
| `src/exo/worker/engines/mlx/generator/batch_generate.py` | modified | Passes `bind_chunk_guard(self.kv_prefix_cache, self.model)` |
| `src/exo/master/placement_memory.py` | new | `estimate_kv_bytes`, `estimate_activation_margin_bytes`, `estimate_node_memory_requirement` |
| `src/exo/master/placement_utils.py` | modified | `_allocate_and_validate_layers` uses `estimate_node_memory_requirement`; `EXO_PLACEMENT_CONTEXT_TOKENS` lever |
| `src/exo/shared/models/model_cards.py` | modified | `num_attention_heads` on `ModelCard`/`ConfigData` (with `text_config` deferral) |
| `src/exo/api/main.py` | modified | `GET /v1/version`, `PUT /v1/memory-guard`, `prefillMemoryGuard` in feature flags |
| `src/exo/api/types/api.py` | modified | `MemoryGuardSetting` body model |
| `src/exo/shared/constants.py` | modified | `EXO_APP_VERSION` (Swift-injected build version) |
| `app/EXO/EXO/ExoProcessController.swift` | modified | Injects `EXO_APP_VERSION = CFBundleShortVersionString` into the exo child env |
| `dashboard/src/routes/+page.svelte` | modified | `Memory Guard:` On/Off toggle inside the Advanced Options section |
| `dashboard/src/routes/advanced/+page.svelte` | modified | Memory-guard tab (always accessible; default landing) |
| `dashboard/src/lib/components/MemoryGuardToggle.svelte` | new | Detailed toggle + explanation for the Advanced route |
| `dashboard/src/lib/components/HeaderNav.svelte` | modified | Build-version badge in the top-left; Advanced link always visible |
| `dashboard/src/lib/stores/app.svelte.ts` | modified | `setMemoryGuard`, `fetchAppVersion`, `appVersion` |
| `src/exo/worker/tests/unittests/test_mlx/test_prefill_memory_headroom.py` | new | 42 tests (Phase 1 + Phase 2) |
| `src/exo/worker/tests/unittests/test_mlx/test_memory_guard.py` | new | 12 tests (ceiling math, reclaim ratios, loaded-model regression, toggle) |
| `src/exo/worker/tests/unittests/test_mlx/test_kv_prefix_cache.py` | modified | 2 model-backed regression tests |
| `src/exo/master/tests/test_placement_memory.py` | new | 15 Phase 3 tests |

### Verification bar (AGENTS.md)

`basedpyright` 0 errors · `ruff check` clean · `ruff format` clean · `pytest`
173 passed (2 model-download skips).

---

## 1. Issue #1 — admission ceiling too aggressive (rejected a model that fit)

### Symptom

On the 2×256 GB cluster with `iogpu.wired_limit_mb=256000`, every prefill was
rejected immediately after warmup, ~1 s into prompt computation:

```
Prefill would require ~201.80 GiB peak (current 201.44 GiB + KV+SDPA 372.0 MiB)
but the prefill ceiling is 187.50 GiB
```

The model weights alone (201 GiB) exceeded the 187.5 GiB ceiling.

### Root cause

The Phase 1 implementation computed the ceiling as
`max_recommended_working_set_size × _PREFILL_MEMORY_THRESHOLD (0.75)` = 187.5
GiB. Two bugs:

1. **`iogpu.wired_limit_mb` was ignored.** Metal's
   `max_recommended_working_set_size` doesn't reflect the operator-raised wired
   limit; on a 256 GB box with `iogpu.wired_limit_mb=256000` both are ~250 GiB,
   but on a box where the operator raised the limit above the default, the
   Metal cap understates the real budget.
2. **The eviction watermark (0.75) was used as the admission hard limit.** A
   loaded model legitimately sits above 0.75 of the budget — resident weights
   are not a transient. Admission should only reject what would *actually* OOM,
   not what merely crosses the conservative cache-trim line.

The model had run fine for long periods without OOM before task #11, proving
the ceiling was wrong, not the model.

### Fix direction

Decouple the admission ceiling from the eviction watermark, and honour
`iogpu.wired_limit_mb`. This led directly to the reclaim-based ceiling rewrite
(Issue #2).

---

## 2. Issue #2 — reclaim-based ceiling rewrite (port of oMLX)

### What changed

Replaced the flat `max_recommended_working_set_size × fraction` ceiling with
oMLX's three-component model (`memory_guard.py`, ported from
`omlx/process_memory_enforcer.py`):

```
hard_limit = min(static, dynamic, metal_cap)
  static  = total_ram - tier_reserve          (aggressive: 4 GiB)
  dynamic = phys_footprint + free + inactive + active × reclaim_ratio   (aggressive: 0.8)
  metal_cap = iogpu.wired_limit_mb if set, else max_recommended_working_set_size
```

The `dynamic` term is the key: it adds the process's own `phys_footprint`
(the loaded model, read via `proc_pid_rusage(RUSAGE_INFO_V4)`) to the host's
reclaimable memory (`vm_statistics64` via ctypes). So a 201 GiB model on a
250 GiB box sees a ceiling of ~201 + reclaimable (~213 GiB), not 187.5 — and
the 372 MiB prefill peak is admitted.

This matches oMLX's page the user quoted:
`Free 0.1 + inactive 71.1 + (active × 80% = 12.2) → ceiling 502.4`
(= phys_footprint ~419 + reclaimable 83).

**Tiers** (configurable via `EXO_MEMORY_GUARD_TIER`, default `balanced`):
- `safe`: 8 GiB reserve, 0.2 active reclaim
- `balanced`: 6 GiB reserve, 0.5 active reclaim
- `aggressive`: 4 GiB reserve, 0.8 active reclaim ← recommended for the
  2×256 GB cluster

Escape hatches: `EXO_DISABLE_PREFILL_GUARD=1` (restore pre-task-#11
behaviour), `EXO_MEMORY_GUARD_CUSTOM_CEILING_BYTES` (absolute override).

### Ship default: OFF

The guard ships **disabled by default** (opt-in via `EXO_ENABLE_PREFILL_GUARD=1`
or the UI toggle). A fresh install behaves exactly as before task #11 — no
preflight rejection, no per-chunk abort — until the operator explicitly enables
it. This keeps the cluster working out-of-the-box while the reclaim model is
validated.

### Verification

Simulated the user's scenario: old ceiling 187.5 GiB (rejected) → new ceiling
213.5 GiB (admitted). 12 `test_memory_guard.py` tests cover the
three-component ceiling, reclaim ratios, the loaded-model regression, and the
disable/custom escape hatches.

---

## 3. Issue #3 — SIGBUS from a truncated ctypes struct

### Symptom

The present version died each time it started computing a prompt: the model
loaded and warmed up correctly, but ~1 s into prompt computation:

```
Error: Runner shutdown before completing command (signal=10 (Bus error: 10))
```

### Root cause

`get_phys_footprint()` declares a `_RusageInfoV4` ctypes struct. The Phase 1
implementation declared only the **9 prefix fields (80 bytes)**, intending to
stop reading at `ri_phys_footprint`. But `proc_pid_rusage(RUSAGE_INFO_V4)`
writes the **full 36-field struct (296 bytes)** regardless of which field the
caller reads — a 216-byte buffer overflow that corrupted the heap and crashed
with SIGBUS every call.

`get_phys_footprint` runs on every prefill (via `current_usage_bytes` + the
dynamic ceiling), so the crash fired ~1 s into the first prompt. Proven:

```
My struct size : 80 bytes
Full struct size: 296 bytes
OVERFLOW bytes : 216 => kernel writes 296 into a 80 buffer = CORRUPTION/SIGBUS
```

oMLX declares all 36 fields; the "prefix only" optimization was wrong.

### Fix

Declared the complete `rusage_info_v4` layout (all 36 fields, 296 bytes),
ported field-for-field from oMLX's `utils.proc_memory._RusageInfoV4`. Verified:
`ctypes.sizeof(_RusageInfoV4) == 296`, `get_phys_footprint()` returns real
values without crashing under pytest and in the bundled app, and the server
stays alive through prefill-triggering calls.

### Lesson

`proc_pid_rusage` always writes the full struct for the requested info tier —
a truncated ctypes buffer is a heap-corruption bug, not a harmless
prefix-only read. Always declare the complete kernel struct layout.

---

## 4. Issue #4 — runtime on/off toggle + build-version badge

### Request

1. An on/off switch for the feature, placed in the Advanced Options section
   alongside `Sharding Strategy:` / `Interconnect:` / `Minimum Devices:`.
2. The `Prefill / Decode` tab retained too.
3. The DMG version shown in the top-left corner so it's easy to tell which
   build is running.

### What changed

**Backend** — `memory_guard.py` gained a runtime-mutable
`_runtime_enabled_override` with `is_guard_enabled()` / `set_guard_enabled()`
accessors; `ceiling_breakdown()` and all guard paths consult
`_is_guard_enabled()`, so the toggle takes effect on the next prefill without
a restart.

**API** — `GET /v1/feature-flags` now includes `"prefillMemoryGuard": bool`;
`PUT /v1/memory-guard {"enabled": true|false}` flips the runtime toggle; new
`GET /v1/version` returns the build version (`EXO_APP_VERSION`).

**Swift** — `ExoProcessController.swift` injects
`EXO_APP_VERSION = CFBundleShortVersionString` into the exo child env.

**Dashboard** — a `Memory Guard:` On/Off radio-style toggle inside the
Advanced Options collapsible (same style as the other options), calling
`PUT /v1/memory-guard` live. The Advanced route page (`/#/advanced`) keeps a
detailed "Memory guard" tab (always accessible; default landing) and the
"Prefill / Decode" tab. `HeaderNav.svelte` shows a small monospace build-version
badge in the top-left corner.

### Verification

`GET /v1/version` → `{"version":"1.0.72-memory-headroom-dev4"}` ✓
`GET /v1/feature-flags` → `{"disaggregation":false,"prefillMemoryGuard":false}`
(default off) ✓
`PUT /v1/memory-guard {"enabled":true}` → `{"prefillMemoryGuard":true}` ✓
(toggle takes effect without restart)

---

## 5. Build/deploy record

| Build | Tag | Key change | Result |
|-------|-----|------------|--------|
| dev1 | `1.0.72-memory-headroom-dev1` | Phases 1–3, flat-fraction ceiling | ❌ rejected the 201 GiB model (Issue #1) |
| dev2 | `1.0.72-memory-headroom-dev2` | reclaim-based ceiling rewrite (Issue #2) | ❌ SIGBUS ~1 s into prompt (Issue #3) |
| dev3 | `1.0.72-memory-headroom-dev3` | full 296-byte ctypes struct fix; ship default OFF | ✅ no crash; toggle added |
| dev4 | `1.0.72-memory-headroom-dev4` | On/Off toggle in Advanced Options; version badge | ✅ final |

### Build gotcha learned

Xcode caches the `dist/exo` folder reference in derived data. After rebuilding
PyInstaller, a non-clean `xcodebuild` may embed a **stale** `dist/exo` (the
DMG app's binary timestamp lagged the fresh build). Fix: delete
`app/EXO/build/Build/Products/Release/EXO.app/Contents/Resources/exo` before
the xcodebuild, or clean the derived data. Always verify the DMG app's `exo`
binary serves the newest route (e.g. `curl /v1/version`) before shipping.

### Deploying to the cluster

```bash
scp output/EXO-1.0.72-memory-headroom-dev4.dmg gra@m3msu256a.local:/tmp/
scp output/EXO-1.0.72-memory-headroom-dev4.dmg gra@m3msu256b.local:/tmp/
# On each machine: open the DMG, QUIT the old EXO fully, launch the new one.
```

The guard is **off by default** — the cluster works immediately as before
task #11. To enable: set `EXO_MEMORY_GUARD_TIER=aggressive` on each node, then
toggle **On** from the dashboard's Advanced Options (or
`EXO_ENABLE_PREFILL_GUARD=1` at launch). If anything goes wrong, toggle it
back off — no restart needed.

---

## 6. Current state & next steps

### Done & verified
- ✅ Phase 1: evict-before-prefill + preflight admission (#2251 port + oMLX
  estimate); `PrefillMemoryExceededError` for impossible prompts.
- ✅ Phase 2: per-chunk EWMA transient guard (`PrefillTransientTracker` +
  `guard_prefill_chunk_or_raise`) wired through `prefill()`'s progress
  callback.
- ✅ Phase 3: placement reserves `weights + KV(context) + activation margin`
  per node (`placement_memory.py`); `EXO_PLACEMENT_CONTEXT_TOKENS` is the
  near-limit lever (#1709).
- ✅ Reclaim-based ceiling (oMLX port): honours `iogpu.wired_limit_mb` and the
  process's own `phys_footprint`; tiers (safe/balanced/aggressive).
- ✅ SIGBUS fix (full 296-byte `rusage_info_v4` struct).
- ✅ Ship default OFF; runtime on/off toggle (UI + API); build-version badge.
- ✅ 69 tests (42 prefill-headroom + 12 memory-guard + 15 placement-memory).

### Pending / future
- ⚠️ Dashboard max-context-length control (#2241): backend lever
  (`EXO_PLACEMENT_CONTEXT_TOKENS`) is in place; UI wiring is separate.
- ⚠️ MLA-precise KV estimation: currently over-counts MLA models (safe but
  could cause false placement rejections).
- ⚠️ HTTP 400 mapping for `PrefillMemoryExceededError` (currently a generic
  500).
- ⚠️ Cluster validation of the reclaim-based ceiling on the 2×256 GB pair
  (the math is verified in simulation; needs an empirical run with
  `EXO_MEMORY_GUARD_TIER=aggressive`).

### Lessons
1. **A loaded model is not a transient.** An admission ceiling that charges
   resident weights against a conservative fraction (0.75) rejects models that
   fit. The ceiling must grow with the process's own footprint (reclaim model)
   or admission must exclude resident weights.
2. **`proc_pid_rusage` writes the full struct.** A truncated ctypes buffer is
   heap corruption, not a prefix-only read. Declare the complete kernel struct.
3. **Metal's `max_recommended_working_set_size` ignores `iogpu.wired_limit_mb`.**
   Operators raise the wired ceiling via sysctl; take `max(recommended,
   wired_limit)`.
4. **Ship safety-critical guards disabled by default.** A guard that can
   reject prefills must be opt-in until validated on the target cluster; an
   env escape hatch alone isn't enough when the default-on behaviour is
   worse than the unguarded behaviour.