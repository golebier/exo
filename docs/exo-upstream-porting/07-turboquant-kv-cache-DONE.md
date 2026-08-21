# 07 — TurboQuant KV-Cache Compression (DONE log)

**Status:** The runtime-toggleable TurboQuant settings + `make_kv_cache`
integration are implemented, verified, and shipped, with the full dashboard
control surface (On/Off, bits dropdown, skip-last checkbox) mirroring oMLX's
`ModelSettings.turboquant_kv_*`. Composes directly with the tiered-cache
work (oMLX doc 01) — quantized blocks spill to SSD smaller. This doc is the
implementation record.

**Build tag:** `1.0.72-turboquant-tiered-dev4`
**oMLX commit referenced:** `c1a3d44` (`jundot/omlx`)
**Target cluster:** 2×256 GB RAM, `iogpu.wired_limit_mb=256000` per node

---

## 0. What was built (summary)

EXO already had `QuantizedKVCache` via a blunt global `KV_CACHE_BITS` constant
(PR #1988), but no per-model tuning, no skip-last, no fast path, no runtime
toggle. TurboQuant is oMLX's tuned, per-model, opt-in KV-cache compression
with a skip-last-N-layers option and a fast Apple-Silicon path.

This change adds the **runtime settings + `make_kv_cache` integration** for
TurboQuant, mirroring oMLX's `ModelSettings.turboquant_kv_enabled` /
`turboquant_kv_bits` / `turboquant_skip_last`, wired through EXO's
Memory-Guard API pattern (env default → runtime override →
`/v1/feature-flags` → Svelte store → toggle in Advanced Options). The bits
dropdown offers oMLX's full depth set (2/2.5/3/3.5/4/6/8); half-step depths
round down to the next integer for mlx-lm's `QuantizedKVCache` until the
native TurboQuant attention kernel lands.

### Files added / changed

| File | Status | Purpose |
|------|--------|---------|
| `src/exo/worker/engines/mlx/turboquant.py` | new | TurboQuant runtime settings: `is_turboquant_enabled` / `turboquant_bits` / `turboquant_skip_last` / `effective_kv_bits` (half-step rounding); env defaults + runtime overrides; also holds the tiered-cache settings (see oMLX doc 01-DONE) |
| `src/exo/worker/engines/mlx/cache.py` | modified | `make_kv_cache` consults TurboQuant (overrides legacy `KV_CACHE_BITS`), applies skip-last (keeps final layer full precision), defers to `model.make_cache()` for Qwen3-Next hybrid caches; `_KV_BYTES_PER_ELEMENT` updated so the prefill estimator honours the quantized width |
| `src/exo/api/main.py` | modified | `PUT /v1/turboquant`; `turboquantKv` in feature flags |
| `src/exo/api/types/api.py` | modified | `TurboQuantSetting` body model (mirrors oMLX `ModelSettings.turboquant_kv_*`) |
| `src/exo/api/types/__init__.py` | modified | Re-export `TurboQuantSetting` |
| `dashboard/src/lib/stores/app.svelte.ts` | modified | `setTurboQuant` store method |
| `dashboard/src/routes/+page.svelte` | modified | TurboQuant KV Cache On/Off toggle + bits `<select>` (2–8 bit) + skip-last checkbox, persisted to localStorage |
| `src/exo/worker/tests/unittests/test_mlx/test_turboquant.py` | new | 40 tests (shared with tiered cache): defaults, runtime overrides, bit normalisation, `make_kv_cache` skip-last integration |

### Verification bar (AGENTS.md)

`basedpyright` 0 errors · `ruff check` clean · `ruff format` clean ·
dashboard builds clean · 40 new tests pass · bundled-binary round-trip
verified.

---

## 1. oMLX → EXO config-surface mapping

oMLX surfaces TurboQuant per-model via `ModelSettings` (a dataclass in
`omlx/model_settings.py`). EXO has no per-model JSON settings manager, so
the config is wired globally through the Memory-Guard pattern:

| oMLX `ModelSettings` field | oMLX default | EXO env default | EXO runtime API | EXO dashboard |
|----------------------------|--------------|-----------------|-----------------|---------------|
| `turboquant_kv_enabled` | `False` | `EXO_TURBOQUANT_KV=0` | `PUT /v1/turboquant {enabled}` | TurboQuant KV Cache On/Off |
| `turboquant_kv_bits` | `4` | `EXO_TURBOQUANT_KV_BITS=4` | `PUT /v1/turboquant {bits}` | Bits `<select>` (2/2.5/3/3.5/4/6/8) |
| `turboquant_skip_last` | `True` | `EXO_TURBOQUANT_SKIP_LAST=1` | `PUT /v1/turboquant {skip_last}` | "Skip last layer" checkbox |

Defaults match oMLX exactly: disabled, bits=4, skip_last=True. Ship default
is **off** so a fresh install keeps the existing fp16/bf16 behaviour.

## 2. `make_kv_cache` integration + Qwen3-Next hybrid-cache awareness

`make_kv_cache` now resolves the effective bit depth as
`turboquant.effective_kv_bits() if TurboQuant enabled else KV_CACHE_BITS`,
so TurboQuant takes precedence over the legacy global when enabled. The
`turboquant_skip_last` setting keeps the **final** KVCache layer full
precision (a plain `KVCache`) to prevent corruption on quality-sensitive
models — oMLX's docstring calls this out explicitly and it's the default.

**Qwen3-Next hybrid-cache awareness:** models exposing `make_cache` (hybrid
`ArraysCache`+`KVCache` models like Qwen3-Next) defer to their own
constructor, so the TurboQuant bits apply only to the plain-KV branches.
This is the same seam oMLX's `type_handlers.py` keys on — the design doc's
"explicit handling for Qwen3-Next's hybrid cache" requirement is satisfied
by deferring to `model.make_cache()`.

**Half-step bit depths** (2.5/3.5): mlx-lm's `QuantizedKVCache` only accepts
integer bits, so `effective_kv_bits()` rounds half-steps **down** (2.5→2,
3.5→3). The half-step precision is reclaimed once the native TurboQuant
attention kernel (oMLX `patches/turboquant_attention.py`) is ported; until
then the integer fallback is the documented blunt-instrument path (EXO PR
#1988). The full depth set is kept in the UI so persisted settings stay
forward-compatible.

## 3. Composition with the tiered cache (oMLX doc 01)

TurboQuant composes directly with the tiered cache shipped alongside it
(see [`01-tiered-kv-cache-ssd-DONE.md`](../omlx-porting/01-tiered-kv-cache-ssd-DONE.md)):
quantized KV blocks are smaller, so they spill to SSD smaller and restore
faster. The two toggles are independent in the UI but the design docs cross-
reference this composition explicitly. Both default off.

## 4. What is NOT yet ported (staged behind the flag)

Per the design doc's phased plan:

- **Phase 1 — Correctness hardening (#1990, #2261):** skip KV quant in
  single-node BatchGenerator mode; force clean prefill after chained
  prefix-cache extensions. Both are latent bugs in the current quant path.
- **Phase 3 — Apple Silicon fast path:** tuned quantized KV kernel
  (reference: oMLX `turboquant_attention.py`). Until then the integer
  `QuantizedKVCache` fallback is used.
- **Phase 4 — PDD cache handoff:** wire format for quantized KV in
  disaggregated/remote prefill.

The flag shipped here is the control surface these phases wire into.

## 5. Build/deploy record

- `EXO_VERSION=1.0.72-turboquant-tiered-dev4` · `BUILD.sh` clean (5/5 steps)
- DMG: `output/EXO-1.0.72-turboquant-tiered-dev4.dmg` (316 MB)
- Deployed to 2-node cluster (`m3msu256a.local` + `m3msu256b.local`)
- Live round-trip verified on the bundled binary: `PUT /v1/turboquant`
  reflects in feature flags; `make_kv_cache` builds `QuantizedKVCache` when
  enabled, keeps the last layer full precision when skip_last=True, defers
  to `model.make_cache()` for GLM-5.2's hybrid cache

## 6. JACCL warmup race fix (bonus, blocking the cluster)

During cluster deployment a **pre-existing** JACCL warmup race (commit
`311fda60`, not part of this feature work) blocked model loading at 99 %
(77/78 layers). The in-process `all_sum` warmup completed unilaterally on
rank 0 before rank 1 joined, desynchronising the JACCL group's collective
counter and hanging rank 1's model-loading collectives on an
`IOSurfaceSharedEvent` that never signalled.

**Fix:** removed the in-process warmup entirely. `_probe_rdma_interface`
already validates the RDMA data path in a subprocess (both ranks
participating simultaneously in separate processes); the in-process warmup
was redundant and broke the group. Verified: both ranks now load in
lockstep (66.9 s each, within 80 ms) and report "runner ready" with no hang.

## 7. Current state & next steps

Shipped and live on the cluster. The settings layer + `make_kv_cache`
integration + dashboard controls are the user-facing deliverable; the
correctness hardening (#1990, #2261) and the native fast-path kernel
(Phase 3) are the next implementation milestones.