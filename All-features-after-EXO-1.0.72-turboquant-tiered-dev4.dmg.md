# All Features After `EXO-1.0.72-turboquant-tiered-dev4.dmg`

**Baseline (DMG):** `output/EXO-1.0.72-turboquant-tiered-dev4.dmg` (331,639,878 B, built 2026‑08‑21 14:12)
→ git tag `v1.0.72-turboquant-tiered-dev4` → commit `172524529ffd6d5730a4cfc82bad1ce28dd315ab`
*"feat(mlx): TurboQuant KV compression + tiered KV cache (Hot RAM / Cold SSD) + JACCL warmup fix"*

**Present local code:** working tree at `HEAD` = `81372b9f` (2026‑08‑26) **plus uncommitted changes**
(built through `output/EXO-1.0.72-stall-watchdog-ui-dev1.dmg`, 2026‑09‑01 16:15).

**Scope of this document:** every difference between the baseline DMG and the present
local code, organized by the **task names from the docs task lists**
(`docs/exo-upstream-porting/` tasks 01–12 and `docs/omlx-porting/` tasks 01–09). For
each task: point‑by‑point what is in the present local code **vs** what was (or was not)
in the `turboquant-tiered-dev4` DMG.

**Total delta:** 51 files changed, **+5,077 insertions / −217 deletions** vs the baseline tag
(`git diff --stat v1.0.72-turboquant-tiered-dev4`).

---

## How to read each section

Each task block has three parts where applicable:

- **In `turboquant-tiered-dev4` (baseline):** the state shipped in the DMG.
- **In present local code:** the state in the working tree now.
- **Point‑by‑point delta:** the concrete changes, with files / commits / build artifacts.

> ⚠️ **Important nuance — reverts.** Several TP‑stall fixes were **committed after** the
> baseline DMG (commits `c86f57df`…`b36895da`) and then **reverted in the uncommitted
> working tree** back toward the known‑working decode path. So for some areas the present
> local code is *not* simply "baseline + additions"; it reverts parts of the post‑baseline
> commits. Those reverts are called out explicitly per task.

---

## DMG build lineage after the baseline (context)

The present local code was exercised through this sequence of dev DMGs (all in `output/`),
each isolating one change. They are the empirical record behind the uncommitted working‑tree
diff:

| DMG | Date | What it isolated |
|-----|------|------------------|
| `EXO-1.0.72-cache-efficiency-dev1.dmg` | 08‑21 16:30 | per‑instance cache‑efficiency stat cards |
| `EXO-1.0.72-cache-efficiency-dev2.dmg` | 08‑24 14:42 | reclaim‑based prefix‑cache eviction |
| `EXO-1.0.72-ssd-tier-dev1.dmg` | 08‑24 20:17 | SSD cold tier + TurboQuant hardening + HTTP 400 |
| `EXO-1.0.72-ssd-prefix-restore-dev1.dmg` | 08‑25 06:45 | prefix‑SSD KV restore |
| `EXO-1.0.72-tp-stall-fix-dev1..3.dmg` | 08‑25 | KV deepcopy / flush / gc reclaim fixes |
| `EXO-1.0.72-tp-decode-reclaim-dev1.dmg` | 08‑26 | periodic reclaim during decode |
| `EXO-1.0.72-tp-deadlock-fix-dev1.dmg` | 08‑26 | eviction‑loop collective deadlock |
| `EXO-1.0.72-tp-cachelist-eval-fix-dev1.dmg` | 08‑26 | CacheList KV state eval (root cause) |
| `EXO-1.0.72-jaccl-warmup-restore-dev1.dmg` | 08‑27 | restored in‑process JACCL warmup |
| `EXO-1.0.72-native-kernels-default-off-dev1.dmg` | 08‑27 | native kernels → opt‑in |
| `EXO-1.0.72-revert-to-working-decode-dev1.dmg` | 08‑31 | reverted TP stall fixes to working path |
| `EXO-1.0.72-async-eval-fix-dev1.dmg` | 08‑31 | periodic `mx.clear_cache()` in batch step |
| `EXO-1.0.72-async-eval-instance-flag-dev1.dmg` | 08‑31 | instance flag for async eval |
| `EXO-1.0.72-adaptive-prefill-step-dev1.dmg` | 08‑31 | adaptive prefill step size |
| `EXO-1.0.72-batch-clear-cache-dev1.dmg` | 08‑31 | batch clear‑cache every 128 steps |
| `EXO-1.0.72-stall-watchdog-ui-dev1.dmg` | 09‑01 | per‑request stall‑watchdog UI override |

---

# Part A — oMLX → EXO porting tasks (`docs/omlx-porting/`)

## omlx 01 — Tiered KV cache (Hot RAM + Cold SSD + persistence)

**In `turboquant-tiered-dev4` (baseline):**
- Settings/flag layer only in `turboquant.py` (`is_tiered_cache_enabled`, `hot_cache_only`,
  `ssd_cache_dir`, `ssd_cache_max_size`, `hot_cache_max_size`, `tiered_cache_status`,
  `clear_ssd_cache`).
- API routes `PUT/GET/DELETE /v1/tiered-cache` + `tieredKvCache` feature flag.
- Dashboard On/Off toggle + SSD dir/cap/RAM‑cap inputs + live observability gauge.
- **No actual SSD spill/restore** — the tier was a control surface with no backing store.
- JACCL warmup `all_sum` **removed** from `utils_mlx.py` (the dev4 "JACCL warmup race fix").

**In present local code:**
- Full SSD cold tier (Phases 2–3) shipped as an adjunct to `KVPrefixCache`.
- Longest‑common‑prefix SSD restore added.
- (Uncommitted) JACCL warmup `all_sum` **restored** in `utils_mlx.py` — see omlx 02 / exo 06.

**Point‑by‑point delta:**

1. **NEW `src/exo/worker/engines/mlx/ssd_cache.py` (+741 lines)** — `SSDKVCacheStore`:
   - spill/restore via mlx‑lm's `save_prompt_cache` / `load_prompt_cache` (byte‑exact for
     `KVCache`, `QuantizedKVCache`, `RotatingKVCache`, `ArraysCache`, `CacheList`).
   - SSD‑eligibility for exotic classes (`DeepseekV4Cache` etc. degrade to RAM‑only).
   - cache‑signature guard (model id + per‑layer class names + quant bits/group‑size +
     effective KV bits) — refuses (and removes) a stale SSD block from a different
     model/quant config.
   - restart‑recovery scan (rebuilds in‑RAM index from SSD dir hash‑prefix subdirs on
     construction; removes files missing EXO metadata).
   - LRU size cap + access‑order seeding from file mtime.
   - `.tokens.npy` int32 sidecar per spill for prefix restore.
   - 14 tests in `test_ssd_cache.py`.
   *(commit `c8f1895e`)*
2. **`src/exo/worker/engines/mlx/cache.py` (+391 lines net)** — `KVPrefixCache`:
   - `set_ssd_store` / `set_model_id` wired.
   - evicted RAM entries spill to SSD in `_evict_until_under`.
   - exact‑match SSD restore in `get_kv_cache` (skips re‑prefill, even after restart).
   - `clear()` wipes both tiers.
   - `restore_prefix()` longest‑common‑prefix scan over sidecar token arrays (RAM‑only,
     no disk I/O) then load + signature‑validate the best entry; trim to prefix, prefill
     only the suffix. Non‑trimmable layers (ArraysCache/RotatingKVCache/DeepseekV4Cache)
     refuse partial restore but still serve exact match.
   *(commits `c8f1895e`, `56ef7390`; 9 new prefix‑restore tests)*
3. **`src/exo/worker/engines/mlx/builder.py` (+17 lines)** — wires the store from resolved
   tiered‑cache settings + model id; no‑ops unless `EXO_TIERED_KV_CACHE` set. *(commit `c8f1895e`)*
4. **NOT in present local:** Phase 1 paged‑block manager (`KVCacheBlock`,
   `FreeKVCacheBlockQueue`, chain‑hash, COW); Phase 4 boundary snapshot SSD offload (omlx 08).

---

## omlx 02 — Native Metal kernels (GLM‑5.2)

**In `turboquant-tiered-dev4` (baseline):**
- `mlx_kernels/` build infra + `OMLX_COMMIT.txt` pinned `1f1aff3`.
- Vendored `omlx_custom_kernels/glm_moe_dsa/` + `vendor/glm_moe_dsa/kernels.py` with
  **lazy `_ext` import**.
- Native kernels **default‑ON** (`EXO_NATIVE_GLM_KERNELS=0` to disable).
- JACCL `all_sum` warmup **removed** (dev4 fix for the rendezvous race).
- Issue #3 (JACCL rendezvous race) declared fixed by warmup removal + master lifecycle gating.

**In present local code:**
- Native kernels flipped to **default‑OFF** (opt‑in via `EXO_NATIVE_GLM_KERNELS=1`).
- JACCL warmup `all_sum` **restored** in `utils_mlx.py` (re‑added as a sync point).
- New Issue #4 documented + fixed: periodic `mx.clear_cache()` in the batch decode step.
- DONE doc updated with the Issue #4 post‑mortem.

**Point‑by‑point delta:**

1. **`src/exo/worker/engines/mlx/vendor/glm_moe_dsa/kernels.py`** — `_resolve_native_fast()`
   semantics flipped:
   - baseline: import attempted unless `EXO_NATIVE_GLM_KERNELS` ∈ {0,false,no,off}.
   - present: import **skipped** unless env ∈ {1,true,yes,on}. Comment: native kernels
     "change Metal command‑buffer timing and, under TP, can perturb the collective ordering
     that the sync‑eval fix depends on."
   *(uncommitted; DMG `native-kernels-default-off-dev1`)*
2. **`src/exo/worker/engines/mlx/utils_mlx.py`** — JACCL warmup **restored** in
   `mlx_distributed_init`:
   - baseline comment: "We do NOT run an in‑process warmup collective here: JACCL's
     `all_sum` can complete unilaterally…"
   - present: `if isinstance(bound_instance.instance, MlxJacclInstance): mx.eval(
     mx.distributed.all_sum(mx.array(1.0), group=group))` with comment that without it the
     collective counters drift and after ~40 s of decode (78 MoE layers × all_sum/step) the
     mismatch surfaces as a permanent `Fence::wait` hang.
   *(uncommitted; DMG `jaccl-warmup-restore-dev1`)*
3. **`src/exo/worker/engines/mlx/patches/opt_batch_gen.py` (+30 lines)** — `_patched_step`
   now periodically calls `mx.clear_cache()` every **128 decode steps** (half of
   `stream_generate`'s 256 interval), mirroring stream_generate's graph‑cache clearing.
   Without it the batch decode path accumulates intermediates indefinitely; on TP clusters
   the Metal command buffer fills and JACCL collectives hang (`Fence::wait` deadlock).
   Adds `_get_step_counter` / `_increment_step_counter` helpers + `_CLEAR_CACHE_EVERY=128`.
   *(uncommitted; DMGs `async-eval-fix` / `batch-clear-cache`)*
4. **`src/exo/worker/engines/mlx/vendor/omlx_custom_kernels/glm_moe_dsa/fast.py`** — minor
   formatting (`mx.where` one‑liner).
5. **`test_glm_moe_dsa_native_kernels.py`** — `TestDispatchFallbackBehavior` updated for
   default‑off: new `test_native_default_off_without_env` asserting native is unavailable
   when the env var is unset; docstrings updated.
6. **`docs/omlx-porting/02-native-metal-kernels-DONE.md` (+181/−…)** — added "Issue #4 — TP
   decode stall: extra collectives introduced between prefill and decode (root cause)"
   post‑mortem documenting the revert‑to‑working‑decode fix (see exo 06).

---

## omlx 03 — Multi‑Token Prediction (MTP)
**No change.** Design only in both baseline and present local. Not started.

## omlx 04 — SpecPrefill (sparse prefill)
**No change.** Design only in both. Not started.

## omlx 05 — EnginePool + Model Profiles
**No change.** Design only in both. Not started.

## omlx 06 — GLM adaptive prefill patch
**Partial / indirect.** Not the oMLX adaptive‑prefill patch per se, but present local adds an
**adaptive prefill *step size*** in `generate.py` (see exo 06 / exo 11 below). The oMLX
DSA‑specific adaptive prefill patch is still design‑only.

## omlx 07 — Embedding + Reranker engines
**No change.** Design only in both. Not started.

## omlx 08 — Boundary snapshot SSD offload
**No change.** Design only in both. Not started (Phase 4 of omlx 01).

## omlx 09 — Claude Code context‑scaling + SSE keep‑alive
**No change.** Design only in both. Not started. (The SSE keep‑alive / stall‑watchdog work
under exo 06 is adjacent but distinct.)

---

# Part B — Upstream → EXO porting tasks (`docs/exo-upstream-porting/`)

## exo 06 — GLM‑5.2 long‑context fix + RDMA/Thunderbolt reliability  ⭐ (most churn)

This is the task with the largest delta, including the **revert** nuance.

**In `turboquant-tiered-dev4` (baseline):**
- GLM‑5.2 vendored model + TP sharding + tool calling already in place (landed pre‑baseline
  in `310530ea`/`6ad281cd`/`e060b792`).
- JACCL warmup removed (dev4).
- **No** decode stall watchdog.
- **No** TP stall fixes (deepcopy/flush/gc/eviction‑collective/cachelist‑eval).
- `flush_prefill_for_decode` did **not** exist.
- `_copy_kv_cache` (shallow copy) did **not** exist — `KVPrefixCache` used `deepcopy`.

**In present local code:**
- Decode stall watchdog shipped (committed) **and** gained a per‑request UI override (uncommitted).
- Five TP‑stall fixes (A–E) were committed after baseline, then **reverted in the uncommitted
  working tree** back to the known‑working `deepcopy` + plain decode path.
- Adaptive prefill step size added (uncommitted).
- `Fence::wait` root cause re‑analyzed: the post‑baseline fixes added *extra TP collectives*
  between prefill and decode that desynchronized ranks; the revert removes them.

**Point‑by‑point delta:**

1. **Decode stall watchdog (COMMITTED, kept)** — `src/exo/api/main.py::_token_chunk_stream`
   bounds the wait for the next chunk. Because the mlx `Fence::wait` collective blocks the
   runner's C++ thread, only the separate API/master process can observe the absence of
   chunks.
   - `src/exo/shared/constants.py` (+17): `EXO_DECODE_STALL_TIMEOUT` (120 s default) and
     `EXO_PREFILL_STALL_TIMEOUT` (180 s default); `0` disables.
   - `src/exo/api/tests/test_stall_watchdog.py` (+177): 4 tests (decode stall, prefill stall,
     disabled, normal‑streaming no‑false‑positive).
   - `tests/test_2node.py` (+49): `test_2node_long_context_jaccl` hardware‑gated `@slow`
     ~46k‑token 2‑node TP test.
   *(commit `37047787`; DMG `cache-efficiency-dev2` verified live)*

2. **TP stall Fix A — shallow KV copy (COMMITTED then REVERTED)** — `cache.py::_copy_kv_cache`
   shared MLX array data via `copy.copy` (O(layers) not O(KV bytes)) on `add_kv_cache` /
   `update_kv_cache` / `get_kv_cache`.
   - **Present local (uncommitted):** reverted back to `deepcopy` in all three call sites;
     `_copy_kv_cache` retained but marked `# pyright: ignore[reportUnusedFunction]` with a
     note "Currently unused by the runtime (reverted to `deepcopy` to match the known‑working
     version)." 9 regression tests for the shallow‑copy invariant still pin the helper.
   *(commit `c86f57df`; reverted in DMG `revert-to-working-decode-dev1`)*

3. **TP stall Fix B — flush prefill transients (COMMITTED then REVERTED)** —
   `flush_prefill_for_decode()` did `mx.eval(cache.state)` + `gc.collect()` +
   `mx.clear_cache()` + `mx_barrier` between prefill and decode.
   - **Present local (uncommitted):** the call **removed** from `ExoBatchGenerator.submit`
     (and the `flush_prefill_for_decode` / `bind_chunk_guard` imports dropped from
     `batch_generate.py`). `collect_evalable_cache_arrays()` walker still exists in
     `generate.py` for the helper, but the batch path no longer invokes the flush.
   *(commit `bbb3190e`; reverted in DMG `revert-to-working-decode-dev1`)*

4. **TP stall Fix C — periodic gc/clear_cache during decode (COMMITTED then REVERTED)** —
   `gc.collect()` + `mx.clear_cache()` every 32 decode tokens in `ExoBatchGenerator.step`.
   - **Present local (uncommitted):** removed from `batch_generate.py` (the `import gc`
     dropped, the `% 32` block gone). Replaced by the `opt_batch_gen.py` every‑128‑steps
     `mx.clear_cache()` (omlx 02 above), which is on the *batch `_step`* path rather than
     the wrapper.
   *(commit `4800d95e`; reverted; superseded by `opt_batch_gen` clear‑cache)*

5. **TP stall Fix D — eviction‑loop collective deadlock (COMMITTED then REVERTED)** —
   restructured `_evict_until_under` so both ranks always call the `all_gather` collective
   once per iteration, then a second `all_sum` to agree on whether any rank still has caches.
   - **Present local (uncommitted):** the whole `evict_for_prefill_headroom()` **call** was
     removed from the batch path (it was the source of an extra pre‑prefill collective), so
     the deadlock‑safe loop restructuring is no longer exercised on the request path. The
     `_evict_until_under` changes remain in `cache.py` but are not triggered pre‑prefill.
   *(commit `fa47a075`; call removed in `revert-to-working-decode-dev1`)*

6. **TP stall Fix E — CacheList KV state eval (COMMITTED then REVERTED)** —
   `collect_evalable_cache_arrays()` recursive walker replaced the naive
   `mx.eval([c.state for c in cache])` that `contextlib.suppress` silently skipped when one
   `CacheList` sub‑cache raised. Root cause of the intermittent decode stall.
   - **Present local (uncommitted):** the `flush_prefill_for_decode` call that used it is
     removed from the batch path; `test_flush_prefill_cache_eval.py` (+204) still covers the
     walker. Verified on cluster *before* the revert: 25,520‑token cold prefill + 100 decode
     tokens completed in 145 s with zero stalls.
   *(commit `b36895da`; call removed in `revert-to-working-decode-dev1`)*

7. **Adaptive prefill step size (UNCOMMITTED, NEW)** — `generate.py::prefill`:
   - baseline: `prefill_step_size = 4096` (fixed).
   - present: adaptive by total context — `>64k → 512`, `>32k → 1024`, `>16k → 2048`,
     else `4096`. Reason: MLA attention scores matrix is
     `(num_heads × step × position)` per layer; at 128 heads × 4 bytes, a 4096‑chunk at
     position 60k needs ~126 GB (> the ~51 GB headroom on a 256 GB node with a ~199 GB
     model). Shrinking the step keeps peak attention under ~33 GB.
   *(uncommitted; DMG `adaptive-prefill-step-dev1`)*

8. **Per‑request decode stall watchdog override (UNCOMMITTED, NEW)** — dashboard control +
   API field so an operator can tune/disable the watchdog per model:
   - `src/exo/api/types/api.py`: `ChatCompletionRequest.decode_stall_timeout: float | None`
     (None ⇒ global default; 0 ⇒ disabled; >0 ⇒ custom bound).
   - `src/exo/api/main.py`: `_token_chunk_stream` / `_collect_text_generation_with_stats`
     gain a `decode_stall_timeout` param; threaded through chat (stream/non‑stream) + bench
     paths from `payload.decode_stall_timeout`.
   - `dashboard/src/lib/stores/app.svelte.ts`: `decodeStallTimeout` state +
     `setDecodeStallTimeout()`; included in all three chat request builders.
   - `dashboard/src/routes/+page.svelte` (+257): "Decode Stall Watchdog" radio group
     (Default 120s / Custom / Disabled) + custom seconds input, persisted in launch
     defaults (`LAUNCH_DEFAULTS_KEY`).
   *(uncommitted; DMG `stall-watchdog-ui-dev1`)*

9. **GLM‑5.2 fix‑summary doc (COMMITTED)** — `docs/gra/GLM-5.2-FIX-SUMMARY.md` (+78) and
   `docs/gra/CURRENT-STATE-GLM-5.2-RDMA.md` (+145, new) record the verified cluster state,
   the watchdog, and the post‑dev9 Fixes A–E. `docs/exo-upstream-porting/06-glm-long-context-rdma-reliability.md`
   reconciled (JACCL race marked FIXED; mlx‑jaccl pin marked done).

**Net exo 06 state vs baseline:** the watchdog + adaptive prefill step + per‑request UI
override are net‑new and kept; the five committed stall fixes are net‑reverted on the request
path (helpers/tests retained). The present local decode path is intentionally closer to the
pre‑baseline working version (`6ad281cd`) than to the post‑baseline stall‑fix commits.

---

## exo 07 — TurboQuant KV‑cache compression

**In `turboquant-tiered-dev4` (baseline):**
- `turboquant.py` settings (`is_turboquant_enabled`, `turboquant_bits` 2–8, `turboquant_skip_last`,
  `effective_kv_bits` with half‑step rounding).
- `make_kv_cache` consults TurboQuant (overrides legacy `KV_CACHE_BITS`), applies skip‑last,
  defers to `model.make_cache()` for Qwen3‑Next hybrid caches.
- `PUT /v1/turboquant` route + `turboquantKv` feature flag; dashboard toggle + bits dropdown.
- 40 tests in `test_turboquant.py`.
- **No** correctness hardening for #1990 / #2261.

**In present local code:** correctness hardening landed (committed) and is retained.

**Point‑by‑point delta:**

1. **#1990 — skip KV quantization in single‑node BatchGenerator mode (COMMITTED, kept):**
   `make_kv_cache(force_plain=)`; `ExoBatchGenerator` passes `force_plain=(self.group is None)`.
   mlx‑lm's `BatchGenerator` multi‑sequence trim/extend desyncs `QuantizedKVCache` on a single
   node; distributed mode (one sequence shard/rank) + hybrid‑cache models unaffected.
   *(commit `c8f1895e`)*
   - ⚠️ **Present‑local nuance:** the `force_plain=single_node` line was **removed** in the
     uncommitted revert (`batch_generate.py` now calls `make_kv_cache(self.model)` without
     `force_plain`), so #1990's single‑node guard is currently **not active on the batch
     path** (it lives only in `make_kv_cache`'s signature). The hardening tests in
     `test_turboquant_hardening.py` (+154) still pin the `force_plain` behavior at the unit
     level.
2. **#2261 — force clean prefill after chained prefix‑cache extensions (COMMITTED, kept):**
   `KVPrefixCache` tracks a per‑entry `_chained` flag (set by `update_kv_cache`); `get_kv_cache`
   refuses to reuse a chained+quantized entry for a further partial extension and returns a
   fresh cache. Gated on quantization; plain caches always safe to chain; exact hits unaffected.
   `_cache_is_quantized` + `_ensure_chained_parallel` helpers. *(commit `c8f1895e`)*
3. **NOT in present local:** Phase 3 native fast‑path kernel (oMLX `turboquant_attention.py`);
   Phase 4 PDD cache handoff for quantized KV in disaggregated prefill.

---

## exo 10 — Observability (Prometheus /metrics + cluster stats)

**In `turboquant-tiered-dev4` (baseline):**
- Per‑instance token accounting (`InstanceTokensUpdated` event: prompt_tokens,
  completion_tokens) landed pre‑baseline in `v1.0.72-InstanceTokenUsage-dev2`.
- Dashboard per‑instance token row (↓ in · ↑ out · N req).
- **No** cache‑efficiency aggregation; **no** `/metrics` endpoint; **no** stall‑watchdog UI.

**In present local code:** per‑instance cache‑efficiency stat cards + stall‑watchdog UI
control. Still no Prometheus `/metrics`.

**Point‑by‑point delta:**

1. **Per‑instance cache‑efficiency stat cards (COMMITTED, kept):**
   - `src/exo/shared/types/events.py` (+3): `InstanceTokensUpdated.cached_tokens: int = 0`
     (default for backward‑compatible replay of old persisted events).
   - `src/exo/shared/types/worker/token_usage.py` (+4): `InstanceTokenUsage.cached_tokens`.
   - `src/exo/shared/apply.py` (+2): `apply_instance_tokens_updated` folds `cached_tokens`
     into the running total (first‑request + accumulate branches).
   - `src/exo/worker/runner/runner.py` (+1): `send_chunk` emits `cached_tokens` from
     `usage.prompt_tokens_details.cached_tokens` on the final chunk.
   - `test_apply/test_apply_instance_token_usage.py` (+43): accumulation + legacy‑event
     replay; `test_instance_token_emission.py` (+25) runner emission.
   - Dashboard: `InstanceTokenUsage` store gains `cachedTokens`; instance card (desktop
     sidebar + welcome panel) gets a 3‑cell grid (Prefill / Cached / Efficiency) below the
     token row, gated on `promptTokens > 0`.
   *(commit `4292cefb`; DMG `cache-efficiency-dev1`)*
2. **Per‑request stall‑watchdog UI control (UNCOMMITTED)** — see exo 06 #8; the dashboard
   "Decode Stall Watchdog" radio + custom input is an observability/control surface.
3. **NOT in present local:** `/metrics` Prometheus endpoint; global cluster stats (zenoh
   Last Value).

---

## exo 11 — Memory headroom before prefill + near‑limit placement

**In `turboquant-tiered-dev4` (baseline):**
- Full Phases 1–3 already shipped pre‑baseline (`v1.0.72-memory-headroom-dev4`):
  `memory_guard.py` (reclaim‑based ceiling, `iogpu.wired_limit_mb` + `phys_footprint`,
  tiers safe/balanced/aggressive, ship‑default‑off), `prefill_transient_tracker.py` (EWMA),
  `exceptions.py::PrefillMemoryExceededError`, `cache.py` eviction/preflight/guard helpers,
  `placement_memory.py` + `placement_utils.py` (`EXO_PLACEMENT_CONTEXT_TOKENS`),
  `GET /v1/version`, `PUT /v1/memory-guard`, dashboard Memory Guard toggle + version badge.
- Eviction used `psutil.virtual_memory().percent` vs fixed 0.75/0.85 watermarks (bug: wired
  model pages counted as "used").
- **No** HTTP 400 mapping for `PrefillMemoryExceededError`.

**In present local code:** reclaim‑based eviction fix + HTTP 400 mapping (committed, kept);
preflight admission **removed from the batch request path** in the uncommitted revert.

**Point‑by‑point delta:**

1. **Reclaim‑based prefix‑cache eviction (COMMITTED, kept):**
   - `cache.py::_evict_until_under` now consults `_memory_pressure_exceeds` comparing
     `memory_guard.current_usage_bytes` (= max(phys_footprint, MLX active) — only what this
     process holds) against byte ceilings `eviction_soft_bytes` / `eviction_hard_bytes`,
     interpolating the legacy fractional threshold between soft (before‑prefill) and hard
     (post‑add). Distributed mode aggregates usage+ceilings across the `mx.distributed` Group.
     Falls back to `get_memory_used_percentage` when ceilings unresolvable.
   - `memory_guard.py` (+34): `eviction_hard_bytes()` / `eviction_soft_bytes()` resolve the
     ceilings independently of the preflight‑admission toggle (`EXO_ENABLE_PREFILL_GUARD`) —
     eviction and admission are separate concerns; guard‑off default mustn't disable correct
     eviction.
   - 4 new model‑free regression tests (`TestEvictionPressureModel`); existing eviction tests
     migrated to the byte model.
   *(commit `42762e44`; DMG `cache-efficiency-dev2`; root‑cause doc `ExoCacheEfficiencyEvictionFix.md`)*
2. **HTTP 400 mapping for `PrefillMemoryExceededError` (COMMITTED, kept):**
   - `src/exo/shared/types/chunks.py` (+10): `ErrorChunk.error_code: int = 500` +
     `error_type: str = "InternalServerError"` (defaults preserve historic behavior).
   - `src/exo/worker/engines/mlx/exceptions.py` (+36): `http_error_status_for` /
     `http_error_type_for` (`PrefillMemoryExceededError` → 400 / `PrefillMemoryExceeded`);
     both runner `_send_error` methods (Sequential + BatchGenerator in
     `llm_inference/batch_generator.py` +14) populate them.
   - OpenAI `/v1/chat/completions` (`chat_completions.py` +50): streaming error event carries
     chunk code/type; non‑stream returns the response object directly (not `StreamingResponse`)
     so an `ErrorChunk` raises `HTTPException(code)` *before* headers commit; bench path uses
     `chunk.error_code or 500`.
   - Claude `/v1/messages` (`claude.py` +79): streaming now emits a proper Anthropic
     `event: error` (previously broke + emitted an empty message); non‑stream returns object
     directly with clean `HTTPException(code)`. `claude_api.py` (+21): `ClaudeErrorEvent` /
     `ClaudeErrorBody` + status→Anthropic‑error‑type mapper (`invalid_request_error` for 4xx).
   - `main.py`: non‑stream chat + Claude routes return objects directly.
   - 9 tests `test_error_mapping.py` (+132) + `test_chat_completions_stream.py` (+83).
   *(commit `c8f1895e`; DMG `ssd-tier-dev1`)*
3. **⚠️ Preflight admission removed from the batch path (UNCOMMITTED REVERT):**
   - baseline (and the committed post‑baseline version) called
     `kv_prefix_cache.evict_for_prefill_headroom()` + `preflight_or_raise(...)` before every
     prefill in `ExoBatchGenerator.submit`.
   - **present local:** both calls **removed** from `batch_generate.py` (they were extra TP
     collectives that desynchronized ranks — see exo 06). `evict_for_prefill_headroom` /
     `preflight_or_raise` / `estimate_prefill_peak_bytes` / `raise_if_prefill_exceeds` still
     exist in `cache.py` and are tested (`test_prefill_memory_headroom.py`), but are not
     invoked on the request path.
4. **Adaptive prefill step (UNCOMMITTED)** — see exo 06 #7; the context‑aware step shrink is
   also a memory‑headroom measure (caps peak attention scores).
5. **NOT in present local:** dashboard max‑context‑length control (#2241) UI wiring
   (backend `EXO_PLACEMENT_CONTEXT_TOKENS` lever exists); MLA‑precise KV estimation
   (over‑counts MLA, safe but may false‑reject); cluster validation of the reclaim ceiling
   with `EXO_MEMORY_GUARD_TIER=aggressive`.

---

## Tasks with no implementation change between baseline and present local

These upstream tasks have design docs only in **both** the baseline and the present local
code — no code delta:

| # | Task | Status (both) |
|---|------|---------------|
| exo 01 | Linux CUDA support | ❌ not started |
| exo 02 | Ring Attention for MLX | ❌ not started |
| exo 03 | Speculative decoding (Drafter + MTP + DFlash) | ❌ not started |
| exo 04 | Arbitrary tensor‑parallel splits | ❌ not started |
| exo 05 | Bandwidth/latency‑aware placement | ❌ not started |
| exo 08 | Model support breadth (catalog + GGUF + embeddings) | ❌ not started |
| exo 09 | P2P / Thunderbolt model distribution | ❌ not started |
| exo 12 | GPU offload for prompt processing | ❌ not started |

---

# Part C — Cross‑cutting / non‑task deltas

These don't map cleanly to a single porting task but are real differences in the present
local tree:

1. **glm52 tool‑parser formatting (UNCOMMITTED)** — `tool_parsers/glm52.py` and
   `test_glm52_tool_parser.py`: pure formatting (line‑wrapping the `_ARG_KV_PATTERN` regex
   and the test strings) + trailing‑newline fixes. No behavioral change. (The parser itself
   landed pre‑baseline in the GLM‑5.2 work.)
2. **`app/EXO/EXO.xcodeproj/project.pbxproj` (+4/−…)** — macOS app project bump (version
   string for the new builds).
3. **Docs added (COMMITTED):**
   - `docs/porting-status-and-synthesis.md` (+522, new) — master status + value map across
     both porting tracks.
   - `docs/missing-column-work-DONE.md` (+220, new) — DONE log for omlx 01 SSD tier · exo 07
     hardening · exo 11 HTTP 400.
   - `docs/gra/CURRENT-STATE-GLM-5.2-RDMA.md` (+145, new) — single‑source verified cluster state.
   - `docs/gra/GLM-5.2-FIX-SUMMARY.md` (+78, new) — full fix summary incl. post‑dev9 Fixes A–E.
   - `ExoCacheEfficiencyEvictionFix.md` (+164, new) + `ExoCacheEfficiencyImpl.md` (+144, new)
     — cache‑efficiency design + root‑cause record.
   - `docs/omlx-porting/01-tiered-kv-cache-ssd-DONE.md` / `01-…-ssd.md` updated for Phases 2–3.
   - `docs/omlx-porting/02-native-metal-kernels-DONE.md` updated for Issue #4.

---

# Part D — Summary matrix (present local vs `turboquant-tiered-dev4`)

Legend: **NEW** = net‑added in present local · **REV** = added post‑baseline then reverted
(helper/tests retained) · **KEEP** = added post‑baseline and retained · **—** = no change.

| Task | Delta vs baseline | Key artifacts in present local |
|------|-------------------|--------------------------------|
| omlx 01 Tiered KV cache | **NEW** (SSD tier + prefix restore) | `ssd_cache.py` (+741), `cache.py` spill/restore/`restore_prefix`, `builder.py` wiring; 23 SSD tests |
| omlx 02 Native Metal kernels | **NEW** (default‑off + clear‑cache + warmup restore) | `kernels.py` opt‑in, `utils_mlx.py` warmup, `opt_batch_gen.py` clear‑cache/128; Issue #4 doc |
| omlx 03–09 (excl. 02) | — | design only |
| exo 06 GLM‑5.2 long‑context/RDMA | **NEW** + **REV** | watchdog + UI override + adaptive prefill (**NEW**); Fixes A–E committed then request‑path reverted (**REV**) |
| exo 07 TurboQuant KV‑cache | **KEEP** (#1990/#2261 hardening) | `make_kv_cache(force_plain=)`, `_chained` guard; 8 hardening tests (⚠️ `force_plain` not on batch path post‑revert) |
| exo 10 Observability | **NEW** (cache‑efficiency cards + watchdog UI) | `cached_tokens` event/state/apply + dashboard 3‑cell grid; stall‑watchdog radio |
| exo 11 Memory headroom | **KEEP** + **REV** | reclaim eviction + HTTP 400 mapping (**KEEP**); preflight admission removed from batch path (**REV**) |
| exo 01,02,03,04,05,08,09,12 | — | design only |

---

## Reproduction / verification commands

```bash
# What the baseline DMG maps to
git show --no-patch v1.0.72-turboquant-tiered-dev4

# Full delta (committed + uncommitted) vs the baseline
git diff --stat v1.0.72-turboquant-tiered-dev4

# Committed‑only delta (tag..HEAD)
git log --oneline v1.0.72-turboquant-tiered-dev4..HEAD
git diff --stat v1.0.72-turboquant-tiered-dev4..HEAD

# Uncommitted working‑tree delta on top of HEAD
git diff --stat HEAD
```

**Bottom line:** vs `EXO-1.0.72-turboquant-tiered-dev4.dmg`, the present local code adds the
SSD cold tier + prefix restore (omlx 01), flips native kernels to opt‑in + restores JACCL
warmup + adds periodic clear‑cache (omlx 02), adds the decode stall watchdog + a per‑request
UI override + adaptive prefill step (exo 06), lands TurboQuant #1990/#2261 hardening (exo 07),
adds per‑instance cache‑efficiency stat cards (exo 10), and lands reclaim‑based eviction +
HTTP 400 mapping (exo 11) — while **reverting** the five post‑baseline TP‑stall fixes (A–E)
and the batch‑path preflight admission back to the known‑working decode path, because they
introduced extra TP collectives that desynchronized ranks and caused the very `Fence::wait`
hangs they meant to fix.