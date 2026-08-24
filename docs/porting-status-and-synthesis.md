# Porting Status & Synthesis — oMLX + Upstream EXO

**Date:** 2025-08-24
**Scope:** Consolidated reference covering both porting tracks:
- `docs/omlx-porting/*` — features ported from `jundot/omlx`
- `docs/exo-upstream-porting/*` — features from upstream `exo-explore/exo` issues/PRs

This doc is the **master status + value map**. For implementation-grade detail,
read the individual companion docs linked in each section.

---

## How to read this doc

- **"What we gain"** — 2-sentence value statement per task (the user ask).
- **"Implementation status"** — what is actually shipped in this working tree,
  verified against the source (not just what the design doc claims).
- **"Highest importance"** — ranked recommendation for where to invest next,
  tuned to the 2×256 GB cluster running agentic GLM-5.2.
- **"New in oMLX v0.6.x"** — features shipped in oMLX **after** the porting
  docs were written (2025-08-18, against oMLX `c1a3d44`/`1f1aff3`). These are
  **not** in the present task list and are candidates for new tasks.

---

## Summary table — status at a glance

Legend: ✅ shipped · ⚠️ partial / has blocker · ❌ not started · 🆕 new candidate

### oMLX → EXO (`docs/omlx-porting/`)

| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 01 | Tiered KV cache (RAM+SSD+persist) | ⚠️ | Settings/flag layer + dashboard shipped; paged-block manager + SSD spill + restart-recovery pending |
| 02 | Native Metal kernels (GLM-5.2) | ✅ | Vendored + packaged + lazy-import fixed; JACCL rendezvous race **fixed** (master lifecycle gating + warmup removal); prefill OOM guarded; decode `Fence::wait` mitigated by stall watchdog |
| 03 | Multi-Token Prediction (MTP) | ❌ | Design only; reconcile with exo #03 |
| 04 | SpecPrefill (sparse prefill) | ❌ | Design only |
| 05 | EnginePool + Model Profiles | ❌ | Design only (profiles half is cheap, no refactor needed) |
| 06 | GLM adaptive prefill patch | ❌ | Design only; small, unblocks native-kernel fast path |
| 07 | Embedding + Reranker engines | ❌ | Design only |
| 08 | Boundary snapshot SSD offload | ❌ | Design only; decoupled from #01 |
| 09 | Claude Code context-scaling + SSE keep-alive | ❌ | Design only; tiny effort, high agentic value |

### Upstream → EXO (`docs/exo-upstream-porting/`)

| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 01 | Linux CUDA support | ❌ | Very high effort; reconcile 4 in-flight PRs |
| 02 | Ring Attention for MLX | ❌ | Production PR #2255 exists; needs full-model-per-node |
| 03 | Speculative decoding (Drafter + MTP + DFlash) | ❌ | 3.3–4.3× decode; umbrella for oMLX #03 |
| 04 | Arbitrary TP splits | ❌ | Unblocks 3-node + quantized TP |
| 05 | Bandwidth/latency-aware placement | ❌ | Good-first-issue scope; immediate TPS win |
| 06 | GLM-5.2 long-context + RDMA/TB reliability | ❌ | The flagship-config pain cluster |
| 07 | TurboQuant KV-cache | ⚠️ | Settings + integer fallback shipped; native kernel + hardening pending |
| 08 | Model support breadth (catalog + GGUF + embeddings) | ❌ | Overlaps oMLX #07 |
| 09 | P2P / Thunderbolt model distribution | ❌ | One-node-downloads-others-pull |
| 10 | Observability (Prometheus /metrics) | ❌ | No metrics endpoint today |
| 11 | Memory headroom before prefill + near-limit placement | ✅ | Phases 1–3 full; cluster validation + UI wiring pending |
| 12 | GPU offload for prompt processing | ❌ | Overlaps #01 + #02 |

### 🆕 New candidates from oMLX v0.6.x (not in present tasks)

| Candidate | oMLX release | Why notable |
|-----------|-------------|-------------|
| Decode-fairness during concurrent prefill | v0.6.0 (#2633) | **1.6×–43× decode** during concurrent prefill; default-on; not in any task |
| SSD-backed prompt reuse for distributed ranks | v0.6.0 (#2620) | Extends tiered cache into disaggregated/remote-prefill path |
| Qwen ANE prefill + split tuner | v0.6.1→v0.6.3rc2 | +18–36% prefill on Qwen via Apple Neural Engine |
| Linear `CacheList` prefix storage | v0.6.0 (#2550) | Quadratic→linear (282 GB→15 GB at 84–94k tokens) |
| GDN recurrent-state bounded SSD sidecars | v0.6.0 (#2569/#2644) | 1.93× recurrent-state storage reduction |
| Heterogeneous Metal+CUDA model pools | v0.6.0 (#2591) | Reference impl for exo #01/#12 |
| DFlash 2 | v0.6.3rc1 | Extends exo #03/omlx #03 speculative decoding |

---

## Part 1 — "What we gain" per task (2 sentences each)

### oMLX → EXO

#### 01 — Tiered KV cache (RAM+SSD+persist)
We gain persistence of the long agentic system-prompt KV across restarts and
LRU eviction, so Claude Code/Pi/Codex re-sent contexts skip re-prefill — the
explicit reason oMLX exists. Only the runtime toggle, observability gauge, and
`make_kv_cache` seam ship today; the paged-block manager, safetensors SSD
spill/restore, and restart-recovery scan are still pending behind the flags.

#### 02 — Native Metal kernels (GLM-5.2)
We gain ~1.4–2× (up to ~30× on the slow generic path) prefill on the GLM-5.2
model EXO already vendors, finishing the highest-leverage line of the existing
GLM investment. Vendored + packaged + import-ordering-fixed; the one remaining
blocker is a pre-existing JACCL `all_sum` rendezvous race that hangs rank 1 on
2-node load.

#### 03 — Multi-Token Prediction (MTP)
We gain ~1.3–1.74× decode throughput on Qwen3.5/3.6 and DeepSeek-V4 via a
draft/verify loop folded into the existing batched generator without disturbing
the cache stack. It's the biggest single decode-speedup available and must be
reconciled with the upstream Drafter/DFlash design (exo #03).

#### 04 — SpecPrefill (sparse prefill)
We gain lower TTFT on long prompts by using a small draft model to score token
importance and prefilling only the top-K% with manual RoPE, complementing EXO's
remote-prefill as a local fast path. It's a self-contained patch that doesn't
touch sharding or the cache layers.

#### 05 — EnginePool + Model Profiles
We gain a per-node hot multi-model LRU pool (pin/TTL/memory enforcer) so one
node serves several models concurrently, plus zero-cost `<model>:<profile>`
aliases served from a single loaded model. The profiles half alone gives
dashboard-friendly variants like `qwen3-8b:thinking` with no extra memory and
no pool refactor.

#### 06 — GLM adaptive prefill patch
We gain the one GLM patch EXO deliberately skipped, which adapts prefill
chunking so GLM's DSA sparse path engages correctly at the 2048-token boundary
— unlocking the already-shipped native-kernel fast path for long contexts.
Small, low-effort, and pairs naturally with native kernels.

#### 07 — Embedding + Reranker engines
We gain `/v1/embeddings` and `/v1/rerank` endpoints (BGE-M3, ModernBERT,
XLM-RoBERTa) so EXO serves retrieval models alongside LLMs/VLMs for
RAG-over-EXO. It fills the only engine-type gap (today just `image` + `mlx`
engines exist).

#### 08 — Boundary snapshot SSD offload during prefill
We gain flattened prefill memory spikes for SSM/GatedDeltaNet models by
offloading non-sliceable cache snapshots to SSD mid-prefill and reloading them
block-by-block at completion. It's decoupled from the full paged-SSD cache and
can land independently.

#### 09 — Claude Code context-scaling + SSE keep-alive
We gain correct auto-compact timing for small-context models (scaled reported
token counts) plus SSE keep-alive comments that stop Claude Code from
disconnecting during long prefills. Tiny effort, but high-value for the
agentic workload that is EXO's primary use case.

### Upstream → EXO

#### 01 — Linux CUDA support
We gain the entire non-Mac GPU market (NVIDIA, Framework Desktop, DGX Spark) —
the highest-reaction issue in the repo (56👍). Very high effort (reconcile 4
in-flight PRs + a backend abstraction) but the biggest *reach* win.

#### 02 — Ring Attention for MLX
We gain sequence-parallel prefill that scales TTFT down with ring size,
directly solving the #2208/#781 long-prompt pain that layer-sharding cannot.
Production PR #2255 exists with 33 tests/benchmarks, pure MLX, no CUDA
dependency — but it replicates weights, so each node must hold the full model.

#### 03 — Speculative decoding (Drafter + MTP + DFlash)
We gain 3.3–4.3× decode TPS measured on Qwen 3.5/3.6 via a general drafter
abstraction supporting DFlash + MTP + multi-device coupled drafters. This is
the umbrella design that should absorb oMLX MTP (omlx #03) into one
implementation.

#### 04 — Arbitrary tensor-parallel splits
We gain the ability to shard across 3-node and quantized (group-size-32)
models, which today's `shard_linear`/`shard_inplace` divisibility requirement
blocks. Smart padding unblocks heterogeneous and quantized tensor parallelism.

#### 05 — Bandwidth/latency-aware placement
We gain a placement algorithm that minimizes `Σ(M·Nᵢ/N / Bᵢ)+Lᵢ` instead of
memory-proportional-only, giving an immediate TPS win on every existing cluster
in the memory-bound regime. Good-first-issue scope, low–medium effort.

#### 06 — GLM-5.2 long-context fix + RDMA/TB reliability
We gain a fix for the flagship 2-node sharded config that hangs or garbles on
~45–52k-token agentic prompts (a `mlx Fence::wait` GPU-timeout race) plus
multi-rail/bond RDMA hardening across the whole Thunderbolt reliability
cluster. This is the largest pain cluster and the sibling of the
native-kernel blocker.

#### 07 — TurboQuant KV-cache
We gain per-model, opt-in KV-cache compression (2–8 bit, skip-last) that cuts
KV memory and spills smaller to the tiered SSD cache. Only the runtime toggle
+ integer `QuantizedKVCache` fallback + dashboard ship; the native fast-path
kernel and correctness hardening (#1990, #2261) are pending.

#### 08 — Model support breadth (catalog + GGUF + embeddings)
We gain a unified catalog across HF/LM Studio/Ollama/llama.cpp with
content-based resolution, GGUF via a llama.cpp runner, and embeddings —
consolidating model discovery. Closes the embeddings overlap with oMLX doc 07.

#### 09 — P2P / Thunderbolt model distribution
We gain one-node-downloads-others-pull-over-TB/LAN so every node stops
re-downloading multi-hundred-GB models from HF. Directly addresses #721/#1257
bandwidth and storage pain.

#### 10 — Observability (Prometheus /metrics + cluster stats)
We gain a `/metrics` Prometheus endpoint plus global cluster stats (zenoh Last
Value) for ops — none exist today. Enables real monitoring of the distributed
cluster.

#### 11 — Memory headroom before prefill + near-limit placement
We gain evict-before-prefill admission + per-chunk EWMA transient guard +
placement-time KV/activation headroom so prefix-cache allocations can't starve
prefill and OOM. Fully shipped (reclaim-based ceiling port, ship-default-off,
runtime toggle); pending only cluster validation, UI max-context wiring, and
HTTP 400 mapping.

#### 12 — GPU offload for prompt processing
We gain a heterogeneous prefill-offload pattern where an NVIDIA GPU does
prefill and a Mac does decode. It overlaps CUDA (#01) + Ring Attention (#02)
as the long-prompt-TTFT special case for mixed clusters.

---

## Part 2 — Implementation status (verified in tree)

Verified against the working tree on 2025-08-24, not just the design docs.

### ✅ omlx 01 — Tiered KV cache (settings layer)
**Shipped:**
- `src/exo/worker/engines/mlx/turboquant.py` — runtime settings layer
  (`is_tiered_cache_enabled`, `hot_cache_only`, `ssd_cache_dir`,
  `ssd_cache_max_size`, `hot_cache_max_size`, `tiered_cache_status`,
  `clear_ssd_cache`); env defaults + runtime overrides.
- `src/exo/api/main.py` — `PUT/GET/DELETE /v1/tiered-cache`;
  `tieredKvCache` in `/v1/feature-flags`.
- `src/exo/api/types/api.py` — `TieredCacheSetting` body model.
- Dashboard: On/Off toggle + SSD dir/cap/RAM-cap inputs + live observability
  gauge (used/total, file count, clear button).

**Missing (Phases 1–3):**
- `KVCacheBlock`, `FreeKVCacheBlockQueue` (O(1) LRU), chain hashing, COW.
- `PagedSSDBlockMetadata`, `PagedSSDCacheIndex`, safetensors spill/restore.
- Startup scan + cache-signature validation on model swap.
- Phase 4 = boundary snapshot offload (omlx #08).

**Evidence:** no `PagedSSD` / `FreeKVCacheBlockQueue` / `KVCacheBlock`
symbols exist in `src/exo/worker/engines/mlx/` yet.

### ⚠️ omlx 02 — Native Metal kernels (built + blocked)
**Shipped:**
- `mlx_kernels/` — `build_kernels.py` (CMake), `OMLX_COMMIT.txt` (pinned
  `1f1aff3…`), `parts.nix`, `README.md`.
- `src/exo/worker/engines/mlx/vendor/omlx_custom_kernels/` — vendored oMLX
  tree (`glm_moe_dsa/`, `common/`).
- `src/exo/worker/engines/mlx/vendor/glm_moe_dsa/kernels.py` — `_FastDispatch`
  with **lazy `_ext` import** (defers Metal init past JACCL init).
- `packaging/pyinstaller/exo.spec` — dual metallib placement
  (package dir + `_internal/`).
- `EXO_NATIVE_GLM_KERNELS=0` kill-switch; `EXO_BUILD_MLX_KERNELS=1` build opt-in.
- `test_glm_moe_dsa_native_kernels.py` — 9 pass, 2 skip.

**Missing/blocker:**
- ✅ **Issue #3 — JACCL `all_sum` rendezvous race** — **FIXED.** The in-process
  warmup was removed (commit `17252452`) and rendezvous is enforced by the
  master's lifecycle gating in `plan.py` (rank 0 not sent `LoadModel` until
  rank 1 reaches `RunnerConnected`). The RDMA data path is validated by
  `_probe_rdma_interface` in a subprocess. Verified on the live cluster:
  36k-token prefill succeeds, cluster serves. See
  `omlx-porting/02-native-metal-kernels-DONE.md` Issue #3.
- ⚠️ **`Fence::wait` decode race** (exo #06 / #2208) — **mitigated, not
  eliminated.** Root cause is in the mlx fork (`rltakashige/mlx-jaccl-fix-small-recv`);
  the collective blocks the C++ thread so no Python-level timeout can interrupt
  it. A decode stall watchdog (`EXO_DECODE_STALL_TIMEOUT` in `_token_chunk_stream`)
  bounds the hang from the API/master side (separate process) so the client
  fails fast and can retry. Definitive fix needs a newer mlx-jaccl commit
  (past `e9835615`, which deadlocks) + cluster validation — outside this repo.

### ✅ exo 07 — TurboQuant KV-cache (settings + integer fallback)
**Shipped:**
- `src/exo/worker/engines/mlx/turboquant.py` — TurboQuant settings
  (`is_turboquant_enabled`, `turboquant_bits`, `turboquant_skip_last`,
  `effective_kv_bits` with half-step rounding).
- `src/exo/worker/engines/mlx/cache.py::make_kv_cache` — consults TurboQuant
  (overrides legacy `KV_CACHE_BITS`), applies skip-last (final layer full
  precision), defers to `model.make_cache()` for Qwen3-Next hybrid caches;
  `_KV_BYTES_PER_ELEMENT` updated for the prefill estimator.
- `PUT /v1/turboquant` route; `turboquantKv` in feature flags.
- Dashboard: On/Off + bits `<select>` (2–8) + skip-last checkbox.
- 40 tests in `test_turboquant.py`.

**Missing:**
- Phase 1 correctness hardening (#1990, #2261) — skip KV-quant in
  single-node BatchGenerator; clean prefill after chained prefix-cache
  extensions.
- Phase 3 native fast-path kernel (oMLX `turboquant_attention.py`); today
  half-step depths (2.5/3.5) round down to integer `QuantizedKVCache`.
- Phase 4 PDD cache handoff for quantized KV in disaggregated prefill.

### ✅ exo 11 — Memory headroom + near-limit placement (Phases 1–3)
**Shipped:**
- `src/exo/worker/engines/mlx/memory_guard.py` — reclaim-based ceiling port
  of oMLX `ProcessMemoryEnforcer` (`min(static, dynamic, metal_cap)`),
  honours `iogpu.wired_limit_mb` + `phys_footprint`; tiers
  (safe/balanced/aggressive); runtime toggle; ship-default-off.
- `src/exo/worker/engines/mlx/prefill_transient_tracker.py` — EWMA,
  outlier rejection, observed-max clamp.
- `src/exo/worker/engines/mlx/exceptions.py` — `PrefillMemoryExceededError`.
- `src/exo/worker/engines/mlx/cache.py` — `evict_for_prefill_headroom`,
  `preflight_or_raise`, `guard_prefill_chunk_or_raise`,
  `estimate_prefill_peak_bytes`, `raise_if_prefill_exceeds`.
- `src/exo/master/placement_memory.py` — `estimate_kv_bytes`,
  `estimate_activation_margin_bytes`, `estimate_node_memory_requirement`.
- `src/exo/master/placement_utils.py` — uses `estimate_node_memory_requirement`;
  `EXO_PLACEMENT_CONTEXT_TOKENS` lever.
- `GET /v1/version`, `PUT /v1/memory-guard`, `prefillMemoryGuard` flag.
- Dashboard: Memory Guard toggle in Advanced Options; build-version badge.
- 69 tests (42 prefill-headroom + 12 memory-guard + 15 placement-memory).

**Missing:**
- ⚠️ Dashboard max-context-length control (#2241) — backend lever exists,
  UI wiring separate.
- ⚠️ MLA-precise KV estimation (currently over-counts MLA; safe but may
  false-reject placement).
- ⚠️ HTTP 400 mapping for `PrefillMemoryExceededError` (currently 500).
- ⚠️ Cluster validation of the reclaim ceiling with
  `EXO_MEMORY_GUARD_TIER=aggressive`.

### Everything else — not started
omlx #03, #04, #05, #06, #07, #08, #09 and exo #01, #02, #03, #04, #05, #06,
#08, #09, #10, #12 have design docs but **no implementation** in the tree.

---

## Part 3 — Highest importance / where we gain the most

Tuned to the flagged config: **2×256 GB cluster, agentic GLM-5.2**.

### 🥇 #1 — Unblock the flagship (exo #06 + native-kernel Issue #3)
The 2-node sharded GLM-5.2 path — the entire point of the native-kernel and
GLM-port investment — was broken on real ~45–52k-token agentic prompts. Two
distinct faults were identified; both are now addressed/mitigated:

1. **JACCL `all_sum` rendezvous race** (`utils_mlx.py`) — **FIXED.** The
   interim fix noted in the TurboQuant-DONE log ("removed the in-process
   warmup entirely") **still holds** and has been formalized: the master's
   lifecycle gating in `plan.py` enforces rank rendezvous before model-load
   collectives. Verified on the live 2-node M3 Ultra cluster (build
   `1.0.72-cache-efficiency-dev2`, commit `42762e44`): a 35,969-token prefill
   completed successfully and the cluster is serving.
2. **`mlx Fence::wait` GPU-timeout race** (exo #06 / #2208) — traces to mlx
   itself, not exo's Python. Candidate fix in `rltakashige/mlx-jaccl-fix-small-recv`
   `address-rdma-gpu-locks` (commit `e9835615`) but it introduces its own
   deadlock. **Mitigation in place:** a decode stall watchdog
   (`EXO_DECODE_STALL_TIMEOUT` / `EXO_PREFILL_STALL_TIMEOUT`) bounds the hang
   from the API/master side so requests fail fast instead of hanging forever.
   Plus multi-rail/bond RDMA hardening (#2160, #2195, #2063) still pending.

**Why first:** nothing else matters until the core config actually loads and
serves long contexts. This is "make the thing we already paid for work," not
additive gain.

### 🥇 #2 — Finish the tiered-SSD plumbing (omlx #01, Phases 1–3)
The control surface is already shipped; completing the paged-block manager +
safetensors spill/restore + restart-recovery is what makes restarts and LRU
evictions stop re-prefilling 50k-token system prompts. For the agentic
re-send-the-same-context workload this is the dominant latency — oMLX's stated
raison d'être — and the single highest-leverage finish line on existing
scaffolding (low remaining integration risk: flags + UI + `make_kv_cache` seam
all done).

### 🥇 #3 — Speculative decoding (exo #03 / omlx #03)
3.3–4.3× decode TPS, self-contained PR #2079, composes with TurboQuant
(settings layer already shipped). If decode speed rather than TTFT is the goal,
this wins. Reconcile oMLX MTP + upstream Drafter/DFlash into one design first.

### Why not Ring Attention (#02) for this cluster?
Ring Attention replicates weights per node — each node must hold the full
model. On 2×256 GB it **cannot fit the 395 GB GLM-5.2**; it only helps
smaller (Llama/Qwen-27B) models. It's the right answer for the long-prompt
TTFT pain in general, but not the highest-leverage move for *this specific*
flagship config right now.

### Single pick for "gain the most"
**Finish the tiered-SSD KV plumbing** — settings layer done (low risk),
directly serves the primary agentic workload, and it's the feature oMLX was
built around.

---

## Part 4 — New in oMLX v0.6.x NOT covered by the present tasks

The porting docs were written 2025-08-18 against oMLX `c1a3d44`/`1f1aff3`.
oMLX v0.6.0 → v0.6.3rc2 shipped afterward. Notable features **missing from the
current task list**:

### 🆕 1 — Decode-fairness during concurrent prefill (v0.6.0, #2633) ⭐ add as Tier-1
Prefill yields GPU time to active decodes and adapts chunk size to a target
stall time. **1.6×–43× decode improvement** during concurrent prefill,
enabled by default. EXO's continuous-batching + prefill interleaving would
benefit directly.

> | decoding + prefilling | solo | before | after |
> |---|---:|---:|---:|
> | Qwen3.6-27B + Qwen3.5-0.8B prefilling 21k | 50 tok/s | 11–18 | 23–24 |
> | Qwen3.5-0.8B + Qwen3.6-27B prefilling 8.4k | 203 tok/s | 1.7–1.8 | 76–78 |
> | DeepSeek-V4-Flash + Qwen3.6-27B prefilling 8.4k | 34 tok/s | 0.3 | 11–12 |

**Recommendation:** add as a new Tier-1 task. It's a large, default-on serving
win that the current docs predate entirely, and it's independent of the
sharding/CUDA work so it can land on the existing 2-node cluster.

### 🆕 2 — SSD-backed prompt reuse for distributed ranks (v0.6.0, #2620)
Each rank stores a chain of cache-boundary snapshots and restores the longest
prefix held by *every* rank when the in-memory prompt cache misses. Incremental
KV segments keep storage linear: a 12k-token GLM-5.2 chain used 1.17 GB instead
of ~6.5 GB with cumulative copies.

Extends the shipped tiered-cache work (omlx #01) straight into EXO's
disaggregated/remote-prefill path — high leverage on existing investment.

### 🆕 3 — Qwen ANE prefill + split tuner (v0.6.1 → v0.6.3rc2)
Experimental Apple Neural Engine prefill for Qwen3.5/3.6/3.8:
- **+18.9% prefill at 32K** (dual-ANE/GPU, M3 Ultra, v0.6.1 #2756).
- Built-in ANE/GPU split tuner (v0.6.2 #2814) — benchmarks on each Mac
  instead of using ratios tuned elsewhere.
- Optional CPU sharing (v0.6.3rc2 #2892) — +33–36% prefill on M3 Ultra.
- Wider quantization support (Q5/Q6/Q8, v0.6.3rc1 #2833/#2889).

Fits EXO's MLX-first ethos. **Caveats:** uses private Apple runtime interfaces
(risk), decode-neutral (prompt-processing optimization only), increases peak
memory ~4–7 GB and model load time. Worth a new task if Qwen-on-Apple-Silicon
is a target.

### 🆕 4 — Linear `CacheList` prefix storage (v0.6.0, #2550)
Mixed hybrid-cache prefix storage went quadratic→linear. On the affected
Inkling workload: 282.7 GB at 84k tokens → 15.3 GB at 94k tokens, with
byte-identical restores. Distinct memory-correctness win adjacent to
omlx #01/#08; relevant once the SSD plumbing lands for hybrid-cache models.

### 🆕 5 — GDN recurrent-state bounded SSD sidecars (v0.6.0, #2569/#2644)
Recurrent state now uses bounded SSD sidecars by default, kept separate from
ordinary KV storage. RHT-INT16 reduces storage 1.93× vs FP32 while restoring
in FP32. Directly extends omlx #08 (boundary snapshot offload) for
GDN/GatedDeltaNet models.

### 🆕 6 — Heterogeneous Metal + CUDA model pools (v0.6.0, #2591)
Apple Silicon + NVIDIA CUDA workers contribute to one logical model-memory
pool with memory-aware contiguous-layer placement, ConnectX discovery, and
NCCL verification. The current path keeps every physical worker in the outer
MLX Ring; the hierarchical Ring-to-NCCL gateway is future work.

oMLX now has a **real reference implementation** for exo #01/#12 to adapt
rather than designing from scratch.

### 🆕 7 — DFlash 2 (v0.6.3rc1)
End-to-end DFlash 2 speculative decoding: checkpoint-derived sliding windows,
per-model block-size overrides, matches the regular batched engine's sampling
(min_p, seeded requests), default sink size 0, SSD prefix-cache writes
functional. Versioned `DFlash2` checkpoints appear in the draft-model picker.

Extends exo #03 / omlx #03; fold into the single speculative-decoding design
rather than treating as a separate task.

### Supporting fixes worth noting if porting adjacent features
- **TurboQuant-KV + Lightning-MTP verification crash** (v0.6.2, #2782) —
  needed if you ship MTP + TurboQuant together (the optimized Qwen
  verification-attention path now safely rejects TurboQuant proxy objects).
- **NAX GPU suffix kernels for M5-family** (v0.6.2) — relevant to omlx #02 on
  M5 hardware; `OMLX_QWEN35_QMM_NAX=0` kill-switch.
- **M5 MLX kernel bug self-test + reroute** (v0.5.2rc2, #2267) — quantized
  MoE matmuls hit a defective GPU kernel on M5; oMLX detects at startup and
  reroutes to known-good kernels.

### New models since the docs (feed exo #08)
DeepSeek-V4-Flash-0731 (with DSpark Lightning MTP, +85.6% code decode),
Thinking Machines Inkling Small, Poolside Laguna S-2.1, Xiaomi MiMo V2.5,
Step-3.7-Flash, Ling 3.0 Flash, Jina Reranker v3.5, Baidu Unlimited-OCR,
DiffusionGemma, MiniMax M3.

---

## Appendix — verification commands used

```bash
# Confirm shipped artifacts exist
ls src/exo/worker/engines/mlx/turboquant.py
ls src/exo/worker/engines/mlx/memory_guard.py
ls src/exo/master/placement_memory.py
ls mlx_kernels/
ls src/exo/worker/engines/mlx/vendor/omlx_custom_kernels/

# Confirm the paged-SSD plumbing is NOT yet present (Phase 1–3 of omlx #01)
grep -rl "PagedSSD\|paged_block\|FreeKVCacheBlockQueue\|KVCacheBlock" \
  src/exo/worker/engines/mlx/   # → no matches

# Confirm API feature-flag routes
grep -n "feature-flags\|turboquant\|tiered-cache\|memory-guard\|/v1/version" \
  src/exo/api/main.py

# Confirm dashboard toggles
grep -n "TurboQuant\|Tiered KV\|Memory Guard" \
  dashboard/src/routes/+page.svelte

# Latest oMLX releases (v0.6.x not in the docs)
curl -s https://api.github.com/repos/jundot/omlx/releases
```

---

## Cross-references

- oMLX porting master index: [`omlx-porting/README.md`](omlx-porting/README.md)
- Upstream porting master index: [`exo-upstream-porting/README.md`](exo-upstream-porting/README.md)
- Tiered KV cache DONE log: [`omlx-porting/01-tiered-kv-cache-ssd-DONE.md`](omlx-porting/01-tiered-kv-cache-ssd-DONE.md)
- Native kernels DONE log: [`omlx-porting/02-native-metal-kernels-DONE.md`](omlx-porting/02-native-metal-kernels-DONE.md)
- TurboQuant DONE log: [`exo-upstream-porting/07-turboquant-kv-cache-DONE.md`](exo-upstream-porting/07-turboquant-kv-cache-DONE.md)
- Memory headroom DONE log: [`exo-upstream-porting/11-memory-headroom-prefill-DONE.md`](exo-upstream-porting/11-memory-headroom-prefill-DONE.md)