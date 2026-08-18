# oMLX → EXO Feature Porting Analysis

**Source repo analyzed:** `jundot/omlx` (commit at time of analysis, cloned to `/tmp/omlx-research`)
**Target repo:** EXO (`/Users/gra/Gra/Rt/Src/exo`)
**Date:** 2025-08-18

This document is the **master index**. It summarizes the full comparison between
oMLX and EXO's MLX stack, ranks the features worth porting by impact and effort,
and links to a dedicated implementation doc for each.

The companion docs (one per feature) live alongside this file and are written to
be **implementation-grade**: they cite exact files in both repos, describe the
oMLX design, EXO's current state, the integration seam, and a concrete phased
plan.

---

## How EXO already overlaps with oMLX (do NOT re-port)

EXO has already ported or independently built the following — these are **not**
gaps and are explicitly excluded from the porting scope:

| Area | EXO location | oMLX equivalent | Notes |
|------|--------------|-----------------|-------|
| GLM-5.2 model code (faithful port) | `src/exo/worker/engines/mlx/vendor/glm_moe_dsa/` | `omlx/patches/glm_moe_dsa/` | Verified cosmetic-only diff (see `../gra/GLM-5.2-RESEARCH-RESULTS.md`) |
| Continuous batching | `src/exo/worker/engines/mlx/generator/batch_generate.py` | `omlx/engine/batched.py` | Both use mlx-lm `BatchGenerator` |
| In-RAM prefix KV cache (with LRU) | `src/exo/worker/engines/mlx/cache.py::KVPrefixCache` | `omlx/cache/prefix_cache.py` | EXO's is simpler but functional |
| SSM / non-trimmable cache snapshots | `cache.py::snapshot_ssm_states`, `copy_snapshot_entry` | `omlx/cache/boundary_snapshot_store.py` | EXO does in-RAM; oMLX does RAM+SSD |
| VLM with media-region-aware caching | `src/exo/worker/engines/mlx/vision.py` | `omlx/engine/vlm.py` | EXO's prefix-cache media validation is arguably more sophisticated |
| Anthropic Messages API | `src/exo/api/adapters/claude.py` | `omlx/api/adapters/anthropic.py` | Full parity |
| OpenAI Responses API | `src/exo/api/adapters/responses.py` | `omlx/api/responses_models.py` | Full parity |
| Disaggregated / remote prefill | `src/exo/worker/engines/mlx/disaggregated/`, `generator/remote_prefill.py` | (oMLX cluster path differs) | Different transport (zenoh vs RDMA) |
| Distributed inference | zenoh + event sourcing (master/worker) | `omlx/cluster/` (experimental) | EXO's is more mature; skip oMLX's |
| Dashboard | `dashboard/` (Svelte 5) | `omlx/admin/` + macOS menubar app | Different audiences; skip |

---

## Ranked features to port

Each row links to its dedicated implementation doc.

### ⭐⭐⭐ Tier 1 — Biggest wins

| # | Feature | Doc | Why it matters |
|---|---------|-----|----------------|
| 1 | **Tiered KV cache: Hot (RAM) + Cold (SSD) with persistence** | [01-tiered-kv-cache-ssd.md](01-tiered-kv-cache-ssd.md) | EXO's cache is RAM-only and lost on restart. SSD tier with prefix-aware restore survives restarts — transformative for agentic/coding workloads. oMLX's stated raison d'être. |
| 2 | **Native Metal custom kernels (GLM-5.2, MiniMax M3, Qwen3.5)** | [02-native-metal-kernels.md](02-native-metal-kernels.md) | EXO already vendors GLM-5.2 model code but **ships no kernels** — they fall back to slow generic attention (~30× slower prefill per oMLX). EXO's dispatch shim already expects `omlx.custom_kernels...fast`. Highest-leverage finish line for existing GLM investment. |

### ⭐⭐ Tier 2 — Strong, targeted improvements

| # | Feature | Doc | Why it matters |
|---|---------|-----|----------------|
| 3 | **Multi-Token Prediction (MTP) speculative decode** | [03-multi-token-prediction.md](03-multi-token-prediction.md) | Draft/verify loop folded into `GenerationBatch.next()` for Qwen3.5/3.6 + DeepSeek-V4. ~1.3–1.74× decode throughput. Keeps cache stack intact. |
| 4 | **SpecPrefill (sparse prefill)** | [04-specprefill.md](04-specprefill.md) | Draft model scores token importance; target prefills only top-K% with manual RoPE. Cuts TTFT on long prompts. Complements EXO's remote prefill as a local fast path. |
| 5 | **Multi-model EnginePool + Model Profiles** | [05-engine-pool-model-profiles.md](05-engine-pool-model-profiles.md) | Per-node hot multi-model LRU pool + pinning + TTL + memory enforcer; `<model>:<profile>` aliases (e.g. `qwen3-8b:thinking`) served from one loaded model, zero extra memory. |
| 6 | **GLM adaptive prefill patch** | [06-glm-adaptive-prefill.md](06-glm-adaptive-prefill.md) | The one GLM patch EXO deliberately skipped. Revisit now that EXO's GLM port is verified correct — unlocks the native-kernel fast path and the >2048-token boundary. |

### ⭐ Tier 3 — Smaller, still interesting

| # | Feature | Doc | Why it matters |
|---|---------|-----|----------------|
| 7 | **Embedding + Reranker engines** | [07-embedding-reranker.md](07-embedding-reranker.md) | EXO has only `image` + `mlx` engines. Adds `/v1/embeddings` + `/v1/rerank` (BGE-M3, ModernBERT, XLM-RoBERTa) for RAG-over-EXO. |
| 8 | **Boundary snapshot SSD offload during prefill** | [08-boundary-snapshot-offload.md](08-boundary-snapshot-offload.md) | Offload ArraysCache/SSM snapshots to SSD *during* prefill to flatten memory spikes. Can be ported independently of the full paged-SSD cache. |
| 9 | **Claude Code context-scaling + SSE keep-alive** | [09-claude-code-context-scaling.md](09-claude-code-context-scaling.md) | Scale reported token counts so auto-compact triggers correctly for small-context models; SSE keep-alive during long prefill. Small but high-value for the agentic use case. |

---

## Recommended implementation order

Sequenced by (impact × leverage on prior work) ÷ effort:

1. **Native Metal kernels** ([02](02-native-metal-kernels.md)) — packaging/build task; EXO's dispatch shim already calls `omlx.custom_kernels.glm_moe_dsa.fast` if importable. Immediate ~30× prefill win on the model EXO already supports.
2. **GLM adaptive prefill patch** ([06](06-glm-adaptive-prefill.md)) — small, unblocks the kernel fast path at the >2048 boundary. Pairs naturally with #1.
3. **SSD cold tier on `KVPrefixCache`** ([01](01-tiered-kv-cache-ssd.md)) — highest architectural value; EXO already does the hard parts (snapshot/restore, prefix match, media validation).
4. **Model profiles** (the cheap half of [05](05-engine-pool-model-profiles.md)) — dashboard-friendly, no memory cost, no engine-pool refactor required.
5. **MTP** ([03](03-multi-token-prediction.md)) — decode speedup on popular Qwen3.5/DeepSeek-V4 families.
6. **SpecPrefill** ([04](04-specprefill.md)) — local prefill fast path before remote prefill.
7. The Tier 3 items in any order.

---

## Cross-cutting notes

- **Licensing:** oMLX is Apache-2.0. EXO's vendored GLM code already credits it. Continue the same attribution pattern.
- **MLX executor / threading:** oMLX uses a single-thread MLX executor (`get_mlx_executor`) to serialize loads. EXO's `bootstrap.py` patches are applied before `mlx_lm.load()`. Any ported patch must respect EXO's existing `apply_mlx_patches()` ordering in `src/exo/worker/runner/bootstrap.py:75-77`.
- **Cache type generality:** EXO's `cache.py` already handles `KVCache`, `QuantizedKVCache`, `RotatingKVCache`, `ArraysCache`, `CacheList`, `DeepseekV4Cache`. The paged/SSD work must preserve this generality — oMLX's `type_handlers.py` (1451 lines) is the reference for block-slice eligibility per cache class.
- **Verification bar:** Per `AGENTS.md`, every port must pass `uv run basedpyright && uv run ruff check && nix fmt && uv run pytest`. Each companion doc lists the tests to add.

---

## Companion documents

- [01-tiered-kv-cache-ssd.md](01-tiered-kv-cache-ssd.md)
- [02-native-metal-kernels.md](02-native-metal-kernels.md)
- [03-multi-token-prediction.md](03-multi-token-prediction.md)
- [04-specprefill.md](04-specprefill.md)
- [05-engine-pool-model-profiles.md](05-engine-pool-model-profiles.md)
- [06-glm-adaptive-prefill.md](06-glm-adaptive-prefill.md)
- [07-embedding-reranker.md](07-embedding-reranker.md)
- [08-boundary-snapshot-offload.md](08-boundary-snapshot-offload.md)
- [09-claude-code-context-scaling.md](09-claude-code-context-scaling.md)