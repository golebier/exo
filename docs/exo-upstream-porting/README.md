# exo-explore/exo — Issues & PRs Analysis: Most Valuable Tasks

**Source repo analyzed:** `exo-explore/exo` (46.9k stars, 3.4k forks, 339 open issues, ~100 open PRs)
**Local fork:** `golebier/exo` (this working tree)
**Method:** GitHub REST + Search API; fetched open issues sorted by comments and by reactions, all open PRs, and the last 100 closed PRs (40 merged). Read issue/PR bodies for the top candidates.
**Date:** 2025-08-18

This is the **master index** for a set of implementation docs. Each task is drawn from real upstream issues/PRs, ranked by value, with the local fork's current state noted so work isn't duplicated.

---

## Repo health & velocity (responds to #2258 "is this maintained?")

**Yes, actively maintained.** The last 100 closed PRs include 40 merges, with heavy activity in Apr–Jun 2026:

- **#2132 libp2p → zenoh** networking migration (Jun 3) — major architectural change, already in this local fork.
- **#2131 rename `exo_pyo3_bindings` → `exo_rs`** + **#2139 pyo3 real modules** — Rust-binding refactor.
- **#2106 background/daemon support**, **#2072 PID file locking**, **#2084 runner stdout/stderr to file logs**.
- **#2000 engine abstraction for mlx and mflux** — multi-backend engine layer.
- **#2061 GLM 4.7 stop tokens**, **#2048 TP collective deadlock fix**, **#1995 integration tests infra**.

So #2258's "not maintained in 4 months" is refuted by the merge log. The backlog is large but movement is real.

---

## Local fork vs. upstream — don't duplicate

This working tree (`golebier/exo`) is **ahead** of upstream on several items people file issues for. These are **excluded** from the porting scope:

| Area | Local status | Upstream issue/PR |
|------|--------------|-------------------|
| GLM-5.2 vendored model + patches | ✅ Done + verified (`vendor/glm_moe_dsa/`) | #2208 (still open upstream) |
| Disaggregated / remote prefill | ✅ Done (`disaggregated/`, `remote_prefill.py`) | (upstream has no equivalent) |
| VLM / multimodal vision input | ✅ Done (`vision.py`, media-region-aware cache) | #1002 (still open upstream) |
| zenoh networking | ✅ Done (merged upstream #2132) | #2132 |
| Rust bindings (`exo_rs`) | ✅ Done | #2131, #2139 |
| DeepSeek-V4 cache handling | ✅ Done (`_copy_v4_cache`) | (partial upstream) |
| KV prefix cache (RAM, LRU, SSM snapshots) | ✅ Done | #2251 addresses its prefill OOM gap |

---

## Ranked tasks (by value × leverage)

Each row links to a dedicated implementation doc.

### ⭐⭐⭐ Tier 1 — Biggest wins

| # | Task | Doc | Evidence | Why it matters |
|---|------|-----|----------|----------------|
| 1 | **Linux CUDA support** | [01-linux-cuda-support.md](01-linux-cuda-support.md) | #913 (56👍, tracking), PR #2103, #2129, #2216, #2053; #904 Framework (43👍), #910 CPU (19👍), #934 Vulkan (19👍) | Highest-reaction issue in the repo. Unlocks the entire non-Mac GPU market (NVIDIA, Framework Desktop, DGX Spark). Multiple PRs in flight. |
| 2 | **Sequence-parallel Ring Attention for MLX** | [02-ring-attention-mlx.md](02-ring-attention-mlx.md) | PR #2255 (closes #39) | Shards the prefill *sequence* across nodes (weights replicated) with overlapped RDMA. Solves long-prompt TTFT — the exact pain in #2208/#781. Production PR: 33 tests, 2- & 3-rank Metal tests, benchmarks. |
| 3 | **Speculative decoding: Drafter + MTP + DFlash** | [03-speculative-decoding-drafter-mtp.md](03-speculative-decoding-drafter-mtp.md) | PR #2079, #2110; issue #1685 | Measured **3.3–4.3× decode TPS** on Qwen 3.5/3.6 (M5 Max), 92%+ accept rate, incl. multi-device TP coupled drafter. The biggest single decode-speedup available. |

### ⭐⭐ Tier 2 — Strong improvements

| # | Task | Doc | Evidence | Why it matters |
|---|------|-----|----------|----------------|
| 4 | **Arbitrary tensor-parallel splits** | [04-arbitrary-tensor-parallel-splits.md](04-arbitrary-tensor-parallel-splits.md) | #953 (HARD); PRs #2064, #2226, #2227 | Current `shard_linear`/`shard_inplace` require dims divisible by group size → blocks 3-node TP, quantized models (gs=32), heterogeneous clusters. Smart padding is the start. |
| 5 | **Bandwidth- & latency-aware placement** | [05-bandwidth-latency-aware-placement.md](05-bandwidth-latency-aware-placement.md) | #957 (good first issue); PRs #2226, #2227, #2254, #2252 | Placement is memory-proportional only. Minimize `Σ (M·Nᵢ/N / Bᵢ) + Lᵢ` instead. Direct TPS win in memory-bound regime; pairs with #1720 manual layers (#2253 PR). |
| 6 | **GLM-5.2 long-context fix + RDMA/Thunderbolt reliability** | [06-glm-long-context-rdma-reliability.md](06-glm-long-context-rdma-reliability.md) | #2208; #1847, #1390, #2067, #1973, #2219, #1459, #1320, #1701, #1534, #1887; PRs #2160 (multi-rail), #2195 (bond), #2063 (TB5) | The largest pain cluster. #2208 is a `mlx Fence::wait` GPU-timeout race at prefill→decode transition. Local fork has the GLM model fix; still needs the mlx-jaccl fix + multi-rail/bond RDMA hardening. |
| 7 | **TurboQuant KV-cache compression** | [07-turboquant-kv-cache.md](07-turboquant-kv-cache.md) | PR #2148 (PDD plan); #1988, #2242, #2261 | KV-cache quantization plan. Ties directly to the oMLX porting analysis (docs/omlx-porting/). Cuts KV memory; Qwen3-Next hybrid-cache handling called out. |
| 8 | **Model support breadth: unified catalog + GGUF + embeddings** | [08-model-support-breadth.md](08-model-support-breadth.md) | PR #2012 (catalog), #2145 (custom cards); #1695 GGUF (7👍), #1047 embeddings (6👍) | Unified catalog across HF/LM Studio/Ollama/llama.cpp with content-based resolution. GGUF via llama.cpp runner. Embeddings for RAG (overlaps oMLX doc 07). |

### ⭐ Tier 3 — Valuable, smaller, or partially covered

| # | Task | Doc | Evidence | Why it matters |
|---|------|-----|----------|----------------|
| 9 | **Peer-to-peer / Thunderbolt model distribution** | [09-model-distribution-p2p.md](09-model-distribution-p2p.md) | #721 (rsync, 11👍), #1257 (why all nodes DL), #1992 PR (p2p), #2117 (TB share) | Avoid every node re-downloading multi-hundred-GB models from HF. One node downloads, others pull over TB/LAN. |
| 10 | **Observability: Prometheus /metrics + cluster stats** | [10-observability-cluster-stats.md](10-observability-cluster-stats.md) | PR #1985, #2144 (zenoh Last Value), #2126 (telemetry); #1700 (global stats) | No metrics endpoint today. `/metrics` + global cluster stats for ops. |
| 11 | **Memory headroom before prefill + near-limit placement** | [11-memory-headroom-prefill.md](11-memory-headroom-prefill.md) | PR #2251, #1709; #2240, #2241 | Prefix-cache allocations can starve prefill activations → OOM. PR #2251 reserves headroom via `EXO_PREFILL_MEMORY_THRESHOLD`. |
| 12 | **GPU offload for prompt processing** | [12-gpu-offload-prompt-processing.md](12-gpu-offload-prompt-processing.md) | #781 (10👍) | NVIDIA GPU does prefill, Mac does decode. Overlaps with CUDA (#01) + ring attention (#02) — a heterogeneous prefill-offload pattern. |

---

## Recommended implementation order

Sequenced by (impact × leverage) ÷ effort, and respecting dependencies:

1. **Ring Attention for MLX** ([02](02-ring-attention-mlx.md)) — production PR exists with tests/benchmarks; biggest TTFT win; no CUDA dependency. Review & adapt #2255.
2. **Speculative decoding (Drafter + MTP + DFlash)** ([03](03-speculative-decoding-drafter-mtp.md)) — 3–4× decode TPS; self-contained PR #2079. Pairs with oMLX MTP doc.
3. **TurboQuant KV cache** ([07](07-turboquant-kv-cache.md)) — PDD plan exists (#2148); composes with the oMLX tiered-cache work.
4. **Bandwidth/latency placement** ([05](05-bandwidth-latency-aware-placement.md)) — good-first-issue scope; immediate TPS win on existing clusters.
5. **Arbitrary TP splits** ([04](04-arbitrary-tensor-parallel-splits.md)) — unblocks 3-node + quantized TP; complements #05.
6. **GLM-5.2 long-context + RDMA reliability** ([06](06-glm-long-context-rdma-reliability.md)) — closes the #2208 pain; needs mlx-jaccl fix + multi-rail.
7. **Linux CUDA** ([01](01-linux-cuda-support.md)) — large effort, multiple PRs to reconcile (#2103, #2129, #2216, #2053); biggest *reach* win.
8. Tier 3 items in any order.

---

## Cross-cutting notes

- **Two PR clusters address the same pain (long-context TTFT):** Ring Attention (#2255), GPU prompt offload (#781), and the GLM/RDMA fix (#2208) all attack long-prompt latency. Ring Attention is the most general; #781 is the heterogeneous-NVIDIA special case; #2208 is the correctness bug underneath.
- **Speculative decoding is double-covered:** upstream PR #2079 (Drafter + MTP + DFlash) and the oMLX MTP doc (`docs/omlx-porting/03-...`). Reconcile into one design — #2079's "drafter abstraction" is the more general framing (supports DFlash + MTP + coupled multi-device).
- **Placement is triple-covered:** #957 (bandwidth/latency), #953 (arbitrary TP), #1720/#2253 (manual). One cohesive placement rewrite addresses all three.
- **Verification bar (AGENTS.md):** every task must pass `uv run basedpyright && uv run ruff check && nix fmt && uv run pytest`. Each companion doc lists tests to add.
- **Attribution:** when adapting an upstream PR, credit the original author in the commit message (e.g. `Adapted from #2255 by @abendrothj`).

---

## Companion documents

- [01-linux-cuda-support.md](01-linux-cuda-support.md)
- [02-ring-attention-mlx.md](02-ring-attention-mlx.md)
- [03-speculative-decoding-drafter-mtp.md](03-speculative-decoding-drafter-mtp.md)
- [04-arbitrary-tensor-parallel-splits.md](04-arbitrary-tensor-parallel-splits.md)
- [05-bandwidth-latency-aware-placement.md](05-bandwidth-latency-aware-placement.md)
- [06-glm-long-context-rdma-reliability.md](06-glm-long-context-rdma-reliability.md)
- [07-turboquant-kv-cache.md](07-turboquant-kv-cache.md)
- [08-model-support-breadth.md](08-model-support-breadth.md)
- [09-model-distribution-p2p.md](09-model-distribution-p2p.md)
- [10-observability-cluster-stats.md](10-observability-cluster-stats.md)
- [11-memory-headroom-prefill.md](11-memory-headroom-prefill.md)
- [12-gpu-offload-prompt-processing.md](12-gpu-offload-prompt-processing.md)