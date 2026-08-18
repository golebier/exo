# 01 — Linux CUDA Support

**Tier:** ⭐⭐⭐ (Tier 1)
**Effort:** Very high (multi-PR reconciliation + backend abstraction)
**Impact:** Very high (unlocks the entire non-Mac GPU market)
**Upstream evidence:**
- #913 `[TRACKING] General Linux CUDA Support` (56👍, 14 comments) — top-reaction issue in the repo
- PR #2103 `feat: Add Native CUDA Support for Linux GPU Inference` (Winston-9527)
- PR #2129 `feat: 3-node heterogeneous cluster support (Metal + CUDA)` (Juliuscply)
- PR #2216 `feat: report CUDA GPU VRAM as node memory on Linux` (darinlarimore)
- PR #2053 `Add CUDA Docker image support` (cameronbergh)
- Related: #904 Framework Desktop (43👍), #910 Linux CPU (19👍), #934 Vulkan (19👍), #907 Raspberry Pi arm64 (25👍), #1102 Mac+DGX Spark, #1380 CXL

---

## What it is

EXO is Apple-Silicon/MLX-first. The largest demand in the issue tracker (by a wide margin) is to run on **NVIDIA/Linux CUDA** — both standalone Linux GPU boxes and **heterogeneous Mac + NVIDIA** clusters (Mac Studio + DGX Spark, Mac + Framework Desktop). The tracking issue #913 has more reactions than any other issue.

Four open PRs attack different slices:
- **#2103** — native CUDA inference on Linux (the core engine).
- **#2129** — 3-node heterogeneous Metal + CUDA (cluster placement across mixed backends).
- **#2216** — report CUDA VRAM as node memory so placement can see it.
- **#2053** — CUDA Docker image (packaging).

Plus adjacent platform asks: Vulkan (#934, 19👍) as a cross-platform fallback, Linux CPU (#910, 19👍), Raspberry Pi/arm64 (#907, 25👍), CXL (#1380).

---

## Why it matters

- **Market reach:** most would-be EXO users have NVIDIA hardware, not Mac Studios. 56 reactions on #913 is the strongest signal in the repo.
- **Heterogeneous clusters:** #1102 (Mac Studio + DGX Spark, 19👍) and #904 (Framework Desktop, 43👍) show the dream is mixing Mac + NVIDIA over Thunderbolt/RDMA. This is the headline use case for #781 (GPU prompt offload, doc 12).
- **Foundation for other tasks:** CUDA is a prerequisite for #781 (NVIDIA prefill offload) and a co-equal path with Ring Attention (doc 02) for long-context TTFT.

---

## Upstream PR landscape (reconciliation needed)

The four PRs overlap and need to be reconciled into one coherent CUDA backend:

| PR | Scope | Status risk |
|----|-------|-------------|
| #2103 | Native CUDA Linux inference engine | Core; largest. Need to review against `exo_rs` Rust-binding refactor (#2131/#2139) which landed after. |
| #2129 | Heterogeneous Metal+CUDA 3-node | Depends on #2103 + placement changes; may conflict with #2064 (asymmetric TP). |
| #2216 | CUDA VRAM → node memory | Small, foundational for placement (#05). Should land first. |
| #2053 | CUDA Docker image | Packaging; depends on #2103. |

**Note:** upstream also merged #2000 `engine abstraction for mlx and mflux` and #2087 `use custom mlx sources for linux`. The CUDA engine should slot into the same engine-abstraction layer (`src/exo/worker/engines/`) alongside `mlx/` and `image/`.

---

## EXO current state (local fork)

- Engines: `src/exo/worker/engines/{mlx, image}/` only. No CUDA engine.
- `src/exo/utils/info_gatherer/info_gatherer.py` references `cuda`/`CUDA` (likely detection only, not inference).
- Placement (`src/exo/master/placement_utils.py`) is memory-proportional and MLX/topology-centric.
- Backend abstraction is partial (mlx + mflux for image).
- Local fork is **ahead** on GLM-5.2/disaggregated prefill but has **no CUDA** at all.

---

## Integration seam

- **Engine layer:** add `src/exo/worker/engines/cuda/` implementing the same engine interface as `mlx/` (continuous batching, KV cache, prefill/generate). Reuse the engine-abstraction pattern from #2000.
- **Backend selection:** a node's backend (MLX/CUDA/CPU/Vulkan) is a capability advertised in topology (`src/exo/shared/topology.py`, `src/exo/shared/types/topology.py`). Placement must be backend-aware (don't MLX-shard onto a CUDA node).
- **Model format:** CUDA path likely uses `llama.cpp`/gguf or HF safetensors via a CUDA runtime (vLLM-style? EXO's own?). Decide: (a) wrap llama.cpp server, (b) native PyTorch+CUDA, (c) Triton. PR #2103's choice is the reference.
- **Heterogeneous placement (#2129):** the master must plan shards across mixed backends — a CUDA node runs the CUDA engine, a Mac runs MLX, and they cooperate via the existing zenoh/collective layer. This is the hardest part.
- **VRAM reporting (#2216):** `info_gatherer` must report CUDA VRAM as `MemoryUsage` so placement (#05) can use it. Land this first.
- **Packaging (#2053):** CUDA Docker image + nix flake extension for Linux+CUDA.

---

## Phased plan

### Phase 1 — VRAM reporting + backend capability (land first, unblocks placement)
- Port #2216: report CUDA VRAM as node memory in `info_gatherer`.
- Add `Backend` capability to topology (`mlx` | `cuda` | `cpu` | `vulkan`).
- **Tests:** node-memory reporting test with mock CUDA device; placement sees VRAM.

### Phase 2 — CUDA inference engine (port #2103)
- Rebase #2103 onto current `exo_rs`/zenoh main.
- Implement `engines/cuda/` behind the engine abstraction (#2000 pattern).
- Continuous batching + KV cache equivalent (decide runtime: llama.cpp vs native).
- **Tests:** single-node CUDA inference parity (output vs MLX within tol); batching.

### Phase 3 — Heterogeneous placement (port #2129)
- Master plans shards across mixed backends.
- Collective/transport: CUDA↔MLX over zenoh (TB/RDMA where available).
- **Tests:** 3-node mixed Metal+CUDA placement; cross-backend collective.

### Phase 4 — Packaging (port #2053) + adjacent backends
- CUDA Docker image; nix flake CUDA support.
- Optional: Vulkan fallback (#934) via the same engine abstraction; Linux CPU (#910).

---

## Risks & open questions

- **PR drift:** #2103 was written before the zenoh migration (#2132) and `exo_rs` refactor (#2131). Expect significant rebase work; the networking/Rust layers it touches have changed.
- **Runtime choice:** wrapping llama.cpp is easiest but breaks EXO's "native MLX" feel and may not support EXO's KV-prefix-cache/disaggregated-prefill innovations. A native path preserves those but is far more work. Decide explicitly.
- **Heterogeneous collectives:** MLX's distributed ops (`mx.distributed`) and CUDA's (NCCL) don't interop natively. Cross-backend collectives likely go through CPU staging or zenoh — latency penalty. This is the core research question for #2129.
- **Quantization divergence:** MLX uses `mlx-community` quants; CUDA typically uses GGUF/AWQ/GPTQ. The unified catalog (doc 08) must reconcile formats per backend.
- **CI:** no Linux+CUDA runners in EXO's CI today. Tests will be hardware-gated (like the GLM native kernels in oMLX doc 02).

---

## Definition of done

- [ ] Phase 1: CUDA VRAM reported as node memory; placement consumes it; tests green.
- [ ] Phase 2: single-node CUDA inference produces output within tol of MLX; batching works.
- [ ] Phase 3: 3-node Metal+CUDA heterogeneous cluster runs a sharded model end-to-end.
- [ ] Phase 4: CUDA Docker image builds; documented quickstart.
- [ ] `basedpyright` + `ruff` + `nix fmt` + `pytest` clean (hardware tests gated).