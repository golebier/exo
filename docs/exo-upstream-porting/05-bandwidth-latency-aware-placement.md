# 05 — Bandwidth- & Latency-Aware Placement

**Tier:** ⭐⭐ (Tier 2)
**Effort:** Low–medium (good-first-issue scope at the core)
**Impact:** High (TPS win on every existing cluster)
**Upstream evidence:**
- #957 `[MEDIUM] Better placement algorithm for pipeline parallelism using memory bandwidth + latency` (good first issue)
- PR #2226 `add flops + memory weighted partitioning strategy`
- PR #2227 `fix: capacity-aware proportional layer allocation`
- PR #2254 `feat: measure link latency and prefer low-latency paths for ring hosts`
- PR #2252 `fix: prefer accelerators in heterogeneous placement`
- #1720 `Manual layer distribution per device` + PR #2253 `feat: support manual pipeline layer allocation`

**Cross-reference:** doc 04 (arbitrary TP splits — placement consumes it), doc 11 (memory headroom).

---

## What it is

EXO's pipeline-parallel placement is **memory-proportional only**. From #957:

> This is our placement algorithm for pipeline parallelism:
> [`src/exo/master/placement_utils.py` L52–L100]
> It places a number of layers proportional to the memory available on each machine. **This is not optimal.**
>
> In order to maximize TPS in the memory-bound regime (low batch_size), we should look at memory bandwidth and latency. The time computing on device `i` is
> `C_i = M · (N_i / N) / B_i`
> where `M` = total model memory, `N` = total layers, `N_i` = layers on device i, `B_i` = device i memory bandwidth. The time communicating between device i and i+1 is `L_i`. Total time for one token across k devices is `Σ_{1..k} (C_i + L_i)`. **This sum is what we want to minimize.**

Current placement ignores bandwidth `B_i` and link latency `L_i` entirely. A Mac Studio M3 Ultra (800 GB/s) and a Mac Mini (100 GB/s) get layers proportional to memory, so the Mini becomes the bottleneck.

---

## Why it matters

- **Every existing cluster benefits** — no new backend needed, pure algorithmic win.
- **Memory-bound regime is the common case** for single-stream decode (batch_size=1), which is exactly chat/agentic workloads.
- **Good-first-issue scoped** — the math is given; it's a constrained optimization swap.
- **Unlocks heterogeneous clusters** (#1720 manual layers, #2064 asymmetric TP, #2252 prefer accelerators) — all need a placement layer that understands more than memory.

---

## Upstream PR landscape

| PR | Adds |
|----|------|
| #2226 | flops + memory weighted partitioning (a cost model) |
| #2227 | capacity-aware proportional layer allocation |
| #2254 | **measure link latency** and prefer low-latency ring hosts (gives `L_i`) |
| #2252 | prefer accelerators in heterogeneous placement |
| #2253 | manual pipeline layer allocation (#1720) |

#2254 is especially important: it gives placement the `L_i` (link latency) data it currently lacks. #2226/#2227 give the cost-model structure. Together they're most of #957.

---

## EXO current state (local fork)

`src/exo/master/placement_utils.py`:
- `filter_cycles_by_memory` — keeps cycles where every node has enough memory.
- (L52–100, per #957) memory-proportional layer allocation.
- `src/exo/shared/types/profiling.py` has `MemoryUsage`, `NodeNetworkInfo` — bandwidth/latency data structures exist but may not be populated/used by placement.
- `src/exo/worker/engines/mlx/auto_parallel.py` does the actual sharding once placement decides shares.

Local fork is ahead on GLM/disaggregated but the placement algorithm is upstream-current (memory-proportional).

---

## Integration seam

- **Inputs:** per-node `B_i` (memory bandwidth — from `info_gatherer`, likely already detected), per-link `L_i` (from #2254's latency measurement).
- **Cost model:** implement `Σ (M·Nᵢ/N / Bᵢ) + Lᵢ` minimization. This is a small constrained optimization (integer layer counts, memory feasibility constraint). Closed-form or greedy; no solver dependency needed.
- **Placement fn:** replace the memory-proportional allocation in `placement_utils.py` with the cost-minimizing one; keep memory feasibility as a hard constraint.
- **Manual override:** #1720/#2253 — allow user-specified per-device layer counts that bypass the optimizer (validate feasibility).
- **Accelerator preference:** #2252 — in heterogeneous Metal+CUDA, prefer the accelerator for the heavier share.
- **Ring host selection:** #2254 — for ring (doc 02), prefer low-latency ring neighbors.

---

## Phased plan

### Phase 1 — Cost-model placement (the #957 core)
- Implement `Σ (M·Nᵢ/N / Bᵢ) + Lᵢ` minimization in `placement_utils.py`.
- Use existing `NodeNetworkInfo` for `L_i`; if absent, fall back to memory-proportional.
- **Tests:** extend `src/exo/master/tests/test_placement_utils.py` — uneven-bandwidth cluster gets bandwidth-weighted shares; total cost < memory-proportional baseline; memory feasibility preserved.

### Phase 2 — Link latency measurement (port #2254)
- Measure round-trip latency between nodes; populate `NodeNetworkInfo.latency`.
- Placement uses measured `L_i`.
- **Tests:** latency measurement reproducible; low-latency path preferred.

### Phase 3 — Manual override (#1720, #2253)
- User-specified per-device layer counts (config or API).
- Validate feasibility (memory); reject infeasible.
- **Tests:** manual allocation honored; infeasible rejected with clear error.

### Phase 4 — Heterogeneous accelerator preference (#2252, #2226)
- In Metal+CUDA clusters, weight the cost model by accelerator type.
- Flops+memory weighted (#2226) for the compute term.
- **Tests:** mixed-backend placement prefers accelerator for heavy share.

---

## Risks & open questions

- **Bandwidth detection:** is `B_i` reliably reported on all platforms (Mac, Linux, CUDA)? If not, placement must degrade gracefully to memory-proportional (never worse than today).
- **Latency measurement overhead:** #2254's measurement must be cheap and not perturb the cluster. Do it once at topology formation, cache.
- **Integer layer counts:** the cost model yields fractional shares; round to integer layers without violating the cost improvement. Edge cases (very small models, few layers) need care.
- **Interaction with TP (doc 04):** TP placement has a different cost model (collective cost, not pipeline relay). Keep pipeline and TP placement separate but share the bandwidth/latency inputs.
- **Interaction with ring (doc 02):** ring placement is admission-only (full model per node) today; #2254's latency preference for ring neighbors is the one ring-side change.
- **Regression:** memory-proportional is "safe." The cost model must never produce a *worse* placement in edge cases — add a "cost no worse than baseline" assertion.

---

## Definition of done

- [ ] Phase 1: cost-model placement lands; `test_placement_utils.py` shows lower cost than memory-proportional on an uneven-bandwidth fixture.
- [ ] Phase 2: link latency measured and used; low-latency preferred.
- [ ] Phase 3: manual per-device layers honored; infeasible rejected.
- [ ] Phase 4: heterogeneous accelerator preference works.
- [ ] No regression vs memory-proportional (cost-assertion test).
- [ ] `basedpyright` + `ruff` + `nix fmt` + `pytest` clean.