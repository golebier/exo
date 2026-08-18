# 02 — Sequence-Parallel Ring Attention for MLX

**Tier:** ⭐⭐⭐ (Tier 1)
**Effort:** Medium (production PR exists to adapt)
**Impact:** Very high (long-prompt TTFT; the #2208/#781 pain)
**Upstream evidence:**
- PR #2255 `feat: sequence-parallel Ring Attention for MLX` (abendrothj) — closes #39
- Related: #2208 (long-context hang/garble), #781 (GPU prompt offload), #957 (placement)

---

## What it is

**Ring Attention** shards the **sequence** across ranks (weights are *replicated* on each node), rotating KV blocks around the ring while accumulating attention with a numerically-stable online-softmax merge. This is the opposite of pipeline/TB sharding (which shards *layers* across nodes). It directly attacks **prefill TTFT on long prompts** — the exact pain in #2208 and #781.

From PR #2255:

> `RingAttentionLayer` wraps only the attention module (never a full decoder block), splits the sequence across ranks, and rotates KV blocks around the ring while accumulating attention with a numerically stable online-softmax merge.
>
> **Overlapped communication:** dedicated CPU send/receive streams (MLX distributed send/recv are CPU ops; unified memory lets them consume Metal-produced KV without copies). Receives are posted *before* the current block's attention is scheduled, and the lazy attention recurrence is explicitly `mx.async_eval`-ed to create the computation window the transfer overlaps with.

This is a **production-quality PR**: 33 unit/integration tests including scheduling-order regression (communication posted before compute is waited on), online-merge equivalence vs full attention, and real 2-rank & 3-rank distributed Metal tests over the MLX ring backend. Plus `exo-bench --sharding ring` support and docs.

---

## Why it matters

Long-context prefill is the #1 UX pain in the tracker:
- #2208: 45k–52k-token agentic prompts stall at the prefill→decode transition.
- #781: "Prompt processing on Mac is a pain especially with larger context window" (10👍).

Pipeline/TB sharding shards layers, so each node still processes the **full sequence** for its layers — prefill time doesn't drop with more nodes. Ring Attention shards the **sequence**, so prefill time scales down with ring size. For agentic workloads (Claude Code, Pi, Codex sending huge system prompts + tool schemas), this is transformative.

It's also **complementary** to existing sharding: ring for prefill (sequence-parallel), pipeline for decode (layer-parallel). And it needs no CUDA — pure MLX distributed.

---

## Upstream PR design (from #2255 body)

### Core component
- **`RingAttentionLayer`** at `src/exo/worker/engines/mlx/ring_attention.py` — wraps only the attention module (never a full decoder block).

### Overlapped communication
- Dedicated CPU send/receive streams. MLX distributed send/recv are CPU ops; unified memory lets them consume Metal-produced KV **without copies**.
- Receives posted **before** the current block's attention is scheduled.
- Lazy attention recurrence explicitly `mx.async_eval`-ed to create the compute window the transfer overlaps.
- Transport peer (ring neighbor) correctly separated from KV origin rank.

### Placement
- `Sharding.Ring` gated on:
  - `MlxRing` transport available
  - ≥2 nodes
  - model-card `supports_ring`
  - **per-node admission of the *full* model size** (weights are replicated; only prefill sequence is sharded)
- So ring requires each node to hold the whole model — a different memory profile than pipeline sharding.

### Model cards
- Ring support declared for verified Llama 3.x cards.
- `supports_ring` defaults **off** for everything else (capability-gated).

### Benchmarks/env
- `MLX_METAL_FAST_SYNCH=1` recommended.
- `exo-bench --sharding ring` support.

---

## EXO current state (local fork)

- **No ring attention.** `rg ring_attention` returns nothing in `src/`.
- Has `src/exo/worker/engines/mlx/auto_parallel.py` with `shard_model` for tensor/pipeline sharding — ring is a new `Sharding` variant.
- `src/exo/shared/types/worker/shards.py` defines `Sharding` enum (Pipeline, Tensor, …) — add `Ring`.
- Local fork is ahead on disaggregated prefill (`remote_prefill.py`) which ring could **compose with** (remote prefill + ring for the local portion).

---

## Integration seam

- **New file:** `src/exo/worker/engines/mlx/ring_attention.py` (port from #2255).
- **Sharding type:** add `Sharding.Ring` to `src/exo/shared/types/worker/shards.py` + `RingShardMetadata`.
- **Placement:** extend `src/exo/master/placement_utils.py` to admit ring when `supports_ring` + `MlxRing` transport + ≥2 nodes + each node holds full model. Cross-reference doc 05 (placement rewrite).
- **Model cards:** add `supports_ring` field to `ModelCard` (`src/exo/shared/models/model_cards.py`); default off; enable for verified Llama 3.x / Qwen3.
- **Generator:** ring applies during **prefill** (sequence-parallel). Decode uses replicated caches (each node has full KV for its replicated weights) — verify the decode path in #2255.
- **Bench:** add `--sharding ring` to `bench/`.

---

## Phased plan

### Phase 1 — Port & adapt #2255
- Rebase #2255 onto current main (zenoh + exo_rs + local GLM/disaggregated work).
- Port `ring_attention.py`, `Sharding.Ring`, placement gate, model-card field.
- Verify the 33 tests still pass; adapt to local fork's `KVPrefixCache` (which has media-region-aware logic — ensure ring composes).
- **Tests:** port the 33 tests; add a media-region + ring interaction test.

### Phase 2 — Verify correctness & compose with disaggregated prefill
- Online-merge equivalence vs full attention (numerical) on real Metal, 2- and 3-rank.
- Compose with local `remote_prefill.py`: ring-shard the local prefill, optionally pull remote KV.
- **Tests:** 3-rank Metal distributed test; ring + remote prefill composition.

### Phase 3 — Benchmarks & model-card enablement
- `exo-bench --sharding ring`: TTFT vs sequence length, 1 vs 2 vs 3 ranks.
- Enable `supports_ring` on verified Qwen3 cards (in addition to Llama 3.x).
- Document the memory caveat (full-model replication per node).

### Phase 4 — Decode-path optimization
- #2255 uses replicated decode caches. Decide whether to keep replicated decode (simple, more memory) or hand off to pipeline sharding for decode (complex, less memory). Benchmark.

---

## Risks & open questions

- **Memory profile:** ring requires **full model replication** per node. For a 395GB GLM-5.2 model (#2208 env), that means every node needs 395GB — ring won't help that case unless combined with pipeline/TB. Ring is best for *mid-size* models (≤ per-node memory) with *long* sequences. Document this clearly so users don't expect ring to help a model that doesn't fit.
- **Compose with pipeline?** Can a model be pipeline-sharded across 2 nodes AND ring-sharded across 2 nodes (4 nodes total)? #2255 doesn't address this. Likely a future "2D sharding" problem.
- **Online-softmax numerical stability:** for very long sequences across many ranks, the online merge must stay stable. #2255 has equivalence tests; re-verify at extreme lengths (52k tokens from #2208).
- **`mx.async_eval` correctness:** the overlap depends on lazy eval semantics; if the local fork's `generate.py` eager-evals anywhere, the overlap breaks and ring degrades to sequential. Audit the prefill path for stray `mx.eval`/`.item()`.
- **Model-card gating:** enabling `supports_ring` on a card without verification risks silent corruption. Keep default-off; enable per verified architecture.

---

## Definition of done

- [ ] Phase 1: #2255 rebased; 33 tests pass on local fork; `Sharding.Ring` + `supports_ring` in place.
- [ ] Phase 2: 2- & 3-rank Metal distributed tests pass; online-merge within tol of full attention at 52k tokens; ring + remote prefill composes.
- [ ] Phase 3: `exo-bench --sharding ring` shows TTFT scaling with ring size; Qwen3 cards enabled.
- [ ] Memory caveat documented (ring ≠ for models that don't fit per node).
- [ ] `basedpyright` + `ruff` + `nix fmt` + `pytest` clean.