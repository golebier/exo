# 04 — Arbitrary Tensor-Parallel Splits

**Tier:** ⭐⭐ (Tier 2)
**Effort:** Medium-high
**Impact:** High (unblocks 3-node TP, quantized models, heterogeneous clusters)
**Upstream evidence:**
- #953 `[HARD] Support arbitrary tensor parallel splits`
- PR #2064 `Asymmetric tensor parallelism for uneven-memory clusters` (team-wcv)
- PR #2226 `add flops + memory weighted partitioning strategy` (Mustafa-cuda-dev)
- PR #2227 `fix: capacity-aware proportional layer allocation` (jgawronek)
- PR #2268 `mlx tensor-parallel: replicate vector-quantized codebooks`

**Cross-reference:** doc 05 (placement) — these two are sibling halves of one sharding rewrite.

---

## What it is

EXO's tensor parallelism uses MLX LM's `shard_linear` / `shard_inplace` to split tensors across nodes. From #953:

> These functions assume that the sharding dimensions are exactly divisible by the group size.
>
> **Limitations:**
> 1. Certain models can't be TP-sharded — particularly **quantized models** (e.g. GLM Air 4.5 `config.json` with gs=32 quantization).
> 2. Most models **can't be parallelised across 3 nodes** — their intermediate size isn't divisible by 3.
> 3. This is a **blocker for smarter TP on heterogeneous devices**.
>
> A potential start is **smart padding**.

So TP today only works for "nice" dims ÷ group_size. Real clusters (3 Macs, quantized models, uneven memory) are left out.

---

## Why it matters

- **3-node clusters:** the most common small-cluster size after 2-node, and it's broken for most models because intermediate_size is rarely divisible by 3.
- **Quantized models:** group-size-32 quants can't be evenly split — and quants are *the* way people run big models on Mac. This blocks the highest-value configs.
- **Heterogeneous clusters:** #2064 (asymmetric TP for uneven memory) + #2226 (flops+memory weighted) + #2227 (capacity-aware) all need arbitrary splits as a foundation. You can't do asymmetric TP if you can't split non-evenly.
- **Unblocks doc 05:** placement rewrite assumes arbitrary splits are possible.

---

## Upstream PR landscape

| PR | Approach |
|----|----------|
| #2064 | Asymmetric TP for uneven-memory clusters (the heterogeneous case) |
| #2226 | flops + memory weighted partitioning (cost model) |
| #2227 | capacity-aware proportional layer allocation (pipeline side) |
| #2268 | replicate vector-quantized codebooks (the quant-specific fix) |

#2268 is notable: vector-quantized (VQ) codebooks can't be naively split (the codebook is shared), so they must be **replicated** across ranks while the rest splits. This is the quantized-model fix.

---

## EXO current state (local fork)

`src/exo/worker/engines/mlx/auto_parallel.py`:
- Uses `shard_linear` / `shard_inplace` from mlx-lm.
- `assert n_heads % o_groups == 0`, `heads_per_group must be divisible by world_size` — the exact divisibility assumptions #953 flags.
- `shard_heads` for MLA-style head splitting.
- Multiple `shard_model` overloads per architecture (lines 631, 638, 698, 949, 984, 1124).
- Has GLM MoE DSA sharding (local fork is ahead here).

So the local fork has rich per-arch sharding but inherits the divisibility limitation.

---

## Integration seam

- **Smart padding:** pad the sharding dim up to a multiple of group_size, shard, then mask the padding out in the forward. This is #953's suggested start.
- **VQ codebook replication:** port #2268 — detect vector-quantized weights and replicate (don't split) the codebook; split only the lookup indices/weights.
- **Asymmetric splits:** port #2064 — allow rank i to get a non-equal share (e.g. 40/30/30 for 3 nodes of unequal memory). Requires the padding approach + a per-rank split spec.
- **Shard metadata:** `src/exo/shared/types/worker/shards.py::TensorShardMetadata` must carry the per-rank split spec (offsets/lengths, replicated flag for codebooks).
- **Placement:** the master computes the split spec (doc 05) and sends it in `TensorShardMetadata`; `auto_parallel.py` applies it.

---

## Phased plan

### Phase 1 — Smart padding for non-divisible dims
- In `auto_parallel.py`, replace direct `shard_linear` with a padded variant: pad to next multiple of group_size, shard, mask padding in forward.
- Handle the 3-node case (intermediate_size % 3 != 0).
- **Tests:** 3-node TP on a model whose intermediate_size isn't divisible by 3; output parity vs single-node; no NaN from padding.

### Phase 2 — Vector-quantized codebook replication (#2268)
- Detect VQ codebooks (group-size quant with shared codebook).
- Replicate codebook across ranks; split the rest.
- **Tests:** GLM Air 4.5 (gs=32) TP-shards correctly; codebook parity across ranks; output parity.

### Phase 3 — Asymmetric splits (#2064, #2226, #2227)
- Per-rank split spec in `TensorShardMetadata`.
- `auto_parallel.py` applies arbitrary per-rank shares.
- Placement (doc 05) computes shares from memory/flops/latency.
- **Tests:** 3-node uneven-memory cluster (e.g. 128/128/64 GB) gets proportional shares; output parity; no rank-starvation stalls.

### Phase 4 — Verification
- TP bit-exactness test exists (`test_tp_bit_exact.py`) — extend to padded, VQ, and asymmetric cases.

---

## Risks & open questions

- **Padding memory cost:** padding wastes a little memory per rank; for quants the waste is small. Ensure padding doesn't push a node over its memory budget (interacts with doc 11).
- **Mask correctness in forward:** a mask bug silently corrupts output. The bit-exactness test is the safety net — make it exhaustive across configs.
- **VQ detection:** reliably detecting VQ codebooks from config/weights is fiddly. Mirror mlx-lm's quantized-layer detection.
- **Asymmetric + collectives:** asymmetric TP changes the all-reduce/collective shape per rank — verify MLX distributed handles non-equal splits (it should, but test). Cf. #2048 collective deadlock fix.
- **Compose with ring (doc 02) & drafter (doc 03):** arbitrary TP + ring + coupled drafter is a 3-way composition. Verify at least TP+drafter works (doc 03 Phase 3 assumes it).

---

## Definition of done

- [ ] Phase 1: 3-node TP works on a non-divisible intermediate_size; bit-exactness passes.
- [ ] Phase 2: gs-32 quantized models TP-shard (codebook replicated); bit-exactness passes.
- [ ] Phase 3: uneven-memory 3-node cluster gets proportional shares; no stalls.
- [ ] `test_tp_bit_exact.py` extended to padded/VQ/asymmetric cases.
- [ ] `basedpyright` + `ruff` + `nix fmt` + `pytest` clean.