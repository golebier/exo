# 12 — GPU Offload for Prompt Processing (Heterogeneous Prefill)

**Tier:** ⭐ (Tier 3)
**Effort:** High (depends on CUDA doc 01 + ring/transport)
**Impact:** High for the Mac+NVIDIA use case
**Upstream evidence:**
- #781 `Feature Request: Integrate GPU Offloading for Prompt Processing` (10👍)

**Cross-reference:** doc 01 (Linux CUDA), doc 02 (Ring Attention), doc 06 (RDMA reliability).

---

## What it is

#781 asks for the **heterogeneous prefill-offload** pattern:

> We can connect a Linux/Windows machine with a dedicated GPU (RTX 5090) to the Mac via Thunderbolt 5. Can we offload **prompt processing** (compute-intensive) to the NVIDIA GPU while using the Mac for **token generation**?
>
> Prompt processing on Mac is a pain especially with larger context windows. Offloading it to an NVIDIA GPU would save significant time.

This is the "NVIDIA does prefill, Mac does decode" split — attractive because prefill is compute-bound (GPU wins big) while decode is memory-bandwidth-bound (Mac's unified memory shines).

---

## Why it matters

- **Long-prompt TTFT is the #1 pain** (#2208, #781). An NVIDIA GPU cuts prefill time dramatically.
- **Heterogeneous Mac+NVIDIA clusters** are the dream config (#1102 Mac+DGX Spark 19👍, #904 Framework 43👍). This pattern makes them compelling: each does what it's best at.
- **Composes with ring attention (doc 02):** ring shards the *sequence* across nodes — an NVIDIA node in the ring contributes its fast prefill to the sequence-parallel split. This is the clean architectural fit for #781.
- **Composes with disaggregated prefill (local fork):** EXO already ships `remote_prefill.py` — the NVIDIA node acts as a remote prefill server, Mac ingests KV. The transport exists; only the NVIDIA engine is missing.

---

## Relationship to other docs

This is the **integration capstone** of three other tasks:
- **doc 01 (CUDA):** provides the NVIDIA inference engine.
- **doc 02 (Ring Attention):** the sequence-parallel prefill that lets NVIDIA contribute to a long prompt.
- **doc 06 (RDMA):** the TB5/RDMA transport that makes Mac↔NVIDIA KV transfer fast.
- **local `remote_prefill.py`:** the existing disaggregated-prefill protocol that ships KV from prefill node to decode node.

#781 is essentially "wire these together for the Mac+NVIDIA case."

---

## EXO current state (local fork)

- `src/exo/worker/engines/mlx/generator/remote_prefill.py` + `disaggregated/` — **remote prefill protocol exists** (local ahead of upstream). A prefill server fills KV, ships to the decode node.
- No CUDA engine (doc 01).
- No ring attention (doc 02).
- So the *protocol* for #781 exists; the *NVIDIA prefill node* doesn't.

---

## Integration seam

Two architectures, pick one (or both):

### A. Disaggregated prefill (remote_prefill path)
- NVIDIA node runs the CUDA engine (doc 01) as a **prefill server**.
- Mac node requests prefill; NVIDIA computes KV, ships to Mac over TB5/RDMA (doc 06).
- Mac decodes with its MLX engine.
- **Fits existing `remote_prefill.py`** — minimal new code once CUDA engine exists.

### B. Ring Attention (sequence-parallel)
- NVIDIA + Mac in a ring; sequence sharded across both.
- NVIDIA handles its sequence slice (fast prefill); Mac handles its slice.
- **Fits doc 02** — needs ring attention + cross-backend collectives (the hard part of doc 01 Phase 3).

Architecture A is simpler (reuse disaggregated prefill); B is more general (both nodes contribute to the same prefill). Start with A.

---

## Phased plan

### Phase 0 — Prerequisites
- doc 01 Phase 2 (CUDA engine) done.
- doc 06 Phase 1 (TB5/RDMA stable) done.

### Phase 1 — Disaggregated NVIDIA prefill (Architecture A)
- NVIDIA node runs CUDA engine as prefill server.
- Mac requests prefill via `remote_prefill.py`; KV shipped over TB5.
- **Tests:** NVIDIA prefill → Mac decode end-to-end; KV round-trip parity; TTTT vs Mac-only prefill.

### Phase 2 — Ring with NVIDIA (Architecture B, optional)
- NVIDIA + Mac in a ring (doc 02 + cross-backend collective from doc 01 Phase 3).
- **Tests:** 2-node ring (NVIDIA + Mac) prefill; sequence-parallel correctness.

### Phase 3 — Auto-placement
- Placement (doc 05) routes long-prompt requests to NVIDIA-prefill + Mac-decode automatically when the topology has both.
- **Tests:** heterogeneous cluster auto-uses NVIDIA for prefill.

---

## Risks & open questions

- **KV format divergence:** CUDA (llama.cpp/vLLM) KV format ≠ MLX KV format. The disaggregated-prefill protocol (`disaggregated/protocol.py`) must translate, or both engines must share a KV format. This is the core technical risk — EXO's `ingest_into_mlx_cache` assumes MLX KV.
- **Cross-backend collective (Architecture B):** MLX distributed ↔ NCCL doesn't interop natively. Ring across Mac+NVIDIA needs CPU-staged or zenoh transport — latency penalty. Architecture A avoids this.
- **TB5 Mac↔NVIDIA:** TB5 to a Linux/NVIDIA box works (TB-over-USB4 to Linux), but RDMA (JACCL) is Apple-only. The transport is likely TCP/zenoh, not JACCL. Verify throughput suffices for KV transfer.
- **Model duplication:** NVIDIA and Mac both need the model weights (disaggregated) or each its slice (ring). For a 395GB model, that's expensive. Prefill-offload is best for *mid-size* models where both can hold the weights.
- **When to offload:** only worth it for long prompts (short prompts: Mac-only prefill is fine, no transfer overhead). Add a threshold (`EXO_REMOTE_PREFILL_MIN_TOKENS` exists at 1000 — tune for the NVIDIA case).

---

## Definition of done

- [ ] Phase 1: NVIDIA prefill → Mac decode end-to-end; TTFT < Mac-only for prompts > N tokens.
- [ ] KV round-trip parity (CUDA→MLX format translation correct).
- [ ] Phase 2 (optional): 2-node ring (NVIDIA+Mac) prefill.
- [ ] Phase 3: auto-placement uses NVIDIA for prefill when available.
- [ ] `basedpyright` + `ruff` + `nix fmt` + `pytest` clean.