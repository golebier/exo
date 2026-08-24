# GLM-5.2 Long-Context RDMA Reliability — Verified Current State

**Date:** 2026-08-24
**Cluster:** 2× M3 Ultra 256GB (`m3msu256a.local` + `m3msu256b.local`)
**Build:** `1.0.72-cache-efficiency-dev2` (commit `42762e44`)
**Model:** `GLM-5.2-oQ4`, 2-node tensor-parallel (JACCL/RDMA)

This doc records the **verified** state of the two named blockers for the
flagship 2-node sharded GLM-5.2 path on ~45–52k-token agentic prompts. It
supersedes the "pending fix" status in the older docs, which were stale.

---

## TL;DR

| Blocker | Status | Evidence |
|---------|--------|----------|
| JACCL `all_sum` rendezvous race (model-load) | ✅ **Fixed** | Commit `17252452` + `plan.py` lifecycle gating; 36k prefill verified on live cluster |
| `mlx Fence::wait` GPU-timeout race (decode) | ⚠️ **Mitigated** (in-repo) | Decode stall watchdog bounds the hang; definitive fix needs newer mlx-jaccl (outside repo) |
| Prefill OOM at high token counts | ✅ **Addressed** | `preflight_or_raise` admission guard shipped |
| mlx-jaccl pin (Phase 1) | ✅ **Done** | `pyproject.toml` → `rltakashige/mlx-jaccl-fix-small-recv @ address-rdma-gpu-locks` |

**The cluster loads and serves long contexts.** A 35,969-token prefill
completed successfully (11:44→11:47) followed by decode. The runner process
is idle-healthy (parked on `_PySemaphore_Wait`, not wedged in a collective).

---

## 1. JACCL `all_sum` rendezvous race (model-load) — FIXED

### What it was
The in-process warmup `all_sum(mx.array(1.0))` in `utils_mlx.py` could
complete unilaterally on rank 0 (the coordinator) in ~1ms before rank 1
joined the group. Rank 1 then hung on `IOSurfaceSharedEvent
waitUntilSignaledValue` waiting for a partner that had already torn down its
side. Model never loaded; dashboard stuck on PREPARING.

### The fix (two parts)
1. **Removed the in-process warmup `all_sum`** from `utils_mlx.py`
   (commit `17252452`). The RDMA data path is validated exclusively by
   `_probe_rdma_interface`, which runs a full `all_sum` in a **subprocess**
   with both ranks participating simultaneously — sidestepping the in-process
   coordinator race entirely.
2. **Master-side lifecycle gating** in `plan.py`: rank 0 is not sent
   `LoadModel` until rank 1 has reached `RunnerConnected`. This guarantees
   both ranks are past distributed init before any model-load collective runs.

### Why the retry-loop approach was rejected
The original proposal was a value-checking retry loop (repeat `all_sum(1.0)`
until `result == world_size`). This was rejected because the coordinator's
degenerate 1ms return is **cached and deterministic** — it repeats on every
retry without re-exchanging. Only the master's out-of-band lifecycle gating
can force a real rendezvous.

### Verification
- Both nodes run commit `42762e44` (includes the `17252452` fix).
- Model load succeeds; 35,969-token prefill completed (11:44→11:47).
- No `Fence::wait`/`IOSurface`/rendezvous-hang in recent logs.
- Runner process (PID 88334) sampled: idle on `_PySemaphore_Wait` (queue
  wait), tokenizer rayon threads asleep — not wedged.

See `../omlx-porting/02-native-metal-kernels-DONE.md` Issue #3 for the full
post-mortem.

---

## 2. `mlx Fence::wait` GPU-timeout race (decode) — MITIGATED

### What it is
At the prefill→decode transition, mlx's `Fence::wait` can time out if a
peer's Metal command buffer stalls. The collective blocks the **C++ thread**,
so no Python-level timeout can interrupt it. If rank 1's `Fence::wait` hangs,
rank 0's per-token collective (`agree_on_cancellations` → `all_gather`, fired
every `check_for_cancel_every` tokens) also hangs — both ranks stop emitting
chunks and the API client hangs forever.

### Why it can't be fixed from exo Python
The root cause is in the mlx fork (`rltakashige/mlx-jaccl-fix-small-recv`).
The collective blocks the C++ thread; an in-process watchdog is also blocked.
A definitive fix requires a newer mlx-jaccl commit (past `e9835615`, which
introduces its own deadlock) + cluster validation — both outside this repo.

### The mitigation (in-repo): decode stall watchdog
Added to `src/exo/api/main.py::_token_chunk_stream` (the central async
generator all streaming responses flow through). The master/API is a
**separate process** from the GPU-blocked runner, so it can observe the
absence of chunks:

- `EXO_DECODE_STALL_TIMEOUT` (default 120s): after decode starts, if no chunk
  arrives within this window, the request fails fast with an `ErrorChunk`
  ("decode stalled... please retry") instead of hanging forever.
- `EXO_PREFILL_STALL_TIMEOUT` (default 180s): bounds the wait before decode
  starts (prefill progress heartbeat or first decode token).
- Set either to `0` to disable.

This doesn't unblock the GPU, but it **bounds the hang for the user**: the
client gets an error and can retry, rather than waiting indefinitely.

### Current status on the live cluster
Not reproduced in recent logs (Aug 24). The Aug 21 `stderr.log` crashes were
**prefill-time GPU OOM** (`kIOGPUCommandBufferCallbackErrorOutOfMemory`) +
ring transport aborts — a different failure mode, addressed by the preflight
admission guard (see below). The decode `Fence::wait` race remains a
theoretical risk on the highest token counts (~45–52k ceiling).

See `../exo-upstream-porting/06-glm-long-context-rdma-reliability.md` for the
phased plan and upstream PR landscape.

---

## 3. Prefill OOM at high token counts — ADDRESSED

### What it was
At ~45–52k tokens, prefill peak activations exceeded available GPU memory →
`kIOGPUCommandBufferCallbackErrorOutOfMemory` crash. Recorded in the Aug 21
`stderr.log` (2 OOM crashes, 2 ring aborts).

### The fix
`preflight_or_raise` admission guard in `batch_generate.py`: before prefill,
rejects a prompt whose prefill peak won't fit (after prefix-cache headroom
eviction) with a clean error instead of an OOM crash. Shipped in the running
build.

---

## 4. What remains (outside this repo)

- **Definitive `Fence::wait` fix:** a newer mlx-jaccl commit past `e9835615`
  (which deadlocks) + cluster validation. Coordinate with @rltakashige.
- **52k-token reproducer test:** hardware-gated 2-node test
  (`tests/test_2node.py`) exercising the ~45–52k agentic prompt path. Cannot
  run in CI.
- **Multi-rail/bond RDMA** (#2160, #2195), **TB5 discovery** (#2063),
  **inactivity recovery** (#1973): upstream EXO PRs not yet ported.

---

## Key files

- `src/exo/api/main.py::_token_chunk_stream` — decode stall watchdog
- `src/exo/shared/constants.py` — `EXO_DECODE_STALL_TIMEOUT` / `EXO_PREFILL_STALL_TIMEOUT`
- `src/exo/worker/engines/mlx/utils_mlx.py` — distributed init (warmup removed)
- `src/exo/worker/plan.py` — master lifecycle gating (rendezvous enforcement)
- `src/exo/worker/engines/mlx/generator/batch_generate.py` — `preflight_or_raise` admission guard
- `pyproject.toml` — mlx-jaccl pin