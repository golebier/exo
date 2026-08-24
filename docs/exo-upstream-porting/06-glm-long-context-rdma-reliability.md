# 06 — GLM-5.2 Long-Context Fix + RDMA/Thunderbolt Reliability

**Tier:** ⭐⭐ (Tier 2)
**Effort:** Medium (mlx-jaccl fix + multi-rail/bond RDMA)
**Impact:** Very high for the 2-node+ sharded use case (the flagship config)
**Upstream evidence:**
- #2208 `Long-context requests to 2-node Tensor/MlxJaccl instances hang or produce garbled output (mlx Fence::wait GPU-timeout race)`
- RDMA/TB reliability cluster: #1847, #1390, #2067, #1973, #2219, #1459, #1320, #1701, #1534, #1887
- PRs: #2160 `stripe RDMA traffic across multiple Thunderbolt cables (multi-rail jaccl)`, #2195 `bond parallel RDMA links in JACCL device matrix`, #2063 `Thunderbolt 5 + nested-hub discovery; JACCL init retry; RDMA host selection`

**Cross-reference:** `../gra/GLM-5.2-RESEARCH-RESULTS.md` (local — GLM model side verified correct). This doc covers the **networking/RDMA** side, which the local fork has *not* fixed.

---

## What it is

The flagship EXO use case — a big model sharded across 2+ Mac Studios over Thunderbolt/RDMA — is **broken for long contexts**. #2208 is the definitive report:

- 3× Mac Studio M3 Ultra 512GB, Thunderbolt full mesh, GLM-5.2-mxfp4 (78 layers, 395GB), 2-node Tensor/JACCL sharding.
- ~45k–52k-token agentic prompts (large system prompt + tool schemas + history, as Claude Code/Pi send).
- Prefill proceeds normally in 4096-token chunks (40–80% CPU on both shards).
- **At the prefill→decode transition** (the single expensive forward over the entire KV cache, requiring an RDMA all-reduce between ranks) the request either:
  - **hangs indefinitely** near idle (52,417-token prompt stalled at 49,152/52,417 — 94% — zero output for 60s+), or
  - **returns garbled/corrupted** output.
- Short prompts (dashboard chat) never trigger it → "the dashboard works but every other client is broken."

**Root cause:** traces to **mlx itself** (`rltakashige/mlx-jaccl-fix-small-recv` @ `address-rdma-gpu-locks`), not exo's Python code. Specifically a `mlx Fence::wait` GPU-timeout race. A candidate fix (commit `e9835615`) introduces its own deadlock.

The local fork fixed the **GLM model side** (indexer sharing — see `../gra/GLM-5.2-RESEARCH-RESULTS.md`), but the **RDMA/mtx race is still present**.

---

## The broader RDMA/TB reliability cluster

#2208 is the headline, but there's a large cluster of RDMA/Thunderbolt pain:

| Issue | Symptom |
|-------|---------|
| #1847 | jaccl RDMA crashes on Mac Studio M3 Ultra (errno=2, 60, 22) — recurring |
| #1390 | RDMA fails on macOS when running from source |
| #2067 | Models fail to load / crash after load with RDMA 5 + M4 Max |
| #1973 | RDMA ports stop working after inactivity |
| #2219 | MLX TCP ring: 5-node instance wedges on first real request (warmup succeeds) |
| #1459 | No valid Tensor+RDMA config for 3 nodes |
| #1320 | Interest in Linux RDMA device discovery |
| #1701 | TB5 nodes can't see each other unless on same subnet |
| #1534 | Not seeing Mac minis/Studios — no firewall |
| #1887 | Linux undiscoverable to Mac cluster |

Themes: RDMA crashes/timeouts, inactivity failures, discovery across subnets/OSes, 3-node configs broken.

---

## Upstream PR landscape

| PR | Fix |
|----|-----|
| #2160 | **Stripe RDMA traffic across multiple Thunderbolt cables** (multi-rail jaccl) — bandwidth + redundancy |
| #2195 | **Bond parallel RDMA links** in JACCL device matrix — failover across links |
| #2063 | **TB5 + nested-hub discovery**; JACCL init retry; RDMA host selection — discovery + init robustness |

#2160 + #2195 together address the bandwidth/reliability side (multi-rail + bond = if one TB link drops, traffic continues). #2063 addresses discovery/init (TB5 hubs, retry on init failure). None directly fix the `Fence::wait` race — that's a mlx-jaccl upstream fix (`rltakashige/mlx-jaccl-fix-small-recv`).

---

## EXO current state (local fork)

- **GLM model side: fixed + verified** (`vendor/glm_moe_dsa/`, indexer sharing — see `../gra/GLM-5.2-RESEARCH-RESULTS.md`). The "garbled output" half of #2208 (shared-layer random indexer weights) is addressed.
- **mlx-jaccl pin: DONE.** `pyproject.toml` pins mlx to `rltakashige/mlx-jaccl-fix-small-recv @ address-rdma-gpu-locks` (Phase 1 pin complete).
- **Prefill OOM: addressed.** The preflight admission guard (`preflight_or_raise`) rejects a prompt whose prefill peak won't fit instead of crashing with `kIOGPUCommandBufferCallbackErrorOutOfMemory`. Verified shipped in the running build.
- **Model-load rendezvous race (doc 02 Issue #3): fixed.** In-process warmup `all_sum` removed (commit `17252452`); master lifecycle gating in `plan.py` enforces rendezvous. See `../omlx-porting/02-native-metal-kernels-DONE.md` Issue #3.
- **`Fence::wait` decode race: mitigated (in-repo), not eliminated.** The mlx `Fence::wait` GPU-timeout race at prefill→decode is a mlx-fork root cause that cannot be fixed from exo Python (the collective blocks the C++ thread; no Python-level timeout can interrupt it). A **decode stall watchdog** (`EXO_DECODE_STALL_TIMEOUT` / `EXO_PREFILL_STALL_TIMEOUT` in `src/exo/api/main.py::_token_chunk_stream`) now bounds the hang from the master/API side (a separate process): if no chunk arrives within the timeout, the request fails fast with an error so the client can retry. A definitive fix requires a newer mlx-jaccl commit (past `e9835615` which introduces its own deadlock) + cluster validation — both outside this repo.
- Has the `disaggregated/` + `remote_prefill.py` path (local ahead) — but that's a *different* prefill path; the hang is in the TP-collective path.
- zenoh migration done (local + upstream #2132).
- No multi-rail/bond RDMA (#2160, #2195 not ported).
- No TB5 nested-hub discovery (#2063 not ported).

---

## Integration seam

- **mlx-jaccl fix:** pin mlx to `rltakashige/mlx-jaccl-fix-small-recv` @ `address-rdma-gpu-locks` (or the merged equivalent) for the JACCL/RDMA path. This is a dependency pin, not exo code. Verify it doesn't regress the non-RDMA path.
- **Multi-rail (#2160):** stripe a single collective's traffic across multiple TB cables between the same pair of nodes. Requires JACCL to expose multiple links between peers.
- **Bond (#2195):** treat parallel RDMA links as one bonded device with failover. If one link errors (#1847 errno crashes), traffic continues on the others.
- **TB5 discovery (#2063):** nested-hub topology discovery (TB5 docks/chains), JACCL init retry on transient failure, RDMA host selection.
- **Inactivity recovery (#1973):** heartbeat/keep-alive on RDMA links; re-init on idle timeout.
- **Cross-subnet discovery (#1701, #1887):** don't require same-subnet for TB-discovered peers; allow explicit peer dialing (cf. PR #2243 `EXO_ZENOH_CONNECT` unicast).

---

## Phased plan

### Phase 1 — Reproduce #2208 + pin mlx-jaccl fix
- ~~Reproduce the 52k-token 2-node JACCL hang on the local fork.~~
- ~~Pin the mlx-jaccl fix; verify the hang resolves without introducing the `e9835615` deadlock.~~ ✅ **Pin done** (`pyproject.toml` → `rltakashige/mlx-jaccl-fix-small-recv @ address-rdma-gpu-locks`).
- **Mitigation in place:** decode stall watchdog (`EXO_DECODE_STALL_TIMEOUT`) bounds the `Fence::wait` hang from the API side; the cluster loads and serves long contexts (35,969-token prefill verified). A definitive mlx-fork fix remains open.
- **Tests:** long-context (52k token) 2-node TP test — TODO (hardware-gated; see `tests/test_2node.py`).

### Phase 2 — Multi-rail + bond (port #2160, #2195)
- Stripe collectives across multiple TB links between a peer pair.
- Bond links with failover; survive a single-link drop.
- **Tests:** 2-node with 2 TB links; kill one link mid-generation → continues; bandwidth scales with link count.

### Phase 3 — Discovery robustness (port #2063)
- TB5 nested-hub discovery; JACCL init retry.
- **Tests:** nested-hub topology discovery; transient init failure retried.

### Phase 4 — Inactivity + cross-subnet (#1973, #1701, #1887)
- RDMA link keep-alive + re-init on idle.
- Cross-subnet TB peer dialing (compose with #2243 unicast).
- **Tests:** idle-then-request recovers; cross-subnet peer discovered.

---

## Risks & open questions

- **mlx-jaccl pin:** pinning a fork of mlx is operationally heavy. Track upstream mlx for the official fix and un-pin ASAP.
- **`e9835615` deadlock:** the candidate fix introduces its own deadlock. Need the *next* fix, or a config that avoids the deadlock window. This is the crux — coordinate with @rltakashige (the mlx-jaccl author, also active in #2079/#2061).
- **Multi-rail ordering:** striping a collective across links can reorder arrivals; verify MLX distributed tolerates this or add re-sequencing.
- **Bond failover latency:** if failover triggers mid-collective, does the collective retry or fail? Define the contract.
- **Compose with ring (doc 02):** ring attention uses MLX distributed send/recv over the ring; the `Fence::wait` race affects this too. The mlx-jaccl fix should help ring as well — verify.
- **Reproduce cost:** 2× Mac Studio M3 Ultra 512GB is expensive hardware. The reproducer needs that environment; CI can't cover it. Mark as hardware-gated.

---

## Definition of done

- [x] Phase 1 (partial): mlx-jaccl pin applied; cluster loads + serves long contexts (36k prefill verified); decode stall watchdog mitigates the `Fence::wait` hang. **Open:** 52k reproducer + definitive mlx-fork fix (past `e9835615`).
- [ ] Phase 2: multi-rail stripes; bond survives single-link drop; bandwidth scales.
- [ ] Phase 3: TB5 nested-hub discovery works; init retry recovers transient failure.
- [ ] Phase 4: idle-then-request recovers; cross-subnet peer discovered.
- [ ] Ring attention (doc 02) also stable on the mlx-jaccl pin.
- [ ] `basedpyright` + `ruff` + `nix fmt` + `pytest` clean (hardware tests gated).