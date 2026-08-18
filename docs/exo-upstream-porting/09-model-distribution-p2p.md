# 09 — Peer-to-Peer / Thunderbolt Model Distribution

**Tier:** ⭐ (Tier 3)
**Effort:** Medium
**Impact:** Medium-high for multi-node clusters (saves redundant multi-hundred-GB downloads)
**Upstream evidence:**
- #721 `Feature Request - Add Rsync` (11👍)
- #1257 `Why should all nodes download the model directly from Hugging Face?`
- PR #1992 `feat: peer-to-peer model distribution` (adurham)
- #2117 `Share model via thunderbolt instead of re-download`
- #2050 `Option to disable launch at login and automatic browser opening` (tangential UX)

---

## What it is

Today every EXO node downloads the model from HuggingFace independently. For a 395GB model (#2208 env) across 3 nodes, that's ~1.2TB of HF downloads — slow, rate-limited (#2009 "targeted tweaks to address HF rate limits" was merged), and wasteful when the nodes are on a fast Thunderbolt/LAN mesh.

The ask: **one node downloads, others pull from it** over TB/LAN (rsync, p2p, or TB share). Four issues + one PR converge on this.

---

## Why it matters

- **Download is the #1 onboarding pain** for multi-node clusters. 3× redundant 395GB downloads (plus HF rate limits) make cluster setup miserable.
- **Thunderbolt mesh is fast** (40–80 Gbps) — pulling from a peer is often faster than from HF.
- **Ties to unified catalog (doc 08):** content-based resolution tells nodes "this peer has the exact model you need" — no re-download.
- **Ties to disaggregated prefill (local fork's strength):** once a peer has the weights, it can also serve prefill.

---

## Upstream PR landscape

| PR/Issue | Approach |
|----------|----------|
| #1992 | **peer-to-peer model distribution** (adurham) — the core PR |
| #721 | rsync (simplest; 11👍) |
| #2117 | share via Thunderbolt specifically |
| #1257 | the core "why all nodes DL" complaint |
| #2009 (merged) | HF rate-limit tweaks (mitigation, not solution) |

#1992 is the PR to review; #721 (rsync) is the minimal fallback.

---

## EXO current state (local fork)

- `src/exo/download/` — HF-focused download (`download_utils.py`, `test_safetensors_index.py`).
- No p2p / peer distribution.
- Has disaggregated prefill (ships *KV*, not weights) — a different but related path.
- zenoh networking in place (could carry weight chunks).

---

## Integration seam

- **Seed node:** the node that downloads from HF becomes the seed; others pull from it.
- **Transport:** rsync (#721, simplest) over SSH, or zenoh-based chunk transfer (compose with existing transport), or TB-direct.
- **Content verification:** reuse content-based resolution (doc 08) — peer advertises a model by content hash; requester verifies after transfer.
- **Shard-level:** for sharded models, a node only needs its shard — distribute only the needed weight files (compose with `auto_parallel.py` shard plan).
- **Catalog:** the unified catalog (doc 08) shows which peers have which model.

---

## Phased plan

### Phase 1 — rsync fallback (#721)
- Add an `exo sync-model <model> <peer>` CLI that rsyncs the model dir from a peer.
- Document the manual seed-peer workflow.
- **Tests:** rsync round-trip; content hash matches after sync.

### Phase 2 — P2P distribution (port #1992)
- Automatic peer discovery of model availability (via zenoh/state).
- Node requesting a model pulls from a peer if available, else HF.
- **Tests:** 3-node cluster, 1 downloads, 2 pull from it; no HF hit on the 2.

### Phase 3 — Shard-aware distribution
- Distribute only the weight files each node needs for its shard.
- Compose with `auto_parallel.py` shard plan.
- **Tests:** sharded model, each node receives only its shard files.

### Phase 4 — Thunderbolt-fast path (#2117)
- Prefer TB-connected peers for transfer (lowest latency, highest bandwidth).
- **Tests:** TB peer preferred over LAN peer.

---

## Risks & open questions

- **Auth/SSH:** rsync needs SSH trust between nodes. EXO's cluster already does TOFU-style trust (zenoh); align the rsync path or use zenoh transport to avoid SSH setup.
- **Partial downloads / resume:** large transfers must resume on interruption. rsync handles this; a zenoh path must too.
- **Disk space on seed:** seed must hold the full model even if it only serves a shard. Acceptable (seed is usually the master).
- **Concurrency:** multiple peers pulling simultaneously from one seed — throttle to avoid saturating the seed.
- **Hash verification cost:** re-hashing a 395GB model after transfer is slow; hash per-file (safetensors index already exists — `test_safetensors_index.py`).

---

## Definition of done

- [ ] Phase 1: `exo sync-model` rsync works; documented.
- [ ] Phase 2: automatic p2p; no redundant HF downloads in a 3-node test.
- [ ] Phase 3: shard-aware (only needed files transferred).
- [ ] Phase 4: TB peer preferred.
- [ ] `basedpyright` + `ruff` + `nix fmt` + `pytest` clean.