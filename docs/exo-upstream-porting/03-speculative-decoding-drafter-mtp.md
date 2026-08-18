# 03 — Speculative Decoding: Drafter Abstraction + MTP + DFlash

**Tier:** ⭐⭐⭐ (Tier 1)
**Effort:** Medium-high (large PR to adapt; ties to oMLX MTP)
**Impact:** Very high (3.3–4.3× decode TPS measured)
**Upstream evidence:**
- PR #2079 `Drafter abstraction + Gemma 4 MTP + Qwen 3.5/3.6 DFlash + multi-device coupled-drafter speculative decoding` (team-wcv)
- PR #2110 `Add native MTP for Qwen3.6 MLX models` (ffrappo)
- Issue #1685 `Speculative Speculative Decoding` (3👍)
- Upstream maintainer signal (in #2079): @rltakashige: "We're also looking into MTP, and we'll handle the issues you've mentioned at the same time as well as use something like this."

**Cross-reference:** `docs/omlx-porting/03-multi-token-prediction.md` (oMLX's MTP). **Reconcile the two into one design.**

---

## What it is

Speculative decoding uses a small **drafter** model to propose tokens that the target model verifies in a batched forward, accepting multiple tokens per step. PR #2079 lands a full stack:

1. **Drafter abstraction** — a general `drafter_model_id` ModelCard foundation; in-process tuning; asymmetric pipelined drafter.
2. **Gemma 4 MTP** — multi-token prediction heads (model-native speculative).
3. **Qwen 3.5 / 3.6 DFlash coupled drafters** — a coupled-drafter variant for Qwen DFlash.
4. **Multi-device tensor-parallel coupled-drafter dispatch** — speculative decode that works *across* a TP-sharded model.

The drafter abstraction is the more general framing — it subsumes both MTP (native heads) and DFlash (separate drafter model), and works under TP sharding. This is strictly more capable than the oMLX MTP-only design.

---

## Measured impact (from #2079, M5 Max hardware)

### Single-device DFlash A/B (Qwen 3.5 / 3.6)
| Target | Quant | Arch | Target gen_tps | DFlash gen_tps | Speedup | Accept |
|--------|-------|------|---------------:|---------------:|--------:|-------:|
| Qwen 3.5 4B | 8bit | dense | 97.24 | 404.38 | **4.16×** | 93.2% |
| Qwen 3.6 27B | 8bit | dense | 14.98 | 49.13 | **3.28×** | 92.6% |
| Qwen 3.6 35B-A3B | 8bit | MoE | 87.70 | 377.49 | **4.30×** | 92.6% |

### Multi-device DFlash (Qwen 3.5 122B-A10B, 2-node TP + JACCL/RDMA)
Also demonstrates the coupled drafter working under TP sharding across nodes.

These are the largest decode-speedup numbers available anywhere in the EXO ecosystem.

---

## Why it matters

- **Decode is the steady-state cost** of any chat/agentic workload. 3–4× TPS is transformative for UX.
- **MoE wins are largest** (4.30× on Qwen 3.6 35B-A3B) because the drafter avoids firing experts for rejected tokens — MoE's per-token cost is high, so speculation saves the most.
- **Multi-device TP support** means this isn't single-node-only — it works on the sharded clusters that are EXO's reason to exist.
- **Generalizes oMLX's MTP:** the oMLX doc (`docs/omlx-porting/03-...`) ports only native MTP heads. #2079's drafter abstraction supports MTP *and* separate drafter models *and* DFlash, under TP. Adopt #2079 as the design; treat the oMLX MTP doc as a reference for the native-heads sub-path.

---

## Upstream PR design (from #2079 body)

- **Foundation:** `drafter_model_id` ModelCard field (from the closed #2065, included as first commit `f383ef0a`).
- **Drafter abstraction:** pluggable drafter — MTP head (model-native) OR separate drafter model OR DFlash coupled.
- **In-process tuning:** calibrate the drafter at runtime.
- **Asymmetric pipelined drafter:** drafter runs ahead in a pipeline with the target.
- **Production hardening:** rollback on reject (cache trim; `ArraysCache` snapshot/restore — EXO already has `snapshot_ssm_states`/`copy_snapshot_entry`).
- **Gemma 4 MTP:** native MTP heads for Gemma 4.
- **Qwen 3.5/3.6 DFlash:** coupled drafter variant.
- **Multi-device TP coupled-drafter dispatch:** drafter cooperates with the TP collective so verify forwards are batched across ranks.

PR is **self-contained** (two commits: foundation + bundle) and explicitly invites cherry-picking: "feel free to cherry-pick whatever pieces are useful for upstream's own MTP work."

---

## EXO current state (local fork)

- **No speculative decoding, no drafter abstraction.** `rg drafter|speculative|dflash` returns nothing in `src/`.
- Has `KVPrefixCache` with `snapshot_ssm_states` / `trim_cache` — the exact rollback primitives the drafter needs (oMLX's `cache_rollback.py` uses the same pattern). **Cache side is ready.**
- Has `auto_parallel.py` TP sharding — the multi-device TP coupled drafter must compose with this.
- Has `tool_parsers/glm52.py` and per-model generation patches — the drafter must not break tool-call streaming.

---

## Integration seam

- **ModelCard field:** add `drafter_model_id` (+ drafter type: `mtp` | `dflash` | `external`) to `src/exo/shared/models/model_cards.py`.
- **Drafter module:** new `src/exo/worker/engines/mlx/drafter/` (port from #2079) — abstraction, MTP head, DFlash, coupled-TP dispatch.
- **Generator:** hook the draft/verify loop into `batch_generate.py`'s `GenerationBatch.next` (same seam oMLX patches, see `docs/omlx-porting/03-...`).
- **Cache rollback:** reuse `snapshot_ssm_states` + `trim_cache` for reject rollback. Verify `ArraysCache` snapshot/restore is correct under the drafter (oMLX's `rollback_state` is the reference).
- **TP composition:** the coupled-drafter dispatch must agree with `auto_parallel.py`'s TP collectives — drafter verify forward batched across ranks. Port #2079's multi-device path.
- **Activation gating:** per-model `drafter_model_id` + `config.json` MTP-head detection (for the MTP sub-path). Env override `EXO_DRAFTER_ENABLED` until stable.

---

## Phased plan

### Phase 1 — Drafter abstraction + single-device DFlash (Qwen 3.5/3.6)
- Port #2079's foundation commit (`drafter_model_id` ModelCard field).
- Port the drafter abstraction + DFlash coupled drafter for Qwen 3.5/3.6.
- Single-device draft/verify loop in `batch_generate.py` (singleton path).
- Cache rollback via existing `snapshot_ssm_states`.
- **Tests:** greedy-identity vs standard step; stochastic acceptance distribution; rollback-on-reject cache integrity; reproduce #2079's 4× TPS on Qwen 3.5 4B.

### Phase 2 — Native MTP heads (Gemma 4, Qwen3.6)
- Port the MTP sub-path (overlaps oMLX MTP doc). Gemma 4 MTP + Qwen3.6 native MTP (#2110).
- Decide: is #2110 (separate native-MTP PR) subsumed by #2079? Reconcile — likely #2079's abstraction wraps #2110's heads.
- **Tests:** MTP greedy identity; MTP under sampler.

### Phase 3 — Multi-device TP coupled drafter
- Port #2079's coupled-drafter TP dispatch.
- Compose with `auto_parallel.py` TP sharding + JACCL/RDMA.
- **Tests:** 2-node TP + drafter reproduce #2079's multi-device numbers; collective ordering (drafter doesn't deadlock TP, cf. #2048 TP collective deadlock fix).

### Phase 4 — Auto-disable on compute-bound hardware
- Like oMLX MTP, speculation can be net-negative on compute-bound single-stream parts (M1/M2 base). Detect & default-off there. (See oMLX doc 03 caveat.)
- Surface drafter accept-rate in logs/dashboard.

---

## Risks & open questions

- **PR size:** #2079 is a large bundle (drafter + MTP + DFlash + multi-device). Land in slices (Phase 1 DFlash-only first) rather than one big merge.
- **Reconcile with oMLX MTP doc:** don't implement two speculative paths. Adopt #2079's abstraction as the single design; the oMLX MTP port becomes the "native MTP head" sub-path within it.
- **Cache rollback correctness:** EXO's `snapshot_ssm_states` deep-copies; #2079/oMLX use lighter snapshots. Verify reject rollback restores `ArraysCache`/`DeepseekV4Cache` exactly — a subtle bug here corrupts silently.
- **TP collective deadlock:** #2048 fixed a TP collective deadlock via `agree_on_tasks` ordering. The coupled drafter adds another collective (verify forward) — ensure it composes with #2048's fix.
- **Tool-call streaming:** drafter must not emit draft tokens as visible content; tool-parser (`glm52.py`, etc.) must see only verified tokens. Test with tool-calling workloads.
- **MoE expert firing:** the 4.3× MoE win assumes rejected draft tokens don't fire experts. Verify the MLX MoE path actually skips experts for rejected positions (otherwise the win evaporates).

---

## Definition of done

- [ ] Phase 1: DFlash on Qwen 3.5 4B reproduces ≥3× TPS (target #2079's 4.16×); greedy-identity passes.
- [ ] Phase 2: Gemma 4 + Qwen3.6 native MTP under the same abstraction.
- [ ] Phase 3: 2-node TP + coupled drafter reproduces #2079's multi-device numbers; no TP deadlock.
- [ ] Phase 4: auto-disable confirmed on M1/M2 base (no regression).
- [ ] Tool-call streaming correct under drafter (verified tokens only).
- [ ] `basedpyright` + `ruff` + `nix fmt` + `pytest` clean.