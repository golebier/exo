# 03 — Multi-Token Prediction (MTP) Speculative Decode

**Tier:** ⭐⭐ (Tier 2)
**Effort:** Medium-high
**Impact:** Medium-high (~1.3–1.74× decode throughput on supported models)
**oMLX source:** `omlx/patches/mlx_lm_mtp/`
**EXO target:** `src/exo/worker/engines/mlx/generator/batch_generate.py`, `generate.py`, new `src/exo/worker/engines/mlx/patches/mtp/`

---

## What it is

MTP (Multi-Token Prediction) heads are extra model modules that predict multiple
future tokens in one forward pass. A **draft/verify loop** uses them for
speculative decoding: the MTP head drafts a token, the backbone verifies it in a
2-token forward, accepted drafts are emitted "for free."

oMLX ports two upstream mlx-lm PRs into runtime monkey-patches:
- **ml-explore/mlx-lm#990** — Qwen3.5 / Qwen3.6 native MTP heads (dense + MoE)
- **Blaizzy/mlx-lm#15** — DeepSeek-V4-Flash native MTP heads

Both follow the same shape: the model gains an `mtp` module + `mtp_forward`
method + an enhanced `__call__` returning hidden states alongside logits. A
separate `mtp_generate_step` drives the draft/verify loop.

---

## Throughput math (from oMLX `batch_generator.py` docstring)

Greedy, accept rate `p`:
- Cost per cycle: 1× backbone (2-token verify) + 1× MTP head ≈ **1.15**
- Tokens per cycle: `1 + p` (accept → draft+bonus; reject → verify_pred only)
- At `p≈1`: **0.575 cost/token → ~1.74× throughput**
- At `p≈0.5`: **~0.77 cost/token → ~1.30× throughput**

**Known limitation (compute-bound single-stream Apple Silicon):** the math
assumes the 2-token verify is nearly free relative to a 1-token forward
(bandwidth-bound regime). On lower-end single-stream parts (M1/M2 base/Pro)
decode is compute-bound, so verify costs ~2× and **MTP can be net-negative
regardless of accept rate**. Wins expected on M3/M4+, MoE models (smaller
per-step backbone), or under continuous batching with spare compute.

---

## oMLX design

### Activation gating
`mlx_lm_mtp/__init__.py`:
- `_MTP_ACTIVE` global, set by `set_mtp_active(bool)` before `mlx_lm.load()`.
- `_MTP_DEPTH` (draft depth; >1 only for models marked `_omlx_mtp_chain`).
- Per-instance marker `_omlx_mtp_decode_enabled` set at load time, read by the
  BatchGenerator at decode time — so later model loads can't change existing
  models. Race-free because MLX executor serializes loads.

Caller (`utils/model_loading.py`) checks `model_settings.mtp_enabled` + the
model's `config.json` for MTP heads + supported `model_type` before invoking
`apply_mlx_lm_mtp_patch()`. Patches are **idempotent**.

### Batch integration (`mlx_lm_mtp/batch_generator.py`)
Patches `mlx_lm.generate.GenerationBatch`:
- `__init__` — untouched (fresh singleton batches may merge into larger
  continuous batches; MTP must not mutate cache state here).
- `next` — when the batch holds exactly one MTP-capable sequence, lazily init
  MTP from the post-prefill state. Emit from per-batch queue first; once empty,
  run a 2-token verify over `[next_main, draft]` with `n_confirmed=1` and a
  single MTP-head forward at the bonus position (accept) or confirmed position
  (reject), refilling the queue from verify outputs.
- `extend` / `filter` — drop MTP state whenever continuous batching reshapes
  ownership. MTP state belongs to one uid in one singleton timeline.

### Concurrency model
- Singleton MTP path for one active sequence.
- Row-wise MTP controller for multi-sequence batches **only when every row is at
  the same target cache position**. Late-join / unaligned batches fall through
  to standard continuous batching.

### Acceptance
- **Greedy (sampler is None):** patched dispatch produces identical tokens to
  the standard step (PR 990's `test_mtp_generate_identity` encodes this).
- **Stochastic (sampler not None):** `min(1, p_target / p_draft)`
  (Leviathan & Chen 2023); on rejection sample from residual
  `max(p_target - p_draft, 0) / Z` so marginal output distribution is exact.

### Cache interaction
- `cache.trim(1)` on `BatchKVCache` only updates `self._idx`; underlying paged
  blocks untouched.
- `ArraysCache.rollback_state` holds `(conv_snap, ssm_snap)` snapshots produced
  by patched `GatedDeltaNet.__call__`, restored on reject.
- Both paths only mutate cache *length*, not block ownership → oMLX's
  `PagedCacheManager` is oblivious; prefix-cache lookups continue normally.

### Files in `omlx/patches/mlx_lm_mtp/`
```
__init__.py                  # activation flags, apply_mlx_lm_mtp_patch
batch_generator.py           # GenerationBatch.next/extend/filter patches
cache_rollback.py            # ArraysCache rollback snapshots
deepseek_v4_dspark.py        # DeepSeek-V4 model-side hooks
deepseek_v4_model.py
gemma4_text_model.py
glm_moe_dsa_model.py         # GLM-5.2 MTP (if applicable)
nemotron_h_chain.py          # Nemotron-H hybrid-cache MTP
nemotron_h_model.py
norm_repair.py
prompt_priming.py
qwen35_model.py              # Qwen3.5 model-side hooks
step3p7_model.py
```

---

## EXO current state

- `batch_generate.py` uses mlx-lm `BatchGenerator` directly (via `MlxBatchGenerator`).
- `generate.py` uses `stream_generate` from mlx-lm.
- EXO has no MTP support and no per-model speculative decode.
- EXO's `KVPrefixCache` handles `ArraysCache` rollback via `snapshot_ssm_states`
  — the same primitive oMLX's `cache_rollback.py` uses, so the cache side is
  compatible.

---

## Integration seam in EXO

- **Patch application:** add to `src/exo/worker/engines/mlx/patches/__init__.py`
  alongside `apply_glm_moe_dsa_patch()`. Respect bootstrap ordering
  (`bootstrap.py:75-77`).
- **Activation:** gate on a per-model setting (EXO would need a
  `ModelSettings`-equivalent — see doc 05) + `config.json` MTP-head detection.
  Until EXO has per-model settings, use an env var `EXO_MTP_ENABLED=1` +
  auto-detect MTP heads in config.
- **Model-side hooks:** port the relevant `*_model.py` for each family EXO
  supports (start with Qwen3.5).
- **Batch patch:** port `batch_generator.py`'s `GenerationBatch` patches. EXO's
  `batch_generate.py` imports `MlxBatchGenerator` — patch the same symbols.

---

## Phased plan

### Phase 1 — Qwen3.5 MTP (the primary target)
- Port `mlx_lm_mtp/__init__.py` (flags + `apply_mlx_lm_mtp_patch`).
- Port `qwen35_model.py` model-side hooks.
- Port `batch_generator.py` singleton path (one sequence).
- Port `cache_rollback.py` for `ArraysCache` (EXO already has
  `snapshot_ssm_states`; adapt).
- Env-gate: `EXO_MTP_ENABLED=1` + config auto-detect.
- **Tests:** greedy-identity test vs standard step (match oMLX's
  `test_mlx_lm_mtp_patch.py`); stochastic distribution test; rollback-on-reject
  cache integrity test.

### Phase 2 — Multi-sequence row-wise controller
- Port the aligned-batch row-wise MTP path.
- Ensure late-join falls through correctly (EXO's continuous batching admits new
  sequences mid-flight).
- **Tests:** mixed alignment batch; late-join during MTP.

### Phase 3 — DeepSeek-V4 / other families
- Port `deepseek_v4_dspark.py` + `deepseek_v4_model.py`.
- Port GLM-5.2 MTP if/when applicable (`glm_moe_dsa_model.py`).

### Phase 4 — Auto-disable on compute-bound hardware
- Detect part class (M1/M2 base/Pro → disable; M3/M4+ → enable) and default off
  to avoid the net-negative regime oMLX warns about.
- Surface the decision in logs.

---

## Risks & open questions

- **Net-negative on low-end hardware:** must default-off or auto-detect. Don't
  ship enabled-by-default on all Apple Silicon.
- **Continuous-batch interaction:** EXO's `batch_generate.py` admits new
  sequences mid-flight. The oMLX patch drops MTP state on `extend`/`filter`;
  verify EXO's admission path triggers those.
- **`ArraysCache` rollback correctness:** EXO's `snapshot_ssm_states` deep-copies;
  oMLX's `rollback_state` uses lighter snapshots. Decide which to use — correctness
  over speed first.
- **Cache-block interaction:** if EXO ports the paged cache (doc 01) first, MTP
  must coexist. oMLX's docstring asserts paged-block ownership is unaffected;
  re-verify with EXO's media-region-aware blocks.
- **Sampler parity:** EXO's `make_sampler` / `make_logits_processors` come from
  mlx-lm; the stochastic acceptance math must compose with EXO's
  `ban_token_ids` / stop-sequence logic in `generate.py`.

---

## Definition of done

- [ ] Qwen3.5 MTP greedy-identity test passes (output bit-identical to standard
      step with sampler=None).
- [ ] Stochastic acceptance distribution test passes (marginal ≈ target within
      sampling tol).
- [ ] Cache integrity after reject (rollback restores `ArraysCache` exactly).
- [ ] Decode tok/s benchmark on M3/M4+ shows ≥1.2× on Qwen3.5 with MTP.
- [ ] Auto-disable confirmed on M1/M2 base (no regression vs standard decode).
- [ ] `basedpyright` + `ruff` + `nix fmt` + `pytest` clean.