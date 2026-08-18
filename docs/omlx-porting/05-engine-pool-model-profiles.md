# 05 — Multi-Model EnginePool + Model Profiles

**Tier:** ⭐⭐ (Tier 2)
**Effort:** Medium-high (pool), Low (profiles alone)
**Impact:** Medium
**oMLX source:** `omlx/engine_pool.py`, `omlx/model_profiles.py`, `omlx/model_settings.py`, `omlx/engine/{base,batched,embedding,reranker,vlm}.py`, `omlx/memory_monitor.py`
**EXO target:** `src/exo/worker/runner/` (new pool), `src/exo/api/`, `src/exo/shared/types/`

---

## What it is

Two related features that oMLX combines but EXO can adopt independently:

### A. Multi-model EnginePool
Serve **multiple models concurrently from one node** with:
- Pre-load memory checking (refuse to load what won't fit).
- **LRU eviction** of least-recently-used models when memory runs low.
- **Model pinning** — keep specific models always loaded.
- **Per-model TTL** — auto-unload after idle timeout.
- `ProcessMemoryEnforcer` — total memory limit (default: system RAM − 8GB),
  memory-guard tiers (`safe` / `balanced`).

### B. Model Profiles
Named bundles of per-model settings (sampling, chat-template kwargs, quant
knobs, thinking budget, …). A profile can be **exposed as its own model id**:
`/v1/models` lists `<model>:<profile>` (e.g. `qwen3-8b:thinking`), served on the
**same engine** as the base model with the profile's settings overlaid per
request — **zero extra memory, no reload**. When the base model has an alias,
the exposed id is `<alias>:<profile>`; the directory-name form keeps working.

---

## Why it fits EXO

EXO's current model is "master places **one** model instance across nodes." There's
no per-node hot multi-model pool, and no profile/alias concept. Two concrete wins:

1. **Profiles are nearly free** and dashboard-friendly. Serve `model:thinking`
   and `model:fast` from one loaded model — no second load, perfect for the
   EXO dashboard's model selector.
2. **A per-node pool** lets a single beefy node hold a coding model + a small
   fast model + an embedding model simultaneously, with LRU swapping — better
   UX than EXO's "one model at a time per cluster."

Profiles (B) can land **without** the full pool (A); they only need per-request
settings overlay on the already-loaded model.

---

## oMLX design

### `engine_pool.py` — `EnginePool`
Key methods (from analysis):
- `EngineEntry` — per-model state (model_id, path, type, engine_type, …).
- `discover_models(...)` — auto-detect LLM/VLM/embedding/reranker from dirs.
- `get_engine(model_id)` / `acquire(model_id)` / `release_engine(model_id)` —
  lease-based concurrency (in-flight requests block unload).
- `_find_lru_victim()` — pick eviction target.
- `set_pinned(model_id, pinned)` — pinning.
- `request_unload(model_id)` / `unload_if_idle_unpinned(model_id)` — TTL path.
- `_schedule_pending_unload_locked` / `_unload_pending_if_idle_locked` —
  deferred unload waiting for in-flight to drain.
- `_current_ceiling()` / `_admission_soft_target()` — memory admission control.
- `ProcessMemoryEnforcer` — background total-memory enforcement.

Engine types: `batched` (LLM), `vlm`, `embedding`, `reranker`, `audio_stt/tts/sts`,
plus `simple` and `dflash`.

### `model_profiles.py` — profile field model
```python
UNIVERSAL_PROFILE_FIELDS = (
    "max_context_window", "max_tokens", "temperature", "top_p", "top_k",
    "min_p", "repetition_penalty", "presence_penalty", "force_sampling",
    "enable_thinking", "preserve_thinking", "thinking_budget_enabled",
    "thinking_budget_tokens", "reasoning_parser", "guided_grammar_enabled",
    "guided_grammar", "max_tool_result_tokens", "chat_template_kwargs",
    "forced_ct_kwargs",
)
MODEL_SPECIFIC_PROFILE_FIELDS = (
    "turboquant_kv_enabled", "turboquant_kv_bits", "turboquant_skip_last",
    "qwen35_ane_prefill_enabled", "qwen35_ane_prefill_sequence_length",
    "qwen35_ane_prefill_fraction", ...
)
```
- `ModelProfile` dataclass (named bundle of allowed fields).
- `GlobalTemplate` (universal fields shared across models).
- `UNIVERSAL_FIELDS_SET | MODEL_SPECIFIC_PROFILE_FIELDS` = `PROFILE_FIELDS_SET`
  (allowlist; everything else excluded from profiles/templates).

### `model_settings.py` — `ModelSettings` + `ModelSettingsManager`
- `apply_profile(...)` — overlay a profile onto settings.
- `merge_chat_template_request_kwargs(...)` — request-level kwarg merge.
- `get_exposed_profile_source_model_id(...)` — resolve `<model>:<profile>` →
  physical model id (used by `EnginePool.resolve_model_id`).

### `engine_pool.py::resolve_model_id`
Resolution order: exact id → alias → cluster deployment id → **exposed profile
model ids** (resolve to the physical model they overlay) → case-insensitive match.

---

## EXO current state

- EXO's `Node` runs one model instance placed by the master across nodes
  (`src/exo/master/`, `src/exo/worker/plan.py`).
- `src/exo/shared/types/worker/downloads.py` + `src/exo/download/` handle model
  download.
- API: `src/exo/api/main.py` routes chat completions / responses / claude /
  ollama to the current model instance.
- No per-model settings object, no profile concept, no per-node multi-model.
- EXO does have an `_entry_resident_size` / `current_model_memory` analogue
  conceptually in `plan.py` placement, but not a hot runtime pool.

---

## Integration seam in EXO

This is the **most architecturally invasive** of the docs because it touches
EXO's master/worker placement model. Two scoping options:

### Scope 1 — Profiles only (recommended first; low risk)
- Add `ModelSettings` (sampling + chat-template kwargs + thinking budget +
  quant knobs) to EXO shared types.
- Add `ModelProfile` + a profile store (JSON under XDG config).
- In API request handling (`api/adapters/chat_completions.py` etc.), parse
  `<model>:<profile>` from the requested model id, resolve to the physical
  model, overlay profile settings on the request params before dispatch.
- The master/worker placement is untouched (still one physical model); profiles
  are a pure request-time overlay.
- Dashboard: list `<model>:<profile>` entries in the model selector; allow
  creating/editing profiles.

### Scope 2 — Per-node EnginePool (higher risk)
- A node may hold multiple model instances (subject to memory). The master's
  placement must learn "node already has model X resident" and prefer routing
  to resident models.
- This changes event-sourcing state (`src/exo/shared/types/state.py`,
  `src/exo/shared/apply.py`) to track per-node resident-model sets + LRU/pinned/TTL.
- New events: `ModelLoadedOnNode`, `ModelEvictedFromNode`, `ModelPinnedOnNode`.
- New commands from API: `LoadModelOnNode`, `UnloadModel`, `PinModel`.
- Lease/concurrency: in-flight requests block unload (port oMLX's
  `acquire`/`release` lease pattern).

Start with Scope 1. Only attempt Scope 2 once profiles are shipped and there's
measured demand for multi-model-per-node.

---

## Phased plan

### Phase 1 — ModelSettings + Profiles (Scope 1)
- Define `ModelSettings` in `src/exo/shared/types/` (Pydantic, frozen, strict —
  per AGENTS.md). Port the field allowlists from `model_profiles.py`, trimmed to
  what EXO supports.
- `ModelProfile` + `GlobalTemplate` dataclasses + JSON persistence.
- API: parse `<model>:<profile>` in request model id; resolve via
  `get_exposed_profile_source_model_id`-equivalent; overlay settings.
- Dashboard: profile CRUD UI.
- **Tests:** profile overlay correctness; `<model>:<profile>` resolution; alias
  + profile interaction; settings persistence round-trip.

### Phase 2 — Per-node memory accounting (prep for Scope 2, no behavior change)
- Add per-node resident-memory reporting to worker events (EXO already reports
  memory pressure via `get_memory_used_percentage`).
- Master tracks resident set per node (informational).

### Phase 3 — EnginePool (Scope 2)
- Worker-side `EnginePool` with LRU/pin/TTL + lease-based concurrency.
- New events/commands for load/unload/pin.
- Master routing prefers resident models; falls back to placement.
- `ProcessMemoryEnforcer`-equivalent (EXO already has `_MEMORY_THRESHOLD` for
  KV cache; generalize to total model memory).
- **Tests:** LRU eviction under load; pin prevents eviction; TTL unload after
  idle; lease blocks unload mid-request; OOM refusal.

---

## Risks & open questions

- **Placement-model coupling:** EXO's master places model instances across
  nodes for sharding. A per-node pool must not conflict with sharded placement
  (a sharded model spans nodes; a pooled model is per-node unsharded). Decide:
  pool applies only to unsharded (single-node) models; sharded models stay on
  the existing path.
- **Distributed memory:** EXO aggregates memory pressure across
  `mx.distributed.Group`. The pool's eviction must use *local* pressure for
  local eviction, not cluster-wide (otherwise a remote spike evicts a local
  model). oMLX is single-node here; EXO must adapt.
- **Profile field allowlist:** EXO's settings differ from oMLX's (no turboquant
  yet, etc.). Define EXO's own allowlist; don't blindly copy oMLX's.
- **API compatibility:** `<model>:<profile>` must not break existing clients
  that pass plain model ids. Resolution falls through to the base model.
- **Ollama adapter:** `api/adapters/ollama.py` has its own model-id semantics;
  ensure profiles compose.

---

## Definition of done

- [ ] Phase 1: `ModelSettings` + `ModelProfile` shipped; `<model>:<profile>`
      served from one loaded model with zero extra memory; dashboard CRUD works.
- [ ] Phase 2: per-node resident-memory reported (informational, no behavior change).
- [ ] Phase 3 (if pursued): EnginePool with LRU/pin/TTL; lease blocks unload;
      OOM refusal; master prefers resident models.
- [ ] `basedpyright` + `ruff` + `nix fmt` + `pytest` clean.