# Plan: Per-Instance Cache-Efficiency Stat Cards (Prefill / Cached / Efficiency)

## Goal
Below the existing per-instance token row (`↓ in · ↑ out · N req`) on each
instance card in the dashboard, show a compact three-cell stat block —
**Prefill · Cached · Efficiency** — mirroring oMLX's admin "Status" page stat
cards (`omlx/admin/templates/dashboard/_status.html` +
`omlx/server_metrics.py::_build_snapshot`). The block reports, per instance and
cumulative from creation until kill:

| Card | Meaning |
|------|---------|
| **Prefill** | Cumulative prompt tokens processed (`prompt_tokens`). |
| **Cached** | Share of those served from the KV prefix cache (`cached_tokens`). |
| **Efficiency** | `cached_tokens / prompt_tokens * 100`, in %. |

## Why
EXO already tracked `cached_tokens` **per request** (in the OpenAI/Responses
`Usage.prompt_tokens_details.cached_tokens`, wired through the MLX generator at
`src/exo/worker/engines/mlx/generator/{generate,batch_generate}.py`) but never
**aggregated** it, so there was no way to see cache hit-rate at a glance. oMLX
surfaces exactly this as its headline stat. The per-instance token accounting
landed in `v1.0.72-InstanceTokenUsage-dev2` is the natural place to fold the
cached share in: same event, same state field, same UI row.

## Current state (before this change)
- `InstanceTokensUpdated` event (`src/exo/shared/types/events.py`) carried only
  `prompt_tokens` + `completion_tokens` deltas.
- `InstanceTokenUsage` state model
  (`src/exo/shared/types/worker/token_usage.py`) stored only `prompt_tokens`,
  `completion_tokens`, `total_tokens`, `request_count`.
- `apply_instance_tokens_updated` (`src/exo/shared/apply.py`) folded those four.
- `Runner.send_chunk` (`src/exo/worker/runner/runner.py`) emitted the event
  from the final chunk's `Usage` but dropped `prompt_tokens_details.cached_tokens`.
- The dashboard instance card rendered `↓ in · ↑ out · N req` and nothing else.
- **Events are persisted to disk** via `DiskEventLog`
  (`src/exo/utils/disk_event_log.py`, msgpack records) and **replayed on
  restart**, so any new field on a persisted event **must** default to keep old
  logs replayable.

## Design decision
Reuse the existing event-sourced accumulator — add `cached_tokens` to the
existing `InstanceTokensUpdated` event and `InstanceTokenUsage` state, rather
than inventing a parallel metrics store. This keeps a single source of truth,
survives restarts via replay, and costs one extra int per event.

## Backend changes

### 1. Types
**`src/exo/shared/types/events.py`** — `InstanceTokensUpdated` gains:
```python
cached_tokens: int = 0  # default → old persisted events still replay
```
**`src/exo/shared/types/worker/token_usage.py`** — `InstanceTokenUsage` gains:
```python
cached_tokens: int = 0
```
Both default to `0` so the msgpack event log on disk from before this change
deserializes cleanly (verified by `test_old_events_without_cached_tokens_replay_as_zero`).

### 2. Apply
**`src/exo/shared/apply.py`** — `apply_instance_tokens_updated` folds
`cached_tokens` into the running total in both the first-request and
accumulate branches.

### 3. Emission
**`src/exo/worker/runner/runner.py`** — `send_chunk` now reads
`usage.prompt_tokens_details.cached_tokens` (already present on the final
`TokenChunk`/`ToolCallChunk` `Usage`) and passes it into the event. No new
plumbing: the value was already reaching the runner, just not propagated.

### 4. API exposure
No new endpoint. `State` is served to the dashboard via
`model_dump(by_alias=True)`; `FrozenModel` uses `to_camel`, so `cached_tokens`
serializes as `cachedTokens` automatically and lands in the existing
`instanceTokenUsage` map the dashboard already polls.

## Frontend changes

### 5. Store (`dashboard/src/lib/stores/app.svelte.ts`)
`InstanceTokenUsage` interface gains `cachedTokens: number`.

### 6. Instance card UI (`dashboard/src/routes/+page.svelte`)
Below the `↓ in · ↑ out · N req` row (both render sites: desktop sidebar and
welcome panel), add a 3-column grid gated on `tokenUsage && promptTokens > 0`
(so fresh instances don't show `0 / 0 / 0.0%`):
```svelte
{#if tokenUsage && tokenUsage.promptTokens > 0}
  {@const cacheEfficiency =
    (tokenUsage.cachedTokens ?? 0) / tokenUsage.promptTokens * 100}
  <div class="mt-1.5 grid grid-cols-3 gap-1.5" ...>
    <div>Prefill → {formatTokenCount(tokenUsage.promptTokens)}</div>
    <div>Cached  → {formatTokenCount(tokenUsage.cachedTokens ?? 0)}</div>
    <div>Efficiency → {cacheEfficiency.toFixed(1)}%</div>
  </div>
{/if}
```
Styling matches the existing card aesthetic (mono, `white/40` labels, `white/70`
values, `bg-white/5` cells, tiny `[9px]`/`[11px]` type). Reuses the existing
`formatTokenCount()` helper (`1.2k`, `3.4M`).

## Edge cases & decisions
- **Backward-compatible replay**: `cached_tokens = 0` default on both the event
  and state model. Old events replay with zero cached share; new events carry
  the real value. Covered by a dedicated test.
- **Multi-shard / tensor-parallel**: unchanged — only rank-0 emits
  `InstanceTokensUpdated` (guard already in `Runner.send_chunk` via
  `_final_chunk_usage` + engine filtering), so cached tokens aren't
  double-counted.
- **Prefill-server / disaggregated instances**: cached tokens are attributed to
  the instance whose engine yields the final chunk (the decode instance), same
  as prompt/completion tokens — consistent with the v1 decision in
  `ExoInstanceTokenCounterImpl.md`.
- **Cancelled/errored requests**: not counted (no `usage` on error chunks).
- **`prompt_tokens == 0` guard**: avoids divide-by-zero and hides the block on
  instances that haven't served a real request yet.
- **Efficiency ratio**: matches oMLX's `server_metrics.py` —
  `cached / prompt * 100`, not `cached / (prompt + completion)`.
- **Persistence**: event-sourced as before; no new retention concerns.

## Testing
- **`src/exo/shared/tests/test_apply/test_apply_instance_token_usage.py`**:
  - `_delta` helper takes `cached`.
  - `test_accumulate_cached_tokens` — accumulates and derives the ratio.
  - `test_old_events_without_cached_tokens_replay_as_zero` — deserializes a
    legacy event JSON (no `cachedTokens` field) and asserts it replays as 0.
  - serialization roundtrip equality updated.
- **`src/exo/worker/tests/unittests/test_runner/test_instance_token_emission.py`**:
  - `_usage` helper takes `cached`.
  - `test_final_token_chunk_emits_cached_tokens` — flows cached tokens from
    chunk usage into the event.
  - existing emission tests assert `cached_tokens`.

## Pre-commit gates (per AGENTS.md)
`uv run basedpyright && uv run ruff check && ruff format && uv run pytest` —
all green. (`nix fmt` is the repo's formatter wrapper; `ruff format` is the
underlying Python formatter it invokes — `nix` isn't available in every
environment, so `ruff format` is run directly as the equivalent.)

## Artifact
`output/EXO-1.0.72-cache-efficiency-dev1.dmg` — Release build, code-signed
"Sign to Run Locally", `CFBundleShortVersionString = 1.0.72-cache-efficiency-dev1`.
Verified by mounting the DMG and confirming `cached_tokens` is present in the
bundled `exo.shared.types.events` module inside `PYZ.pyz`.