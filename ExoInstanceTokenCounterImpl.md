# Plan: Per-Instance Token Accounting (In/Out) in the EXO UI

## Goal
On each **Instance card** in the dashboard (the panel showing `F40816FA · Jundot/GLM-5.2… · Tensor · MLX RDMA · READY`), show a **cumulative** tally of input/output tokens processed by that instance from creation until it is killed (deleted). The counter resets to 0 when a new instance is created and disappears when the instance is deleted.

## Current state (what already exists)
- **Per-request usage exists**: The MLX engine produces a `Usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`, + reasoning/audio details) on the final chunk of every generation (`src/exo/worker/runner/llm_inference/model_output_parsers.py`, `src/exo/shared/types/chunks.py`).
- **Per-request stats reach the dashboard**: `GenerationStats` (prompt_tps, generation_tps, prompt_tokens, generation_tokens) is emitted as an SSE comment and consumed for the *current* response only (`app.svelte.ts:1831`, `api/adapters/chat_completions.py`).
- **Nothing is accumulated per instance.** `State` (`shared/types/state.py`) has `instances`, `runners`, `tasks`, … but no token counters. `Instance`/`BaseInstance` (`shared/types/worker/instances.py`) has no usage field.
- **Instance lifecycle is event-sourced**: `InstanceCreated` adds to `state.instances`; `InstanceDeleted` removes it (`shared/apply.py:212`).
- **Dashboard pulls everything from `/state`** (polled in `app.svelte.ts:1310`), and instance cards render from `instanceData` (`routes/+page.svelte:5098` desktop sidebar, `:6217` welcome panel).
- **Attribution gap**: The API process creates `TextGeneration(task_params=…)` and only checks *that a* instance exists (`_validate_model_has_instance`). The **master/worker** binds the task to a specific `instance_id`. The **worker engine** (`SequentialGenerator`/`BatchGenerator`) already has an `event_sender` and sends `ChunkGenerated` events, and is constructed with a `bound_instance` (so it knows its `instance_id`).

## Design decision
**Event-sourced accumulator in `State`** (not an API-local dict), because:
- It's the only approach that stays correct in multi-node setups (dashboard may talk to node A while inference runs on node B; only the event log is globally broadcast).
- It survives API/master restarts (event log replays).
- It matches the existing architecture (pure `apply()`, immutable `State`).

A new lightweight event is emitted once per **completed** generation (not per token), so traffic is negligible.

## Backend changes

### 1. New types
**`src/exo/shared/types/worker/token_usage.py`** (new) — a small frozen model:
```python
class InstanceTokenUsage(FrozenModel):
    instance_id: InstanceId
    prompt_tokens: int        # cumulative input
    completion_tokens: int    # cumulative output
    total_tokens: int         # cumulative sum
    request_count: int        # number of completed generations
```

**`src/exo/shared/types/events.py`** — add event:
```python
class InstanceTokensUpdated(BaseEvent):
    instance_id: InstanceId
    prompt_tokens: int        # delta for this request
    completion_tokens: int    # delta for this request
    # (total derived; request_count increments by 1)
```
Add to the `Event` discriminated union.

### 2. State + apply
**`src/exo/shared/types/state.py`** — add field:
```python
instance_token_usage: Mapping[InstanceId, InstanceTokenUsage] = {}
```
**`src/exo/shared/apply.py`** — add handler `apply_instance_tokens_updated` that accumulates into `state.instance_token_usage` (create entry if absent, sum deltas, +1 request_count). Also: `apply_instance_deleted` should drop the entry for that instance (it already drops the `Instance`; add the token-usage key removal) — so the counter vanishes on kill as requested.

### 3. Emission point (worker)
The cleanest single emission point is where the **final** chunk (the one with `finish_reason` + `usage`) is produced. Two viable spots; pick one:

- **Option 3a (preferred): in `model_output_parsers.py`** — when the parser yields the final `TokenChunk`/`ToolCallChunk` with `usage is not None`, emit `InstanceTokensUpdated`. The parser currently doesn't have `instance_id` or `event_sender`; thread them in (the parser functions already receive context, add `instance_id` + an `event_sender`/callback).
- **Option 3b: in `batch_generator.py`** — the generators already hold `event_sender` and send `ChunkGenerated`. Add `instance_id` to the generator dataclass (from `bound_instance.instance.instance_id` at construction in `bootstrap.py`), and when sending the final `ChunkGenerated` whose chunk has `usage`, also send `InstanceTokensUpdated`.

Recommend **3b**: minimal new plumbing (generators already send events and already need only `instance_id` added), keeps parsers pure.

Details for 3b:
- `bootstrap.py`: pass `bound_instance.instance.instance_id` into `SequentialGenerator`/`BatchGenerator` construction (add `instance_id: InstanceId` field).
- In the spots that send the final `ChunkGenerated` with a chunk whose `finish_reason is not None` and `chunk.usage is not None`, emit:
  ```python
  if self.device_rank == 0 and chunk.usage is not None:
      self.event_sender.send(InstanceTokensUpdated(
          instance_id=self.instance_id,
          prompt_tokens=chunk.usage.prompt_tokens,
          completion_tokens=chunk.usage.completion_tokens,
      ))
  ```
  Guard with `device_rank == 0` (same as existing `ChunkGenerated` sends) so only the lead shard reports, avoiding multi-counting in tensor-parallel instances.
- Cover all adapters' final chunks: `TokenChunk` and `ToolCallChunk` (both carry `usage`). Image generation uses `ImageGenerationStats` separately — out of scope for v1 (can add later).

### 4. API exposure
No new endpoint needed: `/state` already returns the full `State`, so `instanceTokenUsage` ships automatically. (Optionally expose `/state/instanceTokenUsage` for cheap polling — already supported by the generic `get_state(path)`.)

## Frontend changes

### 5. Store (`dashboard/src/lib/stores/app.svelte.ts`)
- Extend `RawStateResponse` (around line 222) with:
  ```ts
  instanceTokenUsage?: Record<string, {
    instanceId: string;
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
    requestCount: number;
  }>;
  ```
- Add `instanceTokenUsage = $state<Record<string, InstanceTokenUsage>>({});` and populate it in `fetchState()` next to `instances`.

### 6. Instance card UI (`routes/+page.svelte`)
Two render locations (desktop sidebar ~`:5098` and welcome panel ~`:6217`) — extract a shared snippet or small component to avoid duplication. Below the node-names row (after `instanceInfo.nodeNames`), add a compact token stats row, shown only when usage exists and `> 0`:
```svelte
{#if instanceTokenUsage[id]}
  {@const u = instanceTokenUsage[id]}
  {#if u.totalTokens > 0}
    <div class="mt-1 flex items-center gap-2 text-[11px] font-mono text-white/55">
      <span title="Input tokens">↓ {formatTokenCount(u.promptTokens)}</span>
      <span class="text-white/20">·</span>
      <span title="Output tokens">↑ {formatTokenCount(u.completionTokens)}</span>
      <span class="text-white/20">·</span>
      <span title="Completed requests">{u.requestCount} req</span>
    </div>
  {/if}
{/if}
```
- Add a `formatTokenCount()` helper (e.g. `1.2k`, `3.4M`) for compactness.
- Styling matches existing card aesthetic (mono, `white/55`, tiny). Use `↓`/`↑` or `IN`/`OUT` labels consistent with the rest of the UI.

## Edge cases & decisions
- **Multi-shard / tensor-parallel**: only `device_rank == 0` emits → no double counting.
- **Prefill-server / disaggregated (DSML) instances**: token attribution should follow the **decode** instance (the one producing output). Verify during implementation that the emission point is on the decode path; if prefill instances also report, decide whether to attribute input tokens to the prefill instance. For v1, attribute both to the instance whose engine yields the final chunk.
- **Instance deleted**: counter removed from state (matches "till is killed"). If a lingering final snapshot is desired later, that's a separate enhancement (e.g., log to traces).
- **Cancelled/errored requests**: only emit on chunks that carry `usage` (successful completions). Errors/cancels have no `usage` → not counted. Decide if partial output should count (recommend: no, keep it simple).
- **Reasoning tokens**: already inside `completion_tokens_details.reasoning_tokens`; the cumulative `completion_tokens` already includes them, so no extra work.
- **Persistence across restart**: handled by event-sourcing (event log replay). If the event log is bounded/rotated, confirm older `InstanceTokensUpdated` events aren't pruned while the instance is still alive (check event log retention in master).
- **Performance**: one tiny event per completed generation; `apply()` does a dict copy + sum — negligible vs. per-token `ChunkGenerated` traffic.

## Testing
- **Unit (`shared/tests`)**: `apply_instance_tokens_updated` accumulates correctly; `apply_instance_deleted` clears the entry; idempotency of replay.
- **Worker test**: extend an existing runner test (e.g. `test_runner/test_event_ordering.py` or `test_finish_reason_sse.py`) to assert an `InstanceTokensUpdated` event is emitted with correct deltas and only on `device_rank == 0`.
- **API test**: `api/tests/test_chat_completions_stream.py` — assert `/state` reflects accumulated usage after a streamed completion.
- **Dashboard**: manual screenshot via Playwright (per AGENTS.md) showing the token row on a live instance; verify it increments across multiple messages and disappears on DELETE.

## Pre-commit gates (per AGENTS.md)
`uv run basedpyright && uv run ruff check && nix fmt && uv run pytest` — note `State` uses `strict=True, extra="forbid"`, so the new field must be added everywhere `State` is constructed/validated; `basedpyright` strict will catch missing thread-throughs of `instance_id`/`event_sender`.

## Suggested implementation order
1. Types (`InstanceTokenUsage`, `InstanceTokensUpdated`) + `Event` union + `State` field + `apply` handlers + tests.
2. Thread `instance_id` into generators (`bootstrap.py`, `batch_generator.py`, `runner.py`) and emit `InstanceTokensUpdated` on final chunk; worker test.
3. Frontend store + `RawStateResponse` + `fetchState`.
4. Card UI (both locations) + `formatTokenCount` + Playwright screenshot.
5. Run all gates; manual multi-message verification.