# Idle-unload + custom image models — Blueprint (2026-11-24)

Operational contract for idle-unload (v2) and custom image-model onboarding.
Every rule below was learned the hard way — verified live on `m3msu512` (.174).

## Semantics (agreed, do not regress)

1. **Unload != delete.** The instance record, shard assignments and idle policy
   survive; only the runner process (weights in RAM) is torn down.
   `InstanceDeleted` is reserved for genuine teardowns.
2. **Auto-reload on request.** A request to an unloaded instance clears the flag
   (`InstanceWeightsLoadRequested` -> `apply` removes it from
   `state.instance_weights_unloaded`) and workers respawn runners from disk cache.
   Reason recorded: `reason=idle_reload`.
3. **Idle clock starts at the first sweeper tick**, not the first busy task.
   Instances that never ran a task since master boot must still reach their
   timeout (this silently disabled unload for e.g. `exolabs/Qwen-Image`).
4. **`Shutdown` tasks are never "usage".** Excluded from busy-checks in BOTH the
   master sweeper and the API `_instance_busy` — a lingering Shutdown task must
   never block unload.
5. **Sweeper fires once.** After the flag is set the sweeper must skip the
   instance (`continue`) — no re-emitting `InstanceWeightsUnloaded` every 10s
   tick (observed live: 418 events, log spam, repeated worker disruption).
6. **State must be resolved explicitly on shutdown.** The runner process dies
   right after `shutdown()` and its final status event races with process death.
   The worker `Shutdown` arm must send `TaskStatusUpdated(Complete)` AND
   `RunnerStatusUpdated(RunnerShutdown)` itself, or the task lingers `Running`
   and the runner row stays `RunnerShuttingDown` forever in `/state`.

## What "unloaded" means — verify RAM, not the UI

The UI flag (`weightsUnloaded: true`) is a state claim, NOT proof of memory
release. Before declaring idle-unload "working":

```bash
# on the node: the runner process must be GONE, RSS must drop
ps aux | grep 'multiprocessing-fork' | grep -v grep   # one entry per loaded model
# expect: the unloaded model's runner pid absent, only live models resident
```

Acceptance: a 57 GB model that unloads returns ~57 GB within seconds (allow a
minute for macOS to reclaim inactive pages). If RAM is still held, find the
process holding it before touching code.

## Custom image-model onboarding (qwen21 engine)

- The qwen21 engine reads **flat filenames** from the repo root
  (`qwen-image-2.1-turbo-MLX-8bit.safetensors` for the transformer,
  `text_encoders/...`, `vae/...`). Cards must declare the transformer component
  with `component_path='./'`.
- **Download gotcha (upstream `is_image_model` filter):** image models skip
  root-level `*.safetensors` ("weights live in component subdirs") — this
  silently drops root-component files and produces a runner crash loop
  (`Failed to open file ...MLX-8bit.safetensors`, retried forever,
  `RunnerFailed` xN). Fixed: root-level safetensors are kept when their size
  matches a root component's `storage_size.in_bytes`.
- **Never trust a "FAILED then retry" state without checking the runner log:**
  `~/.exo/exo_log/exo.log` (worker) + `~/.exo/exo_log/runner_log/stderr.log`.
  The real error is a `RuntimeError` traceback there, not in `/state`.
- Verify before declaring a model "loadable":
  `ls -la ~/.exo/models/<org>--<repo>/` must contain the exact file names the
  engine expects — compare against `weights.py` constants, not the HF repo UI.

## Event-log forensics (quick reference)

```bash
# count event occurrences in the master event log (msgpack, grep raw bytes)
python3 -c "d=open('$HOME/.exo/event_log/master/events.bin','rb').read(); \
print({n: d.count(n) for n in [b'InstanceWeightsUnloaded', b'InstanceWeightsLoadRequested', b'RunnerShutdown']})"
```

- `RunnerShuttingDown` without a matching removal = the shutdown race (rule 6).
- Hundreds of the same event = sweeper re-fire (rule 5).

## Things agreed AWAY (superseded — do not resurrect)

- v1 "unload = delete instance + delete policy": replaced by v2 keep-instance
  semantics (rule 1). Any code path that deletes an instance as part of idle
  handling is a regression.
- Sweeper emits unload only once per unloaded state; do not move to a
  "delete runners" strategy.
- Default timeout is 1800 s (30 min) via `DEFAULT_IDLE_UNLOAD_SECONDS`
  (`src/exo/api/main.py`); `0` disables, `<5` invalid. Do not reintroduce
  default-off.