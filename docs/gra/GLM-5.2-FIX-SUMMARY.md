# GLM-5.2 on exo — Full Fix Summary

Model: `avlp12/GLM-5.2-Alis-MLX-Dynamic-2.3bpw` (architecture `glm_moe_dsa`)
Cluster: 2 x MSU nodes (rank 1 and rank 0), tensor-parallel.
Final artifact: `EXO-1.0.72-GLM-5.2-dev9.dmg`
SHA256: `3aee21938c047953cb6e2c4b87de09b0bbe612dd7c17595c361616df0940a003`

This document describes every fix applied, in the order they were
discovered and resolved, with root cause, the change, and verification.

---

## Fix 1 — Vendored GLM-5.2 model registration (infrastructure)

### Problem
mlx-lm has no built-in `glm_moe_dsa` model. GLM-5.2 ships optimized
MLX code in its HF repo, but exo pins mlx-lm and must not mutate it.

### Change
Added a vendored model package + a monkey-patch that registers it as
`mlx_lm.models.glm_moe_dsa` at runtime (before `mlx_lm.load()` imports
it), without touching the pinned package.

- `src/exo/worker/engines/mlx/vendor/glm_moe_dsa/`
  - `glm_moe_dsa_model.py` — top-level `Model` (extends vendored
    `deepseek_v32.Model`), `GlmMoeDsaModel`, `GlmMoeDsaAttention`,
    `GlmMoeDsaDecoderLayer`. Owns the DSA indexer sharing.
  - `deepseek_v32.py` — vendored DeepSeek-V3.2 backbone
    (`DeepseekV32Model`, `DeepseekV32MLP`, `SwitchGLU`, MLA attention).
    Also contains `sanitize()` (Fix 2) and `_use_glm_moe_fused_gate_up`.
  - `sparse_mla.py` — sparse MLA attention with top-k mask + fallback.
  - `switch_layers.py` — `SwitchGLU` (fused `gate_up_proj` when enabled).
  - `kernels.py` — `fast` facade; all `glm_fast.has(...)` checks return
    False because exo ships no native GLM kernels, so the standard
    attention path with sparse top-k mask is used.
- `src/exo/worker/engines/mlx/patches/glm_moe_dsa/__init__.py` —
  `apply_glm_moe_dsa_patch()` registers the vendored module under
  `mlx_lm.models.glm_moe_dsa` (idempotent).
- `src/exo/worker/engines/mlx/patches/__init__.py` —
  `apply_mlx_patches()` now calls `apply_glm_moe_dsa_patch()`. This is
  invoked from `bootstrap.py` at worker startup.
- `packaging/pyinstaller/exo.spec` — added `collect_submodules("exo")`
  to `HIDDEN_IMPORTS` so the vendored package + patches are bundled.

The public module name stays `glm_moe_dsa` so mlx-lm's normal model
loader finds it; the helper modules stay private under
`exo.worker.engines.mlx.vendor.glm_moe_dsa` and do not replace
`mlx_lm.models.deepseek_v32` or the shared MoE layers used by other
model families.

---

## Fix 2 — Quantization-config remap for fused `gate_up_proj` (warmup crash)

### Problem (dev3 failure)
`sanitize()` fuses `gate_proj` + `up_proj` into `gate_up_proj`
(enabled by `_use_glm_moe_fused_gate_up` for `glm_moe_dsa`), but the
model's `quantization` config only has entries for the *separate*
projections. After fusion, `nn.quantize`'s `class_predicate` could not
find `gate_proj`/`up_proj` and fell back to top-level defaults
(`group_size=64, bits=4`) instead of the per-layer
`group_size=128, bits=2`. This caused a shape mismatch at warmup:

```
[gather_qmm] Last dimension of first input with shape (..., 6144) does
not match the expanded quantized matrix (3072, 2048) computed from
shape (256,2048,384) with group_size=64, bits=4 and transpose=true
```

Verified on the MSU before the fix: `gate_up_proj` had
`gs=64, bits=4, input_dims=3072` (wrong) instead of `gs=128, bits=2`.

### Change
`src/exo/worker/engines/mlx/vendor/glm_moe_dsa/deepseek_v32.py` →
`sanitize()`: after fusing the weights, remap the per-layer
quantization specs from the separate projections onto the fused module:

```python
quant_cfg = getattr(self.args, "quantization", None)
if isinstance(quant_cfg, dict):
    gate_spec = quant_cfg.pop(f"{switch_prefix}.gate_proj", None)
    up_spec   = quant_cfg.pop(f"{switch_prefix}.up_proj", None)
    fused_spec = gate_spec or up_spec      # they share the same spec
    if fused_spec is not None:
        quant_cfg[f"{switch_prefix}.gate_up_proj"] = fused_spec
```

`self.args.quantization` is the same dict object as the `config
["quantization"]` used by `class_predicate`, and `sanitize()` runs
before `_quantize`, so the mutation is visible to `nn.quantize`.

(Note: `_use_glm_shared_fused_gate_up` returns False for GLM, so the
shared experts keep separate `gate_proj`/`up_proj` and need no remap;
the shared-expert fusion branch in `sanitize` is left in place for
completeness but is not exercised by this model.)

### Verification
dev4 warmed up successfully (50 tokens) — crash gone.

---

## Fix 3 — Segmented tensor-parallel sharding for fused `gate_up_proj` (garbage output)

### Problem (dev4 failure)
Warmup succeeded but the model produced a stream of numbers instead of
coherent text. Single-node `mlx_lm` generation (no sharding) produced
correct thinking + answers, confirming a tensor-parallel sharding bug.

`DeepSeekShardingStrategy.shard_model` sharded the fused
`gate_up_proj` with the default `all_to_sharded_linear_in_place`
(`segments=1`), which splits along the output axis so rank 0 gets the
gate half and rank 1 gets the up half. But `SwitchGLU.__call__` does
`mx.split(x_gate_up, 2, axis=-1)` expecting *both* halves present on
*every* rank — so each rank only had half the activation and the GLU
computation was broken.

### Change
`src/exo/worker/engines/mlx/auto_parallel.py`:
1. `GlmMoeDsaModel` added to the `isinstance` check that selects
   `DeepSeekShardingStrategy`.
2. The vendored `DeepseekV32MLP` added to the dense-MLP sharding check.
3. For the MoE branch, the sharding now detects the fused
   `gate_up_proj` and shards it with a custom predicate using
   `segments=[moe_intermediate, moe_intermediate]` via `shard_inplace`,
   so each rank owns a slice of *both* the gate and the up outputs —
   matching the downstream `mx.split(x_gate_up, 2, axis=-1)`:

```python
if hasattr(layer.mlp.switch_mlp, "gate_up_proj"):
    moe_intermediate = layer.mlp.config.moe_intermediate_size
    fused_segments = [moe_intermediate, moe_intermediate]
    def _fused_all_to_shard(path, weight, segs=fused_segments):
        if path.endswith("bias"):
            return weight.ndim - 1, segs
        return max(weight.ndim - 2, 0), segs
    shard_inplace(layer.mlp.switch_mlp.gate_up_proj,
                  sharding=_fused_all_to_shard, group=self.group)
else:
    self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.gate_proj)
    self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.up_proj)
```

The shared-experts branch is likewise made fusion-aware (handles
`gate_up_proj` if present, else falls back to separate projections).
Attention `embed_q`/`unembed_out` head sharding and the (non-sharded)
indexer are untouched.

### Verification
dev5 produced coherent reasoning text on the 2-node cluster.

---

## Fix 4 — EOS token IDs for the GLM-5 family

### Problem
The model's tokenizer config did not expose usable EOS token IDs, so
generation never terminated cleanly.

### Change
`src/exo/worker/engines/mlx/utils_mlx.py` →
`get_eos_token_ids_for_model`:
- `glm-5` (GLM-5.2 / 5.1): `[154820, 154827, 154829]`
  (`<|endoftext|>`, `<|user|>`, `<|observation|>`)
- `glm` (GLM-4.7 and older): `[151336, 151329, 151338]`

### Verification
`test_eos_token_ids.py` covers GLM-5.2 (Alis/mxfp4/DQ4plus), GLM-5.1,
and GLM-4.7.

---

## Fix 5 — Tool-calling (the final issue)

### Problem
The model emitted bash commands as raw markdown text in the `content`
field instead of structured `tool_calls`, so the tool parser never
intercepted them. Two distinct root causes:

#### 5A — Wrong tool-call delimiter tokens

The `glm52` parser in
`src/exo/worker/engines/mlx/tool_parsers/glm52.py` used backtick-XCTS
as `tool_call_start`/`tool_call_end`, but the GLM-5.2 model actually
emits special tokens that `tokenizer.decode` renders as plain ASCII
angle-bracket strings WITHOUT the pipe character:

| token id | decoded text | bytes |
|---|---|---|
| 154843 | OPEN-tool_call-CLOSE | `3c 74 6f 6f 6c 5f 63 61 6c 6c 3e` |
| 154844 | OPEN-/tool_call-CLOSE | `3c 2f 74 6f 6f 6c 5f 63 61 6c 6c 3e` |
| 154847 | OPEN-arg_key-CLOSE | `3c 61 72 67 5f 6b 65 79 3e` |
| 154848 | OPEN-/arg_key-CLOSE | `3c 2f 61 72 67 5f 6b 65 79 3e` |
| 154849 | OPEN-arg_value-CLOSE | `3c 61 72 67 5f 76 61 6c 75 65 3e` |
| 154850 | OPEN-/arg_value-CLOSE | `3c 2f 61 72 67 5f 76 61 6c 75 65 3e` |

(The tokenizer `added_tokens` metadata records these WITH pipes, e.g.
`<|tool_call|>`, but `tokenizer.decode` strips the pipe — so the parser
must match the no-pipe decoded form. The chat template also uses the
no-pipe form.)

`parse_tool_calls` checks
`response.text.startswith(tool_parser.start_parsing)`, so with the
wrong backtick-XCTS string it never matched and the whole
OPEN-tool_call ... CLOSE block leaked as a `TokenChunk` into `content`.

The argument layout is also different:
`func_name`, then repeated
`OPEN-arg_key-K-CLOSE-OPEN-arg_value-V-CLOSE` pairs (with closing
arg_key and arg_value tags), not the flat key/value split the old
parser assumed.

#### 5B — Generation never stops after a tool call

After emitting the tool-call-end token, GLM-5.2 does NOT emit EOS. It
hallucinates a tool-response block and keeps generating (verified: a
400-token run ended with `finish_reason: "length"`). So even with the
correct tokens the parsed tool call would be emitted, but then the
runaway tokens leaked as text.

### Changes

**5A — corrected parser**
`src/exo/worker/engines/mlx/tool_parsers/glm52.py` rewritten:
- `tool_call_start` / `tool_call_end` use the no-pipe decoded forms
  (built from `chr()` to keep the source unambiguous).
- `parse_tool_call` finds the function name before the first
  `arg_key` tag, then extracts each
  `arg_key K close-arg_key arg_value V close-arg_value` pair with a
  compiled regex (`re.escape` on the tag constants).
- Returns `arguments={}` (not `{"raw": ...}`) when no args are present,
  matching the original contract.
- The GLM-5.2 tokenizer branch in `utils_mlx.load_tokenizer_for_model_id`
  selects this parser for any `glm-5.2` / `glm-5.1` model id.

**5B1 — stop sequence in the generator**
`src/exo/worker/engines/mlx/generator/generate.py` -> `mlx_generate`
now adds the tokenizer's `tool_call_end` as a stop sequence whenever
`task.tools` is provided:

```python
if task.tools:
    tool_call_end = getattr(tokenizer, "tool_call_end", None)
    if isinstance(tool_call_end, str) and tool_call_end not in stop_sequences:
        stop_sequences.append(tool_call_end)
```

This makes generation stop as soon as the model emits the tool-call end
token.

**5B2 — recover the tool call when the end marker is trimmed**
The stop-sequence logic trims the matched stop string from the yielded
text, so the tool-call-end token arrives at `parse_tool_calls` with an
empty (or partial) `text`. The parser's `endswith(end_parsing)` check
therefore fails, and the old code would dump the accumulated tool-call
text as a raw `TokenChunk`.

In `src/exo/worker/runner/llm_inference/model_output_parsers.py` ->
`parse_tool_calls`, when generation stops (`finish_reason` set) while
`in_tool_call` is True, re-append the end marker and parse the complete
tool call before falling back to raw text:

```python
if response.finish_reason is not None:
    combined = "".join(tool_call_text_parts)
    if not combined.endswith(tool_parser.end_parsing):
        combined = combined + tool_parser.end_parsing
    parsed = tool_parser.parse(combined.strip(), tools=tools)
    if parsed:
        accumulated_tool_calls.extend(parsed)
        yield ToolCallResponse(
            tool_calls=accumulated_tool_calls,
            usage=response.usage, stats=response.stats,
        )
        accumulated_tool_calls.clear()
        continue
    # ...fall back to yielding raw text...
```

### Verification
End-to-end on a single MSU node (dev9):

- `content: ""`, clean `reasoning_content`, a single parsed
  `tool_calls` entry (`bash` / `{"command": "pwd"}`),
  `finish_reason: "tool_calls"`.
- A multi-tool request returned TWO parallel tool calls
  (`ls -la ~` and `date`) with correct arguments and
  `finish_reason: "tool_calls"`, no text leakage.

---

## Tests added

| file | covers |
|---|---|
| `test_mlx/test_glm52_tool_parser.py` | glm52 delimiter constants + single/multi/empty-arg parsing |
| `test_mlx/test_eos_token_ids.py` | `get_eos_token_ids_for_model` for GLM-5.2 / 5.1 / 4.7 families |
| `test_mlx/test_glm_moe_dsa_indexer.py` | vendored GLM DSA indexer behaviour |

All pre-commit checks pass:
`uv run basedpyright` (0 errors), `uv run ruff check`, and the MLX
unit-test suite (43 passed).

---

## Build artifacts

| version | fix milestone | SHA256 |
|---|---|---|
| dev3 | (failed warmup) | — |
| dev4 | quant-config remap (Fix 2) — warmup OK, garbage output | — |
| dev5 | segmented TP sharding (Fix 3) — coherent text | `8436d48d9e60dde8ded01202d36ae350d82176fe8df0bfc3ac4d0834eccc1586` |
| dev6 | glm52 parser tokens (wrong, with pipe) | `216b519bdc7b6703df3e086d48be043e2f1f6181bc51cd6884980d63a0b5263a` |
| dev7 | + stop-sequence + parser recovery (still wrong tokens) | — |
| dev8 | glm52 tokens corrected to no-pipe form | — |
| **dev9** | **final: all fixes** | `3aee21938c047953cb6e2c4b87de09b0bbe612dd7c17595c361616df0940a003` |

Build command: `EXO_VERSION=1.0.72-GLM-5.2-dev9 ./BUILD.sh`
-> `output/EXO-1.0.72-GLM-5.2-dev9.dmg`.

---

## Known limitation — 2-node discovery

IPv6 multicast peer discovery between the two MSU nodes is
currently not propagating (each node only discovers itself; both elect
themselves master). This is an **environmental networking issue**
(multicast routing on the MSU subnet), unrelated to the code fixes.
The TP-sharding fix (Fix 3) is in place and was verified previously;
the tool-calling fixes (Fix 5) were verified end-to-end single-node.

To run the model on a single node:
```
curl -X POST http://localhost:52415/place_instance \
  -H "Content-Type: application/json" \
  -d '{"model_id":"avlp12/GLM-5.2-Alis-MLX-Dynamic-2.3bpw","sharding":"Tensor","instance_meta":"MlxRing","min_nodes":1}'
```
then issue `/v1/chat/completions` requests with `tools`.

---

## File change index

Modified (tracked):
- `packaging/pyinstaller/exo.spec` — collect `exo` submodules for the bundle.
- `src/exo/worker/engines/mlx/auto_parallel.py` — GLM DSA TP sharding (Fix 3).
- `src/exo/worker/engines/mlx/generator/generate.py` — tool-call-end stop sequence (Fix 5B1).
- `src/exo/worker/engines/mlx/patches/__init__.py` — register GLM DSA patch (Fix 1).
- `src/exo/worker/engines/mlx/utils_mlx.py` — GLM-5 EOS IDs (Fix 4) + glm52 tokenizer branch.
- `src/exo/worker/runner/llm_inference/model_output_parsers.py` — recover tool call on stop (Fix 5B2).

New (vendored model + tests + cards):
- `src/exo/worker/engines/mlx/patches/glm_moe_dsa/__init__.py` — patch registration (Fix 1).
- `src/exo/worker/engines/mlx/vendor/glm_moe_dsa/` — vendored model (Fix 1; Fix 2 lives in `deepseek_v32.py`).
- `src/exo/worker/engines/mlx/tool_parsers/glm52.py` — corrected parser (Fix 5A).
- `src/exo/worker/tests/unittests/test_mlx/test_glm52_tool_parser.py`
- `src/exo/worker/tests/unittests/test_mlx/test_eos_token_ids.py`
- `src/exo/worker/tests/unittests/test_mlx/test_glm_moe_dsa_indexer.py`
- `resources/inference_model_cards/avlp12--GLM-5.2-Alis-MLX-Dynamic-2.3bpw.toml`
- `resources/inference_model_cards/mlx-community--GLM-5.2-DQ4plus-q8.toml`
- `resources/inference_model_cards/mlx-community--GLM-5.2-mxfp4.toml`

---

## Post-Dev9 Fixes — TP Collective Stall (Intermittent)

After dev9, the cluster intermittently hit `decode stalled: no tokens for
120.0s (hung distributed collective)`. Several fixes were applied in sequence;
the final root cause was a broken cache-state evaluation in
`flush_prefill_for_decode`.

### Fix A — Shallow KV copy on TP critical path (commit `c86f57df`)

Eliminated multi-GB `deepcopy` of KV caches on `add_kv_cache` /
`update_kv_cache` / `get_kv_cache`. MLX arrays are immutable;
`copy.copy(layer)` shares array data — O(layers) not O(KV bytes).

### Fix B — Flush prefill transients before decode (commit `bbb3190e`)

`flush_prefill_for_decode()` — `mx.eval([c.state ...])` + `gc.collect()` +
`mx.clear_cache()` + `mx_barrier` between prefill and decode. Reclaims Metal
buffer pool filled by long-prefill intermediate activations.

### Fix C — gc.collect before clear_cache + periodic reclaim (commit `4800d95e`)

Added `gc.collect()` before `mx.clear_cache()` in the flush (release Python
refs so Metal buffers are reclaimable) and periodic `gc.collect()` +
`mx.clear_cache()` every 32 decode tokens in `ExoBatchGenerator.step()`.

### Fix D — TP collective deadlock in KV eviction loop (commit `fa47a075`)

The `_evict_until_under` while loop short-circuited on LOCAL
`len(self.caches) > 0` BEFORE calling `_memory_pressure_exceeds` (which runs
`all_gather` COLLECTIVE). When ranks had different cache entry counts, one
rank exited without calling `all_gather` while the other blocked inside it.
Fixed by restructuring the loop so both ranks ALWAYS call the collective once
per iteration, then do a second collective to agree on whether any rank still
has caches to evict.

### Fix E — CacheList KV state evaluation (commit `b36895da`) — ROOT CAUSE

**Root cause of the remaining intermittent stall.** GLM-5.2 / DeepSeek-V3 MLA
models use `CacheList` (a tuple of two `KVCache` objects per layer). The
`flush_prefill_for_decode` function did:

```python
with contextlib.suppress(Exception):
    mx.eval([c.state for c in cache])
```

But `CacheList.state` delegates to each sub-cache's `.state`, and
`KVCache.state` raises `AttributeError` when `keys is None` (uninitialised
indexer cache on shared layers, or the second sub-cache before first use). A
single raising sub-cache aborted the *entire* list comprehension, so
`contextlib.suppress` silently skipped evaluation of **all** layers.

The full prefill lazy graph (KV writes, `trim(2)` ops) was left pending. The
first decode step then evaluated the entire prefill graph + decode forward as
one giant Metal command buffer. On one rank this completed in time; on the
other it intermittently stalled, hanging the JACCL TP collective
(`Fence::wait`) → decode stall watchdog.

**Fix:** Replaced the naive `c.state` comprehension with
`collect_evalable_cache_arrays()`, a recursive walker that:
- Recurses into `CacheList.caches` (and any wrapper exposing `.caches`)
- Tries `.state` on each *leaf* cache individually
- Skips caches that raise (uninitialised — `keys is None` — nothing to eval)
- Flattens nested state tuples/lists into a flat list of `mx.array`s

This guarantees every *populated* cache layer is force-evaluated regardless
of sibling caches' state, so the prefill graph is fully materialised before
`gc.collect()` + `mx.clear_cache()` + `mx_barrier`.

**Verification on cluster:** 25,520-token cold prefill (0.1% cache hit) + 100
decode tokens completed in 145s with **zero** stall errors. Prefix-cache hit
(99.9% cached) completed in 7.7s. All 5 test scenarios passed with no
`stall`/`hung`/`Fence` errors in the logs.

**Artifact:** `EXO-1.0.72-tp-cachelist-eval-fix-dev1.dmg`
