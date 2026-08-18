# 07 — TurboQuant KV-Cache Compression

**Tier:** ⭐⭐ (Tier 2)
**Effort:** Medium (PDD plan exists; composes with oMLX tiered-cache)
**Impact:** Medium-high (KV memory → longer contexts / bigger batches)
**Upstream evidence:**
- PR #2148 `docs: add TurboQuant MLX PDD plan` (ianleon) — design plan only
- #1988 `feat: EXO_KV_CACHE_BITS env var + step=16384 for QuantizedKVCache` (adurham)
- #1990 `fix: skip KV cache quantization in single-node BatchGenerator mode`
- PR #2242 `Expose existing KV-cache quantization behind an env-var opt-in`
- PR #2261 `Force a clean prefill after chained KV prefix-cache extensions`

**Cross-reference:** `docs/omlx-porting/07-...` (oMLX TurboQuant), `docs/omlx-porting/01-tiered-kv-cache-ssd.md` (tiered cache this composes with).

---

## What it is

**TurboQuant** is oMLX's KV-cache compression scheme (quantized KV cache with per-model tunable bits, skip-last-n-layers option, and a fast Apple-Silicon path). PR #2148 is a **design plan (PDD)** for bringing TurboQuant-style KV compression to EXO's MLX runner, outlining:

- phased work for benchmarking,
- cache adapter integration,
- Apple Silicon fast path,
- PDD cache handoff,
- Qwen3-Next hybrid-cache handling,
- default-off rollout constraints.

EXO already has `QuantizedKVCache` (via `EXO_KV_CACHE_BITS`, #1988) — but it's a blunt instrument. TurboQuant is the tuned, per-model, fast-path version.

---

## Why it matters

- **KV memory dominates** long-context serving. A 52k-token context (#2208) on a 78-layer model is a huge KV footprint; quantizing it 4× buys either longer contexts or more concurrent requests.
- **Composes with tiered cache:** the oMLX SSD tier (doc 01) and boundary snapshot offload (oMLX doc 08) both benefit — quantized blocks are smaller to serialize/spill.
- **Default-off rollout:** KV quantization can degrade quality on some models; the PDD plan's default-off + per-model tuning is the safe path.
- **Qwen3-Next hybrid cache:** called out specifically — hybrid `ArraysCache`+`KVCache` models need special handling (EXO already has `has_non_kv_caches` / `snapshot_ssm_states` for this).

---

## Upstream PR landscape

| PR/Issue | Role |
|----------|------|
| #2148 | The design plan (PDD). Read first. |
| #1988 | `EXO_KV_CACHE_BITS` env + step=16384 for `QuantizedKVCache` (the blunt baseline) |
| #1990 | Skip KV quant in single-node BatchGenerator mode (correctness fix) |
| #2242 | Expose KV quant behind env opt-in (gating) |
| #2261 | Force clean prefill after chained KV prefix-cache extensions (correctness — quantized cache + prefix chaining bug) |

#1990 and #2261 are correctness fixes for the existing `QuantizedKVCache` path — they must land before/with TurboQuant. #1988/#2242 are the gating/UX.

---

## EXO current state (local fork)

`src/exo/worker/engines/mlx/cache.py`:
- `make_kv_cache(model, max_kv_size, keep)` — constructs `KVCache` / `QuantizedKVCache` / `RotatingKVCache` based on `KV_CACHE_BITS` / `CACHE_GROUP_SIZE` (constants.py).
- Already handles `QuantizedKVCache` (group_size + bits).
- `KVPrefixCache` stores caches; `snapshot_ssm_states` handles non-KV layers.
- `KV_CACHE_BITS` / `KV_GROUP_SIZE` in `constants.py`; `_MEMORY_THRESHOLD` for eviction.

So the *plumbing* for quantized KV exists; what's missing is the **TurboQuant tuning + fast path + per-model config + correctness hardening** (#1990, #2261).

---

## Integration seam

- **Constants → per-model:** move `KV_CACHE_BITS` / `CACHE_GROUP_SIZE` from global `constants.py` to per-model settings (ties to oMLX doc 05 `ModelSettings`: `turboquant_kv_enabled`, `turboquant_kv_bits`, `turboquant_skip_last`).
- **Skip-last:** don't quantize the last N layers (quality-sensitive). Add `turboquant_skip_last`.
- **Fast path:** Apple-Silicon-tuned quantized KV kernel (overlaps oMLX native kernels, doc 02 of oMLX porting — but that's GLM-specific; TurboQuant is general KV).
- **PDD cache handoff:** the disaggregated/remote prefill path (`remote_prefill.py`) ships KV blocks over the wire — quantized blocks are smaller. Define the wire format for quantized KV.
- **Correctness:** port #1990 (skip quant in single-node BatchGenerator) and #2261 (clean prefill after chained prefix-cache extensions) — both are latent bugs in the current quant path.

---

## Phased plan

### Phase 1 — Correctness hardening (port #1990, #2261)
- Skip KV quantization in single-node BatchGenerator mode (#1990).
- Force clean prefill after chained KV prefix-cache extensions (#2261).
- **Tests:** single-node BatchGenerator with `KV_CACHE_BITS` set → correct output; chained prefix-cache extension → correct output (no stale quantized state).

### Phase 2 — Per-model TurboQuant settings
- Port `turboquant_kv_enabled` / `_bits` / `_skip_last` (from oMLX `model_settings.py`).
- `make_kv_cache` reads per-model config instead of global constant.
- **Tests:** per-model bits applied; skip-last preserves last N layers full-precision; output parity vs full-precision within tol.

### Phase 3 — Apple Silicon fast path
- Tuned quantized KV kernel (reference: oMLX `turboquant_attention.py`).
- Benchmark tok/s + memory vs baseline `QuantizedKVCache`.
- **Tests:** numerical equivalence vs baseline; benchmark ≥ baseline at lower memory.

### Phase 4 — PDD cache handoff + compose with tiered cache
- Wire format for quantized KV in `disaggregated/protocol.py`.
- Compose with oMLX tiered cache (docs/omlx-porting/01) — quantized blocks spill to SSD smaller.
- **Tests:** remote prefill ships quantized KV; round-trip parity; SSD spill of quantized blocks.

---

## Risks & open questions

- **Quality regression:** aggressive KV quantization degrades output on some models (especially small models, long contexts). The per-model tuning + default-off is mandatory; add a quality regression benchmark.
- **Skip-last tuning:** which layers to skip is model-specific. Start conservative (skip last 2–4); let users tune.
- **Compose with drafter (doc 03):** speculative decode trims/restores KV on reject; quantized KV trim/restore must be correct. Verify `trim_cache` works on `QuantizedKVCache` (it should, but test under the drafter).
- **Compose with ring (doc 02):** ring rotates KV blocks between ranks; quantized blocks must round-trip across ranks without dequant/requant overhead (or define the overhead).
- **PDD plan is docs-only:** #2148 is a plan, not code. Treat it as the spec; implementation is greenfield.

---

## Definition of done

- [ ] Phase 1: #1990 + #2261 ported; latent quant-path bugs fixed; tests green.
- [ ] Phase 2: per-model TurboQuant settings; skip-last; parity within tol.
- [ ] Phase 3: fast path; benchmark shows memory cut ≥2× with ≤2% quality regression.
- [ ] Phase 4: quantized KV over remote prefill; SSD spill of quantized blocks.
- [ ] `basedpyright` + `ruff` + `nix fmt` + `pytest` clean.