import copy
import gc
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Protocol, cast

import mlx.core as mx
import numpy as np
import psutil
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)
from mlx_lm.models.deepseek_v4 import (
    DeepseekV4Cache,
)
from mlx_lm.models.deepseek_v4 import (
    _CompressorBranch as CompressorBranch,  # type: ignore
)
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.models.model_cards import ModelId
from exo.shared.types.memory import Memory
from exo.worker.engines.mlx import memory_guard, turboquant
from exo.worker.engines.mlx.constants import CACHE_GROUP_SIZE, KV_CACHE_BITS
from exo.worker.engines.mlx.exceptions import PrefillMemoryExceededError
from exo.worker.engines.mlx.prefill_transient_tracker import PrefillTransientTracker
from exo.worker.engines.mlx.ssd_cache import SSDKVCacheStore
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from exo.worker.engines.mlx.vision import MediaRegion


# Fraction of device memory above which LRU eviction kicks in.
# Smaller machines need more aggressive eviction.
def _default_memory_threshold() -> float:
    total_gb = Memory.from_bytes(psutil.virtual_memory().total).in_gb
    if total_gb >= 128:
        return 0.85
    if total_gb >= 64:
        return 0.80
    if total_gb >= 32:
        return 0.75
    return 0.70


_MEMORY_THRESHOLD = float(
    os.environ.get("EXO_MEMORY_THRESHOLD", _default_memory_threshold())
)


def _default_prefill_memory_threshold() -> float:
    """Default prefill (soft) watermark: 10 points below the cache threshold.

    Persistent prefix-cache allocations are allowed to fill memory up to
    ``_MEMORY_THRESHOLD``, but prefill needs transient activation headroom on
    top of whatever the cache is holding. Reserving a lower watermark for the
    *start* of prefill (i.e. evicting the cache down to it first) is the fix
    for the prefix-cache-starves-prefill OOM from PR #2251. 0.10 mirrors the
    PR's default gap below ``EXO_MEMORY_THRESHOLD``.

    Note: this is the *legacy* fraction-of-RAM soft watermark used by
    ``_evict_if_needed`` and the before-prefill eviction. The reclaim-based
    admission ceiling (``memory_guard.hard_limit_bytes``) is separate and is
    the binding limit for preflight rejection.
    """
    return max(0.0, _MEMORY_THRESHOLD - 0.10)


# Legacy soft watermark (fraction of RAM) for ``_evict_if_needed`` and
# ``evict_for_prefill_headroom``. Kept for the existing eviction path; the
# preflight admission check uses the reclaim-based ceiling from
# ``memory_guard`` instead.
_PREFILL_MEMORY_THRESHOLD = float(
    os.environ.get("EXO_PREFILL_MEMORY_THRESHOLD", _default_prefill_memory_threshold())
)

# How hard the evict-before-prefill loop will try before giving up and letting
# the preflight estimator decide. Each iteration evicts one LRU entry and
# re-measures; this caps thrash under adversarial pressure. Mirrors oMLX's
# ``_MAX_PREFILL_EVICTION_RETRIES``.
_MAX_PREFILL_EVICTION_RETRIES = 8

# mlx-lm's prefill chunk size; the transient attention peak is set by the
# chunk, not the whole prompt (see oMLX ``prefill_guard._DEFAULT_PREFILL_STEP``).
# EXO hard-codes 4096 in the generators; the estimator uses the same default
# when the caller does not pass one.
_DEFAULT_PREFILL_STEP_SIZE = 4096

# Safety factor applied to the predicted per-chunk transient (oMLX
# ``_PREFILL_TRANSIENT_SAFETY``). The EWMA is a running average that lags
# real growth; multiplying by 1.2 keeps the prediction ahead of it.
_PREFILL_TRANSIENT_SAFETY = 1.2

# Bytes-per-element for the KV cache and the SDPA score matrix. EXO defaults to
# fp16/bf16 (2 bytes) for KV; ``KV_CACHE_BITS`` quantizes KV when set, and
# TurboQuant (``turboquant.effective_kv_bits()``) overrides it when enabled.
# The attention score matrix is the *compute* dtype (fp32 in the unfused
# fallback), so it is sized separately (see ``_estimate_sdpa_activation_bytes``).
_TQ_BITS = turboquant.effective_kv_bits()
_KV_BYTES_PER_ELEMENT = (
    (_TQ_BITS / 8.0)
    if _TQ_BITS is not None
    else (KV_CACHE_BITS / 8.0)
    if KV_CACHE_BITS is not None
    else 2.0
)
_SCORE_BYTES_PER_ELEMENT = 4.0  # unfused fp32 score matrix — safe upper bound


class CacheSnapshot:
    """Snapshot of states at a known token position."""

    def __init__(
        self,
        states: list[
            RotatingKVCache | ArraysCache | CacheList | DeepseekV4Cache | None
        ],
        token_count: int,
    ):
        self.states = states
        self.token_count = token_count


def _detached_copy(a: mx.array) -> mx.array:
    dtype = a.dtype
    if dtype == mx.bfloat16:
        return mx.array(np.array(a.astype(mx.float32))).astype(mx.bfloat16)
    return mx.array(np.array(a))


def copy_rotating_kv_cache(cache: RotatingKVCache) -> RotatingKVCache | None:
    """
    Deepcopy copies the metadata associated with an mx array.
    Specifically, it shares a shared_ptr to the underlying data and
    the mlx graph inputs of the array. This causes a memory leak for rotating
    kv cache. By creating an np array, no metadata is stored so the old cache
    can be cleaned up nicely.
    """
    if cache.keys is None or cache.values is None:
        return None
    n = min(cache.max_size, cache.keys.shape[2])
    k_slice = _detached_copy(cache.keys[..., -n:, :])
    v_slice = _detached_copy(cache.values[..., -n:, :])
    mx.eval(k_slice, v_slice)
    snap = RotatingKVCache.__new__(RotatingKVCache)
    snap.keys = k_slice
    snap.values = v_slice
    snap.offset = cache.offset
    snap._idx = n
    snap.keep = cache.keep
    snap.max_size = cache.max_size
    return snap


def _copy_arrays_cache(ac: ArraysCache) -> ArraysCache:
    entries: list[mx.array | None] = []
    for entry in ac.cache:  # type: ignore[reportUnknownMemberType]
        if entry is None:
            entries.append(None)
            continue
        assert isinstance(entry, mx.array)
        entries.append(_detached_copy(entry))
    copy = ArraysCache(len(entries))
    copy.cache = entries  # type: ignore[reportUnknownMemberType]
    return copy


def _copy_cache_list(cl: CacheList) -> CacheList:
    inners: list[object] = list(cl)  # type: ignore[reportUnknownArgumentType]
    copied: list[object] = []
    for inner in inners:
        if isinstance(inner, RotatingKVCache):
            snap = copy_rotating_kv_cache(inner)
            copied.append(snap if snap is not None else deepcopy(inner))
        elif isinstance(inner, ArraysCache):
            copied.append(_copy_arrays_cache(inner))
        else:
            copied.append(deepcopy(inner))
    return CacheList(*copied)


def _detached_copy_or_none(a: mx.array | None) -> mx.array | None:
    if a is None:
        return None
    out = _detached_copy(a)
    mx.eval(out)
    return out


def _copy_compressor_branch(b: CompressorBranch) -> CompressorBranch:
    out = CompressorBranch.__new__(CompressorBranch)
    out.buffer_kv = _detached_copy_or_none(b.buffer_kv)
    out.buffer_gate = _detached_copy_or_none(b.buffer_gate)
    out.prev_kv = _detached_copy_or_none(b.prev_kv)
    out.prev_gate = _detached_copy_or_none(b.prev_gate)
    out.pool = _detached_copy_or_none(b.pool)
    out.buffer_lengths = deepcopy(b.buffer_lengths)
    out.pool_lengths = deepcopy(b.pool_lengths)
    out.buffer_count = deepcopy(b.buffer_count)
    out._new_pool_lengths = deepcopy(b._new_pool_lengths)
    return out


def _copy_v4_cache(c: DeepseekV4Cache) -> DeepseekV4Cache:
    snap = DeepseekV4Cache.__new__(DeepseekV4Cache)

    local: RotatingKVCache = c.local
    local_snap = copy_rotating_kv_cache(local)
    if local_snap is None:
        local_snap = RotatingKVCache.__new__(RotatingKVCache)
        local_snap.keys = None
        local_snap.values = None
        local_snap.offset = local.offset
        local_snap._idx = 0
        local_snap.keep = local.keep
        local_snap.max_size = local.max_size
    snap.local = local_snap

    snap._branches = {
        key: _copy_compressor_branch(branch) for key, branch in c._branches.items()
    }
    snap._pending_lengths = deepcopy(c._pending_lengths)
    return snap


def copy_snapshot_entry(
    entry: ArraysCache | RotatingKVCache | CacheList | DeepseekV4Cache | None,
) -> ArraysCache | RotatingKVCache | CacheList | DeepseekV4Cache | None:
    match entry:
        case None:
            return None
        case RotatingKVCache():
            snap = copy_rotating_kv_cache(entry)
            return snap if snap is not None else deepcopy(entry)
        case ArraysCache():
            return _copy_arrays_cache(entry)
        case CacheList():
            return _copy_cache_list(entry)
        case DeepseekV4Cache():
            return _copy_v4_cache(entry)


def snapshot_ssm_states(cache: KVCacheType) -> CacheSnapshot:
    states: list[
        RotatingKVCache | ArraysCache | CacheList | DeepseekV4Cache | None
    ] = []
    for c in cache:
        if isinstance(c, ArraysCache):
            states.append(_copy_arrays_cache(c))
        elif isinstance(c, RotatingKVCache):
            states.append(copy_rotating_kv_cache(c))
        elif isinstance(c, CacheList) and not bool(c.is_trimmable()):  # type: ignore[reportUnknownMemberType]
            states.append(_copy_cache_list(c))
        elif isinstance(c, DeepseekV4Cache):
            states.append(_copy_v4_cache(c))
        else:
            states.append(None)
    token_count = cache_length(cache)
    return CacheSnapshot(states=states, token_count=token_count)


def _find_nearest_snapshot(
    snapshots: list[CacheSnapshot],
    target_token_count: int,
) -> CacheSnapshot | None:
    best: CacheSnapshot | None = None
    for snap in snapshots:
        if snap.token_count <= target_token_count and (
            best is None or snap.token_count > best.token_count
        ):
            best = snap
    return best


def is_non_trimmable_cache_entry(c: object) -> bool:
    """A cache entry is non-trimmable if `trim(n)` can't roll back its full
    state — meaning the prefill +2 rollback must snapshot+restore it instead.
    """
    if isinstance(c, (ArraysCache, RotatingKVCache)):
        return True
    if isinstance(c, CacheList):
        return not bool(c.is_trimmable())  # type: ignore[reportUnknownMemberType]
    return isinstance(c, DeepseekV4Cache)


def has_non_kv_caches(cache: KVCacheType) -> bool:
    """Check if a cache contains any ArraysCache (SSM) entries."""
    return any(is_non_trimmable_cache_entry(c) for c in cache)


def _copy_kv_cache(cache: KVCacheType) -> KVCacheType:
    """Copy a KV cache for prefix-cache storage, sharing MLX array data.

    For standard append-only KV caches (``KVCache``/``QuantizedKVCache``) a
    shallow per-layer copy is O(1) in KV size: MLX arrays are immutable and
    ``extend``/``trim`` rebind them (never mutate in place), so sharing array
    data between the stored entry and the live decode cache is safe — decode
    creates new arrays via ``mx.concatenate`` and the stored entry keeps the
    original prompt-KV arrays.  This avoids a multi-GB ``deepcopy`` on the
    post-prefill critical path (between the prefill collective and the decode
    ``mx_barrier``) that stalls the tensor-parallel decode collective under
    memory pressure — the root cause of the ``decode stalled: no tokens for
    120.0s`` watchdog firing on a 2-node TP cluster after a ~50k-token prefill.

    For caches with in-place-mutated containers (``ArraysCache``/SSM,
    ``RotatingKVCache``, non-trimmable ``CacheList``, ``DeepseekV4Cache``) —
    detected by :func:`has_non_kv_caches` — fall back to ``deepcopy``:
    correctness over speed (these are exotic/SSM caches, typically far smaller
    than the KV tiers, and their state can't be safely shared).
    """
    if not has_non_kv_caches(cache):
        return [copy.copy(layer) for layer in cache]
    return deepcopy(cache)


def _cache_is_quantized(cache: KVCacheType) -> bool:
    """Whether any layer of ``cache`` is a ``QuantizedKVCache``.

    Used by the #2261 chained-extension guard: only quantized caches are at
    risk of stale state after a chained extension, so the clean-prefill guard
    is gated on this. A plain ``KVCache`` / ``RotatingKVCache`` (or a hybrid
    cache whose KV branches are all full precision) is always safe to chain.
    """
    return any(isinstance(c, QuantizedKVCache) for c in cache)


class KVPrefixCache:
    def __init__(self, group: mx.distributed.Group | None):
        self.prompts: list[mx.array] = []  # mx array of tokens (ints)
        self.caches: list[KVCacheType] = []
        self._snapshots: list[list[CacheSnapshot] | None] = []
        self._media_regions: list[list["MediaRegion"]] = []
        self._last_used: list[int] = []  # monotonic counter of last access per entry
        # #2261: per-entry flag marking entries produced by extending a previous
        # partial hit (a "chained" extension). When KV quantization is active,
        # a chained quantized entry is not trusted as the base for a *further*
        # partial extension — the accumulated quantization error / state can
        # desync — so ``get_kv_cache`` forces a clean prefill instead of reusing
        # it. Plain (non-quantized) caches are always safe to chain.
        self._chained: list[bool] = []
        self.prefill_tps: list[float] = []
        self._access_counter: int = 0
        self._group = group
        # Cold (SSD) tier for evicted/restarted entries (oMLX doc 01, Phases
        # 2–3). Lazily created via :meth:`set_ssd_store` once the engine knows
        # the model id + resolved SSD settings; ``None`` ⇒ RAM-only behaviour
        # (today's default — the tier is opt-in behind
        # ``EXO_TIERED_KV_CACHE``). When present, evicted RAM entries spill to
        # SSD and an exact-match prefix lookup that misses RAM is restored
        # from SSD instead of being recomputed.
        self._ssd_store: SSDKVCacheStore | None = None
        # Model id for the SSD cache signature (guards against model swaps on
        # restore). Set via :meth:`set_model_id` by the engine builder.
        self.model_id: ModelId | None = None
        # Per-model EWMA of prefill chunk transient bytes/token (Phase 2).
        # Refines the preflight admission estimate after the first prefill
        # and feeds the per-chunk abort guard. One tracker per loaded model,
        # mirroring oMLX's per-scheduler ``PrefillTransientTracker``.
        self.prefill_transient_tracker: PrefillTransientTracker = (
            PrefillTransientTracker()
        )
        # Baseline usage captured before each prefill chunk, so the post-chunk
        # delta can be measured from the progress callback (which fires after
        # the chunk's model() + eval). Set by ``guard_prefill_chunk_or_raise``.
        self._prefill_chunk_baseline_bytes: int = 0

    def set_ssd_store(self, store: SSDKVCacheStore | None) -> None:
        """Wire the cold (SSD) tier (oMLX doc 01, Phases 2–3).

        Called by the engine builder once the model id + resolved SSD settings
        are known. Passing ``None`` disables the tier (RAM-only). The store is
        responsible for its own restart-recovery scan (run at its construction),
        so wiring it here also triggers recovery on the first model load. No-op
        semantics when the tiered-cache master switch is off — the store itself
        reports ``is_active() == False`` and every spill/restore is a no-op.
        """
        self._ssd_store = store
        if store is not None and store.is_active():
            logger.info(
                "SSD KV cold tier active: dir=%s, max_size=%d bytes, recovered=%d entries",
                store.ssd_dir,
                store.max_size_bytes,
                store.status().get("ssd_entries", 0),
            )

    def set_model_id(self, model_id: ModelId | None) -> None:
        """Record the loaded model id for the SSD cache signature.

        Must be called on model load (and reload) so the SSD tier's signature
        guard refuses to restore a block saved under a different model.
        """
        self.model_id = model_id

    def clear(self):
        """Clear all cached prompts and caches.

        Also clears the SSD cold tier (if wired) so a model swap / explicit
        reset wipes both tiers together — a stale SSD block from a previous
        model must not survive a ``clear()``.
        """
        self.prompts.clear()
        self.caches.clear()
        self._snapshots.clear()
        self._media_regions.clear()
        self._last_used.clear()
        self.prefill_tps.clear()
        self._chained.clear()
        self.prefill_transient_tracker.reset()
        self._prefill_chunk_baseline_bytes = 0
        if self._ssd_store is not None and self._ssd_store.is_active():
            self._ssd_store.clear()

    def add_kv_cache(
        self,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        ssm_snapshots: list[CacheSnapshot] | None = None,
        media_regions: list["MediaRegion"] | None = None,
        prefill_tps: float = 0.0,
    ):
        """Add a new cache entry. Evicts LRU entries if memory is high."""
        self._evict_if_needed()
        self.prompts.append(prompt_tokens)
        self.caches.append(_copy_kv_cache(cache))
        self._snapshots.append(ssm_snapshots)
        self._media_regions.append(media_regions or [])
        self.prefill_tps.append(prefill_tps)
        self._access_counter += 1
        self._last_used.append(self._access_counter)
        self._chained.append(False)  # fresh entry, not a chained extension
        logger.info(f"KV cache added: {len(prompt_tokens)} tokens")

    def update_kv_cache(
        self,
        index: int,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        snapshots: list[CacheSnapshot] | None,
        restore_pos: int,
        media_regions: list["MediaRegion"] | None = None,
        prefill_tps: float = 0.0,
    ):
        """Update an existing cache entry in-place."""
        old_snapshots = self._snapshots[index]
        merged: list[CacheSnapshot] = []
        if old_snapshots:
            merged = [s for s in old_snapshots if s.token_count <= restore_pos]
        if snapshots:
            merged.extend(snapshots)

        self.prompts[index] = prompt_tokens
        self.caches[index] = _copy_kv_cache(cache)
        self._snapshots[index] = merged or None
        self._media_regions[index] = media_regions or []
        self.prefill_tps[index] = prefill_tps
        self._access_counter += 1
        self._last_used[index] = self._access_counter
        # #2261: this entry was extended from a previous partial hit — mark it
        # chained so a future partial hit that would extend it *again* forces a
        # clean prefill when KV quantization is active (stale-quantized-state
        # guard). See ``_should_force_clean_prefill``.
        self._chained[index] = True
        logger.info(f"KV cache updated (index {index}): {len(prompt_tokens)} tokens")

    def _get_snapshot(
        self, entry_index: int, target_token_count: int
    ) -> tuple[int, CacheSnapshot | None]:
        if not has_non_kv_caches(self.caches[entry_index]):
            return target_token_count, None

        snapshots = self._snapshots[entry_index]
        if not snapshots:
            return 0, None

        snap = _find_nearest_snapshot(snapshots, target_token_count)
        if snap is not None:
            return snap.token_count, snap

        return 0, None

    def get_kv_cache(
        self,
        model: Model,
        prompt_tokens: mx.array,
        media_regions: list["MediaRegion"] | None = None,
    ) -> tuple[KVCacheType, mx.array, int | None, bool]:
        """Get KV cache for prompt, returning remaining tokens to prefill.

        Returns:
            Tuple of (cache, remaining_tokens, matched_index, is_exact) where:
            - cache: KV cache to use for generation
            - remaining_tokens: tokens that still need prefilling
            - matched_index: index of the matched entry (None if no match)
            - is_exact: True if the full prompt matched the cached entry

        For models with SSM layers (which are ArraysCache in mlx), the cache is trimmed to the
        nearest SSM snapshot position at or before the match point for correctness.
        Same for rotating KV Cache.

        Media region validation: if the token-level prefix match extends into
        a cached media region whose content_hash differs from the query's, the
        match is truncated to the start of that region.
        """
        max_length = len(prompt_tokens)
        query_regions = media_regions or []

        best_index: int | None = None
        best_length = 0
        is_exact = False

        # Find best cache match
        for i, cached_prompt in enumerate(self.prompts):
            length = get_prefix_length(prompt_tokens, cached_prompt)
            if length > 0:
                length = self._validate_media_match(
                    length,
                    self._media_regions[i],
                    query_regions,
                )
            if length >= max_length - 1:
                best_index, best_length = i, length
                is_exact = True
                break
            if length > best_length:
                best_index, best_length = i, length

        if best_index is None:
            # No RAM hit: check the SSD cold tier. Two restore modes
            # (oMLX doc 01):
            #   1. Exact-match restore — the re-run-same-prompt-after-restart
            #      path (Phase 2/3). No trim; full reuse.
            #   2. Prefix restore — the agentic re-send-context path: a long
            #      shared system prompt + a few new turns. Loads the SSD entry
            #      with the longest common prefix, trims it to the prefix, and
            #      only prefills the suffix. Refused for non-trimmable entries
            #      (no snapshot on the SSD tier) — those degrade to a fresh
            #      cache. Best-effort throughout: any miss/ineligibility falls
            #      through to a fresh cache (today's behaviour).
            if self._ssd_store is not None:
                restored, restored_tokens = self._ssd_store.restore(
                    prompt_tokens, model_id=self.model_id
                )
                if restored is not None:
                    logger.info(
                        f"SSD KV cache restored: {restored_tokens}/{max_length} "
                        f"tokens — skipping re-prefill"
                    )
                    self._access_counter += 1
                    self.prompts.append(prompt_tokens)
                    self.caches.append(restored)
                    self._snapshots.append(None)
                    self._media_regions.append(query_regions)
                    self.prefill_tps.append(0.0)
                    self._last_used.append(self._access_counter)
                    self._chained.append(False)  # restored, not a chained ext
                    remaining = prompt_tokens[restored_tokens:]
                    return (
                        restored,
                        remaining,
                        None,
                        restored_tokens >= max_length,
                    )
                # Prefix restore: longest common prefix across SSD entries.
                restored, prefix_len = self._ssd_store.restore_prefix(
                    prompt_tokens, model_id=self.model_id
                )
                if restored is not None and prefix_len > 0:
                    loaded_len = cache_length(restored)
                    if loaded_len > prefix_len:
                        trim_cache(restored, loaded_len - prefix_len, None)
                        for layer in restored:
                            if isinstance(
                                layer,
                                (ArraysCache, RotatingKVCache, DeepseekV4Cache),
                            ):
                                continue
                            if hasattr(layer, "offset"):
                                layer.offset = prefix_len
                    logger.info(
                        f"SSD KV prefix restored: {prefix_len}/{max_length} "
                        f"tokens — prefilling the suffix"
                    )
                    self._access_counter += 1
                    # Store the truncated prompt so a future request that
                    # extends this prefix hits RAM directly.
                    self.prompts.append(prompt_tokens[:prefix_len])
                    self.caches.append(restored)
                    self._snapshots.append(None)
                    self._media_regions.append(query_regions)
                    self.prefill_tps.append(0.0)
                    self._last_used.append(self._access_counter)
                    self._chained.append(False)
                    remaining = prompt_tokens[prefix_len:]
                    return restored, remaining, None, prefix_len >= max_length
            return make_kv_cache(model), prompt_tokens, None, False

        # #2261: force a clean prefill after chained prefix-cache extensions
        # when KV quantization is active. A partial (non-exact) hit that would
        # *extend* an entry which was itself produced by an extension (chained)
        # reuses quantized KV state that has accumulated across two extensions;
        # the quantization boundaries can desync and corrupt output. Refuse to
        # reuse it — return a fresh cache so the whole prompt is recomputed.
        # Exact hits reuse the entry as-is (no extension). Non-quantized caches
        # are always safe to chain, so the guard only applies when quantization
        # is on.
        self._ensure_chained_parallel()
        if (
            not is_exact
            and best_length < max_length
            and self._chained[best_index]
            and _cache_is_quantized(self.caches[best_index])
        ):
            logger.info(
                f"#2261: forcing clean prefill — partial hit on a chained "
                f"quantized entry ({best_length}/{max_length} tokens); "
                f"recomputing to avoid stale quantized KV state."
            )
            return make_kv_cache(model), prompt_tokens, None, False

        # For exact match: trim to max_length-1 so remaining has the last token
        # For partial match: trim to best_length, remaining has suffix to prefill
        # This ensures stream_generate always has at least one token to start with
        has_ssm = has_non_kv_caches(self.caches[best_index])
        cached_length = cache_length(self.caches[best_index])
        if has_ssm:
            target = best_length
        else:
            desired = (max_length - 1) if is_exact else best_length
            target = min(cached_length, desired)
        restore_pos, restore_snap = self._get_snapshot(best_index, target)

        # No usable snapshot — need fresh cache
        if restore_snap is None and has_ssm:
            return make_kv_cache(model), prompt_tokens, None, False

        prompt_cache = _copy_kv_cache(self.caches[best_index])
        tokens_to_trim = cached_length - restore_pos
        if tokens_to_trim > 0:
            trim_cache(prompt_cache, tokens_to_trim, restore_snap)
            # Reset cache offset to match trimmed length
            for c in prompt_cache:
                if isinstance(c, (ArraysCache, RotatingKVCache)):
                    continue
                if isinstance(c, DeepseekV4Cache):
                    continue
                if hasattr(c, "offset"):
                    c.offset = restore_pos

        self._access_counter += 1
        self._last_used[best_index] = self._access_counter
        remaining = prompt_tokens[restore_pos:]

        return prompt_cache, remaining, best_index, is_exact

    @staticmethod
    def _validate_media_match(
        match_length: int,
        cached_regions: list["MediaRegion"],
        query_regions: list["MediaRegion"],
    ) -> int:
        if not cached_regions:
            return match_length

        query_by_start: dict[int, "MediaRegion"] = {
            r.start_pos: r for r in query_regions
        }

        for cached_r in cached_regions:
            if cached_r.start_pos >= match_length:
                break
            query_r = query_by_start.get(cached_r.start_pos)
            if query_r is None:
                continue
            if query_r.content_hash != cached_r.content_hash:
                logger.info(
                    f"Media region mismatch at pos {cached_r.start_pos}: "
                    f"cached={cached_r.content_hash[:12]}... "
                    f"query={query_r.content_hash[:12]}... — "
                    f"truncating match from {match_length} to {cached_r.start_pos}"
                )
                match_length = cached_r.start_pos
                break

        return match_length

    def _evict_if_needed(self):
        """Evict least recently used entries while memory usage is high.

        Called after a completed cache entry is added. For the *before-prefill*
        eviction (PR #2251) use ``evict_for_prefill_headroom`` instead, which
        targets the tighter prefill watermark.
        """
        self._evict_until_under(_MEMORY_THRESHOLD)

    def evict_for_prefill_headroom(self) -> int:
        """Evict LRU entries so prefill has activation headroom.

        This is the #2251 fix: evict **before** ``get_kv_cache`` + prefill, not
        only on ``add_kv_cache``. Persistent prefix-cache allocations can
        otherwise leave too little room for the transient prefill activations
        and the prefill allocation OOMs. Targets the tighter
        ``_PREFILL_MEMORY_THRESHOLD`` (default ``_MEMORY_THRESHOLD - 0.10``),
        and reclaims MLX allocations after eviction so the freed pages are
        actually returned before the dangerous allocation runs.

        Returns the number of entries evicted.
        """
        return self._evict_until_under(_PREFILL_MEMORY_THRESHOLD)

    def preflight_or_raise(
        self,
        model: Model,
        *,
        num_prompt_tokens: int,
        cached_tokens: int,
        prefill_step_size: int = _DEFAULT_PREFILL_STEP_SIZE,
    ) -> None:
        """Refuse a prefill whose peak won't fit, after headroom eviction.

        The oMLX admission gate, run **after** ``evict_for_prefill_headroom``
        and the prefix-cache lookup. Estimate the prefill peak (new KV +
        last-chunk SDPA) and reject with ``PrefillMemoryExceededError`` if
        ``current + peak > hard_limit`` — a clean rejection instead of an OOM
        crash.

        The hard limit is MLX's ``max_recommended_working_set_size`` (the GPU
        allocation cap) scaled by ``_PREFILL_MEMORY_THRESHOLD`` — the same
        fraction the cache was evicted below, so admission and eviction agree
        on one ceiling. Distributed mode: ``get_memory_used_percentage`` is
        already the cluster max, so a remote node above threshold already
        triggered eviction (conservative; oMLX's ``check_collective`` votes
        per-rank).

        Best-effort: when model dims can't be read the estimator returns 0
        and the preflight is a no-op. This matches oMLX's rule that an
        unmeasurable request is never blocked, only an over-large one is.

        Call ``evict_for_prefill_headroom`` **before** the prefix-cache lookup
        so the persistent cache is trimmed before it is consumed (the #2251
        "evict before prefill" fix).
        """
        hard_limit_bytes = prefill_admission_limit_bytes()
        if hard_limit_bytes <= 0:
            return
        current_usage_bytes = _current_process_usage_bytes()
        # Seed the per-chunk baseline so the first progress callback can
        # measure a real delta (Phase 2). ``guard_prefill_chunk_or_raise``
        # re-sets this between chunks.
        self._prefill_chunk_baseline_bytes = current_usage_bytes
        # Refine the transient term with the measured EWMA when samples exist
        # (oMLX ``_admission_transient_bound``): max(static estimate,
        # ewma prediction + recent reclaim). The static estimate is what
        # ``raise_if_prefill_exceeds`` already charges; the EWMA captures
        # real per-chunk costs (MoE staging, buffer-pool fragmentation) the
        # static model misses.
        transient_override = self._admission_transient_bound(
            model,
            new_tokens=max(0, int(num_prompt_tokens) - max(0, int(cached_tokens))),
            kv_len=int(num_prompt_tokens),
            chunk_size=int(prefill_step_size),
            cached_tokens=int(cached_tokens),
        )
        raise_if_prefill_exceeds(
            model,
            num_prompt_tokens=int(num_prompt_tokens),
            cached_tokens=int(cached_tokens),
            prefill_step_size=int(prefill_step_size),
            hard_limit_bytes=hard_limit_bytes,
            current_usage_bytes=current_usage_bytes,
            transient_override_bytes=transient_override,
        )

    def guard_prefill_chunk_or_raise(
        self,
        model: Model,
        *,
        processed: int,
        total: int,
        prefill_step_size: int = _DEFAULT_PREFILL_STEP_SIZE,
        cached_tokens: int = 0,
    ) -> None:
        """Per-chunk prefill guard (Phase 2): measure, then abort the next chunk.

        Called from ``prefill``'s progress callback after each chunk completes
        (the callback fires after the chunk's ``model()`` + ``mx.eval``, so the
        chunk's own peak has already happened — this guard protects the
        *next* chunk). Mirrors oMLX's ``_guard_prefill_chunk`` pass/abort gate,
        adapted to EXO's stream_generate-driven prefill (which does not expose
        chunk shrinking, so we abort rather than shrink).

        1. Measure this chunk's footprint delta vs. the baseline captured
           before it, and feed the EWMA (``record_chunk_transient``).
        2. Predict the *next* chunk's transient
           (``_predicted_chunk_transient``) and abort with
           ``PrefillMemoryExceededError`` if ``current + predicted_next >
           abort_cap`` — a clean error instead of letting the next chunk's
           allocation trip the async Metal OOM.

        The abort cap is the Metal working-set ceiling scaled by
        ``_PREFILL_ABORT_MARGIN`` (above the admission ceiling, so admission
        stays the primary gate and this is the last-resort safety net for a
        climbing baseline mid-prefill). Best-effort: a no-op when model dims
        can't be read or no hard limit is set.
        """
        # processed == 0 is the pre-loop callback (mlx_lm fires it before the
        # first chunk): just refresh the baseline, no measurement/guard.
        if processed <= 0:
            self._prefill_chunk_baseline_bytes = _current_process_usage_bytes()
            return

        post_bytes = _current_process_usage_bytes()
        pre_bytes = self._prefill_chunk_baseline_bytes
        # The chunk that just completed: mlx_lm processes min(step, remaining).
        # Approximate the just-processed chunk size as min(step, processed)
        # (exact for full chunks; a slight over-estimate for the tail, which is
        # the safe direction and tails don't feed the running max anyway).
        chunk_tokens = max(
            1, min(int(prefill_step_size), max(0, int(processed) - int(cached_tokens)))
        )
        self._record_chunk_transient(chunk_tokens, pre_bytes, post_bytes)
        # Baseline for the next chunk is the post-chunk footprint.
        self._prefill_chunk_baseline_bytes = post_bytes

        abort_cap = memory_guard.prefill_abort_cap_bytes()
        if abort_cap <= 0:
            return
        # kv_len for the *next* chunk = total context after this chunk =
        # processed (cached + newly-prefilled so far).
        next_kv_len = int(processed)
        next_chunk = min(int(prefill_step_size), max(0, int(total) - int(processed)))
        if next_chunk <= 0:
            return  # prefill is done; no next chunk to guard
        predicted_next = self._predicted_chunk_transient(
            model, next_chunk, kv_len=next_kv_len
        )
        if predicted_next <= 0:
            return  # can't predict — skip (best-effort)
        if post_bytes + predicted_next <= abort_cap:
            return
        message = (
            f"Prefill chunk would exceed memory at {processed}/{total} tokens: "
            f"current {_format_bytes(post_bytes)} + predicted next-chunk "
            f"transient {_format_bytes(predicted_next)} > abort cap "
            f"{_format_bytes(abort_cap)}. Reduce context length or free memory."
        )
        logger.warning("Per-chunk prefill guard: %s", message)
        raise PrefillMemoryExceededError(
            message=message,
            estimated_bytes=int(post_bytes + predicted_next),
            limit_bytes=int(abort_cap),
        )

    def _record_chunk_transient(
        self, n_tokens: int, pre_bytes: int, post_bytes: int
    ) -> None:
        """Feed one chunk's measured footprint delta into the EWMA tracker.

        Ported from oMLX's ``Scheduler._record_chunk_transient``. Negative
        deltas (MLX pool reclaim larger than the chunk's allocation) are
        excluded from the per-token EWMA but their released footprint is
        retained (``record_reclaim``) until a positive sample confirms
        reallocation — the next predictor prices that one-shot realloc risk.
        """
        delta = post_bytes - pre_bytes
        tracker = self.prefill_transient_tracker
        if delta <= 0:
            tracker.record_reclaim(-delta)
            return
        tracker.clear_reclaim()
        # EXO has no throttle floor-chunk concept (prefill_step_size is fixed);
        # treat every full-step chunk as a floor sample so it feeds the
        # running max (the admission bound). Tail chunks (< step) still update
        # the EWMA but are marked non-floor so they don't inflate the max.
        floor_sample = n_tokens >= _DEFAULT_PREFILL_STEP_SIZE
        tracker.update(n_tokens, delta, floor_sample=floor_sample)

    def _predicted_chunk_transient(
        self, model: Model, n_tokens: int, *, kv_len: int
    ) -> int:
        """Conservative predicted footprint growth for one prefill chunk.

        Ported from oMLX's ``_predicted_chunk_transient``. Take the MAX of the
        measured signals (last-delta per-token, EWMA per-token) and the
        kv_len-aware static estimate (SDPA transient + this chunk's new KV),
        then apply a safety factor. Returns 0 when nothing is known (first
        chunk, no model info) — caller skips the guard.
        """
        if n_tokens <= 0:
            return 0
        tracker = self.prefill_transient_tracker
        per_token = 0.0
        if tracker.last_n_tokens > 0 and tracker.last_delta_bytes > 0:
            per_token = max(per_token, tracker.last_delta_bytes / tracker.last_n_tokens)
        if tracker.bytes_per_token > 0:
            per_token = max(per_token, tracker.bytes_per_token)
        static_per_token = 0.0
        static = estimate_prefill_peak_bytes(
            model,
            n_tokens,
            chunk_size=n_tokens,
            cached_tokens=max(0, kv_len - n_tokens),
        )
        if static > 0:
            static_per_token = float(static) / n_tokens
            per_token = max(per_token, static_per_token)
        if per_token <= 0:
            return 0
        base = per_token * n_tokens * _PREFILL_TRANSIENT_SAFETY
        # One-shot realloc risk: MLX may need to re-allocate the pool the
        # last negative delta returned. Price it once until a positive
        # sample confirms realloc (oMLX ``reallocation_prediction``).
        reallocation = (
            static_per_token * n_tokens * _PREFILL_TRANSIENT_SAFETY
            + tracker.recent_reclaim_bytes
        )
        return int(max(base, reallocation))

    def _admission_transient_bound(
        self,
        model: Model,
        *,
        new_tokens: int,
        kv_len: int,
        chunk_size: int,
        cached_tokens: int,
    ) -> int:
        """Transient charge for admission, refined by measurement.

        Ported from oMLX's ``_admission_transient_bound``. The largest
        floor-size chunk transient observed this session is a floor no
        prediction should undercut (a throttled prefill cannot get under it
        by shrinking). Returns 0 when no EWMA samples exist yet — the caller
        then defers to the static ``raise_if_prefill_exceeds`` charge, so the
        first prefill is admitted on the static estimate alone and later
        prefills are refined by measurement.
        """
        tracker = self.prefill_transient_tracker
        if tracker.samples == 0 and tracker.observed_max_bytes == 0:
            return 0  # no measurements — defer to the static charge
        bound = self._predicted_chunk_transient(
            model, min(int(chunk_size), max(1, int(new_tokens))), kv_len=kv_len
        )
        if tracker.observed_max_bytes > 0:
            bound = max(bound, tracker.observed_max_bytes)
        # Only override when measurement actually refined upward vs the static
        # estimate; otherwise defer to the caller's static charge.
        static = estimate_prefill_peak_bytes(
            model,
            max(1, int(new_tokens)),
            chunk_size=int(chunk_size),
            cached_tokens=int(cached_tokens),
        )
        if bound <= static:
            return 0
        return int(bound)

    def _ensure_chained_parallel(self) -> None:
        """Keep ``_chained`` length-parallel to ``caches``.

        Some test helpers and legacy paths append to ``caches``/``prompts``
        directly (bypassing ``add_kv_cache``); pad ``_chained`` with ``False``
        so the parallel-list invariant holds and the eviction/#2261 paths can't
        index out of bounds. Fresh entries are treated as non-chained.
        """
        while len(self._chained) < len(self.caches):
            self._chained.append(False)
        # Trim if something removed caches without going through the helpers.
        del self._chained[len(self.caches) :]

    def _evict_until_under(self, threshold: float) -> int:
        """Evict LRU entries until pressure is below ``threshold``.

        Shared by the post-add path (``_evict_if_needed``) and the
        before-prefill path (``evict_for_prefill_headroom``). Reclaims MLX's
        Metal buffer pool (``mx.clear_cache``) and runs ``gc.collect`` after
        eviction so the freed memory is visible to the next allocation — the
        oMLX ``_reclaim_pooled_buffers_for_prefill`` analogue.

        Capped at ``_MAX_PREFILL_EVICTION_RETRIES`` evictions per call so a
        stuck pressure reading (e.g. a third-party process holding memory)
        can't evict the entire cache in one go — the preflight estimator
        then decides whether to admit or reject.

        Pressure model: eviction compares the process's real footprint
        (``memory_guard.current_usage_bytes`` = max(phys_footprint, MLX
        active)) against the guard's byte ceilings — ``eviction_soft_bytes``
        for the before-prefill path, ``eviction_hard_bytes`` for the post-add
        path — scaled by ``threshold``'s relative position between the soft
        and hard watermarks. This is the fix for the prefix cache being
        evicted on *every* prefill on a large Apple-Silicon box whose model
        weights are wired via ``iogpu.wired_limit_mb``:
        ``psutil.virtual_memory().percent`` counts those wired pages as
        "used" and reports >75% with the model alone loaded, so the legacy
        percentage threshold evicted the whole cache before each lookup and
        ``prefix_hit_length`` was always 0. ``current_usage_bytes`` only
        counts what *this process* holds, so a legitimately-resident model
        no longer looks like pressure. The byte ceilings are resolved
        independently of the preflight-admission toggle (see
        ``memory_guard.eviction_*_bytes``) so eviction is correct whether or
        not admission rejection is on; the legacy ``get_memory_used_percentage``
        is only the fallback when the reclaim model can't be computed.

        TP deadlock guard: in tensor-parallel mode ``_memory_pressure_exceeds``
        runs an ``all_gather`` (a collective).  The original loop short-
        circuited on the LOCAL ``len(self.caches) > 0`` check *before* the
        collective — so if rank 0 had 0 entries and rank 1 had 1+, rank 0
        exited the loop (and the function) without calling all_gather while
        rank 1 blocked inside all_gather waiting for rank 0.  Rank 0 then
        moved to the next collective (``mx_barrier`` in
        ``flush_prefill_for_decode``) and blocked there.  Mutual deadlock →
        the ``decode stalled: no tokens for 120s`` watchdog.

        The fix: both ranks ALWAYS call ``_memory_pressure_exceeds`` (the
        collective) once per iteration, then do a SECOND collective
        (``all_sum``) to agree on whether ANY rank still has caches to
        evict.  Both ranks exit the loop at the same iteration — same number
        of all_gathers, same number of all_sums — so the collectives can
        never desync.  A rank with 0 caches participates in the collectives
        but skips the local eviction.
        """
        self._ensure_chained_parallel()
        evicted_count = 0
        # Evict LRU entries until below threshold (or the retry cap, whichever
        # comes first — see the docstring on the cap's purpose).
        while evicted_count < _MAX_PREFILL_EVICTION_RETRIES:
            # Collective pressure check — BOTH ranks must always participate,
            # even if this rank has 0 caches.  Without this, a rank with 0
            # caches exits the loop while the other blocks on all_gather →
            # TP collective deadlock (see docstring above).
            if not self._memory_pressure_exceeds(threshold):
                break
            # Both ranks agree pressure is high.  But only ranks with caches
            # can actually evict.  Collective decision: does ANY rank still
            # have a cache to evict?  If not, we're done (pressure persists
            # but nothing more to reclaim — the preflight estimator handles
            # admission/rejection).  This all_sum must run on BOTH ranks so
            # they exit the loop at the same iteration.
            local_can_evict = len(self.caches) > 0
            if self._group is not None:
                any_can_evict_arr = mx.distributed.all_sum(
                    mx.array(1.0 if local_can_evict else 0.0),
                    group=self._group,
                )
                mx.eval(any_can_evict_arr)
                any_can_evict = int(any_can_evict_arr.item()) > 0
            else:
                any_can_evict = local_can_evict
            if not any_can_evict:
                break
            # Evict the LRU entry locally (only if this rank has caches).
            if local_can_evict:
                lru_index = self._last_used.index(min(self._last_used))
                evicted_tokens = len(self.prompts[lru_index])
                # Spill the LRU entry to the SSD cold tier *before* dropping
                # it (oMLX doc 01, Phase 2). Best-effort + eligible-only.
                if self._ssd_store is not None:
                    self._ssd_store.spill(
                        self.prompts[lru_index],
                        self.caches[lru_index],
                        model_id=self.model_id,
                    )
                self.prompts.pop(lru_index)
                self.caches.pop(lru_index)
                self._snapshots.pop(lru_index)
                self._media_regions.pop(lru_index)
                self._last_used.pop(lru_index)
                self.prefill_tps.pop(lru_index)
                self._chained.pop(lru_index)
                logger.info(
                    f"KV cache evicted LRU entry ({evicted_tokens} tokens) "
                    f"due to memory usage (threshold={threshold:.2f})"
                )
            evicted_count += 1

        if evicted_count > 0:
            gc.collect()
            mx.clear_cache()

        return evicted_count

    def _memory_pressure_exceeds(self, threshold: float) -> bool:
        """Whether memory pressure warrants further eviction.

        ``threshold`` is the legacy fractional watermark (0-1) passed by
        ``_evict_if_needed`` (``_MEMORY_THRESHOLD``) and
        ``evict_for_prefill_headroom`` (``_PREFILL_MEMORY_THRESHOLD``). See
        ``_evict_until_under`` for why the reclaim-based byte model replaces
        the naive ``psutil`` percentage on loaded large-memory boxes.
        """
        soft = memory_guard.eviction_soft_bytes()
        hard = memory_guard.eviction_hard_bytes()
        # No byte ceilings resolvable → legacy percentage path so behaviour is
        # unchanged on platforms where the reclaim model can't be computed.
        if soft <= 0 or hard <= 0:
            return self.get_memory_used_percentage() > threshold

        # Map the fractional threshold onto a byte ceiling. The prefill path
        # targets the soft watermark (threshold < _MEMORY_THRESHOLD); the
        # post-add path targets the hard limit. Interpolate linearly between
        # soft and hard by where ``threshold`` sits between
        # ``_PREFILL_MEMORY_THRESHOLD`` and ``_MEMORY_THRESHOLD``.
        lo = _PREFILL_MEMORY_THRESHOLD
        hi = _MEMORY_THRESHOLD
        t = min(1.0, max(0.0, (threshold - lo) / (hi - lo))) if hi > lo else 0.0
        ceiling_bytes = int(soft + t * (hard - soft))
        if ceiling_bytes <= 0:
            return self.get_memory_used_percentage() > threshold

        local_usage = memory_guard.current_usage_bytes()
        if self._group is None:
            return local_usage > ceiling_bytes

        # Cluster-max usage vs cluster-max ceiling (conservative, mirroring the
        # legacy all-gather-of-percentages path): evict if *any* rank is over.
        all_usage = mx.distributed.all_gather(
            mx.array([local_usage], dtype=mx.uint64),
            group=self._group,
        )
        all_ceilings = mx.distributed.all_gather(
            mx.array([ceiling_bytes], dtype=mx.uint64),
            group=self._group,
        )
        max_usage = int(mx.max(all_usage).item())
        min_ceiling = int(mx.min(all_ceilings).item())
        return max_usage > min_ceiling

    def get_memory_used_percentage(self) -> float:
        local_pressure: float = get_memory_used_percentage()

        if self._group is None:
            return local_pressure

        all_pressure = mx.distributed.all_gather(
            mx.array([local_pressure], dtype=mx.float32),
            group=self._group,
        )
        # .item() evals.
        max_pressure = float(mx.max(all_pressure).item())
        return max_pressure


def trim_cache(
    cache: KVCacheType,
    num_tokens: int,
    snapshot: CacheSnapshot | None = None,
) -> None:
    for i, c in enumerate(cache):
        non_trimmable = isinstance(c, (ArraysCache, RotatingKVCache)) or (
            isinstance(c, CacheList) and not bool(c.is_trimmable())  # type: ignore[reportUnknownMemberType]
        )
        if non_trimmable:
            if snapshot is not None and snapshot.states[i] is not None:
                restored = copy_snapshot_entry(snapshot.states[i])
                if restored is not None:
                    cache[i] = restored  # type: ignore
            elif isinstance(c, (ArraysCache, RotatingKVCache)):
                c.state = [None] * len(c.state)
                if isinstance(c, RotatingKVCache):
                    c.offset = 0
                    c._idx = 0
            else:
                # CacheList without a snapshot — zero each inner cache's state
                for inner in c:  # type: ignore[reportUnknownVariableType]
                    if isinstance(inner, (ArraysCache, RotatingKVCache)):
                        inner.state = [None] * len(inner.state)
                        if isinstance(inner, RotatingKVCache):
                            inner.offset = 0
                            inner._idx = 0
        else:
            c.trim(num_tokens)


def encode_prompt(tokenizer: TokenizerWrapper, prompt: str) -> mx.array:
    """Encode a prompt string to token array.

    For chat-templated prompts (which have their own structure markers like
    <|im_user|>, <|im_middle|>, etc.), we should NOT add BOS/EOS tokens as
    that would corrupt the prompt structure.
    """
    # Chat templates define their own structure - don't add BOS/EOS
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    return mx.array(prompt_tokens)


def _entry_length(
    c: KVCache
    | RotatingKVCache
    | QuantizedKVCache
    | ArraysCache
    | CacheList
    | DeepseekV4Cache,
) -> int:
    # Use .offset attribute which KVCache types have (len() not implemented in older QuantizedKVCache).
    if hasattr(c, "offset"):
        return c.offset
    # For CacheList
    if hasattr(c, "size"):
        return int(c.size())  # type: ignore
    return 0


def cache_length(cache: KVCacheType) -> int:
    """Get the number of tokens in a KV cache."""
    return max((_entry_length(c) for c in cache), default=0)


def get_prefix_length(prompt: mx.array, cached_prompt: mx.array) -> int:
    """Find the length of the common prefix between two token arrays."""
    n = min(int(prompt.shape[0]), int(cached_prompt.shape[0]))
    if n == 0:
        return 0

    equal = mx.equal(prompt[:n], cached_prompt[:n]).astype(mx.int32)
    prefix_mask = mx.cumprod(equal)  # stays 1 until first mismatch, then 0 forever
    return int(mx.sum(prefix_mask).item())


def get_available_memory() -> Memory:
    mem: int = psutil.virtual_memory().available
    return Memory.from_bytes(mem)


def get_memory_used_percentage() -> float:
    mem = psutil.virtual_memory()
    # percent is 0-100
    return float(mem.percent / 100)


def get_max_working_set_bytes() -> int:
    """The effective Metal allocation cap on this device.

    Thin delegation to ``memory_guard.get_effective_metal_cap_bytes``:
    ``iogpu.wired_limit_mb`` when the operator raised it (which Metal's
    ``max_recommended_working_set_size`` does not reflect), else Apple's
    recommendation. Kept for backwards compatibility with callers that want
    the Metal cap specifically rather than the full reclaim-based ceiling.
    """
    return memory_guard.get_effective_metal_cap_bytes()


def prefill_admission_limit_bytes() -> int:
    """The hard OOM ceiling the preflight admission check rejects above.

    Delegates to ``memory_guard.hard_limit_bytes``: ``min(static, dynamic,
    metal_cap)`` where ``dynamic = phys_footprint + free + inactive +
    active×reclaim_ratio`` (oMLX's reclaim-based model). This is the fix for a
    model that legitimately fills 80%+ of memory no longer being rejected on
    every prefill: the ceiling grows with the process's own resident footprint
    instead of being a flat fraction of Metal's recommendation. Returns 0 when
    the guard is disabled — callers treat 0 as "no guard".
    """
    return memory_guard.hard_limit_bytes()


def _current_process_usage_bytes() -> int:
    """What this process is holding right now, by the reckoning jetsam uses.

    Delegates to ``memory_guard.current_usage_bytes``: the larger of
    ``phys_footprint`` (the kernel ledger jetsam compares against, which counts
    MLX's Metal allocations) and MLX's active Metal memory (the allocator's
    view, which leads the kernel ledger mid-prefill). Neither alone is the
    whole truth. Using ``phys_footprint`` instead of RSS is the fix for the
    loaded-model case: RSS under-reports IOAccelerator-backed (Metal) memory
    on Apple Silicon UMA.
    """
    return memory_guard.current_usage_bytes()


class _ModelConfigArgs(Protocol):
    """Structural subset of mlx-lm ``ModelArgs`` used for prefill estimation.

    A ``Protocol`` (structural typing) lets us read these fields off any
    mlx-lm config object without importing every model's ``ModelArgs`` class,
    while keeping the strict type-checker happy (no ``Any`` propagation).
    """

    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_size: int
    head_dim: int | None


def _read_model_dims(model: Model) -> tuple[int, int, int, int]:
    """Read ``(num_layers, num_kv_heads, head_dim, num_attention_heads)``.

    Best-effort: reads from the inner mlx-lm model's ``args`` config. Returns
    zeros when dims can't be read, which makes the preflight estimator a
    no-op rather than a source of spurious rejections — the same best-effort
    contract oMLX's ``set_model_info_from_model`` documents. ``head_dim`` is
    derived when the config leaves it None (mlx-lm computes
    ``hidden_size // num_attention_heads``).
    """
    try:
        from exo.worker.engines.mlx.vision import get_inner_model

        inner = get_inner_model(model)  # type: ignore[arg-type]
        args_obj = getattr(inner, "args", None)  # type: ignore[reportAny]
        if args_obj is None:
            return 0, 0, 0, 0
        config = cast(_ModelConfigArgs, args_obj)
        num_layers = int(config.num_hidden_layers or 0)
        num_kv_heads = int(
            config.num_key_value_heads or config.num_attention_heads or 0
        )
        num_attention_heads = int(config.num_attention_heads or 0)
        head_dim = config.head_dim
        if head_dim is None:
            hidden_size = int(config.hidden_size or 0)
            head_dim = (
                hidden_size // num_attention_heads if num_attention_heads > 0 else 0
            )
        return num_layers, num_kv_heads, int(head_dim), num_attention_heads
    except Exception:  # pragma: no cover - defensive
        logger.opt(exception=True).debug(
            "Could not read model dims for prefill-peak estimation"
        )
        return 0, 0, 0, 0


def estimate_kv_bytes_per_token(model: Model) -> float:
    """Resident KV-cache bytes added per prefilled token.

    ``layers * kv_heads * head_dim * dtype * 2`` (keys + values), with the
    KV-cache element width from ``KV_CACHE_BITS`` (fp16/bf16 = 2 bytes by
    default). Returns 0.0 when the dims can't be read — callers must treat 0
    as "can't estimate" and skip the preflight (oMLX's best-effort rule).

    Note: this is the uniform full-attention formula. MLA/latent-KV models
    (GLM-5.2, DeepSeek-V4) store a compressed latent rather than expanded
    K/V, so this over-counts them — the safe direction for a guard. A
    precise MLA estimate (oMLX ``estimate_mla_kv_bytes_per_token``) is a
    follow-up.
    """
    num_layers, num_kv_heads, head_dim, _ = _read_model_dims(model)
    if not (num_layers and num_kv_heads and head_dim):
        return 0.0
    return float(num_layers * num_kv_heads * head_dim) * _KV_BYTES_PER_ELEMENT * 2.0


def _estimate_sdpa_activation_bytes(
    query_tokens: int,
    kv_len: int,
    num_attention_heads: int,
    head_dim: int,
) -> int:
    """Transient SDPA activation bytes for one prefill chunk.

    The spike that drives prefill OOM. MLX only fuses full attention for a
    known set of head dims; unsupported dims fall back to an unfused fp32
    score matrix whose K dimension spans the full key/value context. We
    charge the unfused fp32 bound everywhere — the safe direction
    (over-estimate), matching oMLX's fallback for unsupported prefill head
    dims.

    Shape of the unfused score matrix is
    ``(batch=1, heads, query_tokens, kv_len)`` in fp32, plus the output
    ``(batch=1, heads, query_tokens, head_dim)``.
    """
    if query_tokens <= 0 or kv_len <= 0 or num_attention_heads <= 0 or head_dim <= 0:
        return 0
    scores = num_attention_heads * query_tokens * kv_len * _SCORE_BYTES_PER_ELEMENT
    output = num_attention_heads * query_tokens * head_dim * _KV_BYTES_PER_ELEMENT
    return int(scores + output)


def estimate_prefill_peak_bytes(
    model: Model,
    new_tokens: int,
    *,
    chunk_size: int = _DEFAULT_PREFILL_STEP_SIZE,
    cached_tokens: int = 0,
) -> int:
    """Estimate the per-request prefill peak (new KV + last-chunk SDPA).

    Ported from oMLX's ``MemoryMonitor.estimate_prefill_peak_bytes``. Returns
    only the part attributable to this request's prefill:
    - new KV cache bytes for the ``new_tokens`` being added (the resident
      part chunking cannot reduce), and
    - the SDPA attention activation peak for the last chunk
      (``eff_chunk = min(chunk_size, new_tokens)``), whose K dimension spans
      ``new_tokens + cached_tokens`` because cached positions still
      participate in attention.

    Does NOT include model weights (already in the active baseline) or
    prefix-cached KV that is already resident (counted in the caller's
    current-usage baseline). Returns 0 when model dims can't be read —
    callers treat 0 as "can't estimate" and skip the preflight.
    """
    if new_tokens <= 0:
        return 0
    _, num_kv_heads, head_dim, num_attention_heads = _read_model_dims(model)
    if num_kv_heads == 0 or head_dim == 0:
        return 0

    eff_chunk = min(int(chunk_size), int(new_tokens))
    full_kv_len = int(new_tokens) + max(0, int(cached_tokens))
    attn = _estimate_sdpa_activation_bytes(
        eff_chunk, full_kv_len, num_attention_heads, head_dim
    )

    kv_per_token = estimate_kv_bytes_per_token(model)
    kv = int(kv_per_token * int(new_tokens))
    return attn + kv


def raise_if_prefill_exceeds(
    model: Model,
    *,
    num_prompt_tokens: int,
    cached_tokens: int,
    prefill_step_size: int,
    hard_limit_bytes: int,
    current_usage_bytes: int,
    transient_override_bytes: int = 0,
) -> None:
    """Raise ``PrefillMemoryExceededError`` if a prefill peak won't fit.

    The shared front-door admission check, ported from oMLX's
    ``raise_if_prefill_exceeds``. Compare
    ``current_usage + peak`` against ``hard_limit_bytes``; raise with a clear
    message if it would exceed.

    ``peak`` is normally ``estimate_prefill_peak_bytes(...)`` (the static KV +
    SDPA estimate). When the per-model EWMA tracker has refined the
    transient upward (oMLX ``_admission_transient_bound``), pass that as
    ``transient_override_bytes`` and it replaces the static transient term —
    the new-KV charge still comes from the static estimate.

    No-op when the limit is unset, the model dims can't be read (estimator
    returns 0), or the request fits. The caller supplies
    ``current_usage_bytes`` so this never has to call MLX directly from the
    HTTP/event-loop preflight path.

    ``cached_tokens`` means prompt KV *already resident in current memory*
    (a prefix-cache hit), not merely "tokens that hit a cache". It reduces
    the new-KV charge but extends the SDPA K-dimension.
    """
    if hard_limit_bytes <= 0:
        return
    new_tokens = max(int(num_prompt_tokens) - max(int(cached_tokens), 0), 0)
    if new_tokens == 0:
        return
    static_peak = estimate_prefill_peak_bytes(
        model,
        new_tokens,
        chunk_size=max(1, int(prefill_step_size)),
        cached_tokens=int(cached_tokens),
    )
    if static_peak == 0 and transient_override_bytes <= 0:
        return  # can't estimate — skip, don't spuriously reject
    # The override refines the *transient* (SDPA) term; the new-KV charge is
    # always the static estimate. Split the static peak into its KV and
    # transient parts so the override only replaces the transient.
    override = max(0, int(transient_override_bytes))
    if override > 0 and static_peak > 0:
        # static_peak = kv + sdpa_static; replace sdpa_static with override.
        # Derive kv from the per-token rate so we don't double-count.
        kv_per_token = estimate_kv_bytes_per_token(model)
        kv = int(kv_per_token * int(new_tokens))
        peak = kv + override
    else:
        peak = static_peak if static_peak > 0 else override
    if peak == 0:
        return
    current = max(0, int(current_usage_bytes))
    if current + peak <= hard_limit_bytes:
        return
    message = (
        f"Prefill would require ~{_format_bytes(current + peak)} peak "
        f"(current {_format_bytes(current)} + KV+SDPA {_format_bytes(peak)}) "
        f"but the prefill ceiling is {_format_bytes(hard_limit_bytes)}. "
        f"Reduce context length or free memory."
    )
    logger.warning(
        "Preflight rejected (%d tokens, cached=%d): %s",
        num_prompt_tokens,
        cached_tokens,
        message,
    )
    raise PrefillMemoryExceededError(
        message=message,
        estimated_bytes=int(current + peak),
        limit_bytes=int(hard_limit_bytes),
    )


def _format_bytes(n: int) -> str:
    gb = n / (1024**3)
    if gb >= 1.0:
        return f"{gb:.2f} GiB"
    mb = n / (1024**2)
    return f"{mb:.1f} MiB"


def make_kv_cache(
    model: Model,
    max_kv_size: int | None = None,
    keep: int = 0,
    *,
    force_plain: bool = False,
) -> KVCacheType:
    """Construct the per-layer KV cache list.

    TurboQuant (``turboquant.effective_kv_bits()``) takes precedence over the
    legacy global ``KV_CACHE_BITS`` constant when enabled, mirroring oMLX's
    per-model ``turboquant_kv_enabled`` / ``turboquant_kv_bits``. The
    ``turboquant_skip_last`` setting keeps the final layer full precision to
    prevent corruption on quality-sensitive models (oMLX default: True).

    Qwen3-Next hybrid-cache awareness: models exposing ``make_cache`` (hybrid
    ``ArraysCache``+``KVCache`` models) defer to their own constructor, so the
    TurboQuant bits apply only to the plain-KV branches. This is the same
    seam oMLX's ``type_handlers.py`` keys on.

    ``force_plain`` skips quantization and builds plain ``KVCache`` layers
    regardless of TurboQuant / ``KV_CACHE_BITS``. This is the #1990
    correctness fix: mlx-lm's ``BatchGenerator`` does multi-sequence batched
    trim/extend on a single node, where ``QuantizedKVCache``'s state can
    desync across the batched sequences — so single-node batched mode must not
    quantize. (Distributed mode is unaffected: each rank holds one sequence
    shard.) Hybrid-cache models (``make_cache``) are also unaffected because
    they own their construction.
    """
    assert hasattr(model, "layers")

    if hasattr(model, "make_cache"):
        logger.info("Using MLX LM's make cache")
        return model.make_cache()  # type: ignore

    # TurboQuant overrides the legacy global KV_CACHE_BITS when enabled.
    tq_bits = turboquant.effective_kv_bits()
    effective_bits = tq_bits if tq_bits is not None else KV_CACHE_BITS
    skip_last = turboquant.is_turboquant_enabled() and turboquant.turboquant_skip_last()

    if force_plain:
        # #1990: single-node BatchGenerator mode — quantized KV is unsafe under
        # multi-sequence batched trim/extend. Build plain KVCache regardless of
        # the configured bits.
        if max_kv_size is None:
            logger.info(
                "Using plain KV cache (force_plain; #1990 single-node batched mode)"
            )
            return [KVCache() for _ in model.layers]
        logger.info(
            "Using rotating KV cache (force_plain; #1990) with %s with %s",
            f"{max_kv_size=}",
            f"{keep=}",
        )
        return [RotatingKVCache(max_size=max_kv_size, keep=keep) for _ in model.layers]

    if max_kv_size is None:
        if effective_bits is None:
            logger.info("Using default KV cache")
            return [KVCache() for _ in model.layers]
        else:
            source = "TurboQuant" if tq_bits is not None else "global KV_CACHE_BITS"
            logger.info(
                "Using quantized KV cache (source=%s, bits=%d)", source, effective_bits
            )
            caches: list[KVCache | QuantizedKVCache] = []
            for index, _layer in enumerate(model.layers):
                is_last_layer = index == len(model.layers) - 1
                if skip_last and is_last_layer:
                    caches.append(KVCache())
                else:
                    caches.append(
                        QuantizedKVCache(
                            group_size=CACHE_GROUP_SIZE, bits=effective_bits
                        )
                    )
            return caches
    else:
        logger.info(f"Using rotating KV cache with {max_kv_size=} with {keep=}")
        return [RotatingKVCache(max_size=max_kv_size, keep=keep) for _ in model.layers]
