"""Tests for ``collect_evalable_cache_arrays`` / ``flush_prefill_for_decode``.

The critical bug: GLM-5.2 / DeepSeek-V3 MLA models use ``CacheList`` (a tuple
of two ``KVCache`` objects per layer).  The old
``mx.eval([c.state for c in cache])`` inside ``contextlib.suppress(Exception)``
silently no-op'd because ``KVCache.state`` raises ``AttributeError`` when
``keys is None`` (uninitialised indexer / shared-layer cache), and a single
raising sub-cache aborted the entire list comprehension.  This left the full
prefill lazy graph pending, intermittently stalling the JACCL TP collective.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx_lm.models.cache import CacheList, KVCache, QuantizedKVCache

from exo.worker.engines.mlx.generator.generate import (
    collect_evalable_cache_arrays,
    flatten_state_into,
    flush_prefill_for_decode,
)

EXO_TESTS = "1"


# ─── helpers ──────────────────────────────────────────────────────────────


def _populate(kv: KVCache, seq_len: int = 50, heads: int = 4, dim: int = 64) -> None:
    """Populate a KVCache so ``.state`` doesn't raise (keys is not None)."""
    keys = mx.zeros((1, heads, seq_len, dim))
    values = mx.zeros((1, heads, seq_len, dim))
    k, v = kv.update_and_fetch(keys, values)  # type: ignore[reportAnyMemberAccess]
    mx.eval(k, v)  # type: ignore[reportUnknownArgumentType]


# ─── flatten_state_into ──────────────────────────────────────────────────


def test_flatten_state_into_flat_arrays():
    out: list[mx.array] = []
    flatten_state_into(mx.array(1.0), out)
    flatten_state_into(mx.array(2.0), out)
    assert len(out) == 2


def test_flatten_state_into_nested_tuple():
    out: list[mx.array] = []
    flatten_state_into((mx.array(1.0), (mx.array(2.0), mx.array(3.0))), out)
    assert len(out) == 3


def test_flatten_state_into_nested_list():
    out: list[mx.array] = []
    flatten_state_into([mx.array(1.0), [mx.array(2.0), mx.array(3.0)]], out)
    assert len(out) == 3


def test_flatten_state_into_skips_none_and_scalars():
    out: list[mx.array] = []
    flatten_state_into([None, mx.array(1.0), 42, "str", mx.array(2.0)], out)
    assert len(out) == 2


# ─── collect_evalable_cache_arrays: flat KVCache list ────────────────────


def test_collect_flat_kv_cache_list():
    kv1 = KVCache()
    kv2 = KVCache()
    _populate(kv1)
    _populate(kv2)
    arrays = collect_evalable_cache_arrays([kv1, kv2])
    # Each KVCache.state → (keys, values) → 2 arrays
    assert len(arrays) == 4
    mx.eval(arrays)  # type: ignore[reportArgumentType]  # should not raise


def test_collect_skips_uninitialized_kv_cache():
    kv1 = KVCache()
    kv2 = KVCache()  # uninitialized — keys is None
    _populate(kv1)
    arrays = collect_evalable_cache_arrays([kv1, kv2])
    # Only kv1 contributes (2 arrays); kv2 skipped (state raises)
    assert len(arrays) == 2
    mx.eval(arrays)  # type: ignore[reportArgumentType]


def test_collect_all_uninitialized_returns_empty():
    arrays = collect_evalable_cache_arrays([KVCache(), KVCache(), KVCache()])
    assert arrays == []


# ─── collect_evalable_cache_arrays: CacheList (GLM-5.2 / DeepSeek-V3) ────


def test_collect_cachelist_both_populated():
    """GLM-5.2 routed layer: CacheList(KVCache, KVCache) both populated."""
    kv_a = KVCache()
    kv_b = KVCache()
    _populate(kv_a)
    _populate(kv_b)
    cl = CacheList(kv_a, kv_b)
    arrays = collect_evalable_cache_arrays([cl])
    assert len(arrays) == 4  # 2 KVCache × 2 arrays each
    mx.eval(arrays)  # type: ignore[reportArgumentType]


def test_collect_cachelist_one_uninitialized():
    """The root-cause bug: CacheList(KVCache, KVCache) where the second is
    uninitialized (keys is None).  Old code skipped ALL layers; new code
    still evaluates the first."""
    kv_a = KVCache()
    kv_b = KVCache()  # uninitialized
    _populate(kv_a)
    cl = CacheList(kv_a, kv_b)
    arrays = collect_evalable_cache_arrays([cl])
    # kv_a contributes 2 arrays; kv_b skipped
    assert len(arrays) == 2
    mx.eval(arrays)  # type: ignore[reportArgumentType]


def test_collect_cachelist_all_uninitialized():
    """All sub-caches uninitialized → empty list (no error)."""
    cl = CacheList(KVCache(), KVCache())
    arrays = collect_evalable_cache_arrays([cl])
    assert arrays == []


def test_collect_mixed_cachelist_and_flat():
    """Realistic GLM-5.2 cache: some layers CacheList, some flat KVCache."""
    kv_flat = KVCache()
    _populate(kv_flat)
    kv_a = KVCache()
    kv_b = KVCache()
    _populate(kv_a)
    _populate(kv_b)
    cl = CacheList(kv_a, kv_b)
    # Third layer: CacheList with one uninitialized sub-cache
    kv_c = KVCache()
    _populate(kv_c)
    cl_partial = CacheList(kv_c, KVCache())  # second uninitialized
    cache = [kv_flat, cl, cl_partial]
    arrays = collect_evalable_cache_arrays(cache)
    # kv_flat: 2, cl: 4, cl_partial: 2 (kv_c only) = 8
    assert len(arrays) == 8
    mx.eval(arrays)  # type: ignore[reportArgumentType]


def test_collect_quantized_kv_cache():
    """QuantizedKVCache.state returns (keys, values) where keys/values are
    lists of arrays (quantized representation)."""
    qkv = QuantizedKVCache(group_size=64, bits=8)
    # Populate — QuantizedKVCache uses update_and_fetch too
    keys = mx.zeros((1, 4, 50, 64))
    values = mx.zeros((1, 4, 50, 64))
    k, v = qkv.update_and_fetch(keys, values)  # type: ignore[reportAnyMemberAccess]
    mx.eval(k, v)  # type: ignore[reportUnknownArgumentType]
    arrays = collect_evalable_cache_arrays([qkv])
    assert len(arrays) >= 2  # at least keys + values
    mx.eval(arrays)  # type: ignore[reportArgumentType]


# ─── flush_prefill_for_decode: integration ────────────────────────────────


def test_flush_prefill_for_decode_no_group():
    """flush_prefill_for_decode with group=None should still eval + gc + clear."""
    kv1 = KVCache()
    kv2 = KVCache()
    _populate(kv1)
    _populate(kv2)
    cl = CacheList(kv1, KVCache())  # second uninitialized
    # Should not raise even with an uninitialized sub-cache
    flush_prefill_for_decode([cl], group=None)


def test_flush_prefill_for_decode_empty_cache():
    """All-uninitialized cache should be a no-op (no error)."""
    cl = CacheList(KVCache(), KVCache())
    flush_prefill_for_decode([cl], group=None)


def test_flush_prefill_for_decode_evaluates_pending_ops():
    """Verify that flush actually evaluates pending lazy ops on the cache."""
    kv = KVCache()
    _populate(kv, seq_len=100)
    # Create a pending lazy op: trim leaves a lazy slice in .state
    # (KVCache.state returns keys[..., :offset, :] when offset < keys.shape[2])
    # After populate, offset == 100 == keys.shape[2] (full), so trim to create
    # a partial offset:
    kv.trim(50)
    # Now offset=50 < keys.shape[2], so .state returns a lazy slice
    arrays = collect_evalable_cache_arrays([kv])
    assert len(arrays) == 2
    # Eval should succeed and materialize the lazy slices
    mx.eval(arrays)  # type: ignore[reportArgumentType]
    # Verify the trim took effect
    assert kv.offset == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
