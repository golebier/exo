# type: ignore
"""Tests for TurboQuant correctness hardening (#1990, #2261).

#1990 — skip KV quantization in single-node BatchGenerator mode: mlx-lm's
``BatchGenerator`` does multi-sequence batched trim/extend where
``QuantizedKVCache``'s state can desync, so single-node batched mode must build
plain ``KVCache`` regardless of the configured bits.

#2261 — force a clean prefill after chained prefix-cache extensions (quantized
only): a partial hit that would *extend* an entry which was itself produced by
an extension reuses quantized KV state accumulated across two extensions; the
quantization boundaries can desync and corrupt output. Reuse is refused and the
whole prompt is recomputed. Plain caches are always safe to chain.
"""

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache, QuantizedKVCache

from exo.shared.types.common import ModelId
from exo.worker.engines.mlx import turboquant
from exo.worker.engines.mlx.cache import KVPrefixCache, make_kv_cache


class _FakeModel:
    layers: list[object] = [object(), object(), object()]


@pytest.fixture(autouse=True)
def _reset_turboquant():
    yield
    turboquant.reset_runtime_overrides()


class TestSkipQuantInSingleNodeBatchedMode:
    """#1990: ``make_kv_cache(force_plain=True)`` builds plain KVCache."""

    def test_force_plain_overrides_turboquant(self):
        turboquant.set_turboquant_enabled(True)
        turboquant.set_turboquant_bits(4.0)
        cache = make_kv_cache(_FakeModel(), force_plain=True)
        assert len(cache) == 3
        assert all(isinstance(c, KVCache) for c in cache)
        assert not any(isinstance(c, QuantizedKVCache) for c in cache)

    def test_force_plain_overrides_legacy_kv_cache_bits(self):
        # With KV_CACHE_BITS set globally and force_plain, still plain.
        import exo.worker.engines.mlx.cache as cache_mod

        orig = cache_mod.KV_CACHE_BITS
        cache_mod.KV_CACHE_BITS = 4
        try:
            cache = make_kv_cache(_FakeModel(), force_plain=True)
            assert all(isinstance(c, KVCache) for c in cache)
        finally:
            cache_mod.KV_CACHE_BITS = orig

    def test_no_force_plain_quantizes_when_turboquant_on(self):
        turboquant.set_turboquant_enabled(True)
        turboquant.set_turboquant_bits(4.0)
        cache = make_kv_cache(_FakeModel())  # force_plain defaults to False
        # skip-last keeps the last layer plain; the rest are quantized.
        assert any(isinstance(c, QuantizedKVCache) for c in cache)


class TestChainedExtensionCleanPrefill:
    """#2261: a partial hit on a chained quantized entry forces a clean prefill."""

    def _seed(self, cache: KVPrefixCache, prompt: mx.array, quantized: bool):
        """Add an entry directly (bypassing add_kv_cache to control the cache)."""
        layer = QuantizedKVCache(group_size=64, bits=4) if quantized else KVCache()
        layer.keys = mx.array([[[float(p)] for p in range(len(prompt))]])
        layer.values = mx.array([[[float(p)] for p in range(len(prompt))]])
        layer.offset = len(prompt)
        cache.prompts.append(prompt)
        cache.caches.append([layer])
        cache._snapshots.append(None)
        cache._media_regions.append([])
        cache._last_used.append(1)
        cache.prefill_tps.append(0.0)
        cache._chained.append(False)

    def test_quantized_chained_partial_hit_forces_clean_prefill(self):
        cache = KVPrefixCache(group=None)
        cache.set_model_id(ModelId("m"))
        base = mx.array([1, 2, 3, 4, 5])
        self._seed(cache, base, quantized=True)
        # Simulate a chained extension: mark the entry chained.
        cache._chained[0] = True

        # A longer prompt that partially matches (would extend the chained entry).
        longer = mx.array([1, 2, 3, 4, 5, 6, 7])
        got, remaining, idx, exact = cache.get_kv_cache(_FakeModel(), longer)

        assert idx is None
        assert exact is False
        # Clean prefill: the whole prompt is the remaining tokens.
        assert len(remaining) == len(longer)

    def test_quantized_exact_hit_still_reuses(self):
        """An exact hit on a chained quantized entry is fine (no extension)."""
        cache = KVPrefixCache(group=None)
        cache.set_model_id(ModelId("m"))
        base = mx.array([1, 2, 3, 4, 5])
        self._seed(cache, base, quantized=True)
        cache._chained[0] = True

        # Exact match reuses the entry (trimmed to max_length-1 for generation).
        got, remaining, idx, exact = cache.get_kv_cache(_FakeModel(), base)
        assert idx is not None
        assert exact is True
        assert len(remaining) <= 1  # last token kept for stream_generate

    def test_plain_chained_partial_hit_still_reuses(self):
        """A non-quantized chained entry is always safe to extend (#2261 gated)."""
        cache = KVPrefixCache(group=None)
        cache.set_model_id(ModelId("m"))
        base = mx.array([1, 2, 3, 4, 5])
        self._seed(cache, base, quantized=False)
        cache._chained[0] = True

        longer = mx.array([1, 2, 3, 4, 5, 6, 7])
        got, remaining, idx, exact = cache.get_kv_cache(_FakeModel(), longer)
        assert idx is not None  # reused, not a clean prefill
        assert exact is False
        assert len(remaining) == 2  # the extension tokens

    def test_quantized_non_chained_partial_hit_still_reuses(self):
        """A quantized but *non-chained* (fresh) entry is safe to extend once."""
        cache = KVPrefixCache(group=None)
        cache.set_model_id(ModelId("m"))
        base = mx.array([1, 2, 3, 4, 5])
        self._seed(cache, base, quantized=True)
        # Not chained → first extension is allowed.
        assert cache._chained[0] is False
        longer = mx.array([1, 2, 3, 4, 5, 6, 7])
        got, remaining, idx, exact = cache.get_kv_cache(_FakeModel(), longer)
        assert idx is not None  # reused
        assert len(remaining) == 2

    def test_update_kv_cache_marks_entry_chained(self):
        """``update_kv_cache`` (the extension path) sets the chained flag."""
        cache = KVPrefixCache(group=None)
        base = mx.array([1, 2, 3, 4, 5])
        self._seed(cache, base, quantized=True)
        assert cache._chained[0] is False
        layer = QuantizedKVCache(group_size=64, bits=4)
        layer.keys = mx.array([[[float(p)] for p in range(7)]])
        layer.values = mx.array([[[float(p)] for p in range(7)]])
        layer.offset = 7
        cache.update_kv_cache(
            0, mx.array([1, 2, 3, 4, 5, 6, 7]), [layer], None, restore_pos=5
        )
        assert cache._chained[0] is True
