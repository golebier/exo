# type: ignore
"""Tests for the SSD cold tier of the KV prefix cache (oMLX doc 01, Phases 2–3).

Covers spill/restore round-trip (byte-exactness), restart-recovery scan, cache-
signature refusal on model/quant swap, SSD-eligibility for exotic cache classes,
LRU size-cap eviction, and the ``KVPrefixCache`` integration (evict → spill →
restore skipping re-prefill). The tier is default-off, so every test enables it
explicitly via the ``turboquant`` runtime override.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import KVCache, QuantizedKVCache

from exo.shared.types.common import ModelId
from exo.worker.engines.mlx import turboquant
from exo.worker.engines.mlx.cache import KVPrefixCache, cache_length
from exo.worker.engines.mlx.ssd_cache import SSDKVCacheStore

_TEST_MODEL = ModelId("test-ssd-model")


@pytest.fixture
def ssd_dir(tmp_path: Path) -> Path:
    return tmp_path / "kv_ssd"


@pytest.fixture(autouse=True)
def _enable_tiered_cache(ssd_dir: Path):
    """Enable the SSD tier for every test and restore defaults after."""
    turboquant.set_tiered_cache_enabled(True)
    turboquant.set_hot_cache_only(False)
    turboquant.set_ssd_cache_dir(str(ssd_dir))
    turboquant.set_ssd_cache_max_size("0")  # no cap by default
    yield
    turboquant.reset_runtime_overrides()


def _filled_quant_cache(num_layers: int = 2, tokens: int = 4) -> list[QuantizedKVCache]:
    cache = [QuantizedKVCache(group_size=64, bits=4) for _ in range(num_layers)]
    k = mx.array(np.random.randn(1, 8, tokens, 64).astype(np.float16))
    v = mx.array(np.random.randn(1, 8, tokens, 64).astype(np.float16))
    for c in cache:
        c.keys = k
        c.values = v
        c.offset = tokens
    return cache


def _filled_kv_cache(num_layers: int = 2, tokens: int = 4) -> list[KVCache]:
    cache = [KVCache() for _ in range(num_layers)]
    k = mx.array(np.random.randn(1, 8, tokens, 64).astype(np.float16))
    v = mx.array(np.random.randn(1, 8, tokens, 64).astype(np.float16))
    for c in cache:
        c.keys = k
        c.values = v
        c.offset = tokens
    return cache


def _tokens(n: int) -> mx.array:
    return mx.array(list(range(n)))


class TestSpillRestoreRoundTrip:
    def test_spill_then_restore_is_byte_exact(self, ssd_dir: Path):
        store = SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=0)
        cache = _filled_quant_cache(tokens=8)
        prompt = _tokens(8)

        assert store.spill(prompt, cache, model_id=_TEST_MODEL) is True
        restored, n = store.restore(prompt, model_id=_TEST_MODEL)

        assert restored is not None
        assert n == 8
        assert len(restored) == len(cache)
        for orig, got in zip(cache, restored, strict=True):
            assert isinstance(got, QuantizedKVCache)
            assert got.bits == orig.bits
            assert got.group_size == orig.group_size
            assert got.offset == orig.offset
            assert bool(mx.array_equal(orig.keys, got.keys))
            assert bool(mx.array_equal(orig.values, got.values))

    def test_restore_miss_returns_none(self, ssd_dir: Path):
        store = SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=0)
        store.spill(_tokens(8), _filled_quant_cache(tokens=8), model_id=_TEST_MODEL)
        restored, n = store.restore(_tokens(99), model_id=_TEST_MODEL)
        assert restored is None
        assert n == 0

    def test_has_is_cheap_membership_check(self, ssd_dir: Path):
        store = SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=0)
        prompt = _tokens(8)
        assert store.has(prompt) is False
        store.spill(prompt, _filled_quant_cache(tokens=8), model_id=_TEST_MODEL)
        assert store.has(prompt) is True
        assert store.has(_tokens(99)) is False

    def test_plain_kv_cache_round_trips(self, ssd_dir: Path):
        store = SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=0)
        cache = _filled_kv_cache(tokens=4)
        store.spill(_tokens(4), cache, model_id=_TEST_MODEL)
        restored, n = store.restore(_tokens(4), model_id=_TEST_MODEL)
        assert restored is not None
        assert n == 4
        assert bool(mx.array_equal(cache[0].keys, restored[0].keys))


class TestSignatureGuard:
    def test_model_swap_refuses_restore(self, ssd_dir: Path):
        store = SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=0)
        prompt = _tokens(8)
        store.spill(prompt, _filled_quant_cache(tokens=8), model_id=_TEST_MODEL)
        # Restore under a *different* model id → signature mismatch → refused.
        other = ModelId("different-model")
        restored, n = store.restore(prompt, model_id=other)
        assert restored is None
        assert n == 0
        # The stale block is removed on refusal.
        assert store.has(prompt) is False

    def test_quant_bits_swap_refuses_restore(self, ssd_dir: Path):
        store = SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=0)
        prompt = _tokens(8)
        # Spill with 4-bit, then flip TurboQuant to 2-bit and restore.
        store.spill(prompt, _filled_quant_cache(tokens=8), model_id=_TEST_MODEL)
        turboquant.set_turboquant_enabled(True)
        turboquant.set_turboquant_bits(2.0)
        restored, n = store.restore(prompt, model_id=_TEST_MODEL)
        assert restored is None
        assert n == 0


class TestSSDEligibility:
    def test_exotic_cache_class_is_not_spilled(self, ssd_dir: Path):
        """A cache class mlx-lm can't reconstruct is SSD-ineligible (no spill)."""

        class _ExoticCache(KVCache):
            """A subclass not in mlx_lm.models.cache globals — not restorable."""

        cache = _filled_kv_cache(tokens=4)
        cache.append(_ExoticCache())
        store = SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=0)
        assert store.spill(_tokens(4), cache, model_id=_TEST_MODEL) is False
        # Nothing was written.
        assert list(ssd_dir.rglob("*.safetensors")) == []


class TestRestartRecovery:
    def test_new_store_indexes_existing_files(self, ssd_dir: Path):
        """A freshly-constructed store scans the SSD dir and reuses files."""
        store1 = SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=0)
        prompt = _tokens(8)
        store1.spill(prompt, _filled_quant_cache(tokens=8), model_id=_TEST_MODEL)
        assert store1.has(prompt)

        # Simulate a restart: brand-new store over the same dir.
        store2 = SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=0)
        assert store2.has(prompt)
        restored, n = store2.restore(prompt, model_id=_TEST_MODEL)
        assert restored is not None
        assert n == 8

    def test_legacy_unkeyed_file_is_removed_on_recovery(self, ssd_dir: Path):
        """A file missing the EXO metadata keys can't be validated → removed."""
        # Write a prompt cache with no EXO metadata (simulating an older build).
        from mlx_lm.models.cache import save_prompt_cache

        ssd_dir.mkdir(parents=True, exist_ok=True)
        bogus = ssd_dir / "aa" / "bb" / "legacy.safetensors"
        bogus.parent.mkdir(parents=True, exist_ok=True)
        save_prompt_cache(str(bogus), _filled_kv_cache(tokens=2), metadata={})
        assert bogus.exists()

        store = SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=0)
        # Recovery removed the unkeyed file.
        assert not bogus.exists()
        assert store.status()["ssd_entries"] == 0


class TestLRUSizeCap:
    def _one_entry_size(self, ssd_dir: Path) -> int:
        """Measure one spilled entry's on-disk size using a throwaway subdir."""
        probe_dir = ssd_dir.parent / (ssd_dir.name + "_probe")
        probe = SSDKVCacheStore(ssd_dir=probe_dir, max_size_bytes=0)
        probe.spill(_tokens(4), _filled_kv_cache(tokens=4), model_id=_TEST_MODEL)
        size = probe.status()["ssd_size_bytes"]
        assert size > 0
        return size

    def test_evicts_lru_when_over_cap(self, ssd_dir: Path):
        one_size = self._one_entry_size(ssd_dir)
        # Cap holds one entry but not two → the second spill evicts the LRU.
        # All entries use the same token count so their on-disk sizes match.
        store = SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=one_size + 1)
        p1, p2 = _tokens(4), _tokens(40)  # distinct prompts, same-size cache
        store.spill(p1, _filled_kv_cache(tokens=4), model_id=_TEST_MODEL)
        store.spill(p2, _filled_kv_cache(tokens=4), model_id=_TEST_MODEL)
        assert store.has(p2) is True
        assert store.has(p1) is False  # evicted as LRU

    def test_restore_touches_lru_order(self, ssd_dir: Path):
        one_size = self._one_entry_size(ssd_dir)
        # Cap holds two entries but not three (all entries same on-disk size).
        store = SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=one_size * 2 + 1)
        p1, p2 = _tokens(4), _tokens(40)
        store.spill(p1, _filled_kv_cache(tokens=4), model_id=_TEST_MODEL)
        store.spill(p2, _filled_kv_cache(tokens=4), model_id=_TEST_MODEL)
        # Restore p1 (the older one) so it becomes most-recently-used.
        assert store.restore(p1, model_id=_TEST_MODEL)[0] is not None
        # A third spill exceeds the cap; p2 (now LRU) is evicted, p1 survives.
        store.spill(_tokens(80), _filled_kv_cache(tokens=4), model_id=_TEST_MODEL)
        assert store.has(p1) is True
        assert store.has(p2) is False


class TestKVPrefixCacheIntegration:
    def test_disabled_tier_is_ram_only_noop(self, ssd_dir: Path):
        """With the tier off, eviction/restore are no-ops (today's behaviour)."""
        turboquant.set_tiered_cache_enabled(False)
        cache = KVPrefixCache(group=None)
        cache.set_model_id(_TEST_MODEL)
        cache.set_ssd_store(SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=0))
        # No SSD files should ever be written with the tier off.
        assert list(ssd_dir.rglob("*.safetensors")) == []
        # get_kv_cache with no RAM hit falls through to a fresh cache.
        got, remaining, idx, exact = cache.get_kv_cache(_FakeModel(), _tokens(8))
        assert idx is None
        assert exact is False

    def test_evict_spills_and_restore_skips_prefill(self, ssd_dir: Path, monkeypatch):
        cache = KVPrefixCache(group=None)
        cache.set_model_id(_TEST_MODEL)
        store = SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=0)
        cache.set_ssd_store(store)

        prompt = _tokens(8)
        kv = _filled_quant_cache(tokens=8)
        cache.add_kv_cache(prompt, kv)

        # Force eviction by making every pressure check report over-ceiling.
        assert store.has(prompt) is False
        monkeypatch.setattr(cache, "_memory_pressure_exceeds", lambda _t: True)
        cache._evict_until_under(1.0)
        # Eviction spilled the entry to SSD.
        assert len(cache.caches) == 0
        assert store.has(prompt) is True

        # A new lookup misses RAM but hits SSD → restores, skipping re-prefill.
        got, remaining, idx, exact = cache.get_kv_cache(_FakeModel(), prompt)
        assert len(got) == 2
        assert cache_length(got) == 8
        # Exact match ⇒ no remaining tokens to prefill.
        assert bool(mx.array_equal(remaining, mx.array([]))) or len(remaining) == 0
        # The restored entry is now back in RAM.
        assert len(cache.caches) == 1

    def test_model_swap_does_not_restore_stale_ssd_block(
        self, ssd_dir: Path, monkeypatch
    ):
        cache = KVPrefixCache(group=None)
        cache.set_model_id(_TEST_MODEL)
        store = SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=0)
        cache.set_ssd_store(store)
        prompt = _tokens(8)
        cache.add_kv_cache(prompt, _filled_quant_cache(tokens=8))
        monkeypatch.setattr(cache, "_memory_pressure_exceeds", lambda _t: True)
        cache._evict_until_under(1.0)
        assert store.has(prompt)

        # Swap the model id; the SSD signature guard refuses the restore.
        cache.set_model_id(ModelId("other-model"))
        got, remaining, idx, exact = cache.get_kv_cache(_FakeModel(), prompt)
        assert idx is None
        assert exact is False
        # A fresh cache was returned (re-prefill required), not the stale block.
        assert cache_length(got) == 0


class _FakeModel:
    """Minimal stand-in so ``get_kv_cache``'s ``make_kv_cache(model)`` fallback
    isn't reached on the SSD-hit path (we only exercise the SSD restore branch,
    which never calls ``make_kv_cache``)."""

    layers: list[object] = []
