# type: ignore
"""Tests for the SSD cold tier of the KV prefix cache (oMLX doc 01, Phases 2–3).

Covers spill/restore round-trip (byte-exactness), restart-recovery scan, cache-
signature refusal on model/quant swap, SSD-eligibility for exotic cache classes,
LRU size-cap eviction, and the ``KVPrefixCache`` integration (evict → spill →
restore skipping re-prefill). The tier is default-off, so every test enables it
explicitly via the ``turboquant`` runtime override.
"""

from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache, QuantizedKVCache

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
        # All entries use the same token count so their on-disk sizes match
        # (the tokens sidecar scales with prompt length).
        store = SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=one_size + 1)
        p1, p2 = _tokens(4), mx.array([10, 11, 12, 13])  # distinct, same length
        store.spill(p1, _filled_kv_cache(tokens=4), model_id=_TEST_MODEL)
        store.spill(p2, _filled_kv_cache(tokens=4), model_id=_TEST_MODEL)
        assert store.has(p2) is True
        assert store.has(p1) is False  # evicted as LRU

    def test_restore_touches_lru_order(self, ssd_dir: Path):
        one_size = self._one_entry_size(ssd_dir)
        # Cap holds two entries but not three (all entries same on-disk size).
        store = SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=one_size * 2 + 1)
        p1, p2 = _tokens(4), mx.array([10, 11, 12, 13])
        store.spill(p1, _filled_kv_cache(tokens=4), model_id=_TEST_MODEL)
        store.spill(p2, _filled_kv_cache(tokens=4), model_id=_TEST_MODEL)
        # Restore p1 (the older one) so it becomes most-recently-used.
        assert store.restore(p1, model_id=_TEST_MODEL)[0] is not None
        # A third spill exceeds the cap; p2 (now LRU) is evicted, p1 survives.
        store.spill(
            mx.array([20, 21, 22, 23]), _filled_kv_cache(tokens=4), model_id=_TEST_MODEL
        )
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


class TestPrefixRestore:
    """Longest-common-prefix SSD restore (oMLX doc 01 finish)."""

    @staticmethod
    def _filled_arrays_cache(tokens: int = 4) -> list[ArraysCache]:
        # ArraysCache is SSD-eligible but NOT trimmable — used to exercise the
        # partial-restore refusal path.
        cache = [ArraysCache(size=1) for _ in range(2)]
        for c in cache:
            c.cache = [mx.array(np.random.randn(2, tokens, 8).astype(np.float32))]
        return cache

    def test_finds_longest_common_prefix_entry_is_prefix_of_query(self, ssd_dir: Path):
        """Entry prompt is a strict prefix of the query (the agentic case)."""
        store = SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=0)
        entry_prompt = _tokens(8)  # [0..7]
        store.spill(entry_prompt, _filled_kv_cache(tokens=8), model_id=_TEST_MODEL)
        # Query = entry + 2 new tokens. Exact restore misses (hash differs).
        query = mx.array(list(range(10)))
        assert store.restore(query, model_id=_TEST_MODEL)[0] is None
        restored, prefix_len = store.restore_prefix(query, model_id=_TEST_MODEL)
        assert restored is not None
        assert prefix_len == 8  # the whole entry is the prefix
        assert cache_length(restored) == 8  # no trim needed

    def test_trims_when_entry_diverges_mid_query(self, ssd_dir: Path):
        """Entry shares only a prefix with the query and must be trimmed."""
        store = SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=0)
        store.spill(_tokens(8), _filled_kv_cache(tokens=8), model_id=_TEST_MODEL)
        # Query diverges at token 4: [0,1,2,3,99,...]
        query = mx.array([0, 1, 2, 3, 99, 100, 101, 102])
        restored, prefix_len = store.restore_prefix(query, model_id=_TEST_MODEL)
        assert restored is not None
        assert prefix_len == 4
        # The loaded cache is still at the entry's full offset (8); the caller
        # trims it down to the 4-token prefix.
        assert cache_length(restored) == 8

    def test_picks_longest_among_multiple_entries(self, ssd_dir: Path):
        store = SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=0)
        # Entry A shares 3 tokens; entry B shares 6 tokens with the query.
        store.spill(
            mx.array([0, 1, 2, 50]), _filled_kv_cache(tokens=4), model_id=_TEST_MODEL
        )
        store.spill(
            mx.array([0, 1, 2, 3, 4, 5, 60]),
            _filled_kv_cache(tokens=7),
            model_id=_TEST_MODEL,
        )
        query = mx.array([0, 1, 2, 3, 4, 5, 99])
        restored, prefix_len = store.restore_prefix(query, model_id=_TEST_MODEL)
        assert restored is not None
        assert prefix_len == 6  # entry B's longer prefix wins

    def test_no_common_prefix_returns_none(self, ssd_dir: Path):
        store = SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=0)
        store.spill(_tokens(4), _filled_kv_cache(tokens=4), model_id=_TEST_MODEL)
        # Wholly disjoint prompt.
        restored, prefix_len = store.restore_prefix(
            mx.array([100, 101, 102, 103]), model_id=_TEST_MODEL
        )
        assert restored is None
        assert prefix_len == 0

    def test_refuses_partial_restore_for_non_trimmable_entry(self, ssd_dir: Path):
        """ArraysCache can't be trimmed back to a partial prefix → refuse."""
        store = SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=0)
        store.spill(
            _tokens(8), self._filled_arrays_cache(tokens=8), model_id=_TEST_MODEL
        )
        # Partial prefix (4 of 8) on a non-trimmable entry → refused.
        query = mx.array([0, 1, 2, 3, 99, 100, 101, 102])
        restored, prefix_len = store.restore_prefix(query, model_id=_TEST_MODEL)
        assert restored is None
        assert prefix_len == 0

    def test_exact_match_restored_for_non_trimmable_entry(self, ssd_dir: Path):
        """Exact match needs no trim → offered even for non-trimmable entries."""
        store = SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=0)
        prompt = _tokens(8)
        store.spill(prompt, self._filled_arrays_cache(tokens=8), model_id=_TEST_MODEL)
        restored, prefix_len = store.restore_prefix(prompt, model_id=_TEST_MODEL)
        assert restored is not None
        assert prefix_len == 8

    def test_falls_back_to_exact_when_sidecar_missing(self, ssd_dir: Path):
        """A recovered entry whose sidecar was lost still serves exact restore."""
        store = SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=0)
        prompt = _tokens(8)
        store.spill(prompt, _filled_kv_cache(tokens=8), model_id=_TEST_MODEL)
        # Simulate a lost/unreadable sidecar by clearing the indexed tokens.
        digest = next(iter(store._index))
        store._index[digest] = replace(store._index[digest], prompt_tokens=None)
        # Prefix scan finds nothing, but the exact-hash fallback restores.
        restored, prefix_len = store.restore_prefix(prompt, model_id=_TEST_MODEL)
        assert restored is not None
        assert prefix_len == 8

    def test_prefix_restore_works_after_restart_recovery(self, ssd_dir: Path):
        """The recovery scan loads the tokens sidecar so prefix restore works."""
        store1 = SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=0)
        store1.spill(_tokens(8), _filled_kv_cache(tokens=8), model_id=_TEST_MODEL)
        # Simulate a restart: brand-new store over the same dir.
        store2 = SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=0)
        query = mx.array(list(range(10)))  # entry is a prefix of the query
        restored, prefix_len = store2.restore_prefix(query, model_id=_TEST_MODEL)
        assert restored is not None
        assert prefix_len == 8


class TestPrefixRestoreIntegration:
    """End-to-end prefix restore through ``KVPrefixCache.get_kv_cache``."""

    def test_prefix_restore_prefills_only_suffix(self, ssd_dir: Path, monkeypatch):
        cache = KVPrefixCache(group=None)
        cache.set_model_id(_TEST_MODEL)
        store = SSDKVCacheStore(ssd_dir=ssd_dir, max_size_bytes=0)
        cache.set_ssd_store(store)

        # Spill an 8-token entry, then evict it from RAM to SSD.
        entry_prompt = _tokens(8)
        cache.add_kv_cache(entry_prompt, _filled_quant_cache(tokens=8))
        monkeypatch.setattr(cache, "_memory_pressure_exceeds", lambda _t: True)
        cache._evict_until_under(1.0)
        assert len(cache.caches) == 0
        assert store.has(entry_prompt)

        # Query = entry + 2 new tokens. RAM miss → exact SSD miss → prefix hit.
        query = mx.array(list(range(10)))
        got, remaining, idx, exact = cache.get_kv_cache(_FakeModel(), query)
        assert idx is None
        assert exact is False  # not an exact match of the query
        assert cache_length(got) == 8  # restored to the 8-token prefix
        assert list(np.asarray(np.array(remaining))) == [8, 9]
        # The prefix-restored entry is now in RAM (truncated prompt).
        assert len(cache.caches) == 1
        assert len(cache.prompts[0]) == 8


class _FakeModel:
    """Minimal stand-in so ``get_kv_cache``'s ``make_kv_cache(model)`` fallback
    isn't reached on the SSD-hit path (we only exercise the SSD restore branch,
    which never calls ``make_kv_cache``)."""

    layers: list[object] = []
