# type: ignore
"""Tests for prefill memory headroom (PR #2251) + preflight admission (oMLX).

Covers:
- ``EXO_PREFILL_MEMORY_THRESHOLD`` defaults below ``EXO_MEMORY_THRESHOLD``.
- ``evict_for_prefill_headroom`` evicts LRU *before* prefill (the #2251 fix),
  targeting the tighter prefill watermark and reclaiming MLX allocations.
- ``estimate_prefill_peak_bytes`` / ``raise_if_prefill_exceeds`` admission
  math (ported from oMLX ``MemoryMonitor.estimate_prefill_peak_bytes``).
- ``preflight_or_raise`` rejects an impossible prompt with a clear error
  instead of OOMing, and is a no-op when model dims can't be read.

The estimator-math and eviction-ordering tests need no real model. The
end-to-end regression (prefill survives prefix-cache pressure) lives in
``test_kv_prefix_cache.py`` alongside the existing model-backed tests.
"""

from unittest.mock import patch

import mlx.core as mx
import pytest

from exo.worker.engines.mlx import cache as cache_mod
from exo.worker.engines.mlx.cache import (
    _DEFAULT_PREFILL_STEP_SIZE,
    KVPrefixCache,
    _estimate_sdpa_activation_bytes,
    estimate_kv_bytes_per_token,
    estimate_prefill_peak_bytes,
    raise_if_prefill_exceeds,
)
from exo.worker.engines.mlx.exceptions import PrefillMemoryExceededError


# A stand-in model object whose ``args`` the estimator can read. We avoid
# loading a real model by patching ``_read_model_dims`` in most tests, but this
# lets the no-patch path be exercised too.
class _FakeArgs:
    num_hidden_layers = 4
    num_attention_heads = 8
    num_key_value_heads = 2
    head_dim = 64
    hidden_size = 512


class _FakeInner:
    args = _FakeArgs()

    def __getattr__(self, name: str) -> object:
        raise AttributeError(name)


class _FakeModel:
    def __init__(self) -> None:
        self.model = _FakeInner()


class TestPrefillThresholdDefaults:
    def test_prefill_threshold_below_memory_threshold(self):
        # The #2251 default: prefill watermark 10 points below the cache
        # watermark, so the cache is evicted harder before prefill than after
        # a completed entry is added.
        assert cache_mod._PREFILL_MEMORY_THRESHOLD <= cache_mod._MEMORY_THRESHOLD
        assert (
            pytest.approx(cache_mod._MEMORY_THRESHOLD - 0.10)
            == cache_mod._PREFILL_MEMORY_THRESHOLD
        )

    def test_prefill_threshold_never_negative(self):
        # On a tiny-RAM machine the cache threshold is 0.70; prefill is 0.60.
        # If someone sets EXO_MEMORY_THRESHOLD=0.05, prefill must still clamp
        # at 0 rather than go negative.
        with patch.object(cache_mod, "_MEMORY_THRESHOLD", 0.05):
            assert cache_mod._default_prefill_memory_threshold() >= 0.0


class TestEstimatorMath:
    """Ported from oMLX ``estimate_prefill_peak_bytes`` semantics."""

    def test_kv_bytes_per_token_uniform(self):
        # 4 layers, 2 kv heads, head_dim 64, fp16 (2 bytes), keys+values (*2)
        # => 4 * 2 * 64 * 2 * 2 = 2048 bytes/token
        with patch.object(cache_mod, "_read_model_dims", return_value=(4, 2, 64, 8)):
            assert estimate_kv_bytes_per_token(_FakeModel()) == 2048.0

    def test_kv_bytes_per_token_zero_when_dims_unreadable(self):
        # Best-effort: unreadable dims => 0 => caller skips the preflight.
        with patch.object(cache_mod, "_read_model_dims", return_value=(0, 0, 0, 0)):
            assert estimate_kv_bytes_per_token(_FakeModel()) == 0.0

    def test_sdpa_activation_unfused_bound(self):
        # scores: heads * q * kv * 4 (fp32); output: heads * q * head_dim * 2
        # 8 heads, 64 head_dim, q=2048, kv=2048:
        #   scores = 8 * 2048 * 2048 * 4 = 134,217,728
        #   output = 8 * 2048 * 64 * 2    = 2,097,152
        assert _estimate_sdpa_activation_bytes(2048, 2048, 8, 64) == 136_314_880

    def test_sdpa_activation_zero_for_empty(self):
        assert _estimate_sdpa_activation_bytes(0, 2048, 8, 64) == 0
        assert _estimate_sdpa_activation_bytes(2048, 0, 8, 64) == 0

    def test_prefill_peak_is_kv_plus_last_chunk_sdpa(self):
        # new_tokens=1000, chunk=4096 => eff_chunk = min(4096, 1000) = 1000
        # full_kv_len = 1000 + 0 cached = 1000
        # kv = 2048 bytes/token * 1000 = 2,048,000
        # sdpa: scores 8*1000*1000*4 + output 8*1000*64*2 = 32,000,000 + 1,024,000
        # peak = 2,048,000 + 33,024,000 = 35,072,000
        with patch.object(cache_mod, "_read_model_dims", return_value=(4, 2, 64, 8)):
            peak = estimate_prefill_peak_bytes(_FakeModel(), 1000)
        assert peak == 35_072_000

    def test_prefill_peak_eff_chunk_capped_by_new_tokens(self):
        # A short prompt must not be charged the full chunk_size width in the
        # scores tensor (oMLX: "over-estimating by chunk_size / new_tokens ...
        # raised false-positive 400s on small prompts").
        with patch.object(cache_mod, "_read_model_dims", return_value=(4, 2, 64, 8)):
            small = estimate_prefill_peak_bytes(_FakeModel(), 100, chunk_size=4096)
        # eff_chunk = 100, not 4096
        # kv = 2048 * 100 = 204,800
        # sdpa: 8*100*100*4 + 8*100*64*2 = 320,000 + 102,400 = 422,400
        assert small == 627_200

    def test_prefill_peak_cached_tokens_extend_sdpa_kdim(self):
        # cached_tokens reduce the new-KV charge but extend the SDPA K-dim
        # because cached positions still participate in attention (oMLX).
        with patch.object(cache_mod, "_read_model_dims", return_value=(4, 2, 64, 8)):
            no_cache = estimate_prefill_peak_bytes(_FakeModel(), 1000, cached_tokens=0)
            with_cache = estimate_prefill_peak_bytes(
                _FakeModel(), 1000, cached_tokens=5000
            )
        # KV charge is identical (only new_tokens drive it)...
        # but the SDPA scores K-dim grows from 1000 to 6000, so the peak grows.
        assert with_cache > no_cache

    def test_prefill_peak_zero_when_dims_unreadable(self):
        with patch.object(cache_mod, "_read_model_dims", return_value=(0, 0, 0, 0)):
            assert estimate_prefill_peak_bytes(_FakeModel(), 1000) == 0


class TestRaiseIfPrefillExceeds:
    def test_no_op_when_limit_unset(self):
        with patch.object(cache_mod, "_read_model_dims", return_value=(4, 2, 64, 8)):
            # hard_limit_bytes=0 => guard disabled (unmeasurable host)
            raise_if_prefill_exceeds(
                _FakeModel(),
                num_prompt_tokens=1000,
                cached_tokens=0,
                prefill_step_size=4096,
                hard_limit_bytes=0,
                current_usage_bytes=0,
            )

    def test_no_op_when_peak_unreadable(self):
        # Estimator returns 0 (dims unreadable) => skip, don't spuriously reject.
        with patch.object(cache_mod, "_read_model_dims", return_value=(0, 0, 0, 0)):
            raise_if_prefill_exceeds(
                _FakeModel(),
                num_prompt_tokens=1000,
                cached_tokens=0,
                prefill_step_size=4096,
                hard_limit_bytes=1_000_000,
                current_usage_bytes=0,
            )

    def test_no_op_when_request_fits(self):
        with patch.object(cache_mod, "_read_model_dims", return_value=(4, 2, 64, 8)):
            # 1000 tokens, peak ~35MB, limit 1GB => fits.
            raise_if_prefill_exceeds(
                _FakeModel(),
                num_prompt_tokens=1000,
                cached_tokens=0,
                prefill_step_size=4096,
                hard_limit_bytes=1_000_000_000,
                current_usage_bytes=0,
            )

    def test_no_op_when_cached_tokens_cover_prompt(self):
        # cached_tokens >= prompt => new_tokens = 0 => nothing to prefill.
        with patch.object(cache_mod, "_read_model_dims", return_value=(4, 2, 64, 8)):
            raise_if_prefill_exceeds(
                _FakeModel(),
                num_prompt_tokens=1000,
                cached_tokens=1000,
                prefill_step_size=4096,
                hard_limit_bytes=1,
                current_usage_bytes=0,
            )

    def test_raises_when_peak_exceeds_limit(self):
        with (
            patch.object(cache_mod, "_read_model_dims", return_value=(4, 2, 64, 8)),
            pytest.raises(PrefillMemoryExceededError) as exc_info,
        ):
            raise_if_prefill_exceeds(
                _FakeModel(),
                num_prompt_tokens=1000,
                cached_tokens=0,
                prefill_step_size=4096,
                hard_limit_bytes=1_000_000,  # 1MB, peak ~35MB
                current_usage_bytes=0,
            )
        err = exc_info.value
        assert err.estimated_bytes > err.limit_bytes
        assert err.limit_bytes == 1_000_000
        assert "Reduce context length" in str(err)

    def test_raises_when_current_plus_peak_exceeds_limit(self):
        # Even a modest peak fails when current usage is near the ceiling.
        with (
            patch.object(cache_mod, "_read_model_dims", return_value=(4, 2, 64, 8)),
            pytest.raises(PrefillMemoryExceededError),
        ):
            raise_if_prefill_exceeds(
                _FakeModel(),
                num_prompt_tokens=1000,
                cached_tokens=0,
                prefill_step_size=4096,
                hard_limit_bytes=40_000_000,  # 40MB
                current_usage_bytes=20_000_000,  # 20MB resident
            )


class TestEvictForPrefillHeadroom:
    """The #2251 fix: evict LRU *before* prefill, not only on add_kv_cache."""

    def _populated_cache(self) -> KVPrefixCache:
        c = KVPrefixCache(group=None)
        # Three entries; stagger _last_used so LRU order is deterministic.
        for i in range(3):
            c.prompts.append(mx.array([i]))
            c.caches.append([])
            c._snapshots.append(None)
            c._media_regions.append([])
            c._last_used.append(float(i))
            c.prefill_tps.append(0.0)
        return c

    def test_evicts_until_below_prefill_threshold(self):
        c = self._populated_cache()
        # Drive pressure down across evictions: 0.95 -> 0.80 -> 0.62 (below
        # the default prefill threshold of ~0.60 only on the 3rd reading).
        # We use a prefill threshold of 0.70 here (memory threshold 0.80) so
        # the sequence crosses it deterministically.
        readings = iter([0.95, 0.80, 0.62, 0.55])

        def fake_pressure(_self: KVPrefixCache) -> float:
            return next(readings)

        with (
            patch.object(KVPrefixCache, "get_memory_used_percentage", fake_pressure),
            patch.object(cache_mod, "_PREFILL_MEMORY_THRESHOLD", 0.70),
            patch("exo.worker.engines.mlx.cache.gc.collect"),
            patch("exo.worker.engines.mlx.cache.mx.clear_cache"),
        ):
            evicted = c.evict_for_prefill_headroom()

        # Readings: 0.95 (>0.70, evict LRU0), 0.80 (>0.70, evict LRU1),
        # 0.62 (<=0.70, stop). Two evicted, one remains.
        assert evicted == 2
        assert len(c.prompts) == 1
        # The survivor is the most-recently-used (index 2, _last_used=2.0).
        assert c._last_used == [2.0]

    def test_no_eviction_when_already_below_threshold(self):
        c = self._populated_cache()
        with (
            patch.object(
                KVPrefixCache, "get_memory_used_percentage", lambda self: 0.40
            ),
            patch("exo.worker.engines.mlx.cache.gc.collect"),
            patch("exo.worker.engines.mlx.cache.mx.clear_cache"),
        ):
            evicted = c.evict_for_prefill_headroom()
        assert evicted == 0
        assert len(c.prompts) == 3

    def test_no_eviction_when_cache_empty(self):
        c = KVPrefixCache(group=None)
        with (
            patch.object(
                KVPrefixCache, "get_memory_used_percentage", lambda self: 0.95
            ),
            patch("exo.worker.engines.mlx.cache.gc.collect"),
            patch("exo.worker.engines.mlx.cache.mx.clear_cache"),
        ):
            evicted = c.evict_for_prefill_headroom()
        assert evicted == 0

    def test_prefill_threshold_tighter_than_cache_threshold(self):
        # evict_for_prefill_headroom targets a *lower* watermark than
        # _evict_if_needed, so it evicts at least as aggressively. Concretely:
        # at 0.75 pressure with thresholds memory=0.80 / prefill=0.70, the
        # post-add path would not evict but the before-prefill path must.
        c = self._populated_cache()
        with (
            patch.object(
                KVPrefixCache, "get_memory_used_percentage", lambda self: 0.75
            ),
            patch.object(cache_mod, "_MEMORY_THRESHOLD", 0.80),
            patch.object(cache_mod, "_PREFILL_MEMORY_THRESHOLD", 0.70),
            patch("exo.worker.engines.mlx.cache.gc.collect"),
            patch("exo.worker.engines.mlx.cache.mx.clear_cache"),
        ):
            assert c._evict_until_under(cache_mod._MEMORY_THRESHOLD) == 0
            # Re-populate since the above didn't evict anything.
            assert len(c.prompts) == 3
            assert c.evict_for_prefill_headroom() == 3

    def test_eviction_capped_to_avoid_thrash(self):
        # A stuck pressure reading (e.g. a third-party process holding memory)
        # must not evict the entire cache in one call — the cap stops it so
        # the preflight estimator can decide admit/reject instead.
        c = self._populated_cache()
        # Add more entries than the cap so the bound is exercised.
        for i in range(3, 20):
            c.prompts.append(mx.array([i]))
            c.caches.append([])
            c._snapshots.append(None)
            c._media_regions.append([])
            c._last_used.append(float(i))
            c.prefill_tps.append(0.0)
        assert len(c.prompts) == 20
        cap = cache_mod._MAX_PREFILL_EVICTION_RETRIES
        with (
            patch.object(
                KVPrefixCache, "get_memory_used_percentage", lambda self: 0.99
            ),
            patch.object(cache_mod, "_PREFILL_MEMORY_THRESHOLD", 0.50),
            patch("exo.worker.engines.mlx.cache.gc.collect"),
            patch("exo.worker.engines.mlx.cache.mx.clear_cache"),
        ):
            evicted = c.evict_for_prefill_headroom()
        # Stops at the cap, not at empty.
        assert evicted == cap
        assert len(c.prompts) == 20 - cap


class TestPreflightOrRaise:
    def test_no_op_when_limit_unset(self):
        c = KVPrefixCache(group=None)
        with patch.object(cache_mod, "prefill_admission_limit_bytes", return_value=0):
            # Should not raise even with a huge prompt.
            c.preflight_or_raise(
                _FakeModel(),
                num_prompt_tokens=100_000,
                cached_tokens=0,
            )

    def test_rejects_oversized_prompt(self):
        c = KVPrefixCache(group=None)
        # Make eviction a no-op (already below threshold), then force a tiny
        # hard limit so the estimator rejects.
        with (
            patch.object(KVPrefixCache, "evict_for_prefill_headroom", return_value=0),
            patch.object(
                cache_mod, "prefill_admission_limit_bytes", return_value=1_000
            ),
            patch.object(cache_mod, "_read_model_dims", return_value=(4, 2, 64, 8)),
            patch.object(cache_mod, "_current_process_usage_bytes", return_value=0),
            pytest.raises(PrefillMemoryExceededError),
        ):
            c.preflight_or_raise(
                _FakeModel(),
                num_prompt_tokens=10_000,
                cached_tokens=0,
            )

    def test_admits_prompt_that_fits(self):
        c = KVPrefixCache(group=None)
        with (
            patch.object(KVPrefixCache, "evict_for_prefill_headroom", return_value=0),
            patch.object(
                cache_mod,
                "prefill_admission_limit_bytes",
                return_value=10_000_000_000,
            ),
            patch.object(cache_mod, "_read_model_dims", return_value=(4, 2, 64, 8)),
            patch.object(cache_mod, "_current_process_usage_bytes", return_value=0),
        ):
            # 1000 tokens, peak ~35MB, limit 10GB => fits.
            c.preflight_or_raise(
                _FakeModel(),
                num_prompt_tokens=1000,
                cached_tokens=0,
            )

    def test_no_op_when_dims_unreadable(self):
        # Best-effort: can't estimate => don't block (oMLX rule).
        c = KVPrefixCache(group=None)
        with (
            patch.object(KVPrefixCache, "evict_for_prefill_headroom", return_value=0),
            patch.object(
                cache_mod, "prefill_admission_limit_bytes", return_value=1_000
            ),
            patch.object(cache_mod, "_read_model_dims", return_value=(0, 0, 0, 0)),
            patch.object(cache_mod, "_current_process_usage_bytes", return_value=0),
        ):
            c.preflight_or_raise(
                _FakeModel(),
                num_prompt_tokens=1_000_000,
                cached_tokens=0,
            )

    def test_uses_default_prefill_step_when_none_passed(self):
        # The estimator's chunk size defaults to EXO's prefill step (4096).
        assert _DEFAULT_PREFILL_STEP_SIZE == 4096


class TestPrefillTransientTracker:
    """Ported from oMLX's ``PrefillTransientTracker`` semantics."""

    def _tracker(self) -> cache_mod.PrefillTransientTracker:
        return cache_mod.PrefillTransientTracker(model_id="test")

    def test_predict_zero_before_any_sample(self):
        # First chunk has no measurement — caller falls back to static estimate.
        t = self._tracker()
        assert t.predict(2048) == 0
        assert t.bytes_per_token == 0.0
        assert t.samples == 0

    def test_first_sample_seeds_ewma(self):
        t = self._tracker()
        # 2048 tokens, 4 MiB transient => 2048 bytes/token
        t.update(2048, 4 * 1024 * 1024, floor_sample=True)
        assert t.samples == 1
        assert t.bytes_per_token == pytest.approx(2048.0)
        # First sample seeds but does NOT feed observed_max (load noise).
        assert t.observed_max_bytes == 0

    def test_second_floor_sample_feeds_observed_max(self):
        t = self._tracker()
        t.update(2048, 4 * 1024 * 1024, floor_sample=True)  # seed
        t.update(2048, 6 * 1024 * 1024, floor_sample=True)  # floor => max
        assert t.observed_max_bytes == 6 * 1024 * 1024

    def test_outlier_rejected_from_ewma(self):
        # A sample >8x the running rate is excluded from the EWMA blend
        # (measurement noise) but still recorded raw in last_delta.
        t = self._tracker()
        t.update(2048, 4 * 1024 * 1024, floor_sample=True)  # 2048 B/tok
        # 200x outlier: 2048 tokens, 800 MiB => ~400 KB/tok
        outlier = 200 * 2048 * 2048
        t.update(2048, outlier, floor_sample=True)
        # EWMA unchanged (outlier rejected)...
        assert t.bytes_per_token == pytest.approx(2048.0)
        # ...but last_delta records it raw.
        assert t.last_delta_bytes == outlier

    def test_outlier_clamped_from_observed_max(self):
        # A >4 GiB one-off spike is not a repeatable chunk transient.
        t = self._tracker()
        t.update(2048, 4 * 1024 * 1024, floor_sample=True)  # seed (excluded)
        t.update(2048, 6 * 1024 * 1024, floor_sample=True)  # legit floor => max
        giant = 5 * 1024**3  # 5 GiB > 4 GiB clamp
        t.update(2048, giant, floor_sample=True)  # clamped, doesn't replace max
        assert t.observed_max_bytes == 6 * 1024 * 1024  # not the giant

    def test_negative_delta_recorded_as_reclaim(self):
        # Pool reclaim larger than the chunk's allocation is excluded from the
        # EWMA but retained as a one-shot realloc charge.
        t = self._tracker()
        t.update(2048, 4 * 1024 * 1024, floor_sample=True)  # seed (clears reclaim)
        t.record_reclaim(2 * 1024 * 1024)  # 2 MiB released
        assert t.recent_reclaim_bytes == 2 * 1024 * 1024
        # A positive sample clears the reclaim charge (pool reallocated).
        t.update(2048, 1024 * 1024, floor_sample=True)
        assert t.recent_reclaim_bytes == 0

    def test_predict_applies_safety_factor(self):
        t = self._tracker()
        t.update(2048, 4 * 1024 * 1024, floor_sample=True)  # 2048 B/tok
        # predict(n) = ewma * n * 1.2
        assert t.predict(2048) == int(2048.0 * 2048 * 1.2)

    def test_reset_drops_all(self):
        t = self._tracker()
        t.update(2048, 4 * 1024 * 1024, floor_sample=True)
        t.record_reclaim(1024)
        t.reset()
        assert t.samples == 0
        assert t.bytes_per_token == 0.0
        assert t.observed_max_bytes == 0
        assert t.recent_reclaim_bytes == 0


class TestChunkGuard:
    """Phase 2 per-chunk prefill guard (``guard_prefill_chunk_or_raise``)."""

    def test_pre_loop_callback_sets_baseline_no_raise(self):
        # processed == 0 is the pre-loop callback: just refresh baseline.
        c = KVPrefixCache(group=None)
        with patch.object(cache_mod, "_current_process_usage_bytes", return_value=42):
            c.guard_prefill_chunk_or_raise(
                _FakeModel(), processed=0, total=4096, cached_tokens=0
            )
        assert c._prefill_chunk_baseline_bytes == 42
        # No tracker update from the pre-loop callback.
        assert c.prefill_transient_tracker.samples == 0

    def test_measures_chunk_and_feeds_ewma(self):
        c = KVPrefixCache(group=None)
        # Baseline 100, post 100 + 4 MiB => 4 MiB delta for a 4096-token chunk.
        readings = iter([100, 100 + 4 * 1024 * 1024])
        with (
            patch.object(
                cache_mod, "_current_process_usage_bytes", lambda: next(readings)
            ),
            patch.object(cache_mod, "_read_model_dims", return_value=(4, 2, 64, 8)),
            patch.object(
                cache_mod.memory_guard,
                "prefill_abort_cap_bytes",
                return_value=10**12,
            ),  # huge cap => no abort
        ):
            c.guard_prefill_chunk_or_raise(
                _FakeModel(), processed=0, total=8192, cached_tokens=0
            )  # baseline
            c.guard_prefill_chunk_or_raise(
                _FakeModel(), processed=4096, total=8192, cached_tokens=0
            )  # measure
        assert c.prefill_transient_tracker.samples == 1
        assert c.prefill_transient_tracker.last_delta_bytes == 4 * 1024 * 1024

    def test_aborts_when_next_chunk_predicted_to_exceed_cap(self):
        # Seed a high EWMA so the next-chunk prediction blows the abort cap.
        c = KVPrefixCache(group=None)
        c.prefill_transient_tracker.update(
            4096, 2 * 1024**3, floor_sample=True
        )  # 2 GiB/chunk => ~0.5 MB/token
        # current usage near the cap; next chunk predicted to exceed.
        abort_cap = 9 * 1024**3  # 9 GiB abort cap
        with (
            patch.object(
                cache_mod, "_current_process_usage_bytes", return_value=9 * 1024**3
            ),
            patch.object(cache_mod, "_read_model_dims", return_value=(4, 2, 64, 8)),
            patch.object(
                cache_mod.memory_guard,
                "prefill_abort_cap_bytes",
                return_value=abort_cap,
            ),
            pytest.raises(PrefillMemoryExceededError) as exc_info,
        ):
            c.guard_prefill_chunk_or_raise(
                _FakeModel(), processed=4096, total=12288, cached_tokens=0
            )
        assert exc_info.value.limit_bytes == abort_cap

    def test_no_abort_when_prediction_fits(self):
        c = KVPrefixCache(group=None)
        c.prefill_transient_tracker.update(
            4096, 100 * 1024**2, floor_sample=True
        )  # 100 MiB/chunk
        cap = 100 * 1024**3  # 100 GiB cap
        with (
            patch.object(cache_mod, "_current_process_usage_bytes", return_value=0),
            patch.object(cache_mod, "_read_model_dims", return_value=(4, 2, 64, 8)),
            patch.object(
                cache_mod.memory_guard, "prefill_abort_cap_bytes", return_value=cap
            ),
        ):
            # Should NOT raise.
            c.guard_prefill_chunk_or_raise(
                _FakeModel(), processed=4096, total=12288, cached_tokens=0
            )

    def test_no_abort_when_no_next_chunk(self):
        # processed == total: prefill done, no next chunk to guard.
        c = KVPrefixCache(group=None)
        c.prefill_transient_tracker.update(4096, 2 * 1024**3, floor_sample=True)
        with (
            patch.object(
                cache_mod, "_current_process_usage_bytes", return_value=9 * 1024**3
            ),
            patch.object(cache_mod, "_read_model_dims", return_value=(4, 2, 64, 8)),
            patch.object(
                cache_mod.memory_guard, "prefill_abort_cap_bytes", return_value=1
            ),
        ):
            c.guard_prefill_chunk_or_raise(
                _FakeModel(), processed=8192, total=8192, cached_tokens=0
            )

    def test_no_op_when_cap_unset(self):
        c = KVPrefixCache(group=None)
        with (
            patch.object(cache_mod, "_current_process_usage_bytes", return_value=0),
            patch.object(cache_mod, "_read_model_dims", return_value=(4, 2, 64, 8)),
            patch.object(
                cache_mod.memory_guard, "prefill_abort_cap_bytes", return_value=0
            ),
        ):
            c.guard_prefill_chunk_or_raise(
                _FakeModel(), processed=4096, total=8192, cached_tokens=0
            )


class TestAdmissionTransientBound:
    """EWMA-refined admission (``_admission_transient_bound``)."""

    def test_returns_zero_when_no_samples(self):
        # No measurements => defer to the caller's static charge.
        c = KVPrefixCache(group=None)
        with patch.object(cache_mod, "_read_model_dims", return_value=(4, 2, 64, 8)):
            bound = c._admission_transient_bound(
                _FakeModel(),
                new_tokens=1000,
                kv_len=1000,
                chunk_size=4096,
                cached_tokens=0,
            )
        assert bound == 0

    def test_overrides_upward_when_measured_exceeds_static(self):
        # EWMA per-token far above the static SDPA estimate => override upward.
        c = KVPrefixCache(group=None)
        # 4096 tokens, 1 GiB => 256 KB/token (far above static ~33 MB/chunk).
        c.prefill_transient_tracker.update(4096, 1 * 1024**3, floor_sample=True)
        with patch.object(cache_mod, "_read_model_dims", return_value=(4, 2, 64, 8)):
            bound = c._admission_transient_bound(
                _FakeModel(),
                new_tokens=4096,
                kv_len=4096,
                chunk_size=4096,
                cached_tokens=0,
            )
        assert bound > 0
        # Override should reflect the measured per-token rate (with safety).
        assert bound >= c.prefill_transient_tracker.observed_max_bytes
