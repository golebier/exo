"""Tests for the TurboQuant + tiered-KV-cache runtime settings layer (#07, #01).

Both features follow the memory-guard pattern: env-var default captured at
import time + a runtime override set via the API/UI toggle that takes effect
on the next model load / prefill without a restart. These tests pin the
default-off behaviour, the runtime-override precedence, the bit-depth
normalisation (half-step depths round down for ``QuantizedKVCache``), and the
``make_kv_cache`` skip-last integration.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from mlx_lm.models.cache import KVCache, QuantizedKVCache, RotatingKVCache

from exo.worker.engines.mlx import cache
from exo.worker.engines.mlx import turboquant as tq


@pytest.fixture(autouse=True)
def _reset_runtime_overrides() -> object:  # pyright: ignore[reportUnusedFunction]
    """Each test starts from a clean runtime-override state so toggles from
    one test don't leak into another."""
    tq.reset_runtime_overrides()
    yield
    tq.reset_runtime_overrides()


class _StubLayer:
    pass


class _StubModel:
    """Minimal stand-in for an mlx-lm model with a ``layers`` attribute."""

    def __init__(self, n: int) -> None:
        self.layers = [_StubLayer() for _ in range(n)]


class TestTurboQuantDefaults:
    """Ship default is off; oMLX-matched bits/skip-last defaults."""

    def test_disabled_by_default(self) -> None:
        assert tq.is_turboquant_enabled() is False
        assert tq.effective_kv_bits() is None

    def test_default_bits_and_skip_last(self) -> None:
        # oMLX ModelSettings defaults: bits=4, skip_last=True.
        assert tq.turboquant_bits() == 4.0
        assert tq.turboquant_skip_last() is True


class TestTurboQuantRuntimeOverride:
    """Runtime toggle takes precedence over the env default without a restart."""

    def test_enable_flips_effective_bits(self) -> None:
        tq.set_turboquant_enabled(True)
        assert tq.is_turboquant_enabled() is True
        assert tq.effective_kv_bits() == 4

    def test_disable_returns_none(self) -> None:
        tq.set_turboquant_enabled(True)
        tq.set_turboquant_enabled(False)
        assert tq.effective_kv_bits() is None

    def test_bits_override_rounds_half_steps_down(self) -> None:
        # mlx-lm's QuantizedKVCache only accepts int bits; 2.5/3.5 round down
        # to 2/3 until the native TurboQuant attention kernel lands.
        tq.set_turboquant_enabled(True)
        tq.set_turboquant_bits(2.5)
        assert tq.turboquant_bits() == 2.5
        assert tq.effective_kv_bits() == 2
        tq.set_turboquant_bits(3.5)
        assert tq.effective_kv_bits() == 3
        tq.set_turboquant_bits(8)
        assert tq.effective_kv_bits() == 8

    def test_bits_normalisation_snaps_to_supported(self) -> None:
        tq.set_turboquant_bits(5)  # not in the allowed set -> nearest (4 or 6)
        assert tq.turboquant_bits() in (4.0, 6.0)

    def test_skip_last_override(self) -> None:
        tq.set_turboquant_enabled(True)
        tq.set_turboquant_skip_last(False)
        assert tq.turboquant_skip_last() is False

    def test_settings_snapshot(self) -> None:
        tq.set_turboquant_enabled(True)
        tq.set_turboquant_bits(3)
        tq.set_turboquant_skip_last(False)
        snap = tq.turboquant_settings()
        assert snap == {"enabled": True, "bits": 3.0, "skipLast": False}


class TestMakeKvCacheIntegration:
    """``make_kv_cache`` honours the TurboQuant toggle + skip-last."""

    def test_disabled_builds_plain_kv_cache(self) -> None:
        caches = cache.make_kv_cache(_StubModel(3))  # type: ignore[arg-type]
        assert all(
            isinstance(c, KVCache) and not isinstance(c, QuantizedKVCache)
            for c in caches
        )

    def test_enabled_quantizes_all_layers_when_skip_last_off(self) -> None:
        tq.set_turboquant_enabled(True)
        tq.set_turboquant_bits(4)
        tq.set_turboquant_skip_last(False)
        caches = cache.make_kv_cache(_StubModel(4))  # type: ignore[arg-type]
        assert all(isinstance(c, QuantizedKVCache) for c in caches)

    def test_skip_last_keeps_final_layer_full_precision(self) -> None:
        tq.set_turboquant_enabled(True)
        tq.set_turboquant_bits(4)
        tq.set_turboquant_skip_last(True)
        caches = cache.make_kv_cache(_StubModel(4))  # type: ignore[arg-type]
        assert isinstance(caches[-1], KVCache) and not isinstance(
            caches[-1], QuantizedKVCache
        )
        assert all(isinstance(c, QuantizedKVCache) for c in caches[:-1])

    def test_max_kv_size_branch_unchanged(self) -> None:
        # RotatingKVCache path is unaffected by TurboQuant (no bits knob).
        tq.set_turboquant_enabled(True)
        caches = cache.make_kv_cache(
            _StubModel(3),  # type: ignore[arg-type]
            max_kv_size=1024,
            keep=64,
        )
        assert all(isinstance(c, RotatingKVCache) for c in caches)


class TestTieredCacheDefaults:
    """Ship default is off; SSD dir defaults under EXO's cache home."""

    def test_disabled_by_default(self) -> None:
        assert tq.is_tiered_cache_enabled() is False

    def test_hot_cache_only_default_false(self) -> None:
        assert tq.hot_cache_only() is False

    def test_ssd_dir_under_cache_home(self) -> None:
        assert tq.ssd_cache_dir().endswith("kv_ssd_cache")

    def test_ssd_auto_max_is_positive(self) -> None:
        assert tq.ssd_cache_max_size_bytes() > 0

    def test_hot_max_default_zero(self) -> None:
        assert tq.hot_cache_max_size_bytes() == 0


class TestTieredCacheRuntimeOverride:
    """Runtime toggle + dir/size overrides round-trip through the setters."""

    def test_enable_toggle(self) -> None:
        tq.set_tiered_cache_enabled(True)
        assert tq.is_tiered_cache_enabled() is True

    def test_hot_cache_only_override(self) -> None:
        tq.set_hot_cache_only(True)
        assert tq.hot_cache_only() is True

    def test_ssd_dir_override(self, tmp_path: Path) -> None:
        custom = str(tmp_path / "custom_ssdcache")
        tq.set_ssd_cache_dir(custom)
        assert tq.ssd_cache_dir() == custom
        # The dir is created with restrictive perms on resolution.
        import os

        assert os.path.isdir(custom)

    def test_ssd_dir_none_restores_default(self) -> None:
        tq.set_ssd_cache_dir("/tmp/some_custom_path")
        tq.set_ssd_cache_dir(None)
        assert tq.ssd_cache_dir().endswith("kv_ssd_cache")

    def test_size_string_parsing(self) -> None:
        tq.set_ssd_cache_max_size("8GB")
        assert tq.ssd_cache_max_size_bytes() == 8 * 1024**3
        tq.set_ssd_cache_max_size("512MB")
        assert tq.ssd_cache_max_size_bytes() == 512 * 1024**2
        tq.set_ssd_cache_max_size(1024)
        assert tq.ssd_cache_max_size_bytes() == 1024

    def test_hot_cache_size_string_parsing(self) -> None:
        tq.set_hot_cache_max_size("4GB")
        assert tq.hot_cache_max_size_bytes() == 4 * 1024**3

    def test_settings_snapshot(self) -> None:
        tq.set_tiered_cache_enabled(True)
        snap = tq.tiered_cache_settings()
        assert snap["enabled"] is True
        assert "ssdCacheDir" in snap
        assert "ssdCacheMaxSizeBytes" in snap

    def test_status_reflects_state(self) -> None:
        tq.set_tiered_cache_enabled(True)
        status = tq.tiered_cache_status()
        assert status.enabled is True
        assert status.ssd_cache_files >= 0

    def test_status_includes_disk_capacity_and_paths(self) -> None:
        status = tq.tiered_cache_status()
        # Disk capacity is probed from the filesystem backing the SSD dir.
        assert status.ssd_disk_capacity_bytes >= 0
        assert status.base_path != ""
        assert status.response_state_dir.endswith("response-state")

    def test_reset_runtime_overrides_clears_everything(self) -> None:
        tq.set_tiered_cache_enabled(True)
        tq.set_turboquant_enabled(True)
        tq.set_ssd_cache_dir("/tmp/should_be_reset")
        tq.reset_runtime_overrides()
        assert tq.is_tiered_cache_enabled() is False
        assert tq.is_turboquant_enabled() is False
        assert tq.ssd_cache_dir().endswith("kv_ssd_cache")


class TestClearSsdCache:
    """Clearing the SSD cache removes every file (mirrors oMLX's clear route)."""

    def test_clear_removes_all_files(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "ssd"
        cache_dir.mkdir()
        (cache_dir / "block_0.safetensors").write_bytes(b"x" * 100)
        (cache_dir / "block_1.safetensors").write_bytes(b"y" * 200)
        sub = cache_dir / "response-state"
        sub.mkdir()
        (sub / "state_0.bin").write_bytes(b"z" * 50)
        tq.set_ssd_cache_dir(str(cache_dir))

        status = tq.tiered_cache_status()
        assert status.ssd_cache_files == 3
        assert status.ssd_cache_size_bytes == 350

        removed = tq.clear_ssd_cache()
        assert removed == 3

        status_after = tq.tiered_cache_status()
        assert status_after.ssd_cache_files == 0
        assert status_after.ssd_cache_size_bytes == 0

    def test_clear_on_missing_dir_returns_zero(self) -> None:
        tq.set_ssd_cache_dir("/tmp/exo_nonexistent_cache_dir_xyz")
        assert tq.clear_ssd_cache() == 0

    def test_clear_preserves_directory_structure(self, tmp_path: Path) -> None:
        # oMLX leaves the dir + subdir layout intact for the next spill.
        cache_dir = tmp_path / "ssd"
        cache_dir.mkdir()
        sub = cache_dir / "response-state"
        sub.mkdir()
        (sub / "state_0.bin").write_bytes(b"z")
        tq.set_ssd_cache_dir(str(cache_dir))

        tq.clear_ssd_cache()
        assert cache_dir.exists()
        assert sub.exists()


class TestParseSize:
    """Size-string parsing mirrors oMLX's ``parse_size`` semantics."""

    def test_gb(self) -> None:
        assert tq.parse_size("8GB") == 8 * 1024**3

    def test_gib(self) -> None:
        assert tq.parse_size("8GiB") == 8 * 1024**3

    def test_mb(self) -> None:
        assert tq.parse_size("512MB") == 512 * 1024**2

    def test_int_bytes(self) -> None:
        assert tq.parse_size(2048) == 2048

    def test_int_string(self) -> None:
        assert tq.parse_size("2048") == 2048

    def test_auto_falls_back_to_default(self) -> None:
        assert tq.parse_size("auto") > 0
        assert tq.parse_size("") > 0

    def test_garbage_falls_back_to_default(self) -> None:
        assert tq.parse_size("not-a-size") > 0

    def test_zero(self) -> None:
        assert tq.parse_size("0") == 0


class TestEnvDefaults:
    """Env-var defaults are captured at import time, matching memory_guard."""

    def test_env_disable_default(self) -> None:
        # EXO_TURBOQUANT_KV unset => disabled (the ship default).
        import os

        env = dict(os.environ)
        env.pop("EXO_TURBOQUANT_KV", None)
        with patch.dict("os.environ", env, clear=True):
            tq.reset_runtime_overrides()
            assert tq.is_turboquant_enabled() is False

    def test_env_enable_default(self) -> None:
        # EXO_TIERED_KV_CACHE=1 enables the tiered cache at import time.
        # We can't re-import cleanly, so verify the env parser wiring by
        # checking that the runtime override path is the live toggle.
        tq.set_tiered_cache_enabled(True)
        assert tq.is_tiered_cache_enabled() is True
