"""Tests for the reclaim-based memory ceiling (port of oMLX's memory guard).

Covers the three-component ceiling ``min(static, dynamic, metal_cap)`` where
``dynamic = phys_footprint + free + inactive + active × reclaim_ratio``. This
is the fix for the regression where a model that legitimately fills 80%+ of
memory was rejected on every prefill because the old guard used a flat
``max_recommended_working_set_size × 0.75`` ceiling that ignored both the
raised ``iogpu.wired_limit_mb`` and the process's own resident footprint.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from exo.worker.engines.mlx import memory_guard as mg

_GIB = 1024**3


@pytest.fixture(autouse=True)
def _enable_guard() -> object:  # pyright: ignore[reportUnusedFunction]
    """The guard ships disabled-by-default; these ceiling-math tests exercise
    it enabled, patching the master switch on for the module under test."""
    with patch.object(mg, "_DISABLE_PREFILL_GUARD", False):
        yield


def _vm(*, free: float, active: float, inactive: float, wired: float) -> dict[str, int]:
    return {
        "free": int(free * _GIB),
        "active": int(active * _GIB),
        "inactive": int(inactive * _GIB),
        "wired": int(wired * _GIB),
    }


class TestCeilingBreakdown:
    """The three component ceilings and the binding minimum."""

    def test_hard_limit_is_min_of_components(self):
        # static=252, dynamic=213, metal_cap=250 => hard=213 (dynamic binds).
        with (
            patch.object(mg, "get_total_memory_bytes", return_value=256 * _GIB),
            patch.object(mg, "get_effective_metal_cap_bytes", return_value=250 * _GIB),
            patch.object(
                mg,
                "get_macos_vm_stats",
                return_value=_vm(free=0.5, active=5, inactive=8, wired=201),
            ),
            patch.object(mg, "get_phys_footprint", return_value=201 * _GIB),
        ):
            b = mg.ceiling_breakdown("aggressive")
        assert b.static_ceiling == 252 * _GIB  # 256 - 4 GiB reserve
        assert b.metal_cap == 250 * _GIB
        # dynamic = phys_footprint + free + inactive + active*0.8
        expected_dynamic = (
            201 * _GIB + int(0.5 * _GIB) + int(8 * _GIB) + int(5 * _GIB * 0.8)
        )
        assert b.dynamic_ceiling == expected_dynamic
        assert b.hard_limit == b.dynamic_ceiling
        assert b.binding() == "memory currently available"

    def test_metal_cap_binds_when_below_dynamic(self):
        # iogpu.wired_limit low => metal_cap is the min.
        with (
            patch.object(mg, "get_total_memory_bytes", return_value=256 * _GIB),
            patch.object(mg, "get_effective_metal_cap_bytes", return_value=180 * _GIB),
            patch.object(
                mg,
                "get_macos_vm_stats",
                return_value=_vm(free=10, active=5, inactive=10, wired=200),
            ),
            patch.object(mg, "get_phys_footprint", return_value=200 * _GIB),
        ):
            b = mg.ceiling_breakdown("aggressive")
        assert b.hard_limit == 180 * _GIB
        assert b.binding() == "the GPU allocation cap"

    def test_static_binds_on_small_box(self):
        # 16 GiB box with a loaded model: static = 16 - 4 = 12 GiB. Make
        # dynamic larger (lots of reclaimable) so static is the min.
        with (
            patch.object(mg, "get_total_memory_bytes", return_value=16 * _GIB),
            patch.object(mg, "get_effective_metal_cap_bytes", return_value=15 * _GIB),
            patch.object(
                mg,
                "get_macos_vm_stats",
                return_value=_vm(free=4, active=4, inactive=4, wired=2),
            ),
            patch.object(mg, "get_phys_footprint", return_value=2 * _GIB),
        ):
            b = mg.ceiling_breakdown("aggressive")
        assert b.static_ceiling == 12 * _GIB
        # dynamic = 2 + 4 + 4 + 4*0.8 = 13.2 GiB > static(12) => static binds
        assert b.hard_limit == 12 * _GIB
        assert b.binding() == "installed RAM minus the reserve"


class TestReclaimRatios:
    """The tier controls how much active memory is reclaimable."""

    def _ceiling(self, tier: str) -> int:
        with (
            patch.object(mg, "get_total_memory_bytes", return_value=256 * _GIB),
            patch.object(mg, "get_effective_metal_cap_bytes", return_value=512 * _GIB),
            patch.object(
                mg,
                "get_macos_vm_stats",
                return_value=_vm(free=0, active=100, inactive=0, wired=0),
            ),
            patch.object(mg, "get_phys_footprint", return_value=0),
        ):
            return mg.ceiling_breakdown(tier).dynamic_ceiling

    def test_safe_reclaims_20_percent_active(self):
        # 100 GiB active * 0.2 = 20 GiB
        assert self._ceiling("safe") == 20 * _GIB

    def test_balanced_reclaims_50_percent_active(self):
        assert self._ceiling("balanced") == 50 * _GIB

    def test_aggressive_reclaims_80_percent_active(self):
        assert self._ceiling("aggressive") == 80 * _GIB


class TestLoadedModelRegression:
    """The bug the user hit: a 201 GiB model on a 250 GiB box must prefill."""

    def test_model_filling_80_percent_is_admitted(self):
        # 201 GiB model resident, 250 GiB wired limit, ~12 GiB reclaimable.
        # OLD guard: 250 * 0.75 = 187.5 GiB ceiling => REJECTED (model > ceiling).
        # NEW guard: dynamic = 201 + reclaimable => ADMITTED.
        with (
            patch.object(mg, "get_total_memory_bytes", return_value=256 * _GIB),
            patch.object(mg, "get_effective_metal_cap_bytes", return_value=250 * _GIB),
            patch.object(
                mg,
                "get_macos_vm_stats",
                return_value=_vm(free=0.5, active=5, inactive=8, wired=201),
            ),
            patch.object(mg, "get_phys_footprint", return_value=201 * _GIB),
            patch.object(mg, "_MEMORY_GUARD_TIER", "aggressive"),
        ):
            hard = mg.hard_limit_bytes()
            # current = phys_footprint (the loaded model). Patch the mlx
            # active-memory read so current_usage_bytes doesn't touch Metal.
            with patch.object(mg, "current_usage_bytes", return_value=201 * _GIB):
                current = mg.current_usage_bytes()
            peak = 372 * 1024**2  # from the user's error message
            assert current + peak <= hard, (
                f"current {current / _GIB:.1f} + peak {peak / _GIB:.1f} must fit under "
                f"hard {hard / _GIB:.1f} — this is the loaded-model regression"
            )

    def test_rejects_only_what_would_actually_oom(self):
        # Same box, but a huge prefill peak that would exceed even the reclaim.
        with (
            patch.object(mg, "get_total_memory_bytes", return_value=256 * _GIB),
            patch.object(mg, "get_effective_metal_cap_bytes", return_value=250 * _GIB),
            patch.object(
                mg,
                "get_macos_vm_stats",
                return_value=_vm(free=0.5, active=5, inactive=8, wired=201),
            ),
            patch.object(mg, "get_phys_footprint", return_value=201 * _GIB),
            patch.object(mg, "_MEMORY_GUARD_TIER", "aggressive"),
        ):
            hard = mg.hard_limit_bytes()
            with patch.object(mg, "current_usage_bytes", return_value=201 * _GIB):
                current = mg.current_usage_bytes()
            peak = 50 * _GIB  # 201 + 50 = 251 > 213.5 hard => rejected
            assert current + peak > hard


class TestDisableGuard:
    def test_disabled_returns_zero(self):
        with patch.object(mg, "_DISABLE_PREFILL_GUARD", True):
            b = mg.ceiling_breakdown("aggressive")
            assert b.hard_limit == 0
            assert mg.hard_limit_bytes() == 0
            assert mg.soft_limit_bytes() == 0
            assert mg.prefill_abort_cap_bytes() == 0

    def test_soft_and_abort_cap_track_hard_limit(self):
        with (
            patch.object(mg, "get_total_memory_bytes", return_value=256 * _GIB),
            patch.object(mg, "get_effective_metal_cap_bytes", return_value=512 * _GIB),
            patch.object(
                mg,
                "get_macos_vm_stats",
                return_value=_vm(free=0, active=0, inactive=0, wired=0),
            ),
            patch.object(mg, "get_phys_footprint", return_value=0),
            patch.object(mg, "_MEMORY_GUARD_TIER", "balanced"),
        ):
            hard = mg.hard_limit_bytes()
            # balanced soft = 0.90, abort = 0.90
            assert mg.soft_limit_bytes() == int(hard * 0.90)
            assert mg.prefill_abort_cap_bytes() == int(hard * 0.90)

    def test_aggressive_abort_margin_is_0_95(self):
        with (
            patch.object(mg, "get_total_memory_bytes", return_value=256 * _GIB),
            patch.object(mg, "get_effective_metal_cap_bytes", return_value=512 * _GIB),
            patch.object(
                mg,
                "get_macos_vm_stats",
                return_value=_vm(free=0, active=0, inactive=0, wired=0),
            ),
            patch.object(mg, "get_phys_footprint", return_value=0),
            patch.object(mg, "_MEMORY_GUARD_TIER", "aggressive"),
        ):
            hard = mg.hard_limit_bytes()
            assert mg.prefill_abort_cap_bytes() == int(hard * 0.95)


class TestCustomCeiling:
    def test_custom_tier_uses_absolute_override(self):
        # custom with 240 GiB ceiling => dynamic = 240, static = total - 2.
        with (
            patch.object(mg, "get_total_memory_bytes", return_value=256 * _GIB),
            patch.object(mg, "get_effective_metal_cap_bytes", return_value=512 * _GIB),
            patch.object(mg, "_MEMORY_GUARD_CUSTOM_CEILING_BYTES", 240 * _GIB),
            patch.object(mg, "_MEMORY_GUARD_TIER", "custom"),
        ):
            b = mg.ceiling_breakdown("custom")
        assert b.dynamic_ceiling == 240 * _GIB
        assert b.static_ceiling == 254 * _GIB  # 256 - 2 GiB custom reserve
        assert b.hard_limit == 240 * _GIB
