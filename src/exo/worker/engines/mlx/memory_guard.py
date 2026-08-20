# ruff: noqa
"""Reclaim-based memory ceiling for prefill admission (port of oMLX's guard).

oMLX derives the hard ceiling a rank admits against from three constraints and
takes their minimum::

    hard_limit = min(static_ceiling, dynamic_ceiling, metal_cap)

  static_ceiling  = total_ram - tier.static_reserve
  dynamic_ceiling = phys_footprint + free + inactive + active * tier.reclaim_ratio
  metal_cap       = iogpu.wired_limit_mb (if set) else max_recommended_working_set_size

``dynamic_ceiling`` is the key term a naive ``max_recommended * fraction``
guard misses: it adds the process's own ``phys_footprint`` (the loaded model,
which macOS jetsam actually compares against) to the host's reclaimable memory
(``free + inactive + active × reclaim_ratio``). A 201 GiB model on a 250 GiB
box therefore sees a ceiling of ~201 + reclaimable, not ~187 — so a workload
that legitimately fills 80% of memory is no longer rejected on every prefill.

The ``reclaim_ratio`` (0.2 / 0.5 / 0.8 for safe / balanced / aggressive) is the
fraction of *active* pages macOS can compress or swap under pressure. Tiers
also set the static reserve (8 / 6 / 4 GiB) and the soft / abort watermarks.

This module is the single source of truth for the ceiling so admission
(``preflight_or_raise``) and the per-chunk abort (``guard_prefill_chunk_or_raise``)
agree on one number — the same invariant oMLX's ``ProcessMemoryEnforcer``
enforces. EXO has no cluster admission layer or engine-pool eviction, so only
the ceiling + measurement primitives are ported, not the watchdog/enforcer.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import contextlib
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import cast

import psutil

logger = logging.getLogger(__name__)

# ─── vm_statistics64 (host_statistics64) ──────────────────────────────────────
_HOST_VM_INFO64 = 4
_HOST_INFO64_MAX_COUNT = 256
_VM_STATS_MIN_COUNT = 4
_VM_PAGE_SIZE_DEFAULT = 16384

# Mutable module state (lowercase so basedpyright doesn't treat them as
# redefinable constants). ``_libc`` is the typed libc handle; ``_mach_host``
# is the mach host port. Both ``None``/0 on non-Darwin or libc failure.
_libc: ctypes.CDLL | None = None
_mach_host: int = 0
_vm_page_size: int = _VM_PAGE_SIZE_DEFAULT

if sys.platform == "darwin":
    try:
        _libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.dylib")
        _libc.mach_host_self.argtypes = []
        _libc.mach_host_self.restype = ctypes.c_uint
        _libc.host_page_size.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
        _libc.host_page_size.restype = ctypes.c_int
        _libc.host_statistics64.argtypes = [
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
        ]
        _libc.host_statistics64.restype = ctypes.c_int
        _mach_host = int(_libc.mach_host_self())  # type: ignore[arg-type]
        _ps = ctypes.c_uint(0)
        if _libc.host_page_size(_mach_host, ctypes.byref(_ps)) == 0 and _ps.value > 0:
            _vm_page_size = int(_ps.value)
    except Exception:  # pragma: no cover - non-Darwin or libc unavailable
        _libc = None
        _mach_host = 0


def get_macos_vm_stats() -> dict[str, int] | None:
    """Snapshot of mach ``vm_statistics64`` page counters in bytes.

    Returns ``None`` on non-macOS or when the host call fails. The first four
    counters (free / active / inactive / wired) are stable across SDK versions;
    speculative and compressed sit further into the struct and are optional.
    Ported from oMLX's ``psutil_compat.get_macos_vm_stats``.
    """
    if _libc is None or _mach_host == 0:
        return None
    try:
        stats = (ctypes.c_int * _HOST_INFO64_MAX_COUNT)()
        count = ctypes.c_uint(_HOST_INFO64_MAX_COUNT)
        # ctypes calls are typed via argtypes/restype above; the array indexing
        # returns ``int`` (c_int __getitem__). The whole block is wrapped so a
        # host-call failure degrades to ``None``.
        rc: int = _libc.host_statistics64(  # type: ignore[assignment,arg-type]
            _mach_host, _HOST_VM_INFO64, stats, ctypes.byref(count)
        )
        if rc != 0 or count.value < _VM_STATS_MIN_COUNT:
            return None
        ps = _vm_page_size
        return {
            "free": int(cast(object, stats[0])) * ps,  # type: ignore[arg-type]
            "active": int(cast(object, stats[1])) * ps,  # type: ignore[arg-type]
            "inactive": int(cast(object, stats[2])) * ps,  # type: ignore[arg-type]
            "wired": int(cast(object, stats[3])) * ps,  # type: ignore[arg-type]
        }
    except Exception:  # pragma: no cover
        return None


# ─── phys_footprint (proc_pid_rusage) ─────────────────────────────────────────
class _RusageInfoV4(ctypes.Structure):
    """``rusage_info_v4`` layout — see ``<sys/resource.h>``.

    ``ri_phys_footprint`` is the field jetsam compares against; it includes
    anonymous, dirty file-backed, and IOAccelerator (Metal) allocations. The
    **full** struct must be declared: ``proc_pid_rusage(RUSAGE_INFO_V4)``
    writes all 36 fields (296 bytes) regardless of which one the caller reads,
    so a truncated struct would have the kernel overflow the buffer and corrupt
    the heap (SIGBUS). Ported field-for-field from oMLX's
    ``utils.proc_memory._RusageInfoV4``.
    """

    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
        ("ri_cpu_time_qos_default", ctypes.c_uint64),
        ("ri_cpu_time_qos_maintenance", ctypes.c_uint64),
        ("ri_cpu_time_qos_utility", ctypes.c_uint64),
        ("ri_cpu_time_qos_background", ctypes.c_uint64),
        ("ri_cpu_time_qos_legacy", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_initiated", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_interactive", ctypes.c_uint64),
        ("ri_billed_system_time", ctypes.c_uint64),
        ("ri_serviced_system_time", ctypes.c_uint64),
        ("ri_logical_writes", ctypes.c_uint64),
        ("ri_lifetime_max_phys_footprint", ctypes.c_uint64),
        ("ri_instructions", ctypes.c_uint64),
        ("ri_cycles", ctypes.c_uint64),
        ("ri_billed_energy", ctypes.c_uint64),
        ("ri_serviced_energy", ctypes.c_uint64),
        ("ri_interval_max_phys_footprint", ctypes.c_uint64),
        ("ri_runnable_time", ctypes.c_uint64),
    ]


_RUSAGE_INFO_V4 = 4
# ``proc_pid_rusage`` returns ``Any`` via ctypes; we isolate it behind this
# single typed wrapper so the ``Any`` doesn't propagate (strict ``reportAny``).
_proc_pid_rusage: object | None = None

if sys.platform == "darwin":
    try:
        _libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        _libproc.proc_pid_rusage.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        _libproc.proc_pid_rusage.restype = ctypes.c_int
        _proc_pid_rusage = _libproc.proc_pid_rusage
    except OSError:  # pragma: no cover
        _proc_pid_rusage = None


def get_phys_footprint(pid: int | None = None) -> int:
    """Process ``phys_footprint`` in bytes (the jetsam ledger).

    Returns 0 on non-Darwin or libproc failure so callers can safely use
    ``max(active, phys_footprint)``. Ported from oMLX's
    ``utils.proc_memory.get_phys_footprint``.
    """
    if _proc_pid_rusage is None:
        return 0
    info = _RusageInfoV4()
    target_pid = pid if pid is not None else os.getpid()
    rc = int(_proc_pid_rusage(target_pid, _RUSAGE_INFO_V4, ctypes.byref(info)))  # type: ignore[call-arg,arg-type]
    if rc != 0:
        return 0
    return int(info.ri_phys_footprint)  # type: ignore[arg-type]


# ─── Metal cap (iogpu.wired_limit_mb) ─────────────────────────────────────────
def get_iogpu_wired_limit_bytes() -> int:
    """Read the kernel's ``iogpu.wired_limit_mb`` sysctl in bytes.

    Returns 0 when unset (``0`` means "system default") or unreadable. Ported
    from oMLX's ``get_iogpu_wired_limit_bytes``. Operators of large Apple
    Silicon boxes raise this via ``sudo sysctl iogpu.wired_limit_mb=<mib>`` to
    let MLX wire most of RAM; Metal's ``max_recommended_working_set_size`` does
    not reflect the override, so a ceiling that ignores it rejects workloads
    that fit under the raised wired limit.
    """
    if sys.platform != "darwin":
        return 0
    with contextlib.suppress(Exception):
        out = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "iogpu.wired_limit_mb"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if out.returncode == 0:
            mib = int(out.stdout.strip())
            if mib > 0:
                return mib * (1024**2)
    return 0


def _max_metal_working_set_bytes() -> int:
    """Apple's default Metal cap (~75% of RAM) from MLX. 0 when unavailable."""
    try:
        import mlx.core as mx

        info = mx.device_info()
        reported = cast(object, info.get("max_recommended_working_set_size", 0))
        return max(0, int(reported))  # type: ignore[arg-type]
    except Exception:  # pragma: no cover - MLX absent or too old
        return 0


def get_effective_metal_cap_bytes() -> int:
    """Effective per-process Metal allocation cap.

    ``iogpu.wired_limit_mb`` when explicitly set (> 0); otherwise Apple's
    ``max_recommended_working_set_size``. This is the value above which Metal
    rejects allocations, so the ceiling's ``metal_cap`` term keeps the guard
    from planning above what Metal will actually accept. Ported from oMLX's
    ``get_effective_metal_cap_bytes``.
    """
    sysctl_cap = get_iogpu_wired_limit_bytes()
    if sysctl_cap > 0:
        return sysctl_cap
    return _max_metal_working_set_bytes()


def get_total_memory_bytes() -> int:
    """Total physical RAM in bytes."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = int(pages) * int(page_size)
        if total > 0:
            return total
    except (AttributeError, ValueError, OSError):
        pass
    return int(psutil.virtual_memory().total)


# ─── Tiers ────────────────────────────────────────────────────────────────────
# Static reserve for systems at/above the small-system threshold (GiB). ``custom``
# shares a small reserve so the static cap stays sane regardless of the custom
# ceiling. Ported from oMLX's ``_STATIC_RESERVE_LARGE``.
_SMALL_SYSTEM_RESERVE = 4 * 1024**3
_SMALL_SYSTEM_THRESHOLD = 24 * 1024**3
_STATIC_RESERVE_LARGE: dict[str, int] = {
    "safe": 8 * 1024**3,
    "balanced": 6 * 1024**3,
    "aggressive": 4 * 1024**3,
    "custom": 2 * 1024**3,
}

# Fraction of "active" pages counted as reclaimable via macOS compression/swap.
# macOS's compressor averages ~2-3x so ~60-67% is realistically reclaimable;
# 0.8 pushes into swap territory. Ported from ``_ACTIVE_RECLAIM_RATIO``.
_ACTIVE_RECLAIM_RATIO: dict[str, float] = {
    "safe": 0.2,
    "balanced": 0.5,
    "aggressive": 0.8,
    "custom": 0.8,
}

# Soft watermark (fraction of hard_limit) and per-chunk abort margin per tier.
_SOFT_THRESHOLD_BY_TIER: dict[str, float] = {
    "safe": 0.85,
    "balanced": 0.90,
    "aggressive": 0.925,
    "custom": 0.85,
}
_PREFILL_ABORT_MARGIN_BY_TIER: dict[str, float] = {
    "safe": 0.90,
    "balanced": 0.90,
    "aggressive": 0.95,
    "custom": 0.95,
}

_VALID_TIERS = frozenset(_STATIC_RESERVE_LARGE)


def _normalize_tier(value: str) -> str:
    tier = (value or "").strip().lower()
    return tier if tier in _VALID_TIERS else "balanced"


# Operator-selected tier. ``balanced`` matches oMLX's default; operators of a
# dedicated large-Apple-Silicon inference box (where macOS compression/swap of
# active pages is acceptable) set ``aggressive`` to reclaim up to 80% of active.
_MEMORY_GUARD_TIER = _normalize_tier(
    os.environ.get("EXO_MEMORY_GUARD_TIER", "balanced")
)

# Custom absolute ceiling (bytes) for the ``custom`` tier, overriding the
# reclaim-based dynamic ceiling. 0 disables it (use the reclaim formula).
_MEMORY_GUARD_CUSTOM_CEILING_BYTES = int(
    os.environ.get("EXO_MEMORY_GUARD_CUSTOM_CEILING_BYTES", "0")
)

# Master switch for the prefill memory guard. **Defaults to disabled** so a
# fresh install behaves exactly as before task #11 (no preflight rejection,
# no per-chunk abort) — the reclaim-based ceiling is opt-in until validated on
# the target cluster. Enable explicitly with ``EXO_ENABLE_PREFILL_GUARD=1``
# (or the legacy ``EXO_DISABLE_PREFILL_GUARD=0``). When disabled, the preflight
# admission check and the per-chunk abort are skipped entirely.
_DISABLE_PREFILL_GUARD = os.environ.get("EXO_DISABLE_PREFILL_GUARD", "1") in (
    "1",
    "true",
    "TRUE",
    "yes",
) and os.environ.get("EXO_ENABLE_PREFILL_GUARD", "") not in ("1", "true", "TRUE", "yes")

# Runtime override for the guard enable state, set via the API/UI toggle.
# ``None`` means "defer to the env-var default above``; ``True``/``False`` is an
# explicit runtime flip that takes effect on the next prefill without a restart.
# Consulted by ``_is_guard_enabled`` so every ceiling/abort call honours it.
_runtime_enabled_override: bool | None = None


def _is_guard_enabled() -> bool:
    """Whether the prefill guard is active, honouring the runtime toggle."""
    if _runtime_enabled_override is not None:
        return _runtime_enabled_override
    return not _DISABLE_PREFILL_GUARD


def is_guard_enabled() -> bool:
    """Public accessor for the API/feature-flag surface."""
    return _is_guard_enabled()


def set_guard_enabled(enabled: bool) -> None:
    """Runtime toggle for the API/UI (takes effect on the next prefill).

    Overrides the env-var default (``EXO_ENABLE_PREFILL_GUARD``). The master
    env escape hatch ``EXO_DISABLE_PREFILL_GUARD=1`` still wins when the
    override is cleared, so a process started with the guard forced off cannot
    be re-enabled via the UI.
    """
    global _runtime_enabled_override
    _runtime_enabled_override = bool(enabled)


@dataclass(frozen=True)
class CeilingBreakdown:
    """The three component ceilings and the binding ``hard_limit``."""

    static_ceiling: int
    dynamic_ceiling: int
    metal_cap: int
    hard_limit: int

    def binding(self) -> str:
        """Which constraint is actually stopping us, for error messages."""
        hard = self.hard_limit
        for value, label in (
            (self.dynamic_ceiling, "memory currently available"),
            (self.metal_cap, "the GPU allocation cap"),
            (self.static_ceiling, "installed RAM minus the reserve"),
        ):
            if value and value == hard:
                return label
        return "the admission ceiling"


def _static_ceiling_bytes(tier: str) -> int:
    system_bytes = get_total_memory_bytes()
    if system_bytes <= 0:
        return 0
    if tier == "custom":
        return max(0, system_bytes - _STATIC_RESERVE_LARGE["custom"])
    if system_bytes < _SMALL_SYSTEM_THRESHOLD:
        reserve = _SMALL_SYSTEM_RESERVE
    else:
        reserve = _STATIC_RESERVE_LARGE[tier]
    return max(0, system_bytes - reserve)


def _dynamic_ceiling_bytes(tier: str) -> int:
    """Tier-aware reclaimable-memory ceiling.

    ``phys_footprint + free + inactive + active × reclaim_ratio``. The
    ``phys_footprint`` term is the process's own resident memory (the loaded
    model) — this is what makes a 201 GiB model on a 250 GiB box admissible:
    the ceiling grows with what the process already legitimately holds.
    Falls back to ``phys_footprint + psutil.available`` when vm_stat is
    unreadable, then to the static ceiling.
    """
    if tier == "custom" and _MEMORY_GUARD_CUSTOM_CEILING_BYTES > 0:
        return max(0, _MEMORY_GUARD_CUSTOM_CEILING_BYTES)
    omlx_usage = get_phys_footprint()
    stats = get_macos_vm_stats()
    if stats is None:
        try:
            available = int(psutil.virtual_memory().available)
        except Exception:  # pragma: no cover
            return _static_ceiling_bytes(tier)
        return max(0, omlx_usage + available)
    ratio = _ACTIVE_RECLAIM_RATIO[tier]
    reclaimable = stats["free"] + stats["inactive"] + int(stats["active"] * ratio)
    return max(0, omlx_usage + reclaimable)


def ceiling_breakdown(tier: str = _MEMORY_GUARD_TIER) -> CeilingBreakdown:
    """Compute the hard ceiling and its three components.

    ``hard_limit = min(static, dynamic, metal_cap)`` (non-zero values only).
    Returns all-zero when the guard is disabled — callers treat 0 as "no
    limit". Ported from oMLX's ``ProcessMemoryEnforcer._get_ceiling_breakdown``.
    """
    if not _is_guard_enabled():
        return CeilingBreakdown(0, 0, 0, 0)
    resolved = _normalize_tier(tier)
    static_ceiling = _static_ceiling_bytes(resolved)
    dynamic_ceiling = _dynamic_ceiling_bytes(resolved)
    metal_cap = get_effective_metal_cap_bytes()
    candidates = [c for c in (static_ceiling, dynamic_ceiling, metal_cap) if c > 0]
    hard_limit = min(candidates) if candidates else 0
    return CeilingBreakdown(
        static_ceiling=static_ceiling,
        dynamic_ceiling=dynamic_ceiling,
        metal_cap=metal_cap,
        hard_limit=hard_limit,
    )


def hard_limit_bytes() -> int:
    """The hard OOM ceiling admission rejects above (0 = guard disabled)."""
    return ceiling_breakdown().hard_limit


def soft_limit_bytes() -> int:
    """Soft watermark — the cache is evicted below this before prefill.

    ``hard_limit × _SOFT_THRESHOLD_BY_TIER``. This is the "evict before
    prefill" headroom target (PR #2251): the cache is trimmed to the soft
    watermark, then admission checks against the hard limit. Defaults sit a
    few points below the hard limit so a loaded model whose weights alone are
    above the soft watermark simply can't be evicted to it (the early-bail in
    ``_evict_until_under`` prevents thrash); admission then decides.
    """
    hard = hard_limit_bytes()
    if hard <= 0:
        return 0
    return int(hard * _SOFT_THRESHOLD_BY_TIER[_MEMORY_GUARD_TIER])


def prefill_abort_cap_bytes() -> int:
    """Per-chunk abort cap — ``hard_limit × abort_margin`` (0 = disabled)."""
    hard = hard_limit_bytes()
    if hard <= 0:
        return 0
    return int(hard * _PREFILL_ABORT_MARGIN_BY_TIER[_MEMORY_GUARD_TIER])


def current_usage_bytes() -> int:
    """What this process is holding right now, by the reckoning jetsam uses.

    The larger of ``phys_footprint`` (the ledger jetsam compares against,
    which counts MLX's Metal allocations) and MLX's active Metal memory (the
    allocator's view, which leads the kernel ledger mid-prefill). Neither
    alone is the whole truth. Ported from oMLX's ``current_usage_bytes``.
    """
    phys = get_phys_footprint()
    active = 0
    with contextlib.suppress(Exception):
        import mlx.core as mx

        active = int(cast(object, mx.get_active_memory()))  # type: ignore[arg-type]
    return max(phys, active)
