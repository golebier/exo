# ruff: noqa
"""TurboQuant KV-cache compression + tiered (Hot RAM / Cold SSD) KV cache.

This module is the runtime settings layer for two composable oMLX features
(see ``docs/omlx-porting/01-tiered-kv-cache-ssd.md`` and
``docs/exo-upstream-porting/07-turboquant-kv-cache.md``):

1. **TurboQuant** — quantized KV cache with per-model tunable bits, a
   skip-last-N-layers option (quality-sensitive layers stay full precision),
   and Qwen3-Next hybrid-cache awareness. EXO already had the blunt
   ``KV_CACHE_BITS`` global; TurboQuant is the tuned, opt-in, runtime-toggleable
   version mirroring oMLX's ``ModelSettings.turboquant_kv_*``.

2. **Tiered KV cache** — a Cold SSD tier under the existing RAM-only
   ``KVPrefixCache`` so evicted blocks spill to disk and are restored on a
   future prefix hit — even after a restart. Mirrors oMLX's
   ``CacheSettings`` (``enabled`` / ``hot_cache_only`` / ``ssd_cache_dir`` /
   ``ssd_cache_max_size`` / ``hot_cache_max_size``).

Both features follow the **same pattern as ``memory_guard``**: env-var default
captured at import time + a runtime override set via the API/UI toggle that
takes effect on the next model load / prefill without a restart. The API
surface (``/v1/feature-flags`` + ``PUT /v1/turboquant`` +
``PUT /v1/tiered-cache``) is the exact analogue of the memory-guard toggle.

NOTE: This module is the **settings/feature-flag layer only**. The actual
paged-block manager, safetensors SSD spill/restore, restart-recovery scan, and
the TurboQuant attention fast-path kernels are large ports (phased plans in the
docs above) and land separately. What ships here is the runtime-toggleable
configuration + the ``make_kv_cache`` integration point that
``cache.py`` consults, so the dashboard controls are live and the heavy
plumbing can be staged behind them without further UI churn.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from exo.shared.constants import EXO_CACHE_HOME

logger = logging.getLogger(__name__)


# ─── TurboQuant KV-cache compression ──────────────────────────────────────────
# Valid bit depths, mirroring oMLX ``ModelSettings.turboquant_kv_bits``. mlx-lm's
# ``QuantizedKVCache`` accepts integer bits; the half-step depths (2.5/3.5) are
# oMLX's vector-quantization depths that fall back to the next-lower integer
# ``QuantizedKVCache`` mode (2/3) until the native TurboQuant attention kernel
# (oMLX ``patches/turboquant_attention.py``) is ported. They are kept in the
# allowed set so dashboards/persisted settings stay forward-compatible.
_VALID_TURBOQUANT_BITS: frozenset[float] = frozenset(
    {2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8.0}
)


def _normalize_bits(value: float) -> float:
    """Snap a bit depth to the nearest supported TurboQuant depth.

    Half-step depths (2.5/3.5) are accepted as-is for settings persistence but
    round **down** to the next integer when handed to ``QuantizedKVCache`` (see
    :func:`effective_kv_bits`). 4 is oMLX's default and the safe quality floor.
    """
    v = float(value)
    # Nearest supported depth.
    return min(_VALID_TURBOQUANT_BITS, key=lambda b: abs(b - v))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# Master switch. **Defaults to disabled** so a fresh install keeps the existing
# ``KV_CACHE_BITS`` behaviour (fp16/bf16). Enable explicitly with
# ``EXO_TURBOQUANT_KV=1`` or the dashboard toggle. When disabled,
# ``make_kv_cache`` ignores the bits/skip-last settings entirely.
_TURBOQUANT_KV_ENABLED_DEFAULT = _env_bool("EXO_TURBOQUANT_KV", False)

# Bit depth and skip-last defaults — match oMLX's ``ModelSettings`` defaults
# (``turboquant_kv_bits=4``, ``turboquant_skip_last=True``). Skip-last keeps the
# final KVCache layer full precision to prevent corruption on quality-sensitive
# models; oMLX's own docstring calls this out explicitly.
_TURBOQUANT_KV_BITS_DEFAULT = _normalize_bits(
    float(os.environ.get("EXO_TURBOQUANT_KV_BITS", "4"))
)
_TURBOQUANT_SKIP_LAST_DEFAULT = _env_bool("EXO_TURBOQUANT_SKIP_LAST", True)

# Runtime overrides (set via the API/UI toggle). ``None`` means "defer to the
# env-var default above``; an explicit value takes effect on the next model
# load without a restart. Consulted by the ``is_*_enabled`` accessors so every
# ``make_kv_cache`` call honours the live toggle.
_runtime_enabled_override: bool | None = None
_runtime_bits_override: float | None = None
_runtime_skip_last_override: bool | None = None


def is_turboquant_enabled() -> bool:
    """Whether TurboQuant KV compression is active, honouring the runtime toggle."""
    if _runtime_enabled_override is not None:
        return _runtime_enabled_override
    return _TURBOQUANT_KV_ENABLED_DEFAULT


def set_turboquant_enabled(enabled: bool) -> None:
    """Runtime toggle for the API/UI (takes effect on the next model load).

    Overrides the env-var default (``EXO_TURBOQUANT_KV``). A model already
    loaded with a different cache dtype is not retroactively requantized; the
    next ``make_kv_cache`` call (next model load, or a clean prefill after the
    prefix cache is invalidated) honours the new setting.
    """
    global _runtime_enabled_override
    _runtime_enabled_override = bool(enabled)
    logger.info(
        "TurboQuant KV %s (bits=%s, skip_last=%s)",
        "enabled" if enabled else "disabled",
        turboquant_bits(),
        turboquant_skip_last(),
    )


def turboquant_bits() -> float:
    """Configured TurboQuant bit depth (2/2.5/3/3.5/4/6/8)."""
    if _runtime_bits_override is not None:
        return _runtime_bits_override
    return _TURBOQUANT_KV_BITS_DEFAULT


def set_turboquant_bits(bits: float) -> None:
    """Runtime-set the TurboQuant bit depth (validated against the allowed set)."""
    global _runtime_bits_override
    _runtime_bits_override = _normalize_bits(bits)


def turboquant_skip_last() -> bool:
    """Whether the last KVCache layer is kept full precision."""
    if _runtime_skip_last_override is not None:
        return _runtime_skip_last_override
    return _TURBOQUANT_SKIP_LAST_DEFAULT


def set_turboquant_skip_last(skip: bool) -> None:
    """Runtime-set whether the last KVCache layer skips quantization."""
    global _runtime_skip_last_override
    _runtime_skip_last_override = bool(skip)


def effective_kv_bits() -> int | None:
    """Integer bit depth to hand to mlx-lm's ``QuantizedKVCache``, or ``None``.

    Returns ``None`` when TurboQuant is disabled (so ``make_kv_cache`` builds a
    plain fp16/bf16 ``KVCache``). Half-step depths (2.5/3.5) round **down** to
    the next integer because mlx-lm's ``QuantizedKVCache`` only accepts ints;
    the half-step precision is reclaimed once the native TurboQuant attention
    kernel (oMLX ``patches/turboquant_attention.py``) is ported. Until then the
    integer fallback is the documented blunt-instrument path (EXO PR #1988).
    """
    if not is_turboquant_enabled():
        return None
    return int(turboquant_bits() // 1)


def turboquant_settings() -> dict[str, bool | float]:
    """Snapshot of the live TurboQuant settings for the feature-flag surface."""
    return {
        "enabled": is_turboquant_enabled(),
        "bits": turboquant_bits(),
        "skipLast": turboquant_skip_last(),
    }


# ─── Tiered KV cache (Hot RAM + Cold SSD + persistence) ───────────────────────
# Default SSD cache dir: under EXO's XDG cache home, mirroring oMLX's
# ``~/.omlx/cache``. Created lazily (restrictive 0700 perms — KV state can leak
# prompt content, per the tiered-cache doc's security note).
def _default_ssd_cache_dir() -> Path:
    return EXO_CACHE_HOME / "kv_ssd_cache"


def _default_ssd_cache_max_size_bytes() -> int:
    """oMLX's "auto" default: 10% of the SSD capacity backing the cache dir.

    Falls back to 8 GiB when the capacity can't be probed (non-Darwin, weird
    filesystems). 8 GiB is a conservative floor that holds a few thousand
    quantized KV blocks — enough to survive an agentic-session restart.
    """
    try:
        cache_dir = _default_ssd_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        usage = os.statvfs(str(cache_dir))
        total = usage.f_blocks * usage.f_frsize
        if total > 0:
            return int(total * 0.10)
    except OSError:
        pass
    return 8 * 1024**3


def _parse_bytes(value: str | int) -> int:
    """Parse a size string ("auto"/"8GB"/"512MB"/bytes) to bytes.

    "auto" resolves to the oMLX default (10% of SSD). Recognises the usual
    binary suffixes (KB/MB/GB/TB, case-insensitive, optional ``iB``).
    """
    if isinstance(value, int):
        return max(0, value)
    raw = value.strip().lower()
    if raw in ("", "auto"):
        return _default_ssd_cache_max_size_bytes()
    units = {
        "tb": 1024**4,
        "gb": 1024**3,
        "mb": 1024**2,
        "kb": 1024,
        "tib": 1024**4,
        "gib": 1024**3,
        "mib": 1024**2,
        "kib": 1024,
    }
    for suffix, factor in units.items():
        if raw.endswith(suffix):
            number = raw[: -len(suffix)].strip()
            try:
                return max(0, int(float(number) * factor))
            except ValueError:
                break
    try:
        return max(0, int(raw))
    except ValueError:
        return _default_ssd_cache_max_size_bytes()


# Tiered-cache master switch. **Defaults to disabled** to preserve today's
# RAM-only behaviour; the SSD tier is opt-in until the paged-SSD plumbing
# (phase 2/3 of the tiered-cache doc) is fully ported. When disabled,
# ``KVPrefixCache`` behaves exactly as before regardless of the dir/size knobs.
_TIERED_CACHE_ENABLED_DEFAULT = _env_bool("EXO_TIERED_KV_CACHE", False)

# ``hot_cache_only`` mirrors oMLX: when True the SSD tier is never written and
# the cache is cleared on restart (today's behaviour). When False and the tier
# is enabled, evicted blocks spill to SSD. Defaults to False (spill allowed)
# so the only gate is the master switch.
_TIERED_CACHE_HOT_ONLY_DEFAULT = _env_bool("EXO_TIERED_KV_CACHE_HOT_ONLY", False)

_SSD_CACHE_DIR_DEFAULT = os.environ.get(
    "EXO_SSD_CACHE_DIR", str(_default_ssd_cache_dir())
)
_SSD_CACHE_MAX_SIZE_DEFAULT = _parse_bytes(
    os.environ.get("EXO_SSD_CACHE_MAX_SIZE", "auto")
)
# Hot (RAM) cache byte budget. "0"/disabled means "no explicit cap — fall back
# to the existing reclaim-based ``_MEMORY_THRESHOLD`` eviction", matching oMLX's
# ``hot_cache_max_size="0"`` semantics.
_HOT_CACHE_MAX_SIZE_DEFAULT = _parse_bytes(
    os.environ.get("EXO_TIERED_KV_CACHE_HOT_MAX_SIZE", "0")
)

# Runtime overrides for the tiered-cache settings.
_runtime_tiered_enabled_override: bool | None = None
_runtime_tiered_hot_only_override: bool | None = None
_runtime_ssd_cache_dir_override: str | None = None
_runtime_ssd_cache_max_size_override: int | None = None
_runtime_hot_cache_max_size_override: int | None = None


def is_tiered_cache_enabled() -> bool:
    """Whether the SSD cold tier is active, honouring the runtime toggle."""
    if _runtime_tiered_enabled_override is not None:
        return _runtime_tiered_enabled_override
    return _TIERED_CACHE_ENABLED_DEFAULT


def set_tiered_cache_enabled(enabled: bool) -> None:
    """Runtime toggle for the API/UI (takes effect on the next eviction)."""
    global _runtime_tiered_enabled_override
    _runtime_tiered_enabled_override = bool(enabled)
    logger.info(
        "Tiered KV cache %s (ssd_dir=%s, ssd_max=%d, hot_only=%s)",
        "enabled" if enabled else "disabled",
        ssd_cache_dir(),
        ssd_cache_max_size_bytes(),
        hot_cache_only(),
    )


def hot_cache_only() -> bool:
    """Whether the SSD tier is bypassed (RAM-only, cleared on restart)."""
    if _runtime_tiered_hot_only_override is not None:
        return _runtime_tiered_hot_only_override
    return _TIERED_CACHE_HOT_ONLY_DEFAULT


def set_hot_cache_only(hot_only: bool) -> None:
    global _runtime_tiered_hot_only_override
    _runtime_tiered_hot_only_override = bool(hot_only)


def ssd_cache_dir() -> str:
    """Resolved SSD cache directory (created with 0700 perms on first use)."""
    if _runtime_ssd_cache_dir_override is not None:
        path = Path(_runtime_ssd_cache_dir_override).expanduser()
    else:
        path = Path(_SSD_CACHE_DIR_DEFAULT).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
        # Restrictive perms: KV state can leak prompt content (tiered-cache
        # doc, security note). best-effort on non-POSIX.
        os.chmod(path, 0o700)
    except OSError:
        pass
    return str(path)


def set_ssd_cache_dir(path: str | None) -> None:
    """Runtime-set the SSD cache dir (None restores the env/default)."""
    global _runtime_ssd_cache_dir_override
    _runtime_ssd_cache_dir_override = path if path else None


def ssd_cache_max_size_bytes() -> int:
    """Max SSD cache size in bytes (10% of SSD when "auto")."""
    if _runtime_ssd_cache_max_size_override is not None:
        return _runtime_ssd_cache_max_size_override
    return _SSD_CACHE_MAX_SIZE_DEFAULT


def set_ssd_cache_max_size(value: str | int) -> None:
    global _runtime_ssd_cache_max_size_override
    _runtime_ssd_cache_max_size_override = _parse_bytes(value)


def hot_cache_max_size_bytes() -> int:
    """Hot (RAM) cache byte budget (0 = use the existing reclaim threshold)."""
    if _runtime_hot_cache_max_size_override is not None:
        return _runtime_hot_cache_max_size_override
    return _HOT_CACHE_MAX_SIZE_DEFAULT


def set_hot_cache_max_size(value: str | int) -> None:
    global _runtime_hot_cache_max_size_override
    _runtime_hot_cache_max_size_override = _parse_bytes(value)


def tiered_cache_settings() -> dict[str, bool | str | int]:
    """Snapshot of the live tiered-cache settings for the feature-flag surface."""
    return {
        "enabled": is_tiered_cache_enabled(),
        "hotCacheOnly": hot_cache_only(),
        "ssdCacheDir": ssd_cache_dir(),
        "ssdCacheMaxSizeBytes": ssd_cache_max_size_bytes(),
        "hotCacheMaxSizeBytes": hot_cache_max_size_bytes(),
    }


@dataclass(frozen=True)
class TieredCacheStatus:
    """Observability snapshot mirroring oMLX's runtime-cache stats block.

    Hot/cold block counts and hit rates are populated once the paged-SSD
    manager is ported (phase 2/3). The SSD file count/size/disk-capacity and
    base-path fields are live now so the dashboard's observability block
    reflects the configured state and lets operators decide when to clear.
    """

    enabled: bool
    hot_cache_only: bool
    ssd_cache_dir: str
    ssd_cache_max_size_bytes: int
    hot_cache_max_size_bytes: int
    # Live usage (scanned from the SSD dir on each call).
    hot_cache_entries: int = 0
    hot_cache_size_bytes: int = 0
    ssd_cache_files: int = 0
    ssd_cache_size_bytes: int = 0
    # Backing SSD capacity (for the "used / total" gauge). 0 when unreadable.
    ssd_disk_capacity_bytes: int = 0
    # Active base path (EXO cache home) + response-state subdir, mirroring
    # oMLX's ``base_path`` / ``ssd_cache_dir`` / ``response-state directory``
    # observability rows.
    base_path: str = ""
    response_state_dir: str = ""


def _disk_capacity_bytes(path: Path) -> int:
    """Total bytes on the filesystem backing ``path`` (0 when unreadable)."""
    try:
        usage = os.statvfs(str(path))
        return int(usage.f_blocks * usage.f_frsize)
    except OSError:
        return 0


def tiered_cache_status() -> TieredCacheStatus:
    """Live status for the dashboard observability block."""
    ssd_dir = Path(ssd_cache_dir())
    ssd_files = 0
    ssd_size = 0
    if ssd_dir.exists():
        try:
            # Recursive scan — oMLX stores blocks under hash-prefix subdirs and
            # response-state in a subdir, so a top-level iterdir under-counts.
            for entry in ssd_dir.rglob("*"):
                if entry.is_file():
                    ssd_files += 1
                    ssd_size += entry.stat().st_size
        except OSError:
            pass
    return TieredCacheStatus(
        enabled=is_tiered_cache_enabled(),
        hot_cache_only=hot_cache_only(),
        ssd_cache_dir=str(ssd_dir),
        ssd_cache_max_size_bytes=ssd_cache_max_size_bytes(),
        hot_cache_max_size_bytes=hot_cache_max_size_bytes(),
        ssd_cache_files=ssd_files,
        ssd_cache_size_bytes=ssd_size,
        ssd_disk_capacity_bytes=_disk_capacity_bytes(ssd_dir),
        base_path=str(EXO_CACHE_HOME),
        response_state_dir=str(ssd_dir / "response-state"),
    )


def clear_ssd_cache() -> int:
    """Delete every file in the SSD cache dir; return the file count removed.

    Mirrors oMLX's ``clear_ssd_cache`` route: it walks the configured SSD
    cache directory and unlinks every file (including the ``response-state``
    subdir), leaving the directory structure intact for the next spill. Safe
    to call whether or not the tiered cache is enabled — a disabled tier
    simply has nothing to clear. Returns 0 when the dir doesn't exist.
    """
    ssd_dir = Path(ssd_cache_dir())
    removed = 0
    if not ssd_dir.exists():
        return 0
    try:
        for entry in ssd_dir.rglob("*"):
            if entry.is_file() or entry.is_symlink():
                try:
                    entry.unlink()
                    removed += 1
                except OSError as exc:
                    logger.warning("Failed to remove SSD cache file %s: %s", entry, exc)
    except OSError as exc:
        logger.warning("Failed to walk SSD cache dir %s: %s", ssd_dir, exc)
    logger.info("Cleared %d SSD cache file(s) from %s", removed, ssd_dir)
    return removed


TieredCacheMode = Literal["auto", "ssd_sidecar", "embedded"]

# Public alias for the size-string parser (tested + used by the dashboard's
# size-input round-trip). The leading-underscore name is kept for internal
# call sites; this is the supported external name.
parse_size = _parse_bytes


def reset_runtime_overrides() -> None:
    """Clear every runtime override so the env-var defaults take over again.

    Intended for tests and a future "reset to defaults" dashboard action.
    Restores the exact post-import state: TurboQuant + tiered cache disabled,
    bits=4, skip_last=True, SSD dir/size from env, hot-cache cap 0.
    """
    global \
        _runtime_enabled_override, \
        _runtime_bits_override, \
        _runtime_skip_last_override
    global _runtime_tiered_enabled_override, _runtime_tiered_hot_only_override
    global _runtime_ssd_cache_dir_override, _runtime_ssd_cache_max_size_override
    global _runtime_hot_cache_max_size_override
    _runtime_enabled_override = None
    _runtime_bits_override = None
    _runtime_skip_last_override = None
    _runtime_tiered_enabled_override = None
    _runtime_tiered_hot_only_override = None
    _runtime_ssd_cache_dir_override = None
    _runtime_ssd_cache_max_size_override = None
    _runtime_hot_cache_max_size_override = None
