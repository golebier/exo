# type: ignore
# ruff: noqa
# SPDX-License-Identifier: Apache-2.0
"""Fast-kernel dispatch for the GLM MoE DSA model.

This module provides a dispatch layer that uses native GLM-5.2 kernels when
available (from oMLX's custom_kernels) and falls back to mlx.fast otherwise.
EXO does not ship the native kernels, so the fallback path is used. The
fallback means the sparse MLA and exact block attention paths return None,
which causes the model to fall through to the standard scaled_dot_product
attention path with the sparse top-k mask applied.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


def _detach_import_error(exc: Exception) -> Exception:
    """Keep the diagnostic message without retaining import caller frames."""
    exc.__traceback__ = None
    exc.__cause__ = None
    exc.__context__ = None
    return exc


# EXO vendors oMLX's native GLM kernels under its own namespace
# (exo.worker.engines.mlx.vendor.omlx_custom_kernels.glm_moe_dsa) and builds
# the nanobind ``_ext`` extension only when EXO_BUILD_MLX_KERNELS=1 is set
# (Darwin + Metal toolchain required). When the extension is absent the
# vendored ``fast`` module still imports, but ``is_native_available()``
# returns False and ``has_symbol()`` reports only the ``mx.fast`` symbols —
# so the GLM-specific native kernels (sparse MLA, exact block attention, etc.)
# report unavailable and the model code falls through to the standard
# attention path. An upstream ``omlx`` install is also accepted as a fallback
# so users who pip-installed oMLX with custom kernels get them transparently.
#
# The native ``_ext`` import is DEFERRED (lazy) and DEFAULT-OFF (opt-in via
# ``EXO_NATIVE_GLM_KERNELS=1``). ``_ext.so`` links ``Metal.framework`` +
# ``libmlx.dylib`` and importing it eagerly initialises the Metal device —
# which MUST NOT happen before ``mx.distributed.init(backend="jaccl")`` in the
# runner. The jaccl GPU-RDMA backend needs to own the Metal device's first
# initialisation; an earlier eager import (issue: JACCL warmup all_sum hangs
# on rank 1, plus Metal command-buffer OOM during prefill) corrupts that
# setup. The first kernel lookup happens during model load/inference, well
# after distributed init, so resolving lazily on first access is safe.
# Native kernels are disabled by default because they change Metal
# command-buffer timing and, under TP, can perturb the collective ordering
# that the sync-eval fix depends on.
_native_fast: Any = None
_native_import_error: Exception | None = None
_native_resolved: bool = False


def _resolve_native_fast() -> Any:
    """Lazily import the native GLM ``fast`` module on first access.

    Returns the native ``fast`` module, or ``None`` when unavailable/disabled.
    Resolution runs at most once; the result is cached module-level.
    """
    global _native_fast, _native_import_error, _native_resolved
    if _native_resolved:
        return _native_fast
    _native_resolved = True

    import os

    # Native kernels are **opt-in** (default-off). The native ``_ext``
    # extension changes Metal command-buffer timing and, under TP, can
    # perturb the collective ordering that the sync-eval fix in
    # ``opt_batch_gen._patched_step`` depends on. Operators who want the
    # native sparse-MLA / exact-block-attention fast path set
    # ``EXO_NATIVE_GLM_KERNELS=1``.
    if os.environ.get("EXO_NATIVE_GLM_KERNELS", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        logger.info(
            "EXO_NATIVE_GLM_KERNELS not enabled: native GLM kernels are "
            "default-off (opt-in via =1); using standard attention fallback."
        )
        return None

    # Prefer EXO's vendored native kernels (built with EXO_BUILD_MLX_KERNELS=1).
    try:
        from exo.worker.engines.mlx.vendor.omlx_custom_kernels.glm_moe_dsa import (  # noqa: F401
            fast as _native_fast,
        )
    except Exception as exc:  # pragma: no cover - vendored fast always imports
        _native_fast = None
        _native_import_error = _detach_import_error(exc)
    else:
        _native_import_error = None

    if _native_fast is None:
        # Fall back to an upstream omlx install (users who pip-installed oMLX
        # with custom kernels get them transparently).
        try:
            from omlx.custom_kernels.glm_moe_dsa import fast as _native_fast  # type: ignore[import-not-found]  # noqa: F401
        except Exception as exc:  # pragma: no cover - upstream omlx not installed
            _native_fast = None
            _native_import_error = _detach_import_error(exc)
        else:
            _native_import_error = None

    # Surface availability once, at first resolution (during model load, after
    # distributed init — safe to initialise Metal here).
    if _native_fast is not None and _native_fast.is_native_available():
        logger.info(
            "GLM-5.2 native Metal kernels available; all required fast symbols "
            "resolved (sparse MLA + exact block attention fast path active)"
        )
    else:
        detail = "extension not built"
        if _native_import_error is not None:
            detail = str(_native_import_error).splitlines()[0][:160]
        logger.info(
            "GLM-5.2 native Metal kernels not available (%s); "
            "falling back to standard attention path with sparse top-k mask.",
            detail,
        )

    return _native_fast


class _FastDispatch:
    """Dispatch to native GLM kernels when available, else mlx.fast."""

    def __getattr__(self, name: str) -> Any:
        native_fast = _resolve_native_fast()
        if native_fast is not None and native_fast.has_symbol(name):
            try:
                return getattr(native_fast, name)
            except AttributeError:
                pass
        return getattr(mx.fast, name)

    def __dir__(self) -> list[str]:
        names = set(dir(mx.fast))
        native_fast = _resolve_native_fast()
        if native_fast is not None:
            names.update(dir(native_fast))
        return sorted(names)

    def has(self, name: str) -> bool:
        """Return True if the symbol is available (native or mlx.fast)."""
        native_fast = _resolve_native_fast()
        return (native_fast is not None and native_fast.has_symbol(name)) or hasattr(
            mx.fast, name
        )

    def missing(self, required: tuple[str, ...]) -> list[str]:
        """Return the list of required symbols that are not available."""
        return [name for name in required if not self.has(name)]

    def native_available(self) -> bool:
        """Return True if native GLM kernels are available."""
        native_fast = _resolve_native_fast()
        return native_fast is not None and native_fast.is_native_available()

    def native_import_error(self) -> Exception | None:
        """Return the import error if native kernels failed to load."""
        native_fast = _resolve_native_fast()
        if native_fast is not None:
            return native_fast.import_error()
        return _native_import_error


fast = _FastDispatch()

__all__ = ["fast"]
