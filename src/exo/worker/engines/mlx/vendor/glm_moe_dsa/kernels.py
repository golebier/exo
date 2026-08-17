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


# EXO does not ship oMLX's native GLM kernels. The fast dispatch falls back
# to mx.fast for the symbols it provides, and returns None for the GLM-specific
# native kernels (sparse MLA, exact block attention, etc.) so the model code
# falls through to the standard attention path.
try:
    from omlx.custom_kernels.glm_moe_dsa import fast as _native_fast  # type: ignore[import-not-found]  # noqa: F401
except Exception as exc:  # pragma: no cover - native extension not shipped
    _native_fast = None
    _native_import_error = _detach_import_error(exc)
else:
    _native_import_error = None


class _FastDispatch:
    """Dispatch to native GLM kernels when available, else mlx.fast."""

    def __getattr__(self, name: str) -> Any:
        if _native_fast is not None and _native_fast.has_symbol(name):
            try:
                return getattr(_native_fast, name)
            except AttributeError:
                pass
        return getattr(mx.fast, name)

    def __dir__(self) -> list[str]:
        names = set(dir(mx.fast))
        if _native_fast is not None:
            names.update(dir(_native_fast))
        return sorted(names)

    def has(self, name: str) -> bool:
        """Return True if the symbol is available (native or mlx.fast)."""
        return (_native_fast is not None and _native_fast.has_symbol(name)) or hasattr(
            mx.fast, name
        )

    def missing(self, required: tuple[str, ...]) -> list[str]:
        """Return the list of required symbols that are not available."""
        return [name for name in required if not self.has(name)]

    def native_available(self) -> bool:
        """Return True if native GLM kernels are available."""
        return _native_fast is not None and _native_fast.is_native_available()

    def native_import_error(self) -> Exception | None:
        """Return the import error if native kernels failed to load."""
        if _native_fast is not None:
            return _native_fast.import_error()
        return _native_import_error


fast = _FastDispatch()

__all__ = ["fast"]
