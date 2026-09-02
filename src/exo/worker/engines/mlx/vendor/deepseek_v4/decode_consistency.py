# SPDX-License-Identifier: Apache-2.0
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportConstantRedefinition=false
"""M=1-equivalent reductions for DeepSeek DSpark target verification.

Vendored from oMLX ``omlx/patches/deepseek_v4/decode_consistency.py``.

EXO does not ship the DSpark verification harness, so ``is_armed()`` is
always ``False`` and :func:`matmul` falls through to the plain ``x @ weight_t``.
The module exists only to satisfy ``hyper_connection.py``'s import; the
verification path (``apply()``) is a no-op here.
"""

from __future__ import annotations

import threading
from typing import Any

import mlx.core as mx

_STATE = threading.local()
_PATCHED = False


def set_armed(flag: bool) -> None:
    _STATE.armed = bool(flag)


def is_armed() -> bool:
    return bool(getattr(_STATE, "armed", False))


def matmul(x: mx.array, weight_t: mx.array) -> mx.array:
    if not is_armed():
        result: Any = x @ weight_t
        return result
    width = int(x.shape[-1])
    rows = int(x.size) // width
    if rows <= 1:
        result_fallback: Any = x @ weight_t
        return result_fallback
    flat: Any = x.reshape(rows, width)
    result_batched: Any = (weight_t.T @ flat[..., None]).squeeze(-1)
    return result_batched.reshape((*x.shape[:-1], result_batched.shape[-1]))


def apply() -> bool:
    """No-op in EXO — the DSpark verification harness is not ported."""
    global _PATCHED
    _PATCHED = True
    return True


__all__ = [
    "apply",
    "is_armed",
    "matmul",
    "set_armed",
]
