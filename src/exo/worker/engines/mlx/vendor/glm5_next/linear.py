# SPDX-License-Identifier: Apache-2.0
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportConstantRedefinition=false
"""Model-neutral affine qmm routing for GLM-5.3 projections.

Vendored from oMLX ``omlx/patches/mlx_vlm_glm5_next_compat/vendor/mlx_vlm/models/glm5_next/linear.py``.

EXO does not ship oMLX's qwen35 native prefill kernels, so the native
affine-qmm fast path is removed and :func:`fused_quantized_matmul` falls back
to :func:`mx.quantized_matmul`. Functionally correct; slower on long prefills
than the native tile, but the GLM-5.3 model loads and runs.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn


def linear_forward(linear: nn.Module, x: mx.array) -> mx.array:
    """Affine-quantized matmul routing for GLM-5.3 projections.

    For an affine-mode ``QuantizedLinear`` with ``biases``, route the
    concatenated projection through :func:`fused_quantized_matmul` and add the
    bias. Otherwise defer to the standard ``nn.Module.__call__``.
    """
    if not (
        isinstance(linear, nn.QuantizedLinear)
        and "bias" in linear
        and getattr(linear, "mode", None) == "affine"
        and getattr(linear, "biases", None) is not None
    ):
        return linear(x)

    weight: mx.array = linear.weight  # type: ignore[reportAny]
    scales: mx.array = linear.scales  # type: ignore[reportAny]
    biases: mx.array = linear.biases  # type: ignore[reportAny]
    bits = int(linear.bits)  # type: ignore[reportAny]
    group_size = int(linear.group_size)  # type: ignore[reportAny]
    out = fused_quantized_matmul(
        x, weight, scales, biases, bits=bits, group_size=group_size
    )
    bias: mx.array = linear.bias  # type: ignore[reportAny]
    return out + bias


def fused_quantized_matmul(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    bits: int,
    group_size: int,
) -> mx.array:
    """Route a concatenated projection through ``mx.quantized_matmul``.

    oMLX's native ``qwen35_q{bits}_affine_qmm_t`` tile is not available in EXO,
    so this always uses the MLX built-in quantized matmul (correct, just slower
    on long prefills).
    """
    result: Any = mx.quantized_matmul(
        x,
        weight,
        scales,
        biases,
        transpose=True,
        group_size=group_size,
        bits=bits,
    )
    return result
