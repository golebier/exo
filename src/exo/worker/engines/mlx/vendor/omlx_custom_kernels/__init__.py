# type: ignore
# ruff: noqa
# SPDX-License-Identifier: Apache-2.0
"""Optional native Metal custom kernels vendored from oMLX.

This package vendors oMLX's ``omlx.custom_kernels`` tree under EXO's own
namespace so EXO can build and ship the native GLM-5.2 (and, later, MiniMax
M3 / Qwen3.5 / Bonsai) Metal kernels without depending on the ``omlx`` Python
package at runtime.

The C++/Metal sources live alongside this package under
``glm_moe_dsa/csrc/`` and ``common/csrc/`` and are compiled by
``mlx_kernels/build_kernels.py`` (or the Darwin-gated nix derivation) into a
nanobind ``_ext`` extension that is installed next to ``fast.py``. When the
extension is absent (default source builds, non-Darwin, or no Xcode) every
``fast.is_native_available()`` check returns ``False`` and the GLM-5.2 model
code transparently falls back to the standard attention path — identical to
EXO's pre-kernel behavior.

Vendored from oMLX commit ``1f1aff3018c097bcbeafca9a483e58d04dee38ba``
(``omlx/custom_kernels/__init__.py``). See ``mlx_kernels/README.md`` for the
build/packaging plan and ``docs/omlx-porting/02-native-metal-kernels.md`` for
the design rationale.
"""

from __future__ import annotations

import importlib

# Phase 1 ships only the GLM-5.2 family (the model EXO already supports).
# Phase 2 adds minimax_m3 / qwen35_prefill / bonsai once EXO serves those
# model families — extend this tuple and vendor their csrc/ trees then.
NATIVE_KERNEL_PACKAGES = ("glm_moe_dsa",)


def native_kernel_status() -> dict[str, dict[str, object]]:
    """Report availability of every optional native kernel extension.

    Source installs only compile the native extensions when built with
    ``EXO_BUILD_MLX_KERNELS=1`` (which additionally needs the Metal
    toolchain on Darwin). Without them the affected model families silently
    fall back to much slower generic paths, so availability is surfaced for
    diagnosability instead of only a log line.

    Never raises: a package that fails to import is reported as unavailable
    with the stringified error.
    """
    status: dict[str, dict[str, object]] = {}
    for name in NATIVE_KERNEL_PACKAGES:
        try:
            fast = importlib.import_module(f"{__name__}.{name}.fast")
            available = bool(fast.is_native_available())
            error = fast.import_error()
        except Exception as exc:  # noqa: BLE001 - status must never break
            available = False
            error = exc
        status[name] = {
            "available": available,
            "import_error": str(error) if error is not None else None,
        }
    return status


__all__ = ["native_kernel_status", "NATIVE_KERNEL_PACKAGES"]
