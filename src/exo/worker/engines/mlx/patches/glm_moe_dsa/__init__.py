# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 glm_moe_dsa monkey-patch for mlx-lm.

Vendors the GLM-5.2 optimized mlx-lm model code without modifying the pinned
mlx-lm package. The public module name stays ``mlx_lm.models.glm_moe_dsa``
so mlx-lm's normal model loader can find it, but the optimized helper modules
remain private to ``exo.worker.engines.mlx.vendor.glm_moe_dsa`` and do not
replace ``mlx_lm.models.deepseek_v32`` or the shared MoE layers used by other
model families.

Ported from jundot/omlx ommlx/patches/glm_moe_dsa/__init__.py.
"""

from __future__ import annotations

import importlib
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

PATCH_SOURCE = "exo vendored GLM-5.2 optimized mlx-lm model"
NATIVE_KERNELS_PACKAGE = "omlx.custom_kernels.glm_moe_dsa"

_applied = False


def _register_module() -> None:
    """Register the vendored GLM-5.2 model as mlx_lm.models.glm_moe_dsa."""
    qualname = "mlx_lm.models.glm_moe_dsa"
    existing = sys.modules.get(qualname)
    if getattr(existing, "_EXO_GLM_DSA_OPTIMIZED", False):
        module: Any = existing
    else:
        module = importlib.import_module(
            "exo.worker.engines.mlx.vendor.glm_moe_dsa.glm_moe_dsa_model"
        )
        module._EXO_GLM_DSA_OPTIMIZED = True

    sys.modules[qualname] = module

    models_pkg = importlib.import_module("mlx_lm.models")
    models_pkg.glm_moe_dsa = module  # type: ignore[attr-defined]
    logger.info("Registered %s from EXO vendored module", qualname)


def apply_glm_moe_dsa_patch() -> bool:
    """Apply the GLM MoE DSA patch. Idempotent.

    Must run before ``mlx_lm.load()`` imports ``mlx_lm.models.glm_moe_dsa``.

    Returns True when EXO registered its vendored module, False when the patch
    was already applied or mlx-lm is unavailable.
    """
    global _applied
    if _applied:
        return False

    try:
        importlib.import_module("mlx_lm")
    except ImportError:
        logger.debug("mlx_lm not importable - glm_moe_dsa patch skipped")
        return False

    _register_module()

    # NOTE: native-kernel availability is NOT probed here. apply_glm_moe_patch
    # runs in the runner BEFORE mx.distributed.init(backend="jaccl"), and
    # importing the native _ext.so initialises the Metal device — which must
    # not happen before the jaccl GPU-RDMA backend owns Metal init (else the
    # JACCL warmup all_sum hangs on rank 1). kernels.fast resolves _ext lazily
    # on first access (during model load, after distributed init) and logs the
    # availability then. See docs/omlx-porting/02-native-metal-kernels.md.

    _applied = True
    return True


__all__ = ["apply_glm_moe_dsa_patch"]
