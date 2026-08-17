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


def _missing_fast_symbols() -> list[str]:
    """Return expected fast symbols missing from native or patched MLX runtime."""
    from exo.worker.engines.mlx.vendor.glm_moe_dsa import kernels

    required = (
        "dsa_indexer_scores",
        "dsa_topk_indices",
        "glm_dsa_sparse_mla_attention",
        "glm_dsa_exact_block_attention",
        "glm_dsa_q8_vup_flat",
        "glm_moe_weighted_sum",
    )
    return kernels.fast.missing(required)


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

    # Log missing native symbols (EXO doesn't ship native GLM kernels, so the
    # sparse MLA / exact block paths fall back to the standard attention path).
    missing = _missing_fast_symbols()
    if missing:
        logger.info(
            "GLM-5.2 native kernels not available (%s missing); "
            "falling back to standard attention path with sparse top-k mask",
            ", ".join(missing),
        )

    _applied = True
    return True


__all__ = ["apply_glm_moe_dsa_patch"]
