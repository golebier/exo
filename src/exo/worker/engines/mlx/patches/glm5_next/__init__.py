# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 (``glm5_next``) monkey-patch for mlx-vlm.

Vendors the GLM-5.3 optimized mlx-vlm model code without modifying the pinned
mlx-vlm package. Exposes ``mlx_vlm.models.glm5_next`` from EXO's vendor tree so
mlx-vlm's model loader finds it, and injects the DeepSeek-V4 runtime support
(``PoolingCache`` into ``mlx_lm.models.cache``; ``mlx_vlm.models.deepseek_v4``)
that the GLM-5.3 language model depends on.

Ported from oMLX ``omlx/patches/mlx_vlm_glm5_next_compat/__init__.py``.

Must run before ``mlx_vlm.load()`` imports ``mlx_vlm.models.glm5_next``.
Invoked from ``exo.worker.engines.mlx.patches.apply_mlx_patches`` at worker
startup (``bootstrap.py``).
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Any

# mlx_vlm / mlx_lm ship no type stubs; the registration shim pokes at their
# internals (MODEL_CONFIG, __path__, cache module attrs) which pyright cannot
# resolve. Disable the stub-dependent strict rules for this file only.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false, reportAny=false

logger = logging.getLogger(__name__)

PR_URL = "https://github.com/Blaizzy/mlx-vlm/pull/2030"
PR_MERGE_SHA = "fa27a9a692770c39fdf57b9a985fad084a90aec2"
# The vendor tree holds ``glm5_next/`` and ``deepseek_v4/`` as subpackages.
# Adding this directory to ``mlx_vlm.models.__path__`` makes both importable
# as ``mlx_vlm.models.glm5_next`` / ``mlx_vlm.models.deepseek_v4``.
_VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor"
_patch_applied = False


def _append_package_path(package: Any, path: Path) -> None:
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return
    path_string = str(path)
    if path_string not in package_path:
        package_path.append(path_string)


def _inject_pooling_cache() -> None:
    """Add PoolingCache + BatchPoolingCache to ``mlx_lm.models.cache``.

    GLM-5.3's sparse-attention layers wrap a ``PoolingCache`` (the pooled
    indexer cache) in a ``CacheList`` alongside a ``KVCache``. mlx-lm does not
    yet ship ``PoolingCache`` (PR 1192 pending), so it is injected here.
    Idempotent.
    """
    import mlx_lm.models.cache as _cache_mod

    if hasattr(_cache_mod, "PoolingCache") and hasattr(_cache_mod, "BatchPoolingCache"):
        return

    from exo.worker.engines.mlx.vendor.deepseek_v4.cache_extras import (
        BatchPoolingCache,
        PoolingCache,
    )

    _cache_mod.PoolingCache = PoolingCache
    _cache_mod.BatchPoolingCache = BatchPoolingCache
    # Also expose at __dict__ level so callers that reload the module after the
    # patch (e.g. ``from mlx_lm.models.cache import PoolingCache``) see them.
    _cache_mod.__dict__["PoolingCache"] = PoolingCache
    _cache_mod.__dict__["BatchPoolingCache"] = BatchPoolingCache
    # Reattach the classes' __module__ so class-name introspection matches what
    # mlx-lm code expects.
    PoolingCache.__module__ = "mlx_lm.models.cache"
    BatchPoolingCache.__module__ = "mlx_lm.models.cache"
    logger.info("PoolingCache / BatchPoolingCache injected into mlx_lm.models.cache")


def _register_vendor_on_mlx_vlm_models() -> None:
    """Put the vendor dir on ``mlx_vlm.models.__path__`` and pre-import the subpackages.

    The GLM-5.3 language model uses relative imports
    (``from ..deepseek_v4.hyper_connection import ...``) that resolve against
    ``mlx_vlm.models``. mlx-vlm 0.4.4 has neither ``glm5_next`` nor
    ``deepseek_v4``, so both vendored subpackages are exposed under the
    ``mlx_vlm.models.*`` names.
    """
    import mlx_vlm.models

    _append_package_path(mlx_vlm.models, _VENDOR_DIR)

    # Pre-import under the public mlx_vlm.models.* names so the relative
    # imports inside language.py (..deepseek_v4.hyper_connection) resolve
    # immediately, and so mlx-vlm's loader finds glm5_next in sys.modules.
    for qualname, exo_path in (
        ("mlx_vlm.models.deepseek_v4", "exo.worker.engines.mlx.vendor.deepseek_v4"),
        ("mlx_vlm.models.glm5_next", "exo.worker.engines.mlx.vendor.glm5_next"),
    ):
        if qualname not in sys.modules:
            try:
                sys.modules[qualname] = importlib.import_module(exo_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("GLM-5.3 vendor import %s failed: %s", qualname, exc)


def apply_glm5_next_patch() -> bool:
    """Expose ``mlx_vlm.models.glm5_next`` from EXO's vendor tree. Idempotent.

    Returns ``True`` when EXO freshly registered the module, ``False`` when the
    patch was already applied or mlx-vlm is unavailable.
    """
    global _patch_applied
    if _patch_applied:
        return False

    try:
        import importlib.util

        if importlib.util.find_spec("mlx_vlm") is None:
            raise ImportError("mlx_vlm not found")
    except ImportError:
        logger.debug("mlx_vlm not importable — glm5_next patch skipped")
        return False

    # 1. Inject PoolingCache / BatchPoolingCache into mlx_lm.models.cache.
    #    GLM-5.3's make_cache imports ``from mlx_lm.models.cache import PoolingCache``.
    _inject_pooling_cache()

    # 2. Expose the vendored glm5_next + deepseek_v4 subpackages under the
    #    mlx_vlm.models.* names (relative imports in language.py depend on this).
    _register_vendor_on_mlx_vlm_models()

    # 3. mlx-vlm has no glm5_next entry in MODEL_CONFIG, so get_message_json()
    #    raises "Unsupported model: glm5_next" on every turn that carries no
    #    image.  GLM-5.3 takes the same list-with-image-first shape as glm4v.
    try:
        from mlx_vlm.prompt_utils import MODEL_CONFIG, MessageFormat

        MODEL_CONFIG.setdefault("glm5_next", MessageFormat.LIST_WITH_IMAGE_FIRST)
    except Exception as exc:  # noqa: BLE001
        logger.debug("GLM-5.3 MODEL_CONFIG registration skipped: %s", exc)

    _patch_applied = True
    logger.info("GLM-5.3 (glm5_next) mlx-vlm compatibility patch applied")
    return True


__all__ = [
    "PR_MERGE_SHA",
    "PR_URL",
    "apply_glm5_next_patch",
]
