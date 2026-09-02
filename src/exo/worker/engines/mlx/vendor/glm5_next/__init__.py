# SPDX-License-Identifier: Apache-2.0
"""Vendored GLM-5.3 (``glm5_next``) model implementation for mlx-vlm.

Vendored from oMLX ``omlx/patches/mlx_vlm_glm5_next_compat/vendor/mlx_vlm/models/glm5_next/``
(based on Blaizzy/mlx-vlm PR #2030 at ``fa27a9a6``). Registered via
``exo.worker.engines.mlx.patches.glm5_next`` as ``mlx_vlm.models.glm5_next``
so mlx-vlm's model loader finds it, without modifying the pinned mlx-vlm package.

The language model reuses GLM-5.2's DSA components (vendored under
``exo.worker.engines.mlx.vendor.glm_moe_dsa``) and adds:
- gated-delta (GDN) linear-attention layers (``gated_delta.py``);
- DeepSeek-V4 HyperConnection per layer (``vendor/deepseek_v4/hyper_connection.py``);
- a pooled indexer cache (``PoolingCache``) wrapped in ``CacheList`` with ``KVCache``.

Lightning MTP is intentionally not included (mirrors oMLX). oMLX's native
qwen35 affine-qmm prefill tile is not ported; ``linear.py`` falls back to
``mx.quantized_matmul``.
"""

from .config import ModelConfig, TextConfig, VisionConfig
from .glm5_next import Model
from .language import LanguageModel
from .processing import Glm5NextImageProcessor, Glm5NextProcessor
from .vision import VisionModel

__all__ = [
    "Model",
    "ModelConfig",
    "TextConfig",
    "VisionConfig",
    "LanguageModel",
    "VisionModel",
    "Glm5NextImageProcessor",
    "Glm5NextProcessor",
]
