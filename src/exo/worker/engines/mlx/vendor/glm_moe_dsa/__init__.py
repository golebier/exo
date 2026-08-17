# Copyright © 2025 Apple Inc.
"""Vendored GLM-5.2 glm_moe_dsa model implementation.

This package contains the GLM-5.2-specific model code ported from oMLX
(jundot/omlx). It is registered via sys.modules to override the thin shim
in the pinned mlx_lm fork so that mlx_lm.load() picks up the optimized
model with indexer sharing support.
"""

from .glm_moe_dsa_model import (
    GlmMoeDsaAttention,
    GlmMoeDsaDecoderLayer,
    GlmMoeDsaModel,
    Model,
    ModelArgs,
)

__all__ = [
    "GlmMoeDsaAttention",
    "GlmMoeDsaDecoderLayer",
    "GlmMoeDsaModel",
    "Model",
    "ModelArgs",
]
