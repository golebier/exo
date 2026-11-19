"""Qwen-Image-2.1 Turbo image model config for EXO."""

from exo.worker.engines.image.config import (
    BlockType,
    ImageModelConfig,
    TransformerBlockConfig,
)

# Qwen-Image-2.1 Turbo: 32 joint blocks, single stream, no CFG.
# Turbo sampler runs a FIXED 8 denoising steps (upstream sample_sigmas),
# so every quality preset maps to 8 steps.
QWEN21_IMAGE_CONFIG = ImageModelConfig(
    model_family="qwen21",
    block_configs=(
        TransformerBlockConfig(
            block_type=BlockType.JOINT, count=32, has_separate_text_output=True
        ),
    ),
    default_steps={"low": 8, "medium": 8, "high": 8},
    num_sync_steps=8,
    guidance_scale=None,  # Turbo: no classifier-free guidance
)