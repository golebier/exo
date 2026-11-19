"""EXO ModelAdapter for the pure-MLX Qwen-Image-2.1 Turbo engine (no mflux).

Duck-types the mflux objects the runner requires: ModelConfig + Config (with
the Turbo scheduler injected post-construction), 64-channel plain spatial
latent packing, single CFG branch, rope table over [text | image] positions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image

from exo.worker.engines.image.config import ImageModelConfig
from exo.worker.engines.image.models.base import (
    ModelAdapter,
    PromptData,
    RotaryEmbeddings,
)
from exo.worker.engines.image.models.qwen21 import weights as weights_mod
from exo.worker.engines.image.models.qwen21.model import Qwen21Model
from exo.worker.engines.image.models.qwen21.scheduler import (
    Qwen21TurboScheduler,
    SAMPLE_SIGMAS,
)
from exo.worker.engines.image.models.qwen21.text_encoder import encode_prompts
from exo.worker.engines.image.models.qwen21.transformer import (
    Qwen21Transformer,
    Qwen21Rope,
)
from exo.worker.engines.image.models.qwen21.wrappers import Qwen21JointBlockWrapper
from exo.worker.engines.image.pipeline.block_wrapper import (
    JointBlockWrapper,
    SingleBlockWrapper,
)


class _LatentCreator:
    """Static namespace matching the mflux LatentCreator duck-contract."""

    @staticmethod
    def create_noise(seed: int, height: int, width: int) -> mx.array:
        return mx.random.normal(
            shape=(1, (height // 16) * (width // 16), 64),
            key=mx.random.key(seed),
        ).astype(mx.bfloat16)  # mflux LatentCreator casts to ModelConfig.precision

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        h, w = height // 16, width // 16
        return mx.transpose(
            latents.reshape(1, h, w, 64), (0, 3, 1, 2)
        ).reshape(1, 64, 1, h, w)

    @staticmethod
    def create_for_txt2img_or_img2img(
        seed: int,
        height: int,
        width: int,
        img2img: Any,
    ) -> mx.array:
        if img2img is None or getattr(img2img, "image_path", None) is None:
            return _LatentCreator.create_noise(seed, height, width)
        raise NotImplementedError("qwen21 adapter supports txt2img only")

    @staticmethod
    def pack_latents(latents: mx.array) -> mx.array:
        return mx.transpose(latents, (0, 2, 3, 1)).reshape(1, -1, 64)


class Qwen21PromptData(PromptData):
    def __init__(self, embeds: mx.array, mask: mx.array):
        self._embeds = embeds
        self._mask = mask

    @property
    def prompt_embeds(self) -> mx.array:
        return self._embeds

    @property
    def pooled_prompt_embeds(self) -> mx.array:
        return self._embeds

    @property
    def negative_prompt_embeds(self) -> mx.array | None:
        return None

    @property
    def negative_pooled_prompt_embeds(self) -> mx.array | None:
        return None

    def get_encoder_hidden_states_mask(self, positive: bool = True) -> mx.array | None:  # noqa: ARG002
        return self._mask

    @property
    def cond_image_grid(self) -> tuple[int, int, int] | list[tuple[int, int, int]] | None:
        return None

    @property
    def conditioning_latents(self) -> mx.array | None:
        return None

    @property
    def kontext_image_ids(self) -> mx.array | None:
        return None

    def get_batched_cfg_data(self) -> None:
        return None

    def get_cfg_branch_data(
        self, positive: bool  # noqa: ARG002 (single branch)
    ) -> tuple[mx.array, mx.array | None, mx.array | None, mx.array | None]:
        return (self._embeds, self._mask, None, None)


class Qwen21ModelAdapter(ModelAdapter[Qwen21Model, Qwen21Transformer]):
    def __init__(
        self,
        config: ImageModelConfig,
        model_id: str,
        local_path: Path,
        quantize: int | None = None,
    ):
        self._config = config
        self._model = Qwen21Model(model_id, local_path, quantize)
        self._transformer = self._model.transformer
        self._rope_cache: dict[tuple[int, int, int], tuple[mx.array, mx.array]] = {}
        self._norm_out_scale: mx.array | None = None

    @property
    def hidden_dim(self) -> int:
        return self._transformer.inner_dim if hasattr(
            self._transformer, "inner_dim"
        ) else 4096

    @property
    def needs_cfg(self) -> bool:
        gs = self._config.guidance_scale
        return gs is not None and gs > 1.0

    def _get_latent_creator(self) -> type:
        return _LatentCreator

    def _ensure_turbo_scheduler(self, runtime_config: Any) -> None:
        """Inject the Turbo schedule into the runner's mflux Config.

        The runner accesses config.scheduler BEFORE the first per-step adapter
        hook (scale_model_input at the top of the diffusion loop), and the base
        adapter's create_latents would resolve 'linear' to LinearScheduler. The
        adapter's own create_latents is the first hook that runs, so the
        injection happens there (idempotent via the _scheduler attribute).
        """
        if getattr(runtime_config, "_scheduler", None) is None:
            runtime_config._scheduler = Qwen21TurboScheduler()  # noqa: SLF001

    def get_joint_block_wrappers(
        self,
        text_seq_len: int,
        encoder_hidden_states_mask: mx.array | None = None,
    ) -> list[JointBlockWrapper[Any]]:
        return [
            Qwen21JointBlockWrapper(block, text_seq_len, encoder_hidden_states_mask)
            for block in self._transformer.transformer_blocks
        ]

    def get_single_block_wrappers(
        self,
        text_seq_len: int,  # noqa: ARG002
    ) -> list[SingleBlockWrapper[Any]]:
        return []

    def slice_transformer_blocks(self, start_layer: int, end_layer: int) -> None:
        self._transformer.transformer_blocks = self._transformer.transformer_blocks[
            start_layer:end_layer
        ]

    def encode_prompt(
        self, prompt: str, negative_prompt: str | None = None
    ) -> Qwen21PromptData:
        del negative_prompt  # Turbo: single branch only
        model = self._model
        (embeds, mask), = encode_prompts(
            model.text_encoder,
            model.tokenizer,
            [prompt],
            model.prompt_cache,
        )
        return Qwen21PromptData(embeds, mask)

    def compute_embeddings(
        self,
        hidden_states: mx.array,
        prompt_embeds: mx.array,
    ) -> tuple[mx.array, mx.array]:
        embedded_hidden = self._transformer.img_in(hidden_states)
        embedded_encoder = self._transformer.txt_in(prompt_embeds)
        return embedded_hidden, embedded_encoder

    def compute_text_embeddings(
        self,
        t: int,
        runtime_config: Any,
        pooled_prompt_embeds: mx.array | None = None,
        hidden_states: mx.array | None = None,
    ) -> mx.array:
        del runtime_config, pooled_prompt_embeds, hidden_states
        batch = 1
        # The runner passes the raw loop index (0..num_steps-1); map it to the
        # Turbo sigma value exactly like mflux 0.24 QwenImage21TurboPipeline.
        idx = min(max(int(t), 0), len(SAMPLE_SIGMAS) - 1)
        timestep = mx.full((batch,), float(SAMPLE_SIGMAS[idx]), dtype=mx.float32)
        return self._time_text(timestep)

    def _time_text(self, timestep: mx.array) -> mx.array:
        batch = timestep.shape[0]
        # Qwen21TimeTextEmbed takes [batch] timestep and returns [batch, 2, projected]
        temb = self._transformer.time_text_embed(timestep)
        self._norm_out_scale = self._transformer.norm_out.scale_for(temb[:, 0, :])
        return self._transformer.modulation_for(temb)

    def compute_rotary_embeddings(
        self,
        prompt_embeds: mx.array,
        runtime_config: Any,
        encoder_hidden_states_mask: mx.array | None = None,
        cond_image_grid: tuple[int, int, int] | list[tuple[int, int, int]] | None = None,
        kontext_image_ids: mx.array | None = None,
    ) -> RotaryEmbeddings:
        del cond_image_grid, kontext_image_ids, encoder_hidden_states_mask
        # The rope table spans [text | image] positions; the text length is the
        # (prefix-dropped) prompt length, exactly like mflux pos_embed(text_len, ...).
        text_seq_len = int(prompt_embeds.shape[1])
        height = runtime_config.height // 16
        width = runtime_config.width // 16
        key = (text_seq_len, height, width)
        if key not in self._rope_cache:
            rope = Qwen21Rope(text_seq_len, height, width)
            self._rope_cache[key] = rope.get()
        return self._rope_cache[key]

    def apply_guidance(
        self,
        noise_positive: mx.array,
        noise_negative: mx.array,
        guidance_scale: float,
    ) -> mx.array:
        raise NotImplementedError("qwen21 turbo does not use CFG")

    def final_projection(
        self,
        hidden_states: mx.array,
        text_embeddings: mx.array,  # noqa: ARG002 (scale stashed by _time_text)
    ) -> mx.array:
        if self._norm_out_scale is None:
            raise RuntimeError("compute_text_embeddings must run before final_projection")
        scale = self._norm_out_scale[:, None, :]
        seq_len = hidden_states.shape[1]
        scale = mx.broadcast_to(scale, (scale.shape[0], seq_len, scale.shape[2]))
        normed = self._transformer.norm_out.norm(hidden_states)
        return self._transformer.proj_out(normed * (1 + scale))

    def decode_latents(
        self,
        latents: mx.array,
        runtime_config: Any,
        seed: int,  # noqa: ARG002
        prompt: str,  # noqa: ARG002
    ) -> Image.Image:
        latents = _LatentCreator.unpack_latents(
            latents, runtime_config.height, runtime_config.width
        )
        decoded = self._model.vae.decode(latents)  # [1, 3, H, W] in [-1, 1]
        # mflux ImageUtil.to_pil equivalent: denormalize to [0, 1] then to
        # uint8 BHWC.
        arr = np.clip(
            np.asarray(decoded.astype(mx.float32)).transpose(0, 2, 3, 1) / 2 + 0.5,
            0.0,
            1.0,
        )
        pixels = (arr[0] * 255).round().astype(np.uint8)
        return Image.fromarray(pixels)

    def create_latents(self, seed: int, runtime_config: Any) -> mx.array:
        self._ensure_turbo_scheduler(runtime_config)
        return _LatentCreator.create_noise(
            seed, runtime_config.height, runtime_config.width
        )