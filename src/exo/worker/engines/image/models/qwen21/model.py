"""Duck-typed model object satisfying EXO's mflux-typed model surface (no mflux).

The runner touches: model.model_config, model.callbacks.start(seed, prompt,
config), model.bits, model.lora_paths, model.lora_scales, model.vae.decode,
model.prompt_cache, model.tokenizers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

from exo.worker.engines.image.models.qwen21 import weights as weights_mod
from exo.worker.engines.image.models.qwen21.text_encoder import (
    Qwen21TextEncoder,
    Qwen21Tokenizer,
)
from exo.worker.engines.image.models.qwen21.transformer import Qwen21Transformer
from exo.worker.engines.image.models.qwen21.vae import Qwen21Vae


class NoOpContext:
    """GenerationContext replacement for the diffusion loop.

    The runner calls ctx.before_loop / ctx.in_loop / ctx.interruption /
    ctx.after_loop around the per-step loop (pipeline/runner.py). All are
    no-ops for this engine (no partial-image streaming, no telemetry).
    """

    def before_loop(self, latents: object) -> None:  # noqa: ARG002
        return None

    def in_loop(self, t: int, latents: object) -> None:  # noqa: ARG002
        return None

    def interruption(self, t: int, latents: object) -> None:  # noqa: ARG002
        return None

    def after_loop(self, latents: object) -> None:  # noqa: ARG002
        return None


class NoOpCallbacks:
    """Callback surface observed by the runner; start() returns the ctx."""

    def start(self, seed: int, prompt: str, config: object) -> NoOpContext:  # noqa: ARG002
        return NoOpContext()

    def step(self, step_index: int, total_steps: int) -> None:  # noqa: ARG002
        return None

    def end(self) -> None:
        return None


class Qwen21ModelConfig:
    """Minimal mflux ModelConfig duck-type consumed by the runner/builders."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.base_model = None
        self.repo_id = model_name
        self.supports_quantization = False
        self.supports_precomputation = False
        self.max_sequence_length = 2048
        self.supports_guidance = False
        self.requires_sigma_shift = False
        self.sigma_min_shift = 0.0
        self.sigma_max_shift = 0.0
        self.terminal_sigma_ratio = 0.0

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Qwen21ModelConfig(model_name={self.model_name!r})"


class Qwen21Model:
    """The model object handed to the EXO distributed machinery."""

    def __init__(
        self,
        model_id: str,
        local_path: Path,
        quantize: int | None = None,  # noqa: ARG002 (8bit is baked in)
    ) -> None:
        self.model_config = Qwen21ModelConfig(model_id)
        self.transformer: Qwen21Transformer = weights_mod.load_transformer(local_path)
        self.text_encoder: Qwen21TextEncoder = weights_mod.load_text_encoder(local_path)
        self.vae: Qwen21Vae = weights_mod.load_vae(local_path)
        processor_dir = weights_mod.load_processor_dir(local_path)
        self.tokenizer = Qwen21Tokenizer(processor_dir)
        self.tokenizers: dict[str, Qwen21Tokenizer] = {"qwen21": self.tokenizer}
        self.prompt_cache: dict[str, tuple[mx.array, mx.array]] = {}
        self.callbacks = NoOpCallbacks()
        self.bits: int | None = None
        self.lora_paths: list[str] | None = None
        self.lora_scales: list[float] | None = None

    def parameters(self) -> Any:  # mx.eval target
        return self.transformer.parameters()


def build_model(
    model_id: str, local_path: Path, quantize: int | None = None
) -> Qwen21Model:
    return Qwen21Model(model_id, local_path, quantize)
