"""Qwen-Image-2.1 Turbo scheduler for the mflux 0.17.5 Config duck-contract.

Euler sampler over the upstream 8-sigma schedule; conforms to the mflux
BaseScheduler duck-typing the runner uses:
  - sigmas property
  - step(noise=..., timestep=..., latents=...)
  - scale_model_input(latents, timestep)  (identity)
Injected post-construction onto the mflux Config (config._scheduler
short-circuits the property getter).
"""

import mlx.core as mx

SAMPLE_SIGMAS: tuple[float, ...] = (
    1.0,
    0.978453,
    0.95418,
    0.926626,
    0.89508,
    0.845148,
    0.704534,
    0.414568,
)


class Qwen21TurboScheduler:
    """8-step Euler with the upstream Turbo sigma schedule."""

    def __init__(self, sigmas: tuple[float, ...] = SAMPLE_SIGMAS) -> None:
        self._sigmas = tuple(sigmas)

    @property
    def sigmas(self) -> mx.array:
        return mx.array(self._sigmas, dtype=mx.float32)

    @property
    def num_steps(self) -> int:
        return len(self._sigmas) - 1

    def step(
        self,
        noise: mx.array,
        timestep: int,
        latents: mx.array,
        seed: int | None = None,  # noqa: ARG002 (mflux BaseScheduler signature)
    ) -> mx.array:
        dt = (self._sigmas[timestep + 1] - self._sigmas[timestep])
        dt_arr = mx.array(dt, dtype=latents.dtype)
        return latents + noise.astype(latents.dtype) * dt_arr

    def scale_model_input(self, latents: mx.array, timestep: int) -> mx.array:  # noqa: ARG002
        return latents

    def set_timesteps(self, num_inference_steps: int) -> None:  # noqa: ARG002
        # Turbo schedule is fixed; ignore external step counts.
        return None


def install_scheduler(config: object, scheduler: Qwen21TurboScheduler | None = None) -> None:
    """Inject the Turbo scheduler into a mflux 0.17.5 Config instance.

    mflux's Config.scheduler property returns the pre-set _scheduler attribute
    when present, so plain attribute assignment is the integration point.
    """
    setattr(config, "_scheduler", scheduler or Qwen21TurboScheduler())