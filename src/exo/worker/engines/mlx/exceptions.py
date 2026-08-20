"""MLX engine exceptions.

Errors raised by the MLX inference path. They surface to the API user via
``ErrorChunk`` (see ``runner/llm_inference/batch_generator.py``) and to the
cluster as a clean failure rather than an OOM crash.
"""

from __future__ import annotations


class PrefillMemoryExceededError(Exception):
    """A prompt's prefill peak would push memory past the prefill ceiling.

    Raised by the preflight admission check (see
    ``cache.raise_if_prefill_exceeds``) so a too-large prompt under
    prefix-cache pressure is rejected with a clear message instead of letting
    the prefill activation allocation OOM the process. Mirrors oMLX's
    ``PrefillMemoryExceededError`` (``omlx/memory_monitor.py``).
    """

    def __init__(
        self,
        *,
        message: str,
        estimated_bytes: int,
        limit_bytes: int,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.estimated_bytes = int(estimated_bytes)
        self.limit_bytes = int(limit_bytes)
