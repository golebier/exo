"""MLX engine exceptions.

Errors raised by the MLX inference path. They surface to the API user via
``ErrorChunk`` (see ``runner/llm_inference/batch_generator.py``) and to the
cluster as a clean failure rather than an OOM crash.
"""

from __future__ import annotations

# OpenAI-style ``type`` strings for the errors this module defines. Used by
# :func:`http_error_status_for` / :func:`http_error_type_for` so the API layer
# can map a runner-side exception to the right HTTP status without importing
# the exception class on the API process (the ``ErrorChunk`` carries the
# resolved code/type across the zenoh boundary instead).
PREFILL_MEMORY_EXCEEDED_ERROR_TYPE = "PrefillMemoryExceeded"


class PrefillMemoryExceededError(Exception):
    """A prompt's prefill peak would push memory past the prefill ceiling.

    Raised by the preflight admission check (see
    ``cache.raise_if_prefill_exceeds``) so a too-large prompt under
    prefix-cache pressure is rejected with a clear message instead of letting
    the prefill activation allocation OOM the process. Mirrors oMLX's
    ``PrefillMemoryExceededError`` (``omlx/memory_monitor.py``).

    This is a **client-recoverable** error (the prompt is too large for the
    current memory pressure / context budget) so it maps to HTTP 400, not the
    blanket 500 the API used to return for every ``ErrorChunk``.
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


def http_error_status_for(exc: Exception) -> int:
    """HTTP status code for a runner-side exception.

    The runner's ``_send_error`` consults this to populate ``ErrorChunk.error_code``
    so the API layer maps client-recoverable errors to 4xx instead of a blanket
    500. Add new client-recoverable MLX errors here. Unknown exceptions stay
    500 (a genuine internal fault the operator must investigate).
    """
    if isinstance(exc, PrefillMemoryExceededError):
        return 400
    return 500


def http_error_type_for(exc: Exception) -> str:
    """OpenAI-style ``type`` string for a runner-side exception.

    Companion to :func:`http_error_status_for`; defaults to
    ``"InternalServerError"`` so an unmapped exception keeps the historic
    error-event shape.
    """
    if isinstance(exc, PrefillMemoryExceededError):
        return PREFILL_MEMORY_EXCEEDED_ERROR_TYPE
    return "InternalServerError"
