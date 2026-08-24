# type: ignore
"""Tests for runner-side exception → HTTP error code/type mapping.

Covers the ``exo 11`` HTTP 400 mapping for ``PrefillMemoryExceededError``: a
too-large prompt rejected by the prefill admission guard must surface to the
API user as a 400 (client-recoverable) instead of the blanket 500 every
``ErrorChunk`` used to map to. The mapping lives in
``exo.worker.engines.mlx.exceptions`` and is consumed by the runner's
``_send_error`` to populate ``ErrorChunk.error_code`` / ``error_type``.
"""

from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.common import ModelId
from exo.shared.types.tasks import TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId
from exo.worker.engines.mlx.exceptions import (
    PREFILL_MEMORY_EXCEEDED_ERROR_TYPE,
    PrefillMemoryExceededError,
    http_error_status_for,
    http_error_type_for,
)

_TEST_MODEL = ModelId("test-model")


def _make_task(command_id: str = "cmd-1") -> TextGeneration:
    return TextGeneration(
        command_id=command_id,
        instance_id=InstanceId("inst-1"),
        task_params=TextGenerationTaskParams(
            model=_TEST_MODEL, input=[InputMessage(role="user", content="hi")]
        ),
    )


class TestHttpErrorMapping:
    def test_prefill_memory_exceeded_maps_to_400(self):
        exc = PrefillMemoryExceededError(
            message="Prefill would require ~10 GiB peak but the ceiling is 8 GiB.",
            estimated_bytes=10 * 1024**3,
            limit_bytes=8 * 1024**3,
        )
        assert http_error_status_for(exc) == 400
        assert http_error_type_for(exc) == PREFILL_MEMORY_EXCEEDED_ERROR_TYPE

    def test_generic_exception_maps_to_500(self):
        exc = RuntimeError("something broke")
        assert http_error_status_for(exc) == 500
        assert http_error_type_for(exc) == "InternalServerError"

    def test_value_error_maps_to_500(self):
        # A plain ValueError is not client-recoverable by default.
        exc = ValueError("bad")
        assert http_error_status_for(exc) == 500
        assert http_error_type_for(exc) == "InternalServerError"


class TestErrorChunkDefaults:
    def test_error_chunk_defaults_to_500_internal(self):
        """An ErrorChunk built without a mapped cause keeps historic behaviour."""
        chunk = ErrorChunk(model=_TEST_MODEL, error_message="boom")
        assert chunk.error_code == 500
        assert chunk.error_type == "InternalServerError"

    def test_error_chunk_accepts_mapped_code(self):
        chunk = ErrorChunk(
            model=_TEST_MODEL,
            error_message="too big",
            error_code=400,
            error_type=PREFILL_MEMORY_EXCEEDED_ERROR_TYPE,
        )
        assert chunk.error_code == 400
        assert chunk.error_type == PREFILL_MEMORY_EXCEEDED_ERROR_TYPE


class TestSendErrorMapping:
    """The runner's ``_send_error`` populates error_code/type from the exception.

    We exercise it through the ``BatchGenerator`` engine's ``_send_error`` (the
    continuous-batching path) by constructing it with a stub event sender and
    invoking the method directly — mirroring how a prefill-memory rejection
    reaches the API as an ``ErrorChunk``.
    """

    def test_batch_send_error_maps_prefill_memory_exceeded(self):
        from unittest.mock import MagicMock

        from exo.shared.types.events import ChunkGenerated
        from exo.worker.runner.llm_inference.batch_generator import BatchGenerator

        engine = BatchGenerator.__new__(BatchGenerator)
        engine.device_rank = 0
        engine.model_id = _TEST_MODEL
        engine.event_sender = MagicMock()

        task = _make_task()
        exc = PrefillMemoryExceededError(
            message="Prefill would require ~10 GiB peak.",
            estimated_bytes=10 * 1024**3,
            limit_bytes=8 * 1024**3,
        )
        engine._send_error(task, exc)

        sent = engine.event_sender.send.call_args.args[0]
        assert isinstance(sent, ChunkGenerated)
        assert isinstance(sent.chunk, ErrorChunk)
        assert sent.chunk.error_code == 400
        assert sent.chunk.error_type == PREFILL_MEMORY_EXCEEDED_ERROR_TYPE
        assert "Prefill would require" in sent.chunk.error_message

    def test_batch_send_error_defaults_generic_to_500(self):
        from unittest.mock import MagicMock

        from exo.shared.types.events import ChunkGenerated
        from exo.worker.runner.llm_inference.batch_generator import BatchGenerator

        engine = BatchGenerator.__new__(BatchGenerator)
        engine.device_rank = 0
        engine.model_id = _TEST_MODEL
        engine.event_sender = MagicMock()

        task = _make_task("cmd-2")
        engine._send_error(task, RuntimeError("kaboom"))

        sent = engine.event_sender.send.call_args.args[0]
        assert isinstance(sent, ChunkGenerated)
        assert sent.chunk.error_code == 500
        assert sent.chunk.error_type == "InternalServerError"
