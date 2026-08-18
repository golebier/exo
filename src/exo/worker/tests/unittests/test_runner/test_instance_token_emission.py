"""Token-usage attribution emitted by ``Runner.send_chunk``.

The runner is the single choke point through which every generation chunk
flows, so it is the natural place to emit ``InstanceTokensUpdated``. These
tests construct a ``Runner`` with a no-op fake engine and drive ``send_chunk``
directly to assert the event is emitted (or not) for each chunk type.
"""

from collections.abc import Iterable
from typing import BinaryIO

from exo.api.types import CompletionTokensDetails, PromptTokensDetails, Usage
from exo.shared.types.chunks import (
    Chunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import CommandId
from exo.shared.types.events import ChunkGenerated, Event, InstanceTokensUpdated
from exo.shared.types.tasks import GenerationTask, Task, TaskId
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
)
from exo.utils.channels import mp_channel
from exo.worker.engines.base import Engine
from exo.worker.runner.runner import Runner

from ...constants import INSTANCE_1_ID, MODEL_A_ID, NODE_A, RUNNER_1_ID
from ..conftest import get_bound_mlx_ring_instance

COMMAND_ID = CommandId("cmd-token-usage")


class _NoOpEngine(Engine):
    """Minimal Engine so we can construct a Runner without MLX."""

    _cancelled_tasks: set[TaskId] = set()

    def warmup(self) -> None: ...

    def submit(self, task: GenerationTask) -> None: ...

    def step(
        self,
    ) -> Iterable[tuple[TaskId, Chunk | CancelledResponse | FinishedResponse]]:
        return []

    def close(self) -> None: ...

    def serve_prefill(self, request: object, wfile: BinaryIO) -> None: ...


class _EventCollector:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def send(self, event: Event) -> None:
        self.events.append(event)

    def close(self) -> None: ...

    def join(self) -> None: ...


def _usage(prompt: int, completion: int) -> Usage:
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0),
    )


def _make_runner() -> tuple[Runner, _EventCollector]:
    bound_instance = get_bound_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        runner_id=RUNNER_1_ID,
        node_id=NODE_A,
    )
    _task_sender, task_receiver = mp_channel[Task]()
    _cancel_sender, _cancel_receiver = mp_channel[TaskId]()
    collector = _EventCollector()

    # Avoid the real prefill server / task reader threads touching real sockets.
    task_receiver.close = lambda: None
    task_receiver.join = lambda: None

    runner = Runner(
        bound_instance,
        _NoOpEngine(),  # pyright: ignore[reportArgumentType]
        collector,  # pyright: ignore[reportArgumentType]
        task_receiver,
    )
    return runner, collector


def _instance_tokens_events(events: list[Event]) -> list[InstanceTokensUpdated]:
    return [e for e in events if isinstance(e, InstanceTokensUpdated)]


def test_final_token_chunk_emits_usage() -> None:
    runner, collector = _make_runner()
    chunk = TokenChunk(
        model=MODEL_A_ID,
        text="hello",
        token_id=0,
        usage=_usage(prompt=12, completion=5),
        finish_reason="stop",
    )

    runner.send_chunk(chunk, COMMAND_ID)

    emitted = _instance_tokens_events(collector.events)
    assert len(emitted) == 1
    assert emitted[0].instance_id == INSTANCE_1_ID
    assert emitted[0].prompt_tokens == 12
    assert emitted[0].completion_tokens == 5


def test_intermediate_token_chunk_without_finish_reason_does_not_emit() -> None:
    runner, collector = _make_runner()
    chunk = TokenChunk(
        model=MODEL_A_ID,
        text="hel",
        token_id=0,
        usage=None,
        finish_reason=None,
    )

    runner.send_chunk(chunk, COMMAND_ID)

    assert _instance_tokens_events(collector.events) == []


def test_final_token_chunk_without_usage_does_not_emit() -> None:
    runner, collector = _make_runner()
    chunk = TokenChunk(
        model=MODEL_A_ID,
        text="hello",
        token_id=0,
        usage=None,
        finish_reason="stop",
    )

    runner.send_chunk(chunk, COMMAND_ID)

    assert _instance_tokens_events(collector.events) == []


def test_tool_call_chunk_emits_usage() -> None:
    runner, collector = _make_runner()
    chunk = ToolCallChunk(
        model=MODEL_A_ID,
        tool_calls=[],
        usage=_usage(prompt=8, completion=3),
    )

    runner.send_chunk(chunk, COMMAND_ID)

    emitted = _instance_tokens_events(collector.events)
    assert len(emitted) == 1
    assert emitted[0].prompt_tokens == 8
    assert emitted[0].completion_tokens == 3


def test_tool_call_chunk_without_usage_does_not_emit() -> None:
    runner, collector = _make_runner()
    chunk = ToolCallChunk(model=MODEL_A_ID, tool_calls=[], usage=None)

    runner.send_chunk(chunk, COMMAND_ID)

    assert _instance_tokens_events(collector.events) == []


def test_prefill_progress_chunk_does_not_emit() -> None:
    runner, collector = _make_runner()
    chunk = PrefillProgressChunk(model=MODEL_A_ID, processed_tokens=4, total_tokens=12)

    runner.send_chunk(chunk, COMMAND_ID)

    assert _instance_tokens_events(collector.events) == []


def test_chunk_generated_always_emitted_alongside_usage() -> None:
    """The original ChunkGenerated event must still be sent for every chunk."""
    runner, collector = _make_runner()
    chunk = TokenChunk(
        model=MODEL_A_ID,
        text="hello",
        token_id=0,
        usage=_usage(prompt=12, completion=5),
        finish_reason="stop",
    )

    runner.send_chunk(chunk, COMMAND_ID)

    chunk_events = [e for e in collector.events if isinstance(e, ChunkGenerated)]
    assert len(chunk_events) == 1
    assert chunk_events[0].chunk is chunk
