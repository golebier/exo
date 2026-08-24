# pyright: reportUnusedFunction=false, reportAny=false
"""Tests for the decode/prefill stall watchdog in ``_token_chunk_stream``.

The watchdog bounds the wait for the next chunk so a hung mlx ``Fence::wait``
collective (which blocks the runner's C++ thread) cannot hang the API client
forever.  See ``EXO_DECODE_STALL_TIMEOUT`` / ``EXO_PREFILL_STALL_TIMEOUT``.
"""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock

import anyio
import pytest

from exo.api import main as api_main
from exo.api.main import API
from exo.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import CommandId, ModelId

_TEST_MODEL = ModelId("test-model")
# Full chunk union matching the API's ``_text_generation_queues`` value type.
_Chunk = TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk


def _make_api() -> API:
    """Create a minimal API instance for exercising ``_token_chunk_stream``."""
    api = object.__new__(API)
    api._text_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._send = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    api.command_sender = AsyncMock()
    return api


async def _get_sender(api: API, command_id: CommandId) -> Any:
    """Wait for ``_token_chunk_stream`` to create the channel, return sender."""
    while command_id not in api._text_generation_queues:  # pyright: ignore[reportPrivateUsage]
        await anyio.sleep(0.001)
    return api._text_generation_queues[command_id]  # pyright: ignore[reportPrivateUsage]


async def _drain(stream: AsyncGenerator[_Chunk, None]) -> list[_Chunk]:
    out: list[_Chunk] = []
    async for chunk in stream:
        out.append(chunk)
    return out


@pytest.mark.parametrize(
    ("monkeypatch_const", "phase"),
    [
        ("EXO_DECODE_STALL_TIMEOUT", "decode"),
        ("EXO_PREFILL_STALL_TIMEOUT", "prefill"),
    ],
)
async def test_stall_watchdog_fails_request_when_no_chunk_arrives(
    monkeypatch: pytest.MonkeyPatch, monkeypatch_const: str, phase: str
) -> None:
    """If no chunk arrives within the timeout, an ErrorChunk is yielded."""
    monkeypatch.setattr(api_main, monkeypatch_const, 0.05)

    api = _make_api()
    command_id = CommandId("stall-cmd")
    results: list[_Chunk] = []

    async def _produce() -> None:
        sender = await _get_sender(api, command_id)
        # Decode phase: send one token so decode_started=True, then stall.
        # Prefill phase: send nothing (stall immediately).
        if phase == "decode":
            await sender.send(
                TokenChunk(model=_TEST_MODEL, token_id=1, text="x", usage=None)
            )

    async def _consume() -> None:
        results.extend(
            await _drain(api._token_chunk_stream(command_id))  # pyright: ignore[reportPrivateUsage]
        )

    async with anyio.create_task_group() as tg:
        tg.start_soon(_produce)
        tg.start_soon(_consume)

    error_chunks = [c for c in results if isinstance(c, ErrorChunk)]
    assert len(error_chunks) == 1, f"expected one ErrorChunk, got {results!r}"
    assert "stalled" in error_chunks[0].error_message
    assert phase in error_chunks[0].error_message


async def test_stall_watchdog_disabled_when_timeout_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When both timeouts are 0, no watchdog fires; stream respects finish."""
    monkeypatch.setattr(api_main, "EXO_DECODE_STALL_TIMEOUT", 0.0)
    monkeypatch.setattr(api_main, "EXO_PREFILL_STALL_TIMEOUT", 0.0)

    api = _make_api()
    command_id = CommandId("no-stall-cmd")
    results: list[_Chunk] = []

    async def _produce() -> None:
        sender = await _get_sender(api, command_id)
        await sender.send(
            TokenChunk(
                model=_TEST_MODEL,
                token_id=1,
                text="hello",
                usage=None,
                finish_reason="stop",
            )
        )

    async def _consume() -> None:
        results.extend(
            await _drain(api._token_chunk_stream(command_id))  # pyright: ignore[reportPrivateUsage]
        )

    async with anyio.create_task_group() as tg:
        tg.start_soon(_produce)
        tg.start_soon(_consume)

    assert len(results) == 1
    assert isinstance(results[0], TokenChunk)
    assert results[0].finish_reason == "stop"


async def test_stall_watchdog_does_not_fire_during_normal_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunks arriving well within the timeout do not trigger the watchdog."""
    monkeypatch.setattr(api_main, "EXO_DECODE_STALL_TIMEOUT", 5.0)
    monkeypatch.setattr(api_main, "EXO_PREFILL_STALL_TIMEOUT", 5.0)

    api = _make_api()
    command_id = CommandId("normal-cmd")
    results: list[_Chunk] = []

    async def _produce() -> None:
        sender = await _get_sender(api, command_id)
        await sender.send(
            PrefillProgressChunk(
                model=_TEST_MODEL, processed_tokens=10, total_tokens=20
            )
        )
        await anyio.sleep(0.01)
        await sender.send(
            TokenChunk(model=_TEST_MODEL, token_id=1, text="a", usage=None)
        )
        await anyio.sleep(0.01)
        await sender.send(
            TokenChunk(
                model=_TEST_MODEL,
                token_id=2,
                text="b",
                usage=None,
                finish_reason="stop",
            )
        )

    async def _consume() -> None:
        results.extend(
            await _drain(api._token_chunk_stream(command_id))  # pyright: ignore[reportPrivateUsage]
        )

    async with anyio.create_task_group() as tg:
        tg.start_soon(_produce)
        tg.start_soon(_consume)

    error_chunks = [c for c in results if isinstance(c, ErrorChunk)]
    assert error_chunks == [], f"no ErrorChunk expected, got {results!r}"
    token_chunks = [c for c in results if isinstance(c, TokenChunk)]
    assert len(token_chunks) == 2
