# pyright: reportAny=false, reportPrivateUsage=false, reportAttributeAccessIssue=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""Live per-session token accounting: helpers and the real chunk stream."""

import asyncio
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import MonkeyPatch

import exo.api.main as main_module
from exo.api.main import API
from exo.api.types import (
    CompletionTokensDetails,
    PromptTokensDetails,
    Usage,
)
from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import ErrorChunk, PrefillProgressChunk, TokenChunk
from exo.shared.types.common import CommandId
from exo.shared.types.worker.instances import InstanceId

_TEST_MODEL = ModelId("test-model")


def _usage(prompt: int, completion: int) -> Usage:
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details=PromptTokensDetails(),
        completion_tokens_details=CompletionTokensDetails(),
    )


class _FakeInstance:
    """Duck-typed stand-in for Instance (only model_id is needed)."""

    def __init__(self, model_id: str) -> None:
        self.shard_assignments = SimpleNamespace(model_id=ModelId(model_id))


def _make_api() -> API:
    api = object.__new__(API)
    api._session_token_usage = {}
    api._session_inflight = {}
    api._command_inflight_usage = {}
    api._lifetime_usage = {}
    api._text_generation_queues = {}
    api._image_generation_queues = {}
    api.state = SimpleNamespace(tasks={}, instances={})
    api._send = AsyncMock()
    api.command_sender = MagicMock()
    api.command_sender.send = AsyncMock()
    api._system_id = "sys"
    return api


class TestSessionIdFromRequest:
    def _request(self, headers: dict[str, str]) -> MagicMock:
        request = MagicMock()
        request.headers = MagicMock()
        request.headers.get = lambda key, default=None: headers.get(key, default)
        return request

    def test_x_pi_session_id_wins(self):
        api = _make_api()
        request = self._request(
            {"x-pi-session-id": "abc", "x-agent-id": "other", "x-session-id": "s"}
        )
        assert api._session_id_from_request(request) == "abc"

    def test_fallback_headers(self):
        api = _make_api()
        assert api._session_id_from_request(self._request({"x-agent-id": "a1"})) == "a1"
        assert (
            api._session_id_from_request(self._request({"x-session-id": "s1"})) == "s1"
        )

    def test_no_header_falls_back_to_dashboard(self):
        api = _make_api()
        assert api._session_id_from_request(self._request({})) == "dashboard"


class TestRecordAndExpose:
    async def test_record_fold_and_expose(self):
        api = _make_api()
        instance_id = InstanceId("inst-1")
        api._record_session_tokens(instance_id, "s-A", _usage(100, 50))
        api._record_session_tokens(instance_id, "s-A", _usage(30, 20))
        api._record_session_tokens(instance_id, "s-B", _usage(10, 5))

        result = await api.get_session_usage()
        by_session = {
            entry["sessionId"]: entry for entry in result["instances"]["inst-1"]
        }
        assert by_session["s-A"]["promptTokens"] == 130
        assert by_session["s-A"]["completionTokens"] == 70
        assert by_session["s-A"]["requestCount"] == 2
        assert by_session["s-B"]["requestCount"] == 1
        assert by_session["s-A"]["lastActiveSecondsAgo"] is not None
        assert by_session["s-A"]["lastActiveSecondsAgo"] < 5

    async def test_inflight_exposed_per_session(self):
        api = _make_api()
        instance_id = InstanceId("inst-1")
        api._record_session_tokens(instance_id, "s-A", _usage(1, 1))
        api._session_inflight["s-A"] = 2
        result = await api.get_session_usage()
        entry = next(
            e for e in result["instances"]["inst-1"] if e["sessionId"] == "s-A"
        )
        assert entry["inFlight"] == 2

    async def test_max_sessions_bound(self):
        api = _make_api()
        instance_id = InstanceId("inst-1")
        for i in range(250):
            api._record_session_tokens(instance_id, f"s-{i}", _usage(1, 1))
        assert len(api._session_token_usage[instance_id]) <= 201


class TestLifetimeUsage:
    def test_lifetime_key_uses_type_and_model(self):
        api = _make_api()
        assert api._lifetime_key(_FakeInstance("m-1")) == "_FakeInstance::m-1"

    def test_fold_persist_and_load_roundtrip(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ):
        lifetime_file = tmp_path / "lifetime_token_usage.json"
        monkeypatch.setattr(main_module, "_LIFETIME_USAGE_FILE", lifetime_file)

        api = _make_api()
        api._lifetime_usage = api._load_lifetime_usage()
        assert api._lifetime_usage == {}

        instance = _FakeInstance("m-1")
        api._fold_lifetime_usage(instance, 100, 40)
        api._fold_lifetime_usage(instance, 10, 5)
        api._fold_lifetime_usage(_FakeInstance("m-2"), 7, 3)
        assert lifetime_file.exists()

        reloaded = _make_api()
        reloaded._lifetime_usage = reloaded._load_lifetime_usage()
        assert reloaded._lifetime_usage["_FakeInstance::m-1"].prompt_tokens == 110
        assert reloaded._lifetime_usage["_FakeInstance::m-1"].completion_tokens == 45
        assert reloaded._lifetime_usage["_FakeInstance::m-1"].request_count == 2
        assert reloaded._lifetime_usage["_FakeInstance::m-2"].request_count == 1

    def test_load_tolerates_missing_and_corrupt_file(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ):
        monkeypatch.setattr(
            main_module, "_LIFETIME_USAGE_FILE", tmp_path / "missing.json"
        )
        api = _make_api()
        assert api._load_lifetime_usage() == {}

        corrupt = tmp_path / "corrupt.json"
        corrupt.write_text("{not json")
        monkeypatch.setattr(main_module, "_LIFETIME_USAGE_FILE", corrupt)
        assert api._load_lifetime_usage() == {}

    async def test_lifetime_exposed_per_current_instance(self):
        api = _make_api()
        instance = _FakeInstance("m-1")
        api.state = SimpleNamespace(
            tasks={},
            instances={InstanceId("inst-1"): instance},
            instance_lifecycle={},
        )
        api._fold_lifetime_usage(instance, 100, 40)
        result = await api.get_session_usage()
        assert result["lifetime"]["inst-1"] == {
            "promptTokens": 100,
            "completionTokens": 40,
            "requestCount": 1,
            "loadCount": 0,
            "unloadCount": 0,
        }


class TestTokenChunkStreamAccounting:
    """Drive the real _token_chunk_stream with a feeder task."""

    @staticmethod
    async def _run_stream(
        api: API,
        command_id: CommandId,
        chunks: Sequence[PrefillProgressChunk | TokenChunk | ErrorChunk],
        session_id: str | None,
        task: MagicMock,
    ) -> None:
        api.state = SimpleNamespace(tasks={f"t-{command_id}": task}, instances={})

        async def feeder() -> None:
            for _ in range(500):
                if command_id in api._text_generation_queues:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("queue never registered")
            sender = api._text_generation_queues[command_id]
            assert sender is not None
            for chunk in chunks:
                await sender.send(chunk)
            sender.close()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(feeder())
            gen = api._token_chunk_stream(command_id, session_id=session_id)
            async for _ in gen:
                pass

    @staticmethod
    def _task_for(command_id: CommandId, instance: str) -> MagicMock:
        task = MagicMock()
        task.command_id = command_id
        task.instance_id = InstanceId(instance)
        return task

    async def test_records_usage_and_clears_inflight(self):
        api = _make_api()
        command_id = CommandId("cmd-1")

        chunks = [
            PrefillProgressChunk(model=_TEST_MODEL, processed_tokens=1, total_tokens=5),
            TokenChunk(model=_TEST_MODEL, token_id=1, text="hello", usage=None),
            TokenChunk(
                model=_TEST_MODEL,
                token_id=2,
                text=" world",
                usage=_usage(120, 45),
                finish_reason="stop",
            ),
        ]
        await self._run_stream(
            api, command_id, chunks, "pi-A", self._task_for(command_id, "inst-1")
        )

        entry = api._session_token_usage[InstanceId("inst-1")]["pi-A"]
        assert entry.prompt_tokens == 120
        assert entry.completion_tokens == 45
        assert entry.request_count == 1
        assert "pi-A" not in api._session_inflight
        assert api._command_inflight_usage == {}

    async def test_streaming_consumer_stop_at_final_chunk_still_records(self):
        """Regression: streaming adapters return right after the terminating
        chunk and never resume the inner generator, so usage attribution and
        in-flight bookkeeping must happen before that chunk is yielded."""
        api = _make_api()
        command_id = CommandId("cmd-stream")
        api.state = SimpleNamespace(
            tasks={"t": self._task_for(command_id, "inst-1")}, instances={}
        )
        chunks = [
            PrefillProgressChunk(model=_TEST_MODEL, processed_tokens=1, total_tokens=9),
            TokenChunk(model=_TEST_MODEL, token_id=1, text="hi", usage=None),
            TokenChunk(
                model=_TEST_MODEL,
                token_id=2,
                text="!",
                usage=_usage(9, 2),
                finish_reason="stop",
            ),
        ]

        async def feeder() -> None:
            for _ in range(500):
                if command_id in api._text_generation_queues:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("queue never registered")
            sender = api._text_generation_queues[command_id]
            assert sender is not None
            for chunk in chunks:
                await sender.send(chunk)
            sender.close()

        gen = api._token_chunk_stream(command_id, session_id="pi-A")
        async with asyncio.TaskGroup() as tg:
            tg.create_task(feeder())
            async for chunk in gen:
                if getattr(chunk, "finish_reason", None) is not None:
                    break  # stop consuming exactly like generate_chat_stream
        await gen.aclose()

        entry = api._session_token_usage[InstanceId("inst-1")]["pi-A"]
        assert entry.prompt_tokens == 9
        assert entry.completion_tokens == 2
        assert entry.request_count == 1
        assert "pi-A" not in api._session_inflight
        assert api._command_inflight_usage == {}

    async def test_live_counters_visible_mid_stream(self):
        """While a request is streaming, /session-usage exposes approximate
        running counters for it, even before any request has completed."""
        api = _make_api()
        command_id = CommandId("cmd-live")
        api.state = SimpleNamespace(
            tasks={"t": self._task_for(command_id, "inst-1")}, instances={}
        )
        release = asyncio.Event()

        async def feeder() -> None:
            for _ in range(500):
                if command_id in api._text_generation_queues:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("queue never registered")
            sender = api._text_generation_queues[command_id]
            assert sender is not None
            await sender.send(
                PrefillProgressChunk(
                    model=_TEST_MODEL, processed_tokens=3, total_tokens=500
                )
            )
            await sender.send(
                TokenChunk(model=_TEST_MODEL, token_id=1, text="a", usage=None)
            )
            await sender.send(
                TokenChunk(model=_TEST_MODEL, token_id=2, text="b", usage=None)
            )
            await release.wait()
            await sender.send(
                TokenChunk(
                    model=_TEST_MODEL,
                    token_id=3,
                    text="c",
                    usage=_usage(500, 3),
                    finish_reason="stop",
                )
            )
            sender.close()

        gen = api._token_chunk_stream(command_id, session_id="pi-A")
        async with asyncio.TaskGroup() as tg:
            tg.create_task(feeder())
            await gen.__anext__()  # prefill progress
            await gen.__anext__()  # token 1
            await gen.__anext__()  # token 2

            result = await api.get_session_usage()
            (entry,) = result["instances"]["inst-1"]
            assert entry["sessionId"] == "pi-A"
            assert entry["promptTokens"] == 0  # nothing recorded yet
            assert entry["inFlight"] == 1
            assert entry["inFlightPromptTokens"] == 500
            assert entry["inFlightCompletionTokens"] == 2

            release.set()
            await gen.__anext__()  # final chunk
            with pytest.raises(StopAsyncIteration):
                await gen.__anext__()

        entry = api._session_token_usage[InstanceId("inst-1")]["pi-A"]
        assert entry.prompt_tokens == 500
        assert entry.completion_tokens == 3
        assert "pi-A" not in api._session_inflight
        assert api._command_inflight_usage == {}

    async def test_abort_mid_stream_clears_live_counters_without_recording(self):
        """Client disconnect before a terminating chunk: no usage attributed,
        but in-flight bookkeeping is cleaned up."""
        api = _make_api()
        command_id = CommandId("cmd-abort")
        api.state = SimpleNamespace(
            tasks={"t": self._task_for(command_id, "inst-1")}, instances={}
        )
        chunks = [
            TokenChunk(model=_TEST_MODEL, token_id=1, text="a", usage=None),
            TokenChunk(model=_TEST_MODEL, token_id=2, text="b", usage=None),
        ]

        async def feeder() -> None:
            for _ in range(500):
                if command_id in api._text_generation_queues:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("queue never registered")
            sender = api._text_generation_queues[command_id]
            assert sender is not None
            for chunk in chunks:
                await sender.send(chunk)
            await asyncio.sleep(10)  # keep the channel open; test aborts first

        gen = api._token_chunk_stream(command_id, session_id="pi-A")
        async with asyncio.TaskGroup() as tg:
            feeder_task = tg.create_task(feeder())
            await gen.__anext__()  # token 1
            await gen.aclose()
            feeder_task.cancel()

        assert api._session_token_usage == {}
        assert "pi-A" not in api._session_inflight
        assert api._command_inflight_usage == {}

    async def test_untracked_session_records_nothing(self):
        api = _make_api()
        command_id = CommandId("cmd-x")
        chunks = [
            TokenChunk(
                model=_TEST_MODEL,
                token_id=1,
                text="x",
                usage=_usage(5, 5),
                finish_reason="stop",
            ),
        ]
        await self._run_stream(
            api, command_id, chunks, None, self._task_for(command_id, "inst-9")
        )
        assert api._session_token_usage == {}
        assert api._session_inflight == {}
        assert api._command_inflight_usage == {}

    async def test_error_chunk_records_nothing(self):
        api = _make_api()
        command_id = CommandId("cmd-err")
        chunks = [ErrorChunk(model=_TEST_MODEL, error_message="boom")]
        await self._run_stream(
            api, command_id, chunks, "pi-A", self._task_for(command_id, "inst-err")
        )
        assert api._session_token_usage == {}
        assert "pi-A" not in api._session_inflight

    async def test_unknown_instance_records_nothing(self):
        api = _make_api()
        command_id = CommandId("cmd-unk")
        task = MagicMock()
        task.command_id = CommandId("not-this-command")
        task.instance_id = InstanceId("inst-9")
        chunks = [
            TokenChunk(
                model=_TEST_MODEL,
                token_id=1,
                text="x",
                usage=_usage(5, 5),
                finish_reason="stop",
            ),
        ]
        await self._run_stream(api, command_id, chunks, "pi-A", task)
        assert api._session_token_usage == {}
