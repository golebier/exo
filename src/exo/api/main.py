import base64
import contextlib
import hashlib
import json
import os
import random
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from typing import Annotated, Any, Literal, cast
from uuid import uuid4

import anyio
from anyio import BrokenResourceError, ClosedResourceError
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from hypercorn.asyncio import serve  # pyright: ignore[reportUnknownVariableType]
from hypercorn.config import Config
from hypercorn.typing import ASGIFramework
from hypercorn.utils import LifespanTimeoutError, ShutdownError
from loguru import logger

from exo.api.adapters.chat_completions import (
    chat_request_to_text_generation,
    collect_chat_response,
    generate_chat_stream,
)
from exo.api.adapters.claude import (
    claude_request_to_text_generation,
    collect_claude_response,
    generate_claude_stream,
)
from exo.api.adapters.ollama import (
    collect_ollama_chat_response,
    collect_ollama_generate_response,
    generate_ollama_chat_stream,
    generate_ollama_generate_stream,
    ollama_generate_request_to_text_generation,
    ollama_request_to_text_generation,
)
from exo.api.adapters.responses import (
    collect_responses_response,
    generate_responses_stream,
    responses_request_to_text_generation,
)
from exo.api.keepalive import with_sse_keepalive
from exo.api.launch_profile import (
    LaunchProfileEntry,
    load_launch_profile,
    remove_from_launch_profile,
    upsert_launch_profile,
)
from exo.api.types import (
    AddCustomModelParams,
    AdvancedImageParams,
    AwaitInstanceReadyMessage,
    AwaitInstanceTimeoutMessage,
    BenchChatCompletionRequest,
    BenchChatCompletionResponse,
    BenchImageGenerationResponse,
    BenchImageGenerationTaskParams,
    CancelCommandResponse,
    CancelDownloadParams,
    CancelDownloadResponse,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CreateInstanceParams,
    CreateInstanceResponse,
    DeleteDownloadResponse,
    DeleteInstanceResponse,
    DeleteTracesRequest,
    DeleteTracesResponse,
    ErrorInfo,
    ErrorResponse,
    FinishReason,
    GenerationStats,
    HuggingFaceSearchResult,
    IdleUnloadStatus,
    ImageData,
    ImageEditsTaskParams,
    ImageGenerationResponse,
    ImageGenerationStats,
    ImageGenerationTaskParams,
    ImageListItem,
    ImageListResponse,
    ImageSize,
    InstanceLinkBody,
    InstanceLinkResponse,
    MemoryGuardSetting,
    ModelList,
    ModelListModel,
    PlaceInstanceParams,
    PlacementPreview,
    PlacementPreviewResponse,
    SetIdleUnloadParams,
    StartDownloadParams,
    StartDownloadResponse,
    TieredCacheSetting,
    ToolCall,
    TraceCategoryStats,
    TraceEventResponse,
    TraceListItem,
    TraceListResponse,
    TraceRankStats,
    TraceResponse,
    TraceStatsResponse,
    TurboQuantSetting,
    Usage,
    normalize_image_size,
)
from exo.api.types.claude_api import (
    ClaudeMessagesRequest,
    ClaudeMessagesResponse,
)
from exo.api.types.ollama_api import (
    OllamaCapability,
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaModelDetails,
    OllamaModelTag,
    OllamaPsModel,
    OllamaPsResponse,
    OllamaShowRequest,
    OllamaShowResponse,
    OllamaTagsResponse,
)
from exo.api.types.openai_responses import (
    ResponsesRequest,
    ResponsesResponse,
)
from exo.master.image_store import ImageStore
from exo.master.placement import place_instance as get_instance_placements
from exo.shared.apply import apply
from exo.shared.constants import (
    DASHBOARD_DIR,
    ENABLE_DISAGGREGATION,
    EXO_APP_VERSION,
    EXO_CACHE_HOME,
    EXO_EVENT_LOG_DIR,
    EXO_IMAGE_CACHE_DIR,
    EXO_MAX_CHUNK_SIZE,
    EXO_TRACING_CACHE_DIR,
)
from exo.shared.election import ElectionMessage
from exo.shared.logging import InterceptLogger
from exo.shared.models import model_cards
from exo.shared.models.model_cards import (
    ModelCard,
    ModelId,
    ModelTask,
)
from exo.shared.tracing import TraceEvent, compute_stats, export_trace, load_trace_file
from exo.shared.types.chunks import (
    ErrorChunk,
    ImageChunk,
    InputImageChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.commands import (
    AddCustomModelCard,
    CancelDownload,
    Command,
    CreateInstance,
    DeleteCustomModelCard,
    DeleteDownload,
    DeleteInstance,
    DeleteInstanceLink,
    DownloadCommand,
    ForwarderCommand,
    ForwarderDownloadCommand,
    ImageEdits,
    ImageGeneration,
    LoadInstanceWeights,
    PlaceInstance,
    SendInputChunk,
    SetInstanceIdlePolicy,
    SetInstanceLink,
    StartDownload,
    TaskCancelled,
    TaskFinished,
    TextGeneration,
)
from exo.shared.types.common import CommandId, Id, NodeId, SystemId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    IndexedEvent,
    InstanceCreated,
    InstanceDeleted,
    InstanceTokensUpdated,
    TracesMerged,
)
from exo.shared.types.instance_lifecycle import InstanceIdlePolicy
from exo.shared.types.instance_link import InstanceLink, InstanceLinkId
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    ImageEdits as ImageEditsTask,
)
from exo.shared.types.tasks import (
    ImageGeneration as ImageGenerationTask,
)
from exo.shared.types.tasks import (
    TaskStatus,
)
from exo.shared.types.tasks import (
    TextGeneration as TextGenerationTask,
)
from exo.shared.types.text_generation import (
    Base64ImageHash,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.downloads import DownloadCompleted
from exo.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceMeta,
    MlxJacclInstance,
)
from exo.shared.types.worker.shards import (
    CfgShardMetadata,
    PipelineShardMetadata,
    Sharding,
)
from exo.utils.banner import print_startup_banner
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.disk_event_log import DiskEventLog
from exo.utils.power_sampler import PowerSampler
from exo.utils.task_group import TaskGroup
from exo.worker.engines.mlx import memory_guard, turboquant

_API_EVENT_LOG_DIR = EXO_EVENT_LOG_DIR / "api"
ONBOARDING_COMPLETE_FILE = EXO_CACHE_HOME / "onboarding_complete"


def _format_to_content_type(image_format: Literal["png", "jpeg", "webp"] | None) -> str:
    return f"image/{image_format or 'png'}"


def _ensure_seed(params: AdvancedImageParams | None) -> AdvancedImageParams:
    """Ensure advanced params has a seed set for distributed consistency."""
    if params is None:
        return AdvancedImageParams(seed=random.randint(0, 2**32 - 1))
    if params.seed is None:
        return params.model_copy(update={"seed": random.randint(0, 2**32 - 1)})
    return params


def _require_disaggregation_enabled() -> None:
    if not ENABLE_DISAGGREGATION:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=(
                "Prefill/decode disaggregation is disabled. "
                "Set ENABLE_DISAGGREGATION=true to enable."
            ),
        )


# Fallback session id for requests without a session header (e.g. dashboard chats).
_DEFAULT_SESSION_ID = "dashboard"
# Upper bound on tracked sessions per instance so the in-memory map stays small.
_MAX_TRACKED_SESSIONS_PER_INSTANCE = 200
# Durable all-time token totals, persisted across restarts. Event-sourced
# per-instance totals reset whenever an instance is recreated (restart,
# topology change); this sidecar keeps the true lifetime numbers per
# (instance type, model).
_LIFETIME_USAGE_FILE = _API_EVENT_LOG_DIR / "lifetime_token_usage.json"


@dataclass
class _SessionTokenUsage:
    """Live token accounting for one session on one instance (in-memory)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    request_count: int = 0
    last_active_ts: float = 0.0


@dataclass
class _CommandInflightUsage:
    """Running counters for one in-flight request.

    Approximate until the request completes: completion tokens are counted
    one per TokenChunk and the prompt estimate comes from prefill progress.
    The exact totals land in the session bucket on the terminating chunk.
    """

    session_id: str
    instance_id: InstanceId | None = None
    prompt_tokens_estimate: int = 0
    completion_tokens: int = 0


@dataclass
class _LifetimeUsage:
    """All-time token totals for one (instance type, model) pair."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    request_count: int = 0


DEFAULT_IDLE_UNLOAD_SECONDS = 1800


def _env_idle_unload_seconds() -> int | None:
    """Cluster-wide default idle-unload timeout (seconds).

    Default is 30 minutes: instances idle longer than this get their weights
    freed from memory (the instance itself stays). Override with
    ``EXO_IDLE_UNLOAD_SECONDS``; ``0`` disables the default entirely.
    """
    raw = os.environ.get("EXO_IDLE_UNLOAD_SECONDS")
    if raw is None or raw == "":
        return DEFAULT_IDLE_UNLOAD_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_IDLE_UNLOAD_SECONDS
    return value if value >= 5 else None


class API:
    def __init__(
        self,
        node_id: NodeId,
        *,
        port: int,
        event_receiver: Receiver[IndexedEvent],
        command_sender: Sender[ForwarderCommand],
        download_command_sender: Sender[ForwarderDownloadCommand],
        # This lets us pause the API if an election is running
        election_receiver: Receiver[ElectionMessage],
    ) -> None:
        self.state = State()
        self._event_log = DiskEventLog(_API_EVENT_LOG_DIR)
        self._system_id = SystemId()
        self.command_sender = command_sender
        self.download_command_sender = download_command_sender
        self.event_receiver = event_receiver
        self.election_receiver = election_receiver
        self.node_id: NodeId = node_id
        self.last_completed_election: int = 0
        self.port = port
        self._sent_image_hashes: set[str] = set()
        # Live per-session token accounting (in-memory). Sessions are
        # identified via the x-pi-session-id header ("dashboard" by default).
        # The Σ line's all-time totals come from the persisted sidecar store
        # below, because the event-sourced per-instance counters reset
        # whenever an instance is recreated.
        self._session_token_usage: dict[InstanceId, dict[str, _SessionTokenUsage]] = {}
        self._session_inflight: dict[str, int] = {}
        self._command_inflight_usage: dict[CommandId, _CommandInflightUsage] = {}
        # Durable lifetime totals (survive restarts / instance recreation).
        self._lifetime_usage: dict[str, _LifetimeUsage] = self._load_lifetime_usage()

        self.paused: bool = False
        self.paused_ev: anyio.Event = anyio.Event()

        self.app = FastAPI()

        @self.app.middleware("http")
        async def _log_requests(  # pyright: ignore[reportUnusedFunction]
            request: Request,
            call_next: Callable[[Request], Awaitable[StreamingResponse]],
        ) -> StreamingResponse:
            logger.debug(f"API request: {request.method} {request.url.path}")
            return await call_next(request)

        self._setup_exception_handlers()
        self._setup_cors()
        self._setup_routes()

        self.app.mount(
            "/",
            StaticFiles(
                directory=DASHBOARD_DIR,
                html=True,
            ),
            name="dashboard",
        )

        self._text_generation_queues: dict[
            CommandId,
            Sender[TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk],
        ] = {}
        self._image_generation_queues: dict[
            CommandId, Sender[ImageChunk | ErrorChunk]
        ] = {}
        self._image_store = ImageStore(EXO_IMAGE_CACHE_DIR)
        self._tg: TaskGroup = TaskGroup()

        # Idle-unload bookkeeping (API-side).
        # Pending PlaceInstance commands awaiting their InstanceCreated event,
        # keyed by model id: (policy, monotonic send time).
        self._pending_idle_policies: dict[str, tuple[InstanceIdlePolicy, float]] = {}
        # Durable launch profile — how instances were placed last session,
        # used by POST /launch-profile/restore to re-create them in one click.
        self._launch_profile: dict[str, LaunchProfileEntry] = load_launch_profile()
        # Last activity timestamp per instance (refreshed when busy).
        self._idle_last_activity: dict[InstanceId, float] = {}
        # Instance ids we deleted via idle-unload (for unload-reason attribution).
        self._idle_unload_inflight: set[InstanceId] = set()

    def reset(self, result_clock: int, event_receiver: Receiver[IndexedEvent]):
        logger.info("Resetting API State")
        self._event_log.close()
        self._event_log = DiskEventLog(_API_EVENT_LOG_DIR)
        self.state = State()
        self._system_id = SystemId()
        self._text_generation_queues = {}
        self._image_generation_queues = {}
        self.unpause(result_clock)
        self.event_receiver.close()
        self.event_receiver = event_receiver
        self._tg.start_soon(self._apply_state)
        self._sent_image_hashes = set()

    def unpause(self, result_clock: int):
        logger.info("Unpausing API")
        self.last_completed_election = result_clock
        self.paused = False
        self.paused_ev.set()
        self.paused_ev = anyio.Event()

    def _setup_exception_handlers(self) -> None:
        self.app.exception_handler(HTTPException)(self.http_exception_handler)

    async def http_exception_handler(
        self, _: Request, exc: HTTPException
    ) -> JSONResponse:
        err = ErrorResponse(
            error=ErrorInfo(
                message=exc.detail,
                type=HTTPStatus(exc.status_code).phrase,
                code=exc.status_code,
            )
        )
        return JSONResponse(err.model_dump(), status_code=exc.status_code)

    def _setup_cors(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self) -> None:
        self.app.get("/node_id")(lambda: self.node_id)
        self.app.post("/instance")(self.create_instance)
        self.app.post("/place_instance")(self.place_instance)
        self.app.get("/instance/placement")(self.get_placement)
        self.app.get("/instance/previews")(self.get_placement_previews)
        self.app.get("/instance/await", response_model=None)(self.await_instance)
        self.app.get("/instance/{instance_id}")(self.get_instance)
        self.app.delete("/instance/{instance_id}")(self.delete_instance)
        self.app.get("/instance/{instance_id}/idle-unload")(self.get_idle_unload)
        self.app.put("/instance/{instance_id}/idle-unload")(self.set_idle_unload)
        self.app.post("/instance/{instance_id}/load")(self.load_instance_weights)
        self.app.get("/launch-profile")(self.get_launch_profile)
        self.app.post("/launch-profile/restore")(self.restore_launch_profile)
        self.app.get("/v1/instance-links")(self.list_instance_links)
        self.app.post("/v1/instance-links")(self.create_instance_link)
        self.app.put("/v1/instance-links/{link_id}")(self.update_instance_link)
        self.app.delete("/v1/instance-links/{link_id}")(self.delete_instance_link)
        self.app.get("/v1/feature-flags")(self.get_feature_flags)
        self.app.put("/v1/memory-guard")(self.set_memory_guard)
        self.app.put("/v1/turboquant")(self.set_turboquant)
        self.app.put("/v1/tiered-cache")(self.set_tiered_cache)
        self.app.get("/v1/tiered-cache")(self.get_tiered_cache)
        self.app.delete("/v1/tiered-cache")(self.clear_tiered_cache)
        self.app.get("/v1/version")(self.get_version)
        self.app.get("/models")(self.get_models)
        self.app.get("/v1/models")(self.get_models)
        self.app.post("/models/add")(self.add_custom_model)
        self.app.delete("/models/custom/{model_id:path}")(self.delete_custom_model)
        self.app.get("/models/search")(self.search_models)
        self.app.post("/v1/chat/completions", response_model=None)(
            self.chat_completions
        )
        self.app.post("/bench/chat/completions", response_model=None)(
            self.bench_chat_completions
        )
        self.app.post("/v1/images/generations", response_model=None)(
            self.image_generations
        )
        self.app.post("/bench/images/generations")(self.bench_image_generations)
        self.app.post("/v1/images/edits", response_model=None)(self.image_edits)
        self.app.post("/bench/images/edits")(self.bench_image_edits)
        self.app.get("/images")(self.list_images)
        self.app.get("/images/{image_id}")(self.get_image)
        self.app.post("/v1/messages", response_model=None)(self.claude_messages)
        self.app.post("/v1/responses", response_model=None)(self.openai_responses)
        self.app.post("/v1/cancel/{command_id}")(self.cancel_command)

        # Ollama API
        self.app.head("/ollama/")(self.ollama_version)
        self.app.head("/ollama/api/version")(self.ollama_version)
        self.app.post("/ollama/v1/chat/completions", response_model=None)(
            self.chat_completions
        )
        self.app.post("/ollama/api/chat", response_model=None)(self.ollama_chat)
        self.app.post("/ollama/api/api/chat", response_model=None)(self.ollama_chat)
        self.app.post("/ollama/api/v1/chat", response_model=None)(self.ollama_chat)
        self.app.post("/ollama/api/generate", response_model=None)(self.ollama_generate)
        self.app.get("/ollama/api/tags")(self.ollama_tags)
        self.app.get("/ollama/api/api/tags")(self.ollama_tags)
        self.app.get("/ollama/api/v1/tags")(self.ollama_tags)
        self.app.post("/ollama/api/show")(self.ollama_show)
        self.app.get("/ollama/api/ps")(self.ollama_ps)
        self.app.get("/ollama/api/version")(self.ollama_version)

        self.app.get("/state")(self.get_state)
        self.app.get("/session-usage")(self.get_session_usage)
        self.app.get("/state/{path:path}")(self.get_state)
        self.app.get("/events")(self.stream_events)
        self.app.post("/download/start")(self.start_download)
        self.app.delete("/download/{node_id}/{model_id:path}")(self.delete_download)
        self.app.post("/download/cancel")(self.cancel_download)
        self.app.get("/v1/traces")(self.list_traces)
        self.app.post("/v1/traces/delete")(self.delete_traces)
        self.app.get("/v1/traces/{task_id}")(self.get_trace)
        self.app.get("/v1/traces/{task_id}/stats")(self.get_trace_stats)
        self.app.get("/v1/traces/{task_id}/raw")(self.get_trace_raw)
        self.app.get("/onboarding")(self.get_onboarding)
        self.app.post("/onboarding")(self.complete_onboarding)

    def get_state(self, path: str = ""):
        if path == "":
            return self.state
        try:
            x = self.state.model_dump(by_alias=True)
            for attr in path.split("/"):
                if attr != "":
                    if isinstance(x, dict):
                        x = x[attr]  # pyright: ignore[reportUnknownVariableType]
                    elif isinstance(x, list):
                        x = x[int(attr)]  # pyright: ignore[reportUnknownVariableType]
            return cast(Any, x)  # pyright: ignore[reportAny]
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"unable to find path '{path.replace('/', '.')}' in state json",
            ) from e

    async def place_instance(self, payload: PlaceInstanceParams):
        command = PlaceInstance(
            model_card=await ModelCard.load(payload.model_id),
            sharding=payload.sharding,
            instance_meta=payload.instance_meta,
            min_nodes=payload.min_nodes,
        )
        idle_unload_seconds = (
            payload.idle_unload_seconds
            if payload.idle_unload_seconds is not None
            else _env_idle_unload_seconds()
        )
        if idle_unload_seconds is not None:
            self._pending_idle_policies[str(payload.model_id)] = (
                InstanceIdlePolicy(
                    idle_unload_seconds=idle_unload_seconds,
                    sharding=str(payload.sharding.value),
                    instance_meta=str(payload.instance_meta.value),
                    min_nodes=payload.min_nodes,
                ),
                time.time(),
            )
        self._record_launch(
            LaunchProfileEntry(
                model_id=str(payload.model_id),
                sharding=str(payload.sharding.value),
                instance_meta=str(payload.instance_meta.value),
                min_nodes=payload.min_nodes,
                idle_unload_seconds=idle_unload_seconds,
                added_at=time.time(),
            )
        )
        await self._send(command)

        return CreateInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            model_card=command.model_card,
        )

    async def create_instance(
        self, payload: CreateInstanceParams
    ) -> CreateInstanceResponse:
        instance = payload.instance
        model_card = await ModelCard.load(instance.shard_assignments.model_id)
        required_memory = model_card.storage_size
        available_memory = self._calculate_total_available_memory()

        if required_memory > available_memory:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient memory to create instance. Required: {required_memory.in_gb:.1f}GB, Available: {available_memory.in_gb:.1f}GB",
            )

        command = CreateInstance(
            instance=instance,
        )
        self._record_launch(self._launch_recipe_from_instance(instance))
        await self._send(command)

        return CreateInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            model_card=model_card,
        )

    async def get_placement(
        self,
        model_id: ModelId,
        sharding: Sharding = Sharding.Pipeline,
        instance_meta: InstanceMeta = InstanceMeta.MlxRing,
        min_nodes: int = 1,
    ) -> Instance:
        model_card = await ModelCard.load(model_id)

        try:
            placements = get_instance_placements(
                PlaceInstance(
                    model_card=model_card,
                    sharding=sharding,
                    instance_meta=instance_meta,
                    min_nodes=min_nodes,
                ),
                node_memory=self.state.node_memory,
                node_network=self.state.node_network,
                node_backends=self.state.node_backends,
                topology=self.state.topology,
                current_instances=self.state.instances,
                download_status=self.state.downloads,
                node_rdma_ctl=self.state.node_rdma_ctl,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        current_ids = set(self.state.instances.keys())
        new_ids = [
            instance_id for instance_id in placements if instance_id not in current_ids
        ]
        if len(new_ids) != 1:
            raise HTTPException(
                status_code=500,
                detail="Expected exactly one new instance from placement",
            )

        return placements[new_ids[0]]

    async def get_placement_previews(
        self,
        model_id: ModelId,
        node_ids: Annotated[list[NodeId] | None, Query()] = None,
    ) -> PlacementPreviewResponse:
        seen: set[tuple[ModelId, Sharding, InstanceMeta, int]] = set()
        previews: list[PlacementPreview] = []
        required_nodes = set(node_ids) if node_ids else None

        if len(list(self.state.topology.list_nodes())) == 0:
            return PlacementPreviewResponse(previews=[])

        try:
            model_card = await ModelCard.load(model_id)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Failed to load model card: {exc}"
            ) from exc
        instance_combinations: list[tuple[Sharding, InstanceMeta, int]] = []
        for sharding in (Sharding.Pipeline, Sharding.Tensor):
            for instance_meta in (InstanceMeta.MlxRing, InstanceMeta.MlxJaccl):
                instance_combinations.extend(
                    [
                        (sharding, instance_meta, i)
                        for i in range(
                            1, len(list(self.state.topology.list_nodes())) + 1
                        )
                    ]
                )
        # TODO: PDD
        # instance_combinations.append((Sharding.PrefillDecodeDisaggregation, InstanceMeta.MlxRing, 1))

        for sharding, instance_meta, min_nodes in instance_combinations:
            try:
                placements = get_instance_placements(
                    PlaceInstance(
                        model_card=model_card,
                        sharding=sharding,
                        instance_meta=instance_meta,
                        min_nodes=min_nodes,
                    ),
                    node_memory=self.state.node_memory,
                    node_network=self.state.node_network,
                    node_backends=self.state.node_backends,
                    topology=self.state.topology,
                    current_instances=self.state.instances,
                    required_nodes=required_nodes,
                    download_status=self.state.downloads,
                    node_rdma_ctl=self.state.node_rdma_ctl,
                )
            except ValueError as exc:
                if (model_card.model_id, sharding, instance_meta, 0) not in seen:
                    previews.append(
                        PlacementPreview(
                            model_id=model_card.model_id,
                            sharding=sharding,
                            instance_meta=instance_meta,
                            instance=None,
                            error=str(exc),
                        )
                    )
                seen.add((model_card.model_id, sharding, instance_meta, 0))
                continue

            current_ids = set(self.state.instances.keys())
            new_instances = [
                instance
                for instance_id, instance in placements.items()
                if instance_id not in current_ids
            ]

            if len(new_instances) != 1:
                if (model_card.model_id, sharding, instance_meta, 0) not in seen:
                    previews.append(
                        PlacementPreview(
                            model_id=model_card.model_id,
                            sharding=sharding,
                            instance_meta=instance_meta,
                            instance=None,
                            error="Expected exactly one new instance from placement",
                        )
                    )
                seen.add((model_card.model_id, sharding, instance_meta, 0))
                continue

            instance = new_instances[0]
            shard_assignments = instance.shard_assignments
            placement_node_ids = list(shard_assignments.node_to_runner.keys())

            memory_delta_by_node: dict[str, int] = {}
            if placement_node_ids:
                total_bytes = model_card.storage_size.in_bytes
                per_node = total_bytes // len(placement_node_ids)
                remainder = total_bytes % len(placement_node_ids)
                for index, node_id in enumerate(sorted(placement_node_ids, key=str)):
                    extra = 1 if index < remainder else 0
                    memory_delta_by_node[str(node_id)] = per_node + extra

            if (
                model_card.model_id,
                sharding,
                instance_meta,
                len(placement_node_ids),
            ) not in seen:
                previews.append(
                    PlacementPreview(
                        model_id=model_card.model_id,
                        sharding=sharding,
                        instance_meta=instance_meta,
                        instance=instance,
                        memory_delta_by_node=memory_delta_by_node or None,
                        error=None,
                    )
                )
            seen.add(
                (
                    model_card.model_id,
                    sharding,
                    instance_meta,
                    len(placement_node_ids),
                )
            )

        return PlacementPreviewResponse(previews=previews)

    def get_instance(self, instance_id: InstanceId) -> dict[str, Any]:
        if instance_id not in self.state.instances:
            raise HTTPException(status_code=404, detail="Instance not found")
        instance = self.state.instances[instance_id]
        return {
            "instance": instance,
            "idleUnload": self._idle_unload_info(instance_id).model_dump(),
        }

    async def await_instance(
        self,
        model_id: ModelId,
        timeout_seconds: float = Query(default=0.0, ge=0.0, le=300.0),
    ) -> StreamingResponse:
        _sleep = 0.1

        async def _stream() -> AsyncGenerator[str, None]:
            deadline = (
                None if timeout_seconds == 0 else anyio.current_time() + timeout_seconds
            )

            while True:
                for instance in self.state.instances.values():
                    if instance.shard_assignments.model_id == model_id:
                        payload = AwaitInstanceReadyMessage(instance=instance)
                        yield f"data: {payload.model_dump_json()}\n\n"
                        return

                if deadline is None:
                    await anyio.sleep(_sleep)
                else:
                    remaining = deadline - anyio.current_time()
                    if remaining <= 0:
                        payload = AwaitInstanceTimeoutMessage(
                            message=f"No instance found for model {model_id}"
                        )
                        yield f"data: {payload.model_dump_json()}\n\n"
                        return

                    await anyio.sleep(min(_sleep, remaining))

        return StreamingResponse(
            with_sse_keepalive(_stream()),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "close",
                "X-Accel-Buffering": "no",
            },
        )

    async def delete_instance(self, instance_id: InstanceId) -> DeleteInstanceResponse:
        if instance_id not in self.state.instances:
            raise HTTPException(status_code=404, detail="Instance not found")

        instance = self.state.instances[instance_id]
        command = DeleteInstance(
            instance_id=instance_id,
        )
        await self._send(command)
        model_id = str(instance.shard_assignments.model_id)
        still_running = any(
            other.instance_id != instance_id
            and other.shard_assignments.model_id == instance.shard_assignments.model_id
            for other in self.state.instances.values()
        )
        if not still_running:
            self._launch_profile = remove_from_launch_profile(model_id)
        return DeleteInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            instance_id=instance_id,
        )

    def _idle_unload_info(self, instance_id: InstanceId) -> IdleUnloadStatus:
        """Idle-unload policy + lifecycle counters for one instance."""
        instance = self.state.instances.get(instance_id)
        if instance is None:
            raise HTTPException(status_code=404, detail="Instance not found")
        policy = self.state.instance_idle_policies.get(instance_id)
        key = f"{type(instance).__name__}::{instance.shard_assignments.model_id}"
        stats = self.state.instance_lifecycle.get(key)
        busy = self._instance_busy(instance_id)
        now = time.time()
        idle_for: float | None = None
        if not busy:
            idle_for = now - self._idle_last_activity.get(instance_id, now)
        return IdleUnloadStatus(
            policy_seconds=(policy.idle_unload_seconds if policy is not None else None),
            idle_seconds=idle_for,
            weights_unloaded=instance_id in self.state.instance_weights_unloaded,
            load_count=stats.load_count if stats is not None else 0,
            unload_count=stats.unload_count if stats is not None else 0,
            last_load_ts=stats.last_load_ts if stats is not None else None,
            last_unload_ts=stats.last_unload_ts if stats is not None else None,
            last_unload_reason=stats.last_unload_reason if stats is not None else None,
        )

    def _instance_busy(self, instance_id: InstanceId) -> bool:
        """An instance is busy while any of its tasks is pending/running."""
        return any(
            task.instance_id == instance_id
            and task.task_status in (TaskStatus.Pending, TaskStatus.Running)
            for task in self.state.tasks.values()
        )

    def get_idle_unload(self, instance_id: InstanceId) -> IdleUnloadStatus:
        return self._idle_unload_info(instance_id)

    async def set_idle_unload(
        self, instance_id: InstanceId, payload: SetIdleUnloadParams
    ) -> IdleUnloadStatus:
        if instance_id not in self.state.instances:
            raise HTTPException(status_code=404, detail="Instance not found")
        policy = InstanceIdlePolicy(idle_unload_seconds=payload.idle_unload_seconds)
        await self._send(SetInstanceIdlePolicy(instance_id=instance_id, policy=policy))
        # Optimistic local update so the response reflects the new policy;
        # the event-sourced state converges to the same value.
        self.state = self.state.model_copy(
            update={
                "instance_idle_policies": {
                    **self.state.instance_idle_policies,
                    instance_id: policy,
                }
            }
        )
        if payload.idle_unload_seconds is not None:
            self._idle_last_activity[instance_id] = time.time()
        # Keep the durable launch profile in sync so a later "restore last
        # session" re-applies the same idle-unload timeout.
        recipe = self._launch_recipe_from_instance(self.state.instances[instance_id])
        if str(recipe.model_id) in self._launch_profile:
            self._launch_profile = upsert_launch_profile(recipe)
        return self._idle_unload_info(instance_id)

    async def load_instance_weights(self, instance_id: InstanceId) -> IdleUnloadStatus:
        """Manually load an idle-unloaded instance's weights back into memory."""
        if instance_id not in self.state.instances:
            raise HTTPException(status_code=404, detail="Instance not found")
        if instance_id in self.state.instance_weights_unloaded:
            await self._send(LoadInstanceWeights(instance_id=instance_id))
        return self._idle_unload_info(instance_id)

    def get_launch_profile(self) -> dict[str, list[LaunchProfileEntry]]:
        return {"entries": list(self._launch_profile.values())}

    async def restore_launch_profile(self) -> dict[str, list[dict[str, str]]]:
        """One-click restore of the previous session: re-place every model in
        the durable launch profile that is not currently running, with the
        exact sharding/instance_meta/min_nodes (and idle-unload policy) it had."""
        results: list[dict[str, str]] = []
        for entry in list(self._launch_profile.values()):
            model_id = ModelId(entry.model_id)
            if any(
                instance.shard_assignments.model_id == model_id
                for instance in self.state.instances.values()
            ):
                results.append(
                    {"model_id": entry.model_id, "action": "already_running"}
                )
                continue
            try:
                model_card = await ModelCard.load(model_id)
            except Exception as exc:
                results.append(
                    {
                        "model_id": entry.model_id,
                        "action": "failed",
                        "detail": str(exc),
                    }
                )
                continue
            effective_idle = (
                entry.idle_unload_seconds
                if entry.idle_unload_seconds is not None
                else _env_idle_unload_seconds()
            )
            if effective_idle is not None:
                self._pending_idle_policies[str(model_id)] = (
                    InstanceIdlePolicy(
                        idle_unload_seconds=effective_idle,
                        sharding=entry.sharding,
                        instance_meta=entry.instance_meta,
                        min_nodes=entry.min_nodes,
                    ),
                    time.time(),
                )
            await self._send(
                PlaceInstance(
                    model_card=model_card,
                    sharding=Sharding[entry.sharding],
                    instance_meta=InstanceMeta[entry.instance_meta],
                    min_nodes=entry.min_nodes,
                )
            )
            logger.info(
                f"Launch-profile restore: placing {entry.model_id} "
                f"({entry.sharding}/{entry.instance_meta}/{entry.min_nodes} nodes"
                + (
                    f", idle_unload={effective_idle}s)"
                    if effective_idle is not None
                    else ")"
                )
            )
            results.append({"model_id": entry.model_id, "action": "placed"})
        return {"results": results}

    async def get_feature_flags(self) -> dict[str, bool]:
        return {
            "disaggregation": ENABLE_DISAGGREGATION,
            "prefillMemoryGuard": memory_guard.is_guard_enabled(),
            "turboquantKv": turboquant.is_turboquant_enabled(),
            "tieredKvCache": turboquant.is_tiered_cache_enabled(),
        }

    async def set_memory_guard(self, body: MemoryGuardSetting) -> dict[str, bool]:
        """Runtime toggle for the prefill memory guard (task #11).

        Takes effect on the next prefill without a restart. The env escape
        hatch ``EXO_DISABLE_PREFILL_GUARD=1`` still wins when the override is
        cleared, so a process started with the guard forced off cannot be
        re-enabled via the UI.
        """
        memory_guard.set_guard_enabled(body.enabled)
        return {"prefillMemoryGuard": memory_guard.is_guard_enabled()}

    async def set_turboquant(self, body: TurboQuantSetting) -> dict[str, object]:
        """Runtime toggle for TurboQuant KV-cache compression (#07).

        Mirrors oMLX's ``ModelSettings.turboquant_kv_*``. Takes effect on the
        next ``make_kv_cache`` call (next model load, or a clean prefill after
        the prefix cache is invalidated); a model already loaded with a
        different cache dtype is not retroactively requantized.
        """
        turboquant.set_turboquant_enabled(body.enabled)
        turboquant.set_turboquant_bits(body.bits)
        turboquant.set_turboquant_skip_last(body.skip_last)
        settings = turboquant.turboquant_settings()
        return {str(k): v for k, v in settings.items()}

    async def set_tiered_cache(self, body: TieredCacheSetting) -> dict[str, object]:
        """Runtime toggle for the tiered (Hot RAM / Cold SSD) KV cache (#01).

        Mirrors oMLX's ``CacheSettings``. The SSD spill/restore plumbing is
        staged behind this flag; the dir/size knobs are live so the dashboard
        reflects the configured state and the heavy paged-SSD work can land
        without further UI churn.
        """
        turboquant.set_tiered_cache_enabled(body.enabled)
        turboquant.set_hot_cache_only(body.hot_cache_only)
        turboquant.set_ssd_cache_dir(body.ssd_cache_dir)
        turboquant.set_ssd_cache_max_size(body.ssd_cache_max_size)
        turboquant.set_hot_cache_max_size(body.hot_cache_max_size)
        settings = turboquant.tiered_cache_settings()
        return {str(k): v for k, v in settings.items()}

    async def get_tiered_cache(self) -> dict[str, object]:
        """Live tiered-cache status for the dashboard observability block."""
        status = turboquant.tiered_cache_status()
        return {
            "enabled": status.enabled,
            "hotCacheOnly": status.hot_cache_only,
            "ssdCacheDir": status.ssd_cache_dir,
            "ssdCacheMaxSizeBytes": status.ssd_cache_max_size_bytes,
            "hotCacheMaxSizeBytes": status.hot_cache_max_size_bytes,
            "hotCacheEntries": status.hot_cache_entries,
            "hotCacheSizeBytes": status.hot_cache_size_bytes,
            "ssdCacheFiles": status.ssd_cache_files,
            "ssdCacheSizeBytes": status.ssd_cache_size_bytes,
            "ssdDiskCapacityBytes": status.ssd_disk_capacity_bytes,
            "basePath": status.base_path,
            "responseStateDir": status.response_state_dir,
        }

    async def clear_tiered_cache(self) -> dict[str, int]:
        """Delete every file in the SSD cache dir (mirrors oMLX's clear route).

        Returns the count of files removed. Safe to call whether or not the
        tier is enabled; a disabled tier simply has nothing to clear.
        """
        removed = turboquant.clear_ssd_cache()
        return {"removed": removed}

    async def get_version(self) -> dict[str, str]:
        """App build version for the dashboard top-left badge (task #11)."""
        return {"version": EXO_APP_VERSION}

    async def list_instance_links(self) -> list[InstanceLink]:
        if not ENABLE_DISAGGREGATION:
            return []
        return list(self.state.instance_links.values())

    async def create_instance_link(
        self, body: InstanceLinkBody
    ) -> InstanceLinkResponse:
        _require_disaggregation_enabled()
        return await self._set_instance_link(InstanceLinkId(), body)

    async def update_instance_link(
        self, link_id: InstanceLinkId, body: InstanceLinkBody
    ) -> InstanceLinkResponse:
        _require_disaggregation_enabled()
        return await self._set_instance_link(link_id, body)

    async def _set_instance_link(
        self, link_id: InstanceLinkId, body: InstanceLinkBody
    ) -> InstanceLinkResponse:
        command = SetInstanceLink(
            link_id=link_id,
            prefill_instances=list(body.prefill_instances),
            decode_instances=list(body.decode_instances),
        )
        await self._send(command)
        return InstanceLinkResponse(
            message="Command received.", command_id=command.command_id
        )

    async def delete_instance_link(
        self, link_id: InstanceLinkId
    ) -> InstanceLinkResponse:
        _require_disaggregation_enabled()
        command = DeleteInstanceLink(link_id=link_id)
        await self._send(command)
        return InstanceLinkResponse(
            message="Command received.", command_id=command.command_id
        )

    async def cancel_command(self, command_id: CommandId) -> CancelCommandResponse:
        """Cancel an active command by closing its stream and notifying workers."""
        sender = self._text_generation_queues.get(
            command_id
        ) or self._image_generation_queues.get(command_id)
        if sender is None:
            raise HTTPException(
                status_code=404,
                detail="Command not found or already completed",
            )

        await self._send(TaskCancelled(cancelled_command_id=command_id))
        sender.close()

        return CancelCommandResponse(
            message="Command cancelled.",
            command_id=command_id,
        )

    async def _token_chunk_stream(
        self, command_id: CommandId, session_id: str | None = None
    ) -> AsyncGenerator[
        TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk, None
    ]:
        """Yield chunks for a given command until completion.

        This is the internal low-level stream used by all API adapters. When
        ``session_id`` is given, the request is counted as in-flight for that
        session with live running counters, and the terminating chunk's usage
        is attributed to the (instance, session) bucket for live per-session
        accounting.
        """
        inflight: _CommandInflightUsage | None = None
        if session_id is not None:
            self._session_inflight[session_id] = (
                self._session_inflight.get(session_id, 0) + 1
            )
            inflight = _CommandInflightUsage(session_id=session_id)
            self._command_inflight_usage[command_id] = inflight
        settled = False
        try:
            self._text_generation_queues[command_id], recv = channel[
                TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk
            ]()

            with recv as token_chunks:
                async for chunk in token_chunks:
                    if inflight is not None:
                        if inflight.instance_id is None:
                            inflight.instance_id = self._instance_for_command(
                                command_id
                            )
                        if isinstance(chunk, PrefillProgressChunk):
                            inflight.prompt_tokens_estimate = max(
                                inflight.prompt_tokens_estimate, chunk.total_tokens
                            )
                        elif isinstance(chunk, TokenChunk):
                            inflight.completion_tokens += 1
                    if isinstance(chunk, PrefillProgressChunk):
                        yield chunk
                        continue
                    if chunk.finish_reason is not None:
                        # Settle everything BEFORE yielding the terminating
                        # chunk: streaming adapters stop consuming at the final
                        # chunk, so the generator may never resume after this
                        # yield. Usage attribution and in-flight bookkeeping
                        # must both happen while we still control the flow.
                        # Same termination semantics as the runner's token
                        # accounting: only the final TokenChunk / ToolCallChunk
                        # carries usage, exactly once per request.
                        if (
                            session_id is not None
                            and isinstance(chunk, (TokenChunk, ToolCallChunk))
                            and chunk.usage is not None
                        ):
                            instance_id = self._instance_for_command(command_id)
                            if instance_id is not None:
                                self._record_session_tokens(
                                    instance_id, session_id, chunk.usage
                                )
                        if session_id is not None:
                            self._settle_inflight(command_id, session_id)
                            settled = True
                        yield chunk
                        break
                    yield chunk

        except anyio.get_cancelled_exc_class():
            command = TaskCancelled(cancelled_command_id=command_id)
            with anyio.CancelScope(shield=True):
                await self.command_sender.send(
                    ForwarderCommand(origin=self._system_id, command=command)
                )
            raise
        finally:
            if session_id is not None and not settled:
                # Aborted before a terminating chunk arrived (client
                # disconnect, error): no usage to attribute, just drop the
                # live in-flight counters.
                self._settle_inflight(command_id, session_id)
            await self._send(TaskFinished(finished_command_id=command_id))
            if command_id in self._text_generation_queues:
                del self._text_generation_queues[command_id]

    async def _collect_text_generation_with_stats(
        self, command_id: CommandId
    ) -> BenchChatCompletionResponse:
        sampler = PowerSampler(get_node_system=lambda: self.state.node_system)
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        model: ModelId | None = None
        finish_reason: FinishReason | None = None

        stats: GenerationStats | None = None

        async with anyio.create_task_group() as tg:
            tg.start_soon(sampler.run)

            async for chunk in self._token_chunk_stream(command_id):
                if isinstance(chunk, PrefillProgressChunk):
                    continue

                sampler.mark_prefill_done()

                if chunk.finish_reason == "error":
                    raise HTTPException(
                        status_code=500,
                        detail=chunk.error_message or "Internal server error",
                    )

                if model is None:
                    model = chunk.model

                if isinstance(chunk, TokenChunk):
                    text_parts.append(chunk.text)

                if isinstance(chunk, ToolCallChunk):
                    tool_calls.extend(
                        ToolCall(
                            id=str(uuid4()),
                            index=i,
                            function=tool,
                        )
                        for i, tool in enumerate(chunk.tool_calls)
                    )

                stats = chunk.stats or stats

                if chunk.finish_reason is not None:
                    finish_reason = chunk.finish_reason

            tg.cancel_scope.cancel()

        combined_text = "".join(text_parts)
        assert model is not None

        return BenchChatCompletionResponse(
            id=command_id,
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=combined_text,
                        tool_calls=tool_calls if tool_calls else None,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            generation_stats=stats,
            power_usage=sampler.result(),
        )

    async def _trigger_notify_user_to_download_model(self, model_id: ModelId) -> None:
        logger.warning(
            "TODO: we should send a notification to the user to download the model"
        )

    async def _send_text_generation_with_images(
        self, task_params: TextGenerationTaskParams
    ) -> TextGeneration:
        task_params = task_params.with_card_sampling_defaults()
        images = task_params.images
        if not images:
            command = TextGeneration(task_params=task_params)
            await self._send(command)
            return command

        hashes = [hashlib.sha256(img.encode("ascii")).hexdigest() for img in images]
        all_hashes = {idx: Base64ImageHash(h) for idx, h in enumerate(hashes)}
        task_params = task_params.model_copy(
            update={"images": [], "image_hashes": all_hashes}
        )
        command = TextGeneration(task_params=task_params)

        new_images: list[tuple[int, str]] = []
        for idx, (img, h) in enumerate(zip(images, hashes, strict=True)):
            if h not in self._sent_image_hashes:
                self._sent_image_hashes.add(h)
                new_images.append((idx, img))

        if not new_images:
            await self._send(command)
            return command

        all_chunks: list[tuple[int, str]] = []
        for img_idx, img_data in new_images:
            for i in range(0, len(img_data), EXO_MAX_CHUNK_SIZE):
                all_chunks.append((img_idx, img_data[i : i + EXO_MAX_CHUNK_SIZE]))

        for global_idx, (img_idx, chunk_data) in enumerate(all_chunks):
            await self._send(
                SendInputChunk(
                    chunk=InputImageChunk(
                        model=task_params.model,
                        command_id=command.command_id,
                        data=chunk_data,
                        chunk_index=global_idx,
                        total_chunks=len(all_chunks),
                        image_index=img_idx,
                    )
                )
            )

        await self._send(command)
        return command

    async def chat_completions(
        self, payload: ChatCompletionRequest, request: Request
    ) -> ChatCompletionResponse | StreamingResponse:
        """OpenAI Chat Completions API - adapter."""
        session_id = self._session_id_from_request(request)
        task_params = await chat_request_to_text_generation(payload)
        validated_model = await self._validate_model_has_instance(task_params.model)
        task_params = task_params.model_copy(update={"model": validated_model})

        command = await self._send_text_generation_with_images(task_params)

        if payload.stream:
            return StreamingResponse(
                with_sse_keepalive(
                    generate_chat_stream(
                        command.command_id,
                        self._token_chunk_stream(
                            command.command_id, session_id=session_id
                        ),
                    ),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return StreamingResponse(
                collect_chat_response(
                    command.command_id,
                    self._token_chunk_stream(command.command_id, session_id=session_id),
                ),
                media_type="application/json",
            )

    def _session_id_from_request(self, request: Request) -> str:
        """Identify the session a request belongs to.

        pi agent (via exo-router) sends x-pi-session-id; x-agent-id and
        x-session-id are accepted as fallbacks. Requests without a session
        header (e.g. dashboard chats) share the "dashboard" bucket.
        """
        for header in ("x-pi-session-id", "x-agent-id", "x-session-id"):
            value = request.headers.get(header)
            if value:
                return value
        return _DEFAULT_SESSION_ID

    def _instance_for_command(self, command_id: CommandId) -> InstanceId | None:
        """Resolve which instance is serving a command from live task state."""
        for task in self.state.tasks.values():
            if getattr(task, "command_id", None) == command_id:
                return task.instance_id
        return None

    def _settle_inflight(self, command_id: CommandId, session_id: str) -> None:
        """Drop one request's live counters and decrement its session's in-flight count."""
        self._command_inflight_usage.pop(command_id, None)
        remaining = self._session_inflight.get(session_id, 0) - 1
        if remaining > 0:
            self._session_inflight[session_id] = remaining
        else:
            self._session_inflight.pop(session_id, None)

    def _record_session_tokens(
        self, instance_id: InstanceId, session_id: str, usage: Usage
    ) -> None:
        """Fold one completed request's usage into the live session bucket."""
        per_instance = self._session_token_usage.setdefault(instance_id, {})
        entry = per_instance.get(session_id)
        if entry is None:
            entry = _SessionTokenUsage()
            per_instance[session_id] = entry
        entry.prompt_tokens += usage.prompt_tokens
        entry.completion_tokens += usage.completion_tokens
        entry.request_count += 1
        entry.last_active_ts = time.time()
        if len(per_instance) > _MAX_TRACKED_SESSIONS_PER_INSTANCE:
            oldest = min(per_instance, key=lambda sid: per_instance[sid].last_active_ts)
            del per_instance[oldest]

    def _lifetime_key(self, instance: Instance) -> str:
        """Stable identity for an instance card across restarts.

        Instance ids are recreated on restart / topology change; the
        (concrete instance type, model) pair is what the user thinks of as
        "the same instance".
        """
        return f"{type(instance).__name__}::{instance.shard_assignments.model_id}"

    def _load_lifetime_usage(self) -> dict[str, _LifetimeUsage]:
        """Load durable lifetime totals from disk (best-effort)."""
        try:
            raw = cast(
                "dict[str, dict[str, int]]",
                json.loads(_LIFETIME_USAGE_FILE.read_text()),
            )
            return {
                str(key): _LifetimeUsage(
                    prompt_tokens=value["promptTokens"],
                    completion_tokens=value["completionTokens"],
                    request_count=value["requestCount"],
                )
                for key, value in raw.items()
            }
        except FileNotFoundError:
            return {}
        except Exception as e:
            logger.warning(
                f"Ignoring unreadable lifetime usage file {_LIFETIME_USAGE_FILE}: {e}"
            )
            return {}

    def _persist_lifetime_usage(self) -> None:
        """Atomically write durable lifetime totals (best-effort)."""
        try:
            _LIFETIME_USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = _LIFETIME_USAGE_FILE.with_name(_LIFETIME_USAGE_FILE.name + ".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        key: {
                            "promptTokens": usage.prompt_tokens,
                            "completionTokens": usage.completion_tokens,
                            "requestCount": usage.request_count,
                        }
                        for key, usage in self._lifetime_usage.items()
                    }
                )
            )
            tmp.replace(_LIFETIME_USAGE_FILE)
        except Exception as e:
            logger.warning(f"Failed to persist lifetime usage: {e}")

    def _fold_lifetime_usage(
        self, instance: Instance, prompt_tokens: int, completion_tokens: int
    ) -> None:
        """Fold one completed request's usage into the durable lifetime totals."""
        key = self._lifetime_key(instance)
        entry = self._lifetime_usage.setdefault(key, _LifetimeUsage())
        entry.prompt_tokens += prompt_tokens
        entry.completion_tokens += completion_tokens
        entry.request_count += 1
        self._persist_lifetime_usage()

    async def get_session_usage(self) -> dict[str, Any]:
        """Live per-session token usage, grouped by instance.

        Consumed by the dashboard: the instance card's "active" line sums the
        sessions that are in flight or were active recently (including the
        running counters of requests still streaming), while the Σ line shows
        the durable lifetime totals from the persisted sidecar store.
        """
        now = time.time()
        # Aggregate live counters per (instance, session).
        inflight: dict[InstanceId, dict[str, list[int]]] = {}
        for command in self._command_inflight_usage.values():
            if command.instance_id is None:
                continue
            per_instance = inflight.setdefault(command.instance_id, {})
            acc = per_instance.setdefault(command.session_id, [0, 0])
            acc[0] += command.prompt_tokens_estimate
            acc[1] += command.completion_tokens
        # Recorded sessions, plus synthetic entries for in-flight sessions
        # that have not completed a request yet.
        per_instance_sessions: dict[InstanceId, dict[str, _SessionTokenUsage]] = {
            instance_id: dict(sessions)
            for instance_id, sessions in self._session_token_usage.items()
        }
        for instance_id, sessions in inflight.items():
            recorded = per_instance_sessions.setdefault(instance_id, {})
            for session_id in sessions:
                recorded.setdefault(session_id, _SessionTokenUsage())

        instances: dict[str, list[dict[str, Any]]] = {}
        for instance_id, sessions in per_instance_sessions.items():
            live = inflight.get(instance_id, {})
            instances[str(instance_id)] = [
                {
                    "sessionId": session_id,
                    "promptTokens": entry.prompt_tokens,
                    "completionTokens": entry.completion_tokens,
                    "requestCount": entry.request_count,
                    "inFlight": self._session_inflight.get(session_id, 0),
                    "inFlightPromptTokens": live.get(session_id, [0, 0])[0],
                    "inFlightCompletionTokens": live.get(session_id, [0, 0])[1],
                    "lastActiveSecondsAgo": (
                        round(now - entry.last_active_ts, 1)
                        if entry.last_active_ts
                        else None
                    ),
                }
                for session_id, entry in sessions.items()
            ]
        # Durable lifetime totals: rows for live instances (keyed by instance
        # id, as before) PLUS every persisted key under its stable lifetime
        # key, so the Σ totals survive idle-unload / instance recreation.
        lifetime: dict[str, dict[str, int]] = {}
        for instance_id, instance in self.state.instances.items():
            key = self._lifetime_key(instance)
            usage = self._lifetime_usage.get(key)
            if usage is None:
                continue
            stats = self.state.instance_lifecycle.get(key)
            lifetime[str(instance_id)] = {
                "promptTokens": usage.prompt_tokens,
                "completionTokens": usage.completion_tokens,
                "requestCount": usage.request_count,
                "loadCount": stats.load_count if stats is not None else 0,
                "unloadCount": stats.unload_count if stats is not None else 0,
            }
        lifetime_keys_emitted = set(self._lifetime_usage) & {
            self._lifetime_key(instance) for instance in self.state.instances.values()
        }
        for key, usage in self._lifetime_usage.items():
            if key in lifetime_keys_emitted:
                continue
            stats = self.state.instance_lifecycle.get(key)
            lifetime[key] = {
                "promptTokens": usage.prompt_tokens,
                "completionTokens": usage.completion_tokens,
                "requestCount": usage.request_count,
                "loadCount": stats.load_count if stats is not None else 0,
                "unloadCount": stats.unload_count if stats is not None else 0,
            }
        return {"instances": instances, "lifetime": lifetime}

    async def bench_chat_completions(
        self, payload: BenchChatCompletionRequest
    ) -> BenchChatCompletionResponse | StreamingResponse:
        task_params = await chat_request_to_text_generation(payload)
        validated_model = await self._validate_model_has_instance(
            ModelId(task_params.model)
        )
        task_params = task_params.model_copy(update={"model": validated_model})

        task_params = task_params.model_copy(
            update={
                "stream": False,
                "bench": True,
                "use_prefix_cache": payload.use_prefix_cache,
            }
        )

        command = await self._send_text_generation_with_images(task_params)

        if payload.stream:
            return StreamingResponse(
                with_sse_keepalive(
                    generate_chat_stream(
                        command.command_id,
                        self._token_chunk_stream(command.command_id),
                    ),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )

        return await self._collect_text_generation_with_stats(command.command_id)

    def _launch_recipe_from_instance(self, instance: Instance) -> LaunchProfileEntry:
        """Derive the re-launchable recipe (model/sharding/meta/nodes) from a
        live instance, including its current idle-unload policy if any."""
        meta = "MlxJaccl" if isinstance(instance, MlxJacclInstance) else "MlxRing"
        sharding = "Tensor"
        for shard in instance.shard_assignments.runner_to_shard.values():
            if isinstance(shard, (PipelineShardMetadata, CfgShardMetadata)):
                sharding = "Pipeline"
                break
        policy = self.state.instance_idle_policies.get(instance.instance_id)
        return LaunchProfileEntry(
            model_id=str(instance.shard_assignments.model_id),
            sharding=sharding,
            instance_meta=meta,
            min_nodes=len(instance.shard_assignments.node_to_runner),
            idle_unload_seconds=(
                policy.idle_unload_seconds if policy is not None else None
            ),
            added_at=time.time(),
        )

    def _record_launch(self, entry: LaunchProfileEntry) -> None:
        self._launch_profile = upsert_launch_profile(entry)

    async def _validate_model_has_instance(self, model_id: ModelId) -> ModelId:
        """Validate a model has an active instance.
        If the model isn't even downloaded, triggers notification to user to download model.


        Raises HTTPException 404 if no instance is found for the model.
        """
        if not any(
            instance.shard_assignments.model_id == model_id
            for instance in self.state.instances.values()
        ):
            # Auto re-place after an idle unload: the model is downloaded and
            # we captured its placement parameters when the policy was set, so
            # transparently re-place instead of 404-ing the request.
            if await self._replace_idle_unloaded(model_id):
                return model_id
            # Check if model is actually downloaded
            model_is_downloaded = any(
                isinstance(download, DownloadCompleted)
                and download.shard_metadata.model_card.model_id == model_id
                for node_downloads in self.state.downloads.values()
                for download in node_downloads
            )
            if not model_is_downloaded:
                await self._trigger_notify_user_to_download_model(model_id)

            raise HTTPException(
                status_code=404, detail=f"No instance found for model {model_id}"
            )
        return model_id

    async def _replace_idle_unloaded(self, model_id: ModelId) -> bool:
        """Re-place a model that was unloaded by the idle-unload sweeper.

        Returns True when a re-placement command was issued. The pending
        policy keyed by model id carries the placement parameters captured at
        the original creation (sharding/instance_meta/min_nodes); re-issuing
        it also re-arms the idle-unload policy for the new instance.
        """
        pending = self._pending_idle_policies.get(str(model_id))
        if pending is None:
            return False
        policy, _sent_at = pending
        await self._send(
            PlaceInstance(
                model_card=await ModelCard.load(model_id),
                sharding=Sharding[policy.sharding],
                instance_meta=InstanceMeta[policy.instance_meta],
                min_nodes=policy.min_nodes,
            )
        )
        logger.info(
            f"Re-placing idle-unloaded model {model_id} "
            f"(idle_unload={policy.idle_unload_seconds}s)"
        )
        return True

    def stream_events(self) -> StreamingResponse:
        def _generate_json_array(events: Iterable[Event]) -> Iterable[str]:
            yield "["
            first = True
            for event in events:
                if not first:
                    yield ","
                first = False
                yield event.model_dump_json()
            yield "]"

        return StreamingResponse(
            _generate_json_array(self._event_log.read_all()),
            media_type="application/json",
        )

    async def get_image(self, image_id: str) -> FileResponse:
        stored = self._image_store.get(Id(image_id))
        if stored is None:
            raise HTTPException(status_code=404, detail="Image not found or expired")
        return FileResponse(path=stored.file_path, media_type=stored.content_type)

    async def list_images(self, request: Request) -> ImageListResponse:
        """List all stored images."""
        stored_images = self._image_store.list_images()
        return ImageListResponse(
            data=[
                ImageListItem(
                    image_id=img.image_id,
                    url=self._build_image_url(request, img.image_id),
                    content_type=img.content_type,
                    expires_at=img.expires_at,
                )
                for img in stored_images
            ]
        )

    def _build_image_url(self, request: Request, image_id: Id) -> str:
        host = request.headers.get("host", f"localhost:{self.port}")
        scheme = "https" if request.url.scheme == "https" else "http"
        return f"{scheme}://{host}/v1/images/{image_id}"

    async def image_generations(
        self, request: Request, payload: ImageGenerationTaskParams
    ) -> ImageGenerationResponse | StreamingResponse:
        """Handle image generation requests.

        When stream=True and partial_images > 0, returns a StreamingResponse
        with SSE-formatted events for partial and final images.
        """
        payload = payload.model_copy(
            update={
                "model": await self._validate_model_has_instance(
                    ModelId(payload.model)
                ),
                "advanced_params": _ensure_seed(payload.advanced_params),
            }
        )

        command = ImageGeneration(
            task_params=payload,
        )
        await self._send(command)

        # Check if streaming is requested
        if payload.stream and payload.partial_images and payload.partial_images > 0:
            return StreamingResponse(
                self._generate_image_stream(
                    request=request,
                    command_id=command.command_id,
                    num_images=payload.n or 1,
                    response_format=payload.response_format or "b64_json",
                ),
                media_type="text/event-stream",
            )

        # Non-streaming: collect all image chunks
        return await self._collect_image_generation(
            request=request,
            command_id=command.command_id,
            num_images=payload.n or 1,
            response_format=payload.response_format or "b64_json",
        )

    async def _generate_image_stream(
        self,
        request: Request,
        command_id: CommandId,
        num_images: int,
        response_format: str,
    ) -> AsyncGenerator[str, None]:
        """Generate SSE stream of partial and final images."""
        # Track chunks: {(image_index, is_partial): {chunk_index: data}}
        image_chunks: dict[tuple[int, bool], dict[int, str]] = {}
        image_total_chunks: dict[tuple[int, bool], int] = {}
        image_metadata: dict[tuple[int, bool], tuple[int | None, int | None]] = {}
        images_complete = 0

        try:
            self._image_generation_queues[command_id], recv = channel[
                ImageChunk | ErrorChunk
            ]()

            with recv as chunks:
                async for chunk in chunks:
                    if chunk.finish_reason == "error":
                        error_response = ErrorResponse(
                            error=ErrorInfo(
                                message=chunk.error_message or "Internal server error",
                                type="InternalServerError",
                                code=500,
                            )
                        )
                        yield f"data: {error_response.model_dump_json()}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    key = (chunk.image_index, chunk.is_partial)

                    if key not in image_chunks:
                        image_chunks[key] = {}
                        image_total_chunks[key] = chunk.total_chunks
                        image_metadata[key] = (
                            chunk.partial_index,
                            chunk.total_partials,
                        )

                    image_chunks[key][chunk.chunk_index] = chunk.data

                    # Check if this image is complete
                    if len(image_chunks[key]) == image_total_chunks[key]:
                        full_data = "".join(
                            image_chunks[key][i] for i in range(len(image_chunks[key]))
                        )

                        partial_idx, total_partials = image_metadata[key]

                        if chunk.is_partial:
                            # Yield partial image event (always use b64_json for partials)
                            event_data = {
                                "type": "partial",
                                "image_index": chunk.image_index,
                                "partial_index": partial_idx,
                                "total_partials": total_partials,
                                "format": str(chunk.format),
                                "data": {
                                    "b64_json": full_data
                                    if response_format == "b64_json"
                                    else None,
                                },
                            }
                            yield f"data: {json.dumps(event_data)}\n\n"
                        else:
                            # Final image
                            if response_format == "url":
                                image_bytes = base64.b64decode(full_data)
                                content_type = _format_to_content_type(chunk.format)
                                stored = self._image_store.store(
                                    image_bytes, content_type
                                )
                                url = self._build_image_url(request, stored.image_id)
                                event_data = {
                                    "type": "final",
                                    "image_index": chunk.image_index,
                                    "format": str(chunk.format),
                                    "data": {"url": url},
                                }
                            else:
                                event_data = {
                                    "type": "final",
                                    "image_index": chunk.image_index,
                                    "format": str(chunk.format),
                                    "data": {"b64_json": full_data},
                                }
                            yield f"data: {json.dumps(event_data)}\n\n"
                            images_complete += 1

                            if images_complete >= num_images:
                                yield "data: [DONE]\n\n"
                                break

                        # Clean up completed image chunks
                        del image_chunks[key]
                        del image_total_chunks[key]
                        del image_metadata[key]

        except anyio.get_cancelled_exc_class():
            command = TaskCancelled(cancelled_command_id=command_id)
            with anyio.CancelScope(shield=True):
                await self.command_sender.send(
                    ForwarderCommand(origin=self._system_id, command=command)
                )
            raise
        finally:
            await self._send(TaskFinished(finished_command_id=command_id))
            if command_id in self._image_generation_queues:
                del self._image_generation_queues[command_id]

    async def _collect_image_chunks(
        self,
        request: Request | None,
        command_id: CommandId,
        num_images: int,
        response_format: str,
        capture_stats: bool = False,
    ) -> tuple[list[ImageData], ImageGenerationStats | None]:
        """Collect image chunks and optionally capture stats."""
        # Track chunks per image: {image_index: {chunk_index: data}}
        # Only track non-partial (final) images
        image_chunks: dict[int, dict[int, str]] = {}
        image_total_chunks: dict[int, int] = {}
        image_formats: dict[int, Literal["png", "jpeg", "webp"] | None] = {}
        images_complete = 0
        stats: ImageGenerationStats | None = None

        try:
            self._image_generation_queues[command_id], recv = channel[
                ImageChunk | ErrorChunk
            ]()

            while images_complete < num_images:
                with recv as chunks:
                    async for chunk in chunks:
                        if chunk.finish_reason == "error":
                            raise HTTPException(
                                status_code=500,
                                detail=chunk.error_message or "Internal server error",
                            )

                        if chunk.is_partial:
                            continue

                        if chunk.image_index not in image_chunks:
                            image_chunks[chunk.image_index] = {}
                            image_total_chunks[chunk.image_index] = chunk.total_chunks
                            image_formats[chunk.image_index] = chunk.format

                        image_chunks[chunk.image_index][chunk.chunk_index] = chunk.data

                        if capture_stats and chunk.stats is not None:
                            stats = chunk.stats

                        if (
                            len(image_chunks[chunk.image_index])
                            == image_total_chunks[chunk.image_index]
                        ):
                            images_complete += 1

                        if images_complete >= num_images:
                            break

            images: list[ImageData] = []
            for image_idx in range(num_images):
                chunks_dict = image_chunks[image_idx]
                full_data = "".join(chunks_dict[i] for i in range(len(chunks_dict)))
                if response_format == "url" and request is not None:
                    image_bytes = base64.b64decode(full_data)
                    content_type = _format_to_content_type(image_formats.get(image_idx))
                    stored = self._image_store.store(image_bytes, content_type)
                    url = self._build_image_url(request, stored.image_id)
                    images.append(ImageData(b64_json=None, url=url))
                else:
                    images.append(
                        ImageData(
                            b64_json=full_data
                            if response_format == "b64_json"
                            else None,
                            url=None,
                        )
                    )

            return (images, stats if capture_stats else None)
        except anyio.get_cancelled_exc_class():
            command = TaskCancelled(cancelled_command_id=command_id)
            with anyio.CancelScope(shield=True):
                await self.command_sender.send(
                    ForwarderCommand(origin=self._system_id, command=command)
                )
            raise
        finally:
            await self._send(TaskFinished(finished_command_id=command_id))
            if command_id in self._image_generation_queues:
                del self._image_generation_queues[command_id]

    async def _collect_image_generation(
        self,
        request: Request,
        command_id: CommandId,
        num_images: int,
        response_format: str,
    ) -> ImageGenerationResponse:
        """Collect all image chunks (non-streaming) and return a single response."""
        images, _ = await self._collect_image_chunks(
            request, command_id, num_images, response_format, capture_stats=False
        )
        return ImageGenerationResponse(data=images)

    async def _collect_image_generation_with_stats(
        self,
        request: Request | None,
        command_id: CommandId,
        num_images: int,
        response_format: str,
    ) -> BenchImageGenerationResponse:
        sampler = PowerSampler(get_node_system=lambda: self.state.node_system)
        images: list[ImageData] = []
        stats: ImageGenerationStats | None = None
        async with anyio.create_task_group() as tg:
            tg.start_soon(sampler.run)
            images, stats = await self._collect_image_chunks(
                request, command_id, num_images, response_format, capture_stats=True
            )
            tg.cancel_scope.cancel()
        return BenchImageGenerationResponse(
            data=images, generation_stats=stats, power_usage=sampler.result()
        )

    async def bench_image_generations(
        self, request: Request, payload: BenchImageGenerationTaskParams
    ) -> BenchImageGenerationResponse:
        payload = payload.model_copy(
            update={
                "model": await self._validate_model_has_instance(
                    ModelId(payload.model)
                ),
                "stream": False,
                "partial_images": 0,
                "advanced_params": _ensure_seed(payload.advanced_params),
            }
        )

        command = ImageGeneration(
            task_params=payload,
        )
        await self._send(command)

        return await self._collect_image_generation_with_stats(
            request=request,
            command_id=command.command_id,
            num_images=payload.n or 1,
            response_format=payload.response_format or "b64_json",
        )

    async def _send_image_edits_command(
        self,
        image: UploadFile,
        prompt: str,
        model: ModelId,
        n: int,
        size: ImageSize,
        response_format: Literal["url", "b64_json"],
        input_fidelity: Literal["low", "high"],
        stream: bool,
        partial_images: int,
        bench: bool,
        quality: Literal["high", "medium", "low"],
        output_format: Literal["png", "jpeg", "webp"],
        advanced_params: AdvancedImageParams | None,
    ) -> ImageEdits:
        """Prepare and send an image edits command with chunked image upload."""
        validated_model = await self._validate_model_has_instance(model)
        advanced_params = _ensure_seed(advanced_params)

        image_content = await image.read()
        image_data = base64.b64encode(image_content).decode("utf-8")

        image_strength = 0.7 if input_fidelity == "high" else 0.3

        data_chunks = [
            image_data[i : i + EXO_MAX_CHUNK_SIZE]
            for i in range(0, len(image_data), EXO_MAX_CHUNK_SIZE)
        ]
        total_chunks = len(data_chunks)

        command = ImageEdits(
            task_params=ImageEditsTaskParams(
                image_data="",
                total_input_chunks=total_chunks,
                prompt=prompt,
                model=validated_model,
                n=n,
                size=size,
                response_format=response_format,
                image_strength=image_strength,
                stream=stream,
                partial_images=partial_images,
                bench=bench,
                quality=quality,
                output_format=output_format,
                advanced_params=advanced_params,
            ),
        )

        logger.info(
            f"Sending input image: {len(image_data)} bytes in {total_chunks} chunks"
        )
        for chunk_index, chunk_data in enumerate(data_chunks):
            await self._send(
                SendInputChunk(
                    chunk=InputImageChunk(
                        model=validated_model,
                        command_id=command.command_id,
                        data=chunk_data,
                        chunk_index=chunk_index,
                        total_chunks=total_chunks,
                    )
                )
            )

        await self._send(command)
        return command

    async def image_edits(
        self,
        request: Request,
        image: UploadFile = File(...),  # noqa: B008
        prompt: str = Form(...),
        model: str = Form(...),
        n: int = Form(1),
        size: str | None = Form(None),
        response_format: Literal["url", "b64_json"] = Form("b64_json"),
        input_fidelity: Literal["low", "high"] = Form("low"),
        stream: str = Form("false"),
        partial_images: str = Form("0"),
        quality: Literal["high", "medium", "low"] = Form("medium"),
        output_format: Literal["png", "jpeg", "webp"] = Form("png"),
        advanced_params: str | None = Form(None),
    ) -> ImageGenerationResponse | StreamingResponse:
        """Handle image editing requests (img2img)."""
        # Parse string form values to proper types
        stream_bool = stream.lower() in ("true", "1", "yes")
        partial_images_int = int(partial_images) if partial_images.isdigit() else 0

        parsed_advanced_params: AdvancedImageParams | None = None
        if advanced_params:
            with contextlib.suppress(Exception):
                parsed_advanced_params = AdvancedImageParams.model_validate_json(
                    advanced_params
                )

        command = await self._send_image_edits_command(
            image=image,
            prompt=prompt,
            model=ModelId(model),
            n=n,
            size=normalize_image_size(size),
            response_format=response_format,
            input_fidelity=input_fidelity,
            stream=stream_bool,
            partial_images=partial_images_int,
            bench=False,
            quality=quality,
            output_format=output_format,
            advanced_params=parsed_advanced_params,
        )

        if stream_bool and partial_images_int > 0:
            return StreamingResponse(
                self._generate_image_stream(
                    request=request,
                    command_id=command.command_id,
                    num_images=n,
                    response_format=response_format,
                ),
                media_type="text/event-stream",
            )

        return await self._collect_image_generation(
            request=request,
            command_id=command.command_id,
            num_images=n,
            response_format=response_format,
        )

    async def bench_image_edits(
        self,
        request: Request,
        image: UploadFile = File(...),  # noqa: B008
        prompt: str = Form(...),
        model: str = Form(...),
        n: int = Form(1),
        size: str | None = Form(None),
        response_format: Literal["url", "b64_json"] = Form("b64_json"),
        input_fidelity: Literal["low", "high"] = Form("low"),
        quality: Literal["high", "medium", "low"] = Form("medium"),
        output_format: Literal["png", "jpeg", "webp"] = Form("png"),
        advanced_params: str | None = Form(None),
    ) -> BenchImageGenerationResponse:
        """Handle benchmark image editing requests with generation stats."""
        parsed_advanced_params: AdvancedImageParams | None = None
        if advanced_params:
            with contextlib.suppress(Exception):
                parsed_advanced_params = AdvancedImageParams.model_validate_json(
                    advanced_params
                )

        command = await self._send_image_edits_command(
            image=image,
            prompt=prompt,
            model=ModelId(model),
            n=n,
            size=normalize_image_size(size),
            response_format=response_format,
            input_fidelity=input_fidelity,
            stream=False,
            partial_images=0,
            bench=True,
            quality=quality,
            output_format=output_format,
            advanced_params=parsed_advanced_params,
        )

        return await self._collect_image_generation_with_stats(
            request=request,
            command_id=command.command_id,
            num_images=n,
            response_format=response_format,
        )

    async def claude_messages(
        self, payload: ClaudeMessagesRequest
    ) -> ClaudeMessagesResponse | StreamingResponse:
        """Claude Messages API - adapter."""
        task_params = await claude_request_to_text_generation(payload)
        validated_model = await self._validate_model_has_instance(
            ModelId(task_params.model)
        )
        task_params = task_params.model_copy(update={"model": validated_model})

        command = await self._send_text_generation_with_images(task_params)

        if payload.stream:
            return StreamingResponse(
                with_sse_keepalive(
                    generate_claude_stream(
                        command.command_id,
                        payload.model,
                        self._token_chunk_stream(command.command_id),
                    ),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return StreamingResponse(
                collect_claude_response(
                    command.command_id,
                    payload.model,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/json",
            )

    async def openai_responses(
        self, payload: ResponsesRequest
    ) -> ResponsesResponse | StreamingResponse:
        """OpenAI Responses API."""
        task_params = await responses_request_to_text_generation(payload)
        validated_model = await self._validate_model_has_instance(task_params.model)
        task_params = task_params.model_copy(update={"model": validated_model})

        command = await self._send_text_generation_with_images(task_params)

        if payload.stream:
            return StreamingResponse(
                with_sse_keepalive(
                    generate_responses_stream(
                        command.command_id,
                        payload.model,
                        self._token_chunk_stream(command.command_id),
                    ),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )

        else:
            return StreamingResponse(
                collect_responses_response(
                    command.command_id,
                    payload.model,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/json",
            )

    async def _ollama_root(self) -> JSONResponse:
        """Respond to HEAD / from Ollama CLI connectivity checks."""
        return JSONResponse(content="Ollama is running")

    async def ollama_chat(
        self, request: Request
    ) -> OllamaChatResponse | StreamingResponse:
        """Ollama Chat API — accepts JSON regardless of Content-Type."""
        body = await request.body()
        payload = OllamaChatRequest.model_validate_json(body)
        task_params = ollama_request_to_text_generation(payload)
        validated_model = await self._validate_model_has_instance(
            ModelId(task_params.model)
        )
        task_params = task_params.model_copy(update={"model": validated_model})

        command = await self._send_text_generation_with_images(task_params)

        if payload.stream:
            return StreamingResponse(
                generate_ollama_chat_stream(
                    command.command_id,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return StreamingResponse(
                collect_ollama_chat_response(
                    command.command_id,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/json",
            )

    async def ollama_generate(
        self, request: Request
    ) -> OllamaGenerateResponse | StreamingResponse:
        """Ollama Generate API — accepts JSON regardless of Content-Type."""
        body = await request.body()
        payload = OllamaGenerateRequest.model_validate_json(body)
        task_params = ollama_generate_request_to_text_generation(payload)
        validated_model = await self._validate_model_has_instance(
            ModelId(task_params.model)
        )
        task_params = task_params.model_copy(update={"model": validated_model})

        command = await self._send_text_generation_with_images(task_params)

        if payload.stream:
            return StreamingResponse(
                generate_ollama_generate_stream(
                    command.command_id,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return StreamingResponse(
                collect_ollama_generate_response(
                    command.command_id,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/json",
            )

    async def ollama_tags(self) -> OllamaTagsResponse:
        """Returns list of models in Ollama tags format. We return the downloaded ones only."""

        downloaded_model_ids: set[ModelId] = set()
        for node_downloads in self.state.downloads.values():
            for dl in node_downloads:
                if isinstance(dl, DownloadCompleted):
                    downloaded_model_ids.add(dl.shard_metadata.model_card.model_id)

        cards = [
            c
            for c in await model_cards.card_cache.list_all()
            if c.model_id in downloaded_model_ids
        ]

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return OllamaTagsResponse(
            models=[
                OllamaModelTag(
                    name=str(card.model_id),
                    model=str(card.model_id),
                    modified_at=now,
                    size=card.storage_size.in_bytes,
                    digest="sha256:000000000000",
                    details=OllamaModelDetails(
                        family=card.family or None,
                        quantization_level=card.quantization or None,
                    ),
                )
                for card in cards
            ]
        )

    async def ollama_show(self, request: Request) -> OllamaShowResponse:
        """Returns model information in Ollama show format."""
        body = await request.body()
        payload = OllamaShowRequest.model_validate_json(body)
        model_name = payload.name or payload.model
        if not model_name:
            raise HTTPException(status_code=400, detail="name or model is required")
        try:
            card = await ModelCard.load(ModelId(model_name))
        except Exception as exc:
            raise HTTPException(
                status_code=404, detail=f"Model not found: {model_name}"
            ) from exc

        capabilities: list[OllamaCapability] = []
        if ModelTask.TextGeneration in card.tasks:
            capabilities.extend(("completion", "tools"))
        if card.vision is not None:
            capabilities.append("vision")

        architecture = card.family or "unknown"
        model_info: dict[str, Any] = {
            "general.architecture": architecture,
            "general.basename": card.base_model or str(card.model_id),
        }
        if card.context_length > 0:
            model_info[f"{architecture}.context_length"] = card.context_length

        return OllamaShowResponse(
            modelfile=f"FROM {card.model_id}",
            template="{{ .Prompt }}",
            details=OllamaModelDetails(
                family=card.family or None,
                quantization_level=card.quantization or None,
            ),
            model_info=model_info,
            capabilities=capabilities,
        )

    async def ollama_ps(self) -> OllamaPsResponse:
        """Returns list of running models (active instances)."""
        models: list[OllamaPsModel] = []
        seen: set[str] = set()
        for instance in self.state.instances.values():
            model_id = str(instance.shard_assignments.model_id)
            if model_id in seen:
                continue
            seen.add(model_id)
            models.append(
                OllamaPsModel(
                    name=model_id,
                    model=model_id,
                    size=0,
                )
            )
        return OllamaPsResponse(models=models)

    async def ollama_version(self) -> dict[str, str]:
        """Returns version information for Ollama API compatibility."""
        return {"version": "1.0.0"}

    def _calculate_total_available_memory(self) -> Memory:
        """Calculate total available memory across all nodes in bytes."""
        total_available = Memory()

        for memory in self.state.node_memory.values():
            total_available += memory.ram_available

        return total_available

    async def get_models(self, status: str | None = Query(default=None)) -> ModelList:
        """Returns list of available models, optionally filtered by being downloaded."""
        cards = await model_cards.card_cache.list_all()

        if status == "downloaded":
            downloaded_model_ids: set[str] = set()
            for node_downloads in self.state.downloads.values():
                for dl in node_downloads:
                    if isinstance(dl, DownloadCompleted):
                        downloaded_model_ids.add(dl.shard_metadata.model_card.model_id)
            cards = [c for c in cards if c.model_id in downloaded_model_ids]

        return ModelList(
            data=[
                ModelListModel(
                    id=card.model_id,
                    hugging_face_id=card.model_id,
                    name=card.model_id.short(),
                    description="",
                    tags=[],
                    storage_size_megabytes=card.storage_size.in_mb,
                    supports_tensor=card.supports_tensor,
                    tasks=[task.value for task in card.tasks],
                    is_custom=card.is_custom,
                    family=card.family,
                    quantization=card.quantization,
                    base_model=card.base_model,
                    capabilities=card.capabilities,
                    reasoning_dialect=card.reasoning_dialect,
                    context_length=card.context_length,
                )
                for card in cards
            ]
        )

    async def add_custom_model(self, payload: AddCustomModelParams) -> ModelListModel:
        """Fetch a model from HuggingFace and save as a custom model card, then sync across the cluster."""
        try:
            card = await ModelCard.fetch_from_hf(payload.model_id)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Failed to fetch model: {exc}"
            ) from exc

        await self.command_sender.send(
            ForwarderCommand(
                origin=self._system_id,
                command=AddCustomModelCard(model_card=card),
            )
        )

        # Immediately update the local cache so the subsequent GET /models
        # returns the new model without waiting for the event round-trip.
        model_cards.card_cache.cc[card.model_id] = card

        return ModelListModel(
            id=card.model_id,
            hugging_face_id=card.model_id,
            name=card.model_id.short(),
            description="",
            tags=[],
            storage_size_megabytes=int(card.storage_size.in_mb),
            supports_tensor=card.supports_tensor,
            tasks=[task.value for task in card.tasks],
            is_custom=True,
        )

    async def delete_custom_model(self, model_id: ModelId) -> JSONResponse:
        """Delete a user-added custom model card and sync deletion across the cluster."""
        card = model_cards.card_cache.get(model_id)
        if card is None or not card.is_custom:
            raise HTTPException(status_code=404, detail="Custom model card not found")

        await self.command_sender.send(
            ForwarderCommand(
                origin=self._system_id,
                command=DeleteCustomModelCard(model_id=model_id),
            )
        )

        return JSONResponse(
            {"message": "Model card deleted", "model_id": str(model_id)}
        )

    async def search_models(
        self, query: str = "", limit: int = 20
    ) -> list[HuggingFaceSearchResult]:
        """Search HuggingFace Hub — tries mlx-community first, falls back to all of HuggingFace."""
        from huggingface_hub import ModelInfo, list_models

        def _to_results(models: Iterable[ModelInfo]) -> list[HuggingFaceSearchResult]:
            return [
                HuggingFaceSearchResult(
                    id=m.id,
                    author=m.author or "",
                    downloads=m.downloads or 0,
                    likes=m.likes or 0,
                    last_modified=str(m.last_modified or ""),
                    tags=list(m.tags or []),
                )
                for m in models
            ]

        # Search mlx-community first
        mlx_results = _to_results(
            list_models(
                search=query or None,
                author="mlx-community",
                sort="downloads",
                limit=limit,
            )
        )
        if mlx_results:
            return mlx_results

        # Fall back to searching all of HuggingFace
        return _to_results(
            list_models(
                search=query or None,
                sort="downloads",
                limit=limit,
            )
        )

    async def run(self):
        shutdown_ev = anyio.Event()

        try:
            async with self._tg as tg:
                logger.info("Starting API")
                tg.start_soon(self._apply_state)
                tg.start_soon(self._pause_on_new_election)
                tg.start_soon(self._cleanup_expired_images)
                print_startup_banner(self.port)
                tg.start_soon(self.run_api, shutdown_ev)
                try:
                    await anyio.sleep_forever()
                finally:
                    with anyio.CancelScope(shield=True):
                        # IMPORTANT: when new queues are added, update this (for proper shutdown semantics)
                        self._shutdown_queues(self._text_generation_queues)
                        self._shutdown_queues(self._image_generation_queues)

                        shutdown_ev.set()
        finally:
            self._event_log.close()
            self.command_sender.close()
            self.event_receiver.close()

    @staticmethod
    def _shutdown_queues[K, V](queues: dict[K, Sender[V]]):
        for v in queues.values():
            v.close()

    async def run_api(self, ev: anyio.Event):
        cfg = Config()
        cfg.bind = [f"0.0.0.0:{self.port}"]
        # nb: shared.logging needs updating if any of this changes
        cfg.accesslog = None
        cfg.errorlog = "-"
        cfg.logger_class = InterceptLogger

        # prevents hangs when mid-request and connection refuses to close
        cfg.graceful_timeout = 2  # seconds
        cfg.shutdown_timeout = 3  # seconds

        with anyio.CancelScope(shield=True):
            try:
                await serve(
                    cast(ASGIFramework, self.app),
                    cfg,
                    shutdown_trigger=ev.wait,
                )
                if not ev.is_set():
                    raise ShutdownError(
                        "Server exited without shutdown trigger - exiting abnormally"
                    )
            except LifespanTimeoutError as e:
                logger.warning(
                    "Graceful server shutdown timed out, some connections forcebly closed"
                )
                logger.opt(exception=e).debug("")

    async def _apply_state(self):
        with self.event_receiver as events:
            async for i_event in events:
                self._event_log.append(i_event.event)
                self.state = apply(self.state, i_event)
                event = i_event.event

                if isinstance(event, InstanceCreated):
                    model_key = str(event.instance.shard_assignments.model_id)
                    pending = self._pending_idle_policies.pop(model_key, None)
                    if pending is not None:
                        policy, _sent = pending
                        await self._send(
                            SetInstanceIdlePolicy(
                                instance_id=event.instance.instance_id,
                                policy=policy,
                            )
                        )
                    self._idle_last_activity[event.instance.instance_id] = time.time()
                elif isinstance(event, InstanceDeleted):
                    self._idle_last_activity.pop(event.instance_id, None)
                    self._idle_unload_inflight.discard(event.instance_id)

                if isinstance(event, InstanceTokensUpdated):
                    # Fold into the durable lifetime store too — the
                    # event-sourced per-instance totals reset when instances
                    # are recreated, this sidecar survives restarts.
                    instance = self.state.instances.get(event.instance_id)
                    if instance is not None:
                        self._fold_lifetime_usage(
                            instance,
                            event.prompt_tokens,
                            event.completion_tokens,
                        )
                if isinstance(event, ChunkGenerated):
                    if queue := self._image_generation_queues.get(
                        event.command_id, None
                    ):
                        assert isinstance(event.chunk, ImageChunk)
                        try:
                            await queue.send(event.chunk)
                        except (BrokenResourceError, ClosedResourceError):
                            self._image_generation_queues.pop(event.command_id, None)
                    if queue := self._text_generation_queues.get(
                        event.command_id, None
                    ):
                        assert not isinstance(event.chunk, ImageChunk)
                        try:
                            await queue.send(event.chunk)
                        except (BrokenResourceError, ClosedResourceError):
                            self._text_generation_queues.pop(event.command_id, None)
                if isinstance(event, InstanceDeleted):
                    self._close_streams_for_instance(event.instance_id)
                if isinstance(event, TracesMerged):
                    self._save_merged_trace(event)

    def _close_streams_for_instance(self, instance_id: InstanceId) -> None:
        """Close any active generation streams for commands running on the given instance."""
        for task in self.state.tasks.values():
            if task.instance_id != instance_id:
                continue
            if not isinstance(
                task, (TextGenerationTask, ImageGenerationTask, ImageEditsTask)
            ):
                continue
            if sender := self._text_generation_queues.pop(task.command_id, None):
                sender.close()
            if sender := self._image_generation_queues.pop(task.command_id, None):
                sender.close()

    def _save_merged_trace(self, event: TracesMerged) -> None:
        traces = [
            TraceEvent(
                name=t.name,
                start_us=t.start_us,
                duration_us=t.duration_us,
                rank=t.rank,
                category=t.category,
            )
            for t in event.traces
        ]
        output_path = EXO_TRACING_CACHE_DIR / f"trace_{event.task_id}.json"
        export_trace(traces, output_path)
        logger.debug(f"Saved merged trace to {output_path}")

    async def _pause_on_new_election(self):
        with self.election_receiver as ems:
            async for message in ems:
                if message.clock > self.last_completed_election:
                    self.paused = True

    async def _cleanup_expired_images(self):
        """Periodically clean up expired images from the store."""
        cleanup_interval_seconds = 300  # 5 minutes
        while True:
            await anyio.sleep(cleanup_interval_seconds)
            removed = self._image_store.cleanup_expired()
            if removed > 0:
                logger.debug(f"Cleaned up {removed} expired images")

    async def _send(self, command: Command):
        while self.paused:
            await self.paused_ev.wait()
        await self.command_sender.send(
            ForwarderCommand(origin=self._system_id, command=command)
        )

    async def _send_download(self, command: DownloadCommand):
        await self.download_command_sender.send(
            ForwarderDownloadCommand(origin=self._system_id, command=command)
        )

    async def start_download(
        self, payload: StartDownloadParams
    ) -> StartDownloadResponse:
        command = StartDownload(
            target_node_id=payload.target_node_id,
            shard_metadata=payload.shard_metadata,
        )
        await self._send_download(command)
        return StartDownloadResponse(command_id=command.command_id)

    async def delete_download(
        self, node_id: NodeId, model_id: ModelId
    ) -> DeleteDownloadResponse:
        command = DeleteDownload(
            target_node_id=node_id,
            model_id=ModelId(model_id),
        )
        await self._send_download(command)
        return DeleteDownloadResponse(command_id=command.command_id)

    async def cancel_download(
        self,
        payload: CancelDownloadParams,
    ) -> CancelDownloadResponse:
        command = CancelDownload(
            target_node_id=payload.target_node_id,
            model_id=payload.model_id,
        )
        await self._send_download(command)
        return CancelDownloadResponse(command_id=command.command_id)

    @staticmethod
    def _get_trace_path(task_id: str) -> Path:
        trace_path = EXO_TRACING_CACHE_DIR / f"trace_{task_id}.json"
        if not trace_path.resolve().is_relative_to(EXO_TRACING_CACHE_DIR.resolve()):
            raise HTTPException(status_code=400, detail=f"Invalid task ID: {task_id}")
        return trace_path

    async def list_traces(self) -> TraceListResponse:
        traces: list[TraceListItem] = []

        for trace_file in sorted(
            EXO_TRACING_CACHE_DIR.glob("trace_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            # Extract task_id from filename (trace_{task_id}.json)
            task_id = trace_file.stem.removeprefix("trace_")
            stat = trace_file.stat()
            created_at = datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat()
            traces.append(
                TraceListItem(
                    task_id=task_id,
                    created_at=created_at,
                    file_size=stat.st_size,
                )
            )

        return TraceListResponse(traces=traces)

    async def get_trace(self, task_id: str) -> TraceResponse:
        trace_path = self._get_trace_path(task_id)

        if not trace_path.exists():
            raise HTTPException(status_code=404, detail=f"Trace not found: {task_id}")

        trace_events = load_trace_file(trace_path)

        return TraceResponse(
            task_id=task_id,
            traces=[
                TraceEventResponse(
                    name=event.name,
                    start_us=event.start_us,
                    duration_us=event.duration_us,
                    rank=event.rank,
                    category=event.category,
                )
                for event in trace_events
            ],
        )

    async def get_trace_stats(self, task_id: str) -> TraceStatsResponse:
        trace_path = self._get_trace_path(task_id)

        if not trace_path.exists():
            raise HTTPException(status_code=404, detail=f"Trace not found: {task_id}")

        trace_events = load_trace_file(trace_path)
        stats = compute_stats(trace_events)

        return TraceStatsResponse(
            task_id=task_id,
            total_wall_time_us=stats.total_wall_time_us,
            by_category={
                category: TraceCategoryStats(
                    total_us=cat_stats.total_us,
                    count=cat_stats.count,
                    min_us=cat_stats.min_us,
                    max_us=cat_stats.max_us,
                    avg_us=cat_stats.avg_us,
                )
                for category, cat_stats in stats.by_category.items()
            },
            by_rank={
                rank: TraceRankStats(
                    by_category={
                        category: TraceCategoryStats(
                            total_us=cat_stats.total_us,
                            count=cat_stats.count,
                            min_us=cat_stats.min_us,
                            max_us=cat_stats.max_us,
                            avg_us=cat_stats.avg_us,
                        )
                        for category, cat_stats in rank_stats.items()
                    }
                )
                for rank, rank_stats in stats.by_rank.items()
            },
        )

    async def get_trace_raw(self, task_id: str) -> FileResponse:
        trace_path = self._get_trace_path(task_id)

        if not trace_path.exists():
            raise HTTPException(status_code=404, detail=f"Trace not found: {task_id}")

        return FileResponse(
            path=trace_path,
            media_type="application/json",
            filename=f"trace_{task_id}.json",
        )

    async def delete_traces(self, request: DeleteTracesRequest) -> DeleteTracesResponse:
        deleted: list[str] = []
        not_found: list[str] = []
        for task_id in request.task_ids:
            trace_path = self._get_trace_path(task_id)
            if trace_path.exists():
                trace_path.unlink()
                deleted.append(task_id)
            else:
                not_found.append(task_id)
        return DeleteTracesResponse(deleted=deleted, not_found=not_found)

    async def get_onboarding(self) -> JSONResponse:
        return JSONResponse({"completed": ONBOARDING_COMPLETE_FILE.exists()})

    async def complete_onboarding(self) -> JSONResponse:
        ONBOARDING_COMPLETE_FILE.parent.mkdir(parents=True, exist_ok=True)
        ONBOARDING_COMPLETE_FILE.write_text("true")
        return JSONResponse({"completed": True})
