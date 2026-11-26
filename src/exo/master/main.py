import time
from datetime import datetime, timedelta, timezone

import anyio
from loguru import logger

from exo.master.placement import (
    add_instance_to_placements,
    cancel_unnecessary_downloads,
    delete_instance,
    get_transition_events,
    place_instance,
)
from exo.master.placement_utils import find_ip_prioritised
from exo.routing.event_router import (
    EventRouterBrokenResourceError,
    EventRouterClosedResourceError,
)
from exo.shared.apply import apply
from exo.shared.constants import EXO_EVENT_LOG_DIR, EXO_TRACING_ENABLED
from exo.shared.models.model_cards import card_cache
from exo.shared.types.commands import (
    AddCustomModelCard,
    CreateInstance,
    DeleteCustomModelCard,
    DeleteInstance,
    DeleteInstanceLink,
    ForwarderCommand,
    ForwarderDownloadCommand,
    ImageEdits,
    ImageGeneration,
    LoadInstanceWeights,
    PlaceInstance,
    RequestEventLog,
    SendInputChunk,
    SetInstanceIdlePolicy,
    SetInstanceLink,
    SetRamGuard,
    TaskCancelled,
    TaskFinished,
    TestCommand,
    TextGeneration,
)
from exo.shared.types.common import CommandId, ModelId, NodeId, SessionId, SystemId
from exo.shared.types.events import (
    CustomModelCardAdded,
    CustomModelCardDeleted,
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    InputChunkReceived,
    InstanceCreated,
    InstanceDeleted,
    InstanceIdlePolicySet,
    InstanceLifecycleRecorded,
    InstanceLinkCreated,
    InstanceLinkDeleted,
    InstanceWeightsLoadRequested,
    InstanceWeightsUnloaded,
    LocalForwarderEvent,
    NodeGatheredInfo,
    NodeTimedOut,
    RamGuardSet,
    TaskCreated,
    TaskDeleted,
    TaskStatusUpdated,
    TraceEventData,
    TracesCollected,
    TracesMerged,
)
from exo.shared.types.instance_lifecycle import InstanceUnloadReason
from exo.shared.types.instance_link import InstanceLink
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    ImageEdits as ImageEditsTask,
)
from exo.shared.types.tasks import (
    ImageGeneration as ImageGenerationTask,
)
from exo.shared.types.tasks import (
    Shutdown,
    TaskId,
    TaskStatus,
)
from exo.shared.types.tasks import (
    TextGeneration as TextGenerationTask,
)
from exo.shared.types.worker.instances import (
    InstanceId,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.utils.channels import Receiver, Sender
from exo.utils.disk_event_log import DiskEventLog
from exo.utils.event_buffer import MultiSourceBuffer
from exo.utils.task_group import TaskGroup


def _prefill_endpoint_for(state: State, decode_instance_id: InstanceId) -> str | None:
    decode = state.instances.get(decode_instance_id)
    if decode is None:
        return None
    decode_node = next(iter(decode.shard_assignments.node_to_runner.keys()), None)
    if decode_node is None:
        return None

    sources: set[InstanceId] = set()
    for link in state.instance_links.values():
        if decode_instance_id in link.decode_instances:
            sources.update(link.prefill_instances)
    sources.discard(decode_instance_id)

    in_flight = {TaskStatus.Pending, TaskStatus.Running}
    task_counts: dict[InstanceId, int] = {
        src_id: sum(
            1
            for task in state.tasks.values()
            if task.instance_id == src_id and task.task_status in in_flight
        )
        for src_id in sources
    }
    for src_id in sorted(sources, key=lambda sid: task_counts[sid]):
        instance = state.instances.get(src_id)
        if instance is None:
            continue
        for node_id, runner_id in instance.shard_assignments.node_to_runner.items():
            port = state.prefill_server_ports.get(runner_id)
            if port is None:
                continue
            ip = find_ip_prioritised(
                decode_node, node_id, state.topology, state.node_network, ring=True
            )
            if ip is None:
                continue
            return f"{ip}:{port}"
    return None


class Master:
    def __init__(
        self,
        node_id: NodeId,
        session_id: SessionId,
        *,
        command_receiver: Receiver[ForwarderCommand],
        event_sender: Sender[Event],
        local_event_receiver: Receiver[LocalForwarderEvent],
        global_event_sender: Sender[GlobalForwarderEvent],
        download_command_sender: Sender[ForwarderDownloadCommand],
    ):
        self.node_id = node_id
        self.session_id = session_id
        self.state = State()
        self._tg: TaskGroup = TaskGroup()
        self.command_task_mapping: dict[CommandId, TaskId] = {}
        self.command_receiver = command_receiver
        self.local_event_receiver = local_event_receiver
        self.global_event_sender = global_event_sender
        self.download_command_sender = download_command_sender
        self.event_sender = event_sender
        self._system_id = SystemId()
        self._multi_buffer = MultiSourceBuffer[SystemId, Event]()
        self._event_log = DiskEventLog(EXO_EVENT_LOG_DIR / "master")
        self._pending_traces: dict[TaskId, dict[int, list[TraceEventData]]] = {}
        self._expected_ranks: dict[TaskId, set[int]] = {}
        self._idle_last_activity: dict[InstanceId, float] = {}
        # Consecutive stale observations per node before removal. A single
        # stale window is not enough: a transient gather stall (thread-pool
        # starvation, short GC pause, busy subprocess) can exceed one window
        # while the node is otherwise healthy. Removing on the first miss
        # causes node flapping — the node re-registers on the next gather,
        # its instances resurrect, and any pending deletions loop forever.
        self._node_stale_strikes: dict[NodeId, int] = {}

    async def run(self):
        logger.info("Starting Master")

        try:
            async with self._tg as tg:
                tg.start_soon(self._event_processor)
                tg.start_soon(self._command_processor)
                tg.start_soon(self._plan)
        except* (EventRouterBrokenResourceError, EventRouterClosedResourceError):
            # Event router has been closed (try-star syntax handles error groups)
            pass
        finally:
            self._event_log.close()
            self.global_event_sender.close()
            self.local_event_receiver.close()
            self.command_receiver.close()

    async def shutdown(self):
        logger.info("Stopping Master")
        self._tg.cancel_tasks()

    async def _command_processor(self) -> None:
        with self.command_receiver as commands:
            async for forwarder_command in commands:
                try:
                    logger.info(f"Executing command: {forwarder_command.command}")

                    generated_events: list[Event] = []
                    command = forwarder_command.command
                    instance_task_counts: dict[InstanceId, int] = {}
                    match command:
                        case TestCommand():
                            pass
                        case TextGeneration():
                            # set-difference => prefill-only nodes
                            prefill_only: set[InstanceId] = set()
                            for link in self.state.instance_links.values():
                                prefill_only.update(link.prefill_instances)
                            for link in self.state.instance_links.values():
                                prefill_only.difference_update(link.decode_instances)

                            for instance in self.state.instances.values():
                                # NON-prefill-only instances matching the model ID
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                    and instance.instance_id not in prefill_only
                                    and instance.instance_id
                                    not in self.state.instance_weights_unloaded
                                ):
                                    # count in-flight tasks of that instance
                                    in_flight = {TaskStatus.Pending, TaskStatus.Running}
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                        and task.task_status in in_flight
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            # there are no NON-prefill-only instances matching this model ID
                            if not instance_task_counts:
                                await self._reload_unloaded_for_model(
                                    command.task_params.model
                                )
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            available_instance_ids = sorted(
                                instance_task_counts.keys(),
                                key=lambda instance_id: instance_task_counts[
                                    instance_id
                                ],
                            )

                            decode_instance_id = available_instance_ids[0]
                            task_id = TaskId()
                            params = command.task_params.model_copy(
                                update={
                                    "prefill_endpoint": _prefill_endpoint_for(
                                        self.state, decode_instance_id
                                    ),
                                }
                            )
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=TextGenerationTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        instance_id=decode_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=params,
                                    ),
                                )
                            )
                            self.command_task_mapping[command.command_id] = task_id
                        case ImageGeneration():
                            unloaded_match = False
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                ):
                                    if (
                                        instance.instance_id
                                        in self.state.instance_weights_unloaded
                                    ):
                                        unloaded_match = True
                                        continue
                                    in_flight = {TaskStatus.Pending, TaskStatus.Running}
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                        and task.task_status in in_flight
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if not instance_task_counts:
                                # Unloaded instances still have records in state, so an
                                # unloaded match must trigger a reload here or requests
                                # will be pinned to a zombie instance forever.
                                if unloaded_match:
                                    await self._reload_unloaded_for_model(
                                        command.task_params.model
                                    )
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            available_instance_ids = sorted(
                                instance_task_counts.keys(),
                                key=lambda instance_id: instance_task_counts[
                                    instance_id
                                ],
                            )

                            task_id = TaskId()
                            selected_instance_id = available_instance_ids[0]
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=ImageGenerationTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        instance_id=selected_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=command.task_params,
                                    ),
                                )
                            )

                            self.command_task_mapping[command.command_id] = task_id

                            if EXO_TRACING_ENABLED:
                                selected_instance = self.state.instances.get(
                                    selected_instance_id
                                )
                                if selected_instance:
                                    ranks = set(
                                        shard.device_rank
                                        for shard in selected_instance.shard_assignments.runner_to_shard.values()
                                    )
                                    self._expected_ranks[task_id] = ranks
                        case ImageEdits():
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                ):
                                    in_flight = {TaskStatus.Pending, TaskStatus.Running}
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                        and task.task_status in in_flight
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if not instance_task_counts:
                                await self._reload_unloaded_for_model(
                                    command.task_params.model
                                )
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            available_instance_ids = sorted(
                                instance_task_counts.keys(),
                                key=lambda instance_id: instance_task_counts[
                                    instance_id
                                ],
                            )

                            task_id = TaskId()
                            selected_instance_id = available_instance_ids[0]
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=ImageEditsTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        instance_id=selected_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=command.task_params,
                                    ),
                                )
                            )

                            self.command_task_mapping[command.command_id] = task_id

                            if EXO_TRACING_ENABLED:
                                selected_instance = self.state.instances.get(
                                    selected_instance_id
                                )
                                if selected_instance:
                                    ranks = set(
                                        shard.device_rank
                                        for shard in selected_instance.shard_assignments.runner_to_shard.values()
                                    )
                                    self._expected_ranks[task_id] = ranks
                        case DeleteInstance():
                            placement = delete_instance(command, self.state.instances)
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            for cmd in cancel_unnecessary_downloads(
                                placement, self.state.downloads
                            ):
                                await self.download_command_sender.send(
                                    ForwarderDownloadCommand(
                                        origin=self._system_id, command=cmd
                                    )
                                )
                            generated_events.extend(transition_events)
                        case PlaceInstance():
                            placement = place_instance(
                                command,
                                self.state.topology,
                                self.state.instances,
                                self.state.node_memory,
                                self.state.node_network,
                                self.state.node_backends,
                                download_status=self.state.downloads,
                                node_rdma_ctl=self.state.node_rdma_ctl,
                            )
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            generated_events.extend(transition_events)
                        case CreateInstance():
                            placement = add_instance_to_placements(
                                command,
                                self.state.topology,
                                self.state.instances,
                            )
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            generated_events.extend(transition_events)
                        case SendInputChunk(chunk=chunk):
                            generated_events.append(
                                InputChunkReceived(
                                    command_id=chunk.command_id,
                                    chunk=chunk,
                                )
                            )
                        case TaskCancelled():
                            if (
                                task_id := self.command_task_mapping.get(
                                    command.cancelled_command_id
                                )
                            ) is not None:
                                generated_events.append(
                                    TaskStatusUpdated(
                                        task_status=TaskStatus.Cancelled,
                                        task_id=task_id,
                                    )
                                )
                            else:
                                logger.warning(
                                    f"Nonexistent command {command.cancelled_command_id} cancelled"
                                )
                        case TaskFinished():
                            if (
                                task_id := self.command_task_mapping.pop(
                                    command.finished_command_id, None
                                )
                            ) is not None:
                                generated_events.append(TaskDeleted(task_id=task_id))
                            else:
                                logger.warning(
                                    f"Finished command {command.finished_command_id} finished"
                                )

                        case AddCustomModelCard():
                            generated_events.append(
                                CustomModelCardAdded(model_card=command.model_card)
                            )
                        case DeleteCustomModelCard():
                            generated_events.append(
                                CustomModelCardDeleted(model_id=command.model_id)
                            )
                        case SetInstanceLink():
                            link = InstanceLink(
                                link_id=command.link_id,
                                prefill_instances=list(
                                    dict.fromkeys(command.prefill_instances)
                                ),
                                decode_instances=list(
                                    dict.fromkeys(command.decode_instances)
                                ),
                            )
                            generated_events.append(InstanceLinkCreated(link=link))
                        case DeleteInstanceLink():
                            generated_events.append(
                                InstanceLinkDeleted(link_id=command.link_id)
                            )
                        case LoadInstanceWeights():
                            if command.instance_id in self.state.instances:
                                instance = self.state.instances[command.instance_id]
                                generated_events.append(
                                    InstanceWeightsLoadRequested(
                                        instance_id=command.instance_id
                                    )
                                )
                                generated_events.append(
                                    InstanceLifecycleRecorded(
                                        model_id=str(
                                            instance.shard_assignments.model_id
                                        ),
                                        instance_type=type(instance).__name__,
                                        phase="load",
                                        reason=InstanceUnloadReason.MANUAL,
                                        ts=time.time(),
                                    )
                                )
                        case SetInstanceIdlePolicy():
                            generated_events.append(
                                InstanceIdlePolicySet(
                                    instance_id=command.instance_id,
                                    policy=command.policy,
                                )
                            )
                        case SetRamGuard():
                            generated_events.append(
                                RamGuardSet(percent=command.percent)
                            )
                        case RequestEventLog():
                            # We should just be able to send everything, since other buffers will ignore old messages
                            # rate limit to 1000 at a time
                            end = min(command.since_idx + 1000, len(self._event_log))
                            for i, event in enumerate(
                                self._event_log.read_range(command.since_idx, end),
                                start=command.since_idx,
                            ):
                                await self._send_indexed_event(
                                    IndexedEvent(idx=i, event=event)
                                )
                    for event in generated_events:
                        await self.event_sender.send(event)
                    for event in generated_events:
                        match event:
                            case InstanceCreated():
                                await self.event_sender.send(
                                    InstanceLifecycleRecorded(
                                        model_id=str(
                                            event.instance.shard_assignments.model_id
                                        ),
                                        instance_type=type(event.instance).__name__,
                                        phase="load",
                                        reason=InstanceUnloadReason.PLACEMENT,
                                        ts=time.time(),
                                    )
                                )
                            case InstanceDeleted():
                                await self.event_sender.send(
                                    InstanceLifecycleRecorded(
                                        model_id=str(
                                            self.state.instances[
                                                event.instance_id
                                            ].shard_assignments.model_id
                                        )
                                        if event.instance_id in self.state.instances
                                        else "unknown",
                                        instance_type=type(
                                            self.state.instances[event.instance_id]
                                        ).__name__
                                        if event.instance_id in self.state.instances
                                        else "unknown",
                                        phase="unload",
                                        reason=InstanceUnloadReason.MANUAL,
                                        ts=time.time(),
                                    )
                                )
                            case InstanceWeightsUnloaded():
                                instance = self.state.instances.get(event.instance_id)
                                if instance is not None:
                                    await self.event_sender.send(
                                        InstanceLifecycleRecorded(
                                            model_id=str(
                                                instance.shard_assignments.model_id
                                            ),
                                            instance_type=type(instance).__name__,
                                            phase="unload",
                                            reason=InstanceUnloadReason.IDLE_UNLOAD,
                                            ts=time.time(),
                                        )
                                    )
                            case InstanceWeightsLoadRequested():
                                instance = self.state.instances.get(event.instance_id)
                                if instance is not None:
                                    await self.event_sender.send(
                                        InstanceLifecycleRecorded(
                                            model_id=str(
                                                instance.shard_assignments.model_id
                                            ),
                                            instance_type=type(instance).__name__,
                                            phase="load",
                                            reason=InstanceUnloadReason.IDLE_RELOAD,
                                            ts=time.time(),
                                        )
                                    )
                            case _:
                                pass
                except Exception as e:
                    logger.opt(exception=e).warning("Error in command processor")

    async def _reload_unloaded_for_model(self, model_id: ModelId | str) -> None:
        """Auto-reload an unloaded instance when a request targets its model.

        The request itself fails fast (caller raises); the instance flips
        back to loaded in the background so the retry/re-request succeeds
        once the weights are resident again. The event is sent directly —
        a fail-fast raise must never swallow it.
        """
        for instance_id, instance in self.state.instances.items():
            if (
                instance.shard_assignments.model_id == model_id
                and instance_id in self.state.instance_weights_unloaded
            ):
                logger.info(
                    f"Auto-reloading unloaded instance {instance_id} for {model_id}"
                )
                await self.event_sender.send(
                    InstanceWeightsLoadRequested(instance_id=instance_id)
                )
                return

    # These plan loops are the cracks showing in our event sourcing architecture - more things could be commands
    async def _plan(self) -> None:
        while True:
            # kill broken instances
            connected_node_ids = set(self.state.topology.list_nodes())
            for instance_id, instance in self.state.instances.items():
                for node_id in instance.shard_assignments.node_to_runner:
                    if node_id not in connected_node_ids:
                        await self.event_sender.send(
                            InstanceDeleted(instance_id=instance_id)
                        )
                        break

            # idle-unload sweeper: free the weights of idle instances
            for instance_id in list(self.state.instances):
                if instance_id in self.state.instance_weights_unloaded:
                    # Already unloaded (weights are back only after an
                    # InstanceWeightsLoadRequested clears the flag) - re-emitting
                    # the unload every tick would spam workers and the event log.
                    continue
                policy = self.state.instance_idle_policies.get(instance_id)
                if policy is None or policy.idle_unload_seconds is None:
                    continue
                # Start the idle clock at the first sweeper tick — instances
                # that never ran a task since master boot would otherwise
                # never reach their timeout.
                self._idle_last_activity.setdefault(instance_id, time.time())
                # Busy while a task that consumes or loads weights is
                # pending/running. Shutdown is worker-side runner teardown
                # (it can linger as Running) and must NOT look like usage,
                # or the instance could never unload again after a reload.
                busy = any(
                    task.instance_id == instance_id
                    and task.task_status in (TaskStatus.Pending, TaskStatus.Running)
                    and not isinstance(task, Shutdown)
                    for task in self.state.tasks.values()
                )
                if busy:
                    self._idle_last_activity[instance_id] = time.time()
                    continue
                idle_for = time.time() - self._idle_last_activity[instance_id]
                if idle_for < policy.idle_unload_seconds:
                    continue
                logger.info(
                    f"Idle-unloading instance {instance_id} after {int(idle_for)}s idle"
                )
                # Keep the instance (shard assignments, links, policy survive);
                # only the weights are freed. Workers see the flag and shut
                # their runner processes down.
                await self.event_sender.send(
                    InstanceWeightsUnloaded(instance_id=instance_id)
                )
                await self.event_sender.send(
                    InstanceLifecycleRecorded(
                        model_id=str(
                            self.state.instances[instance_id].shard_assignments.model_id
                        ),
                        instance_type=type(self.state.instances[instance_id]).__name__,
                        phase="unload",
                        reason=InstanceUnloadReason.IDLE_UNLOAD,
                        ts=time.time(),
                    )
                )

            # RAM guard sweeper: when a node's RAM usage crosses the guard
            # threshold, free the weights of the least-recently-active loaded
            # instance on that node (one per tick — the next tick re-checks
            # with fresh memory numbers). This is the last line of defense
            # against a reload/large-model fit thrashing the node into a
            # swap storm that takes the whole machine down.
            guard_percent = self.state.ram_guard_percent
            if guard_percent is not None:
                node_busy: dict[InstanceId, bool] = {
                    instance_id: any(
                        task.instance_id == instance_id
                        and task.task_status in (TaskStatus.Pending, TaskStatus.Running)
                        and not isinstance(task, Shutdown)
                        for task in self.state.tasks.values()
                    )
                    for instance_id in self.state.instances
                }
                for node_id, mem in list(self.state.node_memory.items()):
                    total = mem.ram_total.in_bytes
                    available = mem.ram_available.in_bytes
                    if total <= 0:
                        continue
                    used_pct = (total - available) / total * 100
                    if used_pct < guard_percent:
                        continue
                    # Victim: largest loaded instance on this node that has
                    # no in-flight task (unloading mid-generation would crash
                    # the worker). Unloaded instances are not candidates.
                    victims: list[tuple[int, InstanceId]] = []
                    for instance_id, instance in self.state.instances.items():
                        if instance_id in self.state.instance_weights_unloaded:
                            continue
                        if node_id not in instance.shard_assignments.node_to_runner:
                            continue
                        if node_busy.get(instance_id, False):
                            continue
                        card = card_cache.get(instance.shard_assignments.model_id)
                        size = card.storage_size.in_bytes if card else 0
                        victims.append((size, instance_id))
                    if not victims:
                        logger.warning(
                            f"RAM guard: node {node_id} at {used_pct:.0f}% RAM "
                            "but no safely unloadable instance found"
                        )
                        continue
                    victims.sort(reverse=True)
                    victim_id = victims[0][1]
                    victim_model = str(
                        self.state.instances[victim_id].shard_assignments.model_id
                    )
                    logger.warning(
                        f"RAM guard: node {node_id} at {used_pct:.0f}% RAM "
                        f"(threshold {guard_percent}%) - unloading {victim_model} "
                        f"({victim_id})"
                    )
                    await self.event_sender.send(
                        InstanceWeightsUnloaded(instance_id=victim_id)
                    )
                    await self.event_sender.send(
                        InstanceLifecycleRecorded(
                            model_id=victim_model,
                            instance_type=type(
                                self.state.instances[victim_id]
                            ).__name__,
                            phase="unload",
                            reason=InstanceUnloadReason.RAM_GUARD,
                            ts=time.time(),
                        )
                    )
                    break  # one unload per tick; next tick re-reads memory

            # Converge zombie instances: an instance whose every host node
            # has left the topology can never run again, but it also never
            # times out on its own — it resurrects whenever a stale worker
            # re-registers, and any pending deletion loops forever. Delete
            # it directly so state converges even if command processing is
            # degraded. Idle-unloaded instances keep their hosts and are
            # untouched by this check. NOTE: the "kill broken instances"
            # block above only inspects nodes that still hold runners
            # (node_to_runner); this block covers instances whose hosts are
            # entirely gone, including ones never re-assigned after flap.
            topology_nodes = set(self.state.topology.list_nodes())
            for instance_id, instance in self.state.instances.items():
                match instance:
                    case MlxRingInstance():
                        host_nodes = set(instance.hosts_by_node.keys())
                    case MlxJacclInstance():
                        host_nodes = set(instance.jaccl_coordinators.keys())
                if not host_nodes & topology_nodes:
                    logger.warning(
                        f"Deleting zombie instance {instance_id} — all host "
                        f"nodes {host_nodes} are absent from the topology"
                    )
                    await self.event_sender.send(
                        InstanceDeleted(instance_id=instance_id)
                    )

            # time out dead nodes
            for node_id, last_seen in self.state.last_seen.items():
                now = datetime.now(tz=timezone.utc)
                if now - last_seen > timedelta(seconds=30):
                    strikes = self._node_stale_strikes.get(node_id, 0) + 1
                    self._node_stale_strikes[node_id] = strikes
                    if strikes >= 6:
                        logger.info(
                            f"Manually removing node {node_id} due to inactivity "
                            f"(stale for {strikes} consecutive windows)"
                        )
                        await self.event_sender.send(NodeTimedOut(node_id=node_id))
                else:
                    self._node_stale_strikes.pop(node_id, None)

            await anyio.sleep(10)

    async def _event_processor(self) -> None:
        with self.local_event_receiver as local_events:
            async for local_event in local_events:
                # Discard all events not from our session
                if local_event.session != self.session_id:
                    continue
                self._multi_buffer.ingest(
                    local_event.origin_idx,
                    local_event.event,
                    local_event.origin,
                )
                for event in self._multi_buffer.drain():
                    if isinstance(event, TracesCollected):
                        await self._handle_traces_collected(event)
                        continue

                    logger.debug(f"Master indexing event: {str(event)[:100]}")

                    event = event.model_copy(
                        update={"_master_time_stamp": datetime.now(tz=timezone.utc)}
                    )
                    if isinstance(event, NodeGatheredInfo):
                        event = event.model_copy(
                            update={"when": str(datetime.now(tz=timezone.utc))}
                        )

                    indexed = IndexedEvent(event=event, idx=len(self._event_log))
                    self.state = apply(self.state, indexed)

                    self._event_log.append(event)
                    await self._send_indexed_event(indexed)

    # This function is re-entrant, take care!
    async def _send_indexed_event(self, event: IndexedEvent):
        # Convenience method since this line is ugly
        await self.global_event_sender.send(
            GlobalForwarderEvent(
                origin=self.node_id,
                origin_idx=event.idx,
                session=self.session_id,
                event=event.event,
            )
        )

    async def _handle_traces_collected(self, event: TracesCollected) -> None:
        task_id = event.task_id
        if task_id not in self._pending_traces:
            self._pending_traces[task_id] = {}
        self._pending_traces[task_id][event.rank] = event.traces

        if (
            task_id in self._expected_ranks
            and set(self._pending_traces[task_id].keys())
            >= self._expected_ranks[task_id]
        ):
            await self._merge_and_save_traces(task_id)

    async def _merge_and_save_traces(self, task_id: TaskId) -> None:
        all_trace_data: list[TraceEventData] = []
        for trace_data in self._pending_traces[task_id].values():
            all_trace_data.extend(trace_data)

        await self.event_sender.send(
            TracesMerged(task_id=task_id, traces=all_trace_data)
        )

        del self._pending_traces[task_id]
        if task_id in self._expected_ranks:
            del self._expected_ranks[task_id]
