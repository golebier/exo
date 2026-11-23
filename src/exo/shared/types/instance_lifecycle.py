from exo.utils.pydantic_ext import FrozenModel


class InstanceIdlePolicy(FrozenModel):
    """Per-instance idle-unload policy.

    ``idle_unload_seconds=None`` means the instance is never unloaded
    automatically (default — preserves existing behaviour). Otherwise the
    instance may be deleted by the API idle-unload sweeper once it has been
    idle (no in-flight requests, no runner marked running) for at least the
    given number of seconds.
    """

    idle_unload_seconds: int | None = None
    # Placement parameters captured at creation time so the sweeper can
    # transparently re-place the instance on the next request after an
    # idle unload.
    sharding: str = "Pipeline"
    instance_meta: str = "MlxRing"
    min_nodes: int = 1


class InstanceLifecycleStats(FrozenModel):
    """Load/unload counters for one instance lifecycle ledger entry."""

    load_count: int = 0
    unload_count: int = 0
    last_load_ts: float | None = None
    last_unload_ts: float | None = None
    last_unload_reason: str | None = None


class InstanceUnloadReason:
    """Reason strings for lifecycle ledger entries."""

    PLACEMENT = "placement"
    IDLE_UNLOAD = "idle_unload"
    MANUAL = "manual"
    ERROR = "error"
