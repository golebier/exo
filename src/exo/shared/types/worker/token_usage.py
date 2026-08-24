from exo.shared.types.worker.instances import InstanceId
from exo.utils.pydantic_ext import FrozenModel


class InstanceTokenUsage(FrozenModel):
    """Cumulative token accounting for a single instance.

    Accumulated across every completed generation attributed to the instance,
    from creation until the instance is deleted. The counter is event-sourced:
    each :class:`~exo.shared.types.events.InstanceTokensUpdated` event carries
    a delta for one request and :func:`~exo.shared.apply.apply` folds it into
    this running total.
    """

    instance_id: InstanceId
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    request_count: int
    # Share of ``prompt_tokens`` served from the KV prefix cache rather than
    # recomputed. Defaults to 0 so persisted state from before this field
    # existed still deserializes.
    cached_tokens: int = 0
