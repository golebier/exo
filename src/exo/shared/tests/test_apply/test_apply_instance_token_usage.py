from exo.shared.apply import (
    apply_instance_deleted,
    apply_instance_tokens_updated,
    event_apply,
)
from exo.shared.types.events import (
    InstanceDeleted,
    InstanceTokensUpdated,
)
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.token_usage import InstanceTokenUsage


def _delta(
    instance_id: InstanceId, prompt: int, completion: int, cached: int = 0
) -> InstanceTokensUpdated:
    return InstanceTokensUpdated(
        instance_id=instance_id,
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_tokens=cached,
    )


def test_accumulate_single_instance() -> None:
    state = State()
    instance_id = InstanceId("inst-1")

    state = apply_instance_tokens_updated(_delta(instance_id, 10, 4), state)
    usage = state.instance_token_usage[instance_id]
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 4
    assert usage.total_tokens == 14
    assert usage.request_count == 1
    assert usage.cached_tokens == 0

    state = apply_instance_tokens_updated(_delta(instance_id, 5, 2), state)
    usage = state.instance_token_usage[instance_id]
    assert usage.prompt_tokens == 15
    assert usage.completion_tokens == 6
    assert usage.total_tokens == 21
    assert usage.request_count == 2


def test_accumulate_multiple_instances_independently() -> None:
    state = State()
    a = InstanceId("inst-a")
    b = InstanceId("inst-b")

    state = apply_instance_tokens_updated(_delta(a, 10, 4), state)
    state = apply_instance_tokens_updated(_delta(b, 100, 40), state)
    state = apply_instance_tokens_updated(_delta(a, 1, 1), state)

    assert state.instance_token_usage[a].total_tokens == 16
    assert state.instance_token_usage[a].request_count == 2
    assert state.instance_token_usage[b].total_tokens == 140
    assert state.instance_token_usage[b].request_count == 1


def test_instance_deleted_clears_token_usage() -> None:
    state = State()
    a = InstanceId("inst-a")
    b = InstanceId("inst-b")

    state = apply_instance_tokens_updated(_delta(a, 10, 4), state)
    state = apply_instance_tokens_updated(_delta(b, 100, 40), state)

    state = apply_instance_deleted(InstanceDeleted(instance_id=a), state)

    assert a not in state.instance_token_usage
    assert b in state.instance_token_usage


def test_event_apply_routes_to_handler() -> None:
    """The discriminated union dispatch in `event_apply` must handle the new event."""
    state = State()
    instance_id = InstanceId("inst-1")

    state = event_apply(_delta(instance_id, 7, 3), state)
    assert state.instance_token_usage[instance_id].total_tokens == 10


def test_replay_reproduces_cumulative_counts() -> None:
    """Replaying the event log must reproduce identical cumulative counts."""
    instance_id = InstanceId("inst-replay")
    events = [
        _delta(instance_id, 10, 4),
        _delta(instance_id, 5, 2),
        _delta(instance_id, 20, 8),
    ]

    state = State()
    for event in events:
        state = event_apply(event, state)

    usage = state.instance_token_usage[instance_id]
    assert usage.prompt_tokens == 35
    assert usage.completion_tokens == 14
    assert usage.total_tokens == 49
    assert usage.request_count == 3


def test_accumulate_cached_tokens() -> None:
    """Cached tokens accumulate and derive a cache-efficiency ratio."""
    state = State()
    instance_id = InstanceId("inst-cache")

    state = apply_instance_tokens_updated(
        _delta(instance_id, prompt=100, completion=4, cached=80), state
    )
    state = apply_instance_tokens_updated(
        _delta(instance_id, prompt=50, completion=2, cached=45), state
    )

    usage = state.instance_token_usage[instance_id]
    assert usage.prompt_tokens == 150
    assert usage.cached_tokens == 125
    assert usage.request_count == 2
    assert usage.cached_tokens / usage.prompt_tokens == 125 / 150


def test_old_events_without_cached_tokens_replay_as_zero() -> None:
    """Persisted events from before ``cached_tokens`` existed must still replay."""
    instance_id = InstanceId("inst-legacy")
    legacy_event_json = (
        '{"InstanceTokensUpdated": {'
        f'"instanceId": "{instance_id}",'
        '"promptTokens": 10,'
        '"completionTokens": 4'
        "}}"
    )
    event = InstanceTokensUpdated.model_validate_json(legacy_event_json)

    state = apply_instance_tokens_updated(event, State())
    usage = state.instance_token_usage[instance_id]
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 4
    assert usage.cached_tokens == 0


def test_state_serialization_roundtrip_preserves_usage() -> None:
    instance_id = InstanceId("inst-rt")
    state = State()
    state = apply_instance_tokens_updated(_delta(instance_id, 12, 5), state)

    restored = State.model_validate_json(state.model_dump_json())

    assert restored.instance_token_usage[instance_id] == InstanceTokenUsage(
        instance_id=instance_id,
        prompt_tokens=12,
        completion_tokens=5,
        total_tokens=17,
        request_count=1,
        cached_tokens=0,
    )
