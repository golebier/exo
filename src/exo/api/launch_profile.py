"""Durable launch profile — remembers how instances were placed last session.

Every successful placement (POST /instance or POST /place_instance) upserts a
recipe entry keyed by model id. The profile survives master restarts (plain
JSON sidecar in the event-log dir, same pattern as the lifetime-usage store)
and powers one-click "restore last session": re-issuing PlaceInstance with the
exact sharding/instance_meta/min_nodes (and idle-unload policy) the user used
before, so they never repeat the same launch steps manually.
"""

import json
from pathlib import Path
from typing import cast

from exo.shared.constants import EXO_EVENT_LOG_DIR
from exo.utils.pydantic_ext import FrozenModel

_PROFILE_FILE = EXO_EVENT_LOG_DIR / "launch_profile.json"


class LaunchProfileEntry(FrozenModel):
    model_id: str
    sharding: str  # "Pipeline" | "Tensor"
    instance_meta: str  # "MlxRing" | "MlxJaccl"
    min_nodes: int
    idle_unload_seconds: int | None = None
    added_at: float = 0.0


def profile_path() -> Path:
    return _PROFILE_FILE


def load_launch_profile() -> dict[str, LaunchProfileEntry]:
    try:
        raw = _PROFILE_FILE.read_text()
        data = cast(dict[str, object], json.loads(raw))
        return {
            model_id: LaunchProfileEntry.model_validate(entry)
            for model_id, entry in data.items()
        }
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def save_launch_profile(profile: dict[str, LaunchProfileEntry]) -> None:
    """Atomically persist the profile (tmp + rename, like _lifetime_usage)."""
    _PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _PROFILE_FILE.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({k: v.model_dump() for k, v in profile.items()}, indent=2)
    )
    tmp.replace(_PROFILE_FILE)


def upsert_launch_profile(entry: LaunchProfileEntry) -> dict[str, LaunchProfileEntry]:
    profile = load_launch_profile()
    profile[entry.model_id] = entry
    save_launch_profile(profile)
    return profile


def remove_from_launch_profile(model_id: str) -> dict[str, LaunchProfileEntry]:
    profile = load_launch_profile()
    if profile.pop(model_id, None) is not None:
        save_launch_profile(profile)
    return profile
