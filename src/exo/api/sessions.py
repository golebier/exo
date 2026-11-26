"""Named session snapshots — "saved sessions" of instance launch recipes.

Builds on the durable launch profile (launch_profile.py), which already IS the
auto-maintained "last session": every successful placement upserts a recipe
entry keyed by model id. A saved session is a frozen copy of those entries
under a user-chosen name, so the user can review each instance (model,
sharding, instance meta, node count, idle-unload policy) and launch them one
by one — not only all at once.

Plain JSON sidecar in the event-log dir, same pattern as launch_profile.json.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import cast

from exo.api.launch_profile import LaunchProfileEntry
from exo.shared.constants import EXO_EVENT_LOG_DIR
from exo.utils.pydantic_ext import FrozenModel

_SESSIONS_FILE = EXO_EVENT_LOG_DIR / "sessions.json"


class SessionSnapshot(FrozenModel):
    id: str
    name: str
    created_at: float
    entries: list[LaunchProfileEntry]


def sessions_path() -> Path:
    return _SESSIONS_FILE


def load_sessions() -> list[SessionSnapshot]:
    try:
        raw = _SESSIONS_FILE.read_text()
        data = cast(list[object], json.loads(raw))
        return [SessionSnapshot.model_validate(item) for item in data]
    except FileNotFoundError:
        return []
    except Exception:
        return []


def save_sessions(sessions: list[SessionSnapshot]) -> None:
    """Atomically persist the sessions (tmp + rename, like launch_profile)."""
    _SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _SESSIONS_FILE.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            [s.model_dump() for s in sessions],
            indent=2,
        )
    )
    tmp.replace(_SESSIONS_FILE)


def save_session(name: str, entries: list[LaunchProfileEntry]) -> list[SessionSnapshot]:
    sessions = load_sessions()
    sessions.append(
        SessionSnapshot(
            id=uuid.uuid4().hex[:12],
            name=name,
            created_at=time.time(),
            entries=list(entries),
        )
    )
    save_sessions(sessions)
    return sessions


def delete_session(session_id: str) -> list[SessionSnapshot]:
    sessions = [s for s in load_sessions() if s.id != session_id]
    save_sessions(sessions)
    return sessions
