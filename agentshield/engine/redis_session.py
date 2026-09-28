"""Redis-backed session store.

The design doc puts ephemeral session state - risk score, short-term tool
history - in Redis so it is shared across API workers and expires on its own.
This store honours the same interface as the in-memory :class:`SessionStore`,
so the engine cannot tell the difference.

It is written against a tiny client interface (``get``, ``set``, ``delete``,
``scan_iter``) which the real ``redis`` package satisfies, and which a fake
satisfies in the tests - so the logic is testable without a live server.
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from typing import Any, Iterable

from agentshield.core.models import ToolCategory
from agentshield.engine.session import (
    HISTORY_LIMIT,
    HistoryEntry,
    Session,
    state_by_name,
)

_KEY_PREFIX = "agentshield:session:"
#: sessions expire after this many seconds of inactivity (24h).
DEFAULT_TTL = 86_400


def session_to_json(session: Session) -> str:
    return json.dumps(
        {
            "session_id": session.session_id,
            "principal_id": session.principal_id,
            "cumulative_risk": session.cumulative_risk,
            "state": session.state.name,
            "created_at": session.created_at.isoformat(),
            "history": [
                {
                    "tool": e.tool,
                    "category": e.category.value,
                    "destination": e.destination,
                    "risk_score": e.risk_score,
                    "decision": e.decision,
                    "at": e.at.isoformat(),
                }
                for e in session.history
            ],
        }
    )


def session_from_json(blob: str) -> Session:
    data = json.loads(blob)
    history: deque[HistoryEntry] = deque(maxlen=HISTORY_LIMIT)
    for e in data.get("history", []):
        history.append(
            HistoryEntry(
                tool=e["tool"],
                category=ToolCategory(e["category"]),
                destination=e.get("destination"),
                risk_score=e["risk_score"],
                decision=e["decision"],
                at=datetime.fromisoformat(e["at"]),
            )
        )
    return Session(
        session_id=data["session_id"],
        principal_id=data.get("principal_id", "anonymous"),
        cumulative_risk=data.get("cumulative_risk", 0),
        state=state_by_name(data.get("state", "NORMAL")),
        history=history,
        created_at=datetime.fromisoformat(data["created_at"])
        if data.get("created_at")
        else datetime.now(timezone.utc),
    )


class _WriteThroughSession(Session):
    """A Session that flushes itself to Redis after every mutation.

    The engine mutates the session it gets from ``get`` (via ``record``) and
    never calls back into the store, so the write-through has to live on the
    session object itself.  This keeps the engine oblivious to the backend.
    """

    _store: "RedisSessionStore"

    def record(self, call, risk_score, decision):  # type: ignore[override]
        transition = super().record(call, risk_score, decision)
        self._store._persist(self)
        return transition

    def reset(self) -> None:  # type: ignore[override]
        super().reset()
        self._store._persist(self)


class RedisSessionStore:
    """Session registry backed by Redis, matching ``SessionStore``'s interface."""

    def __init__(self, client: Any, ttl: int = DEFAULT_TTL) -> None:
        self.client = client
        self.ttl = ttl

    @classmethod
    def from_url(cls, url: str, ttl: int = DEFAULT_TTL) -> "RedisSessionStore":
        import redis  # lazy: only needed when Redis is actually configured

        return cls(redis.Redis.from_url(url, decode_responses=True), ttl=ttl)

    # -- interface expected by the engine -------------------------------
    def get(self, session_id: str, principal_id: str = "anonymous") -> Session:
        blob = self.client.get(_KEY_PREFIX + session_id)
        if blob is not None:
            session = session_from_json(blob)
        else:
            session = Session(session_id=session_id, principal_id=principal_id)
        return self._attach(session)

    def all(self) -> Iterable[Session]:
        sessions = []
        for key in self.client.scan_iter(match=_KEY_PREFIX + "*"):
            blob = self.client.get(key)
            if blob is not None:
                sessions.append(self._attach(session_from_json(blob)))
        return sessions

    def reset(self, session_id: str) -> None:
        session = self.get(session_id)
        session.reset()  # write-through persists it

    def clear(self) -> None:
        for key in list(self.client.scan_iter(match=_KEY_PREFIX + "*")):
            self.client.delete(key)

    # -- internals -------------------------------------------------------
    def _attach(self, session: Session) -> Session:
        session.__class__ = _WriteThroughSession
        session._store = self  # type: ignore[attr-defined]
        return session

    def _persist(self, session: Session) -> None:
        key = _KEY_PREFIX + session.session_id
        blob = session_to_json(session)
        try:
            self.client.set(key, blob, ex=self.ttl)
        except TypeError:  # a minimal fake client without the ex kwarg
            self.client.set(key, blob)
