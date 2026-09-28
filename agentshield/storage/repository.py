"""The persistence interface the rest of the system talks to.

Everything above this file works in terms of :class:`Repository`; whether the
rows land in SQLite or PostgreSQL is decided once, in :mod:`storage.factory`.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from agentshield.core.models import Principal, SecurityAssessment
from agentshield.engine.session import Session


class Repository(ABC):
    """Durable store for calls, decisions, incidents and events."""

    # -- lifecycle -------------------------------------------------------
    @abstractmethod
    def initialize(self) -> None:
        """Create tables if they do not yet exist. Safe to call repeatedly."""

    @abstractmethod
    def close(self) -> None:
        ...

    # -- writes ----------------------------------------------------------
    @abstractmethod
    def save_assessment(self, assessment: SecurityAssessment) -> None:
        """Persist a tool call and its policy decision together."""

    @abstractmethod
    def save_event(
        self, event_type: str, call_id: str | None, session_id: str | None, payload: dict
    ) -> None:
        ...

    @abstractmethod
    def save_incident(self, incident: dict) -> None:
        ...

    @abstractmethod
    def upsert_session(self, session: Session) -> None:
        ...

    @abstractmethod
    def upsert_principal(self, principal: Principal) -> None:
        ...

    @abstractmethod
    def record_approval(self, call_id: str, session_id: str | None, status: str) -> None:
        ...

    # -- reads (for the API) --------------------------------------------
    @abstractmethod
    def recent_decisions(self, limit: int = 50, session_id: str | None = None) -> list[dict]:
        ...

    @abstractmethod
    def list_incidents(self, limit: int = 50) -> list[dict]:
        ...

    @abstractmethod
    def counts(self) -> dict[str, int]:
        """Row counts per table, for the /stats endpoint."""

    # -- context-manager sugar ------------------------------------------
    def __enter__(self) -> "Repository":
        self.initialize()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def dumps(value: Any) -> str:
    """JSON-encode a value for a text/JSON column, tolerating odd types."""
    return json.dumps(value, default=str, ensure_ascii=False)


def loads(value: Any) -> Any:
    """Decode a JSON column that may arrive as text (SQLite) or native (psycopg)."""
    if value is None or isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value
