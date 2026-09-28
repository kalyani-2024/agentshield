"""Wires the Stage 1 engine to Stage 2 persistence and session storage.

A :class:`ShieldService` is an :class:`AgentShield` plus a durable repository
and (optionally) a Redis-backed session store, assembled from :class:`Settings`.
"""

from __future__ import annotations

from typing import Any

from agentshield.api import AgentShield
from agentshield.config import Settings, load_settings
from agentshield.engine.pipeline import SecurityEngine
from agentshield.engine.session import SessionStore
from agentshield.filters.destination_trust import TrustPolicy
from agentshield.storage.factory import build_repository
from agentshield.storage.observer import PersistenceObserver
from agentshield.storage.repository import Repository


class ShieldService:
    """The Stage 2 application object: shield + storage, ready for the API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()

        # 1. durable store
        self.repository: Repository = build_repository(self.settings.database_url)
        self.repository.initialize()

        # 2. session state backend (Redis when configured, else in-memory)
        sessions = self._build_session_store()

        # 3. the security engine, with a configured trust policy
        trust_policy = TrustPolicy(
            allowed_domains=self.settings.allowed_domain_set(),
            denied_domains=self.settings.denied_domain_set(),
        )
        engine = SecurityEngine(sessions=sessions, trust_policy=trust_policy)

        # 4. the Stage 1 facade around that engine
        self.shield = AgentShield(engine=engine)

        # 5. persistence attaches as just another observer
        self.persistence = PersistenceObserver(
            self.repository, session_store=engine.sessions
        )
        self.shield.subscribe(self.persistence)

    def _build_session_store(self):
        if self.settings.uses_redis:
            from agentshield.engine.redis_session import RedisSessionStore

            return RedisSessionStore.from_url(self.settings.redis_url)
        return SessionStore()

    # -- convenience passthroughs ---------------------------------------
    def close(self) -> None:
        self.repository.close()

    @property
    def backend(self) -> dict[str, Any]:
        return {
            "database": "postgres" if self.settings.uses_postgres else "sqlite",
            "database_url": _redact_url(self.settings.database_url),
            "session_store": "redis" if self.settings.uses_redis else "memory",
        }


def _redact_url(url: str) -> str:
    """Hide any password embedded in a connection string before logging it."""
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, tail = rest.split("@", 1)
    if ":" in creds:
        user = creds.split(":", 1)[0]
        creds = f"{user}:***"
    return f"{scheme}://{creds}@{tail}"


def build_service(settings: Settings | None = None) -> ShieldService:
    return ShieldService(settings)
