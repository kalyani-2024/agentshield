"""Wires the security engine to persistence, session storage and the detectors.

A :class:`ShieldService` is an :class:`AgentShield` plus a durable repository,
an optional Redis-backed session store, the configured detection strategy and
the real tool connectors - assembled from :class:`Settings`.
"""

from __future__ import annotations

import logging
from typing import Any

from agentshield.api import AgentShield
from agentshield.config import Settings, load_settings
from agentshield.detection.factory import build_detector, build_judge, url_model
from agentshield.detection.strategy import DetectionStrategy
from agentshield.engine.pipeline import SecurityEngine
from agentshield.engine.session import SessionStore
from agentshield.filters.destination_trust import TrustPolicy
from agentshield.proxy.connectors import build_connectors
from agentshield.storage.factory import build_repository
from agentshield.storage.observer import PersistenceObserver
from agentshield.storage.repository import Repository

logger = logging.getLogger("agentshield.service")


class ShieldService:
    """The application object: shield + storage + detectors, ready for the API."""

    def __init__(
        self,
        settings: Settings | None = None,
        judge: DetectionStrategy | None = None,
    ) -> None:
        self.settings = settings or load_settings()

        # 1. durable store
        self.repository: Repository = build_repository(self.settings.database_url)
        self.repository.initialize()

        # 2. session state backend (Redis when configured, else in-memory)
        sessions = self._build_session_store()

        # 3. the detector: model-first, rules only as the fallback
        if judge is None and self.settings.detector in ("hybrid", "llm"):
            judge = build_judge(
                self.settings.llm_provider,
                self.settings.llm_api_key,
                self.settings.llm_model,
                self.settings.llm_base_url,
                self.settings.llm_timeout,
            )
        model_dir = self.settings.model_dir or None
        detector, self.detector_info = build_detector(
            self.settings.detector,
            model_dir,
            judge,
            self.settings.hybrid_low,
            self.settings.hybrid_high,
        )
        self.detector = detector
        for note in self.detector_info.notes:
            logger.warning("detector: %s", note)

        # 4. the security engine, with a configured trust policy
        trust_policy = TrustPolicy(
            allowed_domains=self.settings.allowed_domain_set(),
            denied_domains=self.settings.denied_domain_set(),
        )
        engine = SecurityEngine(
            sessions=sessions,
            trust_policy=trust_policy,
            detector=detector,
            url_model=None if self.settings.detector == "rules" else url_model(model_dir),
        )

        # 5. the facade around that engine, with the real tools behind the proxy
        self.shield = AgentShield(engine=engine)
        tools, self.disabled_tools = build_connectors(self.settings)
        for name, tool in tools:
            self.shield.register(tool, name)
        for name, reason in self.disabled_tools.items():
            logger.info("tool %s is off: %s", name, reason)

        # 6. persistence attaches as just another observer
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
            "detector": self.detector_info.to_dict(),
            "detector_fallbacks": getattr(self.detector, "fallbacks", 0),
            "tools": self.shield.proxy.registry.names(),
            "tools_disabled": self.disabled_tools,
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


def build_service(
    settings: Settings | None = None, judge: DetectionStrategy | None = None
) -> ShieldService:
    return ShieldService(settings, judge)
