"""Runtime configuration, read from the environment with safe defaults.

Nothing here holds a secret literal.  Connection strings come from the
environment, so the same code runs against SQLite on a laptop and PostgreSQL in
a deployment without a change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


@dataclass(frozen=True)
class Settings:
    """All Stage 2 knobs in one place."""

    #: repository connection string.
    #:   sqlite:///path/to/file.db   (default; zero setup)
    #:   sqlite:///:memory:          (ephemeral)
    #:   postgresql://user:pass@host:5432/agentshield
    database_url: str = _env("AGENTSHIELD_DATABASE_URL", "sqlite:///agentshield.db")

    #: session store backend.  empty -> in-memory; otherwise a redis URL:
    #:   redis://localhost:6379/0
    redis_url: str = _env("AGENTSHIELD_REDIS_URL", "")

    #: HTTP bind address for the API.
    api_host: str = _env("AGENTSHIELD_API_HOST", "127.0.0.1")
    api_port: int = int(_env("AGENTSHIELD_API_PORT", "8000"))

    #: optional shared-secret for the API's X-API-Key header. empty -> open.
    api_key: str = _env("AGENTSHIELD_API_KEY", "")

    #: comma-separated destination allow/deny lists for the default trust policy.
    allowed_domains: str = _env("AGENTSHIELD_ALLOWED_DOMAINS", "")
    denied_domains: str = _env("AGENTSHIELD_DENIED_DOMAINS", "")

    def allowed_domain_set(self) -> set[str]:
        return {d.strip() for d in self.allowed_domains.split(",") if d.strip()}

    def denied_domain_set(self) -> set[str]:
        return {d.strip() for d in self.denied_domains.split(",") if d.strip()}

    @property
    def uses_postgres(self) -> bool:
        return self.database_url.startswith(("postgresql://", "postgres://"))

    @property
    def uses_redis(self) -> bool:
        return bool(self.redis_url)


def load_settings() -> Settings:
    """Build a fresh :class:`Settings` from the current environment."""
    return Settings()
