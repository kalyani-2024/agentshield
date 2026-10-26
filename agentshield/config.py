"""Runtime configuration, read from the environment with safe defaults.

Nothing here holds a secret literal.  Connection strings and API keys come from
the environment, so the same code runs against SQLite on a laptop and
PostgreSQL + Redis in a deployment without a change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _setting(name: str, default: str = ""):
    """A field whose default is read from the environment when Settings is built."""
    return field(default_factory=lambda: _env(name, default))


def _flag(name: str, default: str):
    return field(
        default_factory=lambda: _env(name, default).lower() in ("1", "true", "yes", "on")
    )


@dataclass(frozen=True)
class Settings:
    """Every deployment knob in one place."""

    #: repository connection string.
    #:   sqlite:///path/to/file.db   (default; zero setup)
    #:   sqlite:///:memory:          (ephemeral)
    #:   postgresql://user:pass@host:5432/agentshield
    database_url: str = _setting("AGENTSHIELD_DATABASE_URL", "sqlite:///agentshield.db")

    #: session store backend.  empty -> in-memory; otherwise a redis URL:
    #:   redis://localhost:6379/0
    redis_url: str = _setting("AGENTSHIELD_REDIS_URL")

    #: HTTP bind address for the API.
    api_host: str = _setting("AGENTSHIELD_API_HOST", "127.0.0.1")
    api_port: int = field(default_factory=lambda: int(_env("AGENTSHIELD_API_PORT", "8000")))

    #: optional shared-secret for the API's X-API-Key header. empty -> open.
    api_key: str = _setting("AGENTSHIELD_API_KEY")

    #: comma-separated destination allow/deny lists for the default trust policy.
    allowed_domains: str = _setting("AGENTSHIELD_ALLOWED_DOMAINS")
    denied_domains: str = _setting("AGENTSHIELD_DENIED_DOMAINS")

    #: injection detector: hybrid | ml | llm | rules  (see detection/factory.py)
    detector: str = _setting("AGENTSHIELD_DETECTOR", "hybrid")
    #: where the trained model files live; empty -> the packaged models
    model_dir: str = _setting("AGENTSHIELD_MODEL_DIR")
    #: the hybrid detector asks the LLM judge when the model's probability
    #: falls strictly between these two bounds
    hybrid_low: float = field(default_factory=lambda: float(_env("AGENTSHIELD_HYBRID_LOW", "0.15")))
    hybrid_high: float = field(
        default_factory=lambda: float(_env("AGENTSHIELD_HYBRID_HIGH", "0.85"))
    )

    #: LLM judge.  provider: groq | gemini | openai.  Empty -> whichever
    #: provider has a key in the environment (GROQ_API_KEY, GEMINI_API_KEY, ...).
    llm_provider: str = _setting("AGENTSHIELD_LLM_PROVIDER")
    llm_model: str = _setting("AGENTSHIELD_LLM_MODEL")
    llm_api_key: str = _setting("AGENTSHIELD_LLM_API_KEY")
    llm_base_url: str = _setting("AGENTSHIELD_LLM_BASE_URL")
    llm_timeout: float = field(
        default_factory=lambda: float(_env("AGENTSHIELD_LLM_TIMEOUT", "15"))
    )

    #: tool connectors the proxy may execute: any of files, email, database, http.
    #: A listed connector whose settings are missing stays off (see /v1/tools).
    tools: str = _setting("AGENTSHIELD_TOOLS", "files,email,database,http")
    #: the only directory the file tools can read or write
    workspace_dir: str = _setting("AGENTSHIELD_WORKSPACE_DIR", "workspace")
    #: database the query_database tool reads (read-only); empty -> tool off
    tool_database_url: str = _setting("AGENTSHIELD_TOOL_DATABASE_URL")
    #: SMTP server for send_email; empty host -> tool off
    smtp_host: str = _setting("AGENTSHIELD_SMTP_HOST")
    smtp_port: int = field(default_factory=lambda: int(_env("AGENTSHIELD_SMTP_PORT", "587")))
    smtp_username: str = _setting("AGENTSHIELD_SMTP_USERNAME")
    smtp_password: str = _setting("AGENTSHIELD_SMTP_PASSWORD")
    #: From address; defaults to the username
    smtp_from: str = _setting("AGENTSHIELD_SMTP_FROM")
    smtp_starttls: bool = _flag("AGENTSHIELD_SMTP_STARTTLS", "true")
    smtp_ssl: bool = _flag("AGENTSHIELD_SMTP_SSL", "false")
    #: HTTP tool: seconds per request, and whether internal addresses may be
    #: reached (off: the firewall must not become a way into the network)
    http_timeout: float = field(
        default_factory=lambda: float(_env("AGENTSHIELD_HTTP_TIMEOUT", "15"))
    )
    http_allow_private: bool = _flag("AGENTSHIELD_HTTP_ALLOW_PRIVATE", "false")

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


def load_dotenv(path: str | Path = ".env") -> list[str]:
    """Read ``KEY=value`` lines from a .env file into the environment.

    Variables already set in the environment win, so a shell export or a
    container setting is never overridden by the file.  Returns the names set.
    """
    file = Path(path)
    if not file.is_file():
        return []
    loaded = []
    for line in file.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip().removeprefix("export ").strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if name and value and not os.environ.get(name):
            os.environ[name] = value
            loaded.append(name)
    return loaded


def load_settings() -> Settings:
    """Build a fresh :class:`Settings` from the current environment."""
    return Settings()
