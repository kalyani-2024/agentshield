"""Pick a repository backend from a connection string."""

from __future__ import annotations

from agentshield.storage.repository import Repository
from agentshield.storage.sqlite_repo import SQLiteRepository


def build_repository(database_url: str) -> Repository:
    """Return the repository implied by ``database_url``.

    Accepted forms::

        sqlite:///agentshield.db      -> file-backed SQLite (default)
        sqlite:///:memory:            -> in-memory SQLite
        postgresql://user:pw@host/db  -> PostgreSQL via psycopg
        postgres://...                -> same
    """
    url = database_url.strip()

    if url.startswith("sqlite:///"):
        path = url[len("sqlite:///"):]
        return SQLiteRepository(path or "agentshield.db")

    if url.startswith(("postgresql://", "postgres://")):
        from agentshield.storage.postgres_repo import PostgresRepository

        return PostgresRepository(url)

    raise ValueError(
        f"unsupported database_url {url!r}; "
        f"expected sqlite:///... or postgresql://..."
    )
