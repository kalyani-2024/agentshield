"""PostgreSQL-backed repository - the deployment target from the design doc.

Uses psycopg (v3).  The connection string is supplied through configuration and
never hard-coded here.
"""

from __future__ import annotations

from typing import Any

from agentshield.storage.schema import POSTGRES
from agentshield.storage.sql_base import SqlRepository


class PostgresRepository(SqlRepository):
    placeholder = "%s"
    dialect = POSTGRES

    def __init__(self, dsn: str) -> None:
        super().__init__()
        self.dsn = dsn

    def _connect(self):
        import psycopg  # imported lazily so SQLite users need no driver

        return psycopg.connect(self.dsn)

    def _json(self, value: Any) -> Any:
        from psycopg.types.json import Jsonb

        return Jsonb(value)
