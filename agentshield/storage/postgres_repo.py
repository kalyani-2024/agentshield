"""PostgreSQL-backed repository - the deployment target from the design doc.

Uses psycopg (v3).  The connection string is supplied through configuration and
never hard-coded here.
"""

from __future__ import annotations

import time
from typing import Any

from agentshield.storage.schema import POSTGRES
from agentshield.storage.sql_base import SqlRepository


class PostgresRepository(SqlRepository):
    placeholder = "%s"
    dialect = POSTGRES

    def __init__(self, dsn: str, connect_attempts: int = 10) -> None:
        super().__init__()
        self.dsn = dsn
        self.connect_attempts = connect_attempts

    def _connect(self):
        import psycopg  # imported lazily so SQLite users need no driver

        # in a container stack the database can come up a moment after the API
        for attempt in range(1, self.connect_attempts + 1):
            try:
                return psycopg.connect(self.dsn)
            except psycopg.OperationalError:
                if attempt == self.connect_attempts:
                    raise
                time.sleep(min(5.0, 0.5 * attempt))

    def _json(self, value: Any) -> Any:
        from psycopg.types.json import Jsonb

        return Jsonb(value)
