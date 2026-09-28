"""SQLite-backed repository - the zero-setup default."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from agentshield.storage.schema import SQLITE
from agentshield.storage.sql_base import SqlRepository


class SQLiteRepository(SqlRepository):
    """Runs out of the box: a single file, or ``:memory:`` for tests."""

    placeholder = "?"
    dialect = SQLITE

    def __init__(self, path: str = "agentshield.db") -> None:
        super().__init__()
        self.path = path

    def _connect(self):
        if self.path not in (":memory:", "file::memory:?cache=shared"):
            parent = Path(self.path).expanduser().resolve().parent
            parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn
