"""In-memory test doubles for the tool connectors.

Unit tests use these so they run offline and fast; the application itself only
registers the real connectors in agentshield/proxy/connectors.py.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from agentshield.proxy.tools import Tool


class InMemoryFileTool(Tool):
    """A fake filesystem, so the demo can be dangerous without being dangerous."""

    name = "read_file"
    description = "Read a file from the workspace"

    def __init__(self, files: dict[str, str] | None = None) -> None:
        self.files = files or {}

    def execute(self, **arguments: Any) -> Any:
        path = str(arguments.get("path", ""))
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]


class InMemoryWriteTool(Tool):
    name = "write_file"
    description = "Write a file in the workspace"

    def __init__(self, store: dict[str, str]) -> None:
        self.store = store

    def execute(self, **arguments: Any) -> Any:
        path = str(arguments.get("path", ""))
        self.store[path] = str(arguments.get("content", ""))
        return {"written": path, "bytes": len(self.store[path])}


class MockEmailTool(Tool):
    name = "send_email"
    description = "Send an email on the user's behalf"

    def __init__(self) -> None:
        self.outbox: list[dict[str, Any]] = []

    def execute(self, **arguments: Any) -> Any:
        message = {
            "recipient": arguments.get("recipient"),
            "subject": arguments.get("subject", ""),
            "body": arguments.get("body", ""),
        }
        self.outbox.append(message)
        return {"sent": True, "recipient": message["recipient"]}


class MockSearchTool(Tool):
    name = "web_search"
    description = "Search the web"

    def __init__(self, results: dict[str, str] | None = None) -> None:
        self.results = results or {}

    def execute(self, **arguments: Any) -> Any:
        query = str(arguments.get("query", ""))
        return self.results.get(query, f"no results for {query!r}")


class MockDatabaseTool(Tool):
    """A read-only SQL database, seeded with a little fake customer data."""

    name = "query_database"
    description = "Run a read-only SQL query against the sandbox database"

    _SEED = (
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, email TEXT, plan TEXT)",
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, total REAL, "
        "placed_on TEXT)",
        "INSERT INTO customers VALUES (1, 'Maya Rao', 'maya@example.com', 'pro'), "
        "(2, 'Sam Lee', 'sam@example.com', 'free'), (3, 'Priya N', 'priya@example.com', 'pro')",
        "INSERT INTO orders VALUES (1, 1, 120.50, '2026-01-14'), (2, 1, 35.00, '2026-02-02'), "
        "(3, 3, 410.00, '2026-02-20')",
    )

    def __init__(self, seed: tuple[str, ...] | None = None, max_rows: int = 100) -> None:
        self.max_rows = max_rows
        self.queries: list[str] = []
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        for statement in seed if seed is not None else self._SEED:
            self._conn.execute(statement)
        self._conn.commit()
        # the sandbox database can be read but never changed through the tool
        self._conn.execute("PRAGMA query_only = ON")

    def execute(self, **arguments: Any) -> Any:
        query = str(arguments.get("query") or arguments.get("sql") or "")
        if not query.strip():
            raise ValueError("query_database needs a 'query' argument")
        self.queries.append(query)
        cursor = self._conn.execute(query)
        columns = [column[0] for column in cursor.description or []]
        rows = [dict(zip(columns, row)) for row in cursor.fetchmany(self.max_rows)]
        return {"columns": columns, "rows": rows, "row_count": len(rows)}


class MockHttpTool(Tool):
    """A stand-in for an external HTTP API.  Nothing leaves the process."""

    name = "http_request"
    description = "Call an external HTTP API"

    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        #: url -> canned response body
        self.responses = responses or {}
        #: every request the agent made, for inspection in tests and demos
        self.requests: list[dict[str, Any]] = []

    def execute(self, **arguments: Any) -> Any:
        url = str(arguments.get("url") or arguments.get("endpoint") or "")
        if not url:
            raise ValueError("http_request needs a 'url' argument")
        method = str(arguments.get("method", "GET")).upper()
        self.requests.append(
            {"method": method, "url": url, "body": arguments.get("body", arguments.get("data"))}
        )
        body = self.responses.get(url, f"sandbox response for {method} {url}")
        return {"status": 200, "url": url, "body": body}
