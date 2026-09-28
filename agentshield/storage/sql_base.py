"""Shared SQL logic for the SQLite and PostgreSQL repositories.

The two backends differ only in how they connect and in the parameter marker
(``?`` vs ``%s``), so all the statement-building lives here once.
"""

from __future__ import annotations

from typing import Any, Sequence

from agentshield.core.models import Principal, SecurityAssessment
from agentshield.engine.session import Session
from agentshield.storage.repository import Repository, dumps, loads
from agentshield.storage.schema import Dialect, ddl_statements


class SqlRepository(Repository):
    """Backend-agnostic implementation over a DB-API 2.0 connection."""

    #: parameter marker for this backend's driver
    placeholder: str = "?"
    dialect: Dialect

    def __init__(self) -> None:
        self._conn: Any = None

    # -- subclasses provide these ---------------------------------------
    def _connect(self) -> Any:  # pragma: no cover - trivial
        raise NotImplementedError

    def _upsert_suffix(self, conflict_col: str, update_cols: Sequence[str]) -> str:
        """Dialect-specific ``ON CONFLICT`` clause for upserts."""
        assignments = ", ".join(f"{c}=excluded.{c}" for c in update_cols)
        return f" ON CONFLICT({conflict_col}) DO UPDATE SET {assignments}"

    # -- lifecycle -------------------------------------------------------
    def initialize(self) -> None:
        if self._conn is None:
            self._conn = self._connect()
        cur = self._conn.cursor()
        for statement in ddl_statements(self.dialect):
            cur.execute(statement)
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- helpers ---------------------------------------------------------
    def _q(self, sql: str) -> str:
        """Rewrite the ``?`` markers in a statement to this backend's marker."""
        if self.placeholder == "?":
            return sql
        return sql.replace("?", self.placeholder)

    def _json(self, value: Any) -> Any:
        """Encode a value for a JSON column. SQLite stores text; Postgres
        overrides this to use a JSONB adapter."""
        return dumps(value)

    def _execute(self, sql: str, params: Sequence[Any] = ()) -> Any:
        cur = self._conn.cursor()
        cur.execute(self._q(sql), tuple(params))
        return cur

    def _commit(self) -> None:
        self._conn.commit()

    def _rows(self, cur: Any) -> list[dict]:
        columns = [c[0] for c in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]

    # -- writes ----------------------------------------------------------
    def save_assessment(self, assessment: SecurityAssessment) -> None:
        call = assessment.call
        self._execute(
            "INSERT INTO tool_calls "
            "(call_id, session_id, principal_id, tool, category, destination, "
            " arguments, source_framework, user_authorized) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            + self._upsert_suffix("call_id", ("destination", "category")),
            [
                call.call_id,
                call.session_id,
                call.principal.id,
                call.tool,
                call.category.value,
                call.destination,
                self._json(call.arguments),
                call.source_framework,
                1 if call.user_authorized else 0,
            ],
        )
        self._execute(
            "INSERT INTO policy_decisions "
            "(call_id, session_id, decision, risk_score, session_state, "
            " resulting_state, reason, signals, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                call.call_id,
                call.session_id,
                assessment.decision.value,
                assessment.risk_score,
                assessment.session_state,
                assessment.resulting_state,
                assessment.reason,
                self._json([s.to_dict() for s in assessment.signals]),
                assessment.latency_ms,
            ],
        )
        self._commit()

    def save_event(
        self, event_type: str, call_id: str | None, session_id: str | None, payload: dict
    ) -> None:
        self._execute(
            "INSERT INTO security_events (type, call_id, session_id, payload) "
            "VALUES (?, ?, ?, ?)",
            [event_type, call_id, session_id, self._json(payload)],
        )
        self._commit()

    def save_incident(self, incident: dict) -> None:
        self._execute(
            "INSERT INTO incidents (kind, session_id, tool, risk_score, reason, detail) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                incident.get("kind", "UNKNOWN"),
                incident.get("session_id"),
                incident.get("tool"),
                incident.get("risk_score"),
                incident.get("reason"),
                self._json(incident),
            ],
        )
        self._commit()

    def upsert_session(self, session: Session) -> None:
        self._execute(
            "INSERT INTO sessions (session_id, principal_id, state, cumulative_risk) "
            "VALUES (?, ?, ?, ?)"
            + self._upsert_suffix(
                "session_id", ("state", "cumulative_risk", "principal_id")
            ),
            [
                session.session_id,
                session.principal_id,
                session.state.name,
                session.cumulative_risk,
            ],
        )
        self._commit()

    def upsert_principal(self, principal: Principal) -> None:
        self._execute(
            "INSERT INTO principals (id, display_name, roles, allowed_tools) "
            "VALUES (?, ?, ?, ?)"
            + self._upsert_suffix(
                "id", ("display_name", "roles", "allowed_tools")
            ),
            [
                principal.id,
                principal.display_name,
                self._json(list(principal.roles)),
                self._json(sorted(principal.allowed_tools)),
            ],
        )
        self._commit()

    def record_approval(self, call_id: str, session_id: str | None, status: str) -> None:
        decided = status in ("APPROVED", "DENIED", "EXECUTED")
        now_expr = self.dialect.now if decided else "NULL"
        self._execute(
            f"INSERT INTO approvals (call_id, session_id, status, decided_at) "
            f"VALUES (?, ?, ?, {now_expr})"
            + self._upsert_suffix("call_id", ("status", "decided_at")),
            [call_id, session_id, status],
        )
        self._commit()

    # -- reads -----------------------------------------------------------
    def recent_decisions(self, limit: int = 50, session_id: str | None = None) -> list[dict]:
        if session_id:
            cur = self._execute(
                "SELECT * FROM policy_decisions WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                [session_id, limit],
            )
        else:
            cur = self._execute(
                "SELECT * FROM policy_decisions ORDER BY id DESC LIMIT ?", [limit]
            )
        rows = self._rows(cur)
        for row in rows:
            row["signals"] = loads(row.get("signals"))
        return rows

    def list_incidents(self, limit: int = 50) -> list[dict]:
        cur = self._execute("SELECT * FROM incidents ORDER BY id DESC LIMIT ?", [limit])
        rows = self._rows(cur)
        for row in rows:
            row["detail"] = loads(row.get("detail"))
        return rows

    def counts(self) -> dict[str, int]:
        tables = (
            "principals", "sessions", "tool_calls", "policy_decisions",
            "incidents", "security_events", "approvals",
        )
        result: dict[str, int] = {}
        for table in tables:
            cur = self._execute(f"SELECT COUNT(*) FROM {table}")
            result[table] = int(cur.fetchone()[0])
        return result
