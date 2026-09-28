"""Database schema.

The design document lists these persistent tables: users, agents, sessions,
tool_calls, policy_decisions, incidents, security_events, approvals.  Stage 2
implements the ones the engine actually writes; the rest are created so the
schema is complete and the later stages have somewhere to write.

Two dialects are supported.  They differ only in a few type names, so the DDL is
templated over a small dialect table rather than duplicated.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dialect:
    name: str
    pk: str            # auto-incrementing primary key
    json: str          # JSON column type
    timestamp: str     # timestamp column type
    now: str           # server-side "now" default expression


SQLITE = Dialect(
    name="sqlite",
    pk="INTEGER PRIMARY KEY AUTOINCREMENT",
    json="TEXT",
    timestamp="TEXT",
    now="CURRENT_TIMESTAMP",
)

POSTGRES = Dialect(
    name="postgres",
    pk="BIGSERIAL PRIMARY KEY",
    json="JSONB",
    timestamp="TIMESTAMPTZ",
    now="now()",
)


def ddl_statements(d: Dialect) -> list[str]:
    """Return the ``CREATE TABLE IF NOT EXISTS`` statements for a dialect."""
    return [
        f"""
        CREATE TABLE IF NOT EXISTS principals (
            id            TEXT PRIMARY KEY,
            display_name  TEXT,
            roles         {d.json},
            allowed_tools {d.json},
            created_at    {d.timestamp} DEFAULT {d.now}
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id       TEXT PRIMARY KEY,
            principal_id     TEXT,
            state            TEXT NOT NULL,
            cumulative_risk  INTEGER NOT NULL DEFAULT 0,
            created_at       {d.timestamp} DEFAULT {d.now},
            updated_at       {d.timestamp} DEFAULT {d.now}
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS tool_calls (
            id             {d.pk},
            call_id        TEXT UNIQUE NOT NULL,
            session_id     TEXT NOT NULL,
            principal_id   TEXT,
            tool           TEXT NOT NULL,
            category       TEXT,
            destination    TEXT,
            arguments      {d.json},
            source_framework TEXT,
            user_authorized  INTEGER NOT NULL DEFAULT 0,
            created_at     {d.timestamp} DEFAULT {d.now}
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS policy_decisions (
            id             {d.pk},
            call_id        TEXT NOT NULL,
            session_id     TEXT NOT NULL,
            decision       TEXT NOT NULL,
            risk_score     INTEGER NOT NULL,
            session_state  TEXT,
            resulting_state TEXT,
            reason         TEXT,
            signals        {d.json},
            latency_ms     REAL,
            evaluated_at   {d.timestamp} DEFAULT {d.now}
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS incidents (
            id           {d.pk},
            kind         TEXT NOT NULL,
            session_id   TEXT,
            tool         TEXT,
            risk_score   INTEGER,
            reason       TEXT,
            detail       {d.json},
            created_at   {d.timestamp} DEFAULT {d.now}
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS security_events (
            id          {d.pk},
            type        TEXT NOT NULL,
            call_id     TEXT,
            session_id  TEXT,
            payload     {d.json},
            at          {d.timestamp} DEFAULT {d.now}
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS approvals (
            id           {d.pk},
            call_id      TEXT UNIQUE NOT NULL,
            session_id   TEXT,
            status       TEXT NOT NULL,
            requested_at {d.timestamp} DEFAULT {d.now},
            decided_at   {d.timestamp}
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_tool_calls_session ON tool_calls (session_id)",
        "CREATE INDEX IF NOT EXISTS ix_decisions_session ON policy_decisions (session_id)",
        "CREATE INDEX IF NOT EXISTS ix_events_session ON security_events (session_id)",
        "CREATE INDEX IF NOT EXISTS ix_incidents_session ON incidents (session_id)",
    ]
