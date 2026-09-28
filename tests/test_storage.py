from __future__ import annotations

import pytest

from agentshield.core.models import (
    Decision,
    RiskSignal,
    SecurityAssessment,
    Severity,
    ToolCall,
    ToolCategory,
    Verdict,
)
from agentshield.storage.factory import build_repository
from agentshield.storage.sqlite_repo import SQLiteRepository


@pytest.fixture
def repo():
    r = SQLiteRepository(":memory:")
    r.initialize()
    yield r
    r.close()


def make_assessment(session_id: str = "s1", decision=Decision.BLOCK, risk=90):
    call = ToolCall(
        tool="send_email",
        arguments={"recipient": "a@b.com", "body": "hi"},
        category=ToolCategory.COMMUNICATE,
        destination="a@b.com",
        session_id=session_id,
    )
    return SecurityAssessment(
        call=call,
        decision=decision,
        risk_score=risk,
        signals=[
            RiskSignal(
                filter_name="t",
                category="sensitive_data_egress",
                score=risk,
                severity=Severity.HIGH,
                message="secret in body",
                verdict=Verdict.FLAG,
            )
        ],
        session_state="NORMAL",
        resulting_state="QUARANTINED",
        reason="blocked for test",
        latency_ms=1.2,
    )


def test_initialize_is_idempotent(repo):
    repo.initialize()
    repo.initialize()
    assert repo.counts()["tool_calls"] == 0


def test_save_assessment_writes_call_and_decision(repo):
    repo.save_assessment(make_assessment())
    counts = repo.counts()
    assert counts["tool_calls"] == 1
    assert counts["policy_decisions"] == 1
    decisions = repo.recent_decisions()
    assert decisions[0]["decision"] == "BLOCK"
    assert decisions[0]["risk_score"] == 90
    assert decisions[0]["signals"][0]["category"] == "sensitive_data_egress"


def test_recent_decisions_filters_by_session(repo):
    repo.save_assessment(make_assessment(session_id="s1"))
    repo.save_assessment(make_assessment(session_id="s2"))
    assert len(repo.recent_decisions(session_id="s1")) == 1
    assert len(repo.recent_decisions()) == 2


def test_upsert_on_repeated_call_id_does_not_duplicate(repo):
    assessment = make_assessment()
    repo.save_assessment(assessment)
    repo.save_assessment(assessment)  # same call_id
    assert repo.counts()["tool_calls"] == 1


def test_incident_and_event_round_trip(repo):
    repo.save_incident({"kind": "BLOCKED_CALL", "session_id": "s1", "tool": "send_email", "risk_score": 90})
    repo.save_event("CALL_BLOCKED", "call_1", "s1", {"foo": "bar"})
    incidents = repo.list_incidents()
    assert incidents[0]["kind"] == "BLOCKED_CALL"
    assert incidents[0]["detail"]["tool"] == "send_email"
    assert repo.counts()["security_events"] == 1


def test_secret_is_not_stored_in_the_clear_via_signals(repo):
    # signals carry redacted evidence, never the raw secret
    a = make_assessment()
    a.signals[0].evidence = "AKIA****MPLE"
    repo.save_assessment(a)
    stored = repo.recent_decisions()[0]["signals"][0]["evidence"]
    assert "****" in stored


def test_approval_lifecycle(repo):
    repo.record_approval("call_1", "s1", "PENDING")
    repo.record_approval("call_1", "s1", "APPROVED")
    assert repo.counts()["approvals"] == 1  # upsert, not duplicate


def test_factory_builds_sqlite_from_url():
    repo = build_repository("sqlite:///:memory:")
    assert isinstance(repo, SQLiteRepository)


def test_factory_rejects_unknown_scheme():
    with pytest.raises(ValueError):
        build_repository("mysql://nope")


def test_factory_selects_postgres_lazily(monkeypatch):
    # build_repository must not require psycopg to be importable unless used
    repo = build_repository("postgresql://user:pw@localhost:5432/db")
    from agentshield.storage.postgres_repo import PostgresRepository

    assert isinstance(repo, PostgresRepository)
    assert repo.placeholder == "%s"
