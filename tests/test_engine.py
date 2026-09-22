from __future__ import annotations

import pytest

from agentshield.core.models import (
    Decision,
    RiskSignal,
    Severity,
    ToolCall,
    ToolCategory,
    Verdict,
)
from agentshield.engine import (
    EventType,
    RiskAggregator,
    SecurityEngine,
    Session,
    SessionStore,
)
from agentshield.engine.events import IncidentObserver, MetricsObserver
from agentshield.engine.session import (
    NormalState,
    QuarantinedState,
    RestrictedState,
    state_for_risk,
)
from agentshield.filters.base import SecurityFilter


def signal(category: str, score: int) -> RiskSignal:
    return RiskSignal(
        filter_name="t",
        category=category,
        score=score,
        severity=Severity.HIGH,
        message=category,
    )


# -- risk aggregation ----------------------------------------------------
def test_no_signals_means_no_risk():
    assert RiskAggregator().aggregate([]).total == 0


def test_strongest_signal_sets_the_floor():
    breakdown = RiskAggregator().aggregate([signal("sensitive_data_egress", 80)])
    assert breakdown.total == 80
    assert breakdown.base == 80


def test_corroborating_signals_add_less_than_they_score():
    single = RiskAggregator().aggregate([signal("suspicious_sequence", 50)]).total
    pair = RiskAggregator().aggregate(
        [signal("suspicious_sequence", 50), signal("tool_call_burst", 50)]
    ).total
    assert single < pair < single + 50


def test_correlated_categories_earn_a_bonus():
    plain = RiskAggregator().aggregate(
        [signal("indirect_prompt_injection", 40), signal("tool_call_burst", 10)]
    )
    correlated = RiskAggregator().aggregate(
        [signal("indirect_prompt_injection", 40), signal("sensitive_data_egress", 10)]
    )
    assert correlated.correlation > 0
    assert plain.correlation == 0
    assert correlated.total > plain.total


def test_score_is_clamped_to_100():
    signals = [signal("sensitive_data_egress", 100) for _ in range(6)]
    assert RiskAggregator().aggregate(signals, state_surcharge=40).total == 100


# -- session state machine ----------------------------------------------
@pytest.mark.parametrize(
    ("risk", "expected"),
    [(0, "NORMAL"), (10, "NORMAL"), (47, "SUSPICIOUS"), (76, "RESTRICTED"), (94, "QUARANTINED")],
)
def test_state_ladder(risk, expected):
    assert state_for_risk(risk).name == expected


def test_state_only_tightens_within_a_session():
    session = Session(session_id="s")
    call = ToolCall(tool="send_email", category=ToolCategory.COMMUNICATE)
    session.record(call, 95, "BLOCK")
    assert session.state.name == "QUARANTINED"
    session.record(call, 0, "ALLOW")
    assert session.state.name == "QUARANTINED"


def test_reset_returns_a_session_to_normal():
    session = Session(session_id="s")
    session.record(ToolCall(tool="x"), 95, "BLOCK")
    session.reset()
    assert session.state.name == "NORMAL"
    assert session.cumulative_risk == 0
    assert not session.history


def test_quarantine_permits_nothing():
    assert not QuarantinedState().permits(ToolCategory.READ_LOCAL)
    assert RestrictedState().permits(ToolCategory.READ_LOCAL)
    assert not RestrictedState().permits(ToolCategory.COMMUNICATE)
    assert NormalState().permits(ToolCategory.EXECUTE)


def test_sustained_medium_risk_escalates_over_time():
    session = Session(session_id="s")
    call = ToolCall(tool="send_email", category=ToolCategory.COMMUNICATE)
    for _ in range(6):
        session.record(call, 45, "REQUIRE_APPROVAL")
    assert session.state.name in {"SUSPICIOUS", "RESTRICTED"}


# -- engine --------------------------------------------------------------
def test_engine_allows_a_clean_call():
    engine = SecurityEngine()
    call = ToolCall(
        tool="read_file",
        arguments={"path": "notes.txt"},
        category=ToolCategory.READ_LOCAL,
    )
    assessment = engine.evaluate(call)
    assert assessment.decision is Decision.ALLOW
    assert assessment.risk_score == 0


def test_engine_fails_closed_when_a_filter_raises():
    class Exploding(SecurityFilter):
        name = "boom"

        def check(self, call, session):
            raise RuntimeError("filter is broken")

    engine = SecurityEngine(filters=[Exploding()])
    assessment = engine.evaluate(ToolCall(tool="send_email"))
    assert assessment.decision is Decision.BLOCK
    assert "failing closed" in assessment.reason


def test_engine_publishes_events_to_observers():
    engine = SecurityEngine()
    metrics = engine.subscribe(MetricsObserver())
    engine.evaluate(ToolCall(tool="read_file", category=ToolCategory.READ_LOCAL))
    assert metrics.total_evaluated == 1
    assert metrics.events[EventType.CALL_ALLOWED.value] == 1


def test_a_failing_observer_does_not_break_the_pipeline():
    class Broken:
        name = "broken"

        def notify(self, event):
            raise RuntimeError("observer exploded")

    engine = SecurityEngine()
    engine.bus.subscribe(Broken())
    assert engine.evaluate(ToolCall(tool="read_file")).decision is Decision.ALLOW


def test_incident_is_opened_once_per_blocked_call():
    engine = SecurityEngine()
    incidents = engine.subscribe(IncidentObserver())

    class Blocker(SecurityFilter):
        name = "blocker"

        def check(self, call, session):
            from agentshield.core.models import FilterResult

            return FilterResult.from_signals(
                [
                    RiskSignal(
                        filter_name="blocker",
                        category="unauthorized_tool",
                        score=100,
                        severity=Severity.CRITICAL,
                        message="nope",
                        verdict=Verdict.BLOCK,
                    )
                ]
            )

    engine.filters = [Blocker()]
    engine.chain = Blocker()
    engine.evaluate(ToolCall(tool="send_email"))
    blocked = [i for i in incidents.incidents if i["kind"] == "BLOCKED_CALL"]
    assert len(blocked) == 1


def test_sessions_are_shared_by_id():
    store = SessionStore()
    assert store.get("s1") is store.get("s1")
    assert store.get("s1") is not store.get("s2")
