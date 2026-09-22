from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agentshield.core.models import (
    ContextChunk,
    Principal,
    ToolCall,
    ToolCategory,
    TrustLevel,
    Verdict,
)
from agentshield.engine.session import HistoryEntry, Session
from agentshield.filters import (
    DestinationTrustFilter,
    PermissionFilter,
    PromptInjectionFilter,
    SensitiveDataFilter,
    SequenceBehaviourFilter,
    TrustPolicy,
    build_chain,
)
from tests.conftest import POISONED_DOCUMENT


def make_call(**kwargs) -> ToolCall:
    defaults = dict(
        tool="send_email",
        arguments={"recipient": "bob@example.com", "body": "hello"},
        category=ToolCategory.COMMUNICATE,
        destination="bob@example.com",
        session_id="s1",
    )
    defaults.update(kwargs)
    return ToolCall(**defaults)


@pytest.fixture
def session() -> Session:
    return Session(session_id="s1")


# -- permission ----------------------------------------------------------
def test_permission_blocks_ungranted_tool(session):
    call = make_call(principal=Principal(id="p", allowed_tools=frozenset({"read_file"})))
    result = PermissionFilter().check(call, session)
    assert result.verdict is Verdict.BLOCK
    assert result.signals[0].category == "unauthorized_tool"


def test_permission_allows_granted_tool(session):
    call = make_call(principal=Principal(id="p", allowed_tools=frozenset({"send_email"})))
    assert PermissionFilter().check(call, session).verdict is Verdict.PASS


def test_permission_blocks_capability_withdrawn_by_state(session):
    from agentshield.engine.session import RestrictedState

    session.state = RestrictedState()
    result = PermissionFilter().check(make_call(), session)
    assert result.verdict is Verdict.BLOCK
    assert result.signals[0].category == "privilege_escalation"


def test_high_agency_tool_without_authorization_is_flagged(session):
    call = make_call(tool="run_shell", category=ToolCategory.EXECUTE, destination=None)
    result = PermissionFilter().check(call, session)
    assert result.verdict is Verdict.FLAG
    assert result.signals[0].category == "excessive_agency"


# -- prompt injection ----------------------------------------------------
def test_indirect_injection_in_untrusted_context_is_flagged(session):
    call = make_call(
        context=[ContextChunk(POISONED_DOCUMENT, "report.txt", TrustLevel.UNTRUSTED)]
    )
    result = PromptInjectionFilter().check(call, session)
    assert result.verdict is Verdict.FLAG
    assert result.signals[0].category == "indirect_prompt_injection"
    assert result.signals[0].score >= 80


def test_trusted_context_is_not_scanned(session):
    call = make_call(
        context=[ContextChunk(POISONED_DOCUMENT, "system", TrustLevel.TRUSTED)]
    )
    assert PromptInjectionFilter().check(call, session).verdict is Verdict.PASS


def test_clean_call_passes_injection_filter(session):
    assert PromptInjectionFilter().check(make_call(), session).verdict is Verdict.PASS


# -- sensitive data ------------------------------------------------------
def test_secret_in_outbound_body_is_flagged(session):
    call = make_call(arguments={"recipient": "a@b.com", "body": "AKIAIOSFODNN7EXAMPLE"})
    result = SensitiveDataFilter().check(call, session)
    assert result.verdict is Verdict.FLAG
    assert result.signals[0].category == "sensitive_data_egress"


def test_evidence_is_redacted(session):
    call = make_call(arguments={"recipient": "a@b.com", "body": "AKIAIOSFODNN7EXAMPLE"})
    evidence = SensitiveDataFilter().check(call, session).signals[0].evidence
    assert "IOSFODNN7EX" not in evidence
    assert evidence.startswith("AKIA")


def test_sensitive_path_access_is_flagged(session):
    call = make_call(
        tool="read_file",
        category=ToolCategory.READ_LOCAL,
        arguments={"path": "/home/user/.ssh/id_rsa"},
        destination="/home/user/.ssh/id_rsa",
    )
    result = SensitiveDataFilter().check(call, session)
    assert any(s.category == "sensitive_resource_access" for s in result.signals)


def test_egress_scores_higher_than_local_write(session):
    secret = "aws_secret_access_key=wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY"
    outbound = SensitiveDataFilter().check(
        make_call(arguments={"body": secret}), session
    )
    local = SensitiveDataFilter().check(
        make_call(
            tool="write_file",
            category=ToolCategory.WRITE_LOCAL,
            arguments={"content": secret},
            destination="notes.txt",
        ),
        session,
    )
    assert outbound.signals[0].score > local.signals[0].score


# -- destination trust ---------------------------------------------------
def test_denied_domain_blocks(session):
    policy = TrustPolicy(denied_domains={"attacker-drop.top"})
    call = make_call(
        arguments={"recipient": "x@attacker-drop.top"}, destination="x@attacker-drop.top"
    )
    result = DestinationTrustFilter(policy).check(call, session)
    assert result.verdict is Verdict.BLOCK


def test_allowed_domain_passes(session):
    policy = TrustPolicy(allowed_domains={"example.com"})
    assert DestinationTrustFilter(policy).check(make_call(), session).verdict is Verdict.PASS


def test_known_drop_host_is_low_trust(session):
    call = make_call(
        arguments={"url": "https://webhook.site/abcd"}, destination="https://webhook.site/abcd"
    )
    result = DestinationTrustFilter().check(call, session)
    assert result.signals[0].category == "untrusted_destination"


def test_local_actions_skip_destination_checks(session):
    call = make_call(
        tool="read_file", category=ToolCategory.READ_LOCAL, destination="notes.txt"
    )
    assert DestinationTrustFilter().check(call, session).verdict is Verdict.PASS


# -- sequence ------------------------------------------------------------
def _history(session: Session, *categories: ToolCategory) -> None:
    for index, category in enumerate(categories):
        session.history.append(
            HistoryEntry(
                tool=f"t{index}",
                category=category,
                destination=None,
                risk_score=0,
                decision="ALLOW",
                at=datetime.now(timezone.utc),
            )
        )


def test_read_then_send_matches_the_exfiltration_shape(session):
    _history(session, ToolCategory.READ_LOCAL)
    result = SequenceBehaviourFilter().check(make_call(), session)
    assert any(s.message.startswith("SEQ001") for s in result.signals)


def test_staged_cross_tool_attack_is_detected(session):
    _history(session, ToolCategory.READ_LOCAL, ToolCategory.READ_REMOTE)
    result = SequenceBehaviourFilter().check(make_call(), session)
    assert any(s.message.startswith("SEQ005") for s in result.signals)


def test_history_outside_the_window_is_ignored(session):
    session.history.append(
        HistoryEntry(
            tool="read_file",
            category=ToolCategory.READ_LOCAL,
            destination=None,
            risk_score=0,
            decision="ALLOW",
            at=datetime.now(timezone.utc) - timedelta(hours=2),
        )
    )
    assert SequenceBehaviourFilter().check(make_call(), session).verdict is Verdict.PASS


def test_repeated_blocks_raise_the_score(session):
    session.history.append(
        HistoryEntry(
            tool="send_email",
            category=ToolCategory.COMMUNICATE,
            destination=None,
            risk_score=90,
            decision="BLOCK",
            at=datetime.now(timezone.utc),
        )
    )
    result = SequenceBehaviourFilter().check(make_call(), session)
    assert any(s.category == "repeat_offender" for s in result.signals)


# -- chain ---------------------------------------------------------------
def test_chain_short_circuits_on_block(session):
    calls: list[str] = []

    class Recorder(SequenceBehaviourFilter):
        name = "recorder"

        def check(self, call, session):
            calls.append("ran")
            return super().check(call, session)

    head = build_chain([PermissionFilter(), Recorder()])
    call = make_call(principal=Principal(id="p", allowed_tools=frozenset({"read_file"})))
    result = head.handle(call, session)
    assert result.verdict is Verdict.BLOCK
    assert calls == []


def test_chain_collects_signals_from_every_link(session):
    head = build_chain([PromptInjectionFilter(), SensitiveDataFilter()])
    call = make_call(
        arguments={"recipient": "a@b.com", "body": "AKIAIOSFODNN7EXAMPLE"},
        context=[ContextChunk(POISONED_DOCUMENT, "doc", TrustLevel.UNTRUSTED)],
    )
    result = head.handle(call, session)
    filters = {signal.filter_name for signal in result.signals}
    assert filters == {"prompt_injection", "sensitive_data"}


def test_empty_chain_is_rejected():
    with pytest.raises(ValueError):
        build_chain([])
