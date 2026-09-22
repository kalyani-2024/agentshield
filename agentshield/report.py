"""The console verdict box from the design document."""

from __future__ import annotations

from agentshield.core.models import Decision, SecurityAssessment

_WIDTH = 66

_DECISION_LABEL = {
    Decision.ALLOW: "ACTION ALLOWED",
    Decision.REQUIRE_APPROVAL: "USER APPROVAL REQUIRED",
    Decision.BLOCK: "ACTION BLOCKED",
}


def _row(label: str, value: str) -> str:
    body = f" {label:<24}: {value}"
    return f"|{body:<{_WIDTH}}|"


def _line(char: str = "-") -> str:
    return f"+{char * _WIDTH}+"


def _centre(text: str) -> str:
    return f"|{text.center(_WIDTH)}|"


def render(assessment: SecurityAssessment) -> str:
    """Render one assessment as the security panel."""
    call = assessment.call
    categories = {signal.category for signal in assessment.signals}

    lines = [
        _line("="),
        _centre("AGENTSHIELD SECURITY"),
        _line("="),
        _row("Requested action", call.tool.upper()),
        _row("Destination", call.destination or "n/a"),
        _row(
            "Session state",
            assessment.session_state
            if assessment.resulting_state == assessment.session_state
            else f"{assessment.session_state} -> {assessment.resulting_state}",
        ),
        _centre(""),
        _row(
            "Destination trust",
            "LOW" if "untrusted_destination" in categories else "OK",
        ),
        _row(
            "Sensitive information",
            "DETECTED"
            if categories & {"sensitive_data", "sensitive_data_egress",
                             "sensitive_resource_access"}
            else "none",
        ),
        _row("User authorization", "PRESENT" if call.user_authorized else "ABSENT"),
        _row(
            "Prompt-injection context",
            "DETECTED"
            if categories & {"prompt_injection", "indirect_prompt_injection"}
            else "none",
        ),
        _row(
            "Sequence behaviour",
            "SUSPICIOUS" if "suspicious_sequence" in categories else "normal",
        ),
        _centre(""),
        _row("Risk score", f"{assessment.risk_score} / 100"),
        _row("Latency", f"{assessment.latency_ms:.2f} ms"),
        _centre(""),
        _centre(_DECISION_LABEL[assessment.decision]),
        _line("="),
    ]

    if assessment.signals:
        lines.append("  Evidence:")
        for signal in assessment.top_signals():
            lines.append(
                f"    - [{signal.severity.value:<8}] {signal.score:>3}  {signal.message}"
            )
            if signal.evidence:
                lines.append(f"              {signal.evidence[:100]}")
    lines.append(f"  Reason: {assessment.reason}")
    return "\n".join(lines)


def print_report(assessment: SecurityAssessment) -> None:
    print(render(assessment))
