"""Exceptions raised by the AgentShield runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from agentshield.core.models import SecurityAssessment


class AgentShieldError(Exception):
    """Base class for every AgentShield error."""


class ToolCallBlocked(AgentShieldError):
    """Raised when the policy engine refuses a tool call."""

    def __init__(self, assessment: "SecurityAssessment") -> None:
        self.assessment = assessment
        super().__init__(
            f"Blocked {assessment.call.tool} "
            f"(risk {assessment.risk_score}/100): {assessment.reason}"
        )


class ApprovalRequired(AgentShieldError):
    """Raised when a call needs a human decision before it can run."""

    def __init__(self, assessment: "SecurityAssessment") -> None:
        self.assessment = assessment
        super().__init__(
            f"Approval required for {assessment.call.tool} "
            f"(risk {assessment.risk_score}/100): {assessment.reason}"
        )


class AdapterError(AgentShieldError):
    """Raised when a foreign tool call cannot be normalised."""
