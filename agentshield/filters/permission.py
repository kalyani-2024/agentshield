"""Filter 2: permissions, privilege escalation and excessive agency."""

from __future__ import annotations

from agentshield.core.models import FilterResult, Severity, ToolCall, Verdict
from agentshield.engine.session import Session
from agentshield.filters.base import SecurityFilter

#: Capability classes that always need an explicit human decision.
HIGH_AGENCY_TOOLS: frozenset[str] = frozenset(
    {
        "delete_file",
        "drop_table",
        "run_shell",
        "execute_code",
        "transfer_funds",
        "grant_access",
        "rotate_credentials",
    }
)


class PermissionFilter(SecurityFilter):
    """Checks the principal's grants and the session's current privilege state.

    This is the one filter that can block on its own before anything else runs:
    if the principal has no grant for the tool, nothing downstream matters.
    """

    name = "permission"

    def __init__(self, high_agency_tools: frozenset[str] = HIGH_AGENCY_TOOLS) -> None:
        super().__init__()
        self.high_agency_tools = high_agency_tools

    def check(self, call: ToolCall, session: Session) -> FilterResult:
        signals = []

        if not call.principal.may_use(call.tool):
            signals.append(
                self.signal(
                    category="unauthorized_tool",
                    score=100,
                    severity=Severity.CRITICAL,
                    message=(
                        f"principal {call.principal.id!r} has no grant for "
                        f"tool {call.tool!r}"
                    ),
                    evidence=f"allowed={sorted(call.principal.allowed_tools)}",
                    verdict=Verdict.BLOCK,
                )
            )
            return FilterResult.from_signals(signals)

        if not session.state.permits(call.category):
            signals.append(
                self.signal(
                    category="privilege_escalation",
                    score=95,
                    severity=Severity.CRITICAL,
                    message=(
                        f"session is {session.state.name}; capability "
                        f"{call.category.value} is withdrawn in this state"
                    ),
                    evidence=session.state.describe(),
                    verdict=Verdict.BLOCK,
                )
            )
            return FilterResult.from_signals(signals)

        if call.tool in self.high_agency_tools and not call.user_authorized:
            signals.append(
                self.signal(
                    category="excessive_agency",
                    score=55,
                    severity=Severity.HIGH,
                    message=(
                        f"{call.tool!r} is a high-agency action taken without "
                        f"explicit user authorization"
                    ),
                    verdict=Verdict.FLAG,
                )
            )

        return FilterResult.from_signals(signals)
