"""Policy decision: score plus session state in, ALLOW/APPROVAL/BLOCK out."""

from __future__ import annotations

from dataclasses import dataclass

from agentshield.core.models import Decision, RiskSignal, ToolCall, Verdict
from agentshield.engine.risk import RiskBreakdown
from agentshield.engine.session import Session


@dataclass
class PolicyDecision:
    decision: Decision
    reason: str


class PolicyEngine:
    """Applies thresholds, with the session state deciding how strict they are.

    Thresholds live on the state objects (see :mod:`agentshield.engine.session`)
    so that escalating a session automatically tightens every later judgement.
    """

    def __init__(self, fail_closed: bool = True) -> None:
        #: when True, an evaluation error results in BLOCK rather than ALLOW
        self.fail_closed = fail_closed

    def decide(
        self,
        call: ToolCall,
        session: Session,
        breakdown: RiskBreakdown,
        chain_verdict: Verdict,
        signals: list[RiskSignal],
    ) -> PolicyDecision:
        state = session.state

        if chain_verdict is Verdict.BLOCK:
            blocking = next(
                (s for s in signals if s.verdict is Verdict.BLOCK), None
            )
            reason = blocking.message if blocking else "a filter refused the call"
            return PolicyDecision(Decision.BLOCK, reason)

        score = breakdown.total

        if score >= state.block_threshold:
            return PolicyDecision(
                Decision.BLOCK,
                f"risk {score} >= block threshold {state.block_threshold} "
                f"in state {state.name}: {self._headline(breakdown)}",
            )

        if score >= state.approval_threshold:
            if call.user_authorized:
                return PolicyDecision(
                    Decision.ALLOW,
                    f"risk {score} cleared by explicit user authorization",
                )
            return PolicyDecision(
                Decision.REQUIRE_APPROVAL,
                f"risk {score} >= approval threshold {state.approval_threshold} "
                f"in state {state.name}: {self._headline(breakdown)}",
            )

        return PolicyDecision(Decision.ALLOW, f"risk {score} below policy thresholds")

    def on_error(self, error: Exception) -> PolicyDecision:
        """What to do when the pipeline itself fails."""
        if self.fail_closed:
            return PolicyDecision(
                Decision.BLOCK, f"security pipeline error, failing closed: {error}"
            )
        return PolicyDecision(
            Decision.REQUIRE_APPROVAL, f"security pipeline error: {error}"
        )

    @staticmethod
    def _headline(breakdown: RiskBreakdown) -> str:
        return breakdown.reasons[0] if breakdown.reasons else "no dominant signal"
