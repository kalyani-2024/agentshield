"""The security engine: pipes-and-filters over a chain of responsibility."""

from __future__ import annotations

import time

from agentshield.core.models import (
    Decision,
    FilterResult,
    SecurityAssessment,
    ToolCall,
    Verdict,
)
from agentshield.engine.events import (
    AuditObserver,
    EventBus,
    EventType,
    MetricsObserver,
    SecurityEvent,
    SecurityObserver,
)
from agentshield.engine.policy import PolicyEngine
from agentshield.engine.risk import RiskAggregator
from agentshield.engine.session import Session, SessionStore
from agentshield.filters.base import SecurityFilter, build_chain
from agentshield.filters.destination_trust import DestinationTrustFilter, TrustPolicy
from agentshield.filters.permission import PermissionFilter
from agentshield.filters.prompt_injection import PromptInjectionFilter
from agentshield.filters.sensitive_data import SensitiveDataFilter
from agentshield.filters.sequence_behavior import SequenceBehaviourFilter


def default_filters(trust_policy: TrustPolicy | None = None) -> list[SecurityFilter]:
    """The stage 1 pipeline, in evaluation order.

    Permission runs first because it is the cheapest hard stop; the behavioural
    filter runs last because it needs the rest of the picture to be useful.
    """
    return [
        PermissionFilter(),
        PromptInjectionFilter(),
        SensitiveDataFilter(),
        DestinationTrustFilter(trust_policy),
        SequenceBehaviourFilter(),
    ]


class SecurityEngine:
    """Evaluates a normalised :class:`ToolCall` and returns an assessment.

    The engine itself holds no detection logic: it runs the chain, aggregates,
    asks the policy, updates session state and publishes events.  New checks are
    new filters (microkernel/plugin style), not changes here.
    """

    def __init__(
        self,
        filters: list[SecurityFilter] | None = None,
        sessions: SessionStore | None = None,
        aggregator: RiskAggregator | None = None,
        policy: PolicyEngine | None = None,
        bus: EventBus | None = None,
        trust_policy: TrustPolicy | None = None,
    ) -> None:
        self.filters = filters if filters is not None else default_filters(trust_policy)
        self.chain = build_chain(self.filters)
        self.sessions = sessions or SessionStore()
        self.aggregator = aggregator or RiskAggregator()
        self.policy = policy or PolicyEngine()
        self.bus = bus or EventBus()

    # -- wiring ----------------------------------------------------------
    def subscribe(self, observer: SecurityObserver) -> SecurityObserver:
        self.bus.subscribe(observer)
        return observer

    def with_default_observers(self) -> "SecurityEngine":
        """Attach an audit trail and metrics collector; handy for demos/tests."""
        self.subscribe(AuditObserver())
        self.subscribe(MetricsObserver())
        return self

    # -- evaluation ------------------------------------------------------
    def evaluate(self, call: ToolCall) -> SecurityAssessment:
        started = time.perf_counter()
        session = self.sessions.get(call.session_id, call.principal.id)

        try:
            result = self.chain.handle(call, session)
        except Exception as error:  # a broken filter must not open the gate
            decision = self.policy.on_error(error)
            assessment = SecurityAssessment(
                call=call,
                decision=decision.decision,
                risk_score=100 if decision.decision is Decision.BLOCK else 50,
                signals=[],
                session_state=session.state.name,
                reason=decision.reason,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            self._publish(assessment, session)
            return assessment

        breakdown = self.aggregator.aggregate(
            result.signals, state_surcharge=session.state.risk_surcharge
        )
        policy_decision = self.policy.decide(
            call, session, breakdown, result.verdict, result.signals
        )

        assessment = SecurityAssessment(
            call=call,
            decision=policy_decision.decision,
            risk_score=breakdown.total,
            signals=result.signals,
            session_state=session.state.name,
            reason=policy_decision.reason,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        self._publish(assessment, session)
        return assessment

    # -- internals -------------------------------------------------------
    def _publish(self, assessment: SecurityAssessment, session: Session) -> None:
        previous_state = session.state.name
        new_state = session.record(
            assessment.call, assessment.risk_score, assessment.decision.value
        )
        # ``session_state`` stays as it was when the decision was taken; the new
        # state only governs the *next* call.
        assessment.resulting_state = session.state.name

        self.bus.publish(
            SecurityEvent(type=EventType.CALL_EVALUATED, assessment=assessment)
        )
        self.bus.publish(
            SecurityEvent(type=_DECISION_EVENTS[assessment.decision], assessment=assessment)
        )
        if new_state is not None:
            self.bus.publish(
                SecurityEvent(
                    type=EventType.STATE_CHANGED,
                    assessment=assessment,
                    payload={
                        "session_id": session.session_id,
                        "from": previous_state,
                        "to": new_state.name,
                        "cumulative_risk": session.cumulative_risk,
                    },
                )
            )


_DECISION_EVENTS = {
    Decision.ALLOW: EventType.CALL_ALLOWED,
    Decision.BLOCK: EventType.CALL_BLOCKED,
    Decision.REQUIRE_APPROVAL: EventType.APPROVAL_REQUIRED,
}


__all__ = ["SecurityEngine", "default_filters", "FilterResult", "Verdict"]
