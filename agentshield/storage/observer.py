"""A SecurityObserver that writes the event stream to a Repository.

This is how persistence attaches to the engine: the engine already publishes
every decision, so Stage 2 just subscribes a writer.  The engine itself is
untouched.
"""

from __future__ import annotations

import logging

from agentshield.engine.events import EventType, SecurityEvent, SecurityObserver
from agentshield.storage.repository import Repository

logger = logging.getLogger("agentshield.storage")


class PersistenceObserver(SecurityObserver):
    """Persists calls, decisions, incidents and the raw event stream."""

    name = "persistence"

    def __init__(
        self,
        repository: Repository,
        store_events: bool = True,
        session_store: Any | None = None,
    ) -> None:
        self.repo = repository
        self.store_events = store_events
        #: optional session registry, so the sessions table tracks live state
        self.session_store = session_store

    def notify(self, event: SecurityEvent) -> None:
        try:
            self._handle(event)
        except Exception:  # persistence must never break the security path
            logger.exception("failed to persist %s", event.type)

    def _handle(self, event: SecurityEvent) -> None:
        assessment = event.assessment

        if event.type is EventType.CALL_EVALUATED and assessment is not None:
            self.repo.upsert_principal(assessment.call.principal)
            self.repo.save_assessment(assessment)
            if self.session_store is not None:
                self.repo.upsert_session(
                    self.session_store.get(assessment.call.session_id)
                )

        elif event.type is EventType.STATE_CHANGED:
            self.repo.save_incident(
                {
                    "kind": "SESSION_ESCALATION",
                    "session_id": event.payload.get("session_id"),
                    "reason": f"{event.payload.get('from')} -> {event.payload.get('to')}",
                    "risk_score": event.payload.get("cumulative_risk"),
                    **event.payload,
                }
            )

        elif event.type in (EventType.CALL_BLOCKED,) and assessment is not None:
            self.repo.save_incident(
                {
                    "kind": "BLOCKED_CALL",
                    "session_id": assessment.call.session_id,
                    "tool": assessment.call.tool,
                    "risk_score": assessment.risk_score,
                    "reason": assessment.reason,
                    "signals": [s.to_dict() for s in assessment.top_signals()],
                }
            )

        elif event.type is EventType.APPROVAL_REQUIRED and assessment is not None:
            self.repo.record_approval(
                assessment.call.call_id, assessment.call.session_id, "PENDING"
            )

        if self.store_events:
            call_id = assessment.call.call_id if assessment else None
            session_id = (
                assessment.call.session_id if assessment
                else event.payload.get("session_id")
            )
            self.repo.save_event(event.type.value, call_id, session_id, event.payload)
