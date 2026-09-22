"""Observer pattern: everything that reacts to a security decision.

The engine publishes events; observers subscribe.  Adding alerting, SIEM export
or an incident workflow later means adding an observer, not touching the engine.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from agentshield.core.models import Decision, SecurityAssessment

logger = logging.getLogger("agentshield.events")


class EventType(str, Enum):
    CALL_EVALUATED = "CALL_EVALUATED"
    CALL_ALLOWED = "CALL_ALLOWED"
    CALL_BLOCKED = "CALL_BLOCKED"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    STATE_CHANGED = "STATE_CHANGED"
    TOOL_EXECUTED = "TOOL_EXECUTED"
    TOOL_FAILED = "TOOL_FAILED"


@dataclass
class SecurityEvent:
    type: EventType
    assessment: SecurityAssessment | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": self.type.value,
            "at": self.at.isoformat(),
            "payload": self.payload,
        }
        if self.assessment is not None:
            data["assessment"] = self.assessment.to_dict()
        return data


class SecurityObserver(ABC):
    """Subscriber notified whenever the engine emits an event."""

    name: str = "observer"

    @abstractmethod
    def notify(self, event: SecurityEvent) -> None:
        """React to one security event. Must never raise."""


class EventBus:
    """The subject half of the Observer pattern."""

    def __init__(self) -> None:
        self._observers: list[SecurityObserver] = []

    def subscribe(self, observer: SecurityObserver) -> None:
        self._observers.append(observer)

    def unsubscribe(self, observer: SecurityObserver) -> None:
        if observer in self._observers:
            self._observers.remove(observer)

    @property
    def observers(self) -> list[SecurityObserver]:
        return list(self._observers)

    def publish(self, event: SecurityEvent) -> None:
        for observer in self._observers:
            try:
                observer.notify(event)
            except Exception:  # an observer must never break the security path
                logger.exception("observer %s failed on %s", observer.name, event.type)


class AuditObserver(SecurityObserver):
    """Append-only audit trail, in memory and optionally as JSON lines."""

    name = "audit"

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.records: list[dict[str, Any]] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def notify(self, event: SecurityEvent) -> None:
        record = event.to_dict()
        self.records.append(record)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")


class MetricsObserver(SecurityObserver):
    """Counters for the stage 3 benchmark and any dashboard."""

    name = "metrics"

    def __init__(self) -> None:
        self.decisions: Counter[str] = Counter()
        self.events: Counter[str] = Counter()
        self.signals: Counter[str] = Counter()
        self.latencies_ms: list[float] = []

    def notify(self, event: SecurityEvent) -> None:
        self.events[event.type.value] += 1
        assessment = event.assessment
        if event.type is EventType.CALL_EVALUATED and assessment is not None:
            self.decisions[assessment.decision.value] += 1
            self.latencies_ms.append(assessment.latency_ms)
            for signal in assessment.signals:
                self.signals[signal.category] += 1

    @property
    def total_evaluated(self) -> int:
        return sum(self.decisions.values())

    def average_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)

    def snapshot(self) -> dict[str, Any]:
        return {
            "evaluated": self.total_evaluated,
            "decisions": dict(self.decisions),
            "signals": dict(self.signals),
            "avg_latency_ms": round(self.average_latency_ms(), 3),
        }


class AlertObserver(SecurityObserver):
    """Logs a human-readable warning whenever something is refused."""

    name = "alert"

    def __init__(self, sink: logging.Logger | None = None) -> None:
        self.log = sink or logger
        self.alerts: list[str] = []

    def notify(self, event: SecurityEvent) -> None:
        if event.type not in (EventType.CALL_BLOCKED, EventType.APPROVAL_REQUIRED):
            return
        assessment = event.assessment
        if assessment is None:
            return
        message = (
            f"[{event.type.value}] {assessment.call.tool} "
            f"-> {assessment.call.destination or 'n/a'} "
            f"risk={assessment.risk_score} reason={assessment.reason}"
        )
        self.alerts.append(message)
        self.log.warning(message)


class IncidentObserver(SecurityObserver):
    """Opens an incident record for blocked calls and state escalations."""

    name = "incident"

    def __init__(self, threshold: int = 70) -> None:
        self.threshold = threshold
        self.incidents: list[dict[str, Any]] = []

    def notify(self, event: SecurityEvent) -> None:
        assessment = event.assessment
        if event.type is EventType.STATE_CHANGED:
            self.incidents.append(
                {
                    "kind": "SESSION_ESCALATION",
                    "session_id": event.payload.get("session_id"),
                    "state": event.payload.get("to"),
                    "at": event.at.isoformat(),
                }
            )
            return
        # Only the CALL_EVALUATED event is considered, so one call never opens
        # two incidents.
        if event.type is not EventType.CALL_EVALUATED or assessment is None:
            return
        if assessment.decision is Decision.BLOCK or assessment.risk_score >= self.threshold:
            self.incidents.append(
                {
                    "kind": "BLOCKED_CALL",
                    "session_id": assessment.call.session_id,
                    "tool": assessment.call.tool,
                    "risk_score": assessment.risk_score,
                    "reason": assessment.reason,
                    "signals": [s.to_dict() for s in assessment.top_signals()],
                    "at": event.at.isoformat(),
                }
            )
