"""Agent session tracking and the privilege State machine.

A session escalates through NORMAL -> SUSPICIOUS -> RESTRICTED -> QUARANTINED as
risk accumulates, and each state changes how strictly the policy engine judges
the next call.  States never de-escalate automatically inside stage 1: recovery
is an explicit operator action (``SessionStore.reset``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Iterable

from agentshield.core.models import ToolCall, ToolCategory

#: How much of a call's risk score carries over into the session's running risk.
RISK_DECAY = 0.75
#: Number of recent calls kept for sequence analysis.
HISTORY_LIMIT = 25


@dataclass(frozen=True)
class HistoryEntry:
    """A compact record of a past call in this session."""

    tool: str
    category: ToolCategory
    destination: str | None
    risk_score: int
    decision: str
    at: datetime

    def is_external(self) -> bool:
        return self.category in (ToolCategory.COMMUNICATE, ToolCategory.READ_REMOTE)


class SessionState(ABC):
    """Behaviour attached to one privilege level."""

    name: str = "ABSTRACT"
    #: risk needed to enter this state
    entry_threshold: int = 0
    #: added to every risk score evaluated while the session is in this state
    risk_surcharge: int = 0
    #: risk at which a call is escalated from ALLOW to REQUIRE_APPROVAL
    approval_threshold: int = 40
    #: risk at which a call is blocked outright
    block_threshold: int = 70

    @abstractmethod
    def permits(self, category: ToolCategory) -> bool:
        """Whether this state still allows the capability class at all."""

    def describe(self) -> str:
        return (
            f"{self.name}: approval>={self.approval_threshold}, "
            f"block>={self.block_threshold}, surcharge=+{self.risk_surcharge}"
        )


class NormalState(SessionState):
    name = "NORMAL"
    entry_threshold = 0
    risk_surcharge = 0
    approval_threshold = 40
    block_threshold = 70

    def permits(self, category: ToolCategory) -> bool:
        return True


class SuspiciousState(SessionState):
    """Something odd happened; judge the next calls more harshly."""

    name = "SUSPICIOUS"
    entry_threshold = 35
    risk_surcharge = 5
    approval_threshold = 30
    block_threshold = 65

    def permits(self, category: ToolCategory) -> bool:
        return True


class RestrictedState(SessionState):
    """Read-only mode: nothing may leave the boundary without a human."""

    name = "RESTRICTED"
    entry_threshold = 60
    risk_surcharge = 15
    approval_threshold = 20
    block_threshold = 55

    def permits(self, category: ToolCategory) -> bool:
        return category not in (ToolCategory.COMMUNICATE, ToolCategory.EXECUTE)


class QuarantinedState(SessionState):
    """The session is considered compromised; nothing runs."""

    name = "QUARANTINED"
    entry_threshold = 85
    risk_surcharge = 40
    approval_threshold = 0
    block_threshold = 1

    def permits(self, category: ToolCategory) -> bool:
        return False


#: Ordered from most to least severe, so lookup picks the highest match.
STATE_LADDER: tuple[SessionState, ...] = (
    QuarantinedState(),
    RestrictedState(),
    SuspiciousState(),
    NormalState(),
)


def state_for_risk(cumulative_risk: int) -> SessionState:
    """The state a session with this much accumulated risk belongs in."""
    for state in STATE_LADDER:
        if cumulative_risk >= state.entry_threshold:
            return state
    return STATE_LADDER[-1]


@dataclass
class Session:
    """Mutable per-agent-session security state."""

    session_id: str
    principal_id: str = "anonymous"
    cumulative_risk: int = 0
    state: SessionState = field(default_factory=NormalState)
    history: Deque[HistoryEntry] = field(
        default_factory=lambda: deque(maxlen=HISTORY_LIMIT)
    )
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def record(self, call: ToolCall, risk_score: int, decision: str) -> SessionState | None:
        """Append a call to the history and re-evaluate the privilege state.

        Returns the new state when a transition happened, otherwise ``None``.
        """
        self.history.append(
            HistoryEntry(
                tool=call.tool,
                category=call.category,
                destination=call.destination,
                risk_score=risk_score,
                decision=decision,
                at=datetime.now(timezone.utc),
            )
        )
        self.cumulative_risk = max(
            int(self.cumulative_risk * RISK_DECAY),
            risk_score,
            # sustained medium risk should still escalate over time
            min(100, int(self.cumulative_risk * RISK_DECAY) + risk_score // 3),
        )
        return self._transition()

    def _transition(self) -> SessionState | None:
        target = state_for_risk(self.cumulative_risk)
        if target.name == self.state.name:
            return None
        # Privileges only tighten within a session; recovery is explicit.
        if target.entry_threshold < self.state.entry_threshold:
            return None
        self.state = target
        return target

    def recent(self, n: int = HISTORY_LIMIT) -> list[HistoryEntry]:
        return list(self.history)[-n:]

    def categories_seen(self, n: int = HISTORY_LIMIT) -> list[ToolCategory]:
        return [entry.category for entry in self.recent(n)]

    def reset(self) -> None:
        """Operator-initiated recovery back to NORMAL."""
        self.cumulative_risk = 0
        self.state = NormalState()
        self.history.clear()


class SessionStore:
    """In-memory session registry.

    Stage 2 swaps this for Redis behind the same interface; nothing above it
    knows where the state actually lives.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def get(self, session_id: str, principal_id: str = "anonymous") -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            session = Session(session_id=session_id, principal_id=principal_id)
            self._sessions[session_id] = session
        return session

    def all(self) -> Iterable[Session]:
        return self._sessions.values()

    def reset(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is not None:
            session.reset()

    def clear(self) -> None:
        self._sessions.clear()
