"""Canonical data model for AgentShield.

Every framework-specific tool call is normalised into :class:`ToolCall` before it
reaches the security pipeline, so the engine never has to care where a request
came from.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Verdict(str, Enum):
    """What a single security filter thinks of a request."""

    PASS = "PASS"
    FLAG = "FLAG"
    BLOCK = "BLOCK"


class Decision(str, Enum):
    """What the policy engine decides to do with a request."""

    ALLOW = "ALLOW"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    BLOCK = "BLOCK"


class Severity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ToolCategory(str, Enum):
    """Coarse capability class, used for permissions and sequence analysis."""

    READ_LOCAL = "READ_LOCAL"        # read_file, list_dir
    READ_REMOTE = "READ_REMOTE"      # web_search, fetch_url
    WRITE_LOCAL = "WRITE_LOCAL"      # write_file, delete_file
    QUERY_DATA = "QUERY_DATA"        # database queries
    COMMUNICATE = "COMMUNICATE"      # send_email, post_message, webhook
    EXECUTE = "EXECUTE"              # shell / code execution
    UNKNOWN = "UNKNOWN"


class TrustLevel(str, Enum):
    """Provenance of a piece of content the agent is acting on."""

    TRUSTED = "TRUSTED"        # the operator / system prompt
    USER = "USER"              # the human in the loop
    UNTRUSTED = "UNTRUSTED"    # documents, web pages, emails, tool output


@dataclass
class Principal:
    """Who the agent is acting on behalf of."""

    id: str
    display_name: str = ""
    roles: tuple[str, ...] = ()
    #: tool names (or "*") this principal is allowed to invoke
    allowed_tools: frozenset[str] = frozenset({"*"})

    def may_use(self, tool_name: str) -> bool:
        return "*" in self.allowed_tools or tool_name in self.allowed_tools


@dataclass
class ContextChunk:
    """A piece of context that influenced the agent's decision to call a tool."""

    content: str
    source: str = "unknown"
    trust: TrustLevel = TrustLevel.UNTRUSTED


@dataclass
class ToolCall:
    """The normalised internal representation of a requested action."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    principal: Principal = field(default_factory=lambda: Principal(id="anonymous"))
    session_id: str = field(default_factory=lambda: _new_id("sess"))
    call_id: str = field(default_factory=lambda: _new_id("call"))
    category: ToolCategory = ToolCategory.UNKNOWN
    #: where the effect lands: an email address, host, file path, table name...
    destination: str | None = None
    #: context the agent read before deciding on this call
    context: list[ContextChunk] = field(default_factory=list)
    #: True when the human explicitly approved this exact action
    user_authorized: bool = False
    source_framework: str = "internal"
    created_at: datetime = field(default_factory=_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def argument_text(self) -> str:
        """All argument values flattened into one searchable string."""
        return "\n".join(_flatten(self.arguments))

    def context_text(self, only_untrusted: bool = False) -> str:
        chunks = self.context
        if only_untrusted:
            chunks = [c for c in chunks if c.trust is TrustLevel.UNTRUSTED]
        return "\n".join(c.content for c in chunks)

    def searchable_text(self) -> str:
        return f"{self.argument_text()}\n{self.context_text()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "session_id": self.session_id,
            "tool": self.tool,
            "arguments": self.arguments,
            "category": self.category.value,
            "destination": self.destination,
            "principal": self.principal.id,
            "user_authorized": self.user_authorized,
            "source_framework": self.source_framework,
            "created_at": self.created_at.isoformat(),
        }


def _flatten(value: Any) -> list[str]:
    """Depth-first flatten of nested argument structures into strings."""
    if isinstance(value, dict):
        out: list[str] = []
        for key, val in value.items():
            out.append(str(key))
            out.extend(_flatten(val))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(_flatten(item))
        return out
    if value is None:
        return []
    return [str(value)]


@dataclass
class RiskSignal:
    """One piece of evidence produced by a filter."""

    filter_name: str
    category: str
    score: int                      # 0-100, this signal's own risk contribution
    severity: Severity
    message: str
    evidence: str = ""
    verdict: Verdict = Verdict.FLAG

    def to_dict(self) -> dict[str, Any]:
        return {
            "filter": self.filter_name,
            "category": self.category,
            "score": self.score,
            "severity": self.severity.value,
            "message": self.message,
            "evidence": self.evidence[:200],
            "verdict": self.verdict.value,
        }


@dataclass
class FilterResult:
    """What one filter returns to the chain."""

    verdict: Verdict = Verdict.PASS
    signals: list[RiskSignal] = field(default_factory=list)

    @classmethod
    def from_signals(cls, signals: list[RiskSignal]) -> "FilterResult":
        if not signals:
            return cls()
        verdict = Verdict.PASS
        for signal in signals:
            if signal.verdict is Verdict.BLOCK:
                verdict = Verdict.BLOCK
                break
            if signal.verdict is Verdict.FLAG:
                verdict = Verdict.FLAG
        return cls(verdict=verdict, signals=signals)


@dataclass
class SecurityAssessment:
    """The full result of running a tool call through the pipeline."""

    call: ToolCall
    decision: Decision
    risk_score: int
    signals: list[RiskSignal] = field(default_factory=list)
    #: privilege state the decision was made under
    session_state: str = "NORMAL"
    #: privilege state the session holds after this call was recorded
    resulting_state: str = "NORMAL"
    reason: str = ""
    latency_ms: float = 0.0
    evaluated_at: datetime = field(default_factory=_now)

    @property
    def blocked(self) -> bool:
        return self.decision is Decision.BLOCK

    @property
    def allowed(self) -> bool:
        return self.decision is Decision.ALLOW

    def top_signals(self, n: int = 5) -> list[RiskSignal]:
        return sorted(self.signals, key=lambda s: s.score, reverse=True)[:n]

    def to_dict(self) -> dict[str, Any]:
        return {
            "call": self.call.to_dict(),
            "decision": self.decision.value,
            "risk_score": self.risk_score,
            "session_state": self.session_state,
            "resulting_state": self.resulting_state,
            "reason": self.reason,
            "latency_ms": round(self.latency_ms, 2),
            "signals": [s.to_dict() for s in self.signals],
            "evaluated_at": self.evaluated_at.isoformat(),
        }
