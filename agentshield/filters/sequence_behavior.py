"""Filter 5: suspicious sequences and cross-tool attack shapes.

Individually harmless calls can form a dangerous pattern.  The classic one is::

    read_file(credentials.txt) -> search_contacts("external") -> send_email(...)

This filter is the only one that reasons over session history rather than the
call in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from agentshield.core.models import FilterResult, ToolCall, ToolCategory, Verdict
from agentshield.engine.session import Session
from agentshield.filters.base import SecurityFilter, severity_for


@dataclass(frozen=True)
class SequenceRule:
    """A risky ordering of capability classes ending in the current call."""

    id: str
    description: str
    score: int
    #: categories that must have occurred earlier, in this order
    precursors: tuple[ToolCategory, ...]
    #: the category of the call being evaluated
    trigger: ToolCategory


SEQUENCE_RULES: tuple[SequenceRule, ...] = (
    SequenceRule(
        id="SEQ001",
        description="local data read followed by an outbound send (exfiltration shape)",
        score=75,
        precursors=(ToolCategory.READ_LOCAL,),
        trigger=ToolCategory.COMMUNICATE,
    ),
    SequenceRule(
        id="SEQ002",
        description="database query followed by an outbound send (data exfiltration)",
        score=80,
        precursors=(ToolCategory.QUERY_DATA,),
        trigger=ToolCategory.COMMUNICATE,
    ),
    SequenceRule(
        id="SEQ003",
        description="untrusted web content read, then a local write (poisoned write)",
        score=55,
        precursors=(ToolCategory.READ_REMOTE,),
        trigger=ToolCategory.WRITE_LOCAL,
    ),
    SequenceRule(
        id="SEQ004",
        description="read then execute (untrusted content reaching an executor)",
        score=85,
        precursors=(ToolCategory.READ_REMOTE,),
        trigger=ToolCategory.EXECUTE,
    ),
    SequenceRule(
        id="SEQ005",
        description="local read, remote lookup, then send (staged cross-tool attack)",
        score=90,
        precursors=(ToolCategory.READ_LOCAL, ToolCategory.READ_REMOTE),
        trigger=ToolCategory.COMMUNICATE,
    ),
)


class SequenceBehaviourFilter(SecurityFilter):
    """Matches the recent call history against known attack shapes."""

    name = "sequence_behaviour"

    def __init__(
        self,
        rules: tuple[SequenceRule, ...] = SEQUENCE_RULES,
        window: timedelta = timedelta(minutes=10),
        burst_limit: int = 8,
    ) -> None:
        super().__init__()
        self.rules = rules
        self.window = window
        self.burst_limit = burst_limit

    def check(self, call: ToolCall, session: Session) -> FilterResult:
        signals = []
        cutoff = datetime.now(timezone.utc) - self.window
        recent = [e for e in session.recent() if e.at >= cutoff]
        categories = [e.category for e in recent]

        for rule in self.rules:
            if call.category is not rule.trigger:
                continue
            if not _contains_in_order(categories, rule.precursors):
                continue
            signals.append(
                self.signal(
                    category="suspicious_sequence",
                    score=rule.score,
                    severity=severity_for(rule.score),
                    message=f"{rule.id}: {rule.description}",
                    evidence=" -> ".join(
                        [c.value for c in categories[-4:]] + [call.category.value]
                    ),
                    verdict=Verdict.FLAG,
                )
            )

        if len(recent) >= self.burst_limit:
            score = min(70, 25 + (len(recent) - self.burst_limit) * 5)
            signals.append(
                self.signal(
                    category="tool_call_burst",
                    score=score,
                    severity=severity_for(score),
                    message=(
                        f"{len(recent)} tool calls in the last "
                        f"{int(self.window.total_seconds() // 60)} minutes"
                    ),
                    evidence=f"burst_limit={self.burst_limit}",
                    verdict=Verdict.FLAG,
                )
            )

        repeats = [e for e in recent if e.decision == "BLOCK"]
        if repeats:
            score = min(90, 30 + 20 * len(repeats))
            signals.append(
                self.signal(
                    category="repeat_offender",
                    score=score,
                    severity=severity_for(score),
                    message=(
                        f"{len(repeats)} call(s) already blocked in this session; "
                        f"the agent keeps retrying"
                    ),
                    evidence=", ".join(sorted({e.tool for e in repeats})),
                    verdict=Verdict.FLAG,
                )
            )

        return FilterResult.from_signals(signals)


def _contains_in_order(
    haystack: list[ToolCategory], needles: tuple[ToolCategory, ...]
) -> bool:
    """True when every needle appears in ``haystack`` in the given order."""
    it = iter(haystack)
    return all(any(item is needle for item in it) for needle in needles)
