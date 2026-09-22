"""Chain of Responsibility scaffolding for the security pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agentshield.core.models import (
    FilterResult,
    RiskSignal,
    Severity,
    ToolCall,
    Verdict,
)
from agentshield.engine.session import Session


class SecurityFilter(ABC):
    """One link in the security chain.

    Subclasses implement :meth:`check`; the base class owns the chaining, so a
    filter never needs to know who comes after it.
    """

    name: str = "filter"

    def __init__(self) -> None:
        self._next: SecurityFilter | None = None

    def set_next(self, nxt: "SecurityFilter") -> "SecurityFilter":
        """Link ``nxt`` after this filter and return it, so links can chain."""
        self._next = nxt
        return nxt

    @property
    def next_filter(self) -> "SecurityFilter | None":
        return self._next

    def handle(self, call: ToolCall, session: Session) -> FilterResult:
        """Run this filter, then pass down the chain unless it blocked."""
        result = self.check(call, session)
        signals = list(result.signals)

        if result.verdict is Verdict.BLOCK or self._next is None:
            return FilterResult(verdict=result.verdict, signals=signals)

        downstream = self._next.handle(call, session)
        signals.extend(downstream.signals)
        verdict = _worst(result.verdict, downstream.verdict)
        return FilterResult(verdict=verdict, signals=signals)

    @abstractmethod
    def check(self, call: ToolCall, session: Session) -> FilterResult:
        """Inspect the call and return this filter's own verdict and signals."""

    # -- helpers for subclasses ------------------------------------------
    def signal(
        self,
        category: str,
        score: int,
        severity: Severity,
        message: str,
        evidence: str = "",
        verdict: Verdict = Verdict.FLAG,
    ) -> RiskSignal:
        return RiskSignal(
            filter_name=self.name,
            category=category,
            score=max(0, min(100, score)),
            severity=severity,
            message=message,
            evidence=evidence,
            verdict=verdict,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r}>"


def severity_for(score: int) -> Severity:
    """Map a 0-100 risk score onto the shared severity ladder."""
    if score >= 85:
        return Severity.CRITICAL
    if score >= 65:
        return Severity.HIGH
    if score >= 40:
        return Severity.MEDIUM
    if score >= 20:
        return Severity.LOW
    return Severity.INFO


_ORDER = {Verdict.PASS: 0, Verdict.FLAG: 1, Verdict.BLOCK: 2}


def _worst(a: Verdict, b: Verdict) -> Verdict:
    return a if _ORDER[a] >= _ORDER[b] else b


def build_chain(filters: list[SecurityFilter]) -> SecurityFilter:
    """Link filters in order and return the head of the chain."""
    if not filters:
        raise ValueError("a security chain needs at least one filter")
    head = filters[0]
    current = head
    for nxt in filters[1:]:
        current = current.set_next(nxt)
    return head
