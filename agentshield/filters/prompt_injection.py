"""Filter 1: direct and indirect prompt-injection detection."""

from __future__ import annotations

from agentshield.core.models import FilterResult, ToolCall, TrustLevel, Verdict
from agentshield.detection.strategies import EnsembleStrategy
from agentshield.detection.strategy import DetectionStrategy
from agentshield.engine.session import Session
from agentshield.filters.base import SecurityFilter, severity_for


class PromptInjectionFilter(SecurityFilter):
    """Scores the call's arguments and its surrounding context.

    Injection found in *untrusted context* (a document, a web page, an email the
    agent just read) is the indirect case and is treated as more serious than
    the same text appearing in the arguments, because the user never saw it.
    """

    name = "prompt_injection"

    def __init__(
        self,
        strategy: DetectionStrategy | None = None,
        flag_confidence: float = 0.35,
    ) -> None:
        super().__init__()
        self.strategy = strategy or EnsembleStrategy()
        self.flag_confidence = flag_confidence

    def check(self, call: ToolCall, session: Session) -> FilterResult:
        signals = []

        direct = self.strategy.detect(call.argument_text())
        if direct.confidence >= self.flag_confidence:
            signals.append(
                self.signal(
                    category="prompt_injection",
                    score=direct.score,
                    severity=severity_for(direct.score),
                    message=f"injection indicators in tool arguments: {direct.summary()}",
                    evidence=direct.evidence(),
                    verdict=Verdict.FLAG,
                )
            )

        for chunk in call.context:
            if chunk.trust is TrustLevel.TRUSTED:
                continue
            result = self.strategy.detect(chunk.content)
            if result.confidence < self.flag_confidence:
                continue
            # Indirect injection: content the agent ingested is giving orders.
            weight = 1.15 if chunk.trust is TrustLevel.UNTRUSTED else 1.0
            score = min(100, int(result.score * weight))
            signals.append(
                self.signal(
                    category="indirect_prompt_injection"
                    if chunk.trust is TrustLevel.UNTRUSTED
                    else "prompt_injection",
                    score=score,
                    severity=severity_for(score),
                    message=(
                        f"injection indicators in {chunk.trust.value.lower()} content "
                        f"from {chunk.source!r}: {result.summary()}"
                    ),
                    evidence=result.evidence(),
                    verdict=Verdict.FLAG,
                )
            )

        return FilterResult.from_signals(signals)

