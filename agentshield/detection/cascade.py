"""Composite strategies: the hybrid cascade and the fallback chain.

The deployed detector is model-first::

    FallbackStrategy( HybridStrategy(ML, LLM judge) , rules )

The ML model answers every request.  Only the ones it is unsure about go to the
LLM judge.  The hand-written rules are consulted only when a model cannot give
a verdict at all - missing model file, no API key, provider outage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from agentshield.detection.strategy import (
    Channel,
    DetectionResult,
    DetectionStrategy,
    DetectorUnavailable,
)

logger = logging.getLogger("agentshield.detection")


class HybridStrategy(DetectionStrategy):
    """ML first; the LLM judge settles what the model is unsure about.

    A probability at or below ``low`` is accepted as benign and at or above
    ``high`` as malicious, without spending an LLM call.  Anything between is
    the model's grey zone and is escalated to the judge, whose verdict stands.
    If the judge is unavailable the model's own answer is used.
    """

    name = "hybrid"
    threshold = 0.5

    def __init__(
        self,
        ml: DetectionStrategy,
        judge: DetectionStrategy | None = None,
        low: float = 0.15,
        high: float = 0.85,
    ) -> None:
        if not 0.0 <= low < high <= 1.0:
            raise ValueError("need 0 <= low < high <= 1")
        self.ml = ml
        self.judge = judge
        self.low = low
        self.high = high
        self.evaluated = 0
        self.escalated = 0
        self.judge_failures = 0

    def detect(self, text: str, channel: Channel = Channel.UNTRUSTED) -> DetectionResult:
        first = self.ml.detect(text, channel)
        self.evaluated += 1
        if self.judge is None or not self.low < first.confidence < self.high:
            return self._result(first.confidence, first.hits)

        self.escalated += 1
        try:
            second = self.judge.detect(text, channel)
        except DetectorUnavailable as error:
            self.judge_failures += 1
            logger.warning("LLM judge unavailable, keeping the model's verdict: %s", error)
            return self._result(first.confidence, first.hits)
        return self._result(second.confidence, first.hits + second.hits)

    @property
    def escalation_rate(self) -> float:
        return self.escalated / self.evaluated if self.evaluated else 0.0


class FallbackStrategy(DetectionStrategy):
    """Use ``primary``; fall back to ``fallback`` only when it cannot answer.

    This is the piece that makes the rules a safety net instead of the decision
    maker: they run on a request only if the primary detector raised.
    """

    def __init__(self, primary: DetectionStrategy, fallback: DetectionStrategy) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = primary.name
        self.threshold = primary.threshold
        self.fallbacks = 0
        self.last_error: str | None = None

    def detect(self, text: str, channel: Channel = Channel.UNTRUSTED) -> DetectionResult:
        try:
            return self.primary.detect(text, channel)
        except Exception as error:  # any detector fault degrades, never opens
            self.fallbacks += 1
            self.last_error = f"{type(error).__name__}: {error}"
            # warn once, then stay quiet: an outage would otherwise flood the log
            log = logger.warning if self.fallbacks == 1 else logger.debug
            log("detector %r failed, falling back to %r: %s",
                self.primary.name, self.fallback.name, self.last_error)
            return self.fallback.detect(text, channel)


@dataclass
class DetectorInfo:
    """What the factory actually assembled, for /v1/backend and the logs."""

    mode: str
    primary: str
    ml_model: bool
    llm_judge: str | None
    fallback: str
    notes: list[str]

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "primary": self.primary,
            "ml_model": self.ml_model,
            "llm_judge": self.llm_judge,
            "fallback": self.fallback,
            "notes": self.notes,
        }
