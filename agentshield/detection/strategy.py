"""Strategy pattern for prompt-injection detection.

Stage 1 ships the deterministic strategies (regex + heuristic + ensemble).  The
ML classifier and LLM-judge strategies plug into the same interface in later
stages, which is what makes the experimental comparison possible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class DetectionHit:
    """One matched injection indicator."""

    pattern_id: str
    description: str
    weight: int          # 0-100 contribution of this single hit
    excerpt: str = ""


@dataclass
class DetectionResult:
    """Outcome of running a detection strategy over a piece of text."""

    strategy: str
    confidence: float = 0.0          # 0.0 - 1.0
    hits: list[DetectionHit] = field(default_factory=list)

    @property
    def detected(self) -> bool:
        return self.confidence > 0.0

    @property
    def score(self) -> int:
        """Confidence expressed on the shared 0-100 risk scale."""
        return int(round(min(1.0, max(0.0, self.confidence)) * 100))

    def summary(self, limit: int = 3) -> str:
        if not self.hits:
            return "no injection indicators"
        top = sorted(self.hits, key=lambda h: h.weight, reverse=True)[:limit]
        return "; ".join(h.description for h in top)

    def evidence(self) -> str:
        if not self.hits:
            return ""
        best = max(self.hits, key=lambda h: h.weight)
        return best.excerpt


class DetectionStrategy(ABC):
    """Interchangeable prompt-injection detector."""

    name: str = "abstract"

    @abstractmethod
    def detect(self, text: str) -> DetectionResult:
        """Score ``text`` for prompt-injection indicators."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r}>"
