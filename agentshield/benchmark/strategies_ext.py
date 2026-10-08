"""Benchmark-only detection strategies: ML, Hybrid, and an LLM-judge adapter.

All three implement the same :class:`DetectionStrategy` interface as the Stage 1
regex and heuristic detectors, so the comparison harness treats every arm
identically.
"""

from __future__ import annotations

from typing import Callable

from agentshield.benchmark.classifier import NaiveBayesClassifier
from agentshield.detection.strategy import (
    DetectionHit,
    DetectionResult,
    DetectionStrategy,
)


class MLStrategy(DetectionStrategy):
    """Wraps a trained Naive Bayes classifier as a detection strategy."""

    name = "ml"

    def __init__(self, classifier: NaiveBayesClassifier) -> None:
        self.classifier = classifier

    def detect(self, text: str) -> DetectionResult:
        if not text.strip():
            return DetectionResult(strategy=self.name)
        prob = self.classifier.predict_proba(text)
        hits = []
        if prob >= 0.5:
            hits.append(
                DetectionHit(
                    pattern_id="ML",
                    description=f"classifier probability {prob:.2f}",
                    weight=int(prob * 100),
                    excerpt=text[:80],
                )
            )
        return DetectionResult(strategy=self.name, confidence=prob, hits=hits)


class HybridStrategy(DetectionStrategy):
    """Rules OR ML: fast signatures catch the known, the model catches the rest.

    Confidence is the max of the two arms - a detection by either is a detection
    - which is the usual way a rules+ML firewall is wired in practice.
    """

    name = "hybrid"

    def __init__(self, rules: DetectionStrategy, ml: DetectionStrategy) -> None:
        self.rules = rules
        self.ml = ml

    def detect(self, text: str) -> DetectionResult:
        r = self.rules.detect(text)
        m = self.ml.detect(text)
        confidence = max(r.confidence, m.confidence)
        return DetectionResult(
            strategy=self.name,
            confidence=confidence,
            hits=r.hits + m.hits,
        )


class LLMJudgeStrategy(DetectionStrategy):
    """Adapter for an LLM-as-judge detector.

    A real judge calls a model; it needs network and an API key, so it is never
    invoked automatically.  Supply a ``judge`` callable (text -> probability) to
    plug one in - the benchmark will then include it as another arm.
    """

    name = "llm_judge"

    def __init__(self, judge: Callable[[str], float]) -> None:
        self.judge = judge

    def detect(self, text: str) -> DetectionResult:
        prob = float(self.judge(text))
        return DetectionResult(strategy=self.name, confidence=prob)
