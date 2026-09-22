"""Concrete detection strategies available in stage 1."""

from __future__ import annotations

import math
import re

from agentshield.detection.patterns import (
    INJECTION_SIGNATURES,
    SUSPICIOUS_TERMS,
    Signature,
)
from agentshield.detection.strategy import (
    DetectionHit,
    DetectionResult,
    DetectionStrategy,
)

_EXCERPT_PAD = 40


def _excerpt(text: str, start: int, end: int) -> str:
    lo = max(0, start - _EXCERPT_PAD)
    hi = min(len(text), end + _EXCERPT_PAD)
    prefix = "..." if lo > 0 else ""
    suffix = "..." if hi < len(text) else ""
    return f"{prefix}{text[lo:hi].strip()}{suffix}"


def _combine(weights: list[int]) -> float:
    """Probabilistic OR over independent indicators.

    Several weak indicators add up, but the confidence never exceeds 1.0 and a
    single strong indicator is already close to it.
    """
    product = 1.0
    for weight in weights:
        product *= 1.0 - min(0.99, weight / 100.0)
    return 1.0 - product


class RegexStrategy(DetectionStrategy):
    """Signature matching against the curated injection pattern catalogue."""

    name = "regex"

    def __init__(self, signatures: tuple[Signature, ...] = INJECTION_SIGNATURES) -> None:
        self._signatures = signatures

    def detect(self, text: str) -> DetectionResult:
        if not text.strip():
            return DetectionResult(strategy=self.name)

        hits: list[DetectionHit] = []
        for sig in self._signatures:
            match = sig.regex.search(text)
            if match:
                hits.append(
                    DetectionHit(
                        pattern_id=sig.id,
                        description=sig.description,
                        weight=sig.weight,
                        excerpt=_excerpt(text, match.start(), match.end()),
                    )
                )
        return DetectionResult(
            strategy=self.name,
            confidence=_combine([h.weight for h in hits]),
            hits=hits,
        )


class HeuristicStrategy(DetectionStrategy):
    """Lexical scoring: suspicious-term density plus structural tells.

    It catches phrasings the signature catalogue misses, at the cost of being
    noisier - which is exactly the trade-off the stage 3 benchmark measures.
    """

    name = "heuristic"

    #: term-score total that maps to ~0.63 confidence
    _saturation = 45.0

    def __init__(self, terms: dict[str, int] = SUSPICIOUS_TERMS) -> None:
        self._terms = terms

    def detect(self, text: str) -> DetectionResult:
        stripped = text.strip()
        if not stripped:
            return DetectionResult(strategy=self.name)

        lowered = stripped.lower()
        hits: list[DetectionHit] = []
        total = 0

        for term, weight in self._terms.items():
            pattern = re.compile(rf"\b{re.escape(term)}\b")
            match = pattern.search(lowered)
            if match:
                total += weight
                hits.append(
                    DetectionHit(
                        pattern_id=f"HEU-{term}",
                        description=f"suspicious term {term!r}",
                        weight=weight,
                        excerpt=_excerpt(stripped, match.start(), match.end()),
                    )
                )

        structural = self._structural_score(stripped)
        for pattern_id, description, weight, excerpt in structural:
            total += weight
            hits.append(DetectionHit(pattern_id, description, weight, excerpt))

        if not hits:
            return DetectionResult(strategy=self.name)

        # Saturating curve: more evidence -> higher confidence, capped below 1.
        confidence = 1.0 - math.exp(-total / self._saturation)
        return DetectionResult(strategy=self.name, confidence=confidence, hits=hits)

    def _structural_score(self, text: str) -> list[tuple[str, str, int, str]]:
        found: list[tuple[str, str, int, str]] = []

        letters = [c for c in text if c.isalpha()]
        if len(letters) >= 40:
            upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if upper_ratio > 0.6:
                found.append(
                    ("HEU-SHOUT", "shouted imperative text (mostly uppercase)", 12, text[:80])
                )

        imperative = re.search(
            r"(?m)^\s*(please\s+)?(send|delete|upload|forward|execute|run|email)\b.*$",
            text,
            re.IGNORECASE,
        )
        if imperative:
            found.append(
                (
                    "HEU-IMP",
                    "imperative action sentence inside content",
                    15,
                    imperative.group(0)[:120],
                )
            )

        if re.search(r"[​-‏‪-‮﻿]", text):
            found.append(
                ("HEU-HIDDEN", "zero-width or bidirectional control characters", 30, "")
            )

        base64ish = re.search(r"\b[A-Za-z0-9+/]{60,}={0,2}\b", text)
        if base64ish:
            found.append(
                ("HEU-B64", "long base64-like blob embedded in content", 18,
                 base64ish.group(0)[:60] + "...")
            )

        return found


class EnsembleStrategy(DetectionStrategy):
    """Weighted blend of several strategies.

    The blend is a weighted mean of the members' confidences, then lifted toward
    the strongest member so a single high-confidence detector is never diluted
    into silence by quiet peers.
    """

    name = "ensemble"

    def __init__(
        self,
        strategies: list[DetectionStrategy] | None = None,
        weights: list[float] | None = None,
    ) -> None:
        self._strategies = strategies or [RegexStrategy(), HeuristicStrategy()]
        if weights is None:
            weights = [1.0] * len(self._strategies)
        if len(weights) != len(self._strategies):
            raise ValueError("weights must match the number of strategies")
        self._weights = weights

    def detect(self, text: str) -> DetectionResult:
        results = [s.detect(text) for s in self._strategies]
        total_weight = sum(self._weights) or 1.0
        weighted = sum(r.confidence * w for r, w in zip(results, self._weights))
        mean = weighted / total_weight
        strongest = max((r.confidence for r in results), default=0.0)
        confidence = max(mean, 0.85 * strongest)

        hits: list[DetectionHit] = []
        for result in results:
            hits.extend(result.hits)

        return DetectionResult(strategy=self.name, confidence=confidence, hits=hits)


#: Registry so strategies can be selected by name from config or the CLI.
STRATEGY_REGISTRY: dict[str, type[DetectionStrategy]] = {
    RegexStrategy.name: RegexStrategy,
    HeuristicStrategy.name: HeuristicStrategy,
    EnsembleStrategy.name: EnsembleStrategy,
}


def build_strategy(name: str) -> DetectionStrategy:
    """Instantiate a strategy by its registry name."""
    try:
        return STRATEGY_REGISTRY[name]()
    except KeyError:
        raise ValueError(
            f"unknown detection strategy {name!r}; "
            f"available: {sorted(STRATEGY_REGISTRY)}"
        ) from None
