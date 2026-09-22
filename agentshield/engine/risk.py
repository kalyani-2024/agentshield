"""Risk aggregation.

Turning a list of signals into one number is the part of the system a reviewer
will poke at hardest, so the rule is deliberately simple and explainable:

* the strongest single signal sets the floor,
* every additional signal adds a damped contribution (evidence corroborates,
  it does not simply sum),
* co-occurring categories that describe a known attack shape add a bonus,
* the session's current state adds its surcharge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentshield.core.models import RiskSignal

#: Category pairs whose co-occurrence is worse than either alone.
CORRELATION_BONUS: dict[frozenset[str], int] = {
    frozenset({"indirect_prompt_injection", "sensitive_data_egress"}): 20,
    frozenset({"indirect_prompt_injection", "untrusted_destination"}): 18,
    frozenset({"prompt_injection", "sensitive_data_egress"}): 15,
    frozenset({"sensitive_data_egress", "untrusted_destination"}): 18,
    frozenset({"suspicious_sequence", "sensitive_data_egress"}): 15,
    frozenset({"suspicious_sequence", "untrusted_destination"}): 12,
    frozenset({"sensitive_resource_access", "untrusted_destination"}): 15,
    frozenset({"excessive_agency", "indirect_prompt_injection"}): 15,
}

#: Per-category weights, so a filter can be tuned without touching its code.
CATEGORY_WEIGHTS: dict[str, float] = {
    "unauthorized_tool": 1.0,
    "privilege_escalation": 1.0,
    "indirect_prompt_injection": 1.0,
    "prompt_injection": 0.9,
    "sensitive_data_egress": 1.0,
    "sensitive_data": 0.7,
    "sensitive_resource_access": 0.8,
    "untrusted_destination": 0.9,
    "suspicious_sequence": 0.9,
    "repeat_offender": 0.8,
    "tool_call_burst": 0.5,
    "excessive_agency": 0.8,
}

DEFAULT_WEIGHT = 0.8
#: Each extra signal beyond the strongest contributes this fraction of itself.
CORROBORATION_FACTOR = 0.35


@dataclass
class RiskBreakdown:
    """An explainable account of how the final score was reached."""

    base: int = 0
    corroboration: int = 0
    correlation: int = 0
    state_surcharge: int = 0
    total: int = 0
    reasons: list[str] = field(default_factory=list)


class RiskAggregator:
    """Combines signals into a single 0-100 score."""

    def __init__(
        self,
        weights: dict[str, float] = CATEGORY_WEIGHTS,
        correlations: dict[frozenset[str], int] = CORRELATION_BONUS,
        corroboration_factor: float = CORROBORATION_FACTOR,
    ) -> None:
        self.weights = weights
        self.correlations = correlations
        self.corroboration_factor = corroboration_factor

    def aggregate(
        self, signals: list[RiskSignal], state_surcharge: int = 0
    ) -> RiskBreakdown:
        breakdown = RiskBreakdown(state_surcharge=state_surcharge)
        if not signals:
            breakdown.total = min(100, state_surcharge)
            return breakdown

        weighted = sorted(
            (
                (signal, signal.score * self.weights.get(signal.category, DEFAULT_WEIGHT))
                for signal in signals
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )

        strongest, base = weighted[0]
        breakdown.base = int(round(base))
        breakdown.reasons.append(f"{strongest.category} ({strongest.message})")

        corroboration = sum(score for _, score in weighted[1:]) * self.corroboration_factor
        breakdown.corroboration = int(round(corroboration))

        categories = {signal.category for signal in signals}
        for pair, bonus in self.correlations.items():
            if pair <= categories:
                breakdown.correlation += bonus
                breakdown.reasons.append(
                    "correlated: " + " + ".join(sorted(pair)) + f" (+{bonus})"
                )

        total = (
            breakdown.base
            + breakdown.corroboration
            + breakdown.correlation
            + state_surcharge
        )
        breakdown.total = max(0, min(100, int(round(total))))
        return breakdown
