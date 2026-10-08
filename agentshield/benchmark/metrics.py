"""Confusion matrix and the classification metrics the design doc asks for."""

from __future__ import annotations

from dataclasses import dataclass

#: "positive" means the detector flagged the sample as malicious.


@dataclass
class ConfusionMatrix:
    tp: int = 0   # malicious, flagged
    fp: int = 0   # benign, flagged     (false alarm)
    tn: int = 0   # benign, passed
    fn: int = 0   # malicious, passed   (attack got through)

    def add(self, actual_malicious: bool, flagged: bool) -> None:
        if actual_malicious and flagged:
            self.tp += 1
        elif actual_malicious and not flagged:
            self.fn += 1
        elif not actual_malicious and flagged:
            self.fp += 1
        else:
            self.tn += 1

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        """Also the detection rate: of all attacks, how many were caught."""
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def false_positive_rate(self) -> float:
        """Of all benign calls, how many were wrongly flagged."""
        denom = self.fp + self.tn
        return self.fp / denom if denom else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0

    @property
    def attack_success_rate(self) -> float:
        """Of all attacks, how many slipped through (the figure to minimise)."""
        denom = self.tp + self.fn
        return self.fn / denom if denom else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "fpr": round(self.false_positive_rate, 4),
            "accuracy": round(self.accuracy, 4),
            "attack_success_rate": round(self.attack_success_rate, 4),
        }
