"""Run detectors and the full engine over the benchmark and collect metrics."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from agentshield.benchmark.dataset import Sample
from agentshield.benchmark.metrics import ConfusionMatrix
from agentshield.core.models import Decision
from agentshield.detection.llm_judge import QuotaExhausted
from agentshield.detection.ml import ProbabilisticModel
from agentshield.detection.strategy import DetectionStrategy, DetectorUnavailable
from agentshield.engine.pipeline import SecurityEngine
from agentshield.engine.session import SessionStore
from agentshield.filters.destination_trust import TrustPolicy


@dataclass
class Report:
    """Metrics for one detector or one engine configuration."""

    name: str
    matrix: ConfusionMatrix = field(default_factory=ConfusionMatrix)
    per_family: dict[str, ConfusionMatrix] = field(
        default_factory=lambda: defaultdict(ConfusionMatrix)
    )
    latencies_ms: list[float] = field(default_factory=list)
    decisions: Counter = field(default_factory=Counter)
    extra: dict = field(default_factory=dict)

    def add(self, sample: Sample, flagged: bool) -> None:
        self.matrix.add(sample.is_malicious, flagged)
        self.per_family[sample.family].add(sample.is_malicious, flagged)

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]

    def family_rates(self) -> dict[str, dict]:
        """Per family: how many were flagged, out of how many."""
        out = {}
        for family, matrix in sorted(self.per_family.items()):
            flagged = matrix.tp + matrix.fp
            out[family] = {"flagged": flagged, "total": matrix.total,
                           "rate": round(flagged / matrix.total, 4) if matrix.total else 0.0}
        return out

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            **self.matrix.as_dict(),
            "avg_latency_ms": round(self.avg_latency_ms, 4),
            "p95_latency_ms": round(self.p95_latency_ms, 4),
            "families": self.family_rates(),
            "decisions": dict(self.decisions),
            **self.extra,
        }


# ======================================================================
# Detector-level comparison: one piece of content in, one verdict out
# ======================================================================
def evaluate_detector(
    strategy: DetectionStrategy,
    samples: list[Sample],
    name: str | None = None,
    progress: bool = False,
) -> Report:
    """Classify each sample's text and score it against the true label.

    If the detector cannot answer one sample (a malformed reply, say), that
    sample counts as *not detected* and is tallied under ``unanswered``: a
    detector that gives no verdict stops nothing.  An exhausted quota aborts
    the run with :class:`QuotaExhausted`, since the remaining samples would
    all be unanswered for a reason unrelated to the detector.
    """
    report = Report(name or strategy.name)
    unanswered = 0
    for index, sample in enumerate(samples, 1):
        start = time.perf_counter()
        try:
            flagged = strategy.detect(sample.text, sample.detector_channel).flagged
        except QuotaExhausted:
            raise
        except DetectorUnavailable:
            unanswered += 1
            flagged = False
        report.latencies_ms.append((time.perf_counter() - start) * 1000)
        report.add(sample, flagged)
        if progress and index % 50 == 0:
            print(f"    {report.name}: {index}/{len(samples)}", flush=True)
    if unanswered:
        report.extra["unanswered"] = unanswered
    return report


def evaluate_url_model(
    model: ProbabilisticModel, samples: list[Sample], threshold: float = 0.5
) -> Report:
    report = Report("url_model")
    for sample in samples:
        start = time.perf_counter()
        probability = model.predict_proba(sample.text)
        report.latencies_ms.append((time.perf_counter() - start) * 1000)
        report.add(sample, probability >= threshold)
    return report


# ======================================================================
# End-to-end: the whole pipeline, with one detector plugged in
# ======================================================================
def evaluate_engine(
    samples: list[Sample],
    detector: DetectionStrategy,
    url_model: ProbabilisticModel | None,
    name: str,
    trust: TrustPolicy | None = None,
    progress: bool = False,
) -> Report:
    """Replay every interaction through the engine.

    An interaction counts as *stopped* when any of its calls is not ALLOWed -
    blocked outright or held for human approval.  For an attack that is a
    detection; for a benign interaction it is a false positive.  Replay of an
    interaction ends at the first stopped call, as it would in production.

    The trust policy carries no allow or deny list by default: the benchmark
    measures detection, not a hand-tuned list of the test set's hosts.
    """
    engine = SecurityEngine(
        sessions=SessionStore(),
        trust_policy=trust or TrustPolicy(),
        detector=detector,
        url_model=url_model,
    )
    report = Report(name)
    for index, sample in enumerate(samples, 1):
        if progress and index % 100 == 0:
            print(f"    {name} engine: {index}/{len(samples)}", flush=True)
        stopped = False
        for call in sample.build_calls():
            start = time.perf_counter()
            assessment = engine.evaluate(call)
            report.latencies_ms.append((time.perf_counter() - start) * 1000)
            if assessment.decision is not Decision.ALLOW:
                stopped = True
                report.decisions[
                    f"{'attack' if sample.is_malicious else 'benign'}_{assessment.decision.value}"
                ] += 1
                break
        report.add(sample, stopped)
    return report
