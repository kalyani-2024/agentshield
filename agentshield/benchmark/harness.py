"""Run detectors and the full engine over the dataset and collect metrics."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

from agentshield.benchmark.dataset import Sample
from agentshield.benchmark.metrics import ConfusionMatrix
from agentshield.core.models import Decision, ToolCall, ToolCategory
from agentshield.detection.strategy import DetectionStrategy
from agentshield.engine.pipeline import SecurityEngine
from agentshield.engine.session import HistoryEntry, SessionStore
from agentshield.filters.destination_trust import TrustPolicy

# a deployment that knows the usual drop hosts and trusts the corp domain
BENCHMARK_TRUST = TrustPolicy(
    allowed_domains={"example.com", "corp.example.com"},
    denied_domains={
        "attacker-drop.top", "evil.ru", "attacker.xyz", "mailinator.com",
        "pastebin.com", "webhook.site", "exfil.xyz", "requestbin.net", "transfer.sh",
    },
)


@dataclass
class StrategyReport:
    name: str
    matrix: ConfusionMatrix
    avg_latency_ms: float
    samples: int


# ======================================================================
# Detector-level comparison
# ======================================================================
def evaluate_detector(
    strategy: DetectionStrategy, samples: list[Sample], threshold: float = 0.5
) -> StrategyReport:
    """Classify each sample's text and score it against the true label."""
    matrix = ConfusionMatrix()
    total_ns = 0
    for sample in samples:
        start = time.perf_counter_ns()
        result = strategy.detect(sample.text)
        total_ns += time.perf_counter_ns() - start
        flagged = result.confidence >= threshold
        matrix.add(sample.is_malicious, flagged)

    avg_ms = (total_ns / len(samples)) / 1e6 if samples else 0.0
    return StrategyReport(strategy.name, matrix, avg_ms, len(samples))


def compare_detectors(
    strategies: dict[str, DetectionStrategy],
    samples: list[Sample],
    threshold: float = 0.5,
) -> dict[str, StrategyReport]:
    return {
        name: evaluate_detector(strategy, samples, threshold)
        for name, strategy in strategies.items()
    }


# ======================================================================
# End-to-end engine evaluation (the full pipeline, not just a detector)
# ======================================================================
def _seed_precursors(engine: SecurityEngine, call: ToolCall) -> None:
    """Replay a staged attack's earlier steps so the sequence filter can see them."""
    precursors = call.metadata.get("stage_precursors") if call.metadata else None
    if not precursors:
        return
    session = engine.sessions.get(call.session_id, call.principal.id)
    category_map = {
        "read_file": ToolCategory.READ_LOCAL,
        "web_search": ToolCategory.READ_REMOTE,
        "write_file": ToolCategory.WRITE_LOCAL,
    }
    import datetime as _dt

    for tool in precursors:
        session.history.append(
            HistoryEntry(
                tool=tool,
                category=category_map.get(tool, ToolCategory.UNKNOWN),
                destination=None,
                risk_score=10,
                decision="ALLOW",
                at=_dt.datetime.now(_dt.timezone.utc),
            )
        )


def evaluate_engine(
    samples: list[Sample], trust: TrustPolicy = BENCHMARK_TRUST
) -> tuple[StrategyReport, dict[str, ConfusionMatrix]]:
    """Run every sample through the whole engine; a flag = decision != ALLOW."""
    engine = SecurityEngine(sessions=SessionStore(), trust_policy=trust)
    matrix = ConfusionMatrix()
    per_type: dict[str, ConfusionMatrix] = defaultdict(ConfusionMatrix)
    total_ns = 0

    for sample in samples:
        call = sample.build_call()
        _seed_precursors(engine, call)
        start = time.perf_counter_ns()
        assessment = engine.evaluate(call)
        total_ns += time.perf_counter_ns() - start

        flagged = assessment.decision is not Decision.ALLOW
        matrix.add(sample.is_malicious, flagged)
        per_type[sample.attack_type].add(sample.is_malicious, flagged)

    avg_ms = (total_ns / len(samples)) / 1e6 if samples else 0.0
    return StrategyReport("engine", matrix, avg_ms, len(samples)), dict(per_type)
