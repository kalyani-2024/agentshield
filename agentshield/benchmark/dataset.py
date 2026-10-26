"""Loads the benchmark: 500 benign and 500 malicious agent interactions.

The data is not generated here.  ``scripts/fetch_datasets.py`` downloads six
public datasets and ``scripts/build_benchmark.py`` assembles them into
``data/benchmark/agentshield_bench_v1.jsonl``; this module only reads that file.

Each record carries two views of the same interaction:

``text``   what a content detector sees (a prompt, a tool output, a URL)
``steps``  the tool calls themselves, in their framework's wire format, with
           the context the agent had read - what the full engine sees

so one dataset drives both the detector comparison and the end-to-end run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentshield.adapters.frameworks import get_factory
from agentshield.core.models import ContextChunk, Principal, ToolCall, TrustLevel
from agentshield.detection.strategy import Channel

BENIGN = "benign"
MALICIOUS = "malicious"

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
BENCHMARK_PATH = DATA_DIR / "benchmark" / "agentshield_bench_v1.jsonl"
TRAIN_TEXT_PATH = DATA_DIR / "train" / "injection_text.jsonl"

#: families whose ``text`` is a URL rather than prose
URL_FAMILIES = frozenset({"malicious_url", "benign_url"})


@dataclass
class Step:
    """One tool call of an interaction, as the agent framework emitted it."""

    framework: str
    call: Any
    context: list[ContextChunk] = field(default_factory=list)
    user_authorized: bool = False


@dataclass
class Sample:
    """One labelled interaction."""

    id: str
    label: str                       # BENIGN or MALICIOUS
    family: str                      # attack family, or the kind of benign traffic
    source: str                      # the public dataset the record came from
    text: str                        # what a text detector sees
    channel: str                     # "user", "untrusted" or "url": where text came from
    principal: Principal
    steps: list[Step]
    notes: str = ""

    @property
    def is_malicious(self) -> bool:
        return self.label == MALICIOUS

    @property
    def is_url(self) -> bool:
        return self.family in URL_FAMILIES

    @property
    def detector_channel(self) -> Channel:
        return Channel(self.channel)

    def build_calls(self) -> list[ToolCall]:
        """Normalise every step through its framework adapter, in one session."""
        calls = []
        for step in self.steps:
            adapter = get_factory(step.framework).create_adapter()
            calls.append(
                adapter.normalize(
                    step.call,
                    principal=self.principal,
                    session_id=f"bench-{self.id}",
                    context=list(step.context),
                    user_authorized=step.user_authorized,
                )
            )
        return calls


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run scripts/fetch_datasets.py then scripts/build_benchmark.py"
        )
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sample(row: dict) -> Sample:
    principal = Principal(
        id=row["principal"]["id"],
        allowed_tools=frozenset(row["principal"]["allowed_tools"]),
    )
    steps = [
        Step(
            framework=step["framework"],
            call=step["call"],
            context=[
                ContextChunk(c["content"], c["source"], TrustLevel(c["trust"]))
                for c in step["context"]
            ],
            user_authorized=step["user_authorized"],
        )
        for step in row["steps"]
    ]
    return Sample(
        id=row["id"],
        label=row["label"],
        family=row["family"],
        source=row["source"],
        text=row["text"],
        channel=row["channel"],
        principal=principal,
        steps=steps,
        notes=row.get("notes", ""),
    )


def load_benchmark(path: str | Path | None = None) -> list[Sample]:
    """The held-out benchmark, in its fixed order."""
    return [_sample(row) for row in _jsonl(Path(path) if path else BENCHMARK_PATH)]


def load_training_text(
    channel: Channel, path: str | Path | None = None
) -> tuple[list[str], list[str]]:
    """(texts, labels) of one channel's training pool - never overlaps the benchmark."""
    rows = [
        row for row in _jsonl(Path(path) if path else TRAIN_TEXT_PATH)
        if channel.value in row["channels"]
    ]
    return [row["text"] for row in rows], [row["label"] for row in rows]
