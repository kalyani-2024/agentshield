"""Assembles the detector the pipeline runs, from a mode name.

=========  ==========================================================
mode       what answers a request
=========  ==========================================================
hybrid     ML model; grey-zone cases escalate to the LLM judge
ml         ML model only
llm        LLM judge on every request (ML if the judge is down)
rules      the hand-written signatures only
=========  ==========================================================

In every mode except ``rules`` the signatures sit at the end of the chain as
the fallback, never as the first opinion.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from agentshield.detection.cascade import DetectorInfo, FallbackStrategy, HybridStrategy
from agentshield.detection.llm_judge import (
    PROVIDERS,
    JudgeCache,
    LLMJudgeStrategy,
    build_llm_client,
)
from agentshield.detection.ml import (
    INJECTION_MODELS,
    MODEL_DIR,
    URL_MODEL,
    LinearTextModel,
    MLStrategy,
)
from agentshield.detection.strategies import EnsembleStrategy
from agentshield.detection.strategy import Channel, DetectionStrategy, DetectorUnavailable

MODES = ("hybrid", "ml", "llm", "rules")


@lru_cache(maxsize=8)
def _cached_model(path: str) -> LinearTextModel | None:
    return LinearTextModel.load(path) if Path(path).exists() else None


def injection_models(
    model_dir: str | Path | None = None,
) -> dict[Channel, LinearTextModel] | None:
    """The per-channel injection models, or ``None`` unless all are present."""
    folder = Path(model_dir or MODEL_DIR)
    models = {
        channel: _cached_model(str(folder / filename))
        for channel, filename in INJECTION_MODELS.items()
    }
    if any(model is None for model in models.values()):
        return None
    return models


def url_model(model_dir: str | Path | None = None) -> LinearTextModel | None:
    return _cached_model(str(Path(model_dir or MODEL_DIR) / URL_MODEL))


def resolve_llm(provider: str = "", api_key: str = "") -> tuple[str, str]:
    """Pick the provider and key: explicit settings first, then the environment."""
    if provider:
        spec = PROVIDERS.get(provider.lower())
        if api_key or spec is None:
            return provider.lower(), api_key
        for env in spec.key_env:
            if os.environ.get(env, "").strip():
                return provider.lower(), os.environ[env].strip()
        return provider.lower(), ""
    for name, spec in PROVIDERS.items():
        for env in spec.key_env:
            if os.environ.get(env, "").strip():
                return name, api_key or os.environ[env].strip()
    return "", ""


def build_judge(
    provider: str = "",
    api_key: str = "",
    model: str = "",
    base_url: str = "",
    timeout: float = 15.0,
    cache: JudgeCache | None = None,
    retries: int = 2,
) -> LLMJudgeStrategy | None:
    """The LLM judge, or ``None`` when no provider key is available."""
    name, key = resolve_llm(provider, api_key)
    if not name or not key:
        return None
    try:
        client = build_llm_client(name, key, model, base_url, timeout, retries)
    except DetectorUnavailable:
        return None
    return LLMJudgeStrategy(client, cache)


def build_detector(
    mode: str = "hybrid",
    model_dir: str | Path | None = None,
    judge: DetectionStrategy | None = None,
    low: float = 0.15,
    high: float = 0.85,
) -> tuple[DetectionStrategy, DetectorInfo]:
    """Build the detection strategy for ``mode`` and report what was assembled."""
    mode = mode.lower()
    if mode not in MODES:
        raise ValueError(f"unknown detector mode {mode!r}; available: {list(MODES)}")

    rules = EnsembleStrategy()
    models = injection_models(model_dir)
    ml = MLStrategy(models) if models is not None else None
    notes: list[str] = []
    judge_label = getattr(getattr(judge, "client", None), "label", None) if judge else None

    if mode == "rules":
        return rules, DetectorInfo("rules", "rules", False, None, "none", notes)

    if ml is None:
        notes.append("ML model files not found; run scripts/train_models.py")
    if judge is None and mode in ("hybrid", "llm"):
        notes.append("no LLM provider key configured; the judge is disabled")

    if mode == "llm" and judge is not None:
        # judge first; if it is down, the model answers before the rules do
        backup = FallbackStrategy(ml, rules) if ml else rules
        return (FallbackStrategy(judge, backup),
                DetectorInfo(mode, "llm_judge", ml is not None, judge_label, "ml, then rules"
                             if ml else "rules", notes))

    if ml is None:
        if judge is not None:
            return (FallbackStrategy(judge, rules),
                    DetectorInfo(mode, "llm_judge", False, judge_label, "rules", notes))
        notes.append("running on rules alone: no model and no judge available")
        return rules, DetectorInfo(mode, "rules", False, None, "none", notes)

    if mode == "hybrid" and judge is not None:
        primary: DetectionStrategy = HybridStrategy(ml, judge, low, high)
    else:
        primary = ml
    return (FallbackStrategy(primary, rules),
            DetectorInfo(mode, primary.name, True, judge_label, "rules", notes))


def default_detector() -> DetectionStrategy:
    """What the pipeline uses when nothing is configured: the model, offline.

    No LLM here on purpose - a library default must not make network calls.
    """
    return build_detector("ml")[0]
