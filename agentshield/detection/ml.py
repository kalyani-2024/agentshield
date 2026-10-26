"""The machine-learning detector: a linear model over text n-grams.

Training needs scikit-learn (``pip install -e ".[train]"``) and happens once,
offline, in ``scripts/train_models.py``.  Inference is the few lines in
:meth:`LinearTextModel.predict_proba` - plain Python over a JSON file - so the
deployed firewall carries no ML dependency and loads in milliseconds.

Three models ship in ``agentshield/models``:

``injection_user_lr.json``       injection in what the user typed (and in arguments)
``injection_untrusted_lr.json``  injection in tool output, documents and web pages
``url_lr.json``                  malicious-URL lexical model

There are two injection models because the channel changes the question.  On
the user channel an attack is an attempt to override the system's rules; on the
untrusted channel *any* instruction addressed to the agent is the attack.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Protocol

from agentshield.detection.strategy import (
    Channel,
    DetectionHit,
    DetectionResult,
    DetectionStrategy,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
INJECTION_MODELS: dict[Channel, str] = {
    Channel.USER: "injection_user_lr.json",
    Channel.UNTRUSTED: "injection_untrusted_lr.json",
}
URL_MODEL = "url_lr.json"

#: longest input scored in full; anything beyond is cut to keep latency bounded
MAX_CHARS = 20_000

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d{3,}")
_WORD_RE = re.compile(r"[a-z0-9_]+")
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase and mask the parts that identify one attacker, not an attack.

    A model that memorises ``amy.watson@gmail.com`` has learned a dataset, not
    a behaviour; masking keeps "email them to <an address>" and drops the rest.
    """
    text = text[:MAX_CHARS].lower()
    text = _URL_RE.sub(" urltoken ", text)
    text = _EMAIL_RE.sub(" emailtoken ", text)
    return _NUMBER_RE.sub(" numtoken ", text)


def text_features(text: str) -> list[str]:
    """Word 1-2 grams plus within-word character 3-5 grams."""
    words = _WORD_RE.findall(normalize_text(text))
    features = [f"w:{w}" for w in words]
    features.extend(f"w:{a} {b}" for a, b in zip(words, words[1:]))
    for word in words:
        padded = f" {word} "
        for n in (3, 4, 5):
            if len(padded) < n:
                break
            features.extend(f"c:{padded[i:i + n]}" for i in range(len(padded) - n + 1))
    return features


def url_features(url: str) -> list[str]:
    """Character 2-5 grams of the URL, with whitespace removed."""
    compact = _SPACE_RE.sub("", url[:2000].lower())
    features: list[str] = []
    for n in (2, 3, 4, 5):
        features.extend(compact[i:i + n] for i in range(len(compact) - n + 1))
    return features


ANALYZERS: dict[str, Callable[[str], list[str]]] = {
    "text": text_features,
    "url": url_features,
}


class ProbabilisticModel(Protocol):
    """Anything that can score a text; lets a different model be dropped in."""

    def predict_proba(self, text: str) -> float: ...


class LinearTextModel:
    """TF-IDF + logistic regression, with dependency-free inference."""

    def __init__(
        self,
        analyzer: str,
        vocabulary: dict[str, tuple[float, float]],
        bias: float,
        meta: dict | None = None,
    ) -> None:
        if analyzer not in ANALYZERS:
            raise ValueError(f"unknown analyzer {analyzer!r}; available: {sorted(ANALYZERS)}")
        self.analyzer = analyzer
        self._features = ANALYZERS[analyzer]
        #: feature -> (idf, coefficient)
        self.vocabulary = vocabulary
        self.bias = bias
        self.meta = meta or {}

    # -- inference -------------------------------------------------------
    def _contributions(self, text: str) -> tuple[dict[str, float], float]:
        """Per-feature share of the decision value, plus the value itself."""
        counts = Counter(f for f in self._features(text) if f in self.vocabulary)
        if not counts:
            return {}, self.bias
        weighted = {
            feature: (1.0 + math.log(count)) * self.vocabulary[feature][0]
            for feature, count in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
        shares = {
            feature: value / norm * self.vocabulary[feature][1]
            for feature, value in weighted.items()
        }
        return shares, self.bias + sum(shares.values())

    def predict_proba(self, text: str) -> float:
        """P(malicious | text)."""
        _, decision = self._contributions(text)
        if decision >= 0:
            return 1.0 / (1.0 + math.exp(-decision))
        exp = math.exp(decision)
        return exp / (1.0 + exp)

    def explain(self, text: str, top: int = 3) -> list[tuple[str, float]]:
        """The features that pushed hardest towards "malicious"."""
        shares, _ = self._contributions(text)
        # whole words read better in an audit log than character fragments
        words = {f: s for f, s in shares.items() if f.startswith("w:")}
        ranked = sorted((words or shares).items(), key=lambda item: item[1], reverse=True)
        return [(feature, share) for feature, share in ranked[:top] if share > 0]

    # -- persistence -----------------------------------------------------
    def save(self, path: str | Path) -> None:
        payload = {
            "format": "agentshield-linear-v1",
            "analyzer": self.analyzer,
            "bias": self.bias,
            "meta": self.meta,
            "features": list(self.vocabulary),
            "idf": [round(idf, 5) for idf, _ in self.vocabulary.values()],
            "coef": [round(coef, 5) for _, coef in self.vocabulary.values()],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "LinearTextModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "agentshield-linear-v1":
            raise ValueError(f"{path}: not an AgentShield linear model")
        vocabulary = {
            feature: (idf, coef)
            for feature, idf, coef in zip(payload["features"], payload["idf"], payload["coef"])
        }
        return cls(payload["analyzer"], vocabulary, payload["bias"], payload.get("meta"))

    # -- training (offline; needs scikit-learn) --------------------------
    @classmethod
    def fit(
        cls,
        texts: list[str],
        labels: list[bool],
        analyzer: str = "text",
        c: float = 10.0,
        min_df: int = 2,
        max_features: int | None = 120_000,
        meta: dict | None = None,
    ) -> "LinearTextModel":
        """Fit on ``texts`` where ``labels[i]`` is True for malicious."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
        except ImportError as error:  # pragma: no cover - depends on the install
            raise RuntimeError(
                "training needs scikit-learn: pip install -e \".[train]\""
            ) from error

        vectorizer = TfidfVectorizer(
            analyzer=ANALYZERS[analyzer],
            sublinear_tf=True,
            min_df=min_df,
            max_features=max_features,
        )
        matrix = vectorizer.fit_transform(texts)
        classifier = LogisticRegression(C=c, max_iter=5000, class_weight="balanced")
        classifier.fit(matrix, labels)

        coefficients = classifier.coef_[0]
        vocabulary = {
            feature: (float(vectorizer.idf_[index]), float(coefficients[index]))
            for feature, index in vectorizer.vocabulary_.items()
        }
        info = {"c": c, "min_df": min_df, "features": len(vocabulary),
                "train_rows": len(texts), **(meta or {})}
        return cls(analyzer, vocabulary, float(classifier.intercept_[0]), info)


def load_model(name: str, model_dir: str | Path | None = None) -> LinearTextModel | None:
    """Load a packaged model, or ``None`` when it has not been trained yet."""
    path = Path(model_dir) / name if model_dir else MODEL_DIR / name
    if not path.exists():
        return None
    return LinearTextModel.load(path)


class MLStrategy(DetectionStrategy):
    """Wraps trained models as a detection strategy, one model per channel.

    Arguments are text the agent wrote on the user's behalf, so they are scored
    by the user-channel model.  A single model may be passed instead of a
    mapping; it is then used for every channel.
    """

    name = "ml"
    threshold = 0.5

    def __init__(
        self,
        models: ProbabilisticModel | dict[Channel, ProbabilisticModel],
        name: str | None = None,
    ) -> None:
        if not isinstance(models, dict):
            models = {Channel.USER: models, Channel.UNTRUSTED: models}
        if Channel.USER not in models or Channel.UNTRUSTED not in models:
            raise ValueError("MLStrategy needs a model for the USER and UNTRUSTED channels")
        self.models = models
        if name:
            self.name = name

    def model_for(self, channel: Channel) -> ProbabilisticModel:
        return self.models[Channel.UNTRUSTED if channel is Channel.UNTRUSTED else Channel.USER]

    def detect(self, text: str, channel: Channel = Channel.UNTRUSTED) -> DetectionResult:
        if not text.strip():
            return self._result()
        model = self.model_for(channel)
        probability = model.predict_proba(text)
        hits: list[DetectionHit] = []
        if probability >= self.threshold:
            explain = getattr(model, "explain", None)
            reasons = explain(text) if explain else []
            described = ", ".join(repr(feature.split(":", 1)[-1]) for feature, _ in reasons)
            hits.append(
                DetectionHit(
                    pattern_id="ML",
                    description=(
                        f"model probability {probability:.2f}"
                        + (f" (strongest features: {described})" if described else "")
                    ),
                    weight=int(probability * 100),
                    excerpt=text[:120],
                )
            )
        return self._result(confidence=probability, hits=hits)
