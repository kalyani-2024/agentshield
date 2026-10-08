"""A dependency-free ML detector: multinomial Naive Bayes over token n-grams.

This is the "ML detector" arm of the Stage 3 comparison.  It is trained from
scratch on the benchmark's training split, so the experiment is fully
self-contained - no scikit-learn, no model download, no network.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase unigrams plus bigrams, so word order carries some weight."""
    words = _TOKEN_RE.findall(text.lower())
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


class NaiveBayesClassifier:
    """Multinomial NB with Laplace smoothing. Outputs P(malicious | text)."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._log_prior: dict[str, float] = {}
        self._log_likelihood: dict[str, dict[str, float]] = {}
        self._vocab: set[str] = set()
        self._class_totals: dict[str, int] = {}
        self.trained = False

    def fit(self, texts: list[str], labels: list[str]) -> "NaiveBayesClassifier":
        if len(texts) != len(labels):
            raise ValueError("texts and labels must be the same length")

        class_docs: dict[str, int] = defaultdict(int)
        token_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for text, label in zip(texts, labels):
            class_docs[label] += 1
            for token in tokenize(text):
                token_counts[label][token] += 1
                self._vocab.add(token)

        total_docs = len(texts)
        for label, docs in class_docs.items():
            self._log_prior[label] = math.log(docs / total_docs)
            self._class_totals[label] = sum(token_counts[label].values())

        vocab_size = len(self._vocab) or 1
        for label in class_docs:
            denom = self._class_totals[label] + self.alpha * vocab_size
            likelihood: dict[str, float] = {}
            counts = token_counts[label]
            for token in self._vocab:
                likelihood[token] = math.log((counts[token] + self.alpha) / denom)
            self._log_likelihood[label] = likelihood
            # cache the unseen-token fallback for this class
            likelihood["<unk>"] = math.log(self.alpha / denom)

        self.trained = True
        return self

    def _log_score(self, label: str, tokens: list[str]) -> float:
        score = self._log_prior[label]
        likelihood = self._log_likelihood[label]
        unk = likelihood["<unk>"]
        for token in tokens:
            if token in self._vocab:
                score += likelihood[token]
            else:
                score += unk
        return score

    def predict_proba(self, text: str) -> float:
        """Probability that ``text`` is malicious, via a softmax over the two classes."""
        if not self.trained:
            raise RuntimeError("classifier is not trained")
        tokens = tokenize(text)
        log_mal = self._log_score("malicious", tokens)
        log_ben = self._log_score("benign", tokens)
        # numerically stable softmax of two log-scores
        hi = max(log_mal, log_ben)
        mal = math.exp(log_mal - hi)
        ben = math.exp(log_ben - hi)
        return mal / (mal + ben)
