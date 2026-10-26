"""Filter 4: where is this action actually going?"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse

from agentshield.core.models import FilterResult, ToolCall, ToolCategory, Verdict
from agentshield.detection.ml import ProbabilisticModel
from agentshield.engine.session import Session
from agentshield.filters.base import SecurityFilter, severity_for

_EMAIL_RE = re.compile(r"[\w.+-]+@([\w-]+\.[\w.-]+)")
_URL_RE = re.compile(r"https?://([^\s/:?#\"'<>]+)[^\s\"'<>]*", re.IGNORECASE)
_IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

#: Hosts and TLDs commonly used to receive exfiltrated data.
SUSPICIOUS_HOST_HINTS: tuple[str, ...] = (
    "pastebin.com", "ngrok.io", "requestbin", "webhook.site", "burpcollaborator",
    "transfer.sh", "file.io", "anonfiles", "tinyurl.com", "bit.ly",
)
SUSPICIOUS_TLDS: tuple[str, ...] = (".ru", ".tk", ".top", ".xyz", ".zip", ".onion")


class Trust(str, Enum):
    ALLOWED = "ALLOWED"
    NEUTRAL = "NEUTRAL"
    LOW = "LOW"
    DENIED = "DENIED"


@dataclass
class TrustPolicy:
    """Destination allow/deny configuration for one deployment."""

    allowed_domains: set[str] = field(default_factory=set)
    denied_domains: set[str] = field(default_factory=set)
    #: when True, any destination outside ``allowed_domains`` is LOW trust
    allowlist_only: bool = False

    def listed(self, host: str) -> Trust | None:
        """The operator's explicit verdict for a host, if there is one."""
        host = host.lower().strip(".")
        if self._matches(host, self.denied_domains):
            return Trust.DENIED
        if self._matches(host, self.allowed_domains):
            return Trust.ALLOWED
        return None

    def classify(self, host: str) -> Trust:
        host = host.lower().strip(".")
        if not host:
            return Trust.NEUTRAL
        listed = self.listed(host)
        if listed is not None:
            return listed
        if any(hint in host for hint in SUSPICIOUS_HOST_HINTS):
            return Trust.LOW
        if host.endswith(SUSPICIOUS_TLDS):
            return Trust.LOW
        if _IP_RE.match(host):
            return Trust.LOW
        return Trust.LOW if self.allowlist_only else Trust.NEUTRAL

    @staticmethod
    def _matches(host: str, domains: set[str]) -> bool:
        return any(host == d or host.endswith("." + d) for d in domains)


class DestinationTrustFilter(SecurityFilter):
    """Scores the trust of every external destination the call touches.

    The operator's allow and deny lists are policy and always win.  For a URL
    that is on neither list the URL model decides; the lexical hints above
    (drop hosts, odd TLDs, raw IPs) are used only when no model is loaded, and
    for bare e-mail domains, which the URL model was not trained on.
    """

    name = "destination_trust"

    _SCORES = {
        Trust.DENIED: 95,
        Trust.LOW: 60,
        Trust.NEUTRAL: 25,
        Trust.ALLOWED: 0,
    }

    #: model probability at which a URL is treated as malicious
    URL_THRESHOLD = 0.5

    def __init__(
        self,
        policy: TrustPolicy | None = None,
        url_model: ProbabilisticModel | None = None,
    ) -> None:
        super().__init__()
        self.policy = policy or TrustPolicy()
        self.url_model = url_model

    def check(self, call: ToolCall, session: Session) -> FilterResult:
        if call.category not in (ToolCategory.COMMUNICATE, ToolCategory.READ_REMOTE):
            return FilterResult()

        signals = []
        for host, raw in self._destinations(call):
            is_url = raw.lower().startswith(("http://", "https://"))
            if self.url_model is not None and is_url and self.policy.listed(host) is None:
                signals.append(self._model_signal(call, host, raw))
                continue
            trust = self.policy.classify(host)
            score = self._SCORES[trust]
            if trust is Trust.ALLOWED:
                continue
            # An untrusted destination the user never named is worse.
            if not call.user_authorized and trust in (Trust.LOW, Trust.DENIED):
                score = min(100, score + 10)
            signals.append(
                self.signal(
                    # an unlisted host is not a known-bad one; keeping the two
                    # apart stops "unknown" from earning the bonuses meant for "bad"
                    category="unlisted_destination" if trust is Trust.NEUTRAL
                    else "untrusted_destination",
                    score=score,
                    severity=severity_for(score),
                    message=(
                        f"destination {raw!r} has {trust.value} trust "
                        f"for an outbound {call.category.value} action"
                    ),
                    evidence=host,
                    verdict=Verdict.BLOCK if trust is Trust.DENIED else Verdict.FLAG,
                )
            )

        return FilterResult.from_signals(signals)

    def _model_signal(self, call: ToolCall, host: str, url: str):
        probability = self.url_model.predict_proba(url)
        if probability < self.URL_THRESHOLD:
            score = self._SCORES[Trust.NEUTRAL]
            return self.signal(
                category="unlisted_destination",
                score=score,
                severity=severity_for(score),
                message=(
                    f"destination {url!r} is not on the allowlist "
                    f"(URL model {probability:.2f})"
                ),
                evidence=host,
                verdict=Verdict.FLAG,
            )
        score = int(round(probability * 100))
        if not call.user_authorized:
            score = min(100, score + 10)
        return self.signal(
            category="malicious_url",
            score=score,
            severity=severity_for(score),
            message=(
                f"URL {url!r} looks malicious to the URL model "
                f"(probability {probability:.2f})"
            ),
            evidence=host,
            verdict=Verdict.FLAG,
        )

    def _destinations(self, call: ToolCall) -> list[tuple[str, str]]:
        """(host, original) pairs found in the destination field and arguments."""
        found: list[tuple[str, str]] = []
        candidates = [call.destination or ""]
        candidates.extend(str(v) for v in call.arguments.values())

        for candidate in candidates:
            if not candidate:
                continue
            for match in _EMAIL_RE.finditer(candidate):
                found.append((match.group(1), match.group(0)))
            for match in _URL_RE.finditer(candidate):
                found.append((match.group(1), match.group(0)))
            if not _EMAIL_RE.search(candidate) and not _URL_RE.search(candidate):
                parsed = urlparse(candidate)
                if parsed.netloc:
                    found.append((parsed.netloc, candidate))

        # de-duplicate by host, keeping the first spelling seen
        seen: dict[str, str] = {}
        for host, raw in found:
            seen.setdefault(host.lower(), raw)
        return list(seen.items())
