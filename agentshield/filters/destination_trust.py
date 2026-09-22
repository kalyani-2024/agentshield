"""Filter 4: where is this action actually going?"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse

from agentshield.core.models import FilterResult, ToolCall, ToolCategory, Verdict
from agentshield.engine.session import Session
from agentshield.filters.base import SecurityFilter, severity_for

_EMAIL_RE = re.compile(r"[\w.+-]+@([\w-]+\.[\w.-]+)")
_URL_RE = re.compile(r"https?://([^\s/:?#]+)", re.IGNORECASE)
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

    def classify(self, host: str) -> Trust:
        host = host.lower().strip(".")
        if not host:
            return Trust.NEUTRAL
        if self._matches(host, self.denied_domains):
            return Trust.DENIED
        if self._matches(host, self.allowed_domains):
            return Trust.ALLOWED
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
    """Scores the trust of every external destination the call touches."""

    name = "destination_trust"

    _SCORES = {
        Trust.DENIED: 95,
        Trust.LOW: 60,
        Trust.NEUTRAL: 25,
        Trust.ALLOWED: 0,
    }

    def __init__(self, policy: TrustPolicy | None = None) -> None:
        super().__init__()
        self.policy = policy or TrustPolicy()

    def check(self, call: ToolCall, session: Session) -> FilterResult:
        if call.category not in (ToolCategory.COMMUNICATE, ToolCategory.READ_REMOTE):
            return FilterResult()

        signals = []
        for host, raw in self._destinations(call):
            trust = self.policy.classify(host)
            score = self._SCORES[trust]
            if trust is Trust.ALLOWED:
                continue
            # An untrusted destination the user never named is worse.
            if not call.user_authorized and trust in (Trust.LOW, Trust.DENIED):
                score = min(100, score + 10)
            signals.append(
                self.signal(
                    category="untrusted_destination",
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
