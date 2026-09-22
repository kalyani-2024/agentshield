"""Filter 3: sensitive data and credential/PII exposure."""

from __future__ import annotations

from agentshield.core.models import FilterResult, ToolCall, ToolCategory, Verdict
from agentshield.detection.patterns import (
    SECRET_SIGNATURES,
    SENSITIVE_PATH_SIGNATURES,
)
from agentshield.engine.session import Session
from agentshield.filters.base import SecurityFilter, severity_for

#: Argument names whose values are the "payload" of a call.
PAYLOAD_KEYS: tuple[str, ...] = (
    "body", "content", "text", "message", "data", "payload", "query", "value",
)
#: Argument names that name a target resource rather than carry data.
PATH_KEYS: tuple[str, ...] = ("path", "file", "filename", "filepath", "target", "resource")


class SensitiveDataFilter(SecurityFilter):
    """Finds secrets and PII in a call, and weighs them by where they are going.

    A password appearing in a local write is bad hygiene; the same password in
    the body of an outbound email is exfiltration, so the outbound case scores
    considerably higher.
    """

    name = "sensitive_data"

    #: multiplier applied when the call leaves the trust boundary
    EGRESS_MULTIPLIER = 1.25

    def check(self, call: ToolCall, session: Session) -> FilterResult:
        signals = []
        egress = call.category in (ToolCategory.COMMUNICATE, ToolCategory.READ_REMOTE)

        payload = self._payload_text(call)
        for sig in SECRET_SIGNATURES:
            match = sig.regex.search(payload)
            if not match:
                continue
            score = sig.weight
            if egress:
                score = min(100, int(score * self.EGRESS_MULTIPLIER))
            signals.append(
                self.signal(
                    category="sensitive_data_egress" if egress else "sensitive_data",
                    score=score,
                    severity=severity_for(score),
                    message=(
                        f"{sig.description} present in the call payload"
                        + (" of an outbound action" if egress else "")
                    ),
                    evidence=_redact(match.group(0)),
                    verdict=Verdict.FLAG,
                )
            )

        for path in self._paths(call):
            for sig in SENSITIVE_PATH_SIGNATURES:
                if sig.regex.search(path):
                    signals.append(
                        self.signal(
                            category="sensitive_resource_access",
                            score=sig.weight,
                            severity=severity_for(sig.weight),
                            message=f"access to {sig.description}: {path}",
                            evidence=path,
                            verdict=Verdict.FLAG,
                        )
                    )
                    break

        return FilterResult.from_signals(signals)

    def _payload_text(self, call: ToolCall) -> str:
        """Argument values that carry data, plus everything if none are named."""
        parts = [
            str(value)
            for key, value in call.arguments.items()
            if any(k in key.lower() for k in PAYLOAD_KEYS)
        ]
        return "\n".join(parts) if parts else call.argument_text()

    def _paths(self, call: ToolCall) -> list[str]:
        paths = [
            str(value)
            for key, value in call.arguments.items()
            if any(k in key.lower() for k in PATH_KEYS)
        ]
        if call.category in (ToolCategory.READ_LOCAL, ToolCategory.WRITE_LOCAL) and call.destination:
            paths.append(call.destination)
        return paths


def _redact(secret: str) -> str:
    """Keep enough of a match to identify it in an audit log, not to reuse it."""
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}{'*' * (len(secret) - 8)}{secret[-4:]}"
