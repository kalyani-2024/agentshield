"""Signature catalogue used by the deterministic detection strategies."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Signature:
    id: str
    description: str
    weight: int
    regex: re.Pattern[str]


def _sig(id_: str, description: str, weight: int, pattern: str) -> Signature:
    return Signature(id_, description, weight, re.compile(pattern, re.IGNORECASE))


#: Instruction-override style injections.
INJECTION_SIGNATURES: tuple[Signature, ...] = (
    _sig(
        "PI001",
        "instruction override ('ignore previous instructions')",
        95,
        r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b"
        r"(all\s+)?(previous|prior|earlier|above|preceding|system)\b"
        r"[^.\n]{0,20}\b(instruction|prompt|rule|direction|message)s?\b",
    ),
    _sig(
        "PI002",
        "role / persona hijack",
        70,
        r"\b(you are now|act as|pretend to be|from now on you)\b"
        r"[^.\n]{0,40}\b(dan|admin|root|developer mode|unrestricted|jailbroken)\b",
    ),
    _sig(
        "PI003",
        "fake system / operator turn injected into content",
        80,
        r"(^|\n)\s*(\[|<|#{0,3}\s*)?(system|assistant|developer)\s*(\]|>|:)\s*",
    ),
    _sig(
        "PI004",
        "secrecy instruction ('do not tell the user')",
        75,
        r"\b(do not|don't|never)\b[^.\n]{0,30}\b"
        r"(tell|inform|mention|show|reveal|notify|ask)\b[^.\n]{0,20}\b(the )?(user|human|operator)\b",
    ),
    _sig(
        "PI005",
        "credential harvesting instruction",
        90,
        r"\b(find|search|locate|collect|gather|read|retrieve|extract)\b[^.\n]{0,40}\b"
        r"(credential|password|api[_\s-]?key|secret|token|private key|\.env|ssh key)s?\b",
    ),
    _sig(
        "PI006",
        "exfiltration instruction (send data outside)",
        92,
        r"\b(send|email|post|upload|forward|exfiltrate|transmit|leak)\b[^.\n]{0,40}\b"
        r"(to|at)\b\s*(https?://|www\.|[\w.+-]+@[\w-]+\.[a-z]{2,})",
    ),
    _sig(
        "PI007",
        "authorisation spoofing ('the user has approved')",
        65,
        r"\b(the )?(user|owner|admin|operator)\b[^.\n]{0,25}\b"
        r"(has|have|already)\b[^.\n]{0,15}\b(approved|authorized|authorised|permitted|consented)\b",
    ),
    _sig(
        "PI008",
        "urgency / safety-override framing",
        45,
        r"\b(urgent|immediately|without asking|no confirmation|skip (the )?(approval|confirmation|check)"
        r"|bypass (the )?(security|filter|policy))\b",
    ),
    _sig(
        "PI009",
        "delimiter escape attempt",
        60,
        r"(</?(system|instructions?|prompt)>|```\s*system|-{3,}\s*end of (document|context))",
    ),
    _sig(
        "PI010",
        "encoded payload instruction (base64 / hex decode-and-run)",
        70,
        r"\b(decode|base64|rot13|hex)\b[^.\n]{0,30}\b(and|then)\b[^.\n]{0,20}\b(run|execute|follow|obey)\b",
    ),
)


#: Words that raise suspicion only in aggregate, used by the heuristic strategy.
SUSPICIOUS_TERMS: dict[str, int] = {
    "ignore": 8,
    "instruction": 5,
    "instructions": 6,
    "override": 10,
    "bypass": 12,
    "jailbreak": 20,
    "credentials": 15,
    "password": 14,
    "passwords": 14,
    "api_key": 16,
    "apikey": 16,
    "secret": 10,
    "secrets": 12,
    "token": 8,
    "exfiltrate": 25,
    "silently": 12,
    "covertly": 15,
    "without telling": 18,
    "attacker": 20,
    "payload": 10,
    "urgent": 5,
    "confidential": 8,
    "admin": 6,
    "sudo": 10,
    "curl": 8,
    "wget": 8,
}


#: Sensitive-data detectors, used by the SensitiveDataFilter.
SECRET_SIGNATURES: tuple[Signature, ...] = (
    _sig("SD001", "AWS access key id", 95, r"\bAKIA[0-9A-Z]{16}\b"),
    _sig(
        "SD002",
        "AWS secret access key assignment",
        95,
        r"\baws_secret_access_key\b\s*[:=]\s*\S{20,}",
    ),
    _sig("SD003", "generic API key assignment", 80,
         r"\b(api[_-]?key|apikey|access[_-]?token|auth[_-]?token)\b\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{16,}"),
    _sig("SD004", "password assignment", 75,
         r"\b(password|passwd|pwd)\b\s*[:=]\s*[\"']?\S{6,}"),
    _sig("SD005", "private key block", 98,
         r"-{5}BEGIN [A-Z ]*PRIVATE KEY-{5}"),
    _sig("SD006", "GitHub / Slack style token", 90,
         r"\b(gh[pousr]_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,})\b"),
    _sig("SD007", "bearer token header", 70,
         r"\bAuthorization\b\s*:\s*Bearer\s+\S{10,}"),
    _sig("SD008", "credit card number", 85,
         r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b"),
    _sig("SD009", "US social security number", 80,
         r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b"),
    _sig("SD010", "database connection string with password", 88,
         r"\b(postgres(ql)?|mysql|mongodb(\+srv)?|redis)://[^\s:]+:[^\s@]+@\S+"),
    _sig("SD011", "email address", 25, r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b"),
)


#: File paths that usually hold secrets.
SENSITIVE_PATH_SIGNATURES: tuple[Signature, ...] = (
    _sig("SP001", "environment / secrets file", 70,
         r"(^|[\\/])\.env(\.|$)|(^|[\\/])secrets?\.(ya?ml|json|txt|ini)$"),
    _sig("SP002", "ssh or gpg key material", 85,
         r"(^|[\\/])\.(ssh|gnupg)([\\/]|$)|id_(rsa|ed25519|ecdsa)"),
    _sig("SP003", "cloud credentials file", 85,
         r"(^|[\\/])\.aws([\\/]|$)|credentials(\.json)?$|(^|[\\/])\.kube([\\/]|$)"),
    _sig("SP004", "password store", 80,
         r"(password|passwd|shadow|keystore|keychain|wallet)"),
    _sig("SP005", "browser / token cache", 60,
         r"(cookies?\.sqlite|Login Data|token(s)?\.json)"),
)
