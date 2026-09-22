"""Adapter pattern: foreign tool-call formats in, canonical ToolCall out."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from agentshield.adapters.categories import classify_tool, infer_destination
from agentshield.core.exceptions import AdapterError
from agentshield.core.models import ContextChunk, Principal, ToolCall, TrustLevel


class ToolCallAdapter(ABC):
    """Normalises one framework's tool-call representation."""

    framework: str = "abstract"

    @abstractmethod
    def parse(self, raw: Any) -> tuple[str, dict[str, Any]]:
        """Extract ``(tool_name, arguments)`` from the framework payload."""

    def normalize(
        self,
        raw: Any,
        principal: Principal | None = None,
        session_id: str | None = None,
        context: list[ContextChunk] | None = None,
        user_authorized: bool = False,
    ) -> ToolCall:
        """Build a canonical :class:`ToolCall` from a framework payload."""
        tool, arguments = self.parse(raw)
        if not tool:
            raise AdapterError(f"{self.framework}: payload carries no tool name")

        kwargs: dict[str, Any] = {
            "tool": tool,
            "arguments": arguments,
            "category": classify_tool(tool),
            "destination": infer_destination(arguments),
            "context": context or [],
            "user_authorized": user_authorized,
            "source_framework": self.framework,
        }
        if principal is not None:
            kwargs["principal"] = principal
        if session_id is not None:
            kwargs["session_id"] = session_id
        return ToolCall(**kwargs)

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _as_dict(value: Any, field_name: str, framework: str) -> dict[str, Any]:
        """Arguments arrive either as a dict or as a JSON string."""
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            if not value.strip():
                return {}
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as error:
                raise AdapterError(
                    f"{framework}: {field_name} is not valid JSON: {error}"
                ) from error
            if not isinstance(parsed, dict):
                raise AdapterError(f"{framework}: {field_name} must decode to an object")
            return parsed
        raise AdapterError(f"{framework}: unsupported {field_name} type {type(value).__name__}")


class MessageParser(ABC):
    """Extracts the context the agent read before requesting a tool call.

    This is what makes *indirect* injection detectable: the filters need the
    document, email or page the model just ingested, tagged with its trust.
    """

    framework: str = "abstract"

    @abstractmethod
    def extract_context(self, messages: Any) -> list[ContextChunk]:
        """Turn a framework conversation into trust-tagged context chunks."""


class PolicyTranslator(ABC):
    """Maps framework-native permission metadata onto internal grants."""

    framework: str = "abstract"

    @abstractmethod
    def to_principal(self, raw: Any) -> Principal:
        """Build a :class:`Principal` from framework identity/permission data."""


class DefaultPolicyTranslator(PolicyTranslator):
    """Reads a simple ``{"id", "roles", "allowed_tools"}`` mapping."""

    framework = "generic"

    def to_principal(self, raw: Any) -> Principal:
        if isinstance(raw, Principal):
            return raw
        if not isinstance(raw, dict):
            raise AdapterError("principal data must be a mapping")
        allowed = raw.get("allowed_tools", ["*"])
        return Principal(
            id=str(raw.get("id", "anonymous")),
            display_name=str(raw.get("display_name", "")),
            roles=tuple(raw.get("roles", ())),
            allowed_tools=frozenset(allowed),
        )


class DefaultMessageParser(MessageParser):
    """Handles ``[{"role": ..., "content": ...}]`` style histories.

    ``system`` is trusted, ``user`` is user-trust, and everything else - tool
    output above all - is untrusted, because that is where injected text lives.
    """

    framework = "generic"

    _TRUST_BY_ROLE = {
        "system": TrustLevel.TRUSTED,
        "developer": TrustLevel.TRUSTED,
        "user": TrustLevel.USER,
    }

    def extract_context(self, messages: Any) -> list[ContextChunk]:
        if not isinstance(messages, (list, tuple)):
            raise AdapterError("messages must be a sequence")
        chunks: list[ContextChunk] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            role = str(message.get("role", "tool")).lower()
            chunks.append(
                ContextChunk(
                    content=content,
                    source=str(message.get("name") or role),
                    trust=self._TRUST_BY_ROLE.get(role, TrustLevel.UNTRUSTED),
                )
            )
        return chunks
