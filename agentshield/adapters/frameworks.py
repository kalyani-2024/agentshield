"""Concrete adapters and the Abstract Factory that assembles an integration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentshield.adapters.base import (
    DefaultMessageParser,
    DefaultPolicyTranslator,
    MessageParser,
    PolicyTranslator,
    ToolCallAdapter,
)
from agentshield.core.exceptions import AdapterError
from agentshield.core.models import ContextChunk, TrustLevel


class OpenAIToolCallAdapter(ToolCallAdapter):
    """OpenAI / Groq style: ``{"function": {"name", "arguments": "<json>"}}``."""

    framework = "openai"

    def parse(self, raw: Any) -> tuple[str, dict[str, Any]]:
        if not isinstance(raw, dict):
            raise AdapterError("openai: tool call must be a mapping")
        function = raw.get("function", raw)
        if not isinstance(function, dict):
            raise AdapterError("openai: 'function' must be a mapping")
        name = function.get("name", "")
        arguments = self._as_dict(function.get("arguments"), "arguments", self.framework)
        return str(name), arguments


class GeminiToolCallAdapter(ToolCallAdapter):
    """Gemini style: ``{"functionCall": {"name", "args": {...}}}``."""

    framework = "gemini"

    def parse(self, raw: Any) -> tuple[str, dict[str, Any]]:
        if not isinstance(raw, dict):
            raise AdapterError("gemini: tool call must be a mapping")
        call = raw.get("functionCall") or raw.get("function_call") or raw
        if not isinstance(call, dict):
            raise AdapterError("gemini: 'functionCall' must be a mapping")
        name = call.get("name", "")
        arguments = self._as_dict(
            call.get("args", call.get("arguments")), "args", self.framework
        )
        return str(name), arguments


class MCPToolCallAdapter(ToolCallAdapter):
    """MCP style: a ``tools/call`` JSON-RPC request."""

    framework = "mcp"

    def parse(self, raw: Any) -> tuple[str, dict[str, Any]]:
        if not isinstance(raw, dict):
            raise AdapterError("mcp: request must be a mapping")
        method = raw.get("method")
        if method is not None and method != "tools/call":
            raise AdapterError(f"mcp: unsupported method {method!r}")
        params = raw.get("params", raw)
        if not isinstance(params, dict):
            raise AdapterError("mcp: 'params' must be a mapping")
        name = params.get("name", "")
        arguments = self._as_dict(params.get("arguments"), "arguments", self.framework)
        return str(name), arguments


class GenericToolCallAdapter(ToolCallAdapter):
    """Plain ``{"tool"|"name": ..., "arguments"|"args"|"input": {...}}``."""

    framework = "generic"

    def parse(self, raw: Any) -> tuple[str, dict[str, Any]]:
        if not isinstance(raw, dict):
            raise AdapterError("generic: tool call must be a mapping")
        name = raw.get("tool") or raw.get("name") or ""
        payload = raw.get("arguments")
        if payload is None:
            payload = raw.get("args")
        if payload is None:
            payload = raw.get("input")
        return str(name), self._as_dict(payload, "arguments", self.framework)


class GeminiMessageParser(DefaultMessageParser):
    """Gemini ``contents``: ``{"role": "user"|"model", "parts": [{"text": ...}]}``."""

    framework = "gemini"

    def extract_context(self, messages: Any) -> list[ContextChunk]:
        if not isinstance(messages, (list, tuple)):
            raise AdapterError("gemini: contents must be a sequence")
        chunks: list[ContextChunk] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "model")).lower()
            texts = [
                part.get("text", "")
                for part in message.get("parts", [])
                if isinstance(part, dict) and part.get("text")
            ]
            if not texts:
                continue
            trust = TrustLevel.USER if role == "user" else TrustLevel.UNTRUSTED
            chunks.append(
                ContextChunk(content="\n".join(texts), source=role, trust=trust)
            )
        return chunks


class AgentIntegrationFactory(ABC):
    """Abstract Factory: one coherent integration per agent ecosystem."""

    framework: str = "abstract"

    @abstractmethod
    def create_adapter(self) -> ToolCallAdapter:
        ...

    @abstractmethod
    def create_message_parser(self) -> MessageParser:
        ...

    def create_policy_translator(self) -> PolicyTranslator:
        return DefaultPolicyTranslator()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} framework={self.framework!r}>"


class OpenAIFactory(AgentIntegrationFactory):
    framework = "openai"

    def create_adapter(self) -> ToolCallAdapter:
        return OpenAIToolCallAdapter()

    def create_message_parser(self) -> MessageParser:
        return DefaultMessageParser()


class GeminiFactory(AgentIntegrationFactory):
    framework = "gemini"

    def create_adapter(self) -> ToolCallAdapter:
        return GeminiToolCallAdapter()

    def create_message_parser(self) -> MessageParser:
        return GeminiMessageParser()


class MCPFactory(AgentIntegrationFactory):
    framework = "mcp"

    def create_adapter(self) -> ToolCallAdapter:
        return MCPToolCallAdapter()

    def create_message_parser(self) -> MessageParser:
        return DefaultMessageParser()


class LocalAgentFactory(AgentIntegrationFactory):
    """For in-process agents and the benchmark harness."""

    framework = "local"

    def create_adapter(self) -> ToolCallAdapter:
        return GenericToolCallAdapter()

    def create_message_parser(self) -> MessageParser:
        return DefaultMessageParser()


FACTORY_REGISTRY: dict[str, type[AgentIntegrationFactory]] = {
    OpenAIFactory.framework: OpenAIFactory,
    GeminiFactory.framework: GeminiFactory,
    MCPFactory.framework: MCPFactory,
    LocalAgentFactory.framework: LocalAgentFactory,
}


def get_factory(framework: str) -> AgentIntegrationFactory:
    """Build the integration factory for a framework name."""
    try:
        return FACTORY_REGISTRY[framework.lower()]()
    except KeyError:
        raise AdapterError(
            f"no integration for framework {framework!r}; "
            f"available: {sorted(FACTORY_REGISTRY)}"
        ) from None
