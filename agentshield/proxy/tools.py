"""The tool interface and the registry the secure proxy executes from.

The real implementations live in :mod:`agentshield.proxy.connectors`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """What the agent thinks it is calling; also what the proxy wraps."""

    name: str = "tool"
    description: str = ""

    @abstractmethod
    def execute(self, **arguments: Any) -> Any:
        """Perform the real side effect."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r}>"


class ToolRegistry:
    """Name -> tool lookup used by the proxy."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool, name: str | None = None) -> Tool:
        self._tools[name or tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"no tool registered under {name!r}") from None

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools
