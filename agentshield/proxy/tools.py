"""Tool interface plus the in-memory mock tools used by the demo and tests."""

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


class InMemoryFileTool(Tool):
    """A fake filesystem, so the demo can be dangerous without being dangerous."""

    name = "read_file"
    description = "Read a file from the workspace"

    def __init__(self, files: dict[str, str] | None = None) -> None:
        self.files = files or {}

    def execute(self, **arguments: Any) -> Any:
        path = str(arguments.get("path", ""))
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]


class InMemoryWriteTool(Tool):
    name = "write_file"
    description = "Write a file in the workspace"

    def __init__(self, store: dict[str, str]) -> None:
        self.store = store

    def execute(self, **arguments: Any) -> Any:
        path = str(arguments.get("path", ""))
        self.store[path] = str(arguments.get("content", ""))
        return {"written": path, "bytes": len(self.store[path])}


class MockEmailTool(Tool):
    name = "send_email"
    description = "Send an email on the user's behalf"

    def __init__(self) -> None:
        self.outbox: list[dict[str, Any]] = []

    def execute(self, **arguments: Any) -> Any:
        message = {
            "recipient": arguments.get("recipient"),
            "subject": arguments.get("subject", ""),
            "body": arguments.get("body", ""),
        }
        self.outbox.append(message)
        return {"sent": True, "recipient": message["recipient"]}


class MockSearchTool(Tool):
    name = "web_search"
    description = "Search the web"

    def __init__(self, results: dict[str, str] | None = None) -> None:
        self.results = results or {}

    def execute(self, **arguments: Any) -> Any:
        query = str(arguments.get("query", ""))
        return self.results.get(query, f"no results for {query!r}")


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
