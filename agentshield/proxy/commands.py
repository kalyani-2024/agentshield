"""Command pattern: every tool invocation becomes a recordable object.

Turning calls into objects is what makes approval queues, replay and a full
audit trail possible - a pending command can sit in a queue until a human says
yes, then execute unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from agentshield.core.models import SecurityAssessment, ToolCall
from agentshield.proxy.tools import Tool


class CommandStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"


@dataclass
class ToolCommand(ABC):
    """One invocation, with its security verdict and outcome attached."""

    call: ToolCall
    tool: Tool
    status: CommandStatus = CommandStatus.PENDING
    assessment: SecurityAssessment | None = None
    result: Any = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    executed_at: datetime | None = None

    @abstractmethod
    def describe(self) -> str:
        """A one-line human summary, shown in approval prompts and audit logs."""

    def execute(self) -> Any:
        """Run the underlying tool and record the outcome."""
        try:
            self.result = self.tool.execute(**self.call.arguments)
            self.status = CommandStatus.EXECUTED
        except Exception as error:
            self.status = CommandStatus.FAILED
            self.error = f"{type(error).__name__}: {error}"
            raise
        finally:
            self.executed_at = datetime.now(timezone.utc)
        return self.result

    def approve(self) -> "ToolCommand":
        self.status = CommandStatus.APPROVED
        self.call.user_authorized = True
        return self

    def deny(self) -> "ToolCommand":
        self.status = CommandStatus.DENIED
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call.call_id,
            "type": type(self).__name__,
            "description": self.describe(),
            "status": self.status.value,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
        }


class ReadFileCommand(ToolCommand):
    def describe(self) -> str:
        return f"read file {self.call.arguments.get('path', '?')}"


class WriteFileCommand(ToolCommand):
    def describe(self) -> str:
        return f"write file {self.call.arguments.get('path', '?')}"


class SendEmailCommand(ToolCommand):
    def describe(self) -> str:
        recipient = self.call.arguments.get("recipient", self.call.destination or "?")
        subject = self.call.arguments.get("subject", "")
        return f"send email to {recipient} ({subject!r})"


class DatabaseQueryCommand(ToolCommand):
    def describe(self) -> str:
        query = str(self.call.arguments.get("query", ""))
        return f"database query: {query[:80]}"


class GenericToolCommand(ToolCommand):
    def describe(self) -> str:
        return f"invoke {self.call.tool} with {sorted(self.call.arguments)}"


#: Tool name -> command class.  Unregistered tools fall back to the generic one.
COMMAND_REGISTRY: dict[str, type[ToolCommand]] = {
    "read_file": ReadFileCommand,
    "write_file": WriteFileCommand,
    "send_email": SendEmailCommand,
    "query_database": DatabaseQueryCommand,
    "sql_query": DatabaseQueryCommand,
}


def build_command(call: ToolCall, tool: Tool) -> ToolCommand:
    """Create the right command object for a call."""
    command_cls = COMMAND_REGISTRY.get(call.tool, GenericToolCommand)
    return command_cls(call=call, tool=tool)
