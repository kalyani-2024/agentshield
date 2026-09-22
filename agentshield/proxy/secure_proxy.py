"""Proxy pattern: the agent never touches a real tool directly."""

from __future__ import annotations

from typing import Any

from agentshield.core.exceptions import ApprovalRequired, ToolCallBlocked
from agentshield.core.models import Decision, SecurityAssessment, ToolCall
from agentshield.engine.events import EventType, SecurityEvent
from agentshield.engine.pipeline import SecurityEngine
from agentshield.proxy.commands import CommandStatus, ToolCommand, build_command
from agentshield.proxy.tools import Tool, ToolRegistry


class ApprovalQueue:
    """Holds commands that need a human decision before they may run."""

    def __init__(self) -> None:
        self._pending: dict[str, ToolCommand] = {}

    def add(self, command: ToolCommand) -> str:
        self._pending[command.call.call_id] = command
        return command.call.call_id

    def get(self, call_id: str) -> ToolCommand:
        try:
            return self._pending[call_id]
        except KeyError:
            raise KeyError(f"no pending command {call_id!r}") from None

    def pending(self) -> list[ToolCommand]:
        return list(self._pending.values())

    def approve(self, call_id: str) -> Any:
        """Approve and run a queued command."""
        command = self._pending.pop(call_id)
        command.approve()
        return command.execute()

    def deny(self, call_id: str) -> ToolCommand:
        command = self._pending.pop(call_id)
        return command.deny()

    def __len__(self) -> int:
        return len(self._pending)


class SecureToolProxy:
    """Stands in for the real tool registry and enforces the engine's decision.

    Only ALLOW reaches the tool.  REQUIRE_APPROVAL parks the command in the
    approval queue, and BLOCK never executes anything.
    """

    def __init__(
        self,
        engine: SecurityEngine,
        registry: ToolRegistry | None = None,
        approvals: ApprovalQueue | None = None,
    ) -> None:
        self.engine = engine
        self.registry = registry or ToolRegistry()
        self.approvals = approvals or ApprovalQueue()

    def register(self, tool: Tool, name: str | None = None) -> Tool:
        return self.registry.register(tool, name)

    def invoke(self, call: ToolCall, raise_on_deny: bool = True) -> ToolCommand:
        """Evaluate and, if permitted, execute a call.

        Returns the command in every case, so a caller that prefers inspecting
        ``command.status`` over catching exceptions can pass
        ``raise_on_deny=False``.
        """
        tool = self.registry.get(call.tool)
        command = build_command(call, tool)
        assessment = self.engine.evaluate(call)
        command.assessment = assessment

        if assessment.decision is Decision.BLOCK:
            command.deny()
            if raise_on_deny:
                raise ToolCallBlocked(assessment)
            return command

        if assessment.decision is Decision.REQUIRE_APPROVAL:
            self.approvals.add(command)
            if raise_on_deny:
                raise ApprovalRequired(assessment)
            return command

        return self._execute(command, assessment)

    def _execute(
        self, command: ToolCommand, assessment: SecurityAssessment
    ) -> ToolCommand:
        try:
            command.execute()
        except Exception as error:
            self.engine.bus.publish(
                SecurityEvent(
                    type=EventType.TOOL_FAILED,
                    assessment=assessment,
                    payload={"tool": command.call.tool, "error": str(error)},
                )
            )
            raise
        self.engine.bus.publish(
            SecurityEvent(
                type=EventType.TOOL_EXECUTED,
                assessment=assessment,
                payload={"tool": command.call.tool, "status": command.status.value},
            )
        )
        return command

    def approve(self, call_id: str) -> Any:
        """Human approval path: run a command that was parked by the policy."""
        command = self.approvals.get(call_id)
        result = self.approvals.approve(call_id)
        if command.assessment is not None:
            self.engine.bus.publish(
                SecurityEvent(
                    type=EventType.TOOL_EXECUTED,
                    assessment=command.assessment,
                    payload={
                        "tool": command.call.tool,
                        "status": CommandStatus.EXECUTED.value,
                        "approved_by_human": True,
                    },
                )
            )
        return result

    def deny(self, call_id: str) -> ToolCommand:
        return self.approvals.deny(call_id)
