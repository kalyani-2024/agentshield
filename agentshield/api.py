"""The facade an integrator actually uses.

    shield = AgentShield()
    shield.register(MockEmailTool())
    assessment = shield.inspect(raw_call, framework="openai", messages=history)
"""

from __future__ import annotations

from typing import Any

from agentshield.adapters.frameworks import get_factory
from agentshield.core.models import (
    ContextChunk,
    Principal,
    SecurityAssessment,
    ToolCall,
)
from agentshield.engine.events import (
    AlertObserver,
    AuditObserver,
    IncidentObserver,
    MetricsObserver,
    SecurityObserver,
)
from agentshield.engine.pipeline import SecurityEngine
from agentshield.filters.destination_trust import TrustPolicy
from agentshield.proxy.commands import ToolCommand
from agentshield.proxy.secure_proxy import SecureToolProxy
from agentshield.proxy.tools import Tool


class AgentShield:
    """One object wiring the engine, the observers and the secure proxy."""

    def __init__(
        self,
        trust_policy: TrustPolicy | None = None,
        engine: SecurityEngine | None = None,
        audit_path: str | None = None,
    ) -> None:
        self.engine = engine or SecurityEngine(trust_policy=trust_policy)
        self.audit = AuditObserver(audit_path)
        self.metrics = MetricsObserver()
        self.alerts = AlertObserver()
        self.incidents = IncidentObserver()
        for observer in (self.audit, self.metrics, self.alerts, self.incidents):
            self.engine.subscribe(observer)
        self.proxy = SecureToolProxy(self.engine)

    # -- tools -----------------------------------------------------------
    def register(self, tool: Tool, name: str | None = None) -> Tool:
        return self.proxy.register(tool, name)

    # -- evaluation ------------------------------------------------------
    def normalize(
        self,
        raw_call: Any,
        framework: str = "local",
        messages: Any = None,
        principal: Principal | dict | None = None,
        session_id: str | None = None,
        context: list[ContextChunk] | None = None,
        user_authorized: bool = False,
    ) -> ToolCall:
        """Convert a framework-native call into the canonical form."""
        factory = get_factory(framework)
        adapter = factory.create_adapter()

        chunks = list(context or [])
        if messages is not None:
            chunks.extend(factory.create_message_parser().extract_context(messages))

        resolved = principal
        if principal is not None and not isinstance(principal, Principal):
            resolved = factory.create_policy_translator().to_principal(principal)

        return adapter.normalize(
            raw_call,
            principal=resolved,
            session_id=session_id,
            context=chunks,
            user_authorized=user_authorized,
        )

    def inspect(self, raw_call: Any, framework: str = "local", **kwargs: Any) -> SecurityAssessment:
        """Normalise and evaluate a call without executing anything."""
        return self.engine.evaluate(self.normalize(raw_call, framework, **kwargs))

    def evaluate(self, call: ToolCall) -> SecurityAssessment:
        return self.engine.evaluate(call)

    # -- execution -------------------------------------------------------
    def invoke(self, call: ToolCall, raise_on_deny: bool = True) -> ToolCommand:
        """Evaluate and, when permitted, run the call through the proxy."""
        return self.proxy.invoke(call, raise_on_deny=raise_on_deny)

    def approve(self, call_id: str) -> Any:
        return self.proxy.approve(call_id)

    def deny(self, call_id: str) -> ToolCommand:
        return self.proxy.deny(call_id)

    @property
    def pending_approvals(self) -> list[ToolCommand]:
        return self.proxy.approvals.pending()

    # -- observability ---------------------------------------------------
    def subscribe(self, observer: SecurityObserver) -> SecurityObserver:
        return self.engine.subscribe(observer)

    def stats(self) -> dict[str, Any]:
        return self.metrics.snapshot()

    def reset_session(self, session_id: str) -> None:
        """Operator action: return a quarantined session to NORMAL."""
        self.engine.sessions.reset(session_id)
