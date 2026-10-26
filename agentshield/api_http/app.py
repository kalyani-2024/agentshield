"""HTTP API for AgentShield (FastAPI).

Two ways to use it:

* **Decision as a service** - ``POST /v1/evaluate``.  The agent runtime posts a
  tool call, gets ALLOW / REQUIRE_APPROVAL / BLOCK with the evidence, and runs
  the tool itself.
* **Secure proxy** - ``POST /v1/invoke``.  AgentShield judges the call *and*
  executes it with the real tool connectors, so the agent never touches a tool.

Either way every decision is persisted, and a call that needs approval waits
until a human answers ``POST /v1/approvals/{call_id}``.

Interactive documentation is served at ``/docs``.

Endpoints
---------
GET  /health                      liveness
GET  /v1/backend                  storage, session store, detector and tools in use
GET  /v1/tools                    the tools behind the proxy, and which are off
POST /v1/evaluate                 judge a tool call (no execution)
POST /v1/invoke                   judge a tool call and, if allowed, execute it
GET  /v1/decisions                recent decisions (optionally ?session_id=)
GET  /v1/sessions/{id}            a session's state and decision history
POST /v1/sessions/{id}/reset      operator recovery back to NORMAL
GET  /v1/approvals                approvals (default ?status=PENDING)
GET  /v1/approvals/{call_id}      one approval's status
POST /v1/approvals/{call_id}      {"decision": "approve"|"deny"}
GET  /v1/incidents                opened incidents
GET  /v1/alerts                   recent alert messages
GET  /v1/stats                    in-memory metrics + row counts
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from agentshield.core.exceptions import AdapterError
from agentshield.core.models import ContextChunk, ToolCall, TrustLevel
from agentshield.proxy.commands import CommandStatus
from agentshield.service import ShieldService, build_service


class ContextItem(BaseModel):
    content: str = ""
    source: str = "unknown"
    trust: str = "UNTRUSTED"


class CallRequest(BaseModel):
    """A tool call in its framework's own format, plus what surrounded it."""

    call: Any = Field(default=None, description="The raw tool call payload")
    raw_call: Any = Field(default=None, description="Alias of 'call'")
    framework: str = Field(
        default="local",
        description="openai | groq | gemini | mcp | local | json | custom",
    )
    messages: list[dict[str, Any]] | None = Field(
        default=None, description="Conversation history; tool output is read as untrusted"
    )
    principal: dict[str, Any] | None = None
    session_id: str | None = None
    context: list[ContextItem] = Field(default_factory=list)
    user_authorized: bool = False


class ApprovalDecision(BaseModel):
    decision: Literal["approve", "deny"]


def _parse_context(items: list[ContextItem]) -> list[ContextChunk]:
    chunks: list[ContextChunk] = []
    for item in items:
        try:
            level = TrustLevel(item.trust.upper())
        except ValueError:
            level = TrustLevel.UNTRUSTED
        chunks.append(ContextChunk(content=item.content, source=item.source, trust=level))
    return chunks


def create_app(service: ShieldService | None = None) -> FastAPI:
    svc = service or build_service()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        svc.close()

    app = FastAPI(
        title="AgentShield",
        description="A runtime security firewall for tool-using AI agents.",
        version=_version(),
        lifespan=lifespan,
    )
    app.state.service = svc

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        expected = svc.settings.api_key
        if expected and x_api_key != expected:
            raise HTTPException(status_code=401, detail="unauthorized")

    v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_key)])

    def normalize(body: CallRequest) -> ToolCall:
        raw_call = body.call if body.call is not None else body.raw_call
        if raw_call is None:
            raise HTTPException(status_code=400, detail="missing 'call' in request body")
        try:
            return svc.shield.normalize(
                raw_call,
                framework=body.framework,
                messages=body.messages,
                principal=body.principal,
                session_id=body.session_id,
                context=_parse_context(body.context),
                user_authorized=body.user_authorized,
            )
        except AdapterError as error:
            raise HTTPException(status_code=400, detail=f"adapter: {error}") from error

    # -- meta ------------------------------------------------------------
    @app.get("/health")
    def health():
        return {"status": "ok", "version": _version()}

    @v1.get("/backend")
    def backend():
        return svc.backend

    @v1.get("/tools")
    def tools():
        registry = svc.shield.proxy.registry
        return {
            "enabled": [
                {"name": name, "description": registry.get(name).description}
                for name in registry.names()
            ],
            "disabled": svc.disabled_tools,
        }

    # -- evaluation ------------------------------------------------------
    @v1.post("/evaluate")
    def evaluate(body: CallRequest):
        return svc.shield.evaluate(normalize(body)).to_dict()

    @v1.post("/invoke")
    def invoke(body: CallRequest):
        call = normalize(body)
        if call.tool not in svc.shield.proxy.registry:
            raise HTTPException(
                status_code=404,
                detail=f"no enabled tool named {call.tool!r}; see GET /v1/tools",
            )
        try:
            command = svc.shield.invoke(call, raise_on_deny=False)
        except Exception as error:  # the call was allowed but the tool itself failed
            return {
                "call_id": call.call_id,
                "status": CommandStatus.FAILED.value,
                "executed": False,
                "error": f"{type(error).__name__}: {error}",
            }
        return {
            "call_id": call.call_id,
            "status": command.status.value,
            "executed": command.status is CommandStatus.EXECUTED,
            "result": command.result,
            "assessment": command.assessment.to_dict() if command.assessment else None,
        }

    # -- history / sessions ---------------------------------------------
    @v1.get("/decisions")
    def decisions(limit: int = Query(default=50, ge=1, le=500), session_id: str | None = None):
        return svc.repository.recent_decisions(limit=limit, session_id=session_id)

    @v1.get("/sessions/{session_id}")
    def session_detail(session_id: str):
        session = svc.shield.engine.sessions.get(session_id)
        return {
            "session_id": session.session_id,
            "principal_id": session.principal_id,
            "state": session.state.name,
            "cumulative_risk": session.cumulative_risk,
            "decisions": svc.repository.recent_decisions(session_id=session_id),
        }

    @v1.post("/sessions/{session_id}/reset")
    def session_reset(session_id: str):
        svc.shield.reset_session(session_id)
        return {"session_id": session_id, "state": "NORMAL"}

    # -- approvals -------------------------------------------------------
    @v1.get("/approvals")
    def approvals(status: str = "PENDING", limit: int = Query(default=50, ge=1, le=500)):
        queued = {cmd.call.call_id: cmd for cmd in svc.shield.pending_approvals}
        rows = svc.repository.list_approvals(
            status=None if status.upper() == "ALL" else status.upper(), limit=limit
        )
        for row in rows:
            command = queued.get(row["call_id"])
            # a queued command is one this service will execute once approved
            row["executes_on_approval"] = command is not None
            if command is not None:
                row["description"] = command.describe()
                row["risk_score"] = (
                    command.assessment.risk_score if command.assessment else None
                )
        return rows

    @v1.get("/approvals/{call_id}")
    def approval_status(call_id: str):
        row = svc.repository.get_approval(call_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"no approval for call {call_id!r}")
        return row

    @v1.post("/approvals/{call_id}")
    def decide_approval(call_id: str, body: ApprovalDecision):
        row = svc.repository.get_approval(call_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"no approval for call {call_id!r}")
        if row["status"] != "PENDING":
            raise HTTPException(
                status_code=409, detail=f"call {call_id!r} is already {row['status']}"
            )
        queued = any(cmd.call.call_id == call_id for cmd in svc.shield.pending_approvals)
        response: dict[str, Any] = {"call_id": call_id, "executed": False}

        if body.decision == "deny":
            if queued:
                svc.shield.deny(call_id)
            status = "DENIED"
        elif queued:
            # the call came through /v1/invoke: release it and run it now
            try:
                response["result"] = svc.shield.approve(call_id)
                response["executed"] = True
                status = "EXECUTED"
            except Exception as error:
                response["error"] = f"{type(error).__name__}: {error}"
                status = "FAILED"
        else:
            # the call came through /v1/evaluate: the agent runtime executes it
            status = "APPROVED"

        svc.repository.record_approval(call_id, row.get("session_id"), status)
        response["status"] = status
        return response

    # -- incidents / alerts / stats --------------------------------------
    @v1.get("/incidents")
    def incidents(limit: int = Query(default=50, ge=1, le=500)):
        return svc.repository.list_incidents(limit=limit)

    @v1.get("/alerts")
    def alerts(limit: int = Query(default=50, ge=1, le=500)):
        return svc.shield.alerts.alerts[-limit:]

    @v1.get("/stats")
    def stats():
        return {
            "metrics": svc.shield.stats(),
            "rows": svc.repository.counts(),
            "backend": svc.backend,
        }

    app.include_router(v1)
    return app


def _version() -> str:
    from agentshield import __version__

    return __version__


def main() -> None:  # pragma: no cover - manual entry point
    import uvicorn

    from agentshield.config import load_dotenv, load_settings

    load_dotenv()
    settings = load_settings()
    app = create_app(build_service(settings))
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":  # pragma: no cover
    main()
