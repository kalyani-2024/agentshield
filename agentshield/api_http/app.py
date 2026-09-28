"""HTTP API for AgentShield (Flask).

The API's primary job is *decision-as-a-service*: an agent runtime posts a tool
call and gets back ALLOW / REQUIRE_APPROVAL / BLOCK plus the evidence, and every
decision is persisted.  Execution stays with the agent, so no real tools are
registered server-side.

Endpoints
---------
GET  /health                      liveness
GET  /v1/backend                  which storage backends are active
POST /v1/evaluate                 judge a tool call (no execution)
GET  /v1/decisions                recent decisions (optionally ?session_id=)
GET  /v1/sessions/<id>            a session's decision history
POST /v1/sessions/<id>/reset      operator recovery back to NORMAL
GET  /v1/approvals                pending approvals
POST /v1/approvals/<call_id>      {"decision": "approve"|"deny"}
GET  /v1/incidents                opened incidents
GET  /v1/stats                    in-memory metrics + row counts
"""

from __future__ import annotations

from functools import wraps
from typing import Any

from flask import Flask, g, jsonify, request

from agentshield.core.exceptions import AdapterError
from agentshield.core.models import ContextChunk, TrustLevel
from agentshield.service import ShieldService, build_service


def _parse_context(raw: Any) -> list[ContextChunk]:
    chunks: list[ContextChunk] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        trust = item.get("trust", "UNTRUSTED")
        try:
            level = TrustLevel(str(trust).upper())
        except ValueError:
            level = TrustLevel.UNTRUSTED
        chunks.append(
            ContextChunk(
                content=str(item.get("content", "")),
                source=str(item.get("source", "unknown")),
                trust=level,
            )
        )
    return chunks


def create_app(service: ShieldService | None = None) -> Flask:
    app = Flask(__name__)
    svc = service or build_service()
    app.config["SHIELD_SERVICE"] = svc

    def require_key(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            expected = svc.settings.api_key
            if expected and request.headers.get("X-API-Key") != expected:
                return jsonify({"error": "unauthorized"}), 401
            return fn(*args, **kwargs)

        return wrapper

    # -- meta ------------------------------------------------------------
    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "version": _version()})

    @app.get("/v1/backend")
    @require_key
    def backend():
        return jsonify(svc.backend)

    # -- evaluation ------------------------------------------------------
    @app.post("/v1/evaluate")
    @require_key
    def evaluate():
        body = request.get_json(silent=True) or {}
        raw_call = body.get("call") or body.get("raw_call")
        if raw_call is None:
            return jsonify({"error": "missing 'call' in request body"}), 400
        try:
            call = svc.shield.normalize(
                raw_call,
                framework=body.get("framework", "local"),
                messages=body.get("messages"),
                principal=body.get("principal"),
                session_id=body.get("session_id"),
                context=_parse_context(body.get("context")),
                user_authorized=bool(body.get("user_authorized", False)),
            )
        except AdapterError as error:
            return jsonify({"error": f"adapter: {error}"}), 400

        assessment = svc.shield.evaluate(call)
        return jsonify(assessment.to_dict())

    # -- history / sessions ---------------------------------------------
    @app.get("/v1/decisions")
    @require_key
    def decisions():
        limit = request.args.get("limit", default=50, type=int)
        session_id = request.args.get("session_id")
        return jsonify(svc.repository.recent_decisions(limit=limit, session_id=session_id))

    @app.get("/v1/sessions/<session_id>")
    @require_key
    def session_detail(session_id: str):
        session = svc.shield.engine.sessions.get(session_id)
        return jsonify(
            {
                "session_id": session.session_id,
                "principal_id": session.principal_id,
                "state": session.state.name,
                "cumulative_risk": session.cumulative_risk,
                "decisions": svc.repository.recent_decisions(session_id=session_id),
            }
        )

    @app.post("/v1/sessions/<session_id>/reset")
    @require_key
    def session_reset(session_id: str):
        svc.shield.reset_session(session_id)
        return jsonify({"session_id": session_id, "state": "NORMAL"})

    # -- approvals -------------------------------------------------------
    @app.get("/v1/approvals")
    @require_key
    def approvals():
        pending = [
            {
                "call_id": cmd.call.call_id,
                "session_id": cmd.call.session_id,
                "description": cmd.describe(),
                "risk_score": cmd.assessment.risk_score if cmd.assessment else None,
            }
            for cmd in svc.shield.pending_approvals
        ]
        return jsonify(pending)

    @app.post("/v1/approvals/<call_id>")
    @require_key
    def decide_approval(call_id: str):
        body = request.get_json(silent=True) or {}
        decision = str(body.get("decision", "")).lower()
        if decision not in ("approve", "deny"):
            return jsonify({"error": "decision must be 'approve' or 'deny'"}), 400
        status = "APPROVED" if decision == "approve" else "DENIED"
        try:
            session_id = svc.shield.proxy.approvals.get(call_id).call.session_id
        except KeyError:
            session_id = None
        svc.repository.record_approval(call_id, session_id, status)
        return jsonify({"call_id": call_id, "status": status})

    # -- incidents / stats ----------------------------------------------
    @app.get("/v1/incidents")
    @require_key
    def incidents():
        limit = request.args.get("limit", default=50, type=int)
        return jsonify(svc.repository.list_incidents(limit=limit))

    @app.get("/v1/stats")
    @require_key
    def stats():
        return jsonify(
            {
                "metrics": svc.shield.stats(),
                "rows": svc.repository.counts(),
                "backend": svc.backend,
            }
        )

    return app


def _version() -> str:
    from agentshield import __version__

    return __version__


def main() -> None:  # pragma: no cover - manual entry point
    from agentshield.config import load_settings

    settings = load_settings()
    app = create_app(build_service(settings))
    app.run(host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":  # pragma: no cover
    main()
