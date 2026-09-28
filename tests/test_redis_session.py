"""Redis session store, exercised against an in-process fake client.

The fake implements exactly the slice of the redis interface the store uses
(get / set / delete / scan_iter), so the persistence and rehydration logic is
tested without a live server.
"""

from __future__ import annotations

import fnmatch

from agentshield.core.models import ToolCall, ToolCategory
from agentshield.engine.redis_session import (
    RedisSessionStore,
    session_from_json,
    session_to_json,
)
from agentshield.engine.session import Session


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None):
        self.store[key] = value

    def delete(self, key):
        self.store.pop(key, None)

    def scan_iter(self, match=None):
        for key in list(self.store):
            if match is None or fnmatch.fnmatch(key, match):
                yield key


def a_call(category=ToolCategory.COMMUNICATE):
    return ToolCall(tool="send_email", category=category, session_id="s1")


def test_serialization_round_trip():
    session = Session(session_id="s1", principal_id="p")
    session.record(a_call(ToolCategory.READ_LOCAL), 30, "ALLOW")
    session.record(a_call(), 95, "BLOCK")

    restored = session_from_json(session_to_json(session))
    assert restored.session_id == "s1"
    assert restored.principal_id == "p"
    assert restored.cumulative_risk == session.cumulative_risk
    assert restored.state.name == session.state.name
    assert len(restored.history) == 2
    assert restored.history[0].category is ToolCategory.READ_LOCAL


def test_get_creates_then_persists_on_record():
    store = RedisSessionStore(FakeRedis())
    session = store.get("s1", "p")
    assert session.state.name == "NORMAL"
    session.record(a_call(), 95, "BLOCK")
    # a fresh get reloads from the fake redis and sees the escalation
    reloaded = store.get("s1")
    assert reloaded.state.name == "QUARANTINED"
    assert reloaded.cumulative_risk >= 95


def test_state_only_tightens_across_reloads():
    store = RedisSessionStore(FakeRedis())
    store.get("s1").record(a_call(), 95, "BLOCK")
    assert store.get("s1").state.name == "QUARANTINED"
    store.get("s1").record(a_call(ToolCategory.READ_LOCAL), 0, "ALLOW")
    assert store.get("s1").state.name == "QUARANTINED"


def test_reset_clears_state():
    store = RedisSessionStore(FakeRedis())
    store.get("s1").record(a_call(), 95, "BLOCK")
    store.reset("s1")
    session = store.get("s1")
    assert session.state.name == "NORMAL"
    assert session.cumulative_risk == 0


def test_all_and_clear():
    client = FakeRedis()
    store = RedisSessionStore(client)
    store.get("s1").record(a_call(), 40, "ALLOW")
    store.get("s2").record(a_call(), 40, "ALLOW")
    assert len({s.session_id for s in store.all()}) == 2
    store.clear()
    assert list(store.all()) == []


def test_store_is_transparent_to_the_engine():
    from agentshield.engine.pipeline import SecurityEngine

    engine = SecurityEngine(sessions=RedisSessionStore(FakeRedis()))
    call = ToolCall(tool="read_file", category=ToolCategory.READ_LOCAL, session_id="s1")
    assessment = engine.evaluate(call)
    assert assessment.decision.value == "ALLOW"
