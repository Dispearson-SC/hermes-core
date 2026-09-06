"""Tests for the session-state seam.

Behavioural coverage runs against both shipped implementations through one shared
suite (``StoreUnderTest``) -- the whole point of the ``SessionStore`` Protocol is that
``InMemorySessionStore`` and ``SqliteSessionStore`` are interchangeable, so a test that
only exercised one of them would not prove that. ``HostOwnedStore`` at the bottom is a
third implementation, written from scratch with no import from this seam at all, to
prove the protocol is satisfied structurally rather than through inheritance.

The end-to-end section is the real claim: a second, independent ``AIAgent`` -- a fresh
process would build one exactly like it -- reads the first agent's turn back out of a
store instance it never wrote to, and the model sees it on the next call.
"""

import json
import tempfile

import pytest

from hermes_core.seams.config import DictConfigSource, set_config_source
from hermes_core.seams.credentials import StaticCredentials, set_credential_source
from hermes_core.seams.paths import DirectoryWorkspace, set_workspace
from hermes_core.seams.session_store import InMemorySessionStore, SessionStore, SqliteSessionStore
from hermes_core.testing import Script, install_fake_client


@pytest.fixture(autouse=True)
def isolated_core(monkeypatch):
    set_workspace(DirectoryWorkspace(tempfile.mkdtemp()))
    set_config_source(DictConfigSource({"model": {"default": "fake-model", "provider": "openai"}}))
    set_credential_source(StaticCredentials("sk-test"))
    yield


# -- shared behavioural suite: same tests, both shipped stores -----------------------

@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemorySessionStore()
    return SqliteSessionStore(tmp_path / "sessions.db")


def test_both_shipped_stores_satisfy_the_protocol(store):
    assert isinstance(store, SessionStore)


def test_messages_persist_and_read_back_in_order(store):
    store.create_session(session_id="s1", source="cli", model="m")
    store.append_messages_batch(session_id="s1", messages=[
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ])

    read_back = store.get_messages_as_conversation("s1")

    assert [m["content"] for m in read_back] == ["one", "two", "three"]
    assert [m["role"] for m in read_back] == ["user", "assistant", "user"]


def test_a_tool_result_round_trips_its_name_and_call_id(store):
    store.create_session(session_id="s1", source="cli", model="m")
    store.append_messages_batch(session_id="s1", messages=[
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function"}]},
        {"role": "tool", "content": "21", "tool_name": "get_weather", "tool_call_id": "call_1"},
    ])

    read_back = store.get_messages_as_conversation("s1")

    assert read_back[0]["tool_calls"] == [{"id": "call_1", "type": "function"}]
    assert read_back[1] == {"role": "tool", "content": "21", "name": "get_weather", "tool_call_id": "call_1"}


def test_appending_to_an_unknown_session_does_not_lose_the_messages(store):
    """append_messages_batch has no ``create_session`` precondition in the real call
    sites (the row is created lazily, sometimes after messages are already staged)."""
    store.append_messages_batch(session_id="orphan", messages=[{"role": "user", "content": "hi"}])

    assert [m["content"] for m in store.get_messages_as_conversation("orphan")] == ["hi"]


def test_get_session_returns_none_for_an_unknown_id(store):
    assert store.get_session("does-not-exist") is None


def test_session_title_round_trips(store):
    store.create_session(session_id="s1", source="cli", model="m")
    assert store.get_session_title("s1") is None

    store.set_session_title("s1", "Weather chat")
    store.set_session_title_source("s1", "auto")

    assert store.get_session_title("s1") == "Weather chat"
    assert store.get_session_title_source("s1") == "auto"


def test_update_system_prompt_is_visible_on_the_session_row(store):
    store.create_session(session_id="s1", source="cli", model="m")
    store.update_system_prompt("s1", "You are Hermes.")

    assert store.get_session("s1")["system_prompt"] == "You are Hermes."


def test_end_session_is_first_reason_wins(store):
    store.create_session(session_id="s1", source="cli", model="m")
    store.end_session("s1", "agent_close")
    store.end_session("s1", "compression")  # must not overwrite the first reason

    row = store.get_session("s1")
    assert row["ended_at"] is not None
    assert row["end_reason"] == "agent_close"


def test_queue_token_counts_accumulates(store):
    store.create_session(session_id="s1", source="cli", model="m")
    store.queue_token_counts("s1", input_tokens=10, output_tokens=5)
    store.queue_token_counts("s1", input_tokens=3, output_tokens=0)
    store.flush_token_counts()  # must be safe to call, queued or not

    counts = store.get_session("s1")["token_counts"]
    assert counts["input_tokens"] == 13
    assert counts["output_tokens"] == 5


def test_get_active_message_watermark_counts_stored_rows(store):
    store.create_session(session_id="s1", source="cli", model="m")
    assert store.get_active_message_watermark("s1") == 0

    store.append_messages_batch(session_id="s1", messages=[{"role": "user", "content": "hi"}])

    assert store.get_active_message_watermark("s1") == 1


def test_get_compression_tip_of_an_uncompressed_session_is_itself(store):
    store.create_session(session_id="s1", source="cli", model="m")
    assert store.get_compression_tip("s1") == "s1"


def test_publish_compression_child_is_found_by_get_compression_tip(store):
    store.create_session(session_id="s1", source="cli", model="m")
    store.append_messages_batch(session_id="s1", messages=[{"role": "user", "content": "old"}])

    store.publish_compression_child(
        parent_session_id="s1", child_session_id="s1-compressed", source="cli", model="m",
        system_prompt="summary", messages=[{"role": "user", "content": "summary of old"}],
    )

    assert store.get_compression_tip("s1") == "s1-compressed"
    assert [m["content"] for m in store.get_messages_as_conversation("s1-compressed")] == ["summary of old"]
    # The parent transcript is untouched -- publish_compression_child rotates to a new id,
    # it does not edit the old row in place.
    assert [m["content"] for m in store.get_messages_as_conversation("s1")] == ["old"]


def test_archive_and_compact_replaces_the_transcript_in_place(store):
    store.create_session(session_id="s1", source="cli", model="m")
    store.append_messages_batch(session_id="s1", messages=[
        {"role": "user", "content": "one"}, {"role": "assistant", "content": "two"},
    ])

    store.archive_and_compact("s1", [{"role": "assistant", "content": "[summary]"}])

    assert [m["content"] for m in store.get_messages_as_conversation("s1")] == ["[summary]"]


# -- a conversation survives being reloaded from a fresh store instance --------------

def test_a_disk_backed_conversation_survives_a_fresh_store_instance(tmp_path):
    """The behavioural point of SqliteSessionStore: nothing here is held in the first
    instance's Python objects -- a brand new instance pointed at the same file sees
    everything the first one wrote, as a fresh process (or a resumed session) would."""
    db_path = tmp_path / "sessions.db"

    first = SqliteSessionStore(db_path)
    first.create_session(session_id="s1", source="cli", model="m")
    first.append_messages_batch(session_id="s1", messages=[
        {"role": "user", "content": "hola"}, {"role": "assistant", "content": "hi there"},
    ])
    first.set_session_title("s1", "Greeting")
    first.close()

    second = SqliteSessionStore(db_path)

    assert second.get_session_title("s1") == "Greeting"
    assert [m["content"] for m in second.get_messages_as_conversation("s1")] == ["hola", "hi there"]


# -- a host-supplied store needs no inheritance ---------------------------------------

class HostOwnedStore:
    """A minimal stand-in for a host's own backend (e.g. a thin wrapper over Postgres).

    Deliberately does not import anything from ``hermes_core.seams.session_store`` --
    the point is that satisfying ``SessionStore`` is a matter of having the right
    methods, not of inheriting from anything this package ships.
    """

    def __init__(self):
        self.sessions = {}
        self.messages = {}

    def create_session(self, *, session_id, **_fields):
        self.sessions.setdefault(session_id, {"system_prompt": None, "title": None, "ended_at": None})

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def end_session(self, session_id, reason):
        if session_id in self.sessions:
            self.sessions[session_id]["ended_at"] = reason

    def append_messages_batch(self, *, session_id, messages, **_lease_kwargs):
        self.messages.setdefault(session_id, []).extend(messages)

    def get_messages_as_conversation(self, session_id, *, repair_alternation=True, include_row_ids=False):
        return [{"role": m["role"], "content": m["content"]} for m in self.messages.get(session_id, [])]

    def get_session_title(self, session_id):
        row = self.sessions.get(session_id)
        return row["title"] if row else None

    def set_session_title(self, session_id, title):
        self.sessions[session_id]["title"] = title

    def get_session_title_source(self, session_id):
        return None

    def set_session_title_source(self, session_id, source):
        pass

    def update_system_prompt(self, session_id, system_prompt):
        self.sessions[session_id]["system_prompt"] = system_prompt

    def queue_token_counts(self, session_id, **counts):
        pass

    def flush_token_counts(self):
        pass

    def get_compression_tip(self, session_id):
        return session_id

    def get_active_message_watermark(self, session_id):
        return len(self.messages.get(session_id, []))

    def publish_compression_child(self, **_kwargs):
        pass

    def archive_and_compact(self, session_id, messages, **_extra):
        self.messages[session_id] = list(messages)


def test_a_host_owned_store_satisfies_the_protocol_without_inheriting_anything():
    host_store = HostOwnedStore()

    assert isinstance(host_store, SessionStore)

    host_store.create_session(session_id="s1", source="cli", model="m")
    host_store.append_messages_batch(session_id="s1", messages=[{"role": "user", "content": "hi"}])

    assert [m["content"] for m in host_store.get_messages_as_conversation("s1")] == ["hi"]


def test_an_agent_accepts_a_host_owned_store():
    from hermes_core.run_agent import AIAgent

    host_store = HostOwnedStore()
    agent = AIAgent(
        api_key="sk-test", base_url="https://example.invalid/v1", provider="openai", model="fake-model",
        enabled_toolsets=[], quiet_mode=True, session_db=host_store,
    )

    assert agent._session_db is host_store


# -- end to end: conversation memory through a real turn ------------------------------

def build_agent(**overrides):
    from hermes_core.run_agent import AIAgent

    settings = dict(
        api_key="sk-test", base_url="https://example.invalid/v1", provider="openai", model="fake-model",
        enabled_toolsets=[], quiet_mode=True, max_iterations=5,
    )
    settings.update(overrides)
    return AIAgent(**settings)


def test_an_agent_gets_working_memory_with_no_configuration():
    """The default-wiring claim: an agent built with no ``session_db`` at all still
    gets a real SessionStore (see the ``agent_init`` patch in tools/lift.py) instead of
    persistence silently no-op'ing on every call site guarded by ``if not
    agent._session_db``."""
    agent = build_agent()

    assert isinstance(agent._session_db, InMemorySessionStore)


def test_a_second_agent_on_the_same_store_sees_the_first_agents_turn():
    """Two independent ``AIAgent`` instances sharing one store -- the shape of a
    process restart, or a gateway that builds a fresh agent per request but keeps the
    store alive. The second agent is handed conversation_history loaded straight from
    the store, exactly as a resumed session would build it from ``state.db``."""
    shared_store = InMemorySessionStore()
    session_id = "s1"

    first_agent = build_agent(session_db=shared_store, session_id=session_id)
    install_fake_client(first_agent, Script().text("Hola, soy el core."))
    first_result = first_agent.run_conversation("hola")
    assert first_result["completed"] is True

    # Nothing here reaches into the first agent's Python objects -- only what the
    # store durably recorded during that turn.
    history = shared_store.get_messages_as_conversation(session_id)
    assert [m["role"] for m in history] == ["user", "assistant"]

    second_agent = build_agent(session_db=shared_store, session_id=session_id)
    second_client = install_fake_client(second_agent, Script().text("Siempre a las órdenes."))
    second_result = second_agent.run_conversation(
        "¿te acordás de lo que dijiste antes?", conversation_history=history,
    )

    assert second_result["completed"] is True
    sent_roles = [m["role"] for m in second_client.last_request.messages]
    # The prior turn's user/assistant pair reached the model ahead of the new question --
    # this is the whole claim: memory survives the agent object being thrown away.
    assert sent_roles == ["system", "user", "assistant", "user"]
    assert second_client.last_request.messages[1]["content"] == "hola"
    assert second_client.last_request.messages[2]["content"] == "Hola, soy el core."


def test_the_second_agents_own_turn_is_appended_after_the_first(tmp_path):
    """Persistence is additive across agent instances: the store ends up holding both
    turns, in order, not just the second agent's."""
    db_path = tmp_path / "sessions.db"
    session_id = "s1"

    first_agent = build_agent(session_db=SqliteSessionStore(db_path), session_id=session_id)
    install_fake_client(first_agent, Script().text("Hola, soy el core."))
    first_agent.run_conversation("hola")

    reopened_store = SqliteSessionStore(db_path)
    history = reopened_store.get_messages_as_conversation(session_id)

    second_agent = build_agent(session_db=reopened_store, session_id=session_id)
    install_fake_client(second_agent, Script().text("Todo bien."))
    second_agent.run_conversation("¿cómo va?", conversation_history=history)

    final_history = reopened_store.get_messages_as_conversation(session_id)
    assert [m["role"] for m in final_history] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in final_history] == [
        "hola", "Hola, soy el core.", "¿cómo va?", "Todo bien.",
    ]
