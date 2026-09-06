"""Adversarial coverage for the session-state seam.

``tests/test_session_store.py`` proves the store works. This file tries to prove it
breaks, and checks three specific claims its author made in the module docstring and
handoff notes for ``hermes_core/seams/session_store.py``:

1. ``get_messages_as_conversation``'s ``repair_alternation`` flag is accepted but
   never enforced -- rows come back exactly as stored, regardless of the flag.
2. ``archive_and_compact`` / ``publish_compression_child`` / ``get_compression_tip``
   are correct for exactly one writer, not upstream's multi-process race protection.
3. A landmine: ``session_persistence.py``'s exception handler and
   ``inline_tool_executors.py``'s recall-unavailable path both do a lazy
   ``from hermes_state import ...`` -- a module this extracted core does not ship --
   reachable only when ``append_messages_batch`` raises (or recall is unavailable).

Findings for each claim are recorded next to the test that proves them. Two bugs
found *outside* ``session_store.py`` (in lifted ``hermes_core/agent/*.py`` code) are
included because they are direct consequences of how the store is shaped; they are
marked ``xfail(strict=True)`` per the ground rules -- this file must never edit
``hermes_core/*`` to "fix" them.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import threading

import pytest

from hermes_core.seams.config import DictConfigSource, set_config_source
from hermes_core.seams.credentials import StaticCredentials, set_credential_source
from hermes_core.seams.paths import DirectoryWorkspace, get_hermes_home, set_workspace
from hermes_core.seams.session_store import InMemorySessionStore, SqliteSessionStore
from hermes_core.testing import Script, install_fake_client


@pytest.fixture(autouse=True)
def isolated_core(tmp_path):
    set_workspace(DirectoryWorkspace(str(tmp_path / "workspace")))
    set_config_source(DictConfigSource({"model": {"default": "fake-model", "provider": "openai"}}))
    set_credential_source(StaticCredentials("sk-test"))
    yield


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemorySessionStore()
    return SqliteSessionStore(tmp_path / "adversarial.db")


def build_agent(**overrides):
    from hermes_core.run_agent import AIAgent

    settings = dict(
        api_key="sk-test", base_url="https://example.invalid/v1", provider="openai", model="fake-model",
        enabled_toolsets=[], quiet_mode=True, max_iterations=5,
    )
    settings.update(overrides)
    return AIAgent(**settings)


@pytest.fixture
def weather_tool():
    from hermes_core.tools.registry import registry, tool_result

    received = []

    def handler(args, **_kwargs):
        received.append(args)
        return tool_result(temp_c=21, city=args.get("city"))

    registry.register(
        name="get_weather", toolset="demo",
        schema={
            "name": "get_weather", "description": "Look up the weather in a city.",
            "parameters": {
                "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"],
            },
        },
        handler=handler, override=True,
    )
    yield received
    registry.deregister("get_weather")


# =====================================================================================
# Claim 1: "repair_alternation is accepted but not actually enforced"
#
# VERDICT: TRUE, and it is not a live bug. Upstream's hermes_state_messages.py
# (get_messages_as_conversation, repair_alternation=True) merges a durable
# ``user;user`` wedge at RESTORE time via agent.agent_runtime_helpers.
# repair_message_sequence. This seam's implementation ignores the keyword entirely.
# But hermes_core.agent.turn_iteration_prep._prepare_iteration calls
# repair_message_sequence_with_cursor before EVERY model request regardless of where
# the history came from, so the same healing still happens -- just per-request
# instead of once at restore, exactly the cost upstream's own docstring describes
# ("the defensive pre-request repair re-fires on EVERY request for the rest of the
# session's life"). The durable transcript itself, though, is never healed: a direct
# reader (export, session_search, a host's own tooling) sees the wedge forever.
# =====================================================================================

def test_repair_alternation_flag_is_completely_inert(store):
    """Direct proof of the claim: True vs False produce byte-identical output."""
    store.create_session(session_id="s1", source="cli", model="m")
    store.append_messages_batch(session_id="s1", messages=[
        {"role": "user", "content": "first ask"},
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": "unanswered turn"},
        {"role": "user", "content": "next turn"},
    ])

    with_repair = store.get_messages_as_conversation("s1", repair_alternation=True)
    without_repair = store.get_messages_as_conversation("s1", repair_alternation=False)

    assert [m["role"] for m in with_repair] == ["user", "assistant", "user", "user"]
    assert with_repair == without_repair


def test_the_durable_wedge_survives_forever_but_the_turn_loop_heals_it_pre_request(store):
    """The claim's practical consequence, proven end to end with a real agent turn."""
    store.create_session(session_id="s1", source="cli", model="m")
    store.append_messages_batch(session_id="s1", messages=[
        {"role": "user", "content": "first ask"},
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": "unanswered turn"},
        {"role": "user", "content": "next turn"},
    ])
    history = store.get_messages_as_conversation("s1", repair_alternation=True)
    assert [m["role"] for m in history] == ["user", "assistant", "user", "user"]  # still wedged

    agent = build_agent(session_db=store, session_id="s1")
    client = install_fake_client(agent, Script().text("ok"))

    result = agent.run_conversation("new question", conversation_history=history)

    assert result["completed"] is True
    sent_roles = [m["role"] for m in client.last_request.messages]
    assert sent_roles == ["system", "user", "assistant", "user"]  # healed before the model saw it

    merged_content = client.last_request.messages[-1]["content"]
    assert "unanswered turn" in merged_content
    assert "next turn" in merged_content
    assert "new question" in merged_content
    assert (
        merged_content.index("unanswered turn")
        < merged_content.index("next turn")
        < merged_content.index("new question")
    )


# =====================================================================================
# Claim 2: "correct for one writer, not upstream's multi-process race protection"
#
# VERDICT: the single-writer path holds up well under real thread concurrency (the
# shared RLock does its job for both shipped stores). But the lifted core DOES carry
# one hidden assumption that this store's deliberate omission of the optional
# compression-lock API breaks silently: see the xfail below.
# =====================================================================================

def test_concurrent_appends_from_many_threads_lose_no_messages(store):
    store.create_session(session_id="s1", source="cli", model="m")
    n_threads, n_msgs = 20, 50

    def worker(t):
        batch = [{"role": "user", "content": f"t{t}-m{i}"} for i in range(n_msgs)]
        store.append_messages_batch(session_id="s1", messages=batch)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    contents = [m["content"] for m in store.get_messages_as_conversation("s1")]
    expected = {f"t{t}-m{i}" for t in range(n_threads) for i in range(n_msgs)}
    assert len(contents) == n_threads * n_msgs
    assert set(contents) == expected
    # A single append_messages_batch call is one lock-held unit -- within one thread's
    # own batch, order must be exactly preserved even though threads interleave.
    for t in range(n_threads):
        per_thread = [c for c in contents if c.startswith(f"t{t}-")]
        assert per_thread == [f"t{t}-m{i}" for i in range(n_msgs)]


def test_concurrent_queue_token_counts_accumulates_exactly(store):
    """Read-modify-write under contention: a lost update would undercount tokens."""
    store.create_session(session_id="s1", source="cli", model="m")
    n_threads, per_thread = 25, 40

    def worker():
        for _ in range(per_thread):
            store.queue_token_counts("s1", input_tokens=1)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert store.get_session("s1")["token_counts"]["input_tokens"] == n_threads * per_thread


def test_concurrent_archive_and_compact_calls_are_serialized_not_interleaved(store):
    """Two 'compactions' racing on one session must each see a whole transcript, never
    a half-written one -- proof the RLock actually serializes _replace_rows."""
    store.create_session(session_id="s1", source="cli", model="m")
    store.append_messages_batch(session_id="s1", messages=[{"role": "user", "content": "seed"}])

    def compact(tag, n):
        store.archive_and_compact("s1", [{"role": "assistant", "content": f"{tag}-{i}"} for i in range(n)])

    threads = [threading.Thread(target=compact, args=(f"w{t}", 30)) for t in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    final = [m["content"] for m in store.get_messages_as_conversation("s1")]
    # Whichever thread wrote last, its FULL batch must be intact -- never a mix of
    # two threads' rows (that would mean _replace_rows was not atomic under the lock).
    tags = {c.rsplit("-", 1)[0] for c in final}
    assert len(tags) == 1
    assert len(final) == 30


def test_get_compression_tip_terminates_on_a_cyclic_chain(store):
    """A malformed compression chain should never happen through the public API, but
    a corrupted row could produce one -- get_compression_tip must not hang."""
    store.create_session(session_id="s1", source="cli", model="m")
    store.publish_compression_child(parent_session_id="s1", child_session_id="s2", source="cli", model="m")
    row = store.get_session("s2")
    row["compression_child"] = "s1"  # manufacture a cycle
    store._write_session("s2", row)

    tip = store.get_compression_tip("s1")  # must return promptly, not loop forever

    assert tip in ("s1", "s2")


def test_an_absent_optional_lock_method_reads_as_no_lock_api_not_as_a_failure():
    """Automatic context compaction used to sit out on every cycle, on both stores.

    `_lock_api_is_absent_on_session_db` asked "is this the old pre-locks SessionDB
    class?" by importing it. The import raised, a blanket `except Exception: return
    False` reported "the lock API is NOT absent", `_resolve_lock_api` then touched the
    attribute, got AttributeError, and classified that as a lookup FAILURE rather than
    an absent API -- and `_acquire_compression_lease` treats any lookup failure as
    unsafe and sits out.

    `try_acquire_compression_lock` is an OPTIONAL part of the SessionStore protocol
    that neither shipped store implements, so "absent" is the normal case, not an
    error. tools/lift.py now patches the check to ask the type directly, the way
    turn_facade_lease.py already asks about its own optional methods.
    """
    from hermes_core.agent.conversation_compression import _resolve_lock_api

    plain_store = InMemorySessionStore()
    try_acquire, lookup_error = _resolve_lock_api(plain_store)

    # Desired: an absent optional method reads as "(None, None)" -- no lock API,
    # proceed unlocked -- exactly like every other optional method on this Protocol.
    assert try_acquire is None
    assert lookup_error is None


# =====================================================================================
# Claim 3: the `from hermes_state import ...` landmine
#
# VERDICT: worse than "only fires when append_messages_batch raises". It fires
# exactly there (confirmed), AND separately on the recall-unavailable path. The
# blast radius differs wildly by call site:
#   * turn_finalizer.py's end-of-turn persist wraps _persist_session in a broad
#     except-and-stringify (_guarded_cleanup) -- the crash is swallowed into
#     result["cleanup_errors"], the turn reports completed=True, but the turn's
#     messages are PERMANENTLY LOST from the store (proven below).
#   * every other call site (turn_recovery.py, turn_tool_round.py, tool_executor.py,
#     turn_iteration_prep.py, ...) calls agent._persist_session /
#     agent._flush_messages_to_session_db with NO such guard -- there the
#     ModuleNotFoundError propagates all the way out of run_conversation() and
#     crashes the caller's process. Reproduced directly below without needing to
#     drive the (slow) retry-exhaustion path that triggers it in practice.
# =====================================================================================

class BoomStore(InMemorySessionStore):
    """A store whose append_messages_batch always fails, like a full disk or a
    locked file would -- the exact trigger condition named in the task brief."""

    def append_messages_batch(self, **kwargs):
        raise RuntimeError("disk full (simulated)")


def test_a_persistence_write_failure_is_classified_rather_than_crashing_the_caller():
    """A failing store must not take the conversation down with it.

    `session_persistence.py` classifies a write failure by reaching for upstream's
    `hermes_state`, which this core replaced wholesale and never provided a stand-in
    for. So the classification -- code that only runs once something has already gone
    wrong -- raised ModuleNotFoundError of its own, replacing the real error and
    propagating out of `run_conversation()` from every call site that lacks a guard.

    `hermes_core/seams/session_state.py` now supplies the names these degrade paths
    reach for."""
    agent = build_agent(session_db=BoomStore(), session_id="s1")
    agent._ensure_db_session()

    # The exact call a dozen turn_*.py call sites make after a tool round or a
    # recovered error. It must not raise.
    agent._persist_session([{"role": "user", "content": "hola"}], None)


def test_the_recall_tool_reports_unavailable_rather_than_crashing():
    """A second, unrelated trigger for the same missing module -- worth its own test.

    `session_search` with no store reaches for `format_session_db_unavailable` to tell
    the model recall is unavailable, which is a message the model can react to. Before
    the seam existed it crashed instead, on a path that has nothing to do with a failed
    write."""
    from hermes_core.agent.inline_tool_executors import _session_search

    agent = build_agent(session_db=None, session_id="s1")
    agent._persist_disabled = True  # the exact condition that makes recall unavailable

    _session_search(agent, {"query": "anything"}, None)


def test_a_persistence_failure_during_a_normal_turn_does_not_silently_lose_the_turn():
    """The highest-severity finding in this file, checked from the other side.

    A store that refused the write used to lose the exchange completely and say nothing:
    `run_conversation` returned `completed: True`, the user saw the model's answer, and
    the messages were gone from anywhere durable. The only trace was
    `result["cleanup_errors"]`, an internal field hosts do not surface.

    Two things had to be true for that. The classification path raised its own
    ModuleNotFoundError before upstream's JSONL safety net could run -- fixed by the
    session_state seam. And that safety net only ran for two specific SQLite errors,
    which made sense when the store was always upstream's own SQLite; here the store is
    whatever the host installed and can fail any way it likes. tools/lift.py now
    diverts on any failure that is not about to be retried.

    So the turn still completes -- a memory write failing must not cost the user their
    answer -- but the exchange is on disk rather than gone.
    """
    store = BoomStore()
    agent = build_agent(session_db=store, session_id="s1")
    install_fake_client(agent, Script().text("hola respuesta"))

    result = agent.run_conversation("hola")

    assert result["completed"] is True

    # The failure is classified rather than escaping. `cleanup_errors` stays empty on
    # purpose: it collects exceptions that got away, and this one no longer does. The
    # cause is recorded where the turn-end explanation reads it, so a host can tell a
    # full disk apart from lock contention.
    assert agent._last_persistence_error_cause == "disk"

    # The store genuinely refused everything, so nothing is there.
    assert store.get_messages_as_conversation("s1") == []

    # The exchange survives on disk instead. Both halves: losing only the model's
    # reply would be just as broken as losing both.
    transcript = get_hermes_home() / "sessions" / "s1.jsonl"
    assert transcript.exists(), "the JSONL fallback did not run; the turn was lost"
    written = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines() if line.strip()]
    contents = [m.get("content") for m in written]
    assert "hola" in contents and "hola respuesta" in contents, contents


# =====================================================================================
# Store mechanics: unknown ids, empty/huge batches, hostile content, sqlite edges
# =====================================================================================

def test_every_read_method_is_safe_on_an_unknown_session_id(store):
    assert store.get_session("ghost") is None
    assert store.get_messages_as_conversation("ghost") == []
    assert store.get_session_title("ghost") is None
    assert store.get_session_title_source("ghost") is None
    assert store.get_active_message_watermark("ghost") == 0
    assert store.get_compression_tip("ghost") == "ghost"
    store.end_session("ghost", "whatever")  # must not raise


def test_writes_to_an_unknown_session_id_silently_no_op(store):
    """Not specified on the Protocol either way -- pinned here so a future change to
    this behaviour is a deliberate decision, not an accident. A host that expects
    set_session_title to signal a missing session will not get one: the write is
    dropped with no exception and no created row."""
    store.set_session_title("ghost", "title")
    store.set_session_title_source("ghost", "auto")
    store.update_system_prompt("ghost", "prompt")
    store.queue_token_counts("ghost", input_tokens=5)

    assert store.get_session("ghost") is None


def test_append_messages_batch_with_an_empty_list_is_a_noop(store):
    store.create_session(session_id="s1", source="cli", model="m")
    store.append_messages_batch(session_id="s1", messages=[])
    assert store.get_messages_as_conversation("s1") == []


def test_a_large_batch_preserves_order(store):
    store.create_session(session_id="s1", source="cli", model="m")
    n = 3000
    store.append_messages_batch(session_id="s1", messages=[
        {"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(n)
    ])
    history = store.get_messages_as_conversation("s1")
    assert [m["content"] for m in history] == [str(i) for i in range(n)]


def test_unicode_and_embedded_nul_bytes_round_trip(store):
    store.create_session(session_id="s1", source="cli", model="m")
    tricky = "emoji \U0001F600, RTL مرحبا, embedded NUL: \x00 end"
    store.append_messages_batch(session_id="s1", messages=[{"role": "user", "content": tricky}])

    assert store.get_messages_as_conversation("s1")[0]["content"] == tricky


def test_archive_and_compact_on_a_session_that_was_never_created(store):
    """archive_and_compact has no create_session precondition, matching
    append_messages_batch's documented lazy-row behaviour."""
    store.archive_and_compact("orphan", [{"role": "assistant", "content": "[summary]"}])

    assert [m["content"] for m in store.get_messages_as_conversation("orphan")] == ["[summary]"]
    assert store.get_session("orphan") is None  # still no session row


def test_end_session_before_create_is_lost_not_queued(store):
    """'First reason wins' only applies to calls made against an EXISTING row. A call
    that arrives before create_session is a plain no-op, not a queued intent -- the
    session ends up NOT ended, with the second reason, once created."""
    store.end_session("s1", "first")  # no row yet: no-op
    store.create_session(session_id="s1", source="cli", model="m")
    store.end_session("s1", "second")  # first call against a real row: wins

    row = store.get_session("s1")
    assert row["ended_at"] is not None
    assert row["end_reason"] == "second"


def test_end_session_twice_on_an_unknown_session_stays_a_noop(store):
    store.end_session("ghost", "first")
    store.end_session("ghost", "second")
    assert store.get_session("ghost") is None


def test_flush_token_counts_before_any_queue_is_a_safe_noop(store):
    store.flush_token_counts()  # nothing queued, nothing created -- must not raise
    store.create_session(session_id="s1", source="cli", model="m")
    store.flush_token_counts()
    assert store.get_session("s1")["token_counts"] == {}


def test_archive_and_compact_with_no_messages_clears_the_transcript(store):
    store.create_session(session_id="s1", source="cli", model="m")
    store.append_messages_batch(session_id="s1", messages=[{"role": "user", "content": "one"}])

    store.archive_and_compact("s1", [])

    assert store.get_messages_as_conversation("s1") == []


def test_in_memory_store_tolerates_non_json_serializable_content():
    """Protocol interchangeability is more fragile than it looks: InMemorySessionStore
    happily stores an arbitrary Python object as content (nothing serializes it)."""
    store = InMemorySessionStore()
    store.create_session(session_id="s1", source="cli", model="m")
    store.append_messages_batch(session_id="s1", messages=[{"role": "user", "content": {1, 2, 3}}])

    assert store.get_messages_as_conversation("s1")[0]["content"] == {1, 2, 3}


def test_sqlite_store_raises_on_non_json_serializable_content(tmp_path):
    """... while SqliteSessionStore requires JSON-serializable content, because each
    row is one JSON blob. A code path exercised only against InMemorySessionStore in
    a host's own tests can break the moment it is pointed at SqliteSessionStore --
    the Protocol does not (and structurally cannot) guarantee this away."""
    store = SqliteSessionStore(tmp_path / "s.db")
    store.create_session(session_id="s1", source="cli", model="m")

    with pytest.raises(TypeError):
        store.append_messages_batch(session_id="s1", messages=[{"role": "user", "content": {1, 2, 3}}])


def test_sqlite_store_creates_missing_parent_directories(tmp_path):
    nested = tmp_path / "a" / "b" / "c" / "sessions.db"
    store = SqliteSessionStore(nested)
    store.create_session(session_id="s1", source="cli", model="m")

    assert nested.exists()
    assert store.get_session("s1") is not None


def test_sqlite_store_on_a_corrupt_file_fails_loudly_not_silently(tmp_path):
    corrupt = tmp_path / "corrupt.db"
    corrupt.write_text("not a sqlite database, just garbage bytes" * 20)

    with pytest.raises(sqlite3.DatabaseError):
        SqliteSessionStore(corrupt)


# =====================================================================================
# End to end, adversarially: does memory survive with tool-call linkage intact?
# =====================================================================================

def test_assistant_message_with_tool_calls_and_no_text_round_trips_content_as_none(store):
    """An assistant turn that is ALL tool calls, no text -- content must stay exactly
    None, not silently become '' or a missing key, and the nested tool_calls shape
    (including a non-trivial function/arguments payload) must survive untouched."""
    store.create_session(session_id="s1", source="cli", model="m")
    store.append_messages_batch(session_id="s1", messages=[
        {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Rosario"}'}}],
        },
        {"role": "tool", "content": "21", "tool_name": "get_weather", "tool_call_id": "call_1"},
    ])

    history = store.get_messages_as_conversation("s1")

    assert history[0]["content"] is None
    assert history[0]["tool_calls"] == [
        {"id": "call_1", "type": "function",
         "function": {"name": "get_weather", "arguments": '{"city": "Rosario"}'}},
    ]
    assert history[1]["tool_call_id"] == "call_1"


def test_tool_call_linkage_survives_the_round_trip_between_two_agents(weather_tool):
    """The sharpest version of the memory claim: not just that text round-trips, but
    that a tool-call/tool-result pair keeps EXACTLY the id linkage a real provider
    demands. A store that renumbers or drops tool_call_id produces a conversation a
    provider will reject with a 400 the moment it is resumed -- this proves it does
    not, through a real agent turn on each side, not a hand-built message list."""
    shared_store = InMemorySessionStore()
    session_id = "s1"

    first_agent = build_agent(session_db=shared_store, session_id=session_id, enabled_toolsets=["demo"])
    install_fake_client(
        first_agent,
        Script().calls(("get_weather", {"city": "Rosario"})).text("Hacen 21 grados."),
    )
    first_result = first_agent.run_conversation("¿qué temperatura hace en Rosario?")
    assert first_result["completed"] is True

    history = shared_store.get_messages_as_conversation(session_id)
    assert [m["role"] for m in history] == ["user", "assistant", "tool", "assistant"]

    assistant_call_msg, tool_result_msg = history[1], history[2]
    assert assistant_call_msg["tool_calls"][0]["id"] == tool_result_msg["tool_call_id"]

    second_agent = build_agent(session_db=shared_store, session_id=session_id, enabled_toolsets=["demo"])
    second_client = install_fake_client(second_agent, Script().text("Todo bien."))
    second_result = second_agent.run_conversation("¿lo recordás?", conversation_history=history)

    assert second_result["completed"] is True
    sent = second_client.last_request.messages
    sent_assistant = next(m for m in sent if m.get("role") == "assistant" and m.get("tool_calls"))
    sent_tool = next(m for m in sent if m.get("role") == "tool")
    # The exact linkage a provider validates: the tool message's tool_call_id must
    # match an id in the immediately preceding assistant message's tool_calls.
    assert sent_tool["tool_call_id"] == sent_assistant["tool_calls"][0]["id"]
    assert sent_tool["name"] == "get_weather"


def test_multiple_sequential_tool_rounds_all_keep_correct_linkage_after_reload(tmp_path):
    """Two SEPARATE tool rounds in one turn, reloaded from a fresh SqliteSessionStore
    instance (a real process restart) -- every tool result must still point at its
    own assistant call, not the other round's."""
    from hermes_core.tools.registry import registry, tool_result

    def make(name):
        def handler(args, **_kwargs):
            return tool_result(ok=name)
        return handler

    for name in ("alpha", "beta"):
        registry.register(
            name=name, toolset="demo",
            schema={"name": name, "description": "x", "parameters": {"type": "object", "properties": {}}},
            handler=make(name), override=True,
        )
    try:
        db_path = tmp_path / "sessions.db"
        first_agent = build_agent(
            session_db=SqliteSessionStore(db_path), session_id="s1", enabled_toolsets=["demo"],
        )
        install_fake_client(
            first_agent,
            Script().calls(("alpha", {})).calls(("beta", {})).text("listo"),
        )
        result = first_agent.run_conversation("hacé alpha y despues beta")
        assert result["completed"] is True

        reopened = SqliteSessionStore(db_path)
        history = reopened.get_messages_as_conversation("s1")

        assistant_calls = [m for m in history if m.get("role") == "assistant" and m.get("tool_calls")]
        tool_results = [m for m in history if m.get("role") == "tool"]
        assert len(assistant_calls) == 2
        assert len(tool_results) == 2

        # Each tool result's call id must appear in ITS OWN preceding assistant
        # message, not the other round's -- a shared/renumbered id would still pass a
        # weaker "the id exists somewhere" check but break provider validation.
        call_ids = [msg["tool_calls"][0]["id"] for msg in assistant_calls]
        assert len(set(call_ids)) == 2  # no accidental id collision across rounds
        for assistant_msg, tool_msg in zip(assistant_calls, tool_results):
            assert assistant_msg["tool_calls"][0]["id"] == tool_msg["tool_call_id"]
    finally:
        registry.deregister("alpha")
        registry.deregister("beta")
