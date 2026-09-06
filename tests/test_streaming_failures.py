"""What happens when the LLM response stream dies part-way through.

This core already shipped one bug of exactly this class, silently: the ``relay_llm``
seam returned the provider SDK's raw stream where upstream returns its own wrapper
(see the module docstring of ``hermes_core/seams/relay_llm.py``). Reading
``stream.final_response`` on the raw object raised ``AttributeError`` mid-response,
was classified as a stream that died mid tool-call, and the complete tool call the
model had already sent was silently dropped -- while ``run_conversation()`` returned
``completed: True``. That bug was found against a live provider, not by this suite.

Every test here drives a REAL ``AIAgent.run_conversation()`` with ``stream=True``
against a scripted OpenAI-shaped client (see ``hermes_core.testing.install_fake_client``
and ``hermes_core.testing.StreamDrop``). Nothing here mocks any part of the turn loop
itself: validation, dispatch, truncation recovery and retry are all the real code.

The question each test answers is never just "does it crash" -- a loud crash is a
manageable bug. The dangerous outcome is a turn that reports ``completed: True`` (or
silently drops a tool call) after losing part of what the model actually sent. Where a
test finds real current behaviour worth flagging as surprising (not necessarily wrong),
it is annotated in its docstring/comments rather than weakened.
"""

from __future__ import annotations

import json
import tempfile

import httpx
import pytest

from hermes_core.agent.errors import EmptyStreamError
from hermes_core.seams.config import DictConfigSource, set_config_source
from hermes_core.seams.credentials import StaticCredentials, set_credential_source
from hermes_core.seams.paths import DirectoryWorkspace, set_workspace
from hermes_core.testing import Script, StreamDrop, install_fake_client
from hermes_core.tools.registry import registry, tool_result


@pytest.fixture(autouse=True)
def isolated_core():
    """Same scratch-workspace / in-memory-config isolation as test_agent_turn.py."""
    set_workspace(DirectoryWorkspace(tempfile.mkdtemp()))
    set_config_source(
        DictConfigSource({"model": {"default": "fake-model", "provider": "openai"}})
    )
    set_credential_source(StaticCredentials("sk-test"))
    yield


@pytest.fixture
def weather_tool():
    """Register a tool and record what it is actually called with.

    Recording the *arguments* (not just that the tool ran) is the point: the
    catastrophic outcome this file hunts for is a tool executed with guessed or
    truncated arguments, not just "did it run".
    """
    received = []

    def handler(args, **_kwargs):
        received.append(args)
        return tool_result(temp_c=21, city=args.get("city"))

    registry.register(
        name="get_weather",
        toolset="demo",
        schema={
            "name": "get_weather",
            "description": "Look up the weather in a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
        handler=handler,
        override=True,
    )
    yield received
    registry.deregister("get_weather")


def build_agent(**overrides):
    from hermes_core.run_agent import AIAgent

    settings = dict(
        api_key="sk-test",
        base_url="https://example.invalid/v1",
        provider="openai",
        model="fake-model",
        enabled_toolsets=["demo"],
        quiet_mode=True,
        max_iterations=10,
    )
    settings.update(overrides)
    return AIAgent(**settings)


def tool_messages(request):
    return [m for m in request.messages if m.get("role") == "tool"]


# == 1. stream dies mid-text ==========================================================

def test_dies_mid_text_recovers_via_continuation_and_tells_the_truth():
    """Some deltas arrive, the connection drops silently (no exception -- the SSE just
    stops), and the model is asked to continue. Proves the turn does NOT report the
    partial fragment as if it were the whole answer: it stitches the continuation on
    and only THEN reports completed.
    """
    agent = build_agent(enabled_toolsets=[])
    # after_chunks=2: the role chunk + the first of 3 text pieces. finish_reason and
    # usage never arrive -- exactly a connection that drops mid-flight.
    client = install_fake_client(
        agent,
        Script()
        .text("Hacen 21 grados en Rosario y va a llover.",
              drop=StreamDrop(after_chunks=2))
        .text(" El resto llegó en el reintento."),
        stream=True,
    )

    result = agent.run_conversation("clima")

    assert client.call_count == 2, "the drop must trigger exactly one continuation call"
    assert result["completed"] is True
    # The stitched answer carries text from BOTH the dropped attempt and the
    # continuation -- never just the first (truncated) fragment reported as complete.
    assert "El resto llegó en el reintento" in result["final_response"]
    assert result["final_response"].startswith("Hacen")


def test_dies_mid_text_via_a_raised_error_also_recovers_honestly():
    """Same shape, but the drop raises instead of silently ending the SSE -- the shape
    of a real transport error (``httpx.RemoteProtocolError``) rather than a clean stop.
    Exercises ``_handle_stream_error`` / ``_partial_stream_stub`` instead of
    ``_finish_chat_stream``'s own text-only-drop branch.

    That recovery path (``chat_completion_helpers.py:_partial_stream_stub``) reads the
    partial text back off ``agent._current_streamed_assistant_text``, which is only
    populated as a side effect of a stream-delta callback successfully firing
    (``stream_delivery.py:_fire_stream_delta``, ``if delivered: ...``) -- so a
    ``stream_callback`` must be registered for the ALREADY-delivered text to survive a
    raised mid-stream error. (Without one: see
    ``test_mid_text_error_without_a_stream_callback_loses_delivered_text`` below --
    that is a real gap, not a test artifact.) Only the one piece that streamed before
    the cut ("Buenos Air", the first third of the sentence) can possibly survive; the
    rest was never delivered and legitimately comes from the retry.
    """
    agent = build_agent(enabled_toolsets=[])
    client = install_fake_client(
        agent,
        Script()
        .text("Buenos Aires está a 18 grados",
              drop=StreamDrop(after_chunks=2, error=httpx.RemoteProtocolError("peer closed connection")))
        .text(" y despejado."),
        stream=True,
    )

    result = agent.run_conversation("clima en Buenos Aires", stream_callback=lambda _text: None)

    assert result["completed"] is True
    assert "y despejado." in result["final_response"]
    assert result["final_response"].startswith("Buenos Air")


def test_mid_text_error_keeps_delivered_text_with_no_stream_callback():
    """Text that reached the client must survive a transport error even when nothing
    is listening for deltas -- which is how an embedded host normally runs.

    Upstream recorded a delta into `_current_streamed_assistant_text` only when
    delivery to a callback succeeded (`stream_delivery.py`, `if delivered: ...`). That
    holds there because something is always watching a stream. Here nothing need be: a
    host that only reads what `run_conversation()` returns registers no callback, and
    `_partial_stream_stub` recovers the partial answer from exactly that record. So a
    *raised* mid-stream error discarded every delta that had already arrived -- while
    the turn still reported `completed: True`. A silent stop was fine; a transport
    error lost the text.

    `tools/lift.py` now records unconditionally: bookkeeping the loop depends on is
    not conditional on whether anyone was listening.
    """
    """Same connection-dies-via-exception shape as the test above, but with NO
    ``stream_callback`` registered on the call -- which is exactly how every other test
    in this suite (and most of ``test_agent_turn.py``) calls ``run_conversation()``.
    Text that unambiguously reached the client before the drop ("Buenos Air") should
    still show up in the final answer. It currently does not.
    """
    agent = build_agent(enabled_toolsets=[])
    client = install_fake_client(
        agent,
        Script()
        .text("Buenos Aires está a 18 grados",
              drop=StreamDrop(after_chunks=2, error=httpx.RemoteProtocolError("peer closed connection")))
        .text(" y despejado."),
        stream=True,
    )

    result = agent.run_conversation("clima en Buenos Aires")  # no stream_callback

    assert result["completed"] is True
    assert "Buenos Ai" in result["final_response"]


def test_persistent_mid_text_drop_never_lies_about_completion():
    """The worst outcome named in this task: a truncated answer presented as complete.
    When every continuation attempt also drops, the loop must hit its ceiling (4
    continuation retries, see ``turn_truncation._continue_text``) and report an
    HONEST partial/failed result -- never ``completed: True`` over content it knows
    is incomplete.
    """
    agent = build_agent(enabled_toolsets=[])
    drop = StreamDrop(after_chunks=2)
    script = Script()
    for _ in range(4):  # initial attempt + 3 continuations = the ceiling
        script.text("palabras que nunca llegan completas", drop=drop)
    client = install_fake_client(agent, script, stream=True)

    result = agent.run_conversation("contame algo largo")

    assert client.call_count == 4
    assert result["completed"] is False
    assert result.get("partial") is True or result.get("failed") is True
    # The crucial negative assertion: never a lie dressed up as success.
    assert result["completed"] is not True


# == 2. stream dies mid tool-call: the known-catastrophic case =======================

def test_complete_tool_call_survives_a_stream_that_dies_before_finish_reason(weather_tool):
    """The model emits a COMPLETE tool call (valid JSON, all of it delivered), then the
    stream ends with no finish_reason and no usage chunk -- exactly the shape of the
    bug this task exists to catch. after_chunks=2 keeps the role chunk and the one
    tool-call delta (which already carries the full arguments) and drops the trailing
    finish/usage chunks.

    Recorded finding: the turn loop's dispatch decision (``conversation_loop.py``,
    ``run_tool_round if s.assistant_message.tool_calls else finish_text_response``) is
    driven purely by whether the assembled message HAS tool_calls, not by the
    finish_reason label attached to it. Because the arguments parsed as valid JSON,
    ``_assemble_tool_calls`` never flags them truncated, so the message keeps its
    tool_calls and the finish_reason ends up stamped "stop" instead of "tool_calls" --
    a cosmetic mislabel, but the tool call itself is neither dropped nor executed with
    guessed arguments.
    """
    agent = build_agent()
    client = install_fake_client(
        agent,
        Script()
        .calls(("get_weather", {"city": "Rosario"}), drop=StreamDrop(after_chunks=2))
        .text("Hacen 21 grados en Rosario."),
        stream=True,
    )

    result = agent.run_conversation("¿qué temperatura hace en Rosario?")

    assert result["completed"] is True
    assert result["final_response"] == "Hacen 21 grados en Rosario."
    # The tool ran EXACTLY once, with EXACTLY the arguments the model sent -- not
    # dropped, not guessed, not run twice.
    assert weather_tool == [{"city": "Rosario"}]
    assert len(tool_messages(client.last_request)) == 1


def test_stream_closes_its_connection_after_dying_mid_tool_call(weather_tool):
    """Teardown half of the same scenario: the abandoned stream's connection must be
    torn down, not leaked. ``UnmanagedLlmStream.__next__`` closes on the sentinel that
    marks a silently-ended iterator, so even a "clean" drop (no exception) must still
    result in ``close()`` reaching the raw stream.
    """
    agent = build_agent()
    client = install_fake_client(
        agent,
        Script()
        .calls(("get_weather", {"city": "Rosario"}), drop=StreamDrop(after_chunks=2))
        .text("Hacen 21 grados en Rosario."),
        stream=True,
    )

    agent.run_conversation("¿qué temperatura hace en Rosario?")

    first_stream = client.streams[0]
    assert first_stream.exhausted, "the truncated chunk list must have been fully drained"
    assert first_stream.close_calls >= 1, "the abandoned stream's connection was never closed"


# == 3. stream dies mid-arguments: truncated JSON must never execute =================

def test_truncated_tool_arguments_are_never_executed(weather_tool):
    """The stream cuts off WHILE the JSON arguments are still being written --
    ``{"city": "Ros`` never gets its closing quote/brace, and the connection then dies
    (after_chunks=2: role chunk + the one delta carrying that fragment; no
    finish_reason, no usage). This must never be dispatched with guessed/repaired
    arguments.

    Recorded finding: on THIS (streaming) path, the truncation is caught by
    ``chat_completion_helpers._assemble_tool_calls`` / ``_finish_chat_stream`` --
    ``_repair_tool_call_arguments`` cannot close an unterminated string, gives up and
    returns ``"{}"``, which is treated as unrepairable and flags
    ``has_truncated_tool_args``. The response comes back as a length-truncated stub
    with ``tool_calls=None``, so ``turn_tool_validation.validate_tool_calls`` -- which
    has its OWN truncation heuristic (the ``rstrip().endswith(("}", "]"))`` check) --
    is never even reached for this call: the earlier layer already intercepted it. That
    heuristic in ``turn_tool_validation.py`` is exercised elsewhere (e.g. a provider
    that stamps finish_reason="tool_calls" over already-truncated arguments), not on
    this connection-drop path.
    """
    agent = build_agent()
    client = install_fake_client(
        agent,
        Script()
        .calls(("get_weather", '{"city": "Ros'), drop=StreamDrop(after_chunks=2))
        .calls(("get_weather", {"city": "Rosario"}))
        .text("Hacen 21 grados en Rosario."),
        stream=True,
    )

    result = agent.run_conversation("¿qué temperatura hace en Rosario?")

    assert result["completed"] is True
    assert result["final_response"] == "Hacen 21 grados en Rosario."
    # The tool ran exactly once, and only with the COMPLETE, correct arguments from the
    # retry -- never with a guess derived from the truncated fragment.
    assert weather_tool == [{"city": "Rosario"}]
    assert client.call_count == 3


# == 4. stream ends with no finish reason and no usage (but full content) ============

def test_fully_delivered_text_with_no_terminator_is_treated_as_stalled():
    """``fake_client.py``'s own comment: the loop treats a stream that ended with no
    usage and no finish_reason as one cut off mid-flight, full stop -- it has no way to
    distinguish "every byte arrived, the provider just forgot to send the closing
    frame" from a genuine mid-flight death. after_chunks=4 keeps the role chunk plus
    all 3 text pieces (the COMPLETE answer) and only drops the terminating
    finish_reason/usage chunks.

    This is intentionally conservative, not a lie: the loop pays for an extra
    round-trip rather than ever risking presenting an unterminated stream as complete.
    """
    agent = build_agent(enabled_toolsets=[])
    full_text = "Hoy hace sol y 25 grados."
    client = install_fake_client(
        agent,
        Script()
        .text(full_text, drop=StreamDrop(after_chunks=4))
        # A genuinely empty continuation reply takes an unrelated path (the loop's
        # empty-response retry, turn_empty_response.py) -- give it a little more so
        # the assertion below is about truncation recovery, not that other guard.
        .text(" Nada más que agregar."),
        stream=True,
    )

    result = agent.run_conversation("clima")

    assert client.call_count == 2, "no finish_reason/usage forces exactly one continuation round-trip"
    assert result["completed"] is True
    assert full_text in result["final_response"]


# == 5. stream yields nothing at all ==================================================

def test_empty_stream_fails_loudly_instead_of_lying(monkeypatch):
    """Connects, then closes with not even a role chunk. This is a crash, not a lie:
    ``_finish_chat_stream``'s zero-chunk guard raises ``EmptyStreamError`` rather than
    returning any response object, so it can never be mistaken for a valid (if empty)
    completion.

    ``HERMES_STREAM_RETRIES`` and ``api_max_retries`` are pinned to their floors so the
    exact number of scripted attempts is deterministic rather than incidental.
    """
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    set_config_source(DictConfigSource({
        "model": {"default": "fake-model", "provider": "openai"},
        "agent": {"api_max_retries": 1},
    }))
    agent = build_agent(enabled_toolsets=[])
    client = install_fake_client(
        agent,
        Script().text("nunca sale", drop=StreamDrop(after_chunks=0)),
        stream=True,
    )

    result = agent.run_conversation("hola")

    assert client.call_count == 1
    assert result["completed"] is False
    assert result.get("failed") is True or result.get("error")
    assert "final_response" not in result or not result.get("completed")


# == 6. the completed-response escape hatch ==========================================

def test_provider_that_ignores_stream_true_still_answers_correctly(weather_tool):
    """Some providers answer in one piece despite ``stream=True``. ``UnmanagedLlmStream``
    exists to catch exactly this (``final_response`` / ``completed_response_predicate``)
    and hand the loop a completed response instead of trying to iterate one. Proves the
    other half of the bug that bit this core in production: the seam that used to lose
    this path is what ``relay_llm.py``'s docstring documents.
    """
    agent = build_agent()
    client = install_fake_client(
        agent,
        Script()
        .calls(("get_weather", {"city": "Rosario"}), ignore_stream=True)
        .text("Hacen 21 grados en Rosario.", ignore_stream=True),
        stream=True,
    )

    result = agent.run_conversation("¿qué temperatura hace en Rosario?")

    assert result["completed"] is True
    assert result["final_response"] == "Hacen 21 grados en Rosario."
    assert weather_tool == [{"city": "Rosario"}]
    # The escape hatch never opens a live stream at all.
    assert client.streams == []


# == 7. teardown: close() is idempotent and runs its finalizer exactly once ==========

def test_unmanaged_llm_stream_close_is_idempotent_and_finalizes_once():
    """Direct unit test of ``hermes_core/seams/relay_llm.py``'s ``UnmanagedLlmStream``
    (read-only seam -- not edited by this task). Its docstring promises ``close()`` is
    idempotent and the finalizer runs exactly once, "however it ends". Verify both by
    closing it three times: once implicitly via exhaustion, twice more explicitly, the
    way an interrupt-then-cleanup sequence in ``chat_completion_helpers.py`` does
    (``stream.close()`` on interrupt, then ``_close_managed_stream()`` in a ``finally``).
    """
    from hermes_core.seams.relay_llm import UnmanagedLlmStream

    raw_chunks = iter(["a", "b"])
    closes = []
    finalizes = []

    class _RawStream:
        def __iter__(self):
            return self

        def __next__(self):
            return next(raw_chunks)

        def close(self):
            closes.append(1)

    stream = UnmanagedLlmStream(
        {}, lambda _request: _RawStream(),
        finalizer=lambda: finalizes.append(1),
    )

    collected = list(stream)  # exhausts the iterator -> __next__'s sentinel path closes it
    assert collected == ["a", "b"]
    assert closes == [1]
    assert finalizes == [1]

    # Idempotent: further explicit closes (the interrupt path, then the `finally`
    # cleanup) must not touch the raw stream or the finalizer again.
    stream.close()
    stream.close()
    assert closes == [1]
    assert finalizes == [1]


def test_unmanaged_llm_stream_close_after_abandoning_mid_iteration():
    """A stream abandoned before exhaustion (the turn loop's own interrupt path: break
    out of the ``for chunk in stream`` loop, then call ``stream.close()`` explicitly)
    must still close the raw connection and finalize exactly once -- proving the
    "however it ends" half of the docstring's promise, not just the clean-exhaustion
    half covered above.
    """
    from hermes_core.seams.relay_llm import UnmanagedLlmStream

    raw_chunks = iter(["a", "b", "c", "d"])
    closes = []
    finalizes = []

    class _RawStream:
        def __iter__(self):
            return self

        def __next__(self):
            return next(raw_chunks)

        def close(self):
            closes.append(1)

    stream = UnmanagedLlmStream(
        {}, lambda _request: _RawStream(),
        finalizer=lambda: finalizes.append(1),
    )

    seen = []
    for chunk in stream:
        seen.append(chunk)
        if len(seen) == 2:
            break  # abandon mid-flight, the way an interrupt does
    stream.close()  # the explicit teardown call a real interrupt path makes
    stream.close()  # the `finally` cleanup pass that always runs too

    assert seen == ["a", "b"]
    assert closes == [1], "close() must reach the raw stream exactly once, not zero or many"
    assert finalizes == [1]
