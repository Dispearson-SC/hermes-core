"""Degenerate-model behaviour, driven through a real ``AIAgent.run_conversation()``.

Companion to ``test_agent_turn.py`` (the happy-path contract) and
``test_tool_call_validation.py`` (hallucinated names / malformed arguments). This file
covers the three loop guards that are supposed to catch a model that does not error but
also does not behave: empty responses, repeated tool calls, and iteration-budget
exhaustion -- plus the truncation-repetition detector in ``repetition_guard.py`` and the
tool-calls-plus-text combined message.

For every case the question is not just "does it stop" but "what does the caller see
when it does" -- ``completed``, ``turn_exit_reason``, ``api_calls`` and the transcript in
``messages`` all have to agree with what actually happened, or a host relying on
``completed`` alone gets misled.

Every script is bounded to the exact number of turns the guard under test should
consume. ``ScriptExhausted`` (raised by ``hermes_core.testing.Script`` when the loop asks
for one more turn than scripted) is what surfaces a missing/weaker-than-expected guard --
never a hang, per the run instructions for this file.
"""

from __future__ import annotations

import tempfile

import pytest

from hermes_core.agent.transports.types import NormalizedResponse, Usage
from hermes_core.seams.config import DictConfigSource, set_config_source
from hermes_core.seams.credentials import StaticCredentials, set_credential_source
from hermes_core.seams.paths import DirectoryWorkspace, set_workspace
from hermes_core.testing import Script, install_fake_client
from hermes_core.tools.registry import registry, tool_result


@pytest.fixture(autouse=True)
def isolated_core():
    """Same scratch-workspace isolation as ``test_agent_turn.py``."""
    set_workspace(DirectoryWorkspace(tempfile.mkdtemp()))
    set_config_source(
        DictConfigSource({"model": {"default": "fake-model", "provider": "openai"}})
    )
    set_credential_source(StaticCredentials("sk-test"))
    yield


@pytest.fixture(autouse=True)
def instant_backoff(monkeypatch):
    """Every empty-response / invalid-response retry waits via ``jittered_backoff``
    (5-60s, real wall-clock, slept in 200ms slices by ``interruptible_backoff_sleep``).
    Zero it so the guard tests exercise the *counting*, not the clock -- this is a
    local monkeypatch of a lazily-imported name; nothing under ``hermes_core/`` is
    touched.
    """
    monkeypatch.setattr("hermes_core.agent.retry_utils.jittered_backoff", lambda *a, **k: 0.0)


@pytest.fixture
def weather_tool():
    """A deterministic, idempotent-in-effect tool: same args -> same JSON result.

    Registered under a toolset the agent must opt into (``demo``), same as
    ``test_agent_turn.py`` -- nothing is offered unless a host asks for it.
    """
    received = []

    def handler(args, **_kwargs):
        received.append(dict(args))
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
        max_iterations=5,
    )
    settings.update(overrides)
    return AIAgent(**settings)


def tool_messages(messages):
    return [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]


def empty_turn(finish_reason: str = "stop") -> NormalizedResponse:
    """A model turn with no content, no tool calls, no reasoning -- the plain
    "unsignaled empty" the module docstring in ``empty_response_guard.py`` describes
    (as opposed to a signaled refusal like ``content_filter``)."""
    return NormalizedResponse(content="", tool_calls=None, finish_reason=finish_reason, usage=Usage())


# == 1-2. Empty response: single retry, and the deterministic-empty cap ==============


def test_single_empty_response_retries_then_the_next_turn_is_delivered():
    """One empty reply is just retried -- same request, asked again."""
    agent = build_agent(enabled_toolsets=[])
    client = install_fake_client(agent, Script().text("").text("Hola, soy el core."))

    result = agent.run_conversation("hola")

    assert client.call_count == 2
    assert result["completed"] is True
    assert result["final_response"] == "Hola, soy el core."


def test_two_consecutive_identical_empties_short_circuit_before_the_third_call():
    """The deterministic-empty guard (``empty_response_guard.deterministic_empty``): two
    consecutive empties with the same (model, provider, finish_reason) and no usage/
    generation evidence are treated as deterministic and skip the REMAINING retries --
    it does not wait for the nominal 3-retry budget. The script only defines 2 turns; if
    the guard didn't fire here, the loop would ask for a 3rd and ``ScriptExhausted``
    (an ``AssertionError``) would fail this test instead of hanging.
    """
    agent = build_agent(enabled_toolsets=[])
    client = install_fake_client(agent, Script().text("").text(""))

    result = agent.run_conversation("hola")

    assert client.call_count == 2, "deterministic-empty should cap at 2 calls, not the nominal 3-retry budget"
    assert result["turn_exit_reason"] == "empty_response_exhausted"
    # completed=True even though no real answer was produced: the caller must read
    # turn_exit_reason, not just `completed`, to detect this. The default turn-completion
    # explainer (display.turn_completion_explainer, on unless disabled) replaces the bare
    # "(empty)" sentinel with an explanation -- proven here so a regression that disables
    # it by default would be caught.
    assert result["completed"] is True
    assert result["final_response"] != "(empty)"
    assert "No reply" in result["final_response"]
    assert "empty content after retries" in result["final_response"]


@pytest.mark.parametrize("content", ["", None, "   "], ids=["empty-string", "none", "whitespace-only"])
def test_empty_none_and_whitespace_content_are_the_same_failure_mode(content):
    """``""``, ``None`` and whitespace-only content must not take different paths --
    ``_has_content_after_think_block`` strips before checking truthiness, so all three
    should hit the exact same deterministic-empty cap at 2 calls."""
    agent = build_agent(enabled_toolsets=[])
    turn = NormalizedResponse(content=content, tool_calls=None, finish_reason="stop", usage=Usage())
    client = install_fake_client(agent, Script().raw(turn).raw(turn))

    result = agent.run_conversation("hola")

    assert client.call_count == 2
    assert result["turn_exit_reason"] == "empty_response_exhausted"


def test_non_deterministic_empty_streak_uses_the_full_three_retry_budget():
    """Without a repeated identical (model, provider, finish_reason) signature, the
    deterministic short-circuit never engages and the loop uses the full nominal budget:
    3 retries after the first empty = 4 total calls. Proves the "3" in the module
    docstring is real, and that it is a ceiling ABOVE the 2-call deterministic case, not
    the same number reached a different way."""
    agent = build_agent(enabled_toolsets=[])
    client = install_fake_client(
        agent,
        Script()
        .text("", finish_reason="stop")
        .text("", finish_reason="eos")  # different finish_reason breaks determinism
        .text("", finish_reason="stop")
        .text("Answer after the full retry budget."),
    )

    result = agent.run_conversation("hola")

    assert client.call_count == 4
    assert result["completed"] is True
    assert result["final_response"] == "Answer after the full retry budget."


def test_empty_after_a_tool_round_is_nudged_once_before_the_retry_ladder(weather_tool):
    """A different rule applies right after a tool call returns: one free nudge
    (``_post_tool_empty_retried``) that does NOT consume an empty-retry attempt, before
    falling into the same empty-retry ladder as the no-tool case. This is the scenario
    from item 6 -- "the model calls a tool, gets a result, and then says nothing"."""
    agent = build_agent()
    client = install_fake_client(
        agent,
        Script()
        .calls(("get_weather", {"city": "Rosario"}))
        .text("")  # post-tool empty -> free nudge, not a retry
        .text("Hacen 21 grados."),
    )

    result = agent.run_conversation("¿qué temperatura hace en Rosario?")

    assert client.call_count == 3
    assert result["completed"] is True
    assert result["final_response"] == "Hacen 21 grados."
    assert weather_tool == [{"city": "Rosario"}]


def test_empty_survives_the_post_tool_nudge_then_still_hits_the_deterministic_cap(weather_tool):
    """Layering check: the post-tool nudge is free, but once it's used, a second and
    third consecutive plain empty (same signature) still hits the ordinary
    deterministic-empty cap at 2 -- the free nudge does not reset that budget to zero
    forever, and it is not itself retried."""
    agent = build_agent()
    client = install_fake_client(
        agent,
        Script()
        .calls(("get_weather", {"city": "Rosario"}))
        .text("")  # free post-tool nudge (call 2)
        .text("")  # first counted empty-retry attempt (call 3)
        .text(""),  # second -> deterministic, short-circuits (call 4)
    )

    result = agent.run_conversation("¿qué temperatura hace en Rosario?")

    assert client.call_count == 4, "nudge (free) + 2 counted empties (deterministic cap) = 4 calls"
    assert result["turn_exit_reason"] == "empty_response_exhausted"


# == 3. Tool-call repetition: exact rule, exact threshold, and the near-miss =========


def test_an_identical_tool_loop_is_halted_even_with_no_platform_given(weather_tool):
    """The default must be the safe one, and it was not.

    Upstream reads an unset ``platform`` as "someone is at a terminal" and leaves
    ``hard_stop_enabled`` off, so an identical-tool loop ran until the iteration budget
    ended it -- measured at seven executions with no guardrail entry at all. That
    reading is right for upstream and wrong by construction here: every platform it
    counts as attended (cli, tui, desktop, acp, subagent, api_server) is a surface this
    core deliberately does not carry. An embedded agent has no console and nobody to
    interrupt it, and the host that hits this is the one that never thought about
    platform strings -- on a metered API, that is a bill.

    ``tools/lift.py`` now patches ``_is_non_interactive_platform`` to treat an unset
    platform as unattended. The opt-out survives: see the ``cli`` test below.
    """
    agent = build_agent(max_iterations=8)  # no platform given
    script = Script()
    for _ in range(7):
        script.calls(("get_weather", {"city": "Rosario"}))
    script.text("Gave up.")
    install_fake_client(agent, script)

    result = agent.run_conversation("¿qué temperatura hace en Rosario?")

    assert len(weather_tool) == 5, "the identical-call streak should halt at the fifth"
    assert result["guardrail"]["code"] == "identical_call_streak_halt"
    assert result["turn_exit_reason"] == "guardrail_halt"


def test_a_host_that_declares_an_attended_platform_keeps_the_opt_in_behaviour(weather_tool):
    """The escape hatch. A host that says a human is watching gets upstream's default
    back -- the guard notices but does not halt, leaving the person to intervene."""
    max_iterations = 6
    agent = build_agent(max_iterations=max_iterations, platform="cli")
    script = Script()
    for _ in range(max_iterations):
        script.calls(("get_weather", {"city": "Rosario"}))
    script.text("Gave up.")
    install_fake_client(agent, script)

    result = agent.run_conversation("¿qué temperatura hace en Rosario?")

    assert len(weather_tool) == max_iterations, "an attended platform runs every call"
    assert "guardrail" not in result
    assert str(result["turn_exit_reason"]).startswith("max_iterations_reached")


def test_identical_tool_call_halts_at_the_fifth_repeat_on_a_non_interactive_platform(weather_tool):
    """The rule, with numbers: on a platform NOT in the attended set (e.g. a gateway/
    WhatsApp-style ``platform``), ``non_interactive_hard_stop_enabled`` (default True)
    turns ``hard_stop_enabled`` on automatically. The identical-call streak guard then
    HALTS the turn at the Nth consecutive identical (tool, args, result) call, where N is
    ``ToolCallGuardrailConfig.no_progress_block_after`` (default 5) -- tool-agnostic, not
    limited to tools marked idempotent. A notice is appended into the tool result from
    the 3rd call onward (``STALL_GUARD_IDENTICAL_CALL_THRESHOLD``), but only the 5th
    stops the turn. The script defines exactly 5 calls; a 6th request would raise
    ``ScriptExhausted`` and fail this test if the halt didn't fire.
    """
    agent = build_agent(platform="whatsapp", max_iterations=20)
    same_call = ("get_weather", {"city": "Rosario"})
    client = install_fake_client(agent, Script().calls(same_call).calls(same_call)
                                  .calls(same_call).calls(same_call).calls(same_call))

    result = agent.run_conversation("¿qué temperatura hace en Rosario?")

    assert client.call_count == 5
    assert len(weather_tool) == 5
    assert result["completed"] is True  # the turn ended cleanly, not via the iteration budget
    assert result["turn_exit_reason"] == "guardrail_halt"
    assert "guardrail" in result
    assert result["guardrail"]["code"] == "identical_call_streak_halt"
    assert result["guardrail"]["count"] == 5
    assert result["guardrail"]["tool_name"] == "get_weather"
    assert "get_weather" in result["final_response"]

    # The 5th (last) tool call still ran and its result is in the transcript -- a
    # halted loop must not silently drop the record of a tool that executed (item 5).
    tool_msgs = tool_messages(result["messages"])
    assert len(tool_msgs) == 5
    assert '"city": "Rosario"' in tool_msgs[-1]["content"] or '"Rosario"' in tool_msgs[-1]["content"]


def test_varying_arguments_are_treated_as_progress_never_halt(weather_tool):
    """The near-miss, with the actual rule: the identical-call streak is keyed on
    (tool_name, canonical-JSON args). Different arguments on every call -- even to the
    SAME tool, even every single call -- reset the streak to 1 each time, so neither the
    notice nor the halt ever fires, no matter how many times it repeats. On the same
    non-interactive platform that halts the identical case above, this loop instead runs
    all the way to the iteration budget. A host tuning this needs to know: exact-repeat
    detection catches a stuck loop; a model that varies one field every call (a common
    thrashing pattern) is invisible to this guard.
    """
    max_iterations = 4
    agent = build_agent(platform="whatsapp", max_iterations=max_iterations)
    script = Script()
    for i in range(max_iterations):
        script.calls(("get_weather", {"city": f"City{i}"}))
    script.text("Summary after budget exhausted.")
    install_fake_client(agent, script)

    result = agent.run_conversation("check several cities")

    assert len(weather_tool) == max_iterations
    assert "guardrail" not in result
    assert result["turn_exit_reason"] == f"max_iterations_reached({max_iterations}/{max_iterations})"
    assert result["completed"] is False
    assert result["final_response"] == "Summary after budget exhausted."


# == 4-5. Iteration budget: exact stop point, exhaustion signal, and the last tool ====


def test_iteration_budget_stops_at_exactly_max_iterations_and_says_so(weather_tool):
    """``max_iterations=1``: exactly one tool-calling iteration is allowed, then the
    finalizer's budget fallback makes ONE extra toolless "summarize" call
    (``handle_max_iterations`` in ``chat_completion_helpers.py``) rather than just
    returning nothing. FINDING to document precisely: ``api_calls`` in the result counts
    only the LOOP's iterations (1), not this extra summary call -- so
    ``client.call_count`` (2) and ``result["api_calls"]`` (1) intentionally disagree, and
    a host correlating billed calls to ``api_calls`` will undercount by exactly the
    number of budget-exhausted turns.
    """
    agent = build_agent(max_iterations=1)
    client = install_fake_client(
        agent,
        Script().calls(("get_weather", {"city": "Rosario"})).text("Here is what I found."),
    )

    result = agent.run_conversation("¿qué temperatura hace en Rosario?")

    assert client.call_count == 2  # 1 tool-calling iteration + 1 forced summary call
    assert result["api_calls"] == 1  # does NOT count the summary call
    assert weather_tool == [{"city": "Rosario"}]  # the one allowed tool call still ran
    assert result["completed"] is False
    assert result["turn_exit_reason"] == "max_iterations_reached(1/1)"
    assert result["final_response"] == "Here is what I found."

    # The tool that ran before the budget cut the turn off is not lost.
    tool_msgs = tool_messages(result["messages"])
    assert len(tool_msgs) == 1


def test_iteration_budget_exhaustion_is_distinguishable_from_normal_completion(weather_tool):
    """The concrete signal a host must check: ``turn_exit_reason`` starts with
    "max_iterations_reached(N/M)" on exhaustion and with "text_response(" on a normal
    stop -- ``completed`` alone does not separate "the model finished" from "we cut it
    off", since ``completed`` can be True in both the guardrail-halt case (see above) and
    -- via the pending-verification fallback path -- in some budget-exhaustion cases
    too. This test nails down the plain case: no verification gate, no guardrail, just
    running out of turns."""
    max_iterations = 3
    agent = build_agent(max_iterations=max_iterations, enabled_toolsets=[])
    script = Script()
    for _ in range(max_iterations):
        script.text("still thinking, let me continue", finish_reason="stop")
    # A plain "stop" text reply ends the turn normally (no tool calls to keep looping),
    # so to actually reach the budget with `enabled_toolsets=[]` we instead force tool
    # calls against a toolset the agent has -- reuse weather_tool with enabled toolsets.
    agent = build_agent(max_iterations=max_iterations)
    script = Script()
    for _ in range(max_iterations):
        script.calls(("get_weather", {"city": "Nowhere"}))
    script.text("Final summary.")
    client = install_fake_client(agent, script)

    result = agent.run_conversation("keep checking")

    assert client.call_count == max_iterations + 1
    assert result["completed"] is False
    assert result["turn_exit_reason"] == f"max_iterations_reached({max_iterations}/{max_iterations})"
    assert not str(result["turn_exit_reason"]).startswith("text_response")


# == 7. Adversarial: max_iterations 0 and 1, `length`, tool_calls + text ==============


def test_max_iterations_zero_makes_no_loop_iterations_but_still_bills_one_summary_call():
    """FINDING, sharper than the =1 case: with ``max_iterations=0`` the outer loop body
    never runs at all (the while-condition is false from the start) -- yet
    ``_resolve_budget_fallback`` still treats this as budget-exhausted-with-no-response
    and makes the SAME forced summary call. The user message is never actually shown to
    a "real" turn, only to this one throwaway summarization call, and its answer is
    delivered as ``final_response`` while ``api_calls`` reports 0. A host that gates
    on ``api_calls > 0`` to decide "did we even try" would be wrong here."""
    agent = build_agent(max_iterations=0, enabled_toolsets=[])
    client = install_fake_client(agent, Script().text("I can't help with 0 iterations."))

    result = agent.run_conversation("hola")

    assert client.call_count == 1
    assert result["api_calls"] == 0
    assert result["completed"] is False
    assert result["turn_exit_reason"] == "max_iterations_reached(0/0)"
    assert result["final_response"] == "I can't help with 0 iterations."


def test_max_iterations_one_with_no_tools_completes_normally():
    """Control case for the adversarial ``max_iterations=1``: when the model doesn't
    call a tool, one iteration is enough to finish normally -- no forced summary call,
    ``completed`` is True, exit reason is the healthy ``text_response(...)``. Confirms
    the forced-summary path in the tests above is specifically about running OUT of
    budget, not a tax on every 1-iteration turn."""
    agent = build_agent(max_iterations=1, enabled_toolsets=[])
    client = install_fake_client(agent, Script().text("One and done."))

    result = agent.run_conversation("hola")

    assert client.call_count == 1
    assert result["api_calls"] == 1
    assert result["completed"] is True
    assert str(result["turn_exit_reason"]).startswith("text_response(")
    assert result["final_response"] == "One and done."


def test_truncated_finish_reason_length_is_not_reported_as_completed():
    """``finish_reason=="length"`` (provider token cap) with no tool calls goes through
    ``turn_truncation.recover_from_truncation`` -> ``_continue_text``, which appends a
    continuation nudge and re-asks (up to 4 times). A non-repetitive fragment continues
    normally and stitches into the final answer -- confirming ``bounded_response.py`` is
    NOT this mechanism (it is a bounded HTTP-error-body reader, unrelated to
    finish_reason=="length"; see the report)."""
    agent = build_agent(enabled_toolsets=[])
    first = NormalizedResponse(
        content="This sentence just got cut off because it ran out of to",
        tool_calls=None, finish_reason="length", usage=Usage(),
    )
    client = install_fake_client(agent, Script().raw(first).text("kens, but now it's finished."))

    result = agent.run_conversation("tell me something long")

    assert client.call_count == 2
    assert result["completed"] is True
    assert "cut off" in result["final_response"] and "finished" in result["final_response"]


def test_repetition_dominated_truncation_aborts_instead_of_continuing():
    """``repetition_guard.is_repetition_dominated`` (60+ char window covering >=50% of a
    >=400-char fragment): a model that fills its ENTIRE output budget with one repeated
    phrase must not be handed a "continue" nudge (item #86581 in the module docstring --
    that just produces more repeated text). ``turn_truncation._abort_reason`` checks this
    BEFORE attempting a continuation and ends the turn instead. The script defines only
    ONE turn: if the repetition check didn't fire, ``_continue_text`` would ask for a
    2nd turn and ``ScriptExhausted`` would fail this test.
    """
    agent = build_agent(enabled_toolsets=[])
    repeated_fragment = ("The system encountered an error and retried the operation again. " * 8)
    assert len(repeated_fragment) >= 400
    turn = NormalizedResponse(content=repeated_fragment, tool_calls=None, finish_reason="length", usage=Usage())
    client = install_fake_client(agent, Script().raw(turn))

    result = agent.run_conversation("go")

    assert client.call_count == 1
    assert result["completed"] is False
    assert result.get("partial") is True
    assert "Repetition Detected" in result["final_response"]


def test_tool_calls_and_final_looking_text_in_one_message_are_not_conflated():
    """A model that emits BOTH tool calls and what reads like a final answer in the same
    message (e.g. "Let me check that for you." alongside a get_weather call): the text
    is kept as the tool-call message's content (and as an internal fallback for a later
    empty follow-up) but is NOT treated as the turn's final answer -- the loop still
    executes the tool(s) and waits for the model's real follow-up. Confirms narration
    is preserved rather than silently dropped, without letting it short-circuit the
    turn.
    """
    agent = None
    weather_calls = []

    def handler(args, **_kwargs):
        weather_calls.append(dict(args))
        return tool_result(temp_c=21, city=args.get("city"))

    registry.register(
        name="get_weather", toolset="demo",
        schema={"name": "get_weather", "description": "x",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}},
        handler=handler, override=True,
    )
    try:
        agent = build_agent()
        client = install_fake_client(
            agent,
            Script()
            .calls(("get_weather", {"city": "Rosario"}), content="Let me check that for you.")
            .text("It's 21 degrees in Rosario."),
        )

        result = agent.run_conversation("¿qué temperatura hace en Rosario?")

        assert client.call_count == 2
        assert result["completed"] is True
        assert result["final_response"] == "It's 21 degrees in Rosario."
        assert weather_calls == [{"city": "Rosario"}]

        # The interim narration was kept on the tool-call row, not discarded.
        tool_call_rows = [
            m for m in result["messages"]
            if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")
        ]
        assert any(m.get("content") == "Let me check that for you." for m in tool_call_rows)
    finally:
        registry.deregister("get_weather")
