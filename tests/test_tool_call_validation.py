"""Tests for tool-call validation, lifted from upstream's turn loop.

This is the module the whole extraction is for. Writing a loop that sends messages
and runs the tools a model asks for is a couple of hundred lines. Surviving contact
with a real model is not, and the difference is here:

* the model invents a tool name that does not exist
* the model sends arguments that are not valid JSON
* a router rewrites ``finish_reason: length`` to ``tool_calls``, so truncated
  arguments arrive looking complete
* the model keeps doing one of the above, forever

Each of those has a considered answer upstream, arrived at by breaking in production.
These tests pin that behaviour so the extracted copy cannot quietly lose it.
"""

import json

import pytest

from hermes_core.agent.transports.types import ToolCall
from hermes_core.agent.turn_tool_validation import validate_tool_calls
from hermes_core.testing.fake_host import FakeAssistantMessage, FakeTurnHost


def call(name, arguments="{}", call_id="call_1"):
    if isinstance(arguments, (dict, list)):
        arguments = json.dumps(arguments)
    return ToolCall(id=call_id, name=name, arguments=arguments)


def validate(host, *tool_calls, finish_reason="tool_calls", messages=None):
    message = FakeAssistantMessage(list(tool_calls))
    return (
        validate_tool_calls(
            host,
            message,
            finish_reason,
            messages=messages if messages is not None else [],
            conversation_history=None,
            api_call_count=1,
            effective_task_id="task-1",
        ),
        message,
    )


# -- the ordinary case -------------------------------------------------------------

def test_a_well_formed_call_is_approved():
    host = FakeTurnHost(valid_tool_names={"get_weather"})

    verdict, _ = validate(host, call("get_weather", {"city": "Rosario"}))

    assert verdict.action == "ok"
    assert not verdict.mixed_invalid_batch


def test_counters_reset_after_a_good_turn():
    """A model that recovers must not carry its earlier strikes forward."""
    host = FakeTurnHost(valid_tool_names={"ok_tool"})
    host._invalid_tool_retries = 2
    host._invalid_json_retries = 2

    validate(host, call("ok_tool"))

    assert host._invalid_tool_retries == 0
    assert host._invalid_json_retries == 0


# -- hallucinated tool names -------------------------------------------------------

def test_a_repairable_name_is_corrected_and_runs():
    """The model meant a real tool and got the name slightly wrong.

    Repairing beats erroring: the turn does the work the user asked for instead of
    spending a round trip teaching the model its own tool list.
    """
    host = FakeTurnHost(valid_tool_names={"get_weather"}, repairs={"getWeather": "get_weather"})

    verdict, message = validate(host, call("getWeather"))

    assert verdict.action == "ok"
    assert message.tool_calls[0].name == "get_weather"


def test_an_unrepairable_name_is_answered_with_a_tool_result():
    """The error goes back as a tool-role result, never as a user message.

    Answering with a user message would break role alternation, and every provider
    then has to be coaxed past a conversation that reads as two users in a row.
    """
    messages = []
    host = FakeTurnHost(valid_tool_names={"get_weather"})

    verdict, _ = validate(host, call("nonexistent_tool"), messages=messages)

    assert verdict.action == "continue"
    roles = [m["role"] for m in messages]
    assert roles == ["assistant", "tool"]
    assert "does not exist" in messages[-1]["content"]


def test_the_error_lists_the_real_tools_so_the_model_can_correct_itself():
    messages = []
    host = FakeTurnHost(valid_tool_names={"get_weather", "send_email"})

    validate(host, call("nonexistent_tool"), messages=messages)

    content = messages[-1]["content"]
    assert "get_weather" in content and "send_email" in content


def test_a_blank_tool_name_does_not_get_the_catalog():
    """A blank name means the model echoed tool-call syntax it saw in data.

    Sending the tool list back would feed that loop with more of the same syntax, so
    a blank name gets a terse instruction instead -- while a merely wrong name still
    gets the catalog, because that model can actually correct itself.
    """
    messages = []
    host = FakeTurnHost(valid_tool_names={"get_weather", "send_email"})

    validate(host, call("   "), messages=messages)

    content = messages[-1]["content"]
    assert "get_weather" not in content
    assert "do not re-emit it" in content


def test_three_strikes_ends_the_turn_as_partial():
    """A degenerate model must not loop forever burning tokens."""
    host = FakeTurnHost(valid_tool_names={"get_weather"})
    host._invalid_tool_retries = 2  # this call is the third

    verdict, _ = validate(host, call("nonexistent_tool"))

    assert verdict.action == "return"
    assert verdict.result["partial"] is True
    assert verdict.result["completed"] is False
    assert "invalid tool call" in verdict.result["error"]


def test_the_partial_exit_persists_before_returning():
    """This path never reaches the loop's finalizer, so it must persist itself."""
    host = FakeTurnHost(valid_tool_names={"get_weather"})
    host._invalid_tool_retries = 2

    validate(host, call("nonexistent_tool"))

    assert len(host.persisted) == 1


def test_the_strike_counter_resets_after_giving_up():
    """The next turn starts clean rather than dying immediately."""
    host = FakeTurnHost(valid_tool_names={"get_weather"})
    host._invalid_tool_retries = 2

    validate(host, call("nonexistent_tool"))

    assert host._invalid_tool_retries == 0


# -- mixed batches -----------------------------------------------------------------

def test_one_bad_name_does_not_void_the_valid_calls():
    """Discarding the whole batch throws away real work the model got right."""
    messages = []
    host = FakeTurnHost(valid_tool_names={"get_weather"})

    verdict, _ = validate(
        host,
        call("get_weather", {"city": "Rosario"}, call_id="a"),
        call("nonexistent_tool", call_id="b"),
        messages=messages,
    )

    assert verdict.action == "ok"
    assert verdict.mixed_invalid_batch


def test_a_mixed_batch_does_not_advance_the_strike_counter():
    """Strikes exist to stop a model that cannot call anything correctly.

    A turn with at least one valid call is not that model, so counting it would
    punish partial success and end useful turns early.
    """
    host = FakeTurnHost(valid_tool_names={"get_weather"})
    host._invalid_tool_retries = 2

    verdict, _ = validate(
        host,
        call("get_weather", call_id="a"),
        call("nonexistent_tool", call_id="b"),
    )

    assert verdict.action == "ok"
    assert host._invalid_tool_retries == 0


def test_broken_arguments_on_an_invalid_name_do_not_trigger_a_json_retry():
    """That call will never run, so its arguments are irrelevant.

    Letting them start a whole-turn JSON retry would discard the valid calls over a
    string nobody is going to parse.
    """
    host = FakeTurnHost(valid_tool_names={"get_weather"})

    verdict, _ = validate(
        host,
        call("get_weather", call_id="a"),
        call("nonexistent_tool", arguments="{not json", call_id="b"),
    )

    assert verdict.action == "ok"
    assert host._invalid_json_retries == 0


# -- argument normalisation --------------------------------------------------------

def test_empty_arguments_become_an_empty_object():
    """A common model quirk for tools that take no parameters."""
    host = FakeTurnHost(valid_tool_names={"ping"})

    verdict, message = validate(host, call("ping", arguments=""))

    assert verdict.action == "ok"
    assert message.tool_calls[0].arguments == "{}"


def test_whitespace_only_arguments_become_an_empty_object():
    host = FakeTurnHost(valid_tool_names={"ping"})

    _, message = validate(host, call("ping", arguments="   "))

    assert message.tool_calls[0].arguments == "{}"


def test_a_dict_instead_of_a_json_string_is_serialised():
    """Some providers hand back parsed arguments; downstream expects a string."""
    host = FakeTurnHost(valid_tool_names={"ping"})
    tool_call = ToolCall(id="c", name="ping", arguments="{}")
    tool_call.arguments = {"already": "parsed"}

    _, message = validate(host, tool_call)

    assert message.tool_calls[0].arguments == '{"already": "parsed"}'


# -- malformed JSON ----------------------------------------------------------------

def test_malformed_json_retries_without_touching_the_conversation():
    """Retry first: the same model often gets it right on a second attempt.

    Nothing is appended, so the retry does not pollute history with a failure the
    model never sees.

    The arguments must be *complete* but unparseable -- note the closing brace. An
    unterminated string would be classified as truncation instead, which is a
    different path with a different answer.
    """
    messages = []
    host = FakeTurnHost(valid_tool_names={"ping"})

    verdict, _ = validate(host, call("ping", arguments='{"a": }'), messages=messages)

    assert verdict.action == "continue"
    assert messages == []
    assert host._invalid_json_retries == 1


def test_after_three_failures_the_model_is_told_what_broke():
    """Retrying forever is not an answer; the model needs the parser error.

    It arrives as tool results rather than a partial exit, so the turn can still
    recover instead of ending.
    """
    messages = []
    host = FakeTurnHost(valid_tool_names={"ping"})
    host._invalid_json_retries = 2  # this call is the third

    verdict, _ = validate(host, call("ping", arguments='{"a": }'), messages=messages)

    assert verdict.action == "continue"
    assert [m["role"] for m in messages] == ["assistant", "tool"]
    assert "Invalid JSON arguments" in messages[-1]["content"]
    assert host._invalid_json_retries == 0


def test_the_recovery_message_shows_the_empty_object_form():
    """Most of these failures are a model fumbling "no arguments"."""
    messages = []
    host = FakeTurnHost(valid_tool_names={"ping"})
    host._invalid_json_retries = 2

    validate(host, call("ping", arguments='{"a": }'), messages=messages)

    assert "{}" in messages[-1]["content"]


# -- truncation: the subtle one ----------------------------------------------------

def test_truncated_arguments_are_refused_outright():
    """The dangerous case, and the reason this module is worth inheriting.

    A router can rewrite `finish_reason: length` to `tool_calls`, so a response cut
    off mid-stream arrives looking like a normal tool call. Its arguments parse as
    broken JSON, which would ordinarily mean "retry" -- but the model never finished
    the thought, so retrying re-sends a mutilated turn. Worse, half-complete
    arguments that happened to parse would run a real action against real data.

    Arguments that do not end in `}` or `]` were cut off, so the turn stops.
    """
    host = FakeTurnHost(valid_tool_names={"transfer_funds"})

    verdict, _ = validate(
        host,
        call("transfer_funds", arguments='{"amount": 5000, "to": "acct-'),
        finish_reason="tool_calls",
    )

    assert verdict.action == "return"
    assert verdict.result["partial"] is True
    assert "truncated" in verdict.result["error"].lower()


def test_truncation_does_not_burn_a_json_retry():
    """Truncation is not a malformed-arguments problem, so it uses no strike."""
    host = FakeTurnHost(valid_tool_names={"ping"})

    validate(host, call("ping", arguments='{"a": "unterminated'))

    assert host._invalid_json_retries == 0


def test_truncation_releases_the_turn_s_resources():
    host = FakeTurnHost(valid_tool_names={"ping"})

    validate(host, call("ping", arguments='{"a": "unterminated'))

    assert host.cleaned_up == ["task-1"]


def test_broken_but_complete_json_is_a_retry_not_a_truncation():
    """The boundary: it ends in `}`, so the model finished and simply got it wrong."""
    host = FakeTurnHost(valid_tool_names={"ping"})

    verdict, _ = validate(host, call("ping", arguments='{"a": }'))

    assert verdict.action == "continue"
    assert host._invalid_json_retries == 1


# -- duplicate ids -----------------------------------------------------------------

def test_duplicate_call_ids_are_made_unique():
    """Two calls sharing an id lose one call and its result downstream.

    The pre-API sanitizer keeps only the first entry per id, so the second tool would
    run with its result silently discarded.
    """
    host = FakeTurnHost(valid_tool_names={"ping"})

    _, message = validate(
        host,
        call("ping", call_id="same"),
        call("ping", call_id="same"),
    )

    ids = [c.id for c in message.tool_calls]
    assert len(set(ids)) == 2


# -- every tool call gets an answer -------------------------------------------------

@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("invalid_name", id="unknown tool name"),
        pytest.param("bad_json", id="malformed arguments"),
    ],
)
def test_every_call_receives_a_result(scenario):
    """An unanswered tool_call leaves the conversation malformed for the next turn.

    Providers reject history in which an assistant tool call has no matching result,
    so both error paths must answer every call in the batch -- including the ones
    that were merely alongside the offender.
    """
    messages = []
    host = FakeTurnHost(valid_tool_names={"ping"})

    if scenario == "invalid_name":
        calls = [call("nope", call_id="a"), call("nope2", call_id="b")]
    else:
        host._invalid_json_retries = 2
        calls = [call("ping", arguments='{"x": }', call_id="a"),
                 call("ping", arguments='{"y": }', call_id="b")]

    validate(host, *calls, messages=messages)

    assistant = messages[0]
    results = [m for m in messages if m["role"] == "tool"]
    assert len(results) == len(assistant["tool_calls"])
    assert {m["tool_call_id"] for m in results} == {"a", "b"}
