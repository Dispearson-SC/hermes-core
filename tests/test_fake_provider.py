"""Tests for the scripted provider.

The fake is the instrument every later test measures with, so its own failure modes
matter more than its happy path: a double that quietly answers when it should not
would make a broken loop look correct.
"""

import json

import pytest

from hermes_core.testing import FakeProvider, Script, ScriptExhausted
from hermes_core.agent.transports.base import ProviderTransport
from hermes_core.agent.transports.types import NormalizedResponse


def test_it_is_a_real_transport():
    """The fake must satisfy the same contract a real provider does."""
    provider = FakeProvider()
    assert isinstance(provider, ProviderTransport)
    assert provider.api_mode == "fake"


def test_replays_turns_in_order():
    provider = FakeProvider(
        Script().calls(("get_weather", {"city": "Rosario"})).text("It is 21C.")
    )

    first = provider.complete(messages=[{"role": "user", "content": "weather?"}])
    assert first.finish_reason == "tool_calls"
    assert [c.name for c in first.tool_calls] == ["get_weather"]
    assert json.loads(first.tool_calls[0].arguments) == {"city": "Rosario"}

    second = provider.complete(messages=[])
    assert second.finish_reason == "stop"
    assert second.content == "It is 21C."
    assert provider.exhausted


def test_one_turn_can_request_several_calls_in_parallel():
    """Several specs in one `calls()` are one batch, not consecutive turns."""
    provider = FakeProvider(
        Script().calls(("a", {}), ("b", {}), ("c", {}))
    )
    response = provider.complete(messages=[])

    assert provider.call_count == 1
    assert [c.name for c in response.tool_calls] == ["a", "b", "c"]


def test_call_ids_are_stable_and_unique():
    """Ids are derived from position, so assertions never have to guess one."""
    provider = FakeProvider(Script().calls(("a", {}), ("b", {})).calls(("c", {})))

    first = provider.complete(messages=[])
    second = provider.complete(messages=[])

    ids = [c.id for c in first.tool_calls] + [c.id for c in second.tool_calls]
    assert ids == ["call_0_0", "call_0_1", "call_1_0"]
    assert len(set(ids)) == len(ids)


def test_an_explicit_call_id_wins():
    provider = FakeProvider(Script().calls(("a", {}, "chosen-id")))
    response = provider.complete(messages=[])
    assert response.tool_calls[0].id == "chosen-id"


def test_exhaustion_raises_instead_of_answering():
    """The failure this double exists to catch: a loop that will not stop."""
    provider = FakeProvider(Script().text("done"))
    provider.complete(messages=[])

    with pytest.raises(ScriptExhausted) as excinfo:
        provider.complete(messages=[])

    message = str(excinfo.value)
    assert "turn 2" in message
    assert "script defines 1" in message


def test_records_what_the_loop_sent():
    provider = FakeProvider(Script().text("ok"))
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]

    provider.complete(
        model="some-model",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ],
        tools=tools,
        temperature=0.2,
    )

    request = provider.last_request
    assert request.model == "some-model"
    assert request.tool_names == ["get_weather"]
    assert request.system_prompt == "You are helpful."
    assert request.params == {"temperature": 0.2}
    assert len(request.messages_with_role("user")) == 1


def test_recorded_messages_are_snapshots():
    """Later mutation of the caller's list must not rewrite recorded history."""
    provider = FakeProvider(Script().text("ok"))
    messages = [{"role": "user", "content": "original"}]

    provider.complete(messages=messages)
    messages[0]["content"] = "mutated"

    assert provider.last_request.messages[0]["content"] == "original"


def test_tool_names_survive_a_bare_schema():
    """Some providers take `{"name": ...}` without the `function` wrapper."""
    provider = FakeProvider(Script().text("ok"))
    provider.complete(messages=[], tools=[{"name": "bare_tool", "parameters": {}}])
    assert provider.last_request.tool_names == ["bare_tool"]


def test_errors_can_be_scripted():
    """Retry and error-classification paths need the model call to fail."""
    provider = FakeProvider(Script().error(TimeoutError("upstream timed out")).text("recovered"))

    with pytest.raises(TimeoutError):
        provider.complete(messages=[])

    assert provider.complete(messages=[]).content == "recovered"


def test_raw_lets_a_test_build_any_shape():
    """The escape hatch: a truncated response, which the helpers do not model."""
    truncated = NormalizedResponse(content="half a sen", tool_calls=None, finish_reason="length")
    provider = FakeProvider(Script().raw(truncated))

    assert provider.complete(messages=[]).finish_reason == "length"


def test_last_request_without_a_call_is_an_error():
    with pytest.raises(AssertionError, match="no model call"):
        _ = FakeProvider().last_request


def test_normalize_response_rejects_foreign_objects():
    with pytest.raises(TypeError, match="only handles"):
        FakeProvider().normalize_response({"choices": []})
