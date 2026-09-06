"""Tests for the lifted provider transports.

A transport owns one wire format and nothing else: convert the messages, convert the
tool schemas, build the SDK kwargs, normalise the response. Client construction,
credentials, streaming and retries all live above it.

That narrowness is what makes provider independence real rather than aspirational,
so these tests check the boundary holds -- that each transport translates in both
directions and reaches for nothing else.
"""

from dataclasses import replace

import pytest

from hermes_core.providers.base import OMIT_TEMPERATURE, ProviderProfile
from hermes_core.agent.transports import (
    NormalizedResponse,
    ToolCall,
    build_tool_call,
    get_transport,
    map_finish_reason,
    register_transport,
)
from hermes_core.agent.transports.base import ProviderTransport
from hermes_core.agent.transports.anthropic import AnthropicTransport
from hermes_core.agent.transports.chat_completions import ChatCompletionsTransport


MESSAGES = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is the weather in Rosario?"},
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up the weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


# -- the registry ------------------------------------------------------------------

def test_the_shipped_transports_are_registered():
    """Registration happens on import, so a host names an api_mode and gets a transport."""
    assert get_transport("chat_completions") is not None
    assert get_transport("anthropic_messages") is not None


def test_a_host_can_register_its_own_transport():
    """The extension point: a new wire format is four methods and one call."""

    class CustomTransport(ProviderTransport):
        @property
        def api_mode(self):
            return "custom_wire"

        def convert_messages(self, messages, **kwargs):
            return messages

        def convert_tools(self, tools):
            return tools

        def build_kwargs(self, model, messages, tools=None, **params):
            return {"model": model}

        def normalize_response(self, response, **kwargs):
            return NormalizedResponse(content="", tool_calls=None, finish_reason="stop")

    register_transport("custom_wire", CustomTransport)

    assert get_transport("custom_wire") is not None


def test_an_unknown_api_mode_resolves_to_nothing():
    """Absence is reported, not raised.

    A missing transport usually means a configuration typo, and the caller is better
    placed than the registry to say what should happen -- fall back to a default,
    refuse to start, or name the misconfigured setting.
    """
    assert get_transport("no_such_wire_format") is None


# -- chat completions --------------------------------------------------------------

def test_chat_completions_passes_messages_through():
    """Its own format is the canonical one, so conversion is identity."""
    transport = ChatCompletionsTransport()

    assert transport.convert_messages(MESSAGES) == MESSAGES


def test_chat_completions_builds_sdk_kwargs():
    transport = ChatCompletionsTransport()

    kwargs = transport.build_kwargs("gpt-4o-mini", MESSAGES, TOOLS)

    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["messages"] == MESSAGES
    assert [t["function"]["name"] for t in kwargs["tools"]] == ["get_weather"]


def test_sampling_parameters_need_a_provider_profile():
    """Without a profile the transport sends only what it is sure of.

    Blindly forwarding keyword arguments to the SDK is how a call dies on an
    unrecognised parameter, so sampling settings travel the profile path, where a
    provider's quirks are known.
    """
    transport = ChatCompletionsTransport()

    kwargs = transport.build_kwargs("gpt-4o-mini", MESSAGES, None, temperature=0.4)

    assert "temperature" not in kwargs


def test_a_profile_lets_the_caller_set_temperature():
    transport = ChatCompletionsTransport()
    profile = ProviderProfile(name="demo", api_mode="chat_completions")

    kwargs = transport.build_kwargs(
        "some-model", MESSAGES, None, provider_profile=profile, temperature=0.4
    )

    assert kwargs["temperature"] == 0.4


def test_a_profile_can_override_the_caller_s_temperature():
    """Some models accept exactly one temperature and reject anything else.

    The profile wins, so a host setting a sensible default globally does not have to
    special-case every model that refuses it.
    """
    transport = ChatCompletionsTransport()
    profile = replace(
        ProviderProfile(name="demo", api_mode="chat_completions"), fixed_temperature=1.0
    )

    kwargs = transport.build_kwargs(
        "some-model", MESSAGES, None, provider_profile=profile, temperature=0.4
    )

    assert kwargs["temperature"] == 1.0


def test_a_profile_can_omit_temperature_entirely():
    """Other models reject the parameter's presence, not just its value."""
    transport = ChatCompletionsTransport()
    profile = replace(
        ProviderProfile(name="demo", api_mode="chat_completions"),
        fixed_temperature=OMIT_TEMPERATURE,
    )

    kwargs = transport.build_kwargs(
        "some-model", MESSAGES, None, provider_profile=profile, temperature=0.4
    )

    assert "temperature" not in kwargs


def test_chat_completions_normalises_a_text_answer(openai_response):
    transport = ChatCompletionsTransport()

    normalized = transport.normalize_response(openai_response(content="It is 21C."))

    assert isinstance(normalized, NormalizedResponse)
    assert normalized.content == "It is 21C."
    assert normalized.finish_reason == "stop"
    assert not normalized.tool_calls


def test_chat_completions_normalises_tool_calls(openai_response):
    """Whatever the provider's shape, downstream sees `ToolCall`."""
    transport = ChatCompletionsTransport()
    response = openai_response(
        tool_calls=[("call_abc", "get_weather", '{"city": "Rosario"}')],
        finish_reason="tool_calls",
    )

    normalized = transport.normalize_response(response)

    assert normalized.finish_reason == "tool_calls"
    assert len(normalized.tool_calls) == 1
    call = normalized.tool_calls[0]
    assert (call.id, call.name, call.arguments) == ("call_abc", "get_weather", '{"city": "Rosario"}')


def test_chat_completions_carries_usage(openai_response):
    """Token counts drive budgets and cost reporting, so they must survive."""
    transport = ChatCompletionsTransport()

    normalized = transport.normalize_response(
        openai_response(content="hi", prompt_tokens=11, completion_tokens=3)
    )

    assert normalized.usage.prompt_tokens == 11
    assert normalized.usage.completion_tokens == 3


# -- anthropic ---------------------------------------------------------------------

def test_anthropic_lifts_the_system_message_out():
    """Anthropic takes the system prompt as its own parameter, not as a message.

    Getting this wrong sends the instructions as an ordinary user turn, where the
    model treats them as something the user said rather than as its brief.
    """
    transport = AnthropicTransport()

    system, messages = transport.convert_messages(MESSAGES)

    assert "You are helpful." in str(system)
    assert all(m["role"] != "system" for m in messages)
    assert messages[0]["role"] == "user"


def test_anthropic_converts_tool_schemas_to_its_own_shape():
    """Anthropic uses `input_schema` where OpenAI uses `parameters`."""
    transport = AnthropicTransport()

    converted = transport.convert_tools(TOOLS)

    assert converted[0]["name"] == "get_weather"
    assert "input_schema" in converted[0]
    assert converted[0]["input_schema"]["required"] == ["city"]


def test_anthropic_declares_its_api_mode():
    assert AnthropicTransport().api_mode == "anthropic_messages"


# -- shared helpers ----------------------------------------------------------------

def test_build_tool_call_serialises_dict_arguments():
    call = build_tool_call("id-1", "get_weather", {"city": "Rosario"})

    assert call.arguments == '{"city": "Rosario"}'


def test_build_tool_call_keeps_provider_specifics_out_of_the_shared_shape():
    """Protocol-specific state rides in `provider_data`, not as new top-level fields.

    Gemini's thought signature must be replayed on later calls or the API rejects
    them, so it has to survive -- without every consumer growing a field for it.
    """
    call = build_tool_call("id-1", "t", {}, call_id="codex-123")

    assert call.call_id == "codex-123"
    assert call.provider_data == {"call_id": "codex-123"}


def test_a_tool_call_still_answers_to_the_openai_shape():
    """Upstream reads `tc.function.name` in dozens of places; keep that working."""
    call = ToolCall(id="x", name="get_weather", arguments="{}")

    assert call.function.name == "get_weather"
    assert call.type == "function"


def test_unknown_stop_reasons_fall_back_to_stop():
    """A provider inventing a reason must not produce an unhandled finish state."""
    assert map_finish_reason("something_new", {"end_turn": "stop"}) == "stop"
    assert map_finish_reason(None, {"end_turn": "stop"}) == "stop"
    assert map_finish_reason("end_turn", {"end_turn": "stop"}) == "stop"
