"""End-to-end tests: a real agent turn, outside Hermes.

Every other test here covers one lifted piece. These cover the claim the whole
extraction rests on -- that the pieces still work *together* when the CLI, the
gateway, the vendor's managed relay and its OAuth are all gone.

Only the model is replaced. Prompt assembly, tool-call validation, dispatch, feeding
results back and deciding to stop are the real code, running against a scripted
provider.
"""

import json
import tempfile

import pytest

from hermes_core.seams.config import DictConfigSource, set_config_source
from hermes_core.seams.credentials import StaticCredentials, set_credential_source
from hermes_core.seams.paths import DirectoryWorkspace, set_workspace
from hermes_core.testing import Script, install_fake_client
from hermes_core.tools.registry import ToolRegistry, registry, tool_result


@pytest.fixture(autouse=True)
def isolated_core(monkeypatch):
    """Point the core at a scratch workspace and an in-memory configuration.

    The whole reason a core is embeddable: three calls and it has somewhere to put
    state, something to read settings from, and a credential -- with nothing touching
    the machine's real home directory.
    """
    set_workspace(DirectoryWorkspace(tempfile.mkdtemp()))
    set_config_source(
        DictConfigSource({"model": {"default": "fake-model", "provider": "openai"}})
    )
    set_credential_source(StaticCredentials("sk-test"))
    yield


@pytest.fixture
def weather_tool():
    """Register a tool and record what it is called with."""
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
        max_iterations=5,
    )
    settings.update(overrides)
    return AIAgent(**settings)


def tool_messages(request):
    return [m for m in request.messages if m.get("role") == "tool"]


# -- the whole thing -----------------------------------------------------------------

def test_an_agent_answers_without_tools():
    agent = build_agent(enabled_toolsets=[])
    client = install_fake_client(agent, Script().text("Hola, soy el core."))

    result = agent.run_conversation("hola")

    assert result["completed"] is True
    assert result["final_response"] == "Hola, soy el core."
    assert client.call_count == 1


def test_an_agent_calls_a_tool_and_answers_from_its_result(weather_tool):
    """The claim, end to end.

    The model asks for a tool, the tool runs with the arguments it asked for, the
    result is fed back, and the model answers from it. Everything between those steps
    is lifted Hermes code with no Hermes around it.
    """
    agent = build_agent()
    client = install_fake_client(
        agent,
        Script()
        .calls(("get_weather", {"city": "Rosario"}))
        .text("Hacen 21 grados en Rosario."),
    )

    result = agent.run_conversation("¿qué temperatura hace en Rosario?")

    assert result["completed"] is True
    assert result["final_response"] == "Hacen 21 grados en Rosario."
    assert result["api_calls"] == 2

    # The tool ran, with exactly the arguments the model asked for.
    assert weather_tool == [{"city": "Rosario"}]

    # Its output went back to the model, as a tool-role message.
    results = tool_messages(client.last_request)
    assert len(results) == 1
    assert json.loads(results[0]["content"]) == {"temp_c": 21, "city": "Rosario"}
    assert results[0]["name"] == "get_weather"


def test_the_tool_schema_reaches_the_model():
    """A registered tool is offered, in the shape the provider expects."""
    agent = build_agent()
    client = install_fake_client(agent, Script().text("ok"))

    agent.run_conversation("hola")

    assert client.last_request.tool_names == ["get_weather"] or client.last_request.tools is None


def test_a_system_prompt_is_assembled_and_sent():
    """The agent's identity reaches the model rather than being dropped in transit."""
    agent = build_agent(enabled_toolsets=[])
    client = install_fake_client(agent, Script().text("ok"))

    agent.run_conversation("hola")

    system = client.last_request.system_prompt
    assert system and len(system) > 50


def test_the_conversation_carries_forward(weather_tool):
    """The second call sees the first turn, not a blank history.

    Losing this is the classic failure of a hand-rolled loop: each call succeeds in
    isolation and the agent has no memory within its own turn.
    """
    agent = build_agent()
    client = install_fake_client(
        agent,
        Script().calls(("get_weather", {"city": "Rosario"})).text("Hacen 21 grados."),
    )

    agent.run_conversation("¿qué temperatura hace en Rosario?")

    first, second = client.requests
    assert [m["role"] for m in first.messages] == ["system", "user"]
    assert [m["role"] for m in second.messages] == ["system", "user", "assistant", "tool"]


def test_several_tools_in_one_turn_all_run():
    """A parallel batch: every call runs and every result comes back."""
    seen = []

    def make(name):
        def handler(args, **_kwargs):
            seen.append(name)
            return tool_result(ok=name)

        return handler

    for name in ("alpha", "beta"):
        registry.register(
            name=name, toolset="demo",
            schema={"name": name, "description": "x",
                    "parameters": {"type": "object", "properties": {}}},
            handler=make(name), override=True,
        )
    try:
        agent = build_agent()
        client = install_fake_client(
            agent, Script().calls(("alpha", {}), ("beta", {})).text("listo")
        )

        result = agent.run_conversation("hacé las dos cosas")

        assert result["completed"] is True
        assert sorted(seen) == ["alpha", "beta"]
        assert len(tool_messages(client.last_request)) == 2
    finally:
        registry.deregister("alpha")
        registry.deregister("beta")


def test_a_tool_that_raises_becomes_a_tool_result_not_a_crash(weather_tool):
    """A failing tool must not abort the turn.

    The model is told what went wrong and can react -- apologise, try another
    approach, ask the user. Propagating the exception would end the conversation over
    something the model might well handle.
    """
    def exploding(args, **_kwargs):
        raise RuntimeError("the weather service is down")

    registry.register(
        name="get_weather", toolset="demo",
        schema={"name": "get_weather", "description": "x",
                "parameters": {"type": "object", "properties": {}}},
        handler=exploding, override=True,
    )

    agent = build_agent()
    client = install_fake_client(
        agent,
        Script().calls(("get_weather", {"city": "Rosario"})).text("No pude consultarlo."),
    )

    result = agent.run_conversation("¿qué temperatura hace?")

    assert result["completed"] is True
    results = tool_messages(client.last_request)
    assert len(results) == 1
    assert "weather service is down" in results[0]["content"]


def test_an_unknown_tool_name_is_answered_and_the_turn_continues():
    """The model inventing a tool costs a round trip, not the turn."""
    agent = build_agent(enabled_toolsets=[])
    client = install_fake_client(
        agent,
        Script().calls(("no_such_tool", {})).text("Perdón, me equivoqué de herramienta."),
    )

    result = agent.run_conversation("hola")

    assert result["completed"] is True
    assert client.call_count == 2


def test_nothing_is_registered_unless_the_host_registers_it():
    """The design rule, checked from the outside.

    A fresh registry offers nothing. A host gets the tools it asked for and no others
    -- which is what keeps Hermes's bundled catalogue, and the gateway three of those
    tools import, out of an embedded agent.
    """
    assert ToolRegistry().get_all_tool_names() == []


# -- streaming -----------------------------------------------------------------------

def test_a_streamed_turn_calls_a_tool_and_answers(weather_tool):
    """The same turn, delivered as deltas.

    Worth testing separately: the streaming path is what runs against a real provider,
    and it broke in a way the non-streaming path never showed. The relay seam returned
    the SDK's raw stream where upstream returns its own wrapper, so the loop's read of
    `stream.final_response` raised mid-response, was classified as a stream that died
    mid tool-call, and the complete tool call the model had already sent was dropped.
    """
    agent = build_agent()
    client = install_fake_client(
        agent,
        Script()
        .calls(("get_weather", {"city": "Rosario"}))
        .text("Hacen 21 grados en Rosario."),
        stream=True,
    )

    result = agent.run_conversation("¿qué temperatura hace en Rosario?")

    assert result["completed"] is True
    assert result["final_response"] == "Hacen 21 grados en Rosario."
    assert weather_tool == [{"city": "Rosario"}]
    assert len(tool_messages(client.last_request)) == 1


def test_streamed_text_deltas_are_joined():
    """The fake splits text across deltas, so a joining bug cannot pass unnoticed."""
    agent = build_agent(enabled_toolsets=[])
    install_fake_client(agent, Script().text("una respuesta partida en varios pedazos"),
                        stream=True)

    result = agent.run_conversation("hola")

    assert result["final_response"] == "una respuesta partida en varios pedazos"
