"""Driving the synchronous turn loop from an asynchronous host.

Two claims are worth testing here and neither is about the agent:

1. Awaiting a turn does not freeze the host. A coroutine that calls
   ``agent.run_conversation`` directly stops *the whole server* for the length of the
   turn -- not just that request -- and nothing in the core stops you from writing it.
2. A synchronous tool handler can reach an ``async def`` service on the host's *own*
   loop. The near-misses (``asyncio.run``, ``is_async=True``) each run the coroutine on
   a brand-new loop in a fresh thread, which is wrong in a way that does not fail
   immediately: an asyncpg pool used from the wrong loop corrupts later, elsewhere.

So the tests below check which loop and which thread the work actually ran on, rather
than only that it returned a value -- a fake that answered on any loop would pass the
weaker assertion while shipping the bug.
"""

import asyncio
import tempfile
import threading
import time

import pytest

from hermes_core.seams.async_bridge import (
    HostLoopUnavailable,
    bind_host_loop,
    call_host_async,
    get_host_loop,
    run_conversation_async,
)
from hermes_core.seams.config import DictConfigSource, set_config_source
from hermes_core.seams.credentials import StaticCredentials, set_credential_source
from hermes_core.seams.paths import DirectoryWorkspace, set_workspace
from hermes_core.testing import Script, install_fake_client
from hermes_core.tools.registry import registry, tool_result

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def isolated_core():
    set_workspace(DirectoryWorkspace(tempfile.mkdtemp(prefix="async-")))
    set_config_source(DictConfigSource({"model": {"default": "fake-model", "provider": "openai"}}))
    set_credential_source(StaticCredentials("sk-test"))
    yield


def build_agent(**overrides):
    from hermes_core.run_agent import AIAgent

    settings = dict(
        api_key="sk-test", base_url="https://example.invalid/v1", provider="openai",
        model="fake-model", enabled_toolsets=["demo"], quiet_mode=True, max_iterations=5,
    )
    settings.update(overrides)
    return AIAgent(**settings)


# -- calling in ----------------------------------------------------------------------

async def test_a_turn_can_be_awaited_and_returns_its_result():
    agent = build_agent(enabled_toolsets=[])
    install_fake_client(agent, Script().text("Hola desde un host async."))

    result = await run_conversation_async(agent, "hola")

    assert result["completed"] is True
    assert result["final_response"] == "Hola desde un host async."


async def test_the_event_loop_keeps_serving_while_the_turn_runs():
    """The whole point. A blocking call here would starve every other request.

    The turn is made to take real wall-clock time by having the fake client sleep, and a
    second coroutine counts how often it gets scheduled meanwhile. Calling
    ``run_conversation`` directly instead of awaiting this makes the count 1.
    """
    agent = build_agent(enabled_toolsets=[])
    client = install_fake_client(agent, Script().text("listo"))
    original = client.chat.completions.create

    def slow_create(*args, **kwargs):
        time.sleep(0.3)
        return original(*args, **kwargs)

    client.chat.completions.create = slow_create

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    beat = asyncio.ensure_future(heartbeat())
    try:
        result = await run_conversation_async(agent, "hola")
    finally:
        beat.cancel()

    assert result["completed"] is True
    assert ticks > 5, f"the loop was blocked during the turn (only {ticks} ticks)"


async def test_keyword_arguments_reach_run_conversation():
    agent = build_agent(enabled_toolsets=[])
    client = install_fake_client(agent, Script().text("ok"))

    await run_conversation_async(agent, "hola", system_message="Sos un test.")

    assert any(
        m.get("role") == "system" and "Sos un test." in str(m.get("content", ""))
        for m in client.last_request.messages
    )


async def test_an_exception_from_the_turn_propagates_to_the_awaiter():
    class Boom(Exception):
        pass

    class ExplodingAgent:
        def run_conversation(self, message, **kwargs):
            raise Boom("turno roto")

    with pytest.raises(Boom):
        await run_conversation_async(ExplodingAgent(), "hola")


# -- calling back out ----------------------------------------------------------------

@pytest.fixture
def host_service_tool():
    """A tool whose handler is synchronous but whose work is not -- the shape of any
    host with async repositories."""
    observed = {}

    async def host_service(value):
        observed["ran_on"] = asyncio.get_running_loop()
        observed["thread"] = threading.current_thread().name
        await asyncio.sleep(0)
        return f"servicio:{value}"

    def handler(args, **_kwargs):
        observed["handler_thread"] = threading.current_thread().name
        return tool_result(answer=call_host_async(host_service(args.get("value", ""))))

    registry.register(
        name="host_call",
        toolset="demo",
        schema={
            "name": "host_call", "description": "Reach the host's async service.",
            "parameters": {"type": "object", "properties": {"value": {"type": "string"}}},
        },
        handler=handler,
        override=True,
    )
    yield observed
    registry.deregister("host_call")


async def test_a_sync_tool_handler_reaches_an_async_host_service(host_service_tool):
    """The claim, end to end, inside a real turn."""
    agent = build_agent()
    client = install_fake_client(
        agent,
        Script().calls(("host_call", {"value": "x"})).text("hecho"),
    )

    result = await run_conversation_async(agent, "usá la herramienta")

    assert result["completed"] is True
    tool_results = [m for m in client.last_request.messages if m.get("role") == "tool"]
    assert "servicio:x" in tool_results[0]["content"]


async def test_the_host_service_runs_on_the_hosts_own_loop_not_a_new_one(host_service_tool):
    """The assertion that separates this from the near-misses.

    ``asyncio.run`` and ``is_async=True`` both answer correctly here while running the
    coroutine on a *different* loop -- which is what breaks an asyncpg pool later. So the
    check is loop identity, not the returned value.
    """
    agent = build_agent()
    install_fake_client(agent, Script().calls(("host_call", {"value": "x"})).text("hecho"))
    host_loop = asyncio.get_running_loop()

    await run_conversation_async(agent, "usá la herramienta")

    assert host_service_tool["ran_on"] is host_loop
    # And the handler itself really was off the loop thread, or none of this was needed.
    assert host_service_tool["handler_thread"] != threading.current_thread().name


async def test_an_exception_from_the_host_service_reaches_the_handler_unchanged():
    """A handler must be able to catch its own failures -- a wrapped exception would
    make every host error look like an infrastructure error."""
    class ServiceError(Exception):
        pass

    async def failing():
        raise ServiceError("el repositorio explotó")

    def in_thread():
        with pytest.raises(ServiceError, match="el repositorio explotó"):
            call_host_async(failing())
        return "caught"

    with bind_host_loop():
        assert await asyncio.to_thread(in_thread) == "caught"


# -- the failure modes, reported instead of hung -------------------------------------

async def test_calling_without_a_bound_loop_raises_a_named_error():
    async def anything():
        return 1

    coro = anything()

    def in_thread():
        with pytest.raises(HostLoopUnavailable, match="run_conversation_async"):
            call_host_async(coro)
        return "raised"

    try:
        assert await asyncio.to_thread(in_thread) == "raised"
    finally:
        coro.close()  # refused before it was scheduled, so the caller still owns it


async def test_calling_from_the_loop_thread_raises_instead_of_deadlocking():
    """The deadlock case. Waiting on the loop from the loop is an instant hang, and a
    hung request is far harder to diagnose than a raised error."""
    async def anything():
        return 1

    coro = anything()
    with bind_host_loop():
        with pytest.raises(HostLoopUnavailable, match="deadlock"):
            call_host_async(coro)
    coro.close()


async def test_bind_host_loop_outside_a_loop_says_so():
    def in_thread():
        with pytest.raises(HostLoopUnavailable, match="no running event loop"):
            with bind_host_loop():
                pass
        return "raised"

    assert await asyncio.to_thread(in_thread) == "raised"


# -- binding -------------------------------------------------------------------------

async def test_bind_host_loop_binds_and_unbinds():
    assert get_host_loop() is None
    with bind_host_loop() as loop:
        assert get_host_loop() is loop is asyncio.get_running_loop()
    assert get_host_loop() is None


async def test_the_binding_survives_the_hop_onto_a_thread():
    """A ``ContextVar`` is copied into a thread when the thread starts, which is the
    whole mechanism -- and the reason binding must happen before the hop."""
    with bind_host_loop() as loop:
        assert await asyncio.to_thread(get_host_loop) is loop


async def test_a_binding_made_after_the_hop_never_reaches_the_thread():
    """Documented as a rule; pinned here so it stays true.

    A host that binds inside the worker instead of around it gets ``None`` in the
    handler, and this is why.
    """
    started = threading.Event()
    release = threading.Event()
    seen = {}

    def worker():
        started.set()
        release.wait(2)
        seen["loop"] = get_host_loop()

    task = asyncio.ensure_future(asyncio.to_thread(worker))
    await asyncio.get_running_loop().run_in_executor(None, started.wait, 2)
    with bind_host_loop():
        release.set()
        await task

    assert seen["loop"] is None


async def test_run_conversation_async_binds_the_loop_for_the_turn(host_service_tool):
    """A host using the easy path needs no ``bind_host_loop`` of its own."""
    agent = build_agent()
    install_fake_client(agent, Script().calls(("host_call", {"value": "y"})).text("ok"))

    assert get_host_loop() is None
    await run_conversation_async(agent, "dale")
    assert get_host_loop() is None  # and it is unbound again afterwards
    assert host_service_tool["ran_on"] is asyncio.get_running_loop()


# -- timeout -------------------------------------------------------------------------

async def test_a_timeout_interrupts_the_agent_rather_than_only_abandoning_the_wait():
    """A thread cannot be killed, so a timeout that only stops waiting leaves the turn
    running -- still writing to the session store, still spending tokens."""
    interrupted = threading.Event()
    finish = threading.Event()

    class SlowAgent:
        def interrupt(self, message=None, **kwargs):
            interrupted.set()
            finish.set()
            return True

        def run_conversation(self, message, **kwargs):
            finish.wait(5)
            return {"completed": False}

    with pytest.raises(asyncio.TimeoutError):
        await run_conversation_async(SlowAgent(), "hola", timeout=0.1)

    assert interrupted.is_set()


async def test_an_agent_with_no_interrupt_still_times_out_cleanly():
    """``run_conversation`` is the only method this module requires; a host wrapper or a
    test double may not have ``interrupt``, and that must not turn a timeout into an
    ``AttributeError``."""
    finish = threading.Event()

    class Minimal:
        def run_conversation(self, message, **kwargs):
            finish.wait(1)
            return {"completed": True}

    try:
        with pytest.raises(asyncio.TimeoutError):
            await run_conversation_async(Minimal(), "hola", timeout=0.1)
    finally:
        finish.set()


async def test_a_turn_inside_the_timeout_returns_normally():
    agent = build_agent(enabled_toolsets=[])
    install_fake_client(agent, Script().text("a tiempo"))

    result = await run_conversation_async(agent, "hola", timeout=30)

    assert result["final_response"] == "a tiempo"


async def test_a_timed_out_host_call_cancels_its_coroutine():
    """Otherwise the coroutine keeps running on the host loop with nobody to read its
    result -- a leak that outlives the request that caused it."""
    import concurrent.futures

    cancelled = threading.Event()

    async def slow():
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    def in_thread():
        with pytest.raises(concurrent.futures.TimeoutError):
            call_host_async(slow(), timeout=0.05)
        return "timed out"

    with bind_host_loop():
        assert await asyncio.to_thread(in_thread) == "timed out"
        await asyncio.sleep(0.05)  # let the cancellation land on the loop

    assert cancelled.is_set()
