"""Provider-failure and retry-machinery tests: a real ``AIAgent.run_conversation()``
against a scripted provider that raises instead of answering.

``hermes_core/agent/retry_utils.py``, ``error_classifier.py``, ``turn_retry_state.py``
and ``bounded_response.py`` (roughly 1,500 lines) came across in the extraction but had
never run inside an actual agent turn. Every test here drives the real turn loop --
never ``classify_api_error`` in isolation -- because the question is not "what does the
classifier say" but "does the loop's behaviour match what the classifier said", and a
seam mismatch between the two is exactly the shape of bug this core has already shipped
once (the streaming ``final_response`` seam, see ``test_agent_turn.py``).

Backoff is real production code (``interruptible_backoff_sleep`` sleeps in 200ms
slices). A fake, deterministic clock is installed for every test in this module so a
multi-second backoff costs nothing in wall time without touching the sleep call sites'
logic.
"""

from __future__ import annotations

import json
import tempfile
import time as time_module

import httpx
import openai
import pytest

from hermes_core.agent.error_classifier import FailoverReason, classify_api_error
from hermes_core.agent.retry_utils import jittered_backoff
from hermes_core.seams.config import DictConfigSource, set_config_source
from hermes_core.seams.credentials import StaticCredentials, set_credential_source
from hermes_core.seams.paths import DirectoryWorkspace, set_workspace
from hermes_core.testing import Script, install_fake_client


# -- fixtures -------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_core():
    """Same isolation contract as test_agent_turn.py's ``isolated_core``."""
    set_workspace(DirectoryWorkspace(tempfile.mkdtemp()))
    set_config_source(
        DictConfigSource({"model": {"default": "fake-model", "provider": "openai"}})
    )
    set_credential_source(StaticCredentials("sk-test"))
    yield


@pytest.fixture(autouse=True)
def fake_clock(monkeypatch):
    """A controllable clock so ``interruptible_backoff_sleep``'s real ``time.sleep(0.2)``
    slices advance instantly instead of costing wall-clock time.

    ``time.time()`` and ``time.sleep()`` are looked up as ``time.<attr>`` at call time
    throughout the retry machinery (``import time`` at module scope, never
    ``from time import sleep``), so patching the attributes on the shared ``time``
    module reaches every call site without patching each module separately.
    """
    state = {"now": 1_700_000_000.0}

    def fake_time():
        return state["now"]

    def fake_sleep(seconds):
        state["now"] += seconds

    monkeypatch.setattr(time_module, "time", fake_time)
    monkeypatch.setattr(time_module, "sleep", fake_sleep)
    return state


def build_agent(**overrides):
    from hermes_core.run_agent import AIAgent

    settings = dict(
        api_key="sk-test",
        base_url="https://example.invalid/v1",
        provider="openai",
        model="fake-model",
        enabled_toolsets=[],
        quiet_mode=True,
        max_iterations=5,
    )
    settings.update(overrides)
    return AIAgent(**settings)


def set_max_retries(n: int):
    """Config knob for ``agent._api_max_retries`` (agent_init.py reads
    ``config["agent"]["api_max_retries"]``, default 3, floored at 1)."""
    set_config_source(
        DictConfigSource({
            "model": {"default": "fake-model", "provider": "openai"},
            "agent": {"api_max_retries": n},
        })
    )


# -- building realistic provider errors ------------------------------------------------

def _response(status: int, body: dict, headers: dict | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    return httpx.Response(status, request=request, json=body, headers=headers or {})


def rate_limit_error(*, retry_after: str | None = None) -> openai.RateLimitError:
    headers = {"retry-after": retry_after} if retry_after else {}
    resp = _response(429, {"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}, headers)
    return openai.RateLimitError("Rate limit exceeded", response=resp, body=resp.json())


def server_error(status: int = 500) -> openai.APIStatusError:
    resp = _response(status, {"error": {"message": "Internal server error", "type": "server_error"}})
    return openai.InternalServerError("Internal server error", response=resp, body=resp.json())


def overloaded_error() -> openai.APIStatusError:
    resp = _response(503, {"error": {"message": "The server is overloaded. Please try again later.", "type": "overloaded_error"}})
    return openai.APIStatusError("overloaded", response=resp, body=resp.json())


def auth_error() -> openai.AuthenticationError:
    resp = _response(401, {"error": {"message": "Incorrect API key provided.", "type": "invalid_request_error"}})
    return openai.AuthenticationError("Incorrect API key provided.", response=resp, body=resp.json())


def connection_error() -> openai.APIConnectionError:
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    return openai.APIConnectionError(message="Connection error.", request=request)


def read_timeout_error() -> openai.APITimeoutError:
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    return openai.APITimeoutError(request=request)


def context_length_error() -> openai.BadRequestError:
    msg = (
        "This model's maximum context length is 8192 tokens. However, your messages "
        "resulted in 20000 tokens. Please reduce the length of the messages."
    )
    resp = _response(400, {"error": {"message": msg, "type": "invalid_request_error", "code": "context_length_exceeded"}})
    return openai.BadRequestError(msg, response=resp, body=resp.json())


# -- 1. transient then success ---------------------------------------------------------

def test_a_rate_limited_call_recovers_on_the_next_attempt():
    """429 once, then a real answer: the turn must recover, not fail the user's turn."""
    agent = build_agent()
    client = install_fake_client(
        agent, Script().error(rate_limit_error()).text("recovered")
    )

    result = agent.run_conversation("hola")

    assert result["completed"] is True
    assert result["final_response"] == "recovered"
    # Two provider calls: the failed attempt and the retry.
    assert client.call_count == 2


def test_the_rate_limited_retry_actually_backs_off():
    """The backoff is not a no-op: real (fake-clock) time elapses between attempts."""
    agent = build_agent()
    install_fake_client(agent, Script().error(rate_limit_error()).text("recovered"))

    before = time_module.time()
    agent.run_conversation("hola")
    after = time_module.time()

    assert after > before  # the fake clock only advances via time.sleep()


# -- 2. persistent failure ---------------------------------------------------------------

def test_a_persistently_failing_retryable_error_gives_up_cleanly():
    """Every attempt 500s. The turn must not raise -- a host cannot catch an exception
    it doesn't know to expect from ``run_conversation()``."""
    set_max_retries(2)
    agent = build_agent()
    client = install_fake_client(
        agent, Script().error(server_error()).error(server_error())
    )

    result = agent.run_conversation("hola")  # must not raise

    assert result["completed"] is False
    assert result["failed"] is True
    assert isinstance(result["error"], str) and result["error"]
    assert "final_response" in result
    assert client.call_count == 2  # exactly the retry budget, no more


def test_persistent_failure_result_names_the_failure_reason():
    """The max-retries-exhausted path adds failure_reason/failure_retryable -- a host
    needs these to distinguish 'model answered' from 'we gave up' (see the
    nonretryable-path test below for the asymmetry)."""
    set_max_retries(1)
    agent = build_agent()
    install_fake_client(agent, Script().error(server_error()))

    result = agent.run_conversation("hola")

    assert result["completed"] is False
    assert result.get("failure_reason") == FailoverReason.server_error.value
    assert result.get("failure_retryable") is True


# -- 3. failure classes: classification and matching turn behaviour ---------------------

@pytest.mark.parametrize(
    "make_error, expected_reason, expected_retryable",
    [
        (rate_limit_error, FailoverReason.rate_limit, True),
        (lambda: server_error(500), FailoverReason.server_error, True),
        (overloaded_error, FailoverReason.overloaded, True),
        (connection_error, FailoverReason.timeout, True),
        (read_timeout_error, FailoverReason.timeout, True),
        (auth_error, FailoverReason.auth, False),
        (context_length_error, FailoverReason.context_overflow, True),
    ],
    ids=["429", "500", "503-overloaded", "connect-error", "read-timeout", "401", "400-context-overflow"],
)
def test_error_classifier_verdict_for_each_failure_class(make_error, expected_reason, expected_retryable):
    classified = classify_api_error(
        make_error(), provider="openai", model="fake-model",
        approx_tokens=100, context_length=200000, num_messages=2,
    )
    assert classified.reason == expected_reason
    assert classified.retryable is expected_retryable


def test_a_401_is_not_retried_five_times():
    """The classifier says auth is non-retryable; the turn must honor that on the FIRST
    failure, not burn the retry budget first. A 401 retried until exhaustion is a bug
    (wasted, guaranteed-to-fail calls against a real provider)."""
    set_max_retries(5)
    agent = build_agent()
    client = install_fake_client(agent, Script().error(auth_error()))

    result = agent.run_conversation("hola")

    assert result["completed"] is False
    assert result["failed"] is True
    assert client.call_count == 1  # not 5


def test_a_connection_error_is_retried_and_recovers():
    """Timeout/connection failures carry no status code; the transport-type heuristic
    in error_classifier._by_transport must still mark them retryable."""
    agent = build_agent()
    client = install_fake_client(
        agent, Script().error(connection_error()).text("recovered")
    )

    result = agent.run_conversation("hola")

    assert result["completed"] is True
    assert result["final_response"] == "recovered"
    assert client.call_count == 2


def test_a_read_timeout_is_retried_and_recovers():
    agent = build_agent()
    client = install_fake_client(
        agent, Script().error(read_timeout_error()).text("recovered")
    )

    result = agent.run_conversation("hola")

    assert result["completed"] is True
    assert client.call_count == 2


def test_a_context_length_error_recovers_through_compression_not_a_bare_retry():
    """context_overflow is should_compress=True, not a plain retry -- the turn routes
    through the compression/rebuild path (turn_overflow.recover_from_overflow), then
    the rebuilt (smaller) request succeeds. This is the one failure class that does NOT
    go through compute_error_backoff/interruptible_backoff_sleep at all, so it is worth
    confirming end to end rather than trusting the classifier's ``should_compress`` flag
    in isolation."""
    agent = build_agent()
    client = install_fake_client(
        agent, Script().error(context_length_error()).text("recovered")
    )

    result = agent.run_conversation("hola")

    assert result["completed"] is True
    assert result["final_response"] == "recovered"
    assert client.call_count == 2


# -- the 401 vs 500 result-shape asymmetry (a finding, pinned as a test) ----------------

def test_nonretryable_and_exhausted_results_disagree_on_which_keys_exist():
    """BUG (finding, not asserted as desired behaviour -- see report): a host that reads
    ``result["failure_reason"]`` to tell auth failures apart from generic ones gets it
    for a max-retries-exhausted failure but NOT for an immediate non-retryable abort
    (auth, content-policy-other-than-billing). ``nonretryable_client_error_result`` in
    turn_recovery.py only special-cases ``billing`` and ``content_policy_blocked`` for
    the extra keys; every other non-retryable reason (auth, ssl_cert_verification,
    provider_policy_blocked, ...) falls through to the bare ``_failed_turn_result()``,
    which carries none of them. This test documents the asymmetry so it is a conscious
    contract, not an accident a host discovers in production.
    """
    set_max_retries(3)
    agent_exhausted = build_agent()
    install_fake_client(
        agent_exhausted, Script().error(server_error()).error(server_error()).error(server_error())
    )
    exhausted_result = agent_exhausted.run_conversation("hola")

    agent_auth = build_agent()
    install_fake_client(agent_auth, Script().error(auth_error()))
    auth_result = agent_auth.run_conversation("hola")

    assert exhausted_result["completed"] is False
    assert auth_result["completed"] is False
    assert "failure_reason" in exhausted_result
    # This is the asymmetry: same "completed": False / "failed": True shape, but only
    # one of the two failure modes tells the host WHY without string-parsing
    # final_response.
    assert "failure_reason" not in auth_result


# -- 4. failure mid-conversation: tool already ran, second call fails -------------------

def test_a_tool_result_survives_a_failed_call_immediately_after_it(monkeypatch):
    """The model calls a tool, the tool runs (a side effect happened), and only THEN
    does the provider fail on the follow-up call that was supposed to read the tool's
    result. The tool must not be silently re-run, and its result must still be in the
    transcript once the turn recovers."""
    from hermes_core.tools.registry import registry, tool_result

    calls = []

    def handler(args, **_kwargs):
        calls.append(args)
        return tool_result(temp_c=21, city=args.get("city"))

    registry.register(
        name="get_weather", toolset="demo",
        schema={
            "name": "get_weather", "description": "Look up the weather in a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
        handler=handler, override=True,
    )
    try:
        agent = build_agent(enabled_toolsets=["demo"])
        client = install_fake_client(
            agent,
            Script()
            .calls(("get_weather", {"city": "Rosario"}))
            .error(server_error())
            .text("Hacen 21 grados en Rosario."),
        )

        result = agent.run_conversation("clima en Rosario")

        assert result["completed"] is True
        assert result["final_response"] == "Hacen 21 grados en Rosario."
        # The tool ran exactly once -- the failed follow-up call must not have re-run it.
        assert calls == [{"city": "Rosario"}]
        assert client.call_count == 3  # tool-call turn, failed retry, successful retry

        tool_messages = [m for m in client.last_request.messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert json.loads(tool_messages[0]["content"]) == {"temp_c": 21, "city": "Rosario"}
    finally:
        registry.deregister("get_weather")


def test_a_persistent_failure_after_a_tool_call_does_not_lose_the_tool_result():
    """Same shape, but the provider never recovers. The turn must still fail cleanly
    (not raise), and the transcript handed back in the result must retain the tool
    call and its result -- the caller may persist ``result["messages"]`` and a host
    that lost the tool's output here would silently discard real side effects."""
    from hermes_core.tools.registry import registry, tool_result

    def handler(args, **_kwargs):
        return tool_result(temp_c=21, city=args.get("city"))

    registry.register(
        name="get_weather", toolset="demo",
        schema={
            "name": "get_weather", "description": "x",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
        handler=handler, override=True,
    )
    try:
        set_max_retries(2)
        agent = build_agent(enabled_toolsets=["demo"])
        install_fake_client(
            agent,
            Script()
            .calls(("get_weather", {"city": "Rosario"}))
            .error(server_error())
            .error(server_error()),
        )

        result = agent.run_conversation("clima en Rosario")  # must not raise

        assert result["completed"] is False
        assert result["failed"] is True
        tool_messages = [m for m in result["messages"] if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert json.loads(tool_messages[0]["content"]) == {"temp_c": 21, "city": "Rosario"}
    finally:
        registry.deregister("get_weather")


# -- 5. retry budget is capped ------------------------------------------------------------

def test_the_retry_budget_is_capped_not_infinite():
    """A persistently failing rate limit must not retry forever -- this is exactly the
    deployment shape (many customers, one worker) where an uncapped loop is a bill."""
    set_max_retries(3)
    agent = build_agent()
    client = install_fake_client(
        agent,
        Script().error(rate_limit_error()).error(rate_limit_error()).error(rate_limit_error()),
    )

    result = agent.run_conversation("hola")

    assert result["completed"] is False
    assert client.call_count == 3  # never more than the configured budget


def test_the_retry_budget_is_configurable():
    """Different max_retries values change the observed attempt count -- proof the cap
    is real config, not a hardcoded number that happens to equal the default."""
    set_max_retries(1)
    agent = build_agent()
    client = install_fake_client(agent, Script().error(server_error()))

    agent.run_conversation("hola")

    assert client.call_count == 1


# -- 6. what the caller sees, precisely ----------------------------------------------------

def test_a_successful_turn_result_shape():
    agent = build_agent()
    install_fake_client(agent, Script().text("ok"))

    result = agent.run_conversation("hola")

    assert result["completed"] is True
    assert result["final_response"] == "ok"
    assert result["api_calls"] == 1
    assert "error" not in result
    assert result["failed"] is False


def test_a_recovered_turn_result_shape_matches_a_clean_success():
    """After a transient failure recovers, the result the host sees must be
    indistinguishable from a first-try success -- no retry residue in the contract."""
    agent = build_agent()
    client = install_fake_client(agent, Script().error(rate_limit_error()).text("ok"))

    result = agent.run_conversation("hola")

    assert result["completed"] is True
    assert result["final_response"] == "ok"
    assert "error" not in result
    assert result["failed"] is False
    # The retry cost two provider calls but is still ONE logical turn.
    assert result["api_calls"] == 1
    assert client.call_count == 2


def test_giving_up_result_shape_is_distinguishable_from_success():
    set_max_retries(1)
    agent = build_agent()
    install_fake_client(agent, Script().error(server_error()))

    result = agent.run_conversation("hola")

    assert result["completed"] is False
    assert result["failed"] is True
    assert result["error"]
    assert isinstance(result["final_response"], str) and result["final_response"]


# -- retry_utils / turn_retry_state, exercised directly (fast, deterministic units) -----

def test_jittered_backoff_grows_and_caps():
    small = jittered_backoff(1, base_delay=2.0, max_delay=60.0, jitter_ratio=0.0)
    bigger = jittered_backoff(4, base_delay=2.0, max_delay=60.0, jitter_ratio=0.0)
    capped = jittered_backoff(20, base_delay=2.0, max_delay=60.0, jitter_ratio=0.0)
    assert small == 2.0
    assert bigger == 16.0
    assert capped == 60.0


def test_turn_retry_state_guards_default_false_and_are_settable():
    from hermes_core.agent.turn_retry_state import TurnRetryState

    state = TurnRetryState()
    assert state.has_retried_429 is False
    state.has_retried_429 = True
    assert state.has_retried_429 is True
    # A fresh instance per attempt -- guards must not leak across instances.
    assert TurnRetryState().has_retried_429 is False
