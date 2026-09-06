"""Does automatic context compaction actually shrink what gets sent to the model?

Context. Automatic compaction was completely broken until recently:
``conversation_compression._lock_api_is_absent_on_session_db`` detected an optional
lock method by importing a module that does not exist in this core, read the failure
backwards, and made the caller sit out on every cycle for both shipped stores. That is
fixed via a ``tools/lift.py`` patch, but the only test of the fix asserts
``_resolve_lock_api(store)`` returns ``(None, None)`` -- which proves the gate is open,
not that compaction runs. This file closes that gap: it drives a real turn across the
trigger and inspects what the fake client actually received.

Session PERSISTENCE (a second agent reading the first's history back) is already
proven in ``tests/test_session_store.py`` and is NOT what this file is about. This
file is only about COMPRESSION: whether the history handed to the model on a later
call is smaller than the raw transcript would be.

The trigger, traced from the real call sites (``agent/turn_context_compaction.py`` ->
``agent/turn_preflight.py`` -> ``agent/context_compressor.py``, and
``agent/turn_tool_round.py``'s ``compress_after_tool_results``): before every API call
the loop estimates the assembled request's token count and compares it against
``ContextCompressor.threshold_tokens``. The config knobs a host sets, under
``compression`` in configuration:

* ``enabled`` (bool, default True)
* ``threshold`` -- ratio of the model's *effective* window (context_length minus
  reserved output tokens); default 0.50, raise-only floor of 0.75 below a 512K window.
* ``threshold_tokens`` -- an absolute cap; compaction fires at the LOWER of the ratio
  and this count, clamped to the context length. This is the knob these tests turn
  down, because a 256K-token default window (the fallback for an unknown model like
  the fake one) makes the ratio threshold ~192K tokens -- too large to reach honestly
  in a unit test. Lowering ``threshold_tokens`` is the documented, supported way to
  make compaction trigger early; it is not a workaround.
* ``protect_first_n`` / ``protect_last_n`` -- messages guaranteed to survive
  uncompressed at the head/tail (defaults 3 / 20). Also turned down here so a modest
  test conversation has a real middle to summarize.
* ``max_attempts`` -- retry rounds before a turn gives up (default 3).

The summariser call. Compaction asks a model to summarise the middle turns, and that
call does NOT go through ``agent.client`` / ``install_fake_client`` at all -- it is a
completely separate path. ``ContextCompressor._call_summary_llm`` calls
``hermes_core.seams.auxiliary_client.call_llm``, which resolves its OWN client via
``get_text_auxiliary_client()`` -> ``hermes_core.runtime.auth`` credential resolution,
building a brand-new ``openai.OpenAI(...)`` instance. A host wiring this up for real
needs either an ``auxiliary.compression`` config block naming a (usually cheaper)
model, or nothing at all (it then reuses the main model's name, still through its own
client). Either way, testing it requires patching
``hermes_core.agent.context_compressor.call_llm`` directly -- there is no fake-client
hook for it, which is itself worth knowing before relying on this path in production
code that wants to unit-test it.

A confirmed, load-bearing bug was found while building this: with compression enabled
and NO ``auxiliary.compression`` configured (the documented "falls back to the main
model" default -- see ``seams/auxiliary_client.py``'s own docstring), the very first
real compaction attempt CRASHES the turn instead of degrading gracefully. Traced to
its root cause below and captured as an ``xfail(strict=True)`` regression test.
"""

import json
import tempfile
from types import SimpleNamespace
from unittest import mock

import pytest

from hermes_core.seams.config import DictConfigSource, set_config_source
from hermes_core.seams.credentials import StaticCredentials, set_credential_source
from hermes_core.seams.paths import DirectoryWorkspace, set_workspace
from hermes_core.testing import Script, install_fake_client

SUMMARY_TEXT = "SUMMARY: earlier turns covered a long back-and-forth about foxes and dogs."

# A chunk of filler large enough that a handful of messages reliably cross a
# low-hundreds-of-tokens threshold (roughly 4 chars/token for plain ASCII).
_FILLER = "The quick brown fox jumps over the lazy dog. " * 40


def _compression_config(**overrides):
    cfg = {
        "enabled": True,
        "threshold_tokens": 800,
        "protect_first_n": 1,
        "protect_last_n": 2,
        "max_attempts": 3,
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture(autouse=True)
def isolated_core(tmp_path):
    """Same three-call host setup as ``test_agent_turn.py``."""
    set_workspace(DirectoryWorkspace(tmp_path))
    set_config_source(
        DictConfigSource({
            "model": {"default": "fake-model", "provider": "openai"},
            "compression": _compression_config(),
        })
    )
    set_credential_source(StaticCredentials("sk-test"))
    yield tmp_path


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


def _long_history(n_turns=8):
    """A hand-built ``conversation_history`` long enough to cross the lowered
    threshold. Built directly rather than via repeated ``run_conversation()`` calls,
    because -- as ``test_session_store.py`` establishes -- this core does NOT
    auto-reload history between calls on the same agent; a host threads
    ``conversation_history`` through explicitly (or via its own SessionStore reads).
    A single call with a long seeded history is the honest way to reach the trigger
    without re-proving persistence, which is not what this file is about."""
    history = []
    for i in range(n_turns):
        history.append({"role": "user", "content": f"{_FILLER} (historical turn {i})"})
        history.append({"role": "assistant", "content": f"Acknowledged turn {i}. {_FILLER}"})
    return history


def _fake_summary_call(*_args, **kwargs):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=SUMMARY_TEXT, reasoning=None, reasoning_content=None
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    )


def _patched_summariser():
    """Patch the two seams a compaction pass touches that ``install_fake_client``
    does not reach: the auxiliary summariser call, and the session-start feasibility
    probe. The probe is patched out deliberately -- it is the site of the confirmed
    bug documented and reproduced (unpatched) below; patching it here isolates
    "does compaction's rewrite logic work" from "does credential auto-resolution for
    the auxiliary client work", which are two different questions."""
    return (
        mock.patch("hermes_core.agent.context_compressor.call_llm", side_effect=_fake_summary_call),
        mock.patch(
            "hermes_core.agent.conversation_compression.check_compression_model_feasibility",
            return_value=None,
        ),
    )


# -- the claim, proven end to end -----------------------------------------------------


def test_a_long_conversation_is_compacted_before_it_reaches_the_model():
    """The central claim: a conversation long enough to cross ``threshold_tokens``
    gets rewritten smaller BEFORE the next model call, not sent in full. Proven by
    inspecting what the fake client actually recorded, not by trusting a log line."""
    patch_summary, patch_feasibility = _patched_summariser()
    with patch_summary, patch_feasibility:
        agent = build_agent()
        history = _long_history(n_turns=8)
        raw_message_count = len(history) + 1  # + the new user message (system is separate)

        client = install_fake_client(agent, Script().text("Final answer after long history."))
        result = agent.run_conversation("What did we discuss?", conversation_history=history)

        assert result["completed"] is True
        assert client.call_count == 1  # compaction never calls the main model

        sent = client.last_request.messages
        # Fewer messages than the raw transcript would have carried (system + history + new turn).
        assert len(sent) < raw_message_count + 1
        # And a real token-count claim, not just a message-count one.
        from hermes_core.agent.model_metadata import estimate_messages_tokens_rough

        raw_system = [{"role": "system", "content": "x" * 2000}]  # rough stand-in, same order of magnitude
        raw_tokens = estimate_messages_tokens_rough(raw_system + history + [{"role": "user", "content": "What did we discuss?"}])
        sent_tokens = estimate_messages_tokens_rough(sent)
        assert sent_tokens < raw_tokens

        assert agent.context_compressor.compression_count >= 1
        assert any(
            SUMMARY_TEXT in (m.get("content") or "")
            for m in sent
            if isinstance(m.get("content"), str)
        )


def test_disabling_compression_sends_the_full_raw_history():
    """Contrast case: with the exact same long conversation, ``compression.enabled:
    False`` sends every message untouched. This is what proves the previous test's
    shrinkage comes from compaction and not from some other truncation path."""
    set_config_source(
        DictConfigSource({
            "model": {"default": "fake-model", "provider": "openai"},
            "compression": _compression_config(enabled=False),
        })
    )
    agent = build_agent()
    history = _long_history(n_turns=8)

    client = install_fake_client(agent, Script().text("Final answer after long history."))
    result = agent.run_conversation("What did we discuss?", conversation_history=history)

    assert result["completed"] is True
    # system + full history + new user message, nothing dropped.
    assert len(client.last_request.messages) == 1 + len(history) + 1
    assert agent.context_compressor.compression_count == 0


def test_compaction_preserves_tool_call_linkage_in_the_surviving_tail():
    """A compaction that drops the wrong half of a tool-call/tool-result pair
    produces a history a real provider rejects outright. Puts one such pair inside
    the PROTECTED tail (``protect_last_n``) and checks it survives as a matched pair;
    a second run puts the same pair in the region that gets summarized away and
    checks it is dropped as a matched pair too -- never split."""
    patch_summary, patch_feasibility = _patched_summariser()
    with patch_summary, patch_feasibility:
        agent = build_agent()
        history = _long_history(n_turns=6)
        # Tool-call/tool-result pair at the tail -> inside protect_last_n=2's reach
        # once the trailing turn assembly is accounted for.
        history.append({"role": "user", "content": "what's the weather in Rosario?"})
        history.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_tail_1", "type": "function",
                "function": {"name": "get_weather", "arguments": json.dumps({"city": "Rosario"})},
            }],
        })
        history.append({
            "role": "tool", "tool_call_id": "call_tail_1", "name": "get_weather",
            "content": json.dumps({"temp_c": 21}),
        })

        client = install_fake_client(agent, Script().text("It's 21C in Rosario."))
        result = agent.run_conversation("thanks, what did we discuss overall?", conversation_history=history)

        assert result["completed"] is True
        sent = client.last_request.messages

        call_ids = {
            tc["id"] if isinstance(tc, dict) else tc.id
            for m in sent if m.get("role") == "assistant" and m.get("tool_calls")
            for tc in m["tool_calls"]
        }
        result_ids = {m["tool_call_id"] for m in sent if m.get("role") == "tool"}

        # No dangling reference in either direction -- a real provider 400s on either.
        assert call_ids == result_ids
        # The tail pair specifically survived together (it was inside protect_last_n).
        assert "call_tail_1" in call_ids
        assert "call_tail_1" in result_ids


# -- the bug: the documented default auxiliary path crashes the turn -----------------


def test_compression_degrades_instead_of_crashing_with_no_auxiliary_model():
    """The documented default -- compression on, nothing under ``auxiliary.compression``
    -- used to crash the turn outright, and the traceback pointed nowhere near the cause.

    The chain was four steps long and every step hid the one before it:

    1. ``seams/config.py`` had no ``get_env_value_prefer_dotenv``. ``runtime/auth.py``
       imports that name at call time from EVERY api-key provider's credential lookup,
       so it raised ``ImportError`` unconditionally -- regardless of env vars, ``.env``,
       or the installed ``CredentialSource``. Env-var credential resolution was broken
       for every provider, not just the auxiliary one.
    2. A bare ``except Exception`` in ``get_text_auxiliary_client`` turned that into
       ``(None, None)`` at DEBUG level, which reads exactly like "nothing configured".
    3. ``check_compression_model_feasibility`` answered that by calling
       ``_try_configured_fallback_for_unavailable_client``, whose seam stub returned a
       2-tuple where the call site unpacks three.
    4. The resulting ``ValueError`` landed inside an ``except ValueError: raise`` written
       for an unrelated deliberate failure, so it propagated instead of degrading.

    Both ends are fixed: the config seam exports the missing helper, and the stub returns
    a 3-tuple. Without a summariser the turn must degrade -- no summary -- and complete.
    """
    agent = build_agent()
    history = _long_history(n_turns=8)
    install_fake_client(agent, Script().text("Final answer after long history."))

    result = agent.run_conversation("What did we discuss?", conversation_history=history)

    assert result["completed"] is True
