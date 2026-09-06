"""Adversarial coverage for the plugin engine, on top of ``tests/test_plugins.py``.

Where that file proves the authoring contract holds for a well-behaved plugin, this
file goes after the failure modes: a plugin that fights another plugin or a built-in
for a tool name, a plugin that registers something and then blows up, a hook that
raises or corrupts the payload it was handed, a directory that does not honor the
contract (name/directory mismatch, no ``__init__.py``), corrupted durable state, and
the unload/force-rediscovery gap that ``test_plugins.py`` already pins with a
trip-wire.

Everything here was run, not just reasoned about -- see the module docstring notes on
each test for what was verified experimentally before the assertion was written.
"""

import contextlib
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from hermes_core.runtime.lifecycle import invoke_hook
from hermes_core.runtime.plugins import discover_plugins, get_plugin_manager, unload_plugins
from hermes_core.runtime.plugins_state import PluginState
from hermes_core.seams.config import DictConfigSource, set_config_source
from hermes_core.seams.credentials import StaticCredentials, set_credential_source
from hermes_core.seams.paths import DirectoryWorkspace, set_workspace
from hermes_core.testing import Script, install_fake_client
from hermes_core.tools.registry import registry

FIXTURES = Path(__file__).parent / "fixtures" / "plugins"


def _install_plugin(workspace_root: Path, fixture_name: str, *, as_name: str | None = None) -> None:
    dest = workspace_root / "plugins" / (as_name or fixture_name)
    shutil.copytree(FIXTURES / fixture_name, dest)


@pytest.fixture
def workspace():
    root = Path(tempfile.mkdtemp())
    set_workspace(DirectoryWorkspace(root))
    set_credential_source(StaticCredentials("sk-test"))
    yield root
    # ``registry``'s per-scope tool overlay is process-wide and outlives this test;
    # ``current_scope_key()`` (derived from whichever workspace is set) stays pinned
    # to *this* test's temp dir until some other test calls set_workspace() again, so
    # an unrelated later test that checks "what tools exist" with no workspace of its
    # own would otherwise see this test's plugin tools. unload-all crashes on the
    # unguarded gateway import (that is exactly what several tests below are about),
    # but every registration is already disposed by the time it does -- so this
    # still tears down every tool/hook this test's plugins registered.
    with contextlib.suppress(Exception):
        unload_plugins()


def _enable(*plugin_ids: str) -> None:
    set_config_source(DictConfigSource({
        "model": {"default": "fake-model", "provider": "openai"},
        "plugins": {"enabled": list(plugin_ids)},
    }))


def build_agent(**overrides):
    from hermes_core.run_agent import AIAgent

    settings = dict(
        api_key="sk-test", base_url="https://example.invalid/v1", provider="openai",
        model="fake-model", enabled_toolsets=["e2e_hardening_tools"], quiet_mode=True, max_iterations=6,
    )
    settings.update(overrides)
    return AIAgent(**settings)


# -- a plugin fighting a built-in or another plugin for a tool name -----------------

def test_a_plugin_cannot_silently_override_a_built_in_tool_without_override_true(workspace):
    """Verified by running: register a global tool the way every real built-in does
    (``registry.register(..., scope=None)``, no plugin owner) *before* discovery, then
    load a plugin that tries to claim the same name without ``override=True``.

    The built-in wins: the plugin's ``register()`` does not raise, its *other* tool
    still registers, but the colliding registration is silently rejected (logged, not
    raised) and dispatch still reaches the original handler.
    """
    def _real_read_file(args, **_kwargs):
        from hermes_core.tools.registry import tool_result
        return tool_result(genuine=True)

    # ``registry`` is a process-wide singleton, unaffected by the per-test workspace
    # -- a global (scope=None) registration here would otherwise leak into every
    # later test in the process. Undo it unconditionally.
    registry.register(
        name="read_file", toolset="file",
        schema={"name": "read_file", "description": "the real one",
                "parameters": {"type": "object", "properties": {}}},
        handler=_real_read_file,
    )
    try:
        _install_plugin(workspace, "builtin_collision_plugin")
        _enable("builtin_collision_plugin")

        discover_plugins()  # must not raise despite the rejected collision

        assert json.loads(registry.dispatch("read_file", {})) == {"genuine": True}
        assert json.loads(registry.dispatch("collision_probe", {})) == {"probe": "alive"}
        plugins = {p["key"]: p for p in get_plugin_manager().list_plugins()}
        assert plugins["builtin_collision_plugin"]["enabled"] is True
        assert plugins["builtin_collision_plugin"]["error"] is None
        # Only the tool that actually won registration is credited.
        assert plugins["builtin_collision_plugin"]["tools"] == 1
    finally:
        registry.deregister("read_file")


def test_two_plugins_claiming_the_same_tool_name_the_first_loaded_wins(workspace):
    """Verified by running: ``dup_tool_plugin_a`` and ``dup_tool_plugin_b`` both declare
    ``shared_tool``; directory scanning is ``sorted(path.iterdir())``
    (``plugins_discovery.py``), so ``a`` loads before ``b`` deterministically.

    The second plugin's registration is rejected the same way a built-in collision
    is (logged, not raised) -- neither plugin's ``register()`` fails, and there is no
    exception anywhere in the path. This is safe, but worth pinning explicitly: a
    plugin's tool can be silently shadowed by another plugin loaded earlier, with no
    signal to the shadowed plugin's author beyond a log line.
    """
    _install_plugin(workspace, "dup_tool_plugin_a")
    _install_plugin(workspace, "dup_tool_plugin_b")
    _enable("dup_tool_plugin_a", "dup_tool_plugin_b")

    discover_plugins()  # must not raise

    assert json.loads(registry.dispatch("shared_tool", {})) == {"owner": "dup_tool_plugin_a"}
    plugins = {p["key"]: p for p in get_plugin_manager().list_plugins()}
    assert plugins["dup_tool_plugin_a"]["enabled"] is True
    assert plugins["dup_tool_plugin_a"]["tools"] == 1
    assert plugins["dup_tool_plugin_b"]["enabled"] is True
    assert plugins["dup_tool_plugin_b"]["error"] is None
    assert plugins["dup_tool_plugin_b"]["tools"] == 0  # its registration was rejected, not credited


# -- register-then-raise: no zombie tools ------------------------------------------

def test_a_tool_registered_just_before_register_raises_is_unwound(workspace):
    """``zombie_tool_plugin`` registers a real tool, then raises. Verified by running:
    the ledger's failure-path cleanup in ``_load_plugin_scoped`` disposes every
    registration attributed to the plugin before it is marked failed, so the tool
    never survives as a callable zombie in the ordinary registry."""
    _install_plugin(workspace, "zombie_tool_plugin")
    _enable("zombie_tool_plugin")

    discover_plugins()  # must not raise

    assert "zombie_tool" not in registry.get_all_tool_names()
    plugins = {p["key"]: p for p in get_plugin_manager().list_plugins()}
    assert plugins["zombie_tool_plugin"]["enabled"] is False
    assert "deliberately fails after registering a tool" in plugins["zombie_tool_plugin"]["error"]


# -- hooks: raising, and payload mutation --------------------------------------------

def test_a_raising_hook_callback_does_not_take_the_turn_down(workspace):
    """``hook_raiser_plugin`` records that it ran, then always raises. Verified by
    running: ``PluginManager.invoke_hook`` wraps each callback in its own
    ``try/except Exception`` (``plugins_dispatch.py``), so one raising observer never
    stops sibling observers or propagates to the caller."""
    _install_plugin(workspace, "demo_plugin")
    _install_plugin(workspace, "hook_raiser_plugin")
    _enable("demo_plugin", "hook_raiser_plugin")
    discover_plugins()

    # Must not raise, even though hook_raiser_plugin's callback always does.
    results = invoke_hook(
        "post_tool_call", tool_name="demo_greet", args={"name": "Ada"}, result='{"greeting": "Hello, Ada!"}',
        task_id="", session_id="", tool_call_id="", turn_id="", api_request_id="",
        duration_ms=5, status="ok", error_type=None, error_message=None, middleware_trace=[],
    )
    assert results == []

    # The raising callback still ran (and recorded that it ran) before it raised.
    assert PluginState("hook_raiser_plugin").get("saw_calls", 0) == 1
    # Its well-behaved sibling's hook was unaffected.
    assert PluginState("demo_plugin").get("observed_calls", []) == [
        {"tool_name": "demo_greet", "args": {"name": "Ada"}}
    ]


def test_a_hook_that_mutates_its_payload_corrupts_what_the_next_hook_sees(workspace):
    """``hook_mutator_plugin`` registers two ``post_tool_call`` callbacks: the first
    mutates the ``args`` dict it receives in place, the second records what it saw.

    Verified by running: ``PluginManager.invoke_hook`` passes every callback a view
    over the *same* ``kwargs`` dict (``_invoke_hook_callback`` builds a fresh
    top-level dict per call, but nested values -- ``args`` here -- are the identical
    object, not a copy). There is no per-callback isolation for payload data, only
    for exceptions: a callback that mutates a nested value in the hook payload
    corrupts what every callback registered after it observes for the same event,
    and what the caller's own dict looks like afterward. Two well-behaved plugins
    that both merely *read* ``args`` from a ``post_tool_call`` hook are not isolated
    from a third plugin that mutates it -- the property the plugin design is
    supposed to guarantee (one plugin cannot damage another) does not hold for hook
    payloads, only for hook exceptions.
    """
    _install_plugin(workspace, "hook_mutator_plugin")
    _enable("hook_mutator_plugin")
    discover_plugins()

    caller_owned_args = {"name": "Ada"}
    invoke_hook(
        "post_tool_call", tool_name="whatever", args=caller_owned_args, result="{}",
        task_id="", session_id="", tool_call_id="", turn_id="", api_request_id="",
        duration_ms=1, status="ok", error_type=None, error_message=None, middleware_trace=[],
    )

    # The mutation is visible on the caller's own dict after the fact...
    assert caller_owned_args == {"name": "Ada", "tampered_by": "hook_mutator_plugin"}
    # ...and the second callback observed the mutated value, not the original.
    observed = PluginState("hook_mutator_plugin").get("observed_args")
    assert observed == {"name": "Ada", "tampered_by": "hook_mutator_plugin"}


# -- directory contract violations ---------------------------------------------------

def test_manifest_name_not_directory_name_is_the_plugins_enabled_key(workspace):
    """``mismatched_name_plugin`` is the directory; ``plugin.yaml`` declares
    ``name: totally_different_manifest_name``. Verified by running: for a flat
    (non-category) plugin, ``parse_manifest_file`` sets ``manifest.key`` to the
    manifest's declared ``name`` field, not the directory basename
    (``plugins_manifest.py``: ``key = f"{prefix}/{plugin_dir.name}" if prefix else
    name`` -- ``name`` here is ``data.get("name", plugin_dir.name)``). So
    ``plugins.enabled`` must list the manifest's ``name:``, and enabling by the
    directory name it actually lives in silently does nothing -- discovered, gated
    off, no error surfaced beyond `hermes plugins list`'s reason string."""
    _install_plugin(workspace, "mismatched_name_plugin")
    _enable("totally_different_manifest_name")  # the manifest's name: field

    discover_plugins()

    assert "mismatch_probe" in registry.get_all_tool_names()
    plugins = {p["key"]: p for p in get_plugin_manager().list_plugins()}
    assert plugins["totally_different_manifest_name"]["enabled"] is True


def test_enabling_by_directory_name_instead_of_manifest_name_silently_no_ops(workspace):
    """The inverse of the above, same fixture: enabling the directory's own name
    (what an operator would naturally reach for) does not load the plugin at all --
    it is gated off as "not enabled in config", even though the directory it lives in
    is exactly that name."""
    _install_plugin(workspace, "mismatched_name_plugin")
    _enable("mismatched_name_plugin")  # the directory name, NOT the manifest's name:

    discover_plugins()  # must not raise

    assert "mismatch_probe" not in registry.get_all_tool_names()
    plugins = {p["key"]: p for p in get_plugin_manager().list_plugins()}
    # Gated off under its manifest-derived key, not the directory name.
    assert plugins["totally_different_manifest_name"]["enabled"] is False
    assert "not enabled in config" in plugins["totally_different_manifest_name"]["error"]


def test_a_plugin_directory_with_no_init_py_is_isolated_cleanly(workspace):
    """A directory with a valid ``plugin.yaml`` but no ``__init__.py`` at all.
    Verified by running: ``_load_directory_module`` raises ``FileNotFoundError``
    before any import is attempted, caught by the same per-plugin isolation as every
    other load failure -- discovery does not crash, the plugin is simply disabled
    with a clear error."""
    _install_plugin(workspace, "no_init_plugin")
    _enable("no_init_plugin")

    discover_plugins()  # must not raise

    plugins = {p["key"]: p for p in get_plugin_manager().list_plugins()}
    assert plugins["no_init_plugin"]["enabled"] is False
    assert "No __init__.py" in plugins["no_init_plugin"]["error"]


# -- ctx.state: persistence and corruption -------------------------------------------

def test_plugin_state_persists_across_the_plugin_being_unloaded(workspace):
    """Durable state is disk-backed, not manager-memory-backed: writing through
    ``ctx.state`` during a hook call, then unloading the plugin (targeted unload --
    see the unload/gateway tests below for why *that* is the only unload that
    reliably works), still leaves the value readable through a brand new
    ``PluginState`` instance afterward."""
    _install_plugin(workspace, "demo_plugin")
    _enable("demo_plugin")
    discover_plugins()

    invoke_hook(
        "post_tool_call", tool_name="demo_greet", args={"name": "Rosario"}, result="{}",
        task_id="", session_id="", tool_call_id="", turn_id="", api_request_id="",
        duration_ms=1, status="ok", error_type=None, error_message=None, middleware_trace=[],
    )
    assert PluginState("demo_plugin").get("observed_calls") != []

    unload_plugins("demo_plugin")  # targeted unload: does not touch the gateway import

    # The plugin is gone from the manager, but its durable state survives on disk.
    assert PluginState("demo_plugin").get("observed_calls") == [
        {"tool_name": "demo_greet", "args": {"name": "Rosario"}}
    ]


def test_corrupt_state_json_on_disk_raises_cleanly_instead_of_losing_data_silently(workspace):
    """Verified by running: both ``PluginState.get`` and ``.set`` raise a plain
    ``RuntimeError`` naming the file when the on-disk JSON cannot be parsed -- they
    do not swallow the corruption and return a default (which would silently lose
    whatever the caller thought was there), and they do not crash the process either.
    A plugin whose ``register()`` or hook touches state at the wrong moment would
    have that specific call fail through the ordinary per-plugin/per-callback
    isolation already proven elsewhere in this file, not the whole discovery pass."""
    state = PluginState("demo_plugin")
    state.set("observed_calls", [{"tool_name": "demo_greet", "args": {"name": "Ada"}}])
    state.path.write_text("{not valid json!!", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Cannot parse plugin state"):
        state.get("observed_calls", [])

    with pytest.raises(RuntimeError, match="Cannot parse plugin state"):
        state.set("observed_calls", [])


# -- the unload / force-rediscovery design question, settled by running code --------

def test_a_bare_unload_of_everything_leaves_the_engine_usable(workspace):
    """Unload-ALL is the primitive that was broken, not ``force=True`` specifically.

    ``unload_plugins()`` with no argument -- a plausible host-facing "reset the plugin
    engine" call, independent of discovery -- hit the same unguarded gateway import at
    ``plugins_ledger.py:234``, with zero platform plugins involved. Now guarded, so the
    teardown runs to completion: the tool goes away, and crucially ``self._discovered``
    is reset, which is what lets a later discovery find the plugin again.
    """
    _install_plugin(workspace, "demo_plugin")
    _enable("demo_plugin")
    discover_plugins()
    assert "demo_greet" in registry.get_all_tool_names()

    unload_plugins()  # plugin=None -> unload-all -> _reset_after_unload_all

    assert "demo_greet" not in registry.get_all_tool_names()

    # The half-finished teardown was the real damage: it left the manager believing it
    # was still discovered, so nothing could ever load again. A complete one recovers.
    discover_plugins()
    assert "demo_greet" in registry.get_all_tool_names()


def test_targeted_unload_of_one_plugin_does_not_touch_the_gateway_import(workspace):
    """The asymmetry that matters for a host's dev loop: unloading ONE named plugin
    never reaches ``_reset_after_unload_all`` (that only runs for unload-ALL), so it
    works today with no crash, cleanly removing the plugin's tools and its entry from
    ``list_plugins()``. Verified by running."""
    _install_plugin(workspace, "demo_plugin")
    _enable("demo_plugin")
    discover_plugins()
    assert "demo_greet" in registry.get_all_tool_names()

    found = unload_plugins("demo_plugin")

    assert found is True
    assert "demo_greet" not in registry.get_all_tool_names()
    assert get_plugin_manager().list_plugins() == []


def test_unload_all_then_rediscover_recovers_by_every_documented_path(workspace):
    """The consequence that made the unload bug severe, now checked from the other side.

    The crash used to happen mid-teardown, so ``self._discovered`` stayed True and the
    stale ``LoadedPlugin`` entries were never cleared: ``force=False`` silently no-oped
    forever and ``force=True`` crashed again, identically, forever. With no
    single-plugin reload API, a process restart was the only way back.

    So it is not enough that unload no longer raises. Both discovery paths have to
    actually bring the plugin back, which is what this checks.
    """
    _install_plugin(workspace, "demo_plugin")
    _enable("demo_plugin")
    discover_plugins()
    assert "demo_greet" in registry.get_all_tool_names()

    unload_plugins()
    discover_plugins(force=False)
    assert "demo_greet" in registry.get_all_tool_names(), "force=False must rescan after unload-all"

    unload_plugins()
    discover_plugins(force=True)
    assert "demo_greet" in registry.get_all_tool_names(), "force=True must rescan after unload-all"


# -- end to end: pushing a real turn harder ------------------------------------------

def test_a_failing_plugin_tool_and_a_malformed_result_do_not_take_the_turn_down(workspace):
    """``e2e_hardening_plugin`` offers ``fails_tool`` (raises inside its handler) and
    ``returns_bad_shape_tool`` (returns a raw dict instead of the ``tool_result()``
    JSON-string contract), plus a ``post_tool_call`` hook that records every call it
    observes. Verified by running: the model calls both tools across two turns, both
    degrade to a structured ``{"error": ...}`` tool message through the ordinary
    registry-level safety net (``ToolRegistry.dispatch`` / ``_normalize_handler_result``
    -- not plugin-specific, but exercised here through the plugin path end to end),
    the turn still completes normally, and the plugin's own hook sees ``status=
    "error"`` for both -- a failed plugin tool call is fully observable by a plugin's
    own hook without anything upstream going down."""
    _install_plugin(workspace, "e2e_hardening_plugin")
    _enable("e2e_hardening_plugin")
    discover_plugins()

    agent = build_agent()
    client = install_fake_client(
        agent,
        Script()
        .calls(("fails_tool", {}))
        .calls(("returns_bad_shape_tool", {}))
        .text("done"),
    )

    result = agent.run_conversation("go")

    assert result["completed"] is True
    assert result["final_response"] == "done"

    tool_messages = [m for m in client.last_request.messages if m.get("role") == "tool"]
    assert len(tool_messages) == 2
    fails_content = json.loads(tool_messages[0]["content"])
    assert "error" in fails_content
    assert "fails_tool deliberately raises" in fails_content["error"]
    bad_shape_content = json.loads(tool_messages[1]["content"])
    assert bad_shape_content["error_type"] == "tool_result_contract"

    observed = PluginState("e2e_hardening_plugin").get("observed", [])
    assert [o["tool_name"] for o in observed] == ["fails_tool", "returns_bad_shape_tool"]
    assert all(o["status"] == "error" for o in observed)
