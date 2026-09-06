"""The plugin engine, proven end to end.

``hermes_core.runtime.plugins`` (plus its ``plugins_manifest``/``plugins_discovery``/
``plugins_loader``/``plugins_dispatch``/``plugins_ledger``/``plugins_state`` split, and
the ``lifecycle``/``middleware`` hook contracts) is lifted verbatim from upstream. The
one piece the extraction has to supply is *where discovery looks*: upstream hardcodes
``~/.hermes``, this core resolves the same directories through
``hermes_core.seams.paths`` (``get_hermes_home()``), which a host controls with
``set_workspace(DirectoryWorkspace(path))``.

These tests establish that the wiring holds: a plugin directory placed under
``<workspace>/plugins/<name>/`` is discovered and loaded, a tool it registers is
callable through the ordinary tool registry, a hook it registers actually fires with
the payload the hook contract promises, one plugin's failure never takes down the
others, and a malformed manifest is rejected instead of crashing discovery.

Author's contract for a directory plugin, exercised by ``tests/fixtures/plugins/``:

* ``<dir>/plugin.yaml`` -- at least ``name``; ``version``, ``description``,
  ``provides_tools``, ``provides_hooks`` are declarative and optional.
* ``<dir>/__init__.py`` -- exposes ``register(ctx)``, called once at discovery.
* ``ctx`` is a ``PluginContext``: ``register_tool(name, toolset, schema, handler)``
  puts a tool in the ordinary registry; ``register_hook(hook_name, callback)`` (hook
  names are ``hermes_core.runtime.plugins.VALID_HOOKS``, e.g. ``post_tool_call``)
  subscribes an observer; ``ctx.state`` is durable per-plugin JSON key/value storage.
* The plugin must be opted in via ``plugins.enabled`` in the host's configuration --
  discovering a directory is not the same as loading it.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from hermes_core.runtime.lifecycle import invoke_hook
from hermes_core.runtime.plugins import discover_plugins, get_plugin_manager
from hermes_core.runtime.plugins_state import PluginState
from hermes_core.seams.config import DictConfigSource, set_config_source
from hermes_core.seams.credentials import StaticCredentials, set_credential_source
from hermes_core.seams.paths import DirectoryWorkspace, set_workspace
from hermes_core.testing import Script, install_fake_client
from hermes_core.tools.registry import registry

FIXTURES = Path(__file__).parent / "fixtures" / "plugins"


def _install_plugin(workspace_root: Path, fixture_name: str, *, as_name: str | None = None) -> None:
    """Copy a fixture plugin directory under ``<workspace_root>/plugins/``."""
    dest = workspace_root / "plugins" / (as_name or fixture_name)
    shutil.copytree(FIXTURES / fixture_name, dest)


@pytest.fixture
def workspace():
    """A scratch workspace, isolated per test by construction.

    ``PluginManager`` and ``ToolRegistry`` both key their state off
    ``hermes_home_key()`` (see ``hermes_core.seams.paths``), so a fresh temp directory
    per test is enough isolation -- no explicit plugin-manager reset needed.
    """
    root = Path(tempfile.mkdtemp())
    set_workspace(DirectoryWorkspace(root))
    set_credential_source(StaticCredentials("sk-test"))
    yield root


def _enable(*plugin_ids: str) -> None:
    set_config_source(DictConfigSource({
        "model": {"default": "fake-model", "provider": "openai"},
        "plugins": {"enabled": list(plugin_ids)},
    }))


# -- discovery + loading ---------------------------------------------------------------

def test_a_plugin_directory_is_discovered_and_loaded(workspace):
    _install_plugin(workspace, "demo_plugin")
    _enable("demo_plugin")

    discover_plugins()

    plugins = {p["key"]: p for p in get_plugin_manager().list_plugins()}
    assert "demo_plugin" in plugins
    assert plugins["demo_plugin"]["enabled"] is True
    assert plugins["demo_plugin"]["error"] is None


def test_force_rediscovery_works_without_a_gateway_installed(workspace):
    """Reloading a plugin without restarting the process -- the development inner loop.

    This was broken, and expensively so. ``discover_and_load(force=True)`` calls
    ``unload()`` first, and ``_reset_after_unload_all`` did an unguarded ``from
    gateway.platform_registry import platform_registry`` -- upstream's gateway package,
    which this core does not carry. It raised on every unload-all, with zero platform
    plugins involved, and it raised *mid-teardown*: registrations already disposed, the
    ``self._discovered`` reset never reached. Afterwards ``force=False`` found nothing
    and ``force=True`` raised again, for the life of the process.

    ``tools/lift.py`` now guards that import exactly the way upstream guards its own
    ``tools.registry`` sibling eight lines below it in the same function.
    """
    _install_plugin(workspace, "demo_plugin")
    _enable("demo_plugin")
    discover_plugins()
    assert "demo_greet" in registry.get_all_tool_names()

    discover_plugins(force=True)

    # The point is not that it did not raise -- it is that the plugin came back. A
    # teardown that half-ran would leave the tool deregistered and never reload it.
    assert "demo_greet" in registry.get_all_tool_names()
    assert "demo_plugin" in {p["key"] for p in get_plugin_manager().list_plugins()}


def test_a_directory_with_no_plugin_yaml_is_ignored(workspace):
    (workspace / "plugins" / "not_a_plugin").mkdir(parents=True)
    (workspace / "plugins" / "not_a_plugin" / "readme.txt").write_text("nothing to see here")
    _enable()

    discover_plugins()  # must not raise

    assert get_plugin_manager().list_plugins() == []


# -- tool half of the contract -----------------------------------------------------

def test_a_plugin_registered_tool_appears_in_the_registry_and_is_callable(workspace):
    _install_plugin(workspace, "demo_plugin")
    _enable("demo_plugin")

    discover_plugins()

    assert "demo_greet" in registry.get_all_tool_names()
    result = json.loads(registry.dispatch("demo_greet", {"name": "Ada"}))
    assert result == {"greeting": "Hello, Ada!"}


def test_a_disabled_plugin_registers_nothing(workspace):
    _install_plugin(workspace, "demo_plugin")
    _enable()  # nothing enabled

    discover_plugins()

    assert "demo_greet" not in registry.get_all_tool_names()
    plugins = {p["key"]: p for p in get_plugin_manager().list_plugins()}
    assert plugins["demo_plugin"]["enabled"] is False


# -- hook half of the contract ------------------------------------------------------

def test_a_plugin_registered_hook_fires_with_the_promised_payload(workspace):
    _install_plugin(workspace, "demo_plugin")
    _enable("demo_plugin")
    discover_plugins()

    invoke_hook(
        "post_tool_call", tool_name="demo_greet", args={"name": "Ada"}, result='{"greeting": "Hello, Ada!"}',
        task_id="", session_id="", tool_call_id="", turn_id="", api_request_id="",
        duration_ms=5, status="ok", error_type=None, error_message=None, middleware_trace=[],
    )

    observed = PluginState("demo_plugin").get("observed_calls", [])
    assert observed == [{"tool_name": "demo_greet", "args": {"name": "Ada"}}]


# -- isolation and error handling ----------------------------------------------------

def test_a_plugin_that_raises_on_load_is_isolated(workspace):
    """Upstream's guard: one plugin's register() blowing up must not stop the others."""
    _install_plugin(workspace, "demo_plugin")
    _install_plugin(workspace, "broken_plugin")
    _enable("demo_plugin", "broken_plugin")

    discover_plugins()  # must not raise

    plugins = {p["key"]: p for p in get_plugin_manager().list_plugins()}
    assert plugins["broken_plugin"]["enabled"] is False
    assert "broken_plugin deliberately fails to load" in plugins["broken_plugin"]["error"]
    # The well-behaved sibling is unaffected.
    assert plugins["demo_plugin"]["enabled"] is True
    assert "demo_greet" in registry.get_all_tool_names()


def test_a_malformed_manifest_is_rejected_without_crashing(workspace):
    _install_plugin(workspace, "demo_plugin")
    _install_plugin(workspace, "malformed_plugin")
    _enable("demo_plugin", "malformed_plugin")

    discover_plugins()  # must not raise despite the unparsable YAML

    plugins = {p["key"]: p for p in get_plugin_manager().list_plugins()}
    # The malformed manifest never became a plugin at all -- parse failure drops it
    # before gating/loading, so its register() (which would assert) never runs.
    assert "malformed_plugin" not in plugins
    assert plugins["demo_plugin"]["enabled"] is True


# -- end to end: a real turn -----------------------------------------------------------

def build_agent(**overrides):
    from hermes_core.run_agent import AIAgent

    settings = dict(
        api_key="sk-test", base_url="https://example.invalid/v1", provider="openai",
        model="fake-model", enabled_toolsets=["demo_plugin_tools"], quiet_mode=True, max_iterations=5,
    )
    settings.update(overrides)
    return AIAgent(**settings)


def test_a_plugin_tool_is_called_during_a_real_turn_and_the_hook_observes_it(workspace):
    """The claim end to end: the model calls a plugin-registered tool mid-turn, the
    tool runs through the ordinary dispatch path, and the plugin's post_tool_call hook
    observes the exact call the model made -- with no Hermes CLI/gateway involved."""
    _install_plugin(workspace, "demo_plugin")
    _enable("demo_plugin")
    discover_plugins()

    agent = build_agent()
    client = install_fake_client(
        agent,
        Script().calls(("demo_greet", {"name": "Rosario"})).text("Ya la saludé."),
    )

    result = agent.run_conversation("saludá a Rosario")

    assert result["completed"] is True
    assert result["final_response"] == "Ya la saludé."

    tool_messages = [m for m in client.last_request.messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert json.loads(tool_messages[0]["content"]) == {"greeting": "Hello, Rosario!"}

    observed = PluginState("demo_plugin").get("observed_calls", [])
    assert len(observed) == 1
    assert observed[0]["tool_name"] == "demo_greet"
    assert observed[0]["args"] == {"name": "Rosario"}
