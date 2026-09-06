"""The first-party observability hook: absent by design, and quiet about it.

Upstream forwards every lifecycle event to Nous Portal's relay telemetry through
``hermes_cli/observability/``. This core strips that deliberately, so there is no
observer here -- which makes *absence the normal state*, and upstream's handling of it
wrong for this core: both call sites sit under a bare ``except Exception`` that logs a
warning with a full traceback.

That cost 21 warnings, each with a stack trace, in a single turn that called no tools.
Not a crash -- but noise at that volume teaches whoever reads the logs to filter out
``hermes_core.runtime.lifecycle``, which is the one logger a *real* observer failure
would use. A warning nobody reads is worse than no warning.

There was a second, sharper edge underneath it. The lift pruned the files out of
``hermes_core/runtime/observability/`` and left the directory, and a directory with no
``__init__.py`` is a valid **namespace package**: the import succeeds and yields a module
with nothing in it. So the error was ``ImportError: cannot import name ... (unknown
location)`` rather than ``ModuleNotFoundError`` -- the shape a caller guarding an
optional import does not expect to catch.
"""

import tempfile

import pytest

from hermes_core.seams.config import DictConfigSource, set_config_source
from hermes_core.seams.credentials import StaticCredentials, set_credential_source
from hermes_core.seams.paths import DirectoryWorkspace, set_workspace
from hermes_core.testing import Script, install_fake_client


@pytest.fixture(autouse=True)
def isolated_core():
    set_workspace(DirectoryWorkspace(tempfile.mkdtemp(prefix="lifecycle-")))
    set_config_source(DictConfigSource({"model": {"default": "fake-model", "provider": "openai"}}))
    set_credential_source(StaticCredentials("sk-test"))
    yield


def run_a_turn():
    from hermes_core.run_agent import AIAgent

    agent = AIAgent(
        api_key="sk-test", base_url="https://example.invalid/v1", provider="openai",
        model="fake-model", quiet_mode=True, max_iterations=3,
    )
    install_fake_client(agent, Script().text("hola"))
    return agent.run_conversation("hola")


# -- the noise -------------------------------------------------------------------

def test_a_turn_logs_no_observability_warnings(caplog):
    """The regression guard, stated as a count because the bug was one of volume."""
    with caplog.at_level("WARNING", logger="hermes_core.runtime.lifecycle"):
        result = run_a_turn()

    assert result["completed"] is True
    observability_warnings = [r for r in caplog.records if "observability" in r.getMessage().lower()]
    assert observability_warnings == [], f"{len(observability_warnings)} warnings back"


def test_hooks_still_dispatch_with_no_observer_installed():
    """Quiet must not mean broken: plugin hooks are the other half of this dispatcher and
    have to keep working when the first-party observer is absent."""
    from hermes_core.runtime.lifecycle import has_hook, invoke_hook

    assert invoke_hook("on_session_start", session_id="s-1") == []
    assert has_hook("on_session_start") is False


# -- the namespace-package trap ----------------------------------------------------

def test_the_observability_module_is_absent_rather_than_empty():
    """``ModuleNotFoundError``, not an importable module with nothing in it.

    An empty directory would still import, and the difference is exactly what a caller
    guarding an optional dependency expects to catch.
    """
    with pytest.raises(ModuleNotFoundError):
        __import__("hermes_core.runtime.observability")


def test_no_package_directory_is_left_empty():
    """The general form, so the next pruned package cannot repeat it.

    ``prune_orphans`` deletes files; a directory left behind with no ``__init__.py``
    becomes a namespace package that shadows the honest "not found".
    """
    from pathlib import Path

    import hermes_core

    root = Path(hermes_core.__file__).parent
    empty = [
        directory.relative_to(root).as_posix()
        for directory in root.rglob("*")
        if directory.is_dir()
        and directory.name != "__pycache__"
        and not any(child.name != "__pycache__" for child in directory.iterdir())
    ]

    assert empty == [], f"empty namespace packages: {empty}"


# -- the extension point -----------------------------------------------------------

def test_a_host_can_install_its_own_observer(monkeypatch):
    """Absence is the default, not the only option: dropping in a module with these two
    names is how a host gets its own telemetry, and it is why the probe is a lookup
    rather than a hard-coded ``return``."""
    import sys
    import types

    from hermes_core.runtime import lifecycle

    seen = []
    observer = types.ModuleType("hermes_core.runtime.observability")
    observer.observe_lifecycle = lambda hook_name, **kwargs: seen.append(hook_name)
    observer.handles_hook = lambda hook_name: hook_name == "on_session_start"

    monkeypatch.setitem(sys.modules, "hermes_core.runtime.observability", observer)
    monkeypatch.setattr(lifecycle, "_OBSERVER_MISSING", False)

    lifecycle.invoke_hook("on_session_start", session_id="s-1")

    assert seen == ["on_session_start"]
    assert lifecycle.has_hook("on_session_start") is True
