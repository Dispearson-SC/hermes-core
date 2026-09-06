"""The published surface, checked against itself.

``hermes_core.__init__`` names what this core commits to, and resolves each name lazily
through a module ``__getattr__``. Lazy means *unverified*: a name whose module path is
wrong, or whose object was renamed in the module it points at, raises nothing until the
first host reaches for it. These tests reach for all of them.

They also pin the laziness, because it is load-bearing rather than a nicety: a host that
only wants the tool registry must not pay for the transports, and a cycle inside one
lifted module must not be able to turn ``import hermes_core`` into an error.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import hermes_core
from hermes_core import _EXPORTS

CORE_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("name", sorted(_EXPORTS))
def test_every_published_name_resolves(name):
    """The whole point of the list: a name on it can be imported.

    Parametrised so a failure says *which* name broke, rather than stopping at the
    first one and hiding the rest.
    """
    assert getattr(hermes_core, name) is not None


def test_all_matches_the_export_table():
    assert set(hermes_core.__all__) == set(_EXPORTS) | {"__version__"}


def test_dir_lists_the_surface_for_tab_completion():
    assert set(hermes_core.__dir__()) == set(hermes_core.__all__)


def test_an_unknown_name_raises_attribute_error_not_something_stranger():
    """``__getattr__`` intercepts every miss, so a typo must still look like a typo."""
    with pytest.raises(AttributeError, match="unknown_name"):
        hermes_core.unknown_name


def test_the_entry_point_is_on_the_list():
    """``AIAgent`` was reachable only as ``hermes_core.run_agent.AIAgent`` for a while --
    an internal import path, which is exactly what this list exists to replace."""
    assert "AIAgent" in _EXPORTS
    assert hermes_core.AIAgent.__name__ == "AIAgent"


# -- the laziness ---------------------------------------------------------------------

def test_importing_the_package_does_not_drag_the_agent_in():
    """Measured in a subprocess: the claim is about import side effects, and every other
    test in this suite has already imported half the core by the time it runs."""
    probe = textwrap.dedent(
        """
        import json, sys
        import hermes_core
        heavy = [m for m in sys.modules if m.startswith("hermes_core.") and m not in (
            "hermes_core.seams", "hermes_core.seams._active",
        )]
        print(json.dumps(heavy))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=CORE_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-2000:]
    loaded = __import__("json").loads(result.stdout.strip().splitlines()[-1])

    assert loaded == [], f"import hermes_core pulled in {loaded}"


def test_reaching_for_one_name_does_not_pull_in_the_rest():
    """A host that wants the registry should not pay for the transports."""
    probe = textwrap.dedent(
        """
        import json, sys
        from hermes_core import registry  # noqa: F401
        print(json.dumps(any(m.startswith("hermes_core.agent.transports") for m in sys.modules)))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=CORE_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "false"


def test_a_resolved_name_is_cached_as_a_real_attribute():
    """``__getattr__`` fires once; the second access is a plain lookup, not another
    import."""
    import hermes_core as fresh

    fresh.tool_result  # resolve it
    assert "tool_result" in vars(fresh)
