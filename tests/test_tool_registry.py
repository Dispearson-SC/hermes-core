"""Tests for the lifted tool registry.

These are the first tests of code that came out of Hermes rather than code written
here, so they answer one question above all: does the tool machinery work outside
its original repository, with no Hermes home, no CLI, no gateway, and no config file?

They also pin the design rule the extraction depends on -- that the core registers
nothing on its own -- because that rule is what keeps a host from inheriting 156
bundled tools and, through three of them, the gateway.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from hermes_core.tools.registry import ToolRegistry, registry, tool_error, tool_result


@pytest.fixture
def reg():
    """A registry instance per test.

    The module exposes a process-wide singleton, which is right for an application
    and wrong for tests: state would leak between them. Constructing the class
    directly keeps each test isolated.
    """
    return ToolRegistry()


def echo_handler(args, **_kwargs):
    return tool_result(echoed=args.get("text", ""))


ECHO_SCHEMA = {
    "name": "echo",
    "description": "Echo the given text back.",
    "parameters": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
}


# -- the rule the extraction rests on ---------------------------------------------

def test_a_fresh_registry_is_empty():
    """The core must not auto-discover Hermes's bundled tool catalogue.

    Upstream imports every `tools/*.py` at startup, and three of those bundled tools
    reach into the gateway -- which is how a headless agent ends up loading 396
    modules. Registration here is explicit, so a host gets exactly the tools it asked
    for.
    """
    assert ToolRegistry().get_all_tool_names() == []


def test_the_core_ships_only_its_own_loop_tool():
    """One tool self-registers on import: `todo_list`.

    It is loop machinery rather than a capability -- the agent's own task list, which
    context compression reads and the turn loop treats as one of its internal tools
    (upstream groups it with `memory` and `session_search`, not with the shell or the
    browser). It carries no dependency on any surface.

    Asserted by name so the moment a *second* bundled tool starts registering itself,
    this fails. That is exactly the drift worth catching: it would mean the catalogue
    is creeping back in, and with it the surfaces some of those tools import.

    Measured in a subprocess, and that is the point. The claim is about *import side
    effects*, and `registry` is a process-wide singleton that every other test file in
    this suite legitimately writes to -- registering a demo tool, importing
    `skills_tool`. Read in-process, this test only held because it happened to run
    before them: under `-p randomly` it failed roughly one run in four, reporting
    other tests' tools as drift. A rule the whole extraction rests on cannot be
    guarded by a test that depends on collection order.
    """
    probe = textwrap.dedent(
        """
        import json
        from hermes_core.run_agent import AIAgent  # noqa: F401 -- the heaviest import path
        from hermes_core.tools.registry import registry
        print(json.dumps(sorted(registry.get_all_tool_names())))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    registered = set(json.loads(result.stdout.strip().splitlines()[-1]))

    assert registered <= {"todo_list"}, f"unexpected self-registering tools: {registered}"


# -- registration and lookup -------------------------------------------------------

def test_register_then_look_up(reg):
    reg.register(name="echo", toolset="demo", schema=ECHO_SCHEMA, handler=echo_handler)

    assert reg.get_all_tool_names() == ["echo"]
    assert reg.get_schema("echo") == ECHO_SCHEMA
    assert reg.get_toolset_for_tool("echo") == "demo"

    entry = reg.get_entry("echo")
    assert entry.name == "echo"
    assert entry.toolset == "demo"
    assert entry.handler is echo_handler


def test_an_unknown_tool_resolves_to_nothing(reg):
    assert reg.get_entry("nope") is None
    assert reg.get_schema("nope") is None


def test_deregister_removes_the_tool(reg):
    reg.register(name="echo", toolset="demo", schema=ECHO_SCHEMA, handler=echo_handler)
    reg.deregister("echo")

    assert reg.get_all_tool_names() == []
    assert reg.get_entry("echo") is None


def test_definitions_are_produced_in_openai_shape(reg):
    """What actually goes on the wire to the model."""
    reg.register(name="echo", toolset="demo", schema=ECHO_SCHEMA, handler=echo_handler)

    definitions = reg.get_definitions({"echo"}, quiet=True)

    assert len(definitions) == 1
    assert definitions[0]["type"] == "function"
    assert definitions[0]["function"]["name"] == "echo"
    assert definitions[0]["function"]["parameters"]["required"] == ["text"]


def test_definitions_only_include_what_was_asked_for(reg):
    """The enabled-toolset filter is applied by the caller, name by name."""
    reg.register(name="echo", toolset="demo", schema=ECHO_SCHEMA, handler=echo_handler)
    reg.register(
        name="other", toolset="demo",
        schema={"name": "other", "description": "x", "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **k: tool_result(),
    )

    names = [d["function"]["name"] for d in reg.get_definitions({"echo"}, quiet=True)]
    assert names == ["echo"]


# -- dispatch ----------------------------------------------------------------------

def test_dispatch_runs_the_handler(reg):
    reg.register(name="echo", toolset="demo", schema=ECHO_SCHEMA, handler=echo_handler)

    result = reg.dispatch("echo", {"text": "hola"})

    assert json.loads(result) == {"echoed": "hola"}


def test_dispatch_passes_extra_context_to_the_handler(reg):
    """Per-invocation context reaches the handler as keyword arguments.

    This is the seam a host uses to inject its own dependencies -- a repository, a
    notifier, the acting user -- without the tool importing them.
    """
    seen = {}

    def handler(args, **kwargs):
        seen.update(kwargs)
        return tool_result(ok=True)

    reg.register(
        name="ctx", toolset="demo",
        schema={"name": "ctx", "description": "x", "parameters": {"type": "object", "properties": {}}},
        handler=handler,
    )
    reg.dispatch("ctx", {}, session_id="s-1", actor="tester")

    assert seen["session_id"] == "s-1"
    assert seen["actor"] == "tester"


def test_dispatching_an_unknown_tool_returns_an_error_not_an_exception(reg):
    """A model naming a tool that does not exist is expected, not exceptional.

    Raising here would abort the turn; returning an error lets the loop feed the
    failure back and give the model a chance to correct itself.
    """
    result = reg.dispatch("does_not_exist", {})

    assert "error" in json.loads(result)


# -- availability gating -----------------------------------------------------------

def test_a_failing_check_fn_hides_the_tool(reg):
    """A tool whose backing service is down should not be offered to the model.

    Better than offering it and failing the call: the model never proposes an action
    that cannot succeed.
    """
    reg.register(
        name="gated", toolset="demo",
        schema={"name": "gated", "description": "x", "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **k: tool_result(),
        check_fn=lambda: False,
    )

    assert reg.get_definitions({"gated"}, quiet=True) == []


def test_a_passing_check_fn_keeps_the_tool(reg):
    reg.register(
        name="gated", toolset="demo",
        schema={"name": "gated", "description": "x", "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **k: tool_result(),
        check_fn=lambda: True,
    )

    assert len(reg.get_definitions({"gated"}, quiet=True)) == 1


# -- toolsets ----------------------------------------------------------------------

def test_tools_group_into_toolsets(reg):
    reg.register(name="a", toolset="alpha", schema={"name": "a", "description": "", "parameters": {"type": "object", "properties": {}}}, handler=lambda args, **k: "")
    reg.register(name="b", toolset="alpha", schema={"name": "b", "description": "", "parameters": {"type": "object", "properties": {}}}, handler=lambda args, **k: "")
    reg.register(name="c", toolset="beta", schema={"name": "c", "description": "", "parameters": {"type": "object", "properties": {}}}, handler=lambda args, **k: "")

    assert sorted(reg.get_tool_names_for_toolset("alpha")) == ["a", "b"]
    assert reg.get_tool_names_for_toolset("beta") == ["c"]
    assert "alpha" in reg.get_registered_toolset_names()


def test_a_toolset_alias_resolves(reg):
    """MCP servers register under `mcp-<server>` and alias the bare server name.

    The alias is what lets configuration say `notion` instead of `mcp-notion`.
    """
    reg.register(name="page_read", toolset="mcp-notion", schema={"name": "page_read", "description": "", "parameters": {"type": "object", "properties": {}}}, handler=lambda args, **k: "")
    reg.register_toolset_alias("notion", "mcp-notion")

    assert reg.get_toolset_alias_target("notion") == "mcp-notion"


# -- result helpers ----------------------------------------------------------------

def test_tool_result_encodes_keywords():
    assert json.loads(tool_result(status="ok", count=2)) == {"status": "ok", "count": 2}


def test_tool_result_encodes_a_dict():
    assert json.loads(tool_result({"status": "ok"})) == {"status": "ok"}


def test_tool_error_carries_extra_fields():
    payload = json.loads(tool_error("boom", code="E_TIMEOUT"))

    assert payload["error"] == "boom"
    assert payload["code"] == "E_TIMEOUT"


def test_tool_error_bounds_a_huge_message():
    """An unbounded exception string would bloat history on every retry."""
    payload = json.loads(tool_error("x" * 100_000))

    assert len(payload["error"]) < 100_000


def test_result_helpers_keep_non_ascii_readable():
    """`ensure_ascii=False`, so the model sees the text rather than escape sequences."""
    assert "ñandú" in tool_result(text="ñandú")


# -- isolation ---------------------------------------------------------------------

def test_registries_do_not_share_state(reg):
    other = ToolRegistry()
    reg.register(name="echo", toolset="demo", schema=ECHO_SCHEMA, handler=echo_handler)

    assert other.get_all_tool_names() == []
