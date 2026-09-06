"""Host dependencies reaching a tool handler, without the tool importing them.

The gap these close: a tool is a plain function plus a schema, and until ``extras`` the
only way one could reach a host's repository, notifier or acting user was to import them
directly -- which couples the tool to one application and makes it untestable alone. The
core already passed ``task_id``/``session_id``/``user_task`` down that exact path; this
opens it to the host.

Every test that claims a handler receives something drives a *real turn*, because the
path being tested is the one the turn loop takes (``conversation_loop`` →
``tool_executor`` → ``model_tools._execute_tool`` → ``registry.dispatch``). Calling
``registry.dispatch`` directly would prove only that ``dispatch`` forwards keyword
arguments, which it always did -- and would have passed even before this existed.
"""

import tempfile
import threading

import pytest

from hermes_core.seams import _active
from hermes_core.seams.config import DictConfigSource, set_config_source
from hermes_core.seams.context import RESERVED_TOOL_KWARGS, AgentContext
from hermes_core.seams.credentials import StaticCredentials, set_credential_source
from hermes_core.seams.paths import DirectoryWorkspace, set_workspace
from hermes_core.testing import Script, install_fake_client
from hermes_core.tools.registry import registry, tool_result


@pytest.fixture(autouse=True)
def process_defaults():
    set_workspace(DirectoryWorkspace(tempfile.mkdtemp(prefix="extras-")))
    set_config_source(DictConfigSource({"model": {"default": "fake-model", "provider": "openai"}}))
    set_credential_source(StaticCredentials("key-default"))
    yield


def make_context(**changes) -> AgentContext:
    base = AgentContext(
        workspace=DirectoryWorkspace(tempfile.mkdtemp(prefix="ctx-")),
        config=DictConfigSource({"model": {"default": "fake-model", "provider": "openai"}}),
        credentials=StaticCredentials("key-ctx"),
    )
    return base.derive(**changes) if changes else base


@pytest.fixture
def recording_tool():
    """A registered tool that records every keyword argument it is handed."""
    seen = []

    def handler(args, **kwargs):
        seen.append(kwargs)
        return tool_result(ok=True)

    registry.register(
        name="record_kwargs",
        toolset="demo",
        schema={
            "name": "record_kwargs",
            "description": "Record the context a tool is called with.",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=handler,
        override=True,
    )
    yield seen
    registry.deregister("record_kwargs")


def run_one_tool_turn():
    """Drive a real turn whose only step is calling ``record_kwargs``."""
    from hermes_core.run_agent import AIAgent

    agent = AIAgent(
        api_key="sk-test", base_url="https://example.invalid/v1", provider="openai",
        model="fake-model", enabled_toolsets=["demo"], quiet_mode=True, max_iterations=3,
    )
    install_fake_client(agent, Script().calls(("record_kwargs", {})).text("listo"))
    return agent.run_conversation("usá la herramienta")


# -- the feature ---------------------------------------------------------------------

def test_extras_reach_a_tool_handler_during_a_real_turn(recording_tool):
    """The claim, end to end: what the host put on the context is what the tool gets."""
    repo = object()
    notifier = object()

    with make_context(extras={"repo": repo, "notifier": notifier, "actor": "u-42"}).activate():
        result = run_one_tool_turn()

    assert result["completed"] is True
    assert len(recording_tool) == 1
    kwargs = recording_tool[0]
    assert kwargs["repo"] is repo
    assert kwargs["notifier"] is notifier
    assert kwargs["actor"] == "u-42"


def test_the_cores_own_keyword_arguments_still_arrive(recording_tool):
    """``extras`` adds to what the core passes; it does not replace it."""
    with make_context(extras={"repo": "R"}).activate():
        run_one_tool_turn()

    kwargs = recording_tool[0]
    assert kwargs["repo"] == "R"
    assert "task_id" in kwargs
    assert "session_id" in kwargs


def test_a_turn_outside_any_context_is_unchanged(recording_tool):
    """The single-agent case that never asked for any of this keeps working."""
    run_one_tool_turn()

    kwargs = recording_tool[0]
    assert "task_id" in kwargs
    assert "session_id" in kwargs
    assert "repo" not in kwargs


def test_a_context_without_extras_adds_nothing(recording_tool):
    with make_context().activate():
        run_one_tool_turn()

    assert set(recording_tool[0]) <= RESERVED_TOOL_KWARGS


# -- reserved names ------------------------------------------------------------------

@pytest.mark.parametrize("reserved", sorted(RESERVED_TOOL_KWARGS))
def test_a_reserved_keyword_is_rejected_when_the_context_is_built(reserved):
    """Rejected at construction, not mid-turn.

    A host that shadowed ``session_id`` would hand its own value to a handler with every
    reason to trust the core's, and the handler could not tell the difference. Caught
    where the error can still name the mistake.
    """
    with pytest.raises(ValueError) as excinfo:
        make_context(extras={reserved: "mine"})

    assert reserved in str(excinfo.value)


def test_the_rejection_names_every_collision_and_suggests_a_fix():
    with pytest.raises(ValueError) as excinfo:
        make_context(extras={"session_id": 1, "task_id": 2, "repo": 3})

    message = str(excinfo.value)
    assert "session_id" in message and "task_id" in message
    assert "repo" not in message
    assert "host_" in message  # points at a rename rather than just refusing


def test_derive_revalidates_rather_than_inheriting_a_pass():
    """``derive`` is the usual way a request-scoped context is built, so it is the usual
    place a reserved name would slip in."""
    base = make_context(extras={"repo": "R"})

    with pytest.raises(ValueError):
        base.derive(extras={"repo": "R", "user_task": "mine"})


def test_the_core_keys_win_even_if_a_host_object_smuggles_one_in(recording_tool):
    """The backstop behind the constructor check.

    ``AgentContext`` rejects reserved names, but the merge reads ``extras`` off whatever
    object is active by attribute -- so a host with its own context class could still
    present one. The merge order, not the constructor, is what guarantees a handler's
    ``session_id`` is the core's.
    """
    real = make_context()

    class HostContext:
        """A host's own context class: the seams still find the four ports on it, but
        its ``extras`` never went through ``AgentContext``'s constructor."""

        workspace = real.workspace
        config = real.config
        credentials = real.credentials
        session_store = real.session_store
        extras = {"session_id": "forged", "repo": "R"}

    token = _active.current.set(HostContext())
    try:
        run_one_tool_turn()
    finally:
        _active.current.reset(token)

    kwargs = recording_tool[0]
    assert kwargs["repo"] == "R"
    assert kwargs["session_id"] != "forged"


# -- immutability --------------------------------------------------------------------

def test_extras_cannot_be_edited_through_the_context():
    """A frozen dataclass whose one mutable field could be edited in place is only half
    frozen -- and this one is read from another thread on every tool call."""
    context = make_context(extras={"repo": "R"})

    with pytest.raises(TypeError):
        context.extras["repo"] = "other"


def test_mutating_the_dict_that_was_passed_in_does_not_reach_the_context():
    supplied = {"repo": "R"}
    context = make_context(extras=supplied)

    supplied["repo"] = "other"
    supplied["late"] = "addition"

    assert context.extras["repo"] == "R"
    assert "late" not in context.extras


def test_a_handler_cannot_corrupt_the_context_for_the_next_call(recording_tool):
    """Handlers get a fresh dict per call, so one that mutates its kwargs -- popping a
    key, adding a marker -- cannot affect the next tool call or another tenant."""
    with make_context(extras={"repo": "R"}).activate():
        run_one_tool_turn()
        recording_tool[0]["repo"] = "corrupted"
        recording_tool[0]["injected"] = True
        run_one_tool_turn()

    assert recording_tool[1]["repo"] == "R"
    assert "injected" not in recording_tool[1]


# -- isolation -----------------------------------------------------------------------

def test_two_tenants_in_one_process_each_see_their_own_extras(recording_tool):
    """The reason this rides on ``AgentContext`` instead of a module global."""
    with make_context(extras={"repo": "acme"}).activate():
        run_one_tool_turn()
    with make_context(extras={"repo": "globex"}).activate():
        run_one_tool_turn()

    assert [k["repo"] for k in recording_tool] == ["acme", "globex"]


def test_concurrent_threads_do_not_share_extras(recording_tool):
    """Two requests served at the same time, which is the case a global gets wrong."""
    seen = {}
    barrier = threading.Barrier(2)

    def serve(tenant):
        with make_context(extras={"repo": tenant}).activate():
            barrier.wait()  # both contexts active at once, so a global would collide
            seen[tenant] = _active.get_tool_extras()

    threads = [threading.Thread(target=serve, args=(t,)) for t in ("acme", "globex")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert seen == {"acme": {"repo": "acme"}, "globex": {"repo": "globex"}}


def test_the_extras_reader_returns_a_copy_not_the_contexts_own_mapping():
    """The merge writes the core's keys into what it gets back; it must not be writing
    into the context every later call reads."""
    context = make_context(extras={"repo": "R"})

    with context.activate():
        first = _active.get_tool_extras()
        first["session_id"] = "written-by-the-merge"

        assert _active.get_tool_extras() == {"repo": "R"}


def test_the_extras_reader_is_empty_with_no_context():
    assert _active.get_tool_extras() == {}


# -- a context that carries only extras ----------------------------------------------
#
# The single-tenant shape: ports configured once at startup, only host dependencies vary
# per request. Requiring all four fields made that host restate its own startup
# configuration on every request just to pass `extras` -- which is how a rarely-changed
# value ends up copied into a request handler and quietly drifts from the real one.

def test_a_context_can_carry_extras_and_nothing_else(recording_tool):
    from hermes_core.seams.context import AgentContext

    with AgentContext(extras={"repo": "R"}).activate():
        run_one_tool_turn()

    assert recording_tool[0]["repo"] == "R"


def test_an_unset_port_falls_through_to_the_process_wide_default():
    """Unset means *fall through*, not "no workspace" -- the difference between a host
    that configured once at startup and a host that is broken."""
    from hermes_core.seams.config import load_config
    from hermes_core.seams.context import AgentContext
    from hermes_core.seams.credentials import resolve_credentials
    from hermes_core.seams.paths import get_workspace

    outside = (get_workspace(), load_config(), resolve_credentials("openai").api_key)

    with AgentContext(extras={"repo": "R"}).activate():
        inside = (get_workspace(), load_config(), resolve_credentials("openai").api_key)

    assert inside == outside


def test_one_port_can_be_overridden_while_the_others_fall_through():
    """The mixed case, which is the reason this is per-field rather than all-or-nothing:
    a tenant that needs its own credentials but shares everything else."""
    from hermes_core.seams.context import AgentContext
    from hermes_core.seams.credentials import StaticCredentials, resolve_credentials
    from hermes_core.seams.paths import get_workspace

    outside_workspace = get_workspace()

    with AgentContext(credentials=StaticCredentials("key-tenant")).activate():
        assert resolve_credentials("openai").api_key == "key-tenant"
        assert get_workspace() is outside_workspace

    assert resolve_credentials("openai").api_key == "key-default"


def test_a_fully_specified_context_still_overrides_everything():
    """The multi-tenant shape must not have regressed to falling through."""
    from hermes_core.seams.config import load_config
    from hermes_core.seams.credentials import resolve_credentials

    context = make_context()

    with context.activate():
        assert load_config()["model"]["default"] == "fake-model"
        assert resolve_credentials("openai").api_key == "key-ctx"
