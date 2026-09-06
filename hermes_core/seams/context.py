"""One object holding everything an agent depends on, so several can share a process.

The four seams -- workspace, configuration, credentials, session store -- are proper
ports: each is a ``Protocol`` a host implements however it likes. What they were not,
until this module, was *injected*. Each one was installed into a module-level global by
``set_workspace`` / ``set_config_source`` / ``set_credential_source``, which makes them
service locators. One process could serve exactly one configuration.

That is fine for a single agent in its own process and fatal for anything else. Two
tenants configured in one worker do not coexist; the second silently replaces the
first, including its API key:

    set_config_source(DictConfigSource({"model": {"default": "modelo-A"}}))
    set_credential_source(StaticCredentials("key-A"))
    set_config_source(DictConfigSource({"model": {"default": "modelo-B"}}))
    set_credential_source(StaticCredentials("key-B"))
    load_config()   # modelo-B, for the tenant that asked for modelo-A

An :class:`AgentContext` bundles the four and binds them to the running thread or
asyncio task rather than to the process::

    acme = AgentContext(
        workspace=DirectoryWorkspace("/var/lib/app/tenants/acme"),
        config=DictConfigSource({"model": {"default": "MiniMax-M2.7"}}),
        credentials=StaticCredentials(acme_api_key),
        session_store=SqliteSessionStore("/var/lib/app/tenants/acme/sessions.db"),
    )

    with acme.activate():
        agent = AIAgent(..., session_db=acme.session_store)
        agent.run_conversation("hola")

Everything inside that block reads Acme's settings; another tenant's block, running
concurrently, reads its own. The seam getters still work unchanged outside any block,
falling back to whatever ``set_*`` installed -- so the simple single-agent case needs
none of this and keeps working exactly as before.

**A context is not a security boundary.** It stops tenants from tripping over each
other; it does not stop code that goes looking. Anything that reads the environment
directly, or holds a path from before the switch, is outside its reach. Real isolation
between untrusted tenants is a process boundary, and this does not replace one.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Optional

from hermes_core.seams import _active
from hermes_core.seams.config import ConfigSource
from hermes_core.seams.credentials import CredentialSource
from hermes_core.seams.paths import Workspace
from hermes_core.seams.session_store import SessionStore

__all__ = ["AgentContext", "get_active_context", "activate", "RESERVED_TOOL_KWARGS"]

#: Keyword arguments the core itself passes to every tool handler. ``extras`` may not
#: use these names: a host that shadowed ``session_id`` would hand its own value to a
#: handler that had every reason to trust the core's, and the handler could not tell.
#: Checked when a context is built, so the error names the mistake instead of surfacing
#: as wrong data three layers down mid-turn.
RESERVED_TOOL_KWARGS = frozenset({"task_id", "session_id", "user_task", "enabled_tools"})


@dataclass(frozen=True)
class AgentContext:
    """Everything one agent -- or one tenant's set of agents -- depends on.

    Frozen, because a context is identity: a tenant's settings should not change under
    an agent mid-turn. :meth:`derive` makes a modified copy for the cases that need one.

    Every field is optional, and they are optional for two different reasons.

    ``workspace``, ``config`` and ``credentials`` are read through module-level seam
    functions by lifted code that cannot be handed an argument -- which is exactly why
    they need a context to be *found* through. Unset means the seam falls through to
    whatever ``set_workspace`` / ``set_config_source`` / ``set_credential_source``
    installed process-wide.

    ``session_store`` sits apart: it is already passed to ``AIAgent(session_db=...)``
    directly, so it is carried here for completeness -- a host hands one object around
    instead of four -- rather than because anything looks it up.
    """

    #: The three ports read through module-level seam functions. Each is optional, and
    #: leaving one unset means *fall through to the process-wide default*, not "no
    #: workspace" -- so the two shapes a host actually has both work:
    #:
    #:     # multi-tenant: everything varies per tenant
    #:     AgentContext(workspace=..., config=..., credentials=..., session_store=...)
    #:
    #:     # single-tenant: ports configured once at startup, only dependencies vary
    #:     set_workspace(...); set_config_source(...); set_credential_source(...)
    #:     AgentContext(extras={"repo": repo, "actor": user.id})
    #:
    #: Requiring all three made the second shape restate its own startup configuration
    #: on every request just to pass ``extras``, which is how a rarely-changed value
    #: ends up copied into a request handler and quietly drifts.
    workspace: Optional[Workspace] = None
    config: Optional[ConfigSource] = None
    credentials: Optional[CredentialSource] = None
    session_store: Optional[SessionStore] = None

    #: Optional label for logs and diagnostics. Never used for lookup or isolation --
    #: two contexts with the same name are still two contexts.
    name: str = ""

    #: Host dependencies handed to every tool handler as keyword arguments, for the
    #: whole time this context is active.
    #:
    #: The gap this closes: a tool is a plain function plus a schema, and the only way
    #: it could reach a host's repository, notifier or acting user was to import them --
    #: which couples the tool to one application and makes it untestable on its own. The
    #: core already passed ``task_id``/``session_id``/``user_task`` down that path;
    #: ``extras`` opens it to the host.
    #:
    #:     ctx = base.derive(extras={"repo": incident_repo, "actor": user.id})
    #:     with ctx.activate():
    #:         agent.run_conversation(text)
    #:
    #:     def report_incident(args, *, repo, actor, **_core):
    #:         return tool_result(id=repo.create(args["title"], reported_by=actor))
    #:
    #: Handlers should keep a ``**kwargs`` catch-all: the core adds its own keys, and a
    #: handler that names only its own would break on any core it did not expect.
    #:
    #: Per *context*, not per call -- the natural unit for a request-scoped dependency,
    #: because a context is what a host activates around one request. For anything that
    #: varies within a single turn, the handler reads it itself.
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.extras:
            # Skip the copy in the overwhelmingly common case: most contexts have none,
            # and this runs on every derive() as well as every construction.
            object.__setattr__(self, "extras", MappingProxyType({}))
            return
        collisions = RESERVED_TOOL_KWARGS.intersection(self.extras)
        if collisions:
            raise ValueError(
                f"AgentContext.extras may not use the reserved tool keyword(s) "
                f"{sorted(collisions)}; the core passes those to every handler itself. "
                f"Rename them (e.g. {sorted(collisions)[0]!r} -> "
                f"{'host_' + sorted(collisions)[0]!r})."
            )
        # A frozen dataclass whose one mutable field could still be edited in place is
        # only half frozen -- and this one is read by another thread on every tool call.
        object.__setattr__(self, "extras", MappingProxyType(dict(self.extras)))

    @contextmanager
    def activate(self) -> Iterator["AgentContext"]:
        """Make this the active context for the duration of the block.

        Scoped to the running thread or asyncio task. Re-entrant and nestable: the
        previous context is restored on exit, including when the block raises, so a
        failed turn cannot leave another tenant's settings bound.
        """
        token = _active.current.set(self)
        try:
            yield self
        finally:
            _active.current.reset(token)

    def derive(self, **changes: Any) -> "AgentContext":
        """A copy with some fields replaced -- two agents in one organisation.

        The escalation case: a sales agent and the owner's personal agent share a
        tenant's credentials and configuration but want separate conversation history.
        ``org.derive(session_store=personal_store, name="personal")`` expresses that
        without duplicating the parts that must stay identical.
        """
        return replace(self, **changes)


def get_active_context() -> Optional[AgentContext]:
    """The context bound to this thread or task, or ``None`` outside any block."""
    return _active.get_active()


@contextmanager
def activate(context: Optional[AgentContext]) -> Iterator[Optional[AgentContext]]:
    """Activate a context, or explicitly activate none.

    Passing ``None`` is meaningful: it detaches from the surrounding context so the
    block falls back to the process-wide defaults. Useful for work that belongs to the
    deployment rather than to any tenant.
    """
    if context is None:
        token = _active.current.set(None)
        try:
            yield None
        finally:
            _active.current.reset(token)
        return
    with context.activate() as active:
        yield active
