"""The agent intelligence of Hermes Agent, as an embeddable library.

Extracted from NousResearch/hermes-agent: the reasoning loop, provider transports,
the tool registry and its tool-call validation, MCP, skills, system-prompt assembly,
and context management -- without the CLI, the TUI, the gateway, the desktop app, or
any one vendor's commercial logic.

    from hermes_core import AgentContext, DirectoryWorkspace, DictConfigSource
    from hermes_core import StaticCredentials, SqliteSessionStore, registry, tool_result

    acme = AgentContext(
        workspace=DirectoryWorkspace("/var/lib/app/tenants/acme"),
        config=DictConfigSource({"model": {"default": "MiniMax-M2.7"}}),
        credentials=StaticCredentials(api_key),
        session_store=SqliteSessionStore("/var/lib/app/tenants/acme/sessions.db"),
    )
    with acme.activate():
        AIAgent(...).run_conversation("hola")

**What this list is for.** Most of this package is lifted from upstream and regenerated
by ``tools/lift.py`` whenever upstream moves; upstream itself says its internal import
paths are not a stable API. The names below are the ones this core commits to -- four
ports, their shipped adapters, the context that binds them, the tool registry and the
test doubles. Reaching past them into a lifted module works, and may stop working after
any re-lift, with no warning.

**Still lazy.** Resolved on first access through a module ``__getattr__`` rather than
imported here, so a host that only wants the tool registry does not pay for the
transports, and a cycle inside one lifted module cannot turn ``import hermes_core``
into an error.

See EXTRACTION.md for what came across, what did not, and why.
"""

from typing import TYPE_CHECKING

__version__ = "0.0.2"

#: Public name -> the module it lives in. This mapping *is* the contract; the lazy
#: resolution below is only how it is delivered.
_EXPORTS = {
    # -- the four ports a host implements ---------------------------------------
    "Workspace": "hermes_core.seams.paths",
    "ConfigSource": "hermes_core.seams.config",
    "CredentialSource": "hermes_core.seams.credentials",
    "SessionStore": "hermes_core.seams.session_store",
    "SessionReader": "hermes_core.seams.session_store",
    # -- shipped adapters, enough to run without writing any -------------------
    "DirectoryWorkspace": "hermes_core.seams.paths",
    "DictConfigSource": "hermes_core.seams.config",
    "StaticCredentials": "hermes_core.seams.credentials",
    "EnvCredentials": "hermes_core.seams.credentials",
    "RotatingKeyPool": "hermes_core.seams.credentials",
    "InMemorySessionStore": "hermes_core.seams.session_store",
    "SessionStoreBase": "hermes_core.seams.session_store",
    "AgentMailbox": "hermes_core.seams.mailbox",
    "MailboxMessage": "hermes_core.seams.mailbox",
    "MailboxState": "hermes_core.seams.mailbox",
    "InMemoryMailbox": "hermes_core.seams.mailbox",
    "SqliteMailbox": "hermes_core.seams.mailbox",
    "SqliteSessionStore": "hermes_core.seams.session_store",
    "Credentials": "hermes_core.seams.credentials",
    "NoCredentials": "hermes_core.seams.credentials",
    # -- binding them, per tenant or process-wide -------------------------------
    "AgentContext": "hermes_core.seams.context",
    "get_active_context": "hermes_core.seams.context",
    "activate": "hermes_core.seams.context",
    "set_workspace": "hermes_core.seams.paths",
    "set_config_source": "hermes_core.seams.config",
    "set_credential_source": "hermes_core.seams.credentials",
    # -- running a turn ----------------------------------------------------------
    # The entry point, and for a long time the one public name that was not on this
    # list -- reachable only as `hermes_core.run_agent.AIAgent`, which is exactly the
    # "internal import path" this list exists to replace. Its ~80 keyword parameters
    # are inherited from upstream and are not all part of the contract; the ones the
    # README documents are.
    "AIAgent": "hermes_core.run_agent",
    # -- ... from an async host --------------------------------------------------
    "run_conversation_async": "hermes_core.seams.async_bridge",
    "call_host_async": "hermes_core.seams.async_bridge",
    "bind_host_loop": "hermes_core.seams.async_bridge",
    "get_host_loop": "hermes_core.seams.async_bridge",
    "HostLoopUnavailable": "hermes_core.seams.async_bridge",
    # -- giving the agent something to do ---------------------------------------
    "registry": "hermes_core.tools.registry",
    "ToolRegistry": "hermes_core.tools.registry",
    "tool_result": "hermes_core.tools.registry",
    "tool_error": "hermes_core.tools.registry",
    # -- testing a host's own integration ---------------------------------------
    "FakeProvider": "hermes_core.testing",
    "Script": "hermes_core.testing",
    "StreamDrop": "hermes_core.testing",
    "install_fake_client": "hermes_core.testing",
}

__all__ = ["__version__", *sorted(_EXPORTS)]

if TYPE_CHECKING:  # pragma: no cover - for type checkers and editors only
    from hermes_core.run_agent import AIAgent
    from hermes_core.seams.async_bridge import (
        HostLoopUnavailable, bind_host_loop, call_host_async, get_host_loop,
        run_conversation_async,
    )
    from hermes_core.seams.config import ConfigSource, DictConfigSource, set_config_source
    from hermes_core.seams.context import AgentContext, activate, get_active_context
    from hermes_core.seams.credentials import (
        CredentialSource, Credentials, EnvCredentials, NoCredentials, RotatingKeyPool,
        StaticCredentials, set_credential_source,
    )
    from hermes_core.seams.paths import DirectoryWorkspace, Workspace, set_workspace
    from hermes_core.seams.mailbox import (
        AgentMailbox, InMemoryMailbox, MailboxMessage, MailboxState, SqliteMailbox,
    )
    from hermes_core.seams.session_store import (
        InMemorySessionStore, SessionReader, SessionStore, SessionStoreBase,
        SqliteSessionStore,
    )
    from hermes_core.testing import FakeProvider, Script, StreamDrop, install_fake_client
    from hermes_core.tools.registry import ToolRegistry, registry, tool_error, tool_result


def __getattr__(name: str):
    """Resolve a public name on first access (PEP 562)."""
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_path), name)
    # Cache it so the second access is a plain attribute lookup, not another import.
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
