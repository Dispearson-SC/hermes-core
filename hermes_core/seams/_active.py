"""The currently active :class:`~hermes_core.seams.context.AgentContext`, and nothing else.

This module is deliberately tiny and deliberately imports nothing from the rest of the
package. ``context.py`` has to import the four seam protocols to describe a context,
and those same seams have to ask which context is active -- which is a cycle. Holding
the variable somewhere neither end owns breaks it.

Not a global. ``ContextVar`` is scoped to the running thread or asyncio task and is
copied into new ones, so two organisations served concurrently in one process each see
their own, and neither can overwrite the other's. That is the whole point: with plain
module globals, whichever tenant configured itself last wins for everybody.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Optional

__all__ = ["current", "get_active", "get_tool_extras"]

#: The active context, or ``None`` when nothing has been activated and the seams should
#: fall back to their process-wide defaults.
current: ContextVar[Optional[Any]] = ContextVar("hermes_core_agent_context", default=None)


def get_active() -> Optional[Any]:
    """The active context, or ``None``.

    Every seam getter calls this on a hot path -- the turn loop reads configuration
    inside its own iteration -- so it stays a bare ``ContextVar.get()``.
    """
    return current.get()


def get_tool_extras() -> dict:
    """The active context's ``extras``, as a plain dict, or ``{}``.

    Lives here rather than on ``context.py`` for one reason: the code that calls it is
    ``model_tools._execute_tool``, which is *lifted* -- regenerated from upstream on
    every re-lift and reached through a ``PATCHES`` entry. Pointing that patch at this
    module keeps it to a single deferred import of something that imports nothing back,
    so it cannot participate in a cycle no matter what upstream does to its own imports.

    Returns a copy, never the context's own mapping: the caller merges core-owned keys
    into the result, and a host's context must not pick those up.
    """
    context = current.get()
    if context is None:
        return {}
    extras = getattr(context, "extras", None)
    return dict(extras) if extras else {}
