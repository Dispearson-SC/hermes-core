"""The managed multi-tenant runtime, which this core does not have.

Upstream can run under a hosted relay that multiplexes many profiles through one
process. A session coordinator there hands out conversation leases, tracks turn
boundaries across tenants, and reports task telemetry back to the vendor's service.

An embedded core has none of that: it runs inside the host's process, and the host
owns its own tenancy. So every coordinator call is inert here.

Inert, not removed. The turn loop calls the coordinator on both the way in and the
way out -- acquire and release, begin and end -- inside ``try``/``finally`` blocks
that keep the turn's structure honest. Deleting the calls would mean editing that
control flow in several lifted modules, which is exactly the kind of surgery most
likely to introduce a bug nobody notices until a turn fails halfway.

The lease this hands back is always granted, because with one tenant there is nothing
to arbitrate.
"""

from __future__ import annotations

from typing import Any, Optional

__all__ = ["SESSION_COORDINATOR", "current_profile_key", "apply_tool_request_intercepts"]


class _NullSessionCoordinator:
    """A coordinator for a process that hosts exactly one tenant."""

    def acquire_conversation(self, *_args: Any, **_kwargs: Any) -> None:
        """Grant the lease by returning nothing to release.

        Callers keep the result and pass it to ``release_conversation`` in a
        ``finally``; ``None`` travels through that path unchanged.
        """
        return None

    def release_conversation(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def begin_turn(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def end_turn(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def finish_logical_calls(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def finalize_conversation(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def notify_session_compacted(self, *_args: Any, **_kwargs: Any) -> None:
        """Upstream tells the relay that a session split during compaction.

        Only a coordinator tracking sessions across tenants needs to know; the
        compaction itself already happened.
        """
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<no relay: single-tenant>"


SESSION_COORDINATOR = _NullSessionCoordinator()


def current_profile_key() -> str:
    """The profile this turn belongs to.

    Upstream returns the multiplexed profile's key. There is one workspace here, so
    the answer is constant -- and empty rather than invented, so anything that stores
    it does not record a fictitious tenant.
    """
    return ""


def apply_tool_request_intercepts(*args: Any, **kwargs: Any) -> Optional[Any]:
    """Let the relay rewrite a tool request before it runs.

    Nothing intercepts here. Hosts that want to inspect or rewrite tool requests use
    the plugin middleware, which is part of the core.
    """
    return kwargs.get("args") if "args" in kwargs else (args[0] if args else None)


def get_runtime() -> None:
    """The managed runtime this process is attached to. There is none."""
    return None
