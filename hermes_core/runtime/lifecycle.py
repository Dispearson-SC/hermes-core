"""Hermes lifecycle dispatch for first-party observers and plugins."""

from __future__ import annotations

import importlib
import logging
from typing import Any, List

logger = logging.getLogger(__name__)


#: Set once the first import settles, so the absence is decided one time rather than
#: re-raised on every hook.
_OBSERVER_MISSING = False


def _first_party_observer(name: str):
    """The named function from ``hermes_core.runtime.observability``, or ``None``.

    Upstream's observability package forwards lifecycle events to Nous Portal's relay
    telemetry, which this core strips deliberately -- so there is no observer here, and
    a host that wants one drops in its own ``hermes_core/runtime/observability.py``
    exposing ``observe_lifecycle`` and ``handles_hook``.

    Absence therefore has to be *expected*, not exceptional. Upstream wraps both call
    sites in a bare ``except Exception`` that logs a warning with a full traceback, which
    here fired **21 times with a stack trace in a single turn that called no tools** --
    enough to teach anyone reading the logs to ignore this logger, which is precisely
    where a real observer failure would appear. An observer that exists and raises is
    still warned about, because that one is a genuine fault.
    """
    global _OBSERVER_MISSING
    if _OBSERVER_MISSING:
        return None
    try:
        module = importlib.import_module("hermes_core.runtime.observability")
    except ImportError:
        _OBSERVER_MISSING = True
        return None
    hook = getattr(module, name, None)
    if hook is None:
        _OBSERVER_MISSING = True
    return hook


def _observe(hook_name: str, **kwargs: Any) -> None:
    observe_lifecycle = _first_party_observer("observe_lifecycle")
    if observe_lifecycle is None:
        return
    try:
        observe_lifecycle(hook_name, **kwargs)
    except Exception:
        logger.warning("Built-in observability hook failed", exc_info=True)


def _plugin_hooks(hook_name: str, **kwargs: Any) -> List[Any]:
    from hermes_core.runtime import plugins

    return plugins.invoke_hook(hook_name, **kwargs)


def invoke_hook(hook_name: str, **kwargs: Any) -> List[Any]:
    """Notify first-party observers, then invoke compatibility plugin hooks."""
    _observe(hook_name, **kwargs)
    return _plugin_hooks(hook_name, **kwargs)


def has_hook(hook_name: str) -> bool:
    """Return whether a first-party observer or plugin consumes a hook."""
    handles_hook = _first_party_observer("handles_hook")
    if handles_hook is not None:
        try:
            if handles_hook(hook_name):
                return True
        except Exception:
            logger.warning("Unable to inspect built-in observability hooks", exc_info=True)

    from hermes_core.runtime import plugins

    return plugins.has_hook(hook_name)


def finalize_session(**kwargs: Any) -> List[Any]:
    """Notify observers and hard-close one core-owned Relay conversation."""
    _observe("on_session_finalize", **kwargs)

    session_id = str(kwargs.get("session_id") or "")
    if session_id:
        try:
            from hermes_core.seams import relay_runtime

            relay_runtime.SESSION_COORDINATOR.finalize_conversation(
                profile_key=relay_runtime.current_profile_key(),
                session_id=session_id,
            )
        except Exception:
            logger.warning("Core Relay session finalization failed", exc_info=True)

    return _plugin_hooks("on_session_finalize", **kwargs)
