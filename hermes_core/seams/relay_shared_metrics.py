"""Task metrics reported to the managed relay -- which this core does not talk to.

Upstream brackets every conversation with a pair of calls at the outer execution
boundary, so a hosted deployment can meter and trace a task across processes. That
telemetry is the relay's, sibling to the other ``relay_*`` seams in this package, and
it goes out over the network to a service this core has no connection to.

The pair is deliberately kept rather than patched out of ``turn_facade``. It is a
genuine extension point -- start of task, end of task, with the result or the exception
either way -- and a host that wants its own telemetry has somewhere obvious to put it::

    from hermes_core.seams import relay_shared_metrics
    relay_shared_metrics.start_task_run = my_start
    relay_shared_metrics.finish_task_run = my_finish

Keyword-only, matching upstream, so a host substituting these does not have to care
about argument order.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

__all__ = ["start_task_run", "finish_task_run"]


def start_task_run(
    *,
    session_id: str,
    task_id: str,
    platform: str,
    parent_session_id: str = "",
) -> None:
    """A conversation is beginning. Nothing to report and nowhere to report it."""
    return None


def finish_task_run(
    *,
    session_id: str,
    task_id: str,
    platform: str,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[BaseException] = None,
) -> None:
    """A conversation has ended, by return or by exception.

    Upstream calls this on both paths, and a replacement must stay quiet on both: this
    runs inside the caller's own teardown, so raising here would replace whatever
    actually ended the turn -- including the exception the caller needs to see.
    """
    return None
