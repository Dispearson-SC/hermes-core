"""Tool calls routed through the managed relay -- which this core does not use.

Upstream can run tool execution under a hosted relay that observes each call and may
rewrite its arguments before it runs. With no relay attached, upstream takes the
unmanaged path: run the callback with the arguments as given, and report those same
arguments back as final.

This core is always in that state, so that is what happens here. The function stays
rather than its call site being edited, because the two-value return -- result and
the arguments actually used -- is what the caller records in history, and collapsing
it would change what gets persisted.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

__all__ = ["execute"]


def execute(
    tool_name: str,
    args: Dict[str, Any],
    callback: Callable[[Dict[str, Any]], Any],
    **_kwargs: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """Run one tool call and report the arguments it ran with.

    Returns ``(result, final_args)``. Nothing rewrites the arguments here, so the
    final ones are the ones passed in -- but the caller still writes them to history
    from this return value, so they must come back.
    """
    return callback(args), args
