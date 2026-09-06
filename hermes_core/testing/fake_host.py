"""A minimal stand-in for the object the turn loop calls back into.

Upstream's ``AIAgent`` is a god object: a constructor with roughly ninety keyword
arguments, composed from thirteen mixins, mixing generic LLM orchestration with
Telegram identity, gateway leases and vendor billing state. That size is what makes
the loop look impossible to extract.

Reading what the loop actually *uses* tells a different story. Tool-call validation,
for instance, touches exactly twelve members:

    valid_tool_names            the set of registered tool names
    log_prefix                  a string prefixed to operator-facing lines
    _invalid_tool_retries       strike counter for unknown tool names
    _invalid_json_retries       strike counter for malformed arguments
    _uniquify_tool_call_ids()   de-duplicate ids before anything downstream sees them
    _repair_tool_call()         map a hallucinated name onto a real one, or None
    _build_assistant_message()  the assistant turn, in wire shape
    _persist_session()          durability
    _cleanup_task_resources()   release work tied to an abandoned turn
    _buffer_vprint()            queue an operator-facing line
    _flush_status_buffer()      emit the queued lines
    _vprint()                   emit one line immediately

Twelve members, not ninety. This class implements them and records what happened, so
loop behaviour can be asserted without constructing anything resembling the real
agent. It doubles as the working specification of that surface: as more of the loop
is lifted, whatever it needs gets added here, and this file stays the honest
inventory of the callbacks a host must provide.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

__all__ = ["FakeTurnHost"]


class FakeTurnHost:
    """A recording stand-in for the agent object the turn loop calls back into."""

    def __init__(
        self,
        *,
        valid_tool_names: Optional[Iterable[str]] = None,
        repairs: Optional[Dict[str, str]] = None,
        log_prefix: str = "",
    ) -> None:
        self.valid_tool_names = set(valid_tool_names or ())
        #: Hallucinated name -> real name. Upstream does fuzzy matching; a test
        #: states the mapping outright so the assertion is about the loop's use of
        #: the repair, not about the matcher's heuristics.
        self.repairs = dict(repairs or {})
        self.log_prefix = log_prefix

        self._invalid_tool_retries = 0
        self._invalid_json_retries = 0

        #: Everything the loop said, in order, whether buffered or immediate.
        self.messages_printed: List[str] = []
        #: Calls to _persist_session, as (messages, history) snapshots.
        self.persisted: List[tuple] = []
        #: Task ids passed to _cleanup_task_resources.
        self.cleaned_up: List[Any] = []
        #: Names passed to _repair_tool_call, including ones with no mapping.
        self.repair_attempts: List[str] = []
        self.flushed = 0

        self._buffer: List[str] = []

    # -- identity and state ---------------------------------------------------

    def _uniquify_tool_call_ids(self, tool_calls: List[Any]) -> None:
        """Give every call a distinct id.

        The pre-API sanitizer keeps only the first call and result per id, so two
        calls sharing one id would silently lose the second -- along with its result.
        """
        seen: Dict[str, int] = {}
        for call in tool_calls:
            current = call.id
            if current is None:
                continue
            if current in seen:
                seen[current] += 1
                call.id = f"{current}_{seen[current]}"
            else:
                seen[current] = 0

    def _repair_tool_call(self, name: str) -> Optional[str]:
        self.repair_attempts.append(name)
        return self.repairs.get(name)

    def _build_assistant_message(self, assistant_message: Any, finish_reason: str) -> Dict[str, Any]:
        """The assistant turn in wire shape, tool calls included."""
        return {
            "role": "assistant",
            "content": getattr(assistant_message, "content", None),
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in (getattr(assistant_message, "tool_calls", None) or [])
            ],
            "finish_reason": finish_reason,
        }

    # -- side effects ---------------------------------------------------------

    def _persist_session(self, messages: List[Dict[str, Any]], conversation_history: Any) -> None:
        self.persisted.append((list(messages), conversation_history))

    def _cleanup_task_resources(self, task_id: Any) -> None:
        self.cleaned_up.append(task_id)

    # -- operator output ------------------------------------------------------

    def _buffer_vprint(self, message: str, **_kwargs: Any) -> None:
        self._buffer.append(message)
        self.messages_printed.append(message)

    def _flush_status_buffer(self) -> None:
        self.flushed += 1
        self._buffer.clear()

    def _vprint(self, message: str, *, force: bool = False, **_kwargs: Any) -> None:
        self.messages_printed.append(message)

    # -- assertions -----------------------------------------------------------

    def said(self, fragment: str) -> bool:
        """Whether any operator-facing line contained *fragment*."""
        return any(fragment in message for message in self.messages_printed)


class FakeAssistantMessage:
    """The model's turn, in the shape the loop expects to receive it."""

    def __init__(self, tool_calls: Optional[List[Any]] = None, content: Optional[str] = None) -> None:
        self.tool_calls = tool_calls or []
        self.content = content

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        names = [getattr(c, "name", "?") for c in self.tool_calls]
        return f"FakeAssistantMessage(content={self.content!r}, tool_calls={names})"


def arguments_of(message: Dict[str, Any]) -> Any:
    """Parse a tool-result message's content as JSON, or return it unchanged."""
    try:
        return json.loads(message.get("content", ""))
    except (TypeError, ValueError):
        return message.get("content")
