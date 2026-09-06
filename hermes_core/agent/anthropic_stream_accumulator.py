"""Rebuild an Anthropic message from its stream of server-sent events. It lives in upstream's relay module but is not relay machinery: any consumer of an Anthropic stream has to reassemble the message from its deltas, and getting the block indexing wrong silently drops content. Extracted so the core keeps the real implementation, not a reinvented one.

Extracted verbatim from upstream ``agent/relay_llm.py``; edit there and re-run
the lift rather than editing this file.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any


_ANTHROPIC_APPEND_DELTAS = {"text_delta": "text", "thinking_delta": "thinking", "signature_delta": "signature"}


class AnthropicStreamAccumulator:
    """Rebuild an Anthropic Message from post-intercept SSE events."""

    def __init__(self) -> None:
        self._message: dict[str, Any] = {}
        self._blocks: dict[int, dict[str, Any]] = {}

    def observe(self, event: Any) -> None:
        payload = _jsonable(event)
        if isinstance(payload, dict):
            handler = self._EVENT_HANDLERS.get(payload.get("type"))
            if handler is not None:
                handler(self, payload)

    def _on_message_start(self, payload: dict[str, Any]) -> None:
        message = payload.get("message")
        if isinstance(message, dict):
            self._message.update({k: message[k] for k in ("id", "type", "role", "model", "usage") if k in message})

    def _on_content_block_start(self, payload: dict[str, Any]) -> None:
        index, block = payload.get("index"), payload.get("content_block")
        if isinstance(index, int) and isinstance(block, dict):
            self._blocks[index] = dict(block)

    def _on_content_block_delta(self, payload: dict[str, Any]) -> None:
        index, delta = payload.get("index"), payload.get("delta")
        if not isinstance(index, int) or not isinstance(delta, dict):
            return
        block = self._blocks.setdefault(index, {})
        delta_type = delta.get("type")
        field = _ANTHROPIC_APPEND_DELTAS.get(delta_type)
        if field is not None:
            block[field] = str(block.get(field) or "") + str(delta.get(field) or "")
        elif delta_type == "input_json_delta":
            block["_partial_json"] = str(block.pop("_partial_json", "")) + str(delta.get("partial_json") or "")
        elif delta_type == "citations_delta" and "citation" in delta:
            block.setdefault("citations", []).append(delta["citation"])

    def _on_message_delta(self, payload: dict[str, Any]) -> None:
        delta = payload.get("delta")
        if isinstance(delta, dict):
            self._message.update({k: delta[k] for k in ("stop_reason", "stop_sequence") if k in delta})
        if "usage" in payload:
            usage, current_usage = payload["usage"], self._message.get("usage")
            if isinstance(current_usage, dict) and isinstance(usage, dict):
                usage = {**current_usage, **usage}
            self._message["usage"] = usage

    _EVENT_HANDLERS = {
        "message_start": _on_message_start, "content_block_start": _on_content_block_start,
        "content_block_delta": _on_content_block_delta, "message_delta": _on_message_delta,
    }

    def finalize(self) -> dict[str, Any]:
        blocks = [dict(self._blocks[index]) for index in sorted(self._blocks)]
        for block in blocks:
            partial = block.pop("_partial_json", None)
            if partial is not None:
                with contextlib.suppress(TypeError, ValueError):
                    partial = json.loads(partial)
                block["input"] = partial
        return {**self._message, "content": blocks}

    def response(self, base: Any = None) -> Any:
        """Return the attribute-shaped response consumed by Hermes."""
        assembled = self.finalize()
        content = assembled.pop("content", [])
        merged = {**_jsonable_dict(base), **assembled}
        if content or "content" not in merged:
            merged["content"] = content
        return _namespace(merged)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    model_dump = getattr(type(value), "model_dump", None)
    if callable(model_dump):
        try:
            # warnings=False: pydantic's generic-union warning would leak to the terminal
            # mid-response; TypeError = duck-typed model_dump without pydantic's signature.
            try:
                return _jsonable(value.model_dump(mode="json", warnings=False))
            except TypeError:
                return _jsonable(value.model_dump())
        except Exception:
            pass
    try:
        attributes = {str(key): item for key, item in vars(value).items() if not str(key).startswith("_")}
    except (TypeError, AttributeError):
        return str(value)
    return _jsonable(attributes) if attributes else str(value)


def _jsonable_dict(value: Any) -> dict[str, Any]:
    """``_jsonable`` for values that must be a JSON object; anything else becomes ``{}``."""
    payload = _jsonable(value)
    return payload if isinstance(payload, dict) else {}


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{str(key): _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value
