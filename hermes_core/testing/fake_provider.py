"""A scripted model provider for tests.

Upstream Hermes has no reusable fake provider. Across roughly 3,800 test files each
one hand-rolls a ``MagicMock`` over ``agent.client`` and re-invents the response
shape, so there is no single seam to swap in a deterministic backend. That is a large
part of why the agent loop is hard to test in isolation, and it is why this module
exists before any of the loop is moved.

The design goal is that a test states what the model says, in order, and nothing
else. Everything the loop does in response -- validating tool calls, executing them,
feeding results back, deciding to stop -- is then observable without a network, an
API key, or a real model's nondeterminism.

    script = Script().calls(("get_weather", {"city": "Rosario"})).text("It is 21C.")
    provider = FakeProvider(script)

    # ... drive the loop with `provider` ...

    assert provider.call_count == 2
    assert provider.last_request.tool_names == ["get_weather"]

Exhaustion is an error, never a default response. If the loop asks for one more turn
than the script provides, the loop did not stop when the test expected it to, and a
fake that answered anyway would hide the very bug it was meant to catch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from hermes_core.agent.transports.base import ProviderTransport
from hermes_core.agent.transports.types import NormalizedResponse, ToolCall, Usage

__all__ = [
    "Script",
    "FakeProvider",
    "RecordedRequest",
    "ScriptExhausted",
    "ToolCallSpec",
    "StreamDrop",
]

# provider_data keys the streaming-only fake client (fake_client.py) reads back off a
# scripted NormalizedResponse. Private to the testing package: nothing in hermes_core
# proper looks at these, so stashing them in the field meant for protocol-specific,
# consumer-defined state is safe and needs no change to NormalizedResponse itself.
_STREAM_DROP_KEY = "_test_stream_drop"
_IGNORE_STREAM_KEY = "_test_ignore_stream"


@dataclass
class StreamDrop:
    """Cuts a scripted stream short -- the shape of a connection that dies part-way
    through, which :class:`Script` otherwise has no way to express.

    ``after_chunks`` counts OpenAI-shaped SSE chunks: the same units the real streaming
    path receives one at a time -- a leading role chunk, one chunk per text fragment or
    tool-call delta, the trailing empty finish-reason chunk, and the usage chunk. The
    stream is cut right after that many have been delivered; ``after_chunks=0`` yields
    nothing at all.

    With ``error=None`` the stream just ends there: no finish_reason, no usage chunk,
    exactly like a connection that drops silently. With ``error`` set, the next pull
    past the cut raises it instead of stopping -- the shape of a read that fails
    outright (``httpx.RemoteProtocolError``, a provider's own mid-stream error frame).

    Usage::

        Script().calls(("get_weather", {"city": "Rosario"}),
                        drop=StreamDrop(after_chunks=2))

    That is the role chunk plus the one tool-call delta for a single call -- the model
    has emitted a complete tool call, and the connection dies before the finish_reason
    and usage chunks arrive.
    """

    after_chunks: int
    error: Optional[BaseException] = None


def _stream_provider_data(
    drop: Optional[StreamDrop], ignore_stream: bool, base: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if drop is None and not ignore_stream:
        return base
    data = dict(base) if base else {}
    if drop is not None:
        data[_STREAM_DROP_KEY] = drop
    if ignore_stream:
        data[_IGNORE_STREAM_KEY] = True
    return data


def stream_drop_of(response: Any) -> Optional[StreamDrop]:
    """The :class:`StreamDrop` scripted for this turn's response, if any."""
    data = getattr(response, "provider_data", None)
    return data.get(_STREAM_DROP_KEY) if isinstance(data, dict) else None


def response_ignores_stream(response: Any) -> bool:
    """True when this turn scripts a provider that ignores ``stream=True`` and answers
    in one piece -- the completed-response escape hatch ``UnmanagedLlmStream`` exists
    for."""
    data = getattr(response, "provider_data", None)
    return bool(isinstance(data, dict) and data.get(_IGNORE_STREAM_KEY))

# ("tool_name", {"arg": value}) or ("tool_name", {"arg": ...}, "explicit_call_id")
ToolCallSpec = Union[
    Tuple[str, Dict[str, Any]],
    Tuple[str, Dict[str, Any], str],
    ToolCall,
]


class ScriptExhausted(AssertionError):
    """The loop requested more model turns than the script defined.

    An ``AssertionError`` because in a test this is a failed expectation about the
    loop's behaviour, not a runtime fault in the code under test.
    """


@dataclass
class RecordedRequest:
    """What the loop sent on one model call, captured for assertions."""

    model: str
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]]
    params: Dict[str, Any]

    @property
    def tool_names(self) -> List[str]:
        """Names of the tools offered to the model, in the order they were sent."""
        names: List[str] = []
        for tool in self.tools or []:
            if not isinstance(tool, dict):
                continue
            body = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            name = body.get("name")
            if name:
                names.append(name)
        return names

    @property
    def system_prompt(self) -> Optional[str]:
        """The system message content, if the loop sent one."""
        for message in self.messages:
            if message.get("role") == "system":
                content = message.get("content")
                return content if isinstance(content, str) else json.dumps(content)
        return None

    def messages_with_role(self, role: str) -> List[Dict[str, Any]]:
        return [m for m in self.messages if m.get("role") == role]


class Script:
    """An ordered list of model turns, built fluently.

    Each method appends one turn and returns ``self``, so a whole conversation reads
    as a single expression. One turn is consumed per model call, in order.
    """

    def __init__(self) -> None:
        self._turns: List[Callable[[int], NormalizedResponse]] = []

    def text(
        self, content: str, *, finish_reason: str = "stop",
        drop: Optional[StreamDrop] = None, ignore_stream: bool = False,
    ) -> "Script":
        """The model answers with a final message and stops.

        ``drop``/``ignore_stream`` only affect ``install_fake_client(..., stream=True)``;
        a non-streaming call ignores them (there is no chunk sequence to cut, and
        nothing to "ignore stream" about). See :class:`StreamDrop`.
        """

        def turn(_: int) -> NormalizedResponse:
            return NormalizedResponse(
                content=content,
                tool_calls=None,
                finish_reason=finish_reason,
                usage=Usage(),
                provider_data=_stream_provider_data(drop, ignore_stream),
            )

        self._turns.append(turn)
        return self

    def calls(
        self, *specs: ToolCallSpec, content: Optional[str] = None,
        drop: Optional[StreamDrop] = None, ignore_stream: bool = False,
    ) -> "Script":
        """The model requests one or more tool calls.

        Several specs in one ``calls()`` produce a parallel batch in a single turn --
        the shape that exercises concurrent execution. Chaining separate ``calls()``
        produces sequential turns instead.

        Call ids default to ``call_<turn>_<index>``, stable across runs, so
        assertions on tool-result plumbing never have to guess an id.

        ``drop``/``ignore_stream`` only affect ``install_fake_client(..., stream=True)``;
        see :class:`StreamDrop`.
        """

        def turn(index: int) -> NormalizedResponse:
            return NormalizedResponse(
                content=content,
                tool_calls=[_to_tool_call(s, index, i) for i, s in enumerate(specs)],
                finish_reason="tool_calls",
                usage=Usage(),
                provider_data=_stream_provider_data(drop, ignore_stream),
            )

        self._turns.append(turn)
        return self

    def raw(self, response: NormalizedResponse) -> "Script":
        """Return a response the test constructed verbatim.

        The escape hatch for shapes the helpers do not cover -- a truncated
        ``finish_reason="length"``, provider-specific ``provider_data``, reasoning
        content.
        """
        self._turns.append(lambda _: response)
        return self

    def error(self, exc: BaseException) -> "Script":
        """Raise instead of answering, to drive retry and error-classification paths."""

        def turn(_: int) -> NormalizedResponse:
            raise exc

        self._turns.append(turn)
        return self

    def __len__(self) -> int:
        return len(self._turns)

    def _turn(self, index: int) -> Callable[[int], NormalizedResponse]:
        if index >= len(self._turns):
            raise ScriptExhausted(
                f"the loop requested model turn {index + 1} but the script defines "
                f"{len(self._turns)}. Either the loop failed to stop, or the script "
                f"is missing a turn."
            )
        return self._turns[index]


def _to_tool_call(spec: ToolCallSpec, turn: int, index: int) -> ToolCall:
    if isinstance(spec, ToolCall):
        return spec
    name, arguments = spec[0], spec[1]
    call_id = spec[2] if len(spec) > 2 else f"call_{turn}_{index}"
    return ToolCall(
        id=call_id,
        name=name,
        arguments=arguments if isinstance(arguments, str) else json.dumps(arguments),
    )


class FakeProvider(ProviderTransport):
    """A ``ProviderTransport`` that replays a :class:`Script` and records requests.

    It satisfies the transport contract, so it drops in wherever a real transport
    does. ``normalize_response`` is the identity: the fake already produces the
    normalized shape, because there is no wire format to translate.
    """

    def __init__(self, script: Optional[Script] = None) -> None:
        self.script = script if script is not None else Script()
        self.requests: List[RecordedRequest] = []

    # -- transport contract ---------------------------------------------------

    @property
    def api_mode(self) -> str:
        return "fake"

    def convert_messages(self, messages: List[Dict[str, Any]], **kwargs: Any) -> Any:
        """Pass messages through unchanged; the fake speaks the canonical format."""
        return messages

    def convert_tools(self, tools: List[Dict[str, Any]]) -> Any:
        """Pass tool definitions through unchanged."""
        return tools

    def build_kwargs(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **params: Any,
    ) -> Dict[str, Any]:
        return {"model": model, "messages": messages, "tools": tools, **params}

    def normalize_response(self, response: Any, **kwargs: Any) -> NormalizedResponse:
        if isinstance(response, NormalizedResponse):
            return response
        raise TypeError(
            f"FakeProvider received {type(response).__name__}; it only handles "
            f"responses it produced itself."
        )

    # -- the fake client ------------------------------------------------------

    def complete(
        self,
        *,
        model: str = "fake-model",
        messages: Optional[Sequence[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **params: Any,
    ) -> NormalizedResponse:
        """Consume the next scripted turn and record what was asked.

        This is the seam a host wires to its own call site. It deliberately mirrors
        the shape of a completion call rather than any one SDK's signature.
        """
        index = len(self.requests)
        self.requests.append(
            RecordedRequest(
                model=model,
                messages=[dict(m) for m in (messages or [])],
                tools=tools,
                params=dict(params),
            )
        )
        return self.script._turn(index)(index)

    # -- assertions -----------------------------------------------------------

    @property
    def call_count(self) -> int:
        """How many model calls the loop made."""
        return len(self.requests)

    @property
    def last_request(self) -> RecordedRequest:
        if not self.requests:
            raise AssertionError("no model call was made")
        return self.requests[-1]

    @property
    def exhausted(self) -> bool:
        """True when every scripted turn was consumed -- usually what a test wants."""
        return len(self.requests) >= len(self.script)
