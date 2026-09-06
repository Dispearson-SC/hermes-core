"""A fake OpenAI-shaped client, driven by a :class:`~hermes_core.testing.Script`.

:class:`~hermes_core.testing.FakeProvider` sits at the transport boundary and speaks
the core's normalised shape. The agent, though, holds an SDK client and calls
``client.chat.completions.create(...)``, so an end-to-end test needs a double at
*that* boundary instead -- one that answers in the wire shape and lets the real
transport normalise it.

Using the same ``Script`` for both means a test states the model's behaviour once and
can drive it at whichever level it is testing:

    agent.client = FakeOpenAIClient(
        Script().calls(("get_weather", {"city": "Rosario"})).text("It is 21C.")
    )

Everything downstream -- validation, dispatch, feeding results back, deciding to stop
-- is then the real code, with only the model replaced.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from hermes_core.testing.fake_provider import (
    RecordedRequest,
    Script,
    response_ignores_stream,
    stream_drop_of,
)

__all__ = ["FakeOpenAIClient", "install_fake_client", "ScriptedStream"]


def install_fake_client(
    agent: Any, script: Optional[Script] = None, *, stream: bool = False
) -> "FakeOpenAIClient":
    """Point an agent at a scripted client and return it.

    Setting ``agent.client`` alone is not enough. The turn loop builds a *per-request*
    client for most providers -- so the watchdog can abort that request's sockets
    without touching another in-flight one -- and only a couple of paths reuse the
    long-lived one. A test that patches just the attribute quietly reaches the network
    instead, which is exactly the failure this helper exists to prevent.

        client = install_fake_client(agent, Script().text("hello"))
        agent.run_conversation("hi")
        assert client.call_count == 1

    Streaming is off by default. The loop prefers to stream even with nobody
    consuming the deltas, because a live stream doubles as a health check for a
    stalled connection -- and upstream already turns that off for test doubles for
    the same reason it matters here: those health checks are about a real socket, and
    a scripted answer has none. Pass ``stream=True`` to exercise the streaming path
    itself; the fake then answers in deltas, ending with the trailing usage chunk a
    real provider sends.
    """
    fake = FakeOpenAIClient(script)
    agent.client = fake
    agent._create_request_openai_client = lambda *a, **k: fake
    agent._create_request_anthropic_client = lambda *a, **k: fake
    agent._disable_streaming = not stream
    return fake


def _as_wire_message(response: Any) -> SimpleNamespace:
    """Render a normalised response the way the OpenAI SDK would."""
    tool_calls = [
        SimpleNamespace(
            id=call.id,
            type="function",
            function=SimpleNamespace(name=call.name, arguments=call.arguments),
        )
        for call in (response.tool_calls or [])
    ]
    return SimpleNamespace(
        role="assistant",
        content=response.content,
        tool_calls=tool_calls or None,
        reasoning_content=None,
        reasoning=None,
    )


class ScriptedStream:
    """Iterator returned in place of the provider SDK's stream for a scripted turn.

    Ordinarily this just replays the chunks ``_build_stream_pieces`` built -- plain
    iteration to ``StopIteration``, indistinguishable from a live SSE stream to the code
    under test. A :class:`~hermes_core.testing.fake_provider.StreamDrop` truncates that
    chunk list before this object is built, so what makes this worth a class rather than
    a bare generator is what happens at the cut, and what a test can observe afterwards:

    * the next pull past the cut either raises the drop's ``error`` (once -- a second
      pull raises the plain ``StopIteration`` a real closed stream would) or ends
      iteration outright, with no finish_reason and no usage chunk either way;
    * every ``close()`` -- ordinary exhaustion, an interrupted turn's explicit teardown,
      a retry's cleanup pass -- is counted, so a test can assert the connection was
      actually torn down rather than left half-read.
    """

    def __init__(self, chunks: List[Any], *, error: Optional[BaseException] = None) -> None:
        self._chunks = list(chunks)
        self._pos = 0
        self._error = error
        self._error_raised = False
        self.close_calls = 0

    def __iter__(self) -> "ScriptedStream":
        return self

    def __next__(self) -> Any:
        if self._pos < len(self._chunks):
            piece = self._chunks[self._pos]
            self._pos += 1
            return piece
        if self._error is not None and not self._error_raised:
            self._error_raised = True
            raise self._error
        raise StopIteration

    def close(self) -> None:
        self.close_calls += 1

    @property
    def exhausted(self) -> bool:
        """True once every scripted chunk has been pulled, whether or not an error
        followed."""
        return self._pos >= len(self._chunks)


def _build_stream_pieces(response: Any, model: str, index: int) -> List[Any]:
    """The full, undropped sequence of OpenAI-shaped chunks for one scripted response.

    Text is split into a few deltas rather than one, because a fake that always
    delivers a whole message in a single chunk would hide any bug in how deltas are
    joined.
    """

    def chunk(delta: SimpleNamespace, finish_reason: Any = None) -> SimpleNamespace:
        return SimpleNamespace(
            id=f"chatcmpl-fake-{index}",
            model=model,
            choices=[SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)],
            usage=None,
        )

    def usage_chunk() -> SimpleNamespace:
        """The trailing choiceless chunk that carries token counts.

        Real providers send this last. Omitting it is not a harmless simplification:
        the loop treats a stream that ended with no usage and no finish reason as one
        cut off mid-flight, and enters its continuation path -- so a fake without it
        makes every response look truncated. (That is also exactly the shape a
        ``StreamDrop`` deliberately produces on purpose.)
        """
        usage = response.usage
        return SimpleNamespace(
            id=f"chatcmpl-fake-{index}",
            model=model,
            choices=[],
            usage=SimpleNamespace(
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                total_tokens=getattr(usage, "total_tokens", 0),
            ),
        )

    empty = SimpleNamespace(role=None, content=None, tool_calls=None, reasoning_content=None)
    pieces: List[Any] = []

    if response.tool_calls:
        pieces.append(chunk(SimpleNamespace(role="assistant", content=None, tool_calls=None,
                                             reasoning_content=None)))
        for position, call in enumerate(response.tool_calls):
            pieces.append(chunk(SimpleNamespace(
                role=None, content=None, reasoning_content=None,
                tool_calls=[SimpleNamespace(
                    index=position, id=call.id, type="function",
                    function=SimpleNamespace(name=call.name, arguments=call.arguments),
                )],
            )))
        pieces.append(chunk(empty, finish_reason=response.finish_reason or "tool_calls"))
        pieces.append(usage_chunk())
        return pieces

    content = response.content or ""
    pieces.append(chunk(SimpleNamespace(role="assistant", content="", tool_calls=None,
                                         reasoning_content=None)))
    # Three pieces: enough to prove the caller joins deltas, few enough to stay readable.
    step = max(1, -(-len(content) // 3)) if content else 1
    for start in range(0, len(content), step):
        pieces.append(chunk(SimpleNamespace(role=None, content=content[start:start + step],
                                             tool_calls=None, reasoning_content=None)))
    pieces.append(chunk(empty, finish_reason=response.finish_reason or "stop"))
    pieces.append(usage_chunk())
    return pieces


def _stream_chunks(response: Any, model: str, index: int) -> ScriptedStream:
    """Build the scripted stream for one response, applying its ``StreamDrop`` if any.

    The turn loop streams by default -- it needs first-token latency and the ability
    to abort mid-response -- so a double that only answers in one piece never
    exercises the path that actually runs.
    """
    pieces = _build_stream_pieces(response, model, index)
    drop = stream_drop_of(response)
    if drop is None:
        return ScriptedStream(pieces)
    return ScriptedStream(pieces[: max(0, drop.after_chunks)], error=drop.error)


class _Completions:
    def __init__(self, client: "FakeOpenAIClient") -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create(**kwargs)


class _Chat:
    def __init__(self, client: "FakeOpenAIClient") -> None:
        self.completions = _Completions(client)


class FakeOpenAIClient:
    """Answers ``chat.completions.create`` from a script, and records the requests."""

    def __init__(self, script: Optional[Script] = None, *, api_key: str = "fake-key") -> None:
        self.script = script if script is not None else Script()
        self.requests: List[RecordedRequest] = []
        self.api_key = api_key
        self.chat = _Chat(self)
        #: Every :class:`ScriptedStream` this client has handed out, in order -- so a
        #: test can check ``close_calls`` on the stream a dropped turn actually opened.
        #: Empty for turns answered non-streaming or via the ``ignore_stream`` escape
        #: hatch, since neither one hands back a live stream to close.
        self.streams: List[ScriptedStream] = []

    def _complete_response(self, response: Any, kwargs: Dict[str, Any], index: int) -> SimpleNamespace:
        """The non-streaming-shaped response: used for an ordinary non-streaming call,
        and for a streaming call scripted with ``ignore_stream=True`` -- a provider that
        answers in one piece despite ``stream=True``, which is what exercises
        ``UnmanagedLlmStream``'s completed-response escape hatch."""
        usage = response.usage
        return SimpleNamespace(
            id=f"chatcmpl-fake-{index}",
            model=kwargs.get("model", "fake-model"),
            choices=[
                SimpleNamespace(
                    index=0,
                    message=_as_wire_message(response),
                    finish_reason=response.finish_reason,
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                total_tokens=getattr(usage, "total_tokens", 0),
            ),
        )

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        index = len(self.requests)
        self.requests.append(
            RecordedRequest(
                model=kwargs.get("model", ""),
                messages=[dict(m) for m in (kwargs.get("messages") or [])],
                tools=kwargs.get("tools"),
                params={k: v for k, v in kwargs.items() if k not in ("model", "messages", "tools")},
            )
        )

        response = self.script._turn(index)(index)
        if kwargs.get("stream") and not response_ignores_stream(response):
            stream = _stream_chunks(response, kwargs.get("model", "fake-model"), index)
            self.streams.append(stream)
            return stream

        return self._complete_response(response, kwargs, index)

    # -- assertions -----------------------------------------------------------

    @property
    def call_count(self) -> int:
        return len(self.requests)

    @property
    def last_request(self) -> RecordedRequest:
        if not self.requests:
            raise AssertionError("no model call was made")
        return self.requests[-1]

    @property
    def last_stream(self) -> ScriptedStream:
        """The most recent live stream this client handed out.

        Only populated for a ``stream=True`` turn that was NOT scripted with
        ``ignore_stream=True`` -- that escape hatch never opens a stream for the
        caller to close in the first place. Use ``last_stream.close_calls`` to check
        teardown happened (and ran only as many times as expected), and
        ``last_stream.exhausted`` to see whether every scripted chunk was pulled
        before the turn moved on (an interrupt or a superseded attempt can stop
        short of that).
        """
        if not self.streams:
            raise AssertionError(
                "no stream was created -- was stream=True passed to install_fake_client(), "
                "and was the turn scripted without ignore_stream=True?"
            )
        return self.streams[-1]

    def tool_results_sent(self) -> List[Dict[str, Any]]:
        """Tool-result messages the loop fed back on the most recent call.

        The clearest evidence that a tool actually ran and its output reached the
        model, rather than the loop having stopped somewhere in between.
        """
        return [m for m in self.last_request.messages if m.get("role") == "tool"]

    def tool_result_payloads(self) -> List[Any]:
        """Those results with their JSON content parsed, where it parses."""
        payloads = []
        for message in self.tool_results_sent():
            try:
                payloads.append(json.loads(message.get("content", "")))
            except (TypeError, ValueError):
                payloads.append(message.get("content"))
        return payloads
