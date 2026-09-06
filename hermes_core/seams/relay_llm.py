"""Provider calls routed through the managed relay -- which this core does not use.

Upstream can run under a hosted relay that wraps every provider call so the service
can meter, trace and arbitrate it. With no relay attached, upstream's own code takes
an *unmanaged* path, and this module is that path.

For non-streaming calls the unmanaged path is simply ``callback(request)``. Streaming
is not so simple, and getting it wrong is how this seam first broke against a real
provider: upstream returns its own iterator wrapper even when no relay is attached,
and the turn loop reads two things off it.

* ``final_response`` -- ``None`` for a genuine stream, or the whole response when the
  factory ignored ``stream=True`` and answered in one piece. Some providers do that,
  and the loop unwraps it rather than trying to iterate a complete response.
* ``on_stream_created(raw)`` -- how the loop registers the live stream with its
  abort/close machinery, so a stalled response can have its socket torn down.

Returning the SDK's raw stream instead loses both. The loop reads
``stream.final_response``, gets an ``AttributeError`` mid-response, treats it as a
stream that died mid tool-call, and drops the tool call the model had already sent
complete. Against MiniMax that turned a working turn into "the action was not
executed" -- with no error anywhere pointing at the cause.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, Optional, Tuple

__all__ = [
    "execute",
    "execute_async",
    "execute_current",
    "stream",
    "stream_current",
    "complete_logical_call",
    "active_turn",
    "UnmanagedLlmStream",
]

# Not relay machinery: reassembling an Anthropic message from its event stream is
# something any consumer of that stream must do. Re-exported because the lifted code
# reaches for it by this path.
from hermes_core.agent.anthropic_stream_accumulator import (  # noqa: E402
    AnthropicStreamAccumulator,
)

__all__.append("AnthropicStreamAccumulator")


def execute(
    request: Dict[str, Any],
    callback: Callable[[Dict[str, Any]], Any],
    **_kwargs: Any,
) -> Any:
    """Run one provider attempt directly."""
    return callback(request)


async def execute_async(
    request: Dict[str, Any],
    callback: Callable[[Dict[str, Any]], Any],
    **_kwargs: Any,
) -> Any:
    """Async form. The callback may be a coroutine function or an ordinary one."""
    result = callback(request)
    if hasattr(result, "__await__"):
        return await result
    return result


def execute_current(
    request: Dict[str, Any],
    callback: Callable[[Dict[str, Any]], Any],
    **_kwargs: Any,
) -> Any:
    """Run the attempt against the current turn, of which there is only ever one."""
    return callback(request)


class UnmanagedLlmStream:
    """Upstream's stream wrapper, on the path taken when no relay is attached.

    Iterating yields the provider's chunks untouched. The wrapper exists for what
    surrounds them: the completed-response escape hatch, the hook that hands the live
    stream to the abort machinery, and the finalizer that runs exactly once when the
    stream ends -- however it ends.
    """

    #: ``None`` while this is a real stream. Set when the factory returned a complete
    #: response despite being asked to stream, which some providers do.
    final_response: Any = None

    #: Whether anything rewrote the chunks on their way through. Always false here:
    #: rewriting is what a relay does, and there is none. The Anthropic path reads
    #: this to decide whether it can trust the SDK's own assembled message or has to
    #: rebuild it from the events it saw -- so answering False keeps it on the
    #: cheaper, more faithful path.
    output_modified: bool = False

    def __init__(
        self,
        request: Dict[str, Any],
        stream_factory: Callable[[Dict[str, Any]], Any],
        *,
        on_stream_created: Optional[Callable[[Any], Any]] = None,
        on_chunk: Optional[Callable[[Any], Any]] = None,
        accept_chunk: Optional[Callable[[Any], bool]] = None,
        finalizer: Optional[Callable[[], Any]] = None,
        completed_response_predicate: Optional[Callable[[Any], bool]] = None,
        **_ignored: Any,
    ) -> None:
        self._finalizer = finalizer
        self._on_chunk = on_chunk
        self._accept_chunk = accept_chunk
        self._closed = False
        self._raw: Any = None

        raw = stream_factory(request)
        if completed_response_predicate is not None and completed_response_predicate(raw):
            # The provider answered in full despite `stream=True`. Hand it over as the
            # final response and present an empty iterator, so the caller's loop over
            # chunks finds nothing rather than trying to iterate a whole response.
            self.final_response = raw
            self._iterator: Iterator[Any] = iter(())
            return

        self._raw = raw
        if on_stream_created is not None:
            # This is how the live stream reaches the abort/close machinery. Skipping
            # it leaves a stalled response with no way to tear its socket down.
            on_stream_created(raw)
        self._iterator = iter(raw)

    # -- passthroughs the caller reads off the stream -------------------------

    @property
    def response(self) -> Any:
        """The underlying HTTP response, which error handling reads for status/body."""
        return getattr(self._raw, "response", None)

    # -- iteration ------------------------------------------------------------

    def __iter__(self) -> "UnmanagedLlmStream":
        return self

    def __next__(self) -> Any:
        if self._closed:
            raise StopIteration

        sentinel = object()
        chunk = next(self._iterator, sentinel)
        if chunk is sentinel:
            self.close()
            raise StopIteration

        if self._accept_chunk is not None and not self._accept_chunk(chunk):
            # The caller has seen enough -- a stop sequence, an interrupt. Close so the
            # connection is released rather than left half-read in the pool.
            self.close()
            raise StopIteration

        if self._on_chunk is not None:
            self._on_chunk(chunk)
        return chunk

    # -- teardown -------------------------------------------------------------

    def close(self) -> None:
        """Release the provider stream and run the finalizer, once.

        Idempotent: the caller closes on interrupt and iteration closes on
        exhaustion, and a finalizer that ran twice would double-count the turn.
        """
        if self._closed:
            return
        self._closed = True
        raw_close = getattr(self._raw, "close", None)
        if callable(raw_close):
            try:
                raw_close()
            except Exception:
                # A stream that will not close cleanly must not mask the response the
                # caller already has, nor skip the finalizer below.
                pass
        if self._finalizer is not None:
            self._finalizer()

    def __enter__(self) -> "UnmanagedLlmStream":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def stream(
    request: Dict[str, Any],
    stream_factory: Callable[[Dict[str, Any]], Any],
    **kwargs: Any,
) -> UnmanagedLlmStream:
    """Open a provider stream, wrapped the way the turn loop expects."""
    return UnmanagedLlmStream(request, stream_factory, **kwargs)


def stream_current(
    request: Dict[str, Any],
    stream_factory: Callable[[Dict[str, Any]], Any],
    **kwargs: Any,
) -> UnmanagedLlmStream:
    """Open a stream against the current turn, of which there is only ever one."""
    return UnmanagedLlmStream(request, stream_factory, **kwargs)


def complete_logical_call(api_request_id: str, **_kwargs: Any) -> None:
    """Close out a logical call in the relay's ledger. There is no ledger."""
    return None


def active_turn() -> None:
    """The relay turn this call belongs to. There is never one."""
    return None
