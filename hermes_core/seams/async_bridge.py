"""Driving the synchronous turn loop from an asynchronous application.

``run_conversation`` blocks. Not by oversight: it is lifted from upstream, where the
whole agent runs on threads, and there is no ``async def`` anywhere on the turn path.
Re-lifting keeps it that way, so the core will not become natively async by editing --
it would become async by rewriting thousands of lines that upstream regenerates.

That leaves every async host -- FastAPI, Starlette, aiohttp, Litestar, an async worker --
with the same two problems, and both have exactly one correct answer that is easy to get
wrong. Rather than let each host rediscover them, they are solved here:

**Calling in.** ``await run_conversation_async(agent, "hola")`` runs the turn on a worker
thread. Awaiting it yields control, so the event loop keeps serving other requests for
the whole turn. Calling ``agent.run_conversation`` directly from a coroutine instead
freezes the entire server -- not just that request -- for as long as the model takes.

**Calling back out.** A tool handler must be a plain synchronous function, because that
is what the registry dispatches. A host whose repositories and services are ``async def``
therefore cannot await them from inside a handler, and the obvious escapes are both
wrong:

* ``asyncio.run(coro)`` and ``registry.register(..., is_async=True)`` each run the
  coroutine on a **brand-new event loop in a fresh thread**. Anything bound to the host's
  loop -- an asyncpg pool, an httpx client, most async database sessions -- is being used
  from the wrong loop. The failure is not reliably immediate, which is what makes it
  expensive: it surfaces later as a corrupted pool or an inexplicable hang.
* ``asyncio.run_coroutine_threadsafe(coro, loop)`` is right, but only with the host's real
  loop, and it deadlocks instantly if it is ever reached from the loop thread itself.

:func:`call_host_async` is that call done correctly, with the loop found for you and the
deadlock case raised as an error instead of a hang::

    async def create_incident(title: str) -> str: ...        # the host's own service

    def report_incident(args, *, repo, **_core):             # a plain sync tool handler
        incident_id = call_host_async(repo.create(args["title"]))
        return tool_result(id=incident_id)

The loop reference travels by ``ContextVar``, which is why this works at all: the core
copies the context into every thread it dispatches a tool on (``asyncio.to_thread`` here,
then ``DaemonThreadPoolExecutor.submit``), so a handler several threads deep still finds
the loop its request started on.

**A host that drives the thread itself** -- Starlette's ``run_in_threadpool``, a
``ProcessPoolExecutor``, an existing worker pool -- does not want :func:`run_conversation_async`
and should use :func:`bind_host_loop` instead, which binds the loop without dictating how
the turn gets off it::

    async def handle(request):
        with bind_host_loop():
            return await run_in_threadpool(agent.run_conversation, request.text)

**Cancellation.** A cancelled ``await`` does not stop the turn: the worker thread runs to
completion, because Python cannot interrupt an arbitrary thread. Call ``agent.interrupt()``
to actually stop one -- the turn loop checks for it between steps. ``run_conversation_async``
takes a ``timeout`` that does this for you.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Awaitable, Iterator, Optional, TypeVar

__all__ = [
    "run_conversation_async",
    "call_host_async",
    "bind_host_loop",
    "get_host_loop",
    "HostLoopUnavailable",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: The host's event loop, for sync code running off it. A ``ContextVar`` rather than a
#: module global for the same reason ``AgentContext`` is: one process can serve several
#: loops -- a test suite creating one per case, a worker with a loop per thread -- and a
#: global would hand the wrong one to whichever got there second.
_host_loop: ContextVar[Optional[asyncio.AbstractEventLoop]] = ContextVar(
    "hermes_core_host_loop", default=None
)


class HostLoopUnavailable(RuntimeError):
    """No usable host loop for :func:`call_host_async`.

    Its own class because the two causes have different fixes and a handler may want to
    fall back rather than fail: either nothing bound a loop (the turn was not started
    through :func:`run_conversation_async` or :func:`bind_host_loop`), or the call was
    reached from the loop thread itself, where waiting on it would deadlock.
    """


def get_host_loop() -> Optional[asyncio.AbstractEventLoop]:
    """The bound host loop, or ``None``. For hosts that want to check rather than catch."""
    return _host_loop.get()


@contextmanager
def bind_host_loop(
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> Iterator[asyncio.AbstractEventLoop]:
    """Bind a loop for :func:`call_host_async` for the duration of the block.

    ``loop`` defaults to the running one, so the usual call is ``with bind_host_loop():``
    from inside a coroutine, immediately around whatever moves the turn onto a thread.

    Bind *before* the work leaves the loop thread. A ``ContextVar`` is copied into a
    thread when the thread is created, so a binding made after the turn is already
    running never reaches it -- the handler sees ``None`` and raises.
    """
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            raise HostLoopUnavailable(
                "bind_host_loop() found no running event loop. Call it from inside a "
                "coroutine, or pass the loop explicitly: bind_host_loop(loop)."
            ) from None
    token = _host_loop.set(loop)
    try:
        yield loop
    finally:
        _host_loop.reset(token)


def call_host_async(coro: Awaitable[T], *, timeout: Optional[float] = None) -> T:
    """Run *coro* on the host's event loop from a synchronous tool handler, and wait.

    The one correct way for a sync handler to reach an ``async def`` service. See the
    module docstring for why the obvious alternatives are not.

    Raises :class:`HostLoopUnavailable` when no loop is bound, or when called from the
    loop thread itself -- the second is the deadlock case, reported rather than hung,
    because a hung request is far harder to diagnose than a raised error. Anything the
    coroutine raises propagates unchanged, so a handler can catch its own exceptions
    normally. ``timeout`` raises :class:`concurrent.futures.TimeoutError`; the coroutine
    is cancelled on the host loop first, so a timed-out call does not leak a task.
    """
    loop = _host_loop.get()
    if loop is None:
        raise HostLoopUnavailable(
            "No host event loop is bound. Start the turn with "
            "run_conversation_async(agent, ...), or wrap it in `with bind_host_loop():`, "
            "so a synchronous tool handler can reach the host's async services."
        )
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        raise HostLoopUnavailable(
            "call_host_async() was called from the host event loop's own thread, where "
            "waiting for the result would deadlock. The turn must run off the loop -- "
            "use run_conversation_async(), or await the coroutine directly if this code "
            "is not actually inside a tool handler."
        )

    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout)
    except BaseException:
        # Covers the timeout and the caller being interrupted. Without this the
        # coroutine keeps running on the host loop with nobody left to read its result
        # -- a leak that survives the request that caused it.
        future.cancel()
        raise


async def run_conversation_async(
    agent: Any,
    user_message: Any,
    *,
    timeout: Optional[float] = None,
    **kwargs: Any,
) -> dict:
    """Run one turn on a worker thread and await its result.

    ``kwargs`` are passed straight to ``agent.run_conversation`` -- ``system_message``,
    ``conversation_history``, ``task_id`` and the rest -- and the turn's result dict comes
    back unchanged. The host loop is bound for the duration, so tool handlers can use
    :func:`call_host_async` with no further setup.

    ``timeout`` is in seconds and is a real stop, not an abandoned wait: it calls
    ``agent.interrupt()`` and lets the turn unwind, because a thread cannot be killed. A
    turn already inside a provider request finishes that request first, so returning can
    lag the timeout by up to one request. :class:`asyncio.TimeoutError` is raised either
    way.

    Cancelling the await *without* a timeout does not stop the turn -- see the module
    docstring. Prefer ``timeout``, or call ``agent.interrupt()`` yourself.
    """
    loop = asyncio.get_running_loop()
    token = _host_loop.set(loop)
    try:
        # Set before the copy: to_thread snapshots the context as it is called, and the
        # thread cannot see a binding made afterwards.
        task = asyncio.ensure_future(
            asyncio.to_thread(lambda: agent.run_conversation(user_message, **kwargs))
        )
        if timeout is None:
            return await task
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout)
        except asyncio.TimeoutError:
            _interrupt(agent, reason=f"run_conversation_async timed out after {timeout}s")
            # Shielded, so the thread is still running and the task is still ours to
            # await. Give the interrupt a bounded chance to land before giving up on the
            # result, rather than leaving an orphaned thread writing to the session store.
            try:
                await asyncio.wait_for(asyncio.shield(task), _INTERRUPT_GRACE_SECONDS)
            except asyncio.TimeoutError:
                logger.warning(
                    "Agent turn did not stop within %ss of being interrupted; the worker "
                    "thread is still running and will finish on its own.",
                    _INTERRUPT_GRACE_SECONDS,
                )
            raise
    finally:
        _host_loop.reset(token)


#: How long to wait for an interrupted turn to unwind before reporting the timeout
#: anyway. Long enough for the loop to reach its next interrupt check, short enough that
#: a wedged turn does not hold the request open.
_INTERRUPT_GRACE_SECONDS = 5.0


def _interrupt(agent: Any, *, reason: str) -> None:
    """Ask the agent to stop, tolerating an agent that cannot.

    ``interrupt`` comes from a mixin every ``AIAgent`` has, but this module accepts any
    object with ``run_conversation`` -- a host's own wrapper, a test double -- so its
    absence is a normal case, not an error. Failing here would replace a timeout with a
    confusing ``AttributeError``.
    """
    interrupt = getattr(agent, "interrupt", None)
    if not callable(interrupt):
        logger.debug("Agent %r has no interrupt(); timed-out turn will run to completion.", type(agent).__name__)
        return
    try:
        interrupt(reason)
    except TypeError:
        try:
            interrupt()
        except Exception:
            logger.warning("Agent interrupt() failed after timeout", exc_info=True)
    except Exception:
        logger.warning("Agent interrupt() failed after timeout", exc_info=True)
