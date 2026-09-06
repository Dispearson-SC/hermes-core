"""Upstream's ``hermes_state`` module, reduced to what the lifted code actually reaches for.

Upstream's ``hermes_state`` is a SQLite session store spread over ~25 files, with an
11-mixin ``SessionDB`` class that includes a Telegram-topics mixin and a gateway mixin.
This core replaced all of it with the ``SessionStore`` protocol in
:mod:`hermes_core.seams.session_store`, and did not lift a line of it.

What nobody noticed is that eight lifted modules still reach for ``hermes_state`` --
always as a lazy import inside an error handler or a degrade path::

    agent/conversation_compression.py    SessionDB
    agent/session_persistence.py         StateDbCorruptError, StateDbReplacedError,
                                         classify_persistence_error,
                                         divert_session_transcript_jsonl
    agent/tool_executor.py               classify_persistence_error
    agent/turn_tool_round.py             classify_persistence_error
    agent/turn_explainers.py             _default_db_path
    agent/inline_tool_executors.py       format_session_db_unavailable
    runtime/goals.py                     SessionDB (x2)

Because the module did not exist, every one of those raised ``ModuleNotFoundError``.
That is a nasty failure mode: the code only runs *when something has already gone
wrong*, so the import error replaces the real error and lands somewhere far from its
cause. Two consequences were measured, not guessed:

* **A conversation could be lost silently.** If the store failed to write,
  ``session_persistence`` reached for ``divert_session_transcript_jsonl`` -- upstream's
  fallback that appends the pending messages to a JSONL file so nothing is lost -- and
  got ``ModuleNotFoundError`` instead. ``turn_finalizer`` swallowed that into
  ``result["cleanup_errors"]``, a field most hosts never read, and ``run_conversation``
  returned ``completed: True`` with the user's message and the model's reply gone.
* **Every other call site crashed the caller**, because none of them has that guard.

So this seam exists to make the degrade paths degrade. It is deliberately small: these
are error handlers, and an error handler that is more elaborate than the thing it
handles is its own hazard.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional

from hermes_core.seams.paths import get_hermes_home
from hermes_core.seams.session_store import SqliteSessionStore

__all__ = [
    "SessionDB",
    "StateDbCorruptError",
    "StateDbReplacedError",
    "DeletedWalGenerationError",
    "CompressionSessionClosedError",
    "classify_persistence_error",
    "divert_session_transcript_jsonl",
    "format_session_db_unavailable",
    "is_automatic_end_reason",
    "acquire",
    "release_or_close",
    "_default_db_path",
]


class StateDbReplacedError(RuntimeError):
    """The database file is no longer the one this handle opened.

    Upstream's case is an out-of-band ``cp``/``mv``/restore under a live process. The
    consumer contract is what matters here and it is unusual: on this error a caller
    must **stop writing** rather than retry, because the handle now points at a
    different file and every further write widens the divergence. That is why it is a
    distinct type and not a string.
    """


class DeletedWalGenerationError(StateDbReplacedError):
    """A live handle holds a deleted WAL generation. Subclasses ``StateDbReplacedError``
    so every consumer that diverts transcripts on a replaced store handles it the same
    way."""


class StateDbCorruptError(sqlite3.DatabaseError):
    """Structural corruption was observed and the handle is quarantined.

    Subclasses ``sqlite3.DatabaseError`` -- as upstream's does -- so existing degrade
    paths that catch the SQLite error keep working unchanged. Recovery is a restart on
    a repaired file, not a reopen.
    """


#: Ordered most-specific-first. Order matters: corruption is checked before disk,
#: because "disk image is malformed" contains the word "disk" and would otherwise be
#: bucketed as a full disk -- sending the user to free up space over a damaged file.
_CAUSE_BY_TYPE: tuple[tuple[type, str], ...] = (
    (StateDbReplacedError, "replaced"),
    (StateDbCorruptError, "corrupt"),
)

_CAUSE_BY_PHRASE: tuple[tuple[tuple[str, ...], str], ...] = (
    (("malformed", "not a database", "corrupt"), "corrupt"),
    (("locked", "busy"), "locked"),
    (("compression lock", "compression lease"), "compression"),
    (("session was rotated", "compression closed"), "compression_closed"),
    (("turn lease", "fenced"), "turn_lease"),
    (("disk", "readonly", "read-only", "no space", "permission"), "disk"),
)


def classify_persistence_error(exc_or_str: Any) -> str:
    """Bucket a persistence failure so the caller can say something useful about it.

    The buckets are upstream's, because lifted call sites branch on these exact
    strings: ``locked`` (busy, retry), ``disk`` (full/read-only/permissions),
    ``compression`` (a live lease refused the write), ``compression_closed`` (adopt the
    rotated session id), ``turn_lease`` (fencing, not storage), ``corrupt`` (file
    damage -- a repair path, not a disk-space one), ``replaced`` (stop writing), and
    ``unknown``.

    Matching by type first and phrase second is also upstream's, and it is the right
    way round: a lease refusal contains neither "locked" nor "busy", so only the type
    identifies it, while a string that survived being passed across a process boundary
    has lost its type and only the phrase is left.
    """
    if exc_or_str is None:
        return "unknown"
    for exc_type, cause in _CAUSE_BY_TYPE:
        if isinstance(exc_or_str, exc_type):
            return cause
    text = str(exc_or_str).lower()
    for markers, cause in _CAUSE_BY_PHRASE:
        if any(marker in text for marker in markers):
            return cause
    return "unknown"


def divert_session_transcript_jsonl(
    session_id: str, messages: Iterable[Any]
) -> Optional[Path]:
    """Append messages to ``<workspace>/sessions/<id>.jsonl`` when the store refused them.

    This is the last thing standing between a failed write and a lost conversation, so
    it deliberately depends on nothing but the filesystem -- no store, no database, no
    lock. Returns the path written, or ``None`` when there was nothing to write.
    """
    sid = str(session_id or "").strip()
    if not sid or not messages:
        return None
    sessions_dir = get_hermes_home() / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    path = sessions_dir / f"{sid}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for message in messages:
            if message is None:
                continue
            record = message if isinstance(message, dict) else {"content": str(message)}
            # `default=str` because this runs on the failure path: a message carrying
            # something unserialisable must still be written, in whatever form.
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return path


def format_session_db_unavailable(
    prefix: str = "Session database not available",
) -> str:
    """The message shown when a feature needs a store and the host configured none.

    Upstream appends the captured initialisation error from its own SQLite setup. This
    core has no such global step -- a host installs a store or does not -- so the
    message says the actionable thing instead of nothing.
    """
    return (
        f"{prefix}. This core has no session store configured; a host installs one "
        "via hermes_core.seams.session_store."
    )


def _default_db_path() -> Path:
    """Where a default SQLite store would live, resolved at call time.

    Call time, not import time, because the workspace is swappable: a host that calls
    ``set_workspace`` after import must still get the right path.
    """
    return get_hermes_home() / "state.db"


class CompressionSessionClosedError(RuntimeError):
    """A write targeted a session that compression has already closed and rotated.

    Distinct from a storage failure: the write did not fail, it arrived at the wrong
    session. The caller's recovery is to adopt the successor id, not to retry -- which
    is why ``classify_persistence_error`` buckets it as ``compression_closed`` and not
    as ``locked``.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(
            f"Session {session_id!r} is closed by compression; "
            "write to its successor instead."
        )


#: End reasons written by *automatic* cleanup rather than by someone deliberately
#: ending a conversation. The distinction decides whether a compression writer that
#: can prove it is alive may keep writing to a session already stamped as ended: if a
#: runtime merely went away, yes; if a person ended the conversation, no. Upstream
#: keeps this taxonomy in one place on purpose -- re-deriving it at a call site is how
#: the two cases get confused.
_AUTOMATIC_END_REASONS = frozenset({
    "agent_close", "ws_orphan_reap", "superseded_by_resume", "startup_orphan_reap",
    "tui_shutdown", "ws_disconnect", "idle_timeout", "lru_evict",
})


def is_automatic_end_reason(reason: Any) -> bool:
    """Whether *reason* marks an automatic cleanup rather than a deliberate ending."""
    return isinstance(reason, str) and reason in _AUTOMATIC_END_REASONS


def acquire(db_path: Any = None) -> Any:
    """Upstream's shared-connection registry, which this core does not have.

    Upstream hands several long-lived in-process callers -- a gateway, a scheduler,
    in-process tools -- one shared writer connection per database file, because SQLite
    in WAL mode wants exactly one writer. Here a host owns its store and decides how it
    is shared, so there is nothing to hand out and no refcount to keep.

    Returning ``None`` rather than raising: every call site treats a falsy result as
    "no shared store, use your own", which is precisely the situation.
    """
    return None


def release_or_close(db: Any) -> None:
    """Release a store obtained from the registry above; with no registry, just close it.

    Upstream's is a drop-in for a plain ``db.close()`` that falls back to closing when
    the object is not registry-managed. Nothing is registry-managed here, so this is
    always the fallback -- and it stays tolerant of a store with no ``close`` at all,
    since the protocol does not require one.
    """
    close = getattr(db, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            # This runs on teardown paths, often while another error is already in
            # flight. A store that will not close cleanly must not replace it.
            pass


#: Upstream's concrete SQLite session class. Lifted code uses this name for exactly one
#: thing -- an ``is``-comparison to tell a live object apart from a hot-reloaded stale
#: class -- so pointing it at this core's SQLite store keeps that check meaningful
#: rather than making it vacuous.
SessionDB = SqliteSessionStore
