"""Session-state seam: where the agent's transcript, titles and token accounting live.

Upstream keeps all of this behind ``hermes_state.py`` plus roughly two dozen
``hermes_state_*.py`` siblings -- well over 4,000 lines in the three biggest alone --
hard-bound to one SQLite file (``state.db``) with no interface anywhere in front of
it. It also carries platform-specific code that has no business in a reasoning core:
``SessionTelegramTopicsMixin`` lives on the same class as message persistence.

**Decision: write a smaller implementation instead of lifting upstream's.** Lifting
``hermes_state.py`` through the manifest would mean dragging the Telegram mixin, the
multi-process turn-lease/lock machinery (``acquire_session_turn_lease``,
``refresh_session_turn_lease``, watermark-fenced compaction races -- all built for
several OS processes sharing one SQLite file behind a gateway) and a dozen
gateway-only session-identity columns across, then spending more ``PATCHES``/``DROPS``
entries stripping them back out than this file is long. None of that is what an
embedded, single-process core needs to keep a conversation across turns. Measured
call surface wins: grepping every ``_session_db.``/``session_db.`` use across the
lifted core (below) gives a surface small enough to implement directly and verify
with tests we actually own.

The measured surface, one line per lifted call site:

* ``run_agent.py``                 -- ``create_session``, ``end_session``
* ``agent/session_persistence.py`` -- ``append_messages_batch``, ``get_compression_tip``,
  ``flush_token_counts``
* ``agent/conversation_loop.py``   -- ``get_session_title``, ``get_session``,
  ``update_system_prompt``
* ``agent/conversation_compression.py`` -- ``get_session_title_source``,
  ``set_session_title``, ``set_session_title_source``, ``get_active_message_watermark``,
  ``publish_compression_child``, ``archive_and_compact``, ``update_system_prompt``,
  ``get_messages_as_conversation`` (duck-typed via ``getattr(type(db), ...)``)
* ``agent/turn_facade_lease.py``   -- ``get_session``, ``get_messages_as_conversation``
  (plus the optional lease methods below)
* ``agent/codex_runtime.py``, ``agent/turn_usage.py`` -- ``queue_token_counts``
* ``agent/system_prompt.py``       -- ``db_path`` (an attribute, read via ``getattr``
  as a diagnostic fallback; never required)

Deliberately **not** in ``SessionStore``: ``acquire_session_turn_lease``,
``refresh_session_turn_lease``, ``resolve_resume_session_id``
(``agent/turn_facade_lease.py``). Every call site reaches these through
``callable(getattr(type(db), "acquire_session_turn_lease", None))`` and skips the
whole cross-process-serialization path when it is absent -- upstream itself treats it
as an optional extension, not part of the contract a store must satisfy. A host
running several processes against one store can add these methods to its own class
without touching this Protocol.

Two implementations ship here:

* ``InMemorySessionStore`` -- dicts, one per process, gone when the process exits.
  Used by tests and by any host that wants working conversation memory (session
  titles, token accounting, the ``session_search`` recall tool, a durable-looking
  transcript across many ``run_conversation()`` calls on one long-lived agent)
  without opting into disk I/O.
* ``SqliteSessionStore`` -- the same behaviour on one SQLite file, for a host that
  wants the transcript to survive the process exiting. It stores each session and
  each message row as one JSON blob per record rather than upstream's wide,
  many-columned tables: this core has no gateway multi-tenancy, no Telegram
  metadata and no per-column query needs, so a rigid schema would only be schema to
  maintain. A host with its own database (Postgres, say) implements ``SessionStore``
  directly -- nothing here requires inheriting from either class.

Both implementations share their business logic (``_SessionStoreCore``) and differ
only in the handful of storage primitives listed on it. ``archive_and_compact`` and
``publish_compression_child`` here are honest simplifications of upstream's
watermark/lock-holder-fenced compaction: this store has exactly one writer (the
process holding it), so there is no concurrent-writer race to fence against, and
replacing a session's row set outright is equivalent to upstream's more careful
soft-archive-then-insert dance without needing to reproduce it.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Union, runtime_checkable

from hermes_core.seams.paths import get_hermes_home

__all__ = [
    "SessionStore",
    "SessionReader",
    "SessionStoreBase",
    "InMemorySessionStore",
    "SqliteSessionStore",
]

PathLike = Union[str, "Path"]

# Reasoning/codex fields carried on assistant rows (mirrors
# ``hermes_core.agent.session_persistence._ROW_REASONING_KEYS`` -- duplicated rather
# than imported, so this seam does not reach into lifted, regeneratable code).
_ROW_REASONING_KEYS = (
    "reasoning", "reasoning_content", "reasoning_details", "codex_reasoning_items", "codex_message_items",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@runtime_checkable
class SessionStore(Protocol):
    """What the lifted core actually calls on ``agent._session_db``.

    Every method here is exercised by a real call site (see the module docstring for
    the exact grep). Implement this structurally -- ``@runtime_checkable`` means
    ``isinstance(x, SessionStore)`` works for any object with the right methods, no
    inheritance required.
    """

    def create_session(
        self, *, session_id: str, source: str, model: str,
        model_config: Optional[Dict[str, Any]] = None, system_prompt: Optional[str] = None,
        user_id: Optional[str] = None, session_key: Optional[str] = None, chat_id: Optional[str] = None,
        chat_type: Optional[str] = None, thread_id: Optional[str] = None, display_name: Optional[str] = None,
        origin_json: Optional[str] = None, parent_session_id: Optional[str] = None, cwd: Optional[str] = None,
        profile_name: Optional[str] = None,
    ) -> None:
        """Create the session row. Idempotent: a retried create (transient failure,
        then a later turn retries) must not clobber an already-durable row."""
        ...

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """The session row, or ``None`` if it does not exist. Never raises for a
        missing id -- callers rely on that to distinguish "no row yet" from "broken
        store" (``#84234`` upstream: a probe failure must not read as "row absent")."""
        ...

    def end_session(self, session_id: str, reason: str) -> None:
        """Mark the session ended. First reason wins -- a session already ended keeps
        its original ``end_reason``."""
        ...

    def append_messages_batch(
        self, *, session_id: str, messages: List[Dict[str, Any]],
        compression_lock_holder: Optional[str] = None, turn_lease_holder: Optional[str] = None,
        turn_lease_ttl_seconds: float = 300.0,
    ) -> None:
        """Append rows in order, in one transaction: on failure nothing lands."""
        ...

    def get_messages_as_conversation(
        self, session_id: str, *, repair_alternation: bool = True, include_row_ids: bool = False,
    ) -> List[Dict[str, Any]]:
        """The session's durable rows, as API-ready message dicts, in append order."""
        ...

    def get_session_title(self, session_id: str) -> Optional[str]: ...
    def set_session_title(self, session_id: str, title: str) -> None: ...
    def get_session_title_source(self, session_id: str) -> Optional[str]: ...
    def set_session_title_source(self, session_id: str, source: str) -> None: ...

    def update_system_prompt(self, session_id: str, system_prompt: Optional[str]) -> None:
        """Persist the cached system prompt so a resumed session can compare it
        against a freshly-built one instead of always rebuilding (prefix-cache reuse)."""
        ...

    def queue_token_counts(self, session_id: str, **counts: Any) -> None:
        """Accumulate per-call token deltas for the session."""
        ...

    def flush_token_counts(self) -> None:
        """Drain any queued token-count writes. Called at every persist point;
        must be a cheap no-op when nothing is queued."""
        ...

    def get_compression_tip(self, session_id: str) -> Optional[str]:
        """Walk ``session_id``'s compression-child chain to its live tip. Returns
        ``session_id`` itself when it has no recorded continuation."""
        ...

    def get_active_message_watermark(self, session_id: str) -> int:
        """How many of the session's rows are still active (not superseded by a
        later compaction)."""
        ...

    def publish_compression_child(
        self, *, parent_session_id: str, child_session_id: str, source: str, model: str,
        model_config: Optional[Dict[str, Any]] = None, system_prompt: Optional[str] = None,
        messages: Iterable[Dict[str, Any]] = (), **extra: Any,
    ) -> None:
        """Create ``child_session_id`` as the compacted continuation of
        ``parent_session_id`` and record the link ``get_compression_tip`` walks."""
        ...

    def archive_and_compact(self, session_id: str, messages: List[Dict[str, Any]], **extra: Any) -> None:
        """Replace ``session_id``'s active transcript with ``messages`` in place
        (same id, no rotation)."""
        ...


# -- shared business logic -----------------------------------------------------------

@runtime_checkable
class SessionReader(Protocol):
    """Enumerating conversations -- a *reader's* need, not an agent's.

    Deliberately separate from ``SessionStore``. An agent always knows its own session
    id and never enumerates; a dashboard, an audit or a support view needs exactly the
    opposite. Folding this into ``SessionStore`` would force every host that only wants
    an agent to implement a method for a front end it may not have -- and it did worse
    than that when tried: adding one method to a published Protocol invalidated every
    existing implementation, because structural typing checks the whole surface.

    Both shipped stores implement it, so a host that uses them gets it for nothing::

        for row in store.list_sessions(limit=20):
            print(row["session_id"], row["message_count"], row["started_at"])
            for message in store.get_messages_as_conversation(row["session_id"]):
                ...
    """

    def list_sessions(
        self, *, limit: int = 50, source: str = "", include_ended: bool = True
    ) -> List[Dict[str, Any]]:
        """Sessions newest first, each carrying a ``message_count``."""
        ...


class _SessionStoreCore:
    """``SessionStore`` behaviour, expressed in terms of five storage primitives a
    subclass provides. Nothing below touches a file or a table directly -- that
    keeps ``InMemorySessionStore`` and ``SqliteSessionStore`` from drifting apart."""

    _lock: threading.RLock

    def _read_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def _write_session(self, session_id: str, row: Dict[str, Any]) -> None:
        raise NotImplementedError

    def _read_messages(self, session_id: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def _append_rows(self, session_id: str, rows: List[Dict[str, Any]]) -> None:
        raise NotImplementedError

    def _replace_rows(self, session_id: str, rows: List[Dict[str, Any]]) -> None:
        raise NotImplementedError

    def _all_sessions(self) -> List[Dict[str, Any]]:
        """Every session row, in creation order.

        The sixth primitive, and the only one no *agent* needs -- a turn always knows
        its own session id. It exists for the reader on the other side: a dashboard, an
        audit, a support view. Without it a host can read any conversation it can name
        and enumerate none of them, which makes a front end impossible to build.
        """
        raise NotImplementedError

    # -- reading, for a front end (see SessionReader) ----------------------------

    def list_sessions(
        self, *, limit: int = 50, source: str = "", include_ended: bool = True
    ) -> List[Dict[str, Any]]:
        """Sessions, newest first, each with a ``message_count``.

        The count comes along because a list of conversations that cannot say how long
        each one is forces the caller into one read per row just to render a list.
        """
        rows: List[Dict[str, Any]] = []
        for row in self._all_sessions():
            if source and row.get("source") != source:
                continue
            if not include_ended and row.get("ended_at") is not None:
                continue
            summary = dict(row)
            summary["message_count"] = len(self._read_messages(row.get("session_id", "")))
            rows.append(summary)
        rows.sort(key=lambda r: str(r.get("started_at") or ""), reverse=True)
        return rows[:limit]

    # -- sessions ---------------------------------------------------------------

    def create_session(self, *, session_id: str, source: str, model: str, model_config=None,
                        system_prompt=None, user_id=None, session_key=None, chat_id=None, chat_type=None,
                        thread_id=None, display_name=None, origin_json=None, parent_session_id=None,
                        cwd=None, profile_name=None) -> None:
        with self._lock:
            if self._read_session(session_id) is not None:
                return
            self._write_session(session_id, {
                "session_id": session_id, "source": source, "model": model, "model_config": model_config,
                "system_prompt": system_prompt, "user_id": user_id, "session_key": session_key,
                "chat_id": chat_id, "chat_type": chat_type, "thread_id": thread_id,
                "display_name": display_name, "origin_json": origin_json,
                "parent_session_id": parent_session_id, "cwd": cwd, "profile_name": profile_name,
                "title": None, "title_source": None, "created_at": _now_iso(), "ended_at": None,
                "end_reason": None, "compression_child": None, "token_counts": {},
            })

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._read_session(session_id)

    def end_session(self, session_id: str, reason: str) -> None:
        with self._lock:
            row = self._read_session(session_id)
            if row is None or row.get("ended_at") is not None:  # first reason wins
                return
            row["ended_at"], row["end_reason"] = _now_iso(), reason
            self._write_session(session_id, row)

    # -- messages -----------------------------------------------------------------

    def append_messages_batch(self, *, session_id: str, messages: List[Dict[str, Any]], **_lease_kwargs: Any) -> None:
        if not messages:
            return
        with self._lock:
            self._append_rows(session_id, [dict(m) for m in messages])

    def get_messages_as_conversation(self, session_id: str, *, repair_alternation: bool = True,
                                      include_row_ids: bool = False) -> List[Dict[str, Any]]:
        return [_row_to_wire_message(row, include_row_ids=include_row_ids) for row in self._read_messages(session_id)]

    def get_active_message_watermark(self, session_id: str) -> int:
        return len(self._read_messages(session_id))

    def archive_and_compact(self, session_id: str, messages: List[Dict[str, Any]], **_extra: Any) -> None:
        with self._lock:
            self._replace_rows(session_id, [dict(m) for m in messages])

    # -- titles / prompt ------------------------------------------------------------

    def get_session_title(self, session_id: str) -> Optional[str]:
        row = self._read_session(session_id)
        return row.get("title") if row else None

    def set_session_title(self, session_id: str, title: str) -> None:
        with self._lock:
            row = self._read_session(session_id)
            if row is None:
                return
            row["title"] = title
            self._write_session(session_id, row)

    def get_session_title_source(self, session_id: str) -> Optional[str]:
        row = self._read_session(session_id)
        return row.get("title_source") if row else None

    def set_session_title_source(self, session_id: str, source: str) -> None:
        with self._lock:
            row = self._read_session(session_id)
            if row is None:
                return
            row["title_source"] = source
            self._write_session(session_id, row)

    def update_system_prompt(self, session_id: str, system_prompt: Optional[str]) -> None:
        with self._lock:
            row = self._read_session(session_id)
            if row is None:
                return
            row["system_prompt"] = system_prompt
            self._write_session(session_id, row)

    # -- token accounting -------------------------------------------------------------

    def queue_token_counts(self, session_id: str, **counts: Any) -> None:
        # Upstream queues these and drains them from a background writer because a
        # synchronous UPDATE on a cold state.db stalled the tool loop. Both stores here
        # apply the delta immediately: an in-memory dict update is not a stall, and
        # SqliteSessionStore accepts the (small, rare) synchronous write in exchange for
        # not needing a background thread at all. `flush_token_counts` stays a no-op.
        with self._lock:
            row = self._read_session(session_id)
            if row is None:
                return
            totals = dict(row.get("token_counts") or {})
            for key, value in counts.items():
                if value is None:
                    continue
                totals[key] = (totals.get(key) or 0) + value
            row["token_counts"] = totals
            self._write_session(session_id, row)

    def flush_token_counts(self) -> None:
        return None

    # -- compression chain --------------------------------------------------------

    def get_compression_tip(self, session_id: str) -> Optional[str]:
        seen = set()
        current = session_id
        while True:
            row = self._read_session(current)
            child = row.get("compression_child") if row else None
            if not child or child in seen:
                return current
            seen.add(current)
            current = child

    def publish_compression_child(self, *, parent_session_id: str, child_session_id: str, source: str,
                                   model: str, model_config=None, system_prompt=None, messages: Iterable[Dict[str, Any]] = (),
                                   **extra: Any) -> None:
        with self._lock:
            self.create_session(
                session_id=child_session_id, source=source, model=model, model_config=model_config,
                system_prompt=system_prompt, parent_session_id=parent_session_id,
                cwd=extra.get("cwd"), profile_name=extra.get("profile_name"),
            )
            messages = list(messages)
            if messages:
                self._append_rows(child_session_id, [dict(m) for m in messages])
            parent = self._read_session(parent_session_id)
            if parent is not None:
                parent["compression_child"] = child_session_id
                self._write_session(parent_session_id, parent)


#: The public name for ``_SessionStoreCore``.
#:
#: A host implementing ``SessionStore`` over its own database does not want to write
#: sixteen methods -- it wants to write the six storage primitives above and inherit the
#: rest, which is exactly what this class is for. Telling hosts to subclass a private,
#: underscore-prefixed name would be telling them to depend on something this package
#: reserves the right to move; the alias makes the intended base part of the contract
#: while leaving the internal name (and every existing reference to it) alone.
SessionStoreBase = _SessionStoreCore


def _row_to_wire_message(row: Dict[str, Any], *, include_row_ids: bool) -> Dict[str, Any]:
    """A stored row back into the shape the turn loop feeds to a provider.

    ``api_content`` wins over ``content`` when present -- it is the exact bytes sent
    to the API when they differ from the clean, displayed transcript (see
    ``session_persistence._db_flush_row``); replaying anything else would resend a
    message the model never actually saw.
    """
    content = row.get("api_content")
    if content is None:
        content = row.get("content")
    msg: Dict[str, Any] = {"role": row.get("role"), "content": content}
    if row.get("role") == "tool":
        msg["name"] = row.get("tool_name")
        msg["tool_call_id"] = row.get("tool_call_id")
    elif row.get("tool_calls"):
        msg["tool_calls"] = row["tool_calls"]
    for key in _ROW_REASONING_KEYS:
        if row.get(key) is not None:
            msg[key] = row[key]
    if include_row_ids and isinstance(row.get("_row_id"), int):
        msg["_row_id"] = row["_row_id"]
    return msg


# -- in-memory implementation ---------------------------------------------------------

class InMemorySessionStore(_SessionStoreCore):
    """Dict-backed ``SessionStore``. Nothing survives the process; nothing touches disk.

    This is what an agent gets by default (see the ``agent_init`` patch in
    ``tools/lift.py``) so conversation memory, session titles and token accounting
    work out of the box for a host that has not opted into durable storage.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._messages: Dict[str, List[Dict[str, Any]]] = {}
        self._next_row_id = 1

    def _read_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        row = self._sessions.get(session_id)
        return dict(row) if row is not None else None

    def _write_session(self, session_id: str, row: Dict[str, Any]) -> None:
        self._sessions[session_id] = dict(row)

    def _read_messages(self, session_id: str) -> List[Dict[str, Any]]:
        return [dict(r) for r in self._messages.get(session_id, [])]

    def _append_rows(self, session_id: str, rows: List[Dict[str, Any]]) -> None:
        stored = self._messages.setdefault(session_id, [])
        for row in rows:
            row["_row_id"] = self._next_row_id
            self._next_row_id += 1
            stored.append(row)

    def _replace_rows(self, session_id: str, rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            row["_row_id"] = self._next_row_id
            self._next_row_id += 1
        self._messages[session_id] = rows

    def _all_sessions(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in self._sessions.values()]


# -- SQLite implementation ------------------------------------------------------------

class SqliteSessionStore(_SessionStoreCore):
    """The same behaviour as ``InMemorySessionStore``, persisted to one SQLite file.

    Each session and each message row is stored as a single JSON blob (``data``
    column) rather than upstream's wide, many-columned schema -- see the module
    docstring for why a rigid per-field schema would only be schema to maintain here.
    Row order is the monotonic ``id`` SQLite assigns, so ``ORDER BY id`` reproduces
    append order without a separate sequence column.
    """

    def __init__(self, path: Optional[PathLike] = None) -> None:
        self.db_path = str(path) if path is not None else str(get_hermes_home() / "sessions.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, data TEXT NOT NULL)")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS messages ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, data TEXT NOT NULL)"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS messages_session_id ON messages(session_id)")
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _read_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        cur = self._conn.execute("SELECT data FROM sessions WHERE session_id = ?", (session_id,))
        row = cur.fetchone()
        return json.loads(row[0]) if row else None

    def _write_session(self, session_id: str, row: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (session_id, data) VALUES (?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET data = excluded.data",
                (session_id, json.dumps(row)),
            )
            self._conn.commit()

    def _read_messages(self, session_id: str) -> List[Dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT id, data FROM messages WHERE session_id = ? ORDER BY id", (session_id,)
        )
        rows = []
        for row_id, data in cur.fetchall():
            decoded = json.loads(data)
            decoded["_row_id"] = row_id
            rows.append(decoded)
        return rows

    def _append_rows(self, session_id: str, rows: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT INTO messages (session_id, data) VALUES (?, ?)",
                [(session_id, json.dumps(r)) for r in rows],
            )
            self._conn.commit()

    def _replace_rows(self, session_id: str, rows: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            self._append_rows(session_id, rows)

    def _all_sessions(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT data FROM sessions").fetchall()
        return [json.loads(r[0]) for r in rows]
