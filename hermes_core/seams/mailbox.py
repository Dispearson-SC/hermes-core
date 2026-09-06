"""Durable messages between agents, and between an agent and a person.

Modelled on upstream Hermes's A2A platform plugin (``plugins/platforms/a2a/``, 2,345
lines implementing the open A2A protocol). That plugin cannot be lifted -- it is a
gateway platform adapter, welded to ``register_platform()``, ``MessageEvent`` routing
and a live session object, none of which exist here. Its *design* is worth taking, and
this module takes it, with two deliberate departures.

What is taken from A2A
----------------------
* **A task lifecycle with an ``input_required`` state.** A2A's insight is that "waiting
  on a person" is a first-class state of a request, not an absence of one. That single
  state is what makes an escalation expressible.
* **A context id distinct from the request id.** A2A keys persistence by ``contextId``
  so a thread of related exchanges hangs together while individual requests come and go.
  Here it lets one customer conversation carry several escalations.
* **Scoped reads.** A2A records carry an agent slug and a tenant, and a reader outside
  that scope gets *not found* rather than a permission error. Every read here takes an
  ``org`` for the same reason: one organisation must not be able to probe for another's
  request ids.
* **Terminal requests stay queryable**, so "what did we ask, and what came back" is
  answerable after the fact.

What is deliberately different
------------------------------
* **Durable, not in-memory.** A2A's ``TaskStore`` is an ``OrderedDict`` and its inbound
  HTTP request blocks on a ``Future`` (``protocol.py:299-308``). Conversation *text* is
  persisted to JSONL, but the pending-task state is not: restart the process and a task
  waiting on a human is gone, even though the transcript survived. For an escalation
  that may wait hours for a busy owner to reply, that is the whole problem, so the
  record here is the durable thing.
* **The sender does not block.** A2A's ``a2a_call`` waits up to 300 seconds for a reply.
  A sales agent serving many customers at once cannot hold a turn open waiting for a
  person -- and every "ask a human" mechanism in upstream (approval gateway, clarify,
  MCP elicitation) is a blocked thread with in-memory state, which is why none of them
  fit. :meth:`AgentMailbox.send` returns immediately with a request id; the answer is
  collected on a later turn.

The shape
---------
A sales agent hits a question it cannot answer::

    request_id = mailbox.send(MailboxMessage(
        org="acme", sender="sales", recipient="personal",
        context_id="whatsapp:+5493411234567",
        subject="Does the customer's warranty cover water damage?",
    ))
    # answer the customer now: "let me check that for you"

The owner's personal agent picks it up, asks the human, and answers::

    for request in mailbox.poll(org="acme", recipient="personal"):
        mailbox.mark_input_required(request.request_id, asked="...")
        # ... hours pass, possibly a restart ...
        mailbox.answer(request.request_id, "Yes, up to 6 months.")

The sales agent finds the answer on its next turn with that customer::

    for answered in mailbox.answers_for(org="acme", recipient="sales",
                                        context_id="whatsapp:+5493411234567"):
        ...

Nothing here sends anything anywhere. A mailbox is a durable record; delivery -- a
webhook, a poll, a push -- is the host's, and ``agent/outbound_webhooks.py`` is already
in this core for the "something arrived, go look" signal.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, runtime_checkable

from hermes_core.seams.paths import get_hermes_home

__all__ = [
    "MailboxState",
    "MailboxMessage",
    "AgentMailbox",
    "InMemoryMailbox",
    "SqliteMailbox",
]


class MailboxState:
    """Where a request is in its life.

    A2A's states, trimmed to the ones that mean something without an HTTP task API.
    Strings rather than an enum because they are persisted and read back by hosts and
    front-ends that are not Python.
    """

    #: Sent, nobody has picked it up.
    PENDING = "pending"
    #: An agent has claimed it and is working on it.
    WORKING = "working"
    #: Waiting on a person. The state that makes an escalation expressible.
    INPUT_REQUIRED = "input_required"
    #: Answered.
    COMPLETED = "completed"
    #: Could not be answered -- nobody knew, or the agent gave up.
    FAILED = "failed"
    #: Withdrawn before it was answered.
    CANCELED = "canceled"

    TERMINAL = frozenset({COMPLETED, FAILED, CANCELED})
    OPEN = frozenset({PENDING, WORKING, INPUT_REQUIRED})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MailboxMessage:
    """One request from one agent to another, and its answer when it arrives.

    Both halves live in one record on purpose. Splitting request and response into two
    rows means a reader has to join them to answer "is this still waiting", which is the
    question every caller actually asks.
    """

    #: The organisation. Every read is scoped by it -- see the module docstring.
    org: str
    #: Who is asking. An agent name inside the organisation.
    sender: str
    #: Who is being asked.
    recipient: str
    #: What is being asked, in plain language.
    subject: str

    #: Ties related exchanges together -- typically the end customer's conversation, so
    #: an answer can be routed back to the right person. Free-form; the mailbox never
    #: interprets it.
    context_id: str = ""

    #: Anything the recipient needs that does not belong in prose: the customer's
    #: question verbatim, an order id, a product code. Must be JSON-serialisable.
    payload: Dict[str, Any] = field(default_factory=dict)

    request_id: str = ""
    state: str = MailboxState.PENDING
    created_at: str = ""
    updated_at: str = ""

    #: What the recipient asked a human, once it decided it needed one. Kept separate
    #: from ``subject`` because an agent usually rephrases before asking a person.
    asked: str = ""
    #: The answer, once there is one.
    answer: str = ""
    #: Who or what answered -- an agent name, a person's identifier. For an audit trail:
    #: "the owner said so" and "the agent guessed" must not look alike later.
    answered_by: str = ""

    def __post_init__(self) -> None:
        if not self.request_id:
            self.request_id = uuid.uuid4().hex
        if not self.created_at:
            self.created_at = _now()
        if not self.updated_at:
            self.updated_at = self.created_at

    @property
    def is_open(self) -> bool:
        return self.state in MailboxState.OPEN

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "MailboxMessage":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in row.items() if k in known})


@runtime_checkable
class AgentMailbox(Protocol):
    """Durable request/answer between agents, and between an agent and a person.

    Implement this to put the mailbox wherever the host already keeps state -- Postgres,
    Redis, a queue. Two implementations ship below; both are ordinary implementations of
    this protocol with nothing privileged about them.

    Every method takes ``org`` and must scope by it. A request id from another
    organisation must read as absent, never as forbidden: telling a caller that an id
    exists but is not theirs confirms the id.
    """

    def send(self, message: MailboxMessage) -> str:
        """Record a request and return its id. Never blocks and never waits for a reply."""
        ...

    def get(self, org: str, request_id: str) -> Optional[MailboxMessage]:
        """One request, or ``None`` when it does not exist in this organisation."""
        ...

    def poll(self, org: str, recipient: str, *, limit: int = 20, claim: bool = True) -> List[MailboxMessage]:
        """Requests waiting for *recipient*, oldest first.

        With ``claim`` set, each returned request moves to ``working`` in the same
        breath, so two workers polling at once do not both take the same one.
        """
        ...

    def answers_for(
        self, org: str, recipient: str, *, context_id: str = "", since: str = "", limit: int = 20
    ) -> List[MailboxMessage]:
        """Answered requests that *recipient* originally sent.

        How an agent finds out that something it asked hours ago has come back.
        """
        ...

    def mark_input_required(self, org: str, request_id: str, *, asked: str = "") -> None:
        """This request is now waiting on a person."""
        ...

    def answer(self, org: str, request_id: str, answer: str, *, answered_by: str = "") -> None:
        """Complete a request with its answer."""
        ...

    def fail(self, org: str, request_id: str, reason: str = "", *, answered_by: str = "") -> None:
        """Close a request that will not be answered."""
        ...

    def cancel(self, org: str, request_id: str, reason: str = "") -> None:
        """Withdraw a request that is no longer wanted."""
        ...

    def list(
        self, org: str, *, state: str = "", recipient: str = "", sender: str = "",
        context_id: str = "", limit: int = 50,
    ) -> List[MailboxMessage]:
        """Requests matching every filter given, newest first. For dashboards and audits."""
        ...


class _MailboxCore:
    """Everything both shipped mailboxes do identically.

    Same split as ``SessionStore``: all the policy here, and three storage primitives
    below it. A third backend is those three methods, not a reimplementation of the
    state machine -- which is what keeps two backends from drifting apart.
    """

    # -- storage primitives a backend provides --------------------------------

    def _read(self, org: str, request_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def _write(self, row: Dict[str, Any]) -> None:
        raise NotImplementedError

    def _scan(self, org: str) -> List[Dict[str, Any]]:
        """Every row for an organisation, oldest first."""
        raise NotImplementedError

    # -- the protocol ---------------------------------------------------------

    def send(self, message: MailboxMessage) -> str:
        if not message.org:
            # Refused rather than defaulted: an unscoped request is readable by every
            # organisation, and the failure would show up as a leak, not as an error.
            raise ValueError("a mailbox message must name an org")
        if not message.recipient:
            raise ValueError("a mailbox message must name a recipient")
        message.updated_at = _now()
        self._write(message.to_dict())
        return message.request_id

    def get(self, org: str, request_id: str) -> Optional[MailboxMessage]:
        row = self._read(org, request_id)
        return MailboxMessage.from_dict(row) if row else None

    def poll(self, org: str, recipient: str, *, limit: int = 20, claim: bool = True) -> List[MailboxMessage]:
        out: List[MailboxMessage] = []
        with self._lock:
            for row in self._scan(org):
                if len(out) >= limit:
                    break
                if row.get("recipient") != recipient or row.get("state") != MailboxState.PENDING:
                    continue
                if claim:
                    row = dict(row)
                    row["state"] = MailboxState.WORKING
                    row["updated_at"] = _now()
                    self._write(row)
                out.append(MailboxMessage.from_dict(row))
        return out

    def answers_for(
        self, org: str, recipient: str, *, context_id: str = "", since: str = "", limit: int = 20
    ) -> List[MailboxMessage]:
        # `recipient` here means the original sender: this asks "what came back to me".
        out: List[MailboxMessage] = []
        for row in self._scan(org):
            if row.get("sender") != recipient:
                continue
            if row.get("state") not in MailboxState.TERMINAL:
                continue
            if context_id and row.get("context_id") != context_id:
                continue
            if since and str(row.get("updated_at", "")) <= since:
                continue
            out.append(MailboxMessage.from_dict(row))
        return out[-limit:]

    def _transition(self, org: str, request_id: str, **changes: Any) -> None:
        with self._lock:
            row = self._read(org, request_id)
            if row is None:
                # Silent, like every other scoped read: a request id from another
                # organisation must be indistinguishable from one that never existed.
                return
            row = dict(row)
            row.update(changes)
            row["updated_at"] = _now()
            self._write(row)

    def mark_input_required(self, org: str, request_id: str, *, asked: str = "") -> None:
        self._transition(org, request_id, state=MailboxState.INPUT_REQUIRED, asked=asked)

    def answer(self, org: str, request_id: str, answer: str, *, answered_by: str = "") -> None:
        self._transition(
            org, request_id, state=MailboxState.COMPLETED, answer=answer, answered_by=answered_by
        )

    def fail(self, org: str, request_id: str, reason: str = "", *, answered_by: str = "") -> None:
        self._transition(
            org, request_id, state=MailboxState.FAILED, answer=reason, answered_by=answered_by
        )

    def cancel(self, org: str, request_id: str, reason: str = "") -> None:
        self._transition(org, request_id, state=MailboxState.CANCELED, answer=reason)

    def list(
        self, org: str, *, state: str = "", recipient: str = "", sender: str = "",
        context_id: str = "", limit: int = 50,
    ) -> List[MailboxMessage]:
        rows = self._scan(org)
        if state:
            rows = [r for r in rows if r.get("state") == state]
        if recipient:
            rows = [r for r in rows if r.get("recipient") == recipient]
        if sender:
            rows = [r for r in rows if r.get("sender") == sender]
        if context_id:
            rows = [r for r in rows if r.get("context_id") == context_id]
        # Newest first: a dashboard wants the most recent, and `_scan` is oldest-first
        # because claiming has to be fair.
        return [MailboxMessage.from_dict(r) for r in reversed(rows)][:limit]


class InMemoryMailbox(_MailboxCore):
    """A mailbox that lives and dies with the process.

    For tests and for a single-process host that genuinely does not need to survive a
    restart. Anything that escalates to a human wants :class:`SqliteMailbox` -- the
    whole reason this module exists is that in-memory state loses the escalation.
    """

    def __init__(self) -> None:
        self._rows: Dict[str, Dict[str, Any]] = {}
        self._order: List[str] = []
        self._lock = threading.RLock()

    def _key(self, org: str, request_id: str) -> str:
        return f"{org}\x00{request_id}"

    def _read(self, org: str, request_id: str) -> Optional[Dict[str, Any]]:
        return self._rows.get(self._key(org, request_id))

    def _write(self, row: Dict[str, Any]) -> None:
        key = self._key(row["org"], row["request_id"])
        if key not in self._rows:
            self._order.append(key)
        self._rows[key] = dict(row)

    def _scan(self, org: str) -> List[Dict[str, Any]]:
        return [dict(self._rows[k]) for k in self._order
                if k in self._rows and self._rows[k].get("org") == org]


class SqliteMailbox(_MailboxCore):
    """A mailbox on disk, so an escalation survives a restart.

    One row per request, with the free-form parts in a JSON column. Columns exist only
    for what is filtered or ordered on; everything else would be schema for its own sake.

    ``check_same_thread=False`` with an ``RLock`` around every access, matching
    ``SqliteSessionStore``: a host serving several conversations does so from several
    threads, and one lock is simpler to reason about than a connection pool for a table
    this small.
    """

    def __init__(self, path: Optional[Any] = None) -> None:
        self._path = Path(path) if path is not None else get_hermes_home() / "mailbox.db"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self._path), check_same_thread=False)
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS mailbox (
                org         TEXT NOT NULL,
                request_id  TEXT NOT NULL,
                recipient   TEXT NOT NULL,
                sender      TEXT NOT NULL,
                state       TEXT NOT NULL,
                context_id  TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                data        TEXT NOT NULL,
                PRIMARY KEY (org, request_id)
            )
            """
        )
        # The polling query -- pending work for one recipient, oldest first -- is the
        # one that runs constantly.
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS mailbox_poll ON mailbox (org, recipient, state, created_at)"
        )
        self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _read(self, org: str, request_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._db.execute(
                "SELECT data FROM mailbox WHERE org = ? AND request_id = ?", (org, request_id)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def _write(self, row: Dict[str, Any]) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO mailbox "
                "(org, request_id, recipient, sender, state, context_id, created_at, updated_at, data) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (row["org"], row["request_id"], row["recipient"], row["sender"], row["state"],
                 row.get("context_id", ""), row["created_at"], row["updated_at"],
                 json.dumps(row, ensure_ascii=False)),
            )
            self._db.commit()

    def _scan(self, org: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT data FROM mailbox WHERE org = ? ORDER BY created_at, rowid", (org,)
            ).fetchall()
        return [json.loads(r[0]) for r in rows]
