"""The escalation flow, and the property nothing in upstream Hermes has.

Every "ask a human" mechanism in upstream -- the approval gateway, ``clarify``, MCP
elicitation -- is a blocked thread with process-memory state. Upstream's A2A plugin
persists conversation *text* to JSONL but keeps its task store in an ``OrderedDict``, so
a task waiting on a person dies with the process even though the transcript survives.

An escalation that waits hours for a busy owner and then loses the question to a deploy
is not an escalation. So the test that matters most here is the one that reopens the
store from scratch, mid-flight, and finds the question still waiting.
"""

import tempfile
import threading

import pytest

from hermes_core.seams.mailbox import (
    InMemoryMailbox,
    MailboxMessage,
    MailboxState,
    SqliteMailbox,
)


@pytest.fixture(params=["memory", "sqlite"])
def mailbox(request, tmp_path):
    """Both shipped backends run the same suite -- they share all their policy, so a
    behaviour that differs between them is a bug in one of them."""
    if request.param == "memory":
        yield InMemoryMailbox()
        return
    box = SqliteMailbox(tmp_path / "mailbox.db")
    yield box
    box.close()


def ask(org="acme", sender="sales", recipient="personal", subject="¿Cubre la garantía?", **kw):
    return MailboxMessage(org=org, sender=sender, recipient=recipient, subject=subject, **kw)


# -- the flow ------------------------------------------------------------------------

def test_sending_does_not_block_and_returns_an_id(mailbox):
    """The property that makes this usable at all.

    A sales agent serving many customers cannot hold a turn open waiting for a person,
    which is exactly what upstream's ``a2a_call`` (300s) and every approval wait do.
    """
    request_id = mailbox.send(ask())

    assert request_id
    assert mailbox.get("acme", request_id).state == MailboxState.PENDING


def test_the_whole_escalation_end_to_end(mailbox):
    """Customer asks → sales escalates → owner's agent asks the human → answer returns.

    Three actors, and the record carries who answered: "the owner said so" and "an agent
    guessed" must not look the same in an audit later.
    """
    request_id = mailbox.send(ask(context_id="whatsapp:+54911", payload={"pregunta": "agua"}))

    claimed = mailbox.poll("acme", "personal")
    assert [c.request_id for c in claimed] == [request_id]
    assert claimed[0].payload == {"pregunta": "agua"}

    mailbox.mark_input_required("acme", request_id, asked="¿La garantía cubre daño por agua?")
    assert mailbox.get("acme", request_id).state == MailboxState.INPUT_REQUIRED

    mailbox.answer("acme", request_id, "Sí, hasta 6 meses.", answered_by="owner")

    back = mailbox.answers_for("acme", "sales", context_id="whatsapp:+54911")
    assert len(back) == 1
    assert back[0].answer == "Sí, hasta 6 meses."
    assert back[0].answered_by == "owner"
    assert back[0].state == MailboxState.COMPLETED


def test_polling_claims_so_two_workers_cannot_take_the_same_request(mailbox):
    """Without claiming, two workers both answer the customer -- twice."""
    mailbox.send(ask())

    first = mailbox.poll("acme", "personal")
    second = mailbox.poll("acme", "personal")

    assert len(first) == 1
    assert second == []


def test_concurrent_pollers_never_claim_the_same_request(mailbox):
    ids = {mailbox.send(ask(subject=f"pregunta {i}")) for i in range(20)}
    claimed, lock = [], threading.Lock()

    def worker():
        got = mailbox.poll("acme", "personal", limit=20)
        with lock:
            claimed.extend(m.request_id for m in got)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(claimed) == sorted(ids), "a request was claimed twice or lost"


def test_peeking_without_claiming_leaves_the_request_pending(mailbox):
    """For a dashboard that wants to look without taking the work."""
    mailbox.send(ask())

    peeked = mailbox.poll("acme", "personal", claim=False)

    assert len(peeked) == 1
    assert mailbox.poll("acme", "personal")[0].request_id == peeked[0].request_id


def test_an_unanswerable_request_is_closed_not_left_hanging(mailbox):
    request_id = mailbox.send(ask())
    mailbox.fail("acme", request_id, "el encargado no sabe", answered_by="owner")

    assert mailbox.get("acme", request_id).state == MailboxState.FAILED
    # The sales agent must still learn it came back -- a failure is an answer too.
    assert [m.request_id for m in mailbox.answers_for("acme", "sales")] == [request_id]


def test_a_cancelled_request_stops_being_pollable(mailbox):
    request_id = mailbox.send(ask())
    mailbox.cancel("acme", request_id, "el cliente se fue")

    assert mailbox.poll("acme", "personal") == []
    assert mailbox.get("acme", request_id).state == MailboxState.CANCELED


# -- isolation between organisations -------------------------------------------------

def test_one_organisation_cannot_read_anothers_request(mailbox):
    """A2A's rule, kept: out of scope reads as absent, never as forbidden.

    "Forbidden" confirms the id exists, which is exactly what a prober wants.
    """
    request_id = mailbox.send(ask(org="acme"))

    assert mailbox.get("globex", request_id) is None
    assert mailbox.poll("globex", "personal") == []
    assert mailbox.list("globex") == []


def test_one_organisation_cannot_answer_anothers_request(mailbox):
    request_id = mailbox.send(ask(org="acme"))

    mailbox.answer("globex", request_id, "respuesta inyectada", answered_by="atacante")

    assert mailbox.get("acme", request_id).state == MailboxState.PENDING
    assert mailbox.get("acme", request_id).answer == ""


def test_a_message_with_no_org_is_refused(mailbox):
    """Defaulting an empty org would make the request readable by everyone, and the
    failure would surface as a leak rather than as an error."""
    with pytest.raises(ValueError, match="org"):
        mailbox.send(MailboxMessage(org="", sender="sales", recipient="personal", subject="x"))


def test_a_message_with_no_recipient_is_refused(mailbox):
    with pytest.raises(ValueError, match="recipient"):
        mailbox.send(MailboxMessage(org="acme", sender="sales", recipient="", subject="x"))


# -- reading it back, for a front end ------------------------------------------------

def test_listing_filters_and_returns_newest_first(mailbox):
    mailbox.send(ask(subject="uno", context_id="c1"))
    second = mailbox.send(ask(subject="dos", context_id="c2"))
    mailbox.answer("acme", second, "listo")

    assert [m.subject for m in mailbox.list("acme")] == ["dos", "uno"]
    assert [m.subject for m in mailbox.list("acme", state=MailboxState.COMPLETED)] == ["dos"]
    assert [m.subject for m in mailbox.list("acme", context_id="c1")] == ["uno"]
    assert mailbox.list("acme", recipient="nadie") == []


def test_answers_can_be_collected_incrementally(mailbox):
    """A sales agent asks "what is new since I last looked" on each turn, rather than
    re-reading everything it has ever asked."""
    first = mailbox.send(ask(subject="uno"))
    mailbox.answer("acme", first, "respuesta uno")
    watermark = mailbox.get("acme", first).updated_at

    second = mailbox.send(ask(subject="dos"))
    mailbox.answer("acme", second, "respuesta dos")

    fresh = mailbox.answers_for("acme", "sales", since=watermark)
    assert [m.subject for m in fresh] == ["dos"]


# -- durability, the whole point -----------------------------------------------------

def test_an_escalation_survives_a_process_restart():
    """The property nothing in upstream Hermes has.

    Upstream's approval waits are a ``threading.Event`` over a process-global dict, and
    A2A's task store is an ``OrderedDict``: restart while a person is thinking and the
    question is gone. Here the store is reopened from scratch -- a new object, a new
    connection, nothing shared but the file -- mid-escalation, and the request is still
    waiting with everything it needs to be answered and routed back.
    """
    path = tempfile.mktemp(suffix=".db")

    before = SqliteMailbox(path)
    request_id = before.send(ask(context_id="whatsapp:+54911"))
    before.poll("acme", "personal")
    before.mark_input_required("acme", request_id, asked="¿Cubre daño por agua?")
    before.close()

    # --- the process dies here ---

    after = SqliteMailbox(path)
    try:
        waiting = after.list("acme", state=MailboxState.INPUT_REQUIRED)
        assert len(waiting) == 1
        assert waiting[0].asked == "¿Cubre daño por agua?"
        assert waiting[0].context_id == "whatsapp:+54911", "no way to route the answer back"

        after.answer("acme", request_id, "Sí, hasta 6 meses.", answered_by="owner")
        back = after.answers_for("acme", "sales", context_id="whatsapp:+54911")
        assert back[0].answer == "Sí, hasta 6 meses."
    finally:
        after.close()


def test_a_host_can_implement_the_protocol_without_inheriting_anything():
    """Structural typing, same as ``SessionStore``: Postgres, Redis or a queue is an
    implementation of the protocol, not a subclass of ours."""
    from hermes_core.seams.mailbox import AgentMailbox

    assert isinstance(InMemoryMailbox(), AgentMailbox)
