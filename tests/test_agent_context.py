"""Several agents, several tenants, one process.

Before ``AgentContext`` the four seams were ports without injection: each one lived in
a module-level global, so a process served exactly one configuration and the second
tenant to configure itself silently replaced the first -- API key included. These tests
pin the isolation, and equally pin that the single-agent case did not change.

The distinction worth keeping in mind while reading: a context is an *isolation*
boundary, not a *security* boundary. It stops tenants tripping over each other. It does
not contain code that goes looking.
"""

import tempfile
import threading

import pytest

from hermes_core.seams.config import DictConfigSource, load_config, set_config_source
from hermes_core.seams.context import AgentContext, activate, get_active_context
from hermes_core.seams.credentials import (
    StaticCredentials,
    resolve_credentials,
    set_credential_source,
)
from hermes_core.seams.paths import DirectoryWorkspace, get_hermes_home, set_workspace
from hermes_core.seams.session_store import InMemorySessionStore
from hermes_core.testing import Script, install_fake_client


def make_org(name: str, model: str, key: str, *, store=None) -> AgentContext:
    return AgentContext(
        workspace=DirectoryWorkspace(tempfile.mkdtemp(prefix=f"{name}-")),
        config=DictConfigSource({"model": {"default": model, "provider": "openai"}}),
        credentials=StaticCredentials(key),
        session_store=store,
        name=name,
    )


def snapshot():
    """What the seams report right now -- the three things a tenant must not share."""
    return (
        load_config().get("model", {}).get("default"),
        get_hermes_home().name.split("-")[0],
        str(resolve_credentials("openai").api_key)[-1:],
    )


@pytest.fixture(autouse=True)
def process_defaults():
    """A known process-wide default, so "fell back to the default" is distinguishable
    from "leaked from another tenant"."""
    set_workspace(DirectoryWorkspace(tempfile.mkdtemp(prefix="default-")))
    set_config_source(DictConfigSource({"model": {"default": "modelo-default"}}))
    set_credential_source(StaticCredentials("key-D"))
    yield


# -- isolation -----------------------------------------------------------------------

def test_two_tenants_do_not_see_each_others_settings():
    """The failure this whole module exists to prevent.

    With plain globals the second `set_credential_source` handed tenant B's API key to
    tenant A. Keys are checked by last character rather than by value so a failure
    message never contains a secret.
    """
    acme = make_org("acme", "modelo-A", "key-A")
    globex = make_org("globex", "modelo-B", "key-B")

    with acme.activate():
        assert snapshot() == ("modelo-A", "acme", "A")
    with globex.activate():
        assert snapshot() == ("modelo-B", "globex", "B")
    with acme.activate():
        assert snapshot() == ("modelo-A", "acme", "A"), "globex overwrote acme"


def test_contexts_nest_and_restore():
    acme = make_org("acme", "modelo-A", "key-A")
    globex = make_org("globex", "modelo-B", "key-B")

    with acme.activate():
        with globex.activate():
            assert snapshot() == ("modelo-B", "globex", "B")
        assert snapshot() == ("modelo-A", "acme", "A")


def test_a_raising_block_still_restores_the_previous_context():
    """A failed turn must not leave another tenant's credentials bound.

    This is the one that would be found in production rather than in a test: the
    exception surfaces somewhere unrelated, and the tenant that runs next is the one
    that gets hurt.
    """
    acme = make_org("acme", "modelo-A", "key-A")
    globex = make_org("globex", "modelo-B", "key-B")

    with acme.activate():
        with pytest.raises(RuntimeError):
            with globex.activate():
                raise RuntimeError("turn blew up")
        assert snapshot() == ("modelo-A", "acme", "A")

    assert get_active_context() is None


def test_concurrent_threads_never_see_each_others_context():
    """Two tenants served at once -- the actual deployment shape.

    Each thread reads repeatedly so an interleaving has room to go wrong; a single
    read each would pass even with globals if the timing were kind.
    """
    acme = make_org("acme", "modelo-A", "key-A")
    globex = make_org("globex", "modelo-B", "key-B")
    leaks = []

    def work(context, expected):
        with context.activate():
            for _ in range(50):
                seen = snapshot()
                if seen != expected:
                    leaks.append((context.name, seen))

    threads = [
        threading.Thread(target=work, args=(acme, ("modelo-A", "acme", "A"))),
        threading.Thread(target=work, args=(globex, ("modelo-B", "globex", "B"))),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert leaks == []


def test_a_new_thread_started_inside_a_context_does_not_inherit_it():
    """Documented, because it surprises people.

    ``ContextVar`` values are copied into an asyncio task but NOT into a bare
    ``threading.Thread``: a thread starts with a fresh context and sees the process
    defaults. A host that hands work to a thread pool must re-activate inside the
    worker, and this pins that so nobody discovers it in production.
    """
    acme = make_org("acme", "modelo-A", "key-A")
    seen = []

    with acme.activate():
        thread = threading.Thread(target=lambda: seen.append(snapshot()))
        thread.start()
        thread.join()

    assert seen[0][0] == "modelo-default"


# -- backwards compatibility ---------------------------------------------------------

def test_without_any_context_the_process_defaults_still_apply():
    """The single-agent case, unchanged. Everything already written keeps working."""
    assert get_active_context() is None
    assert snapshot() == ("modelo-default", "default", "D")


def test_activate_none_detaches_from_the_surrounding_context():
    """For deployment-level work that belongs to no tenant."""
    acme = make_org("acme", "modelo-A", "key-A")

    with acme.activate():
        with activate(None):
            assert snapshot() == ("modelo-default", "default", "D")
        assert snapshot() == ("modelo-A", "acme", "A")


# -- one organisation, several agents ------------------------------------------------

def test_derive_shares_the_organisation_but_separates_what_should_differ():
    """A sales agent and the owner's personal agent inside one organisation.

    They must share credentials and configuration -- one tenant, one bill, one set of
    keys -- and must not share conversation history.
    """
    sales_store, personal_store = InMemorySessionStore(), InMemorySessionStore()
    sales = make_org("acme-sales", "modelo-A", "key-A", store=sales_store)
    personal = sales.derive(session_store=personal_store, name="acme-personal")

    assert personal.credentials is sales.credentials
    assert personal.config is sales.config
    assert personal.workspace is sales.workspace
    assert personal.session_store is not sales.session_store
    assert (sales.name, personal.name) == ("acme-sales", "acme-personal")


def test_a_context_is_frozen():
    """Identity must not change under an agent mid-turn; `derive` makes a copy."""
    acme = make_org("acme", "modelo-A", "key-A")
    with pytest.raises(Exception):
        acme.name = "otro"


# -- a real turn ---------------------------------------------------------------------

def test_two_agents_run_real_turns_under_different_contexts():
    """End to end: it is not enough that the seams report the right values in
    isolation -- an actual `AIAgent` has to pick them up while running.
    """
    from hermes_core.run_agent import AIAgent

    def build():
        return AIAgent(
            api_key=resolve_credentials("openai").api_key,
            base_url="https://example.invalid/v1",
            provider="openai",
            model=load_config()["model"]["default"],
            enabled_toolsets=[],
            quiet_mode=True,
            max_iterations=3,
        )

    acme = make_org("acme", "modelo-A", "key-A")
    globex = make_org("globex", "modelo-B", "key-B")

    with acme.activate():
        agent_a = build()
        client_a = install_fake_client(agent_a, Script().text("respuesta de acme"))
        result_a = agent_a.run_conversation("hola")

    with globex.activate():
        agent_b = build()
        client_b = install_fake_client(agent_b, Script().text("respuesta de globex"))
        result_b = agent_b.run_conversation("hola")

    assert result_a["final_response"] == "respuesta de acme"
    assert result_b["final_response"] == "respuesta de globex"
    assert client_a.last_request.model == "modelo-A"
    assert client_b.last_request.model == "modelo-B"
