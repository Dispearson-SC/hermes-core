"""Tests for the credential seam.

The rotating pool exists for the multi-tenant case: many tenants sharing one
provider account will hit a quota, and the answer is more keys plus knowing when to
stop using one. Its failure modes are what matter -- a pool that strands callers, or
that benches keys over faults they did not cause, is worse than no pool.
"""

import pytest

from hermes_core.seams.credentials import (
    Credentials,
    CredentialSource,
    EnvCredentials,
    NoCredentials,
    RotatingKeyPool,
    StaticCredentials,
    get_credential_source,
    label_for,
    resolve_credentials,
    set_credential_source,
)


@pytest.fixture(autouse=True)
def restore_source():
    original = get_credential_source()
    yield
    set_credential_source(original)


# -- secrets must not leak ---------------------------------------------------------

def test_the_key_never_appears_in_a_repr():
    """Credentials end up in tracebacks and log lines; the default repr would print
    the secret in both."""
    credentials = Credentials(api_key="sk-super-secret-value")

    assert "sk-super-secret-value" not in repr(credentials)
    assert "...alue" in repr(credentials)


def test_a_label_identifies_a_key_without_revealing_it():
    assert label_for("sk-abcdefghijkl") == "...ijkl"


def test_a_short_secret_gets_no_label():
    """Four characters of an eight-character secret is most of it."""
    assert label_for("sk-12345") == "<short>"
    assert label_for("") == "<empty>"


# -- the single-key case -----------------------------------------------------------

def test_a_static_key_answers_for_any_provider():
    set_credential_source(StaticCredentials("sk-test", base_url="https://api.example/v1"))

    credentials = resolve_credentials("openai")

    assert credentials.api_key == "sk-test"
    assert credentials.base_url == "https://api.example/v1"


def test_a_missing_key_fails_by_name():
    """The provider is in the message, so the fix is obvious from the error alone."""
    set_credential_source(StaticCredentials(None))

    with pytest.raises(NoCredentials, match="openai"):
        resolve_credentials("openai")


# -- environment-backed ------------------------------------------------------------

def test_environment_variables_are_read_at_resolve_time(monkeypatch):
    """A process that loads its environment late still works."""
    source = EnvCredentials({"openai": "TEST_OPENAI_KEY"})
    set_credential_source(source)
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-from-env")

    assert resolve_credentials("openai").api_key == "sk-from-env"


def test_an_unset_variable_names_the_variable(monkeypatch):
    monkeypatch.delenv("TEST_MISSING_KEY", raising=False)
    set_credential_source(EnvCredentials({"openai": "TEST_MISSING_KEY"}))

    with pytest.raises(NoCredentials, match="TEST_MISSING_KEY"):
        resolve_credentials("openai")


def test_an_unmapped_provider_is_reported():
    set_credential_source(EnvCredentials({"openai": "X"}))

    with pytest.raises(NoCredentials, match="anthropic"):
        resolve_credentials("anthropic")


# -- rotation ----------------------------------------------------------------------

def test_keys_rotate_across_calls():
    pool = RotatingKeyPool({"openai": ["sk-aaaa", "sk-bbbb", "sk-cccc"]})

    handed_out = [pool.resolve("openai").api_key for _ in range(3)]

    assert sorted(handed_out) == ["sk-aaaa", "sk-bbbb", "sk-cccc"]


def test_rotation_is_least_recently_used():
    """Not round-robin: the key just handed out is the last one chosen again.

    Under uneven load -- one busy tenant among many -- a round-robin cursor still
    spreads unevenly, because the cursor advances with calls rather than with time
    since a key was used.
    """
    pool = RotatingKeyPool({"openai": ["sk-aaaa", "sk-bbbb"]})

    first = pool.resolve("openai").api_key
    second = pool.resolve("openai").api_key
    third = pool.resolve("openai").api_key

    assert first != second
    assert third == first


def test_a_rate_limited_key_is_benched():
    pool = RotatingKeyPool({"openai": ["sk-aaaa", "sk-bbbb"]})
    benched = pool.resolve("openai")

    pool.report_failure("openai", benched, rate_limited=True)

    for _ in range(4):
        assert pool.resolve("openai").api_key != benched.api_key


def test_only_rate_limits_bench_a_key():
    """A malformed request or a server error says nothing about the key.

    Benching on any failure shrinks the pool over faults the key did not cause, and
    a bad request repeated across every key would empty it entirely.
    """
    pool = RotatingKeyPool({"openai": ["sk-aaaa", "sk-bbbb"]})
    credentials = pool.resolve("openai")

    pool.report_failure("openai", credentials, rate_limited=False)

    assert pool.cooling_down("openai") == []


def test_an_exhausted_pool_still_answers():
    """Refusing here turns a throttled provider into a hard outage.

    Every key is cooling down, so the one recovering soonest is returned and the
    provider decides -- it may well accept the call.
    """
    pool = RotatingKeyPool({"openai": ["sk-aaaa", "sk-bbbb"]})
    for _ in range(2):
        pool.report_failure("openai", pool.resolve("openai"), rate_limited=True)

    assert len(pool.cooling_down("openai")) == 2
    assert pool.resolve("openai").api_key in ("sk-aaaa", "sk-bbbb")


def test_the_cooldown_expires():
    pool = RotatingKeyPool({"openai": ["sk-aaaa"]}, cooldown_seconds=0.0)
    pool.report_failure("openai", pool.resolve("openai"), rate_limited=True)

    assert pool.cooling_down("openai") == []


def test_cooling_down_reports_labels_not_keys():
    """Diagnostics must not become a place secrets are printed."""
    pool = RotatingKeyPool({"openai": ["sk-aaaa-secret"]})
    pool.report_failure("openai", pool.resolve("openai"), rate_limited=True)

    assert pool.cooling_down("openai") == ["...cret"]


def test_pools_are_per_provider():
    pool = RotatingKeyPool({"openai": ["sk-aaaa"], "anthropic": ["sk-bbbb"]})

    assert pool.resolve("openai").api_key == "sk-aaaa"
    assert pool.resolve("anthropic").api_key == "sk-bbbb"


def test_benching_one_provider_leaves_the_other_alone():
    pool = RotatingKeyPool({"openai": ["sk-aaaa"], "anthropic": ["sk-bbbb"]})

    pool.report_failure("openai", pool.resolve("openai"), rate_limited=True)

    assert pool.cooling_down("anthropic") == []


def test_an_unconfigured_provider_fails_by_name():
    pool = RotatingKeyPool({"openai": ["sk-aaaa"]})

    with pytest.raises(NoCredentials, match="gemini"):
        pool.resolve("gemini")


# -- the protocol ------------------------------------------------------------------

def test_a_host_can_supply_its_own_source():
    """Two methods, no inheritance: a host with its own secret manager implements it."""

    class VaultSource:
        def __init__(self):
            self.failures = []

        def resolve(self, provider):
            return Credentials(api_key=f"from-vault-{provider}")

        def report_failure(self, provider, credentials, *, rate_limited):
            self.failures.append((provider, rate_limited))

    source = VaultSource()
    assert isinstance(source, CredentialSource)

    set_credential_source(source)
    assert resolve_credentials("openai").api_key == "from-vault-openai"


def test_the_shipped_sources_satisfy_the_protocol():
    assert isinstance(StaticCredentials("k"), CredentialSource)
    assert isinstance(EnvCredentials({}), CredentialSource)
    assert isinstance(RotatingKeyPool({}), CredentialSource)
