"""Building an agent against every host that gets its own default headers.

The bug these exist for: ``AIAgent(base_url="https://openrouter.ai/api/v1")`` raised
``AttributeError: module 'hermes_core.seams.auxiliary_client' has no attribute
'build_or_headers'`` while 525 other tests passed. OpenRouter is probably the single most
common destination for anyone embedding this core, and it did not start.

Five hosts were affected, from two separate causes:

* Three header builders (``build_or_headers``, ``build_nvidia_nim_headers``,
  ``_AI_GATEWAY_HEADERS``) were simply missing from the seam that replaces upstream's
  ``agent/auxiliary_client.py``. The seam had been sized by its main caller -- context
  compression -- and these are called by client construction, which nobody checked.
* ``tools/xai_http.py`` was missing from the lift manifest, which cost twice: the module
  was absent, *and* the lift's string-literal rewriter refuses to rewrite a dotted path
  whose module is not in the manifest, so the table's ``"tools.xai_http"`` also stayed
  pointing at the upstream path.

What let all five hide is the same thing: each table entry is gated on a base-URL host
match, so nothing resolves until an agent is pointed at that vendor, and no other test
passes one of these URLs. The lift's import check proved every module imports -- true,
and beside the point.

So the guard has to be construction against the real URLs, one case per host, which is
what the parametrisation below is. A single test covering one host would have shipped
four of the five failures.
"""

import tempfile

import pytest

from hermes_core.seams.config import DictConfigSource, set_config_source
from hermes_core.seams.credentials import StaticCredentials, set_credential_source
from hermes_core.seams.paths import DirectoryWorkspace, set_workspace


@pytest.fixture(autouse=True)
def isolated_core():
    set_workspace(DirectoryWorkspace(tempfile.mkdtemp(prefix="hosts-")))
    set_config_source(DictConfigSource({"model": {"default": "fake-model", "provider": "openai"}}))
    set_credential_source(StaticCredentials("sk-test"))
    yield


def header_table_hosts():
    """Every host in both tables, read from the tables themselves.

    Deliberately not a hand-written list: a hard-coded copy would go stale the moment
    upstream adds a vendor, which is the same way this gap opened. Reading the table
    means a new entry is covered the day it is lifted.
    """
    from hermes_core.agent.agent_init import _HOST_DEFAULT_HEADERS
    from hermes_core.agent.client_lifecycle import _ROUTE_DEFAULT_HEADERS

    return sorted({host for host, _ in _HOST_DEFAULT_HEADERS} | {host for host, _ in _ROUTE_DEFAULT_HEADERS})


@pytest.mark.parametrize("host", header_table_hosts())
def test_an_agent_builds_against_every_host_with_default_headers(host):
    """Construction only -- no request, no network. That is enough: the header factory
    runs during ``__init__``, which is exactly where this failed."""
    from hermes_core.run_agent import AIAgent

    agent = AIAgent(
        api_key="sk-test",
        base_url=f"https://{host}/v1",
        provider="openai",
        model="fake-model",
        quiet_mode=True,
    )

    assert agent.base_url == f"https://{host}/v1"


def test_both_tables_are_covered():
    """A guard on the guard: if either table were empty or unreadable, the
    parametrisation above would silently test nothing and still be green."""
    hosts = header_table_hosts()

    assert len(hosts) >= 8, f"expected the full vendor table, got {hosts}"
    assert "openrouter.ai" in hosts


@pytest.mark.parametrize("host", header_table_hosts())
def test_every_header_factory_returns_a_dict_of_strings(host):
    """Resolving the factory is not enough -- what it returns goes on the wire.

    A factory that resolved but returned ``None`` or a non-string value would fail later,
    inside the HTTP client, where the traceback no longer names the vendor.
    """
    from hermes_core.agent.agent_init import _host_default_headers_factory

    factory = _host_default_headers_factory(f"https://{host}/v1")
    if factory is None:
        pytest.skip(f"{host} is only in the client_lifecycle table")

    headers = factory("sk-test", f"https://{host}/v1")

    assert isinstance(headers, dict)
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items()), headers


# -- the specific builders, since their content is what a vendor's dashboard reads ----

def test_openrouter_headers_carry_the_attribution_its_dashboard_reads():
    from hermes_core.seams.auxiliary_client import build_or_headers

    headers = build_or_headers({})

    assert headers["X-Title"] == "Hermes Agent"
    assert headers["HTTP-Referer"].startswith("https://")
    assert "X-OpenRouter-Cache" not in headers  # off unless the host asks for it


def test_openrouter_response_cache_is_opt_in_through_config():
    from hermes_core.seams.auxiliary_client import build_or_headers

    headers = build_or_headers({"response_cache": True, "response_cache_ttl": 600})

    assert headers["X-OpenRouter-Cache"] == "true"
    assert headers["X-OpenRouter-Cache-TTL"] == "600"


def test_an_out_of_range_cache_ttl_is_dropped_rather_than_sent():
    """Upstream bounds it to 1-86400 s. Sending a nonsense TTL would have the vendor
    reject the request, which is worse than sending no TTL at all."""
    from hermes_core.seams.auxiliary_client import build_or_headers

    headers = build_or_headers({"response_cache": True, "response_cache_ttl": 999_999})

    assert headers["X-OpenRouter-Cache"] == "true"
    assert "X-OpenRouter-Cache-TTL" not in headers


def test_the_environment_overrides_configuration_for_the_cache(monkeypatch):
    from hermes_core.seams.auxiliary_client import build_or_headers

    monkeypatch.setenv("HERMES_OPENROUTER_CACHE", "1")
    monkeypatch.setenv("HERMES_OPENROUTER_CACHE_TTL", "120")

    headers = build_or_headers({"response_cache": False})

    assert headers["X-OpenRouter-Cache"] == "true"
    assert headers["X-OpenRouter-Cache-TTL"] == "120"


def test_nvidia_attribution_is_gated_on_the_cloud_host():
    """``NVIDIA_BASE_URL`` may point at a locally hosted NIM, which must not be told it
    is cloud traffic."""
    from hermes_core.seams.auxiliary_client import build_nvidia_nim_headers

    assert build_nvidia_nim_headers("https://integrate.api.nvidia.com/v1")
    assert build_nvidia_nim_headers("http://localhost:8000/v1") == {}
    assert build_nvidia_nim_headers(None) == {}


def test_the_ai_gateway_headers_report_this_cores_version():
    from hermes_core import __version__
    from hermes_core.seams.auxiliary_client import _AI_GATEWAY_HEADERS

    assert _AI_GATEWAY_HEADERS["User-Agent"] == f"HermesAgent/{__version__}"


def test_the_xai_module_is_reachable_by_the_path_the_table_names():
    """The second cause: one missing manifest entry left the module absent *and* left the
    table's ``"tools.xai_http"`` unrewritten, because the lift will not rewrite a dotted
    path whose module it does not know. Importing by the exact path the table names is
    what proves both halves are fixed."""
    import importlib
    import inspect

    from hermes_core.agent import client_lifecycle

    # The table holds lambdas, so its repr says nothing -- read the path out of the
    # source, which is where the stale string would still be.
    source = inspect.getsource(client_lifecycle)
    assert '"tools.xai_http"' not in source, "the upstream module path was left unrewritten"

    module = importlib.import_module("hermes_core.tools.xai_http")

    assert module.hermes_xai_default_headers()["User-Agent"].startswith("Hermes-Agent/")
