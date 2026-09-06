"""Tests for the configuration seam.

This seam carries 176 of the extracted core's call sites. Its whole value is that
those call sites keep working unedited, so the tests are mostly about preserving
upstream's exact semantics -- including the awkward ones -- rather than about
designing something better.
"""

import pytest

from hermes_core.seams import config as cfg
from hermes_core.seams.config import (
    ConfigSource,
    _parse_env_value,
    is_managed,
    split_model_config_default,
    DictConfigSource,
    cfg_get,
    get_config_source,
    load_config,
    load_config_readonly,
    read_raw_config,
    set_config_source,
)


@pytest.fixture(autouse=True)
def restore_source():
    """Keep the process-wide source from leaking between tests."""
    original = get_config_source()
    yield
    set_config_source(original)


# -- cfg_get: upstream semantics, copied deliberately -----------------------------

def test_cfg_get_traverses_nested_keys():
    assert cfg_get({"model": {"default": "opus"}}, "model", "default") == "opus"


def test_cfg_get_returns_default_on_a_missing_key():
    assert cfg_get({"model": {}}, "model", "default", default="fallback") == "fallback"


def test_cfg_get_returns_an_explicitly_stored_none():
    """`default` applies only when the key is absent -- `dict.get` semantics.

    A stored `None` is a real configured value (a deliberately disabled setting) and
    must not be silently replaced by the default.
    """
    assert cfg_get({"model": {"default": None}}, "model", "default", default="x") is None


def test_cfg_get_tolerates_a_non_dict_anywhere_on_the_path():
    assert cfg_get({"model": "not-a-dict"}, "model", "default", default="d") == "d"
    assert cfg_get(None, "model", default="d") == "d"
    assert cfg_get("not-a-dict", "model", default="d") == "d"


def test_cfg_get_with_no_keys_returns_the_mapping():
    mapping = {"a": 1}
    assert cfg_get(mapping) == mapping


# -- the source protocol ----------------------------------------------------------

def test_the_default_source_is_an_empty_dict():
    """A core that has not been configured reads empty, never raises."""
    set_config_source(DictConfigSource())
    assert load_config() == {}
    assert cfg_get(load_config(), "anything", default="d") == "d"


def test_an_installed_source_answers_the_four_names():
    set_config_source(DictConfigSource({"model": {"default": "sonnet"}}))

    assert load_config()["model"]["default"] == "sonnet"
    assert load_config_readonly()["model"]["default"] == "sonnet"
    assert read_raw_config()["model"]["default"] == "sonnet"


def test_load_config_hands_out_copies():
    """Upstream deepcopies because most call sites mutate the result."""
    set_config_source(DictConfigSource({"model": {"default": "sonnet"}}))

    first = load_config()
    first["model"]["default"] = "mutated"

    assert load_config()["model"]["default"] == "sonnet"


def test_load_config_readonly_does_not_copy():
    """The documented trade: no copy, so callers must not mutate.

    Asserting the aliasing keeps the contract honest -- if this ever starts copying,
    the performance reason the separate function exists has quietly disappeared.
    """
    set_config_source(DictConfigSource({"model": {}}))
    assert load_config_readonly() is load_config_readonly()


def test_raw_is_independent_of_the_merged_view():
    source = DictConfigSource({"model": {"default": "merged"}}, raw={"model": {}})
    set_config_source(source)

    assert load_config()["model"]["default"] == "merged"
    assert read_raw_config()["model"] == {}


def test_raw_is_also_copied():
    set_config_source(DictConfigSource({"a": {"b": 1}}))
    read_raw_config()["a"]["b"] = 99
    assert read_raw_config()["a"]["b"] == 1


# -- environment expansion --------------------------------------------------------

def test_env_references_are_expanded(monkeypatch):
    monkeypatch.setenv("SOME_API_KEY", "secret-value")
    set_config_source(DictConfigSource({"providers": {"key": "${SOME_API_KEY}"}}))

    assert load_config()["providers"]["key"] == "secret-value"


def test_expansion_reaches_into_lists_and_nesting(monkeypatch):
    monkeypatch.setenv("HOST", "example.com")
    set_config_source(DictConfigSource({"urls": [{"base": "https://${HOST}/v1"}]}))

    assert load_config()["urls"][0]["base"] == "https://example.com/v1"


def test_an_unset_variable_is_left_as_written(monkeypatch):
    """A missing credential must look missing.

    Blanking it would produce an empty string, which reads as "not configured" and
    sends the reader hunting through config instead of the environment.
    """
    monkeypatch.delenv("DEFINITELY_UNSET_VAR", raising=False)
    set_config_source(DictConfigSource({"key": "${DEFINITELY_UNSET_VAR}"}))

    assert load_config()["key"] == "${DEFINITELY_UNSET_VAR}"


def test_expansion_can_be_switched_off():
    set_config_source(DictConfigSource({"key": "${LITERAL}"}, expand_env=False))
    assert load_config()["key"] == "${LITERAL}"


def test_non_string_leaves_survive_expansion():
    set_config_source(DictConfigSource({"n": 1, "b": True, "z": None, "f": 1.5}))
    loaded = load_config()
    assert (loaded["n"], loaded["b"], loaded["z"], loaded["f"]) == (1, True, None, 1.5)


# -- swapping the source ----------------------------------------------------------

def test_a_host_can_supply_its_own_source():
    """The point of the seam: configuration need not come from a dict or a file."""

    class CountingSource:
        def __init__(self):
            self.reads = 0

        def load(self):
            self.reads += 1
            return {"model": {"default": "from-host"}}

        def load_readonly(self):
            return self.load()

        def raw(self):
            return {}

    source = CountingSource()
    assert isinstance(source, ConfigSource)  # structural, no inheritance required

    set_config_source(source)
    assert load_config()["model"]["default"] == "from-host"
    assert source.reads == 1


def test_swapping_the_source_takes_effect_immediately():
    set_config_source(DictConfigSource({"v": 1}))
    assert load_config()["v"] == 1

    set_config_source(DictConfigSource({"v": 2}))
    assert load_config()["v"] == 2


def test_the_module_level_names_match_upstream():
    """The four names the extracted call sites import, present and callable.

    If one is renamed, 176 call sites need edits instead of an import rewrite -- so
    this test is guarding the extraction strategy itself.
    """
    for name in ("load_config", "load_config_readonly", "read_raw_config", "cfg_get"):
        assert callable(getattr(cfg, name)), name


# -- defaults, merged the way upstream merges them --------------------------------

def test_defaults_fill_in_what_the_host_left_out():
    set_config_source(
        DictConfigSource({"model": {"default": "sonnet"}}, defaults={"terminal": {"backend": "local"}})
    )

    loaded = load_config()
    assert loaded["model"]["default"] == "sonnet"
    assert loaded["terminal"]["backend"] == "local"


def test_overriding_one_leaf_keeps_its_siblings():
    """The reason the merge recurses instead of replacing whole sections."""
    set_config_source(
        DictConfigSource(
            {"terminal": {"backend": "docker"}},
            defaults={"terminal": {"backend": "local", "timeout": 30}},
        )
    )

    terminal = load_config()["terminal"]
    assert terminal["backend"] == "docker"
    assert terminal["timeout"] == 30


def test_an_empty_section_does_not_blank_its_defaults():
    """`terminal:` with no value parses as None in YAML.

    Treating that as an override would replace the whole default section with None
    and break every reader expecting a mapping.
    """
    set_config_source(
        DictConfigSource({"terminal": None}, defaults={"terminal": {"backend": "local"}})
    )

    assert load_config()["terminal"] == {"backend": "local"}


def test_raw_shows_what_the_host_passed_not_the_defaults():
    set_config_source(DictConfigSource({"model": {}}, defaults={"terminal": {"backend": "local"}}))

    assert "terminal" not in read_raw_config()


# -- model key canonicalisation ----------------------------------------------------

def test_api_base_is_aliased_to_base_url():
    """`api_base` is the name OpenAI-SDK users reach for, and the runtime never reads it.

    Upstream accepted it, confirmed it, then silently ignored it -- requests quietly
    fell back to the default provider.
    """
    set_config_source(DictConfigSource({"model": {"api_base": "https://example/v1"}}))

    assert load_config()["model"]["base_url"] == "https://example/v1"


def test_an_explicit_base_url_is_not_overridden_by_the_alias():
    set_config_source(
        DictConfigSource({"model": {"base_url": "https://real/v1", "api_base": "https://alias/v1"}})
    )

    assert load_config()["model"]["base_url"] == "https://real/v1"


def test_a_dict_valued_model_id_is_flattened():
    """A custom-provider config can nest the id; no reader should meet a dict there."""
    set_config_source(DictConfigSource({"model": {"default": {"model": "glm-5", "provider": "z"}}}))

    assert load_config()["model"]["default"] == "glm-5"


def test_split_model_config_default_separates_model_and_provider():
    assert split_model_config_default({"model": "glm-5", "provider": "zai"}) == ("glm-5", "zai")
    assert split_model_config_default("sonnet") == ("sonnet", "")
    assert split_model_config_default(None) == ("", "")


# -- environment references --------------------------------------------------------

def test_the_cursor_style_env_reference_works(monkeypatch):
    monkeypatch.setenv("TEST_CURSOR_STYLE", "resolved")
    set_config_source(DictConfigSource({"k": "${env:TEST_CURSOR_STYLE}"}))

    assert load_config()["k"] == "resolved"


def test_a_reference_inside_a_longer_string_is_expanded(monkeypatch):
    monkeypatch.setenv("TEST_HOST", "example.com")
    set_config_source(DictConfigSource({"url": "https://${TEST_HOST}/v1"}))

    assert load_config()["url"] == "https://example.com/v1"


# -- .env parsing ------------------------------------------------------------------

def test_env_values_are_unquoted():
    assert _parse_env_value('"sk-quoted"') == "sk-quoted"
    assert _parse_env_value("'sk-single'") == "sk-single"
    assert _parse_env_value("  sk-bare  ") == "sk-bare"


def test_escapes_inside_a_double_quoted_value_are_reversed():
    assert _parse_env_value(r'"a\"b"') == 'a"b'
    assert _parse_env_value(r'"a\\b"') == "a\\b"


def test_a_lone_backslash_stays_literal():
    """An unquoted Windows path must survive with its separators intact."""
    assert _parse_env_value(r'"C:\Users\test"') == r"C:\Users\test"


def test_managed_mode_is_off_for_a_library():
    """Upstream's update advice targets its own installer; it never applies here."""
    assert is_managed() is False
