"""Tests for the credential-reading chain: ``.env`` parsing, the two lookup
precedences, and the seam-to-``auth.py`` integration that broke silently.

This module exists because of a real incident: ``get_env_value_prefer_dotenv`` did
not exist, so every api-key provider's credential lookup raised ``ImportError`` at
call time (``hermes_core/runtime/auth.py``), a bare ``except Exception`` several
layers up swallowed it into ``(None, None)`` at DEBUG level, a fallback stub then
returned a 2-tuple where the caller unpacked three, and the resulting ``ValueError``
was caught by an ``except ValueError: raise`` written for something unrelated. The
function now exists (``hermes_core/seams/config.py``) but nothing had ever executed
it. These tests do.

``tests/test_config_seam.py`` already covers ``_parse_env_value``'s quoting rules in
isolation and ``${VAR}``/``${env:VAR}`` expansion in configuration (set/unset/nested/
``expand_env=False``). This file does not repeat those -- it covers ``load_env``'s
full file-parsing pipeline (which runs code ``_parse_env_value`` alone does not:
comment-stripping, BOM/CRLF/encoding handling, line filtering), the two precedence
functions, the real ``auth.py`` call path, per-workspace/per-tenant resolution, and a
couple of expansion edge cases the other file does not exercise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_core.seams import config as cfg
from hermes_core.seams.config import (
    DictConfigSource,
    get_env_path,
    get_env_value,
    get_env_value_prefer_dotenv,
    load_env,
)
from hermes_core.seams.context import AgentContext
from hermes_core.seams.credentials import StaticCredentials
from hermes_core.seams.paths import DirectoryWorkspace, get_workspace, set_workspace


# -- fixtures -----------------------------------------------------------------------

@pytest.fixture(autouse=True)
def restore_workspace():
    """Keep the process-wide workspace from leaking into other test files."""
    original = get_workspace()
    yield
    set_workspace(original)


@pytest.fixture
def workspace_dir(tmp_path):
    """Point the workspace at an isolated temp directory; hand back its path."""
    set_workspace(DirectoryWorkspace(tmp_path))
    return tmp_path


def _write_env(directory: Path, content: str, *, encoding: str = "utf-8") -> None:
    (directory / ".env").write_text(content, encoding=encoding)


def _write_env_bytes(directory: Path, data: bytes) -> None:
    (directory / ".env").write_bytes(data)


# == 1. .env parsing (load_env) ======================================================

def test_plain_key_value_is_parsed(workspace_dir):
    _write_env(workspace_dir, "API_KEY=sk-plain-value\n")
    assert load_env() == {"API_KEY": "sk-plain-value"}


def test_single_quoted_value_is_taken_literally(workspace_dir):
    _write_env(workspace_dir, "API_KEY='sk-single $NOT_EXPANDED'\n")
    assert load_env()["API_KEY"] == "sk-single $NOT_EXPANDED"


def test_double_quoted_value_unescapes_quote_and_backslash(workspace_dir):
    _write_env(workspace_dir, r'KEY1="a\"b"' + "\n" + r'KEY2="a\\b"' + "\n")
    result = load_env()
    assert result["KEY1"] == 'a"b'
    assert result["KEY2"] == "a\\b"


def test_a_value_containing_an_equals_sign_is_preserved(workspace_dir):
    _write_env(workspace_dir, "URL=https://example.com?a=1&b=2\n")
    assert load_env()["URL"] == "https://example.com?a=1&b=2"


def test_inline_comment_is_stripped_only_when_preceded_by_whitespace(workspace_dir):
    _write_env(workspace_dir, "API_KEY=sk-real-value # trailing comment\n")
    assert load_env()["API_KEY"] == "sk-real-value"


def test_a_hash_with_no_preceding_whitespace_is_not_a_comment(workspace_dir):
    """An unquoted value can itself contain ``#`` -- it must survive intact."""
    _write_env(workspace_dir, "API_KEY=sk-ab#cd\n")
    assert load_env()["API_KEY"] == "sk-ab#cd"


def test_a_hash_inside_quotes_with_no_preceding_space_survives(workspace_dir):
    _write_env(workspace_dir, 'API_KEY="sk-ab#cd"\n')
    assert load_env()["API_KEY"] == "sk-ab#cd"


def test_quoting_protects_a_value_containing_a_space_then_a_hash(workspace_dir):
    """Quotes must protect the whole value, not just the no-internal-space case.

    `load_env()` strips inline comments before unquoting, so a quoted value containing
    a space followed by `#` used to be cut at the space -- losing the closing quote and
    everything after it. `API_KEY="sk-1234 #5678"` came back as `'"sk-1234'`: truncated,
    carrying a stray quote, with no error anywhere. The provider simply rejects a key
    that looks almost right. `_strip_inline_comment` now skips a quoted value as a unit.
    """
    _write_env(workspace_dir, 'API_KEY="sk-1234 #5678"\n')
    assert load_env()["API_KEY"] == "sk-1234 #5678"


def test_blank_lines_and_comment_only_lines_are_ignored(workspace_dir):
    _write_env(workspace_dir, "# a full comment line\n\n   \nKEY=value\n# another\n")
    assert load_env() == {"KEY": "value"}


def test_utf8_bom_does_not_corrupt_the_first_key(workspace_dir):
    """``utf-8-sig`` must strip the BOM, not leave it glued onto the first key name."""
    _write_env_bytes(workspace_dir, "KEY1=value1\nKEY2=value2\n".encode("utf-8-sig"))
    result = load_env()
    assert result == {"KEY1": "value1", "KEY2": "value2"}
    assert "﻿KEY1" not in result


def test_crlf_line_endings_are_handled(workspace_dir):
    _write_env_bytes(workspace_dir, b"KEYA=vala\r\nKEYB=valb\r\n")
    result = load_env()
    assert result == {"KEYA": "vala", "KEYB": "valb"}
    assert "\r" not in result["KEYB"]


def test_leading_and_trailing_whitespace_around_key_and_value_is_stripped(workspace_dir):
    _write_env(workspace_dir, "   KEY   =   value with spaces   \n")
    assert load_env() == {"KEY": "value with spaces"}


def test_export_prefix_is_stripped(workspace_dir):
    _write_env(workspace_dir, "export KEY=exported_value\n")
    assert load_env() == {"KEY": "exported_value"}


def test_a_missing_file_returns_an_empty_dict(workspace_dir):
    # No .env written at all.
    assert load_env() == {}


def test_an_unreadable_file_returns_an_empty_dict(workspace_dir, monkeypatch):
    """Simulates a permission-denied read without depending on OS-specific ACLs."""
    _write_env(workspace_dir, "KEY=value\n")
    target = Path(get_env_path())
    original_read_text = Path.read_text

    def maybe_raise(self, *args, **kwargs):
        if self == target or str(self) == str(target):
            raise PermissionError("simulated: no read access")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", maybe_raise)
    assert load_env() == {}


def test_invalid_utf8_returns_an_empty_dict(workspace_dir):
    _write_env_bytes(workspace_dir, b"KEY=\x80\x81invalid\n")
    assert load_env() == {}


def test_a_duplicated_key_keeps_the_last_value(workspace_dir):
    _write_env(workspace_dir, "DUPKEY=first\nDUPKEY=second\n")
    assert load_env() == {"DUPKEY": "second"}


# == 2. the two precedences ==========================================================

def test_get_env_value_prefers_the_environment_over_dotenv(workspace_dir, monkeypatch):
    _write_env(workspace_dir, "SHARED_KEY=from-dotenv\n")
    monkeypatch.setenv("SHARED_KEY", "from-environment")

    assert get_env_value("SHARED_KEY") == "from-environment"


def test_get_env_value_prefer_dotenv_prefers_dotenv_over_the_environment(workspace_dir, monkeypatch):
    """The reverse precedence: a deliberate ``.env`` edit beats a stale shell export."""
    _write_env(workspace_dir, "SHARED_KEY=from-dotenv\n")
    monkeypatch.setenv("SHARED_KEY", "from-environment")

    assert get_env_value_prefer_dotenv("SHARED_KEY") == "from-dotenv"


def test_get_env_value_falls_back_to_dotenv_when_the_environment_is_unset(workspace_dir, monkeypatch):
    monkeypatch.delenv("ONLY_IN_DOTENV", raising=False)
    _write_env(workspace_dir, "ONLY_IN_DOTENV=dotenv-value\n")

    assert get_env_value("ONLY_IN_DOTENV") == "dotenv-value"


def test_get_env_value_reads_the_environment_when_dotenv_has_nothing(workspace_dir, monkeypatch):
    # No .env file at all.
    monkeypatch.setenv("ONLY_IN_ENV", "env-value")

    assert get_env_value("ONLY_IN_ENV") == "env-value"


def test_prefer_dotenv_falls_back_to_the_environment_when_dotenv_is_unset(workspace_dir, monkeypatch):
    # No .env file at all.
    monkeypatch.setenv("ONLY_IN_ENV", "env-value")

    assert get_env_value_prefer_dotenv("ONLY_IN_ENV") == "env-value"


def test_prefer_dotenv_reads_dotenv_when_the_environment_has_nothing(workspace_dir, monkeypatch):
    monkeypatch.delenv("ONLY_IN_DOTENV", raising=False)
    _write_env(workspace_dir, "ONLY_IN_DOTENV=dotenv-value\n")

    assert get_env_value_prefer_dotenv("ONLY_IN_DOTENV") == "dotenv-value"


def test_both_functions_return_none_when_neither_source_has_the_key(workspace_dir, monkeypatch):
    monkeypatch.delenv("NOWHERE_AT_ALL", raising=False)
    # No .env file at all.

    assert get_env_value("NOWHERE_AT_ALL") is None
    assert get_env_value_prefer_dotenv("NOWHERE_AT_ALL") is None


# == 3. the integration that actually broke: auth.py's real call path ===============

def test_get_anthropic_key_reads_the_dotenv_file_through_the_real_seam(workspace_dir, monkeypatch):
    """``auth.get_anthropic_key`` imports ``get_env_value_prefer_dotenv`` at call
    time -- exactly the import that was missing. This proves the whole chain, from a
    real workspace ``.env`` file to a returned key, works end to end."""
    from hermes_core.runtime.auth import get_anthropic_key

    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    _write_env(workspace_dir, "ANTHROPIC_API_KEY=sk-ant-fake-test-0000\n")

    assert get_anthropic_key() == "sk-ant-fake-test-0000"


def test_get_anthropic_key_returns_empty_string_with_no_credential_anywhere(workspace_dir, monkeypatch):
    from hermes_core.runtime.auth import get_anthropic_key

    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    # No .env file at all.

    assert get_anthropic_key() == ""


def test_resolve_api_key_provider_secret_reads_dotenv_for_real(workspace_dir, monkeypatch):
    """The exact function named in the incident report, exercised end to end: a
    workspace ``.env`` holding a key must come back through the real seam call, not a
    mock of it -- that mock is precisely what would have hidden the missing import."""
    from hermes_core.runtime.auth import PROVIDER_REGISTRY, _resolve_api_key_provider_secret

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _write_env(workspace_dir, "OPENAI_API_KEY=sk-openai-fake-test-1111\n")

    secret, source = _resolve_api_key_provider_secret("openai-api", PROVIDER_REGISTRY["openai-api"])

    assert (secret, source) == ("sk-openai-fake-test-1111", "OPENAI_API_KEY")


def test_resolve_api_key_provider_secret_prefers_dotenv_over_a_stale_shell_export(workspace_dir, monkeypatch):
    from hermes_core.runtime.auth import PROVIDER_REGISTRY, _resolve_api_key_provider_secret

    _write_env(workspace_dir, "OPENAI_API_KEY=sk-openai-fresh-dotenv\n")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-stale-shell-export")

    secret, source = _resolve_api_key_provider_secret("openai-api", PROVIDER_REGISTRY["openai-api"])

    assert secret == "sk-openai-fresh-dotenv"


def test_resolve_api_key_provider_secret_falls_back_to_env_with_no_dotenv(workspace_dir, monkeypatch):
    from hermes_core.runtime.auth import PROVIDER_REGISTRY, _resolve_api_key_provider_secret

    # No .env file at all.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-from-env-only")

    secret, source = _resolve_api_key_provider_secret("openai-api", PROVIDER_REGISTRY["openai-api"])

    assert (secret, source) == ("sk-openai-from-env-only", "OPENAI_API_KEY")


# == 4. workspace-relative resolution ================================================

def test_env_path_is_under_the_active_workspace_root(workspace_dir):
    assert get_env_path() == workspace_dir / ".env"


def test_set_workspace_redirects_which_env_file_is_read(tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    _write_env(other, "REDIRECTED_KEY=redirected-value\n")

    set_workspace(DirectoryWorkspace(other))

    assert load_env() == {"REDIRECTED_KEY": "redirected-value"}


def test_an_agent_context_redirects_the_env_file_independently_of_the_process_default(tmp_path):
    """The interesting case: an active ``AgentContext`` must win over whatever
    ``set_workspace`` installed at the process level, not just over another
    ``set_workspace`` call."""
    process_default = tmp_path / "process-default"
    process_default.mkdir()
    _write_env(process_default, "WHICH=process-default\n")
    set_workspace(DirectoryWorkspace(process_default))

    tenant_dir = tmp_path / "tenant"
    tenant_dir.mkdir()
    _write_env(tenant_dir, "WHICH=tenant\n")
    context = AgentContext(
        workspace=DirectoryWorkspace(tenant_dir),
        config=DictConfigSource({}),
        credentials=StaticCredentials(None),
    )

    assert load_env()["WHICH"] == "process-default"
    with context.activate():
        assert load_env()["WHICH"] == "tenant"
    assert load_env()["WHICH"] == "process-default"


def test_two_tenant_contexts_read_two_different_env_files(tmp_path):
    """Two tenants must not be able to read each other's credentials via ``.env``."""
    dir_a = tmp_path / "tenant-a"
    dir_b = tmp_path / "tenant-b"
    dir_a.mkdir()
    dir_b.mkdir()
    _write_env(dir_a, "TENANT_KEY=key-for-a\n")
    _write_env(dir_b, "TENANT_KEY=key-for-b\n")

    context_a = AgentContext(
        workspace=DirectoryWorkspace(dir_a), config=DictConfigSource({}), credentials=StaticCredentials(None),
    )
    context_b = AgentContext(
        workspace=DirectoryWorkspace(dir_b), config=DictConfigSource({}), credentials=StaticCredentials(None),
    )

    with context_a.activate():
        value_a = get_env_value_prefer_dotenv("TENANT_KEY")
    with context_b.activate():
        value_b = get_env_value_prefer_dotenv("TENANT_KEY")

    assert (value_a, value_b) == ("key-for-a", "key-for-b")


# == 5. ${VAR} expansion -- edge cases not already covered by test_config_seam.py ===

def test_a_set_but_empty_variable_expands_to_an_empty_string(monkeypatch):
    """Different from an *unset* variable (test_config_seam.py already pins that an
    unset reference is left verbatim). A variable that is explicitly set to "" is a
    real, if unhelpful, value -- os.environ.get returns "" rather than None, and the
    expander only leaves a reference alone when the lookup is None."""
    monkeypatch.setenv("EMPTY_BUT_SET", "")
    source = DictConfigSource({"k": "${EMPTY_BUT_SET}"})

    assert source.load()["k"] == ""


def test_several_references_in_one_string_all_expand(monkeypatch):
    monkeypatch.setenv("PART_A", "aa")
    monkeypatch.setenv("PART_B", "bb")
    source = DictConfigSource({"k": "${PART_A}-${PART_B}"})

    assert source.load()["k"] == "aa-bb"


# == 6. adversarial ===================================================================

def test_a_line_without_an_equals_sign_is_silently_ignored(workspace_dir):
    _write_env(workspace_dir, "JUST_A_WORD_NO_EQUALS\nGOOD_KEY=fine\n")
    assert load_env() == {"GOOD_KEY": "fine"}


def test_a_key_with_a_leading_digit_is_accepted_without_validation(workspace_dir):
    """Surprising: load_env() applies no identifier validation to keys -- unlike
    hermes_core/seams/config.py's own _sanitize_env_lines (used only by the separate
    startup loader in hermes_core/runtime/env_loader.py, never by load_env). A
    malformed key that the startup loader would reject is accepted here."""
    _write_env(workspace_dir, "1KEY=onetwo\n")
    assert load_env() == {"1KEY": "onetwo"}


def test_a_key_containing_a_space_is_accepted_without_validation(workspace_dir):
    _write_env(workspace_dir, "MY KEY=spaced\n")
    assert load_env() == {"MY KEY": "spaced"}


def test_a_one_megabyte_value_round_trips_intact(workspace_dir):
    big_value = "x" * (1024 * 1024)
    _write_env(workspace_dir, f"BIGKEY={big_value}\n")

    result = load_env()

    assert len(result["BIGKEY"]) == len(big_value)
    assert result["BIGKEY"] == big_value


def test_a_directory_at_the_env_path_returns_empty_dict_not_a_crash(workspace_dir):
    (workspace_dir / ".env").mkdir()
    assert load_env() == {}


def test_both_lookups_agree_that_a_whitespace_only_value_is_absent(
    workspace_dir, monkeypatch,
):
    """The two precedences must answer identically for identical input.

    They did not. An empty `.env` entry with nothing in the environment gave `""` from
    `get_env_value` and `None` from `get_env_value_prefer_dotenv`. A caller writing
    `if value:` never noticed; one writing `if value is not None:` got a different
    answer depending on which precedence it happened to need. Empty means "not
    configured" throughout this module, and now both say so.
    """
    monkeypatch.delenv("WHITESPACE_ONLY", raising=False)
    _write_env(workspace_dir, "WHITESPACE_ONLY=   \n")

    assert get_env_value("WHITESPACE_ONLY") is None
    assert get_env_value_prefer_dotenv("WHITESPACE_ONLY") is None


def test_an_empty_dotenv_value_does_not_shadow_a_real_environment_variable(
    workspace_dir, monkeypatch,
):
    """An empty `.env` entry must not beat a real value in the environment.

    `.env` wins in this precedence, so an entry left blank -- a commented-out key, a
    template someone never filled in -- would otherwise mask the variable that is
    actually set and send an empty credential to the provider. Empty means "not
    configured", so the lookup falls through.
    """
    monkeypatch.setenv("HALF_SET", "sk-real-value")
    _write_env(workspace_dir, "HALF_SET=   \n")

    assert get_env_value_prefer_dotenv("HALF_SET") == "sk-real-value"


def test_whitespace_only_is_only_falsy_after_dotenv_parsing_strips_it(
    workspace_dir, monkeypatch,
):
    """Not symmetric with the previous two tests, and worth being explicit about: a
    `.env` value of only whitespace becomes "" (falsy) because `load_env()` strips it
    while parsing. A *process environment* variable holding the same literal
    whitespace is never stripped by either lookup function, so it stays truthy: with
    no `.env` file at all, both functions return it verbatim rather than treating it
    as absent. The "whitespace looks empty" intuition only holds for the `.env` side."""
    monkeypatch.setenv("WHITESPACE_ONLY_ENV", "   ")
    # No .env file at all -- isolates the environment side of the asymmetry.

    assert get_env_value("WHITESPACE_ONLY_ENV") == "   "
    assert get_env_value_prefer_dotenv("WHITESPACE_ONLY_ENV") == "   "
