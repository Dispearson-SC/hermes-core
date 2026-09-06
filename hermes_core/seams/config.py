"""Configuration seam.

Upstream Hermes reads configuration through ``hermes_cli.config`` -- a 3912-line
module bound to ``~/.hermes/config.yaml``, an interactive CLI, and a managed-scope
overlay. The agent core does not need any of that. Measured against the upstream
clone, 176 of the core's imports target that module and 84% of them resolve to just
four names: ``load_config``, ``load_config_readonly``, ``cfg_get`` and
``read_raw_config``. All four answer one question: *give me the configuration dict*.

This module keeps those four names and their exact signatures, so extracted call
sites need only an import-path rewrite -- never an edit -- and makes the source of
the dict injectable. An embedding application supplies a plain dict; nothing here
reads a file, a home directory, or an environment unless it is asked to.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from hermes_core.seams._config_semantics import (
    _deep_merge,
    _normalize_root_model_keys,
    split_model_config_default,
)
from hermes_core.seams import _active
from hermes_core.seams.paths import get_config_path, get_env_path

__all__ = [
    "ConfigSource",
    "DictConfigSource",
    "set_config_source",
    "get_config_source",
    "load_config",
    "load_config_readonly",
    "read_raw_config",
    "cfg_get",
    "split_model_config_default",
    "get_config_path",
    "get_env_path",
    "load_env",
    "get_env_value",
    "get_env_value_prefer_dotenv",
    "is_managed",
    "get_compatible_custom_providers",
    "get_custom_provider_tls_settings",
    "get_custom_provider_context_length",
]


@runtime_checkable
class ConfigSource(Protocol):
    """Where the agent core gets its configuration.

    Implementations must be cheap to call: the core reads configuration on hot
    paths, including inside the turn loop.
    """

    def load(self) -> Dict[str, Any]:
        """The merged, effective configuration. Callers may mutate the result."""
        ...

    def load_readonly(self) -> Dict[str, Any]:
        """Same content as :meth:`load`, without the defensive copy.

        Callers promise not to mutate. Implementations may return a shared object.
        """
        ...

    def raw(self) -> Dict[str, Any]:
        """Configuration exactly as authored, with no defaults merged in."""
        ...


class DictConfigSource:
    """The default source: a dict held in memory.

    ``raw`` defaults to the same mapping as ``values`` -- for an embedding
    application there is usually no distinction between "as authored" and
    "effective", since it authors the dict directly.
    """

    def __init__(
        self,
        values: Optional[Dict[str, Any]] = None,
        *,
        raw: Optional[Dict[str, Any]] = None,
        defaults: Optional[Dict[str, Any]] = None,
        expand_env: bool = True,
    ) -> None:
        self._raw: Dict[str, Any] = copy.deepcopy(raw if raw is not None else (values or {}))

        merged = copy.deepcopy(values or {})
        if defaults:
            # Dict-over-dict, so setting one leaf keeps its sibling defaults.
            merged = _deep_merge(copy.deepcopy(defaults), merged)
        if expand_env:
            merged = _expand_env(merged)
        # Canonicalise `model.*` before anyone reads it: aliases resolved, a
        # dict-valued model id flattened, the id settled on `model.default`.
        self._values = _normalize_root_model_keys(merged)

    def load(self) -> Dict[str, Any]:
        return copy.deepcopy(self._values)

    def load_readonly(self) -> Dict[str, Any]:
        return self._values

    def raw(self) -> Dict[str, Any]:
        return copy.deepcopy(self._raw)


#: ``${VAR}`` and the Cursor-style ``${env:VAR}``. Both name an environment variable.
_ENV_REF_RE = re.compile(r"\$\{([^}]+)\}")


def _expand_one_ref(match: "re.Match[str]") -> str:
    inner = match.group(1).strip()
    name = inner[4:].strip() if inner.startswith("env:") else inner
    value = os.environ.get(name)
    # An unresolved reference stays verbatim. Blanking it produces an empty string
    # that reads as "not configured", sending the reader to the config file when the
    # actual problem is an unset environment variable.
    return value if value is not None else match.group(0)


def _expand_env(value: Any) -> Any:
    """Expand environment references in string leaves.

    Upstream also recognises other secret-reference sources (``file:``, ``vault:``,
    ``bitwarden:``) but does not resolve them here either -- those backends inject
    their values into the environment at startup, so a config reference only ever
    needs the environment shape. Anything unrecognised is left as written.
    """
    if isinstance(value, str):
        return _ENV_REF_RE.sub(_expand_one_ref, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


_source: ConfigSource = DictConfigSource()


def set_config_source(source: ConfigSource) -> None:
    """Install the configuration source for this process.

    Call once during setup, before constructing an agent.
    """
    global _source
    _source = source


def get_config_source() -> ConfigSource:
    """The active context's configuration source, else the process-wide default.

    Consulted on hot paths -- the turn loop reads configuration inside its own
    iteration -- so this stays one ContextVar read and an attribute access.
    """
    active = _active.get_active()
    if active is not None:
        return active.config
    return _source


# -- The four names the extracted core calls. Signatures match upstream exactly. --

def load_config() -> Dict[str, Any]:
    """The merged configuration. Safe to mutate."""
    return get_config_source().load()


def load_config_readonly() -> Dict[str, Any]:
    """The merged configuration, without the defensive copy.

    Mutating the result corrupts the value every later caller sees. Use only on
    read-only paths -- which is why upstream gave it a deliberately alarming name.
    """
    return get_config_source().load_readonly()


def read_raw_config() -> Dict[str, Any]:
    """Configuration as authored, with no defaults merged."""
    return get_config_source().raw()


#: Upstream's config module re-exports these, and lifted modules import them from
#: there rather than from the constants module. Re-exported here for the same reason.
from hermes_core.seams.paths import (  # noqa: E402
    get_hermes_home,
    mkdir_under_hermes_home as ensure_hermes_home,
)

#: Upstream's private name for the recursive expander. Kept so lifted call sites work.
_expand_env_vars = _expand_env


def require_readable_config_before_write(config_path: Any = None) -> Dict[str, Any]:
    """Upstream's read-before-write guard, which here guards nothing.

    Upstream refuses to replace a ``config.yaml`` it could not read or parse, because a
    loader that swallows both failures into ``{}`` would let a read-modify-write caller
    silently wipe every user override. A real hazard -- when the core owns the file.

    It does not own one. ``save_config`` below refuses outright, so there is no write to
    guard and nothing to clobber. Returning the current configuration keeps every
    upstream call site working: they use the return value as the base mapping to modify,
    and then fail at the write itself, which is where the refusal belongs.
    """
    return dict(read_raw_config() or {})


def save_config(config: Dict[str, Any]) -> None:
    """Refused: the core does not write the host's configuration.

    Upstream owns ``~/.hermes/config.yaml`` and rewrites it in place. Here the
    configuration belongs to the embedding application -- it may not even be a file --
    so a write would either be lost or would trample something the host manages.

    Raising rather than silently doing nothing: a caller that believes it saved a
    setting and finds it gone next run has a much harder problem to diagnose.
    """
    raise NotImplementedError(
        "hermes_core does not write configuration. Update it in the host application "
        "and install a new ConfigSource with set_config_source()."
    )


def _sanitize_env_lines(text: str) -> str:
    """Strip lines from a ``.env`` body that are not ``KEY=value`` assignments.

    Guards against a file that is not really a ``.env`` -- a stray shell script, a
    pasted transcript -- being parsed as one and yielding nonsense keys.
    """
    kept = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        candidate = line[len("export "):].lstrip() if line.startswith("export ") else line
        key, separator, _value = candidate.partition("=")
        if separator and key.strip() and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key.strip()):
            kept.append(line)
    return "\n".join(kept)


def normalize_extra_headers(value: Any) -> Dict[str, str]:
    """Coerce configured extra HTTP headers into a plain ``str -> str`` mapping.

    Accepts a mapping or a list of ``"Name: value"`` strings, because both shapes
    appear in hand-written configuration. Anything else yields no headers rather than
    an error: a malformed header block should not stop an agent from starting.
    """
    headers: Dict[str, str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if key is not None and item is not None:
                headers[str(key)] = str(item)
    elif isinstance(value, (list, tuple)):
        for entry in value:
            name, separator, item = str(entry).partition(":")
            if separator and name.strip():
                headers[name.strip()] = item.strip()
    return headers


def apply_terminal_config_to_env(config: Optional[Dict[str, Any]] = None) -> None:
    """No-op: the core does not reshape the process environment.

    Upstream exports terminal settings into ``os.environ`` so subprocesses inherit
    them. A library embedded in someone else's process must not mutate shared
    environment state -- it would leak into every other thread and every subprocess
    the host spawns for its own reasons.
    """


#: Terminal settings upstream exports into the environment for subprocesses. Kept as
#: an empty mapping: a library must not reshape the environment of a process it does
#: not own, so there is nothing to export.
TERMINAL_CONFIG_ENV_MAP: Dict[str, str] = {}


def _terminal_env_value(*_args: Any, **_kwargs: Any) -> Optional[str]:
    """No terminal settings are projected into the environment."""
    return None


def get_managed_system() -> Optional[str]:
    """Which package manager installed the agent, if any. Never applies to a library."""
    return None


def check_config_version(*_args: Any, **_kwargs: Any) -> None:
    """Upstream warns when a config file predates the running version.

    The host owns its configuration format here, so there is no version of ours to
    check it against.
    """


def migrate_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the configuration unchanged.

    Upstream rewrites older layouts in the user's file. The one migration the core
    still needs -- canonicalising the ``model`` section -- happens in
    ``_normalize_root_model_keys`` on every load, without writing anything back.
    """
    return config if config is not None else load_config()


def read_user_config_raw(config_path: Any = None) -> Dict[str, Any]:
    """Configuration exactly as authored. Same answer as :func:`read_raw_config` here.

    Upstream distinguishes them because one reads the file bypassing every cache and
    overlay, for write-back round-trips. Nothing is written back here.
    """
    return read_raw_config()


def is_provider_enabled(provider_id: str) -> bool:
    """Whether a provider is permitted by configuration.

    Absent configuration, every provider is allowed: an empty allow-list means
    "unrestricted", not "nothing works".
    """
    allowed = cfg_get(load_config_readonly(), "providers", "enabled", default=None)
    if not isinstance(allowed, (list, tuple)) or not allowed:
        return True
    return str(provider_id).strip().lower() in {str(p).strip().lower() for p in allowed}


def fast_safe_load(stream: Any) -> Any:
    """Parse YAML, preferring libyaml when it is available."""
    import yaml

    try:
        from yaml import CSafeLoader as _Loader  # type: ignore[attr-defined]
    except ImportError:
        from yaml import SafeLoader as _Loader  # type: ignore[assignment]
    return yaml.load(stream, Loader=_Loader)


def get_custom_provider_model_capability(*args: Any, **kwargs: Any) -> Any:
    from hermes_core.runtime.config_providers import get_custom_provider_model_capability as impl

    return impl(*args, **kwargs)


def apply_custom_provider_tls_to_client_kwargs(*args: Any, **kwargs: Any) -> Any:
    from hermes_core.runtime.config_providers import (
        apply_custom_provider_tls_to_client_kwargs as impl,
    )

    return impl(*args, **kwargs)


def apply_custom_provider_extra_headers_to_client_kwargs(*args: Any, **kwargs: Any) -> Any:
    from hermes_core.runtime.config_providers import (
        apply_custom_provider_extra_headers_to_client_kwargs as impl,
    )

    return impl(*args, **kwargs)


def is_managed() -> bool:
    """Whether the agent was installed by a package manager.

    Upstream uses this to tell a user how to update -- via Nix, Homebrew, or its own
    installer. A library embedded in someone else's application is never managed that
    way, so this is always false and the update advice never fires.
    """
    return False


def load_env() -> Dict[str, str]:
    """Read the workspace's ``.env`` file into a dict.

    Not memoised, unlike upstream: there the same file is read hundreds of times per
    interactive menu render, which does not happen in an embedded core. Reading fresh
    means a rotated secret takes effect without an invalidation call.
    """
    values: Dict[str, str] = {}
    try:
        # utf-8-sig so a byte-order mark does not become part of the first key.
        text = Path(get_env_path()).read_text(encoding="utf-8-sig")
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return values

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if separator and key:
            values[key] = _parse_env_value(_strip_inline_comment(value))
    return values


def _strip_inline_comment(value: str) -> str:
    """Drop a trailing ``# comment``, which must be preceded by whitespace.

    Requiring the space keeps a ``#`` inside a value -- common in generated passwords
    and API keys -- from truncating it. That alone was not enough: this runs *before*
    quotes are removed, so a quoted value containing a space then a hash was cut at the
    space, losing its closing quote too::

        API_KEY="sk-1234 #5678"   ->   '"sk-1234'

    Truncated, with a stray quote, and no error anywhere -- the provider just rejects a
    key that looks almost right. So a quoted value is skipped over as a unit, and only
    what follows the closing quote can hold a comment.
    """
    stripped = value.lstrip()
    if stripped[:1] in ("'", '"'):
        quote = stripped[0]
        index = 1
        while index < len(stripped):
            char = stripped[index]
            # Only double quotes take backslash escapes -- a single-quoted value is
            # literal, which is why `'it\'s'` does not close where it looks like it
            # should. `_parse_env_value` below follows the same rule.
            if char == "\\" and quote == '"' and index + 1 < len(stripped):
                index += 2
                continue
            if char == quote:
                closing = index + 1
                return (stripped[:closing] + re.split(r"\s+#", stripped[closing:], maxsplit=1)[0]).strip()
            index += 1
        # No closing quote: malformed. Left exactly as written rather than guessed at,
        # so the value that reaches the provider is the one in the file.
        return stripped.strip()
    return re.split(r"\s+#", stripped, maxsplit=1)[0].strip()


def get_env_value(key: str) -> Optional[str]:
    """A value from the environment, falling back to the workspace ``.env``.

    The environment wins: a host that exports a variable for one process expects that
    to take precedence over a file it may not even know about.
    """
    # `or None` for the same reason as in `get_env_value_prefer_dotenv` below: an empty
    # value means "not configured" throughout this module, and the two lookups must
    # answer identically for identical inputs or a caller's `is not None` check depends
    # on which precedence it happened to pick.
    return os.environ.get(key) or load_env().get(key) or None


def get_env_value_prefer_dotenv(key: str) -> Optional[str]:
    """The same lookup with the precedence reversed: the ``.env`` file wins.

    Used for credentials the agent manages itself, where a deliberate edit to the
    workspace's ``.env`` should beat a stale value inherited from whatever shell
    happened to launch the process. ``get_env_value`` above keeps the opposite
    precedence for everything else, where an exported variable is the explicit act.

    Its absence was expensive and completely silent. ``runtime/auth.py`` imports this
    name at call time from *every* api-key provider's credential lookup, so the missing
    function raised ``ImportError`` on a path that has nothing to do with ``.env`` --
    and a bare ``except Exception`` several layers up turned that into "no client
    available" at DEBUG level. Environment-variable credential resolution was broken
    for every api-key provider, and the first visible symptom was automatic context
    compaction dying with a tuple-unpacking error three modules away.
    """
    # `or None` so an empty value reads as absent here exactly as it does in
    # `get_env_value` above. Without it the two disagreed on the same file: an empty
    # `.env` entry with nothing in the environment gave `""` from one and `None` from
    # the other, so `if value is not None:` behaved differently depending on which
    # precedence a caller happened to use.
    return load_env().get(key) or os.environ.get(key) or None


# Custom (self-hosted, OpenAI-compatible) provider definitions live in configuration,
# so upstream re-exports these from its config module. Imported lazily because
# config_providers reads configuration through this module.

def get_compatible_custom_providers(*args: Any, **kwargs: Any) -> Any:
    from hermes_core.runtime.config_providers import get_compatible_custom_providers as impl

    return impl(*args, **kwargs)


def get_custom_provider_tls_settings(*args: Any, **kwargs: Any) -> Any:
    from hermes_core.runtime.config_providers import get_custom_provider_tls_settings as impl

    return impl(*args, **kwargs)


def get_custom_provider_context_length(*args: Any, **kwargs: Any) -> Any:
    from hermes_core.runtime.config_providers import get_custom_provider_context_length as impl

    return impl(*args, **kwargs)


def _parse_env_value(raw_value: str) -> str:
    """Parse one value from a ``.env`` file.

    Kept under upstream's private name because lifted modules import it that way --
    ``agent/secret_scope.py`` uses it to read a profile's ``.env``. The grammar is
    deliberately small: a bare value, a ``'single-quoted'`` value taken literally, or
    a ``"double-quoted"`` value in which ``\\"`` and ``\\\\`` are unescaped.

    Only backslash-quote and backslash-backslash are escapes. A lone backslash before
    anything else stays a literal backslash, so a Windows path written unquoted
    survives intact instead of losing separators.
    """
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        quoted = value[1:-1]
        parsed: list[str] = []
        i = 0
        while i < len(quoted):
            escaped = quoted[i] == "\\" and quoted[i + 1:i + 2] in ('"', "\\")
            parsed.append(quoted[i + 1] if escaped else quoted[i])
            i += 2 if escaped else 1
        return "".join(parsed)
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1]
    return value


def cfg_get(cfg: Optional[Dict[str, Any]], *keys: str, default: Any = None) -> Any:
    """Traverse nested dict keys safely, returning ``default`` on any miss.

    An explicit ``None`` stored at the key is returned as-is: ``default`` applies
    only when the key is absent, matching ``dict.get`` semantics.
    """
    if not isinstance(cfg, dict):
        return default
    node: Any = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
