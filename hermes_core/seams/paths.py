"""Workspace-location seam.

Upstream resolves "where does this agent keep things on disk" through
``hermes_constants`` -- 1172 lines encoding the ``~/.hermes`` layout, profile
directories, environment overrides and platform quirks. The extracted core needs
almost none of that. The tool registry, for instance, reaches for exactly three
names, and all three answer one question: *which directory is this agent's
workspace, and what is its stable key?*

Baking ``~/.hermes`` into a general-purpose library would hand every host another
product's directory convention. So this module keeps the three names with their
upstream signatures -- lifted call sites need an import rewrite and no edits -- and
puts the answer behind an injectable protocol.

The default is a directory under the user's home, chosen because it is predictable
and inspectable. A host that cares should set its own::

    set_workspace(DirectoryWorkspace("/var/lib/my-app/agent"))
"""

from __future__ import annotations

import logging
import os
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Union, runtime_checkable

from hermes_core.seams import _active

__all__ = [
    "Workspace",
    "DirectoryWorkspace",
    "set_workspace",
    "get_workspace",
    "get_hermes_home",
    "get_hermes_home_override",
    "hermes_home_key",
    "cache_dir",
]

PathLike = Union[str, "os.PathLike[str]"]


@runtime_checkable
class Workspace(Protocol):
    """Where the agent keeps caches and other on-disk state."""

    def root(self) -> Path:
        """The workspace directory. Need not exist yet."""
        ...

    def key(self, path: Optional[PathLike] = None) -> str:
        """A stable identifier for a workspace directory.

        Used to scope process-wide registries so two workspaces in one process do
        not share cached state. Must be stable across calls for the same directory
        and must differ for directories that are genuinely different.
        """
        ...

    def override(self) -> Optional[str]:
        """An explicit override of the workspace root, or ``None``.

        Upstream exposes this so callers can tell a deliberate redirection apart
        from the default location.
        """
        ...


class DirectoryWorkspace:
    """A workspace rooted at one directory.

    Resolution is deferred and cached per input path, and a directory that does not
    exist yet is resolved without being cached -- its real path can still change once
    it is created, for instance if a parent turns out to be a symlink.
    """

    def __init__(self, root: Optional[PathLike] = None, *, override: Optional[str] = None) -> None:
        self._root = Path(root) if root is not None else Path.home() / ".hermes_core"
        self._override = override
        self._key_cache: dict[str, str] = {}

    def root(self) -> Path:
        return self._root

    def key(self, path: Optional[PathLike] = None) -> str:
        candidate = Path(path) if path is not None else self._root
        raw = str(candidate)
        cached = self._key_cache.get(raw)
        if cached is not None:
            return cached
        resolved = str(candidate.resolve(strict=False))
        if candidate.exists():
            self._key_cache[raw] = resolved
        return resolved

    def override(self) -> Optional[str]:
        return self._override


_workspace: Workspace = DirectoryWorkspace()


def set_workspace(workspace: Workspace) -> None:
    """Install the workspace for this process. Call once, before building an agent."""
    global _workspace
    _workspace = workspace


def get_workspace() -> Workspace:
    """The active context's workspace, else whatever ``set_workspace`` installed.

    The context comes first so several tenants can share a process. With none active
    this is exactly the old behaviour, which is what keeps the single-agent case --
    and every existing caller -- working untouched.
    """
    active = _active.get_active()
    if active is not None:
        return active.workspace
    return _workspace


# -- Names the lifted modules call. Signatures match upstream exactly. ------------

def get_hermes_home() -> Path:
    """The workspace root directory, honouring a task-scoped override."""
    scoped = _override.get()
    return scoped if scoped is not None else get_workspace().root()


def get_hermes_home_override() -> Optional[str]:
    """An explicit override of the workspace root, or ``None``."""
    return get_workspace().override()


def hermes_home_key(path: Optional[PathLike] = None) -> str:
    """A stable key for a workspace directory, defaulting to the current one."""
    return get_workspace().key(path)


def cache_dir() -> Path:
    """The cache directory inside the workspace.

    Not an upstream name -- upstream spells this ``get_hermes_home() / "cache"`` at
    each call site. Naming it once means a host that wants caches somewhere else has
    one place to look.
    """
    return get_workspace().root() / "cache"


def get_default_hermes_root() -> Path:
    """The root workspace directory.

    Upstream distinguishes this from ``get_hermes_home()`` because it supports named
    profiles: with ``HERMES_HOME=<root>/profiles/coder``, the root is ``<root>``. This
    core has no profile concept -- a host that wants several agents gives each its own
    workspace -- so the two are the same directory.
    """
    return get_workspace().root()


def get_config_path() -> Path:
    """Where a file-backed configuration would live.

    The core reads configuration through :mod:`hermes_core.seams.config`, which need
    not involve a file at all. This exists because lifted modules ask for the path,
    usually to report it in a diagnostic.
    """
    return get_workspace().root() / "config.yaml"


def get_env_path() -> Path:
    """The workspace's ``.env`` file."""
    return get_workspace().root() / ".env"


def get_skills_dir() -> Path:
    """Where skills live inside the workspace."""
    return get_workspace().root() / "skills"


def mkdir_under_hermes_home(path: PathLike) -> Path:
    """Create a directory, returning it.

    Upstream also refuses to recreate a deleted named profile here. With no profiles
    there is nothing to guard against, so this is a plain ``mkdir -p``.
    """
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def display_hermes_home() -> str:
    """The workspace path for display, shortened to ``~/`` when it is under home.

    ``as_posix()`` rather than ``str()``: on Windows the latter produces mixed
    separators like ``~/AppData\\Local\\hermes/skills``.
    """
    root = get_workspace().root()
    try:
        return "~/" + root.relative_to(Path.home()).as_posix()
    except ValueError:
        return str(root)


def is_termux() -> bool:
    """True inside Termux on Android."""
    prefix = os.getenv("PREFIX", "")
    return bool(os.getenv("TERMUX_VERSION") or "com.termux/files/usr" in prefix)


_container_detected: Optional[bool] = None


def is_container() -> bool:
    """True inside a container. Detected once per process.

    Lifted modules use this to decide whether host-level assumptions hold -- whether
    a browser can be launched, whether a path outside the workspace is meaningful.
    """
    global _container_detected
    if _container_detected is None:
        _container_detected = (
            Path("/.dockerenv").exists()
            or Path("/run/.containerenv").exists()
            or bool(os.getenv("KUBERNETES_SERVICE_HOST"))
            or _cgroup_mentions_a_container()
        )
    return _container_detected


def _cgroup_mentions_a_container() -> bool:
    try:
        content = Path("/proc/1/cgroup").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return any(marker in content for marker in ("docker", "kubepods", "lxc", "containerd"))


_wsl_detected: Optional[bool] = None


def is_wsl() -> bool:
    """True inside WSL. Detected once per process."""
    global _wsl_detected
    if _wsl_detected is None:
        try:
            _wsl_detected = "microsoft" in Path("/proc/version").read_text(
                encoding="utf-8", errors="replace"
            ).lower()
        except OSError:
            _wsl_detected = False
    return _wsl_detected


# -- scoped overrides -------------------------------------------------------------
#
# Upstream lets one task point at a different workspace without disturbing the rest
# of the process, using a context variable rather than os.environ -- the environment
# is shared by every thread, so a "temporary" change there is not temporary for
# anyone else. The same reasoning applies to a host running several agents at once.

_override: ContextVar[Optional[Path]] = ContextVar("hermes_core_workspace_override", default=None)


def set_hermes_home_override(path: Optional[PathLike]) -> Token:
    """Point this task at a different workspace; returns a token to undo it."""
    return _override.set(Path(path) if path is not None else None)


def reset_hermes_home_override(token: Token) -> None:
    """Restore whatever override was in place before."""
    _override.reset(token)


def get_process_hermes_home() -> Path:
    """The workspace of the process, ignoring any task-scoped override.

    For assets that belong to the process rather than to the current task, which must
    stay reachable while a task is scoped elsewhere.
    """
    return get_workspace().root()


def parse_reasoning_effort(effort: Any) -> Optional[Dict[str, Any]]:
    """Turn a reasoning-effort setting into a config fragment.

    ``None`` when the value is empty or unrecognised, so the caller applies its own
    default. ``{"enabled": False}`` for the several ways of saying off -- including
    YAML's bare ``false``, since ``reasoning_effort: false`` plainly means disabled
    and must not be read as "unrecognised, use the default".
    """
    if effort is None or effort is True:
        return None
    if effort is False:
        return {"enabled": False}
    text = str(effort).strip().lower()
    if not text:
        return None
    if text in {"none", "false", "disabled", "off", "no"}:
        return {"enabled": False}
    if text in {"minimal", "low", "medium", "high"}:
        return {"enabled": True, "effort": text}
    return None


#: The provider finish reason for a response cut off by the output-token limit.
#: Named because the turn loop compares against it in several places, and a typo in a
#: bare string there would silently disable truncation handling.
FINISH_REASON_LENGTH = "length"

#: Progress indicator styles the display layer can use. A host driving its own UI
#: reads the turn callbacks instead and never touches this.
INDICATOR_STYLES = ("dots", "spinner", "none")


def resolve_reasoning_config(effort: Any = None, *, default: Any = None) -> Optional[Dict[str, Any]]:
    """The reasoning configuration for a turn, or ``None`` to leave it to the provider."""
    parsed = parse_reasoning_effort(effort)
    return parsed if parsed is not None else parse_reasoning_effort(default)


# -- named profiles ---------------------------------------------------------------
#
# Upstream supports several named profiles under one home and has to remember that a
# profile was deleted, so a later write does not silently recreate its directory. This
# core has one workspace per agent -- a host wanting several gives each its own -- so
# there is no profile to mark, and these report "not deleted" rather than pretending
# to track something.

def mark_named_profile_deleted(*_args: Any, **_kwargs: Any) -> None:
    return None


def clear_named_profile_deleted(*_args: Any, **_kwargs: Any) -> None:
    return None


def named_profile_is_deleted(*_args: Any, **_kwargs: Any) -> bool:
    return False


def secure_parent_dir(path: Path) -> None:
    """Restrict the directory holding *path* to the owner, where that means anything.

    Called before writing a credential file. Upstream's version refuses two directories
    it must never lock down -- the filesystem root and the install tree -- because a
    ``0o700`` there is not a tightened permission but a broken deployment, and both
    have caused production lockouts. Those refusals are kept.

    On Windows ``chmod`` cannot express this, so it is a no-op there rather than a
    misleading success; POSIX ACLs are what actually protect the file, and the host is
    the one that sets those.
    """
    if os.name != "posix":
        return
    parent = Path(path).parent.resolve()
    # `/` and top-level directories mean the workspace resolved somewhere it should
    # not have. Tightening them would do real damage; the misresolution is the bug.
    if parent == Path("/") or len(parent.parts) < 3:
        return
    install_root = PROJECT_ROOT.parent
    if parent == install_root or install_root in parent.parents:
        logging.getLogger(__name__).warning(
            "Not restricting permissions on %s: it sits inside the installed package "
            "(%s). A credential file here means the workspace was misresolved.",
            parent, install_root,
        )
        return
    try:
        parent.chmod(0o700)
    except OSError:
        # A filesystem that will not take the mode (a network mount, a container
        # overlay) must not stop the credential being written.
        pass


#: The installed package directory, for locating assets that ship with the core.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Placeholder id for a tool call recovered from a stream that was cut off mid-flight.
PARTIAL_STREAM_STUB_ID = "partial-stream-stub"

# Well-known aggregator endpoints, referenced by model-metadata lookups that treat
# them as such rather than as one configured provider among many.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"
AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"
