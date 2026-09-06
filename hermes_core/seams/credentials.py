"""Credential seam: how the core learns which key to use for a provider.

Upstream resolves credentials through ``hermes_cli/auth*.py`` -- 8,184 lines across
twelve modules -- and pools them in ``agent/credential_pool.py``, another 2,853.
Neither is generic. The pool alone names specific vendors 152 times and imports
eighteen private symbols from the auth module at import time: it is machinery for
Hermes's own OAuth integrations (device-code flows, single-use refresh grants, quota
probes), not a way to rotate API keys.

An embedded agent almost never needs that. A service running inside an ERP or behind
a messaging platform holds an API key, and what it actually wants from a pool is
narrow: spread load across several keys, and stop using one for a while when the
provider says it is over quota.

So the vendor OAuth stays upstream, and this module provides the two things a host
does need -- a way to answer "which credential for this provider", and a rotating
pool that honours rate limits.

    set_credential_source(
        RotatingKeyPool({"openai": ["sk-a", "sk-b", "sk-c"]}, base_url=...)
    )

A host that already has its own secret manager implements ``CredentialSource``
itself; the protocol is two methods.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, runtime_checkable

from hermes_core.seams import _active

__all__ = [
    "Credentials",
    "CredentialSource",
    "StaticCredentials",
    "EnvCredentials",
    "RotatingKeyPool",
    "NoCredentials",
    "set_credential_source",
    "get_credential_source",
    "resolve_credentials",
    "report_credential_failure",
    "label_for",
]

#: How long a key sits out after the provider reports it is over quota. Long enough
#: for a per-minute window to roll over, short enough that a pool of two keys does not
#: strand the caller.
DEFAULT_COOLDOWN_SECONDS = 60.0


class NoCredentials(RuntimeError):
    """No usable credential is available for a provider.

    Raised rather than returning ``None`` so a missing key fails at the point of use
    with the provider's name in the message, instead of surfacing later as an
    authentication error from the API.
    """


def label_for(secret: str) -> str:
    """A short, non-reversible label for a secret, safe to log.

    Keys are indistinguishable in logs otherwise, which makes "which key got rate
    limited" unanswerable. Only the last four characters are shown, and never for a
    string short enough that those characters are most of it.
    """
    if not secret:
        return "<empty>"
    return f"...{secret[-4:]}" if len(secret) > 8 else "<short>"


@dataclass(frozen=True)
class Credentials:
    """One resolved credential for one provider."""

    api_key: Optional[str]
    base_url: Optional[str] = None
    headers: Mapping[str, str] = field(default_factory=dict)
    #: Identifies this credential in logs without revealing it.
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label and self.api_key:
            object.__setattr__(self, "label", label_for(self.api_key))

    def __repr__(self) -> str:
        """Never render the key.

        A credential ends up in tracebacks, debugger views and log lines; the default
        dataclass repr would print the secret in all three.
        """
        return f"Credentials(label={self.label!r}, base_url={self.base_url!r})"


@runtime_checkable
class CredentialSource(Protocol):
    """Where the core gets credentials."""

    def resolve(self, provider: str) -> Credentials:
        """The credential to use for *provider*.

        Raises :class:`NoCredentials` when none is available.
        """
        ...

    def report_failure(self, provider: str, credentials: Credentials, *, rate_limited: bool) -> None:
        """Tell the source a credential just failed.

        ``rate_limited`` distinguishes "this key is over quota, try another" from
        any other failure. A source that holds one key can ignore this entirely.
        """
        ...


class StaticCredentials:
    """One credential, used for every provider.

    The common case: a service with a single key in its environment.
    """

    def __init__(
        self,
        api_key: Optional[str],
        *,
        base_url: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._credentials = Credentials(
            api_key=api_key, base_url=base_url, headers=dict(headers or {})
        )

    def resolve(self, provider: str) -> Credentials:
        if not self._credentials.api_key:
            raise NoCredentials(f"no API key configured for provider {provider!r}")
        return self._credentials

    def report_failure(self, provider: str, credentials: Credentials, *, rate_limited: bool) -> None:
        """Nothing to do: with one key there is nothing to rotate to."""


class EnvCredentials:
    """Read each provider's key from an environment variable.

    ``{"openai": "OPENAI_API_KEY"}`` maps provider to variable. The variable is read
    at resolve time, not at construction, so a process that loads its environment
    late still works.
    """

    def __init__(
        self,
        variables: Mapping[str, str],
        *,
        base_urls: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._variables = dict(variables)
        self._base_urls = dict(base_urls or {})

    def resolve(self, provider: str) -> Credentials:
        variable = self._variables.get(provider)
        if not variable:
            raise NoCredentials(f"no environment variable mapped for provider {provider!r}")
        value = os.environ.get(variable)
        if not value:
            raise NoCredentials(
                f"environment variable {variable} is unset or empty (provider {provider!r})"
            )
        return Credentials(api_key=value, base_url=self._base_urls.get(provider))

    def report_failure(self, provider: str, credentials: Credentials, *, rate_limited: bool) -> None:
        """Nothing to do: the environment holds one key per provider."""


class RotatingKeyPool:
    """Several keys per provider, rotated, with a cooldown after a rate limit.

    Selection is least-recently-used rather than round-robin, so a key that has just
    been handed out is the last one chosen again. That spreads load evenly even when
    callers resolve at uneven rates -- the multi-tenant shape, where one busy tenant
    would otherwise dominate a round-robin cursor.

    When every key for a provider is cooling down, the one whose cooldown expires
    soonest is returned anyway. Refusing to answer would turn a slow provider into a
    hard outage, and the provider is the right place for that decision: it may well
    accept the call.

    Safe to share across threads.
    """

    def __init__(
        self,
        keys: Mapping[str, Sequence[str]],
        *,
        base_urls: Optional[Mapping[str, str]] = None,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    ) -> None:
        self._keys: Dict[str, List[str]] = {p: list(k) for p, k in keys.items()}
        self._base_urls = dict(base_urls or {})
        self._cooldown = cooldown_seconds
        self._lock = threading.Lock()
        #: (provider, key) -> when it may be used again
        self._cooling: Dict[tuple, float] = {}
        #: (provider, key) -> when it was last handed out
        self._last_used: Dict[tuple, float] = {}

    def providers(self) -> Iterable[str]:
        return tuple(self._keys)

    def resolve(self, provider: str) -> Credentials:
        keys = self._keys.get(provider)
        if not keys:
            raise NoCredentials(f"no keys configured for provider {provider!r}")

        now = time.monotonic()
        with self._lock:
            available = [k for k in keys if self._cooling.get((provider, k), 0.0) <= now]
            if available:
                chosen = min(available, key=lambda k: self._last_used.get((provider, k), 0.0))
            else:
                # Everything is cooling down. Pick whichever recovers first and let
                # the provider decide, rather than failing the turn ourselves.
                chosen = min(keys, key=lambda k: self._cooling.get((provider, k), 0.0))
            self._last_used[(provider, chosen)] = now

        return Credentials(api_key=chosen, base_url=self._base_urls.get(provider))

    def report_failure(self, provider: str, credentials: Credentials, *, rate_limited: bool) -> None:
        """Put a rate-limited key on cooldown.

        Only rate limits trigger it. A malformed request or a server error says
        nothing about the key, and benching it would shrink the pool over a fault it
        did not cause.
        """
        if not rate_limited or not credentials.api_key:
            return
        with self._lock:
            self._cooling[(provider, credentials.api_key)] = time.monotonic() + self._cooldown

    def cooling_down(self, provider: str) -> List[str]:
        """Labels of the keys currently benched, for diagnostics."""
        now = time.monotonic()
        with self._lock:
            return [
                label_for(key)
                for key in self._keys.get(provider, ())
                if self._cooling.get((provider, key), 0.0) > now
            ]


_source: CredentialSource = StaticCredentials(None)


def set_credential_source(source: CredentialSource) -> None:
    """Install the credential source for this process."""
    global _source
    _source = source


def get_credential_source() -> CredentialSource:
    """The active context's credential source, else the process-wide default.

    This is the one that makes the isolation matter: without it, the last tenant to
    configure itself hands its API key to every other tenant in the process.
    """
    active = _active.get_active()
    if active is not None:
        return active.credentials
    return _source


def resolve_credentials(provider: str) -> Credentials:
    return get_credential_source().resolve(provider)


def report_credential_failure(
    provider: str, credentials: Credentials, *, rate_limited: bool = False
) -> None:
    get_credential_source().report_failure(provider, credentials, rate_limited=rate_limited)
