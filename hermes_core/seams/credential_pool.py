"""Credential pooling, in the shape lifted code expects.

Upstream's ``agent/credential_pool.py`` is 2,853 lines and is not a generic key
rotator: it names specific vendors 152 times and imports eighteen private symbols from
the auth module at import time. It manages OAuth device-code grants, single-use
refresh tokens, and provider-specific quota probes.

What the lifted provider-resolution code actually asks of it is much smaller -- load a
pool, ask whether it holds anything, pick an entry, and report whether the pool
belongs to the provider being resolved. That fits on top of
:class:`~hermes_core.seams.credentials.RotatingKeyPool`, so this module presents
upstream's names over the simple pool the core does have.

Refresh is the one thing that genuinely does not carry over: an API key has nothing to
refresh, so ``try_refresh_current`` reports that nothing changed rather than pretending
to renew something.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hermes_core.seams.credentials import (
    Credentials,
    NoCredentials,
    RotatingKeyPool,
    get_credential_source,
    label_for,
    report_credential_failure,
    resolve_credentials,
)

__all__ = [
    "PooledCredential",
    "CredentialPool",
    "load_pool",
    "credential_pool_matches_provider",
    "custom_provider_pool_key_candidates",
    "resolve_runtime_pool_key",
    "get_env_prefer_dotenv",
    "STATUS_OK",
    "STATUS_EXHAUSTED",
    "STATUS_DEAD",
    "AUTH_TYPE_API_KEY",
    "AUTH_TYPE_OAUTH",
    "FAILURE_REASON_BILLING_UNVERIFIED",
]

# Status and kind vocabulary the lifted code compares against. Kept as the same
# strings upstream uses, because they also appear in persisted state and in log lines
# that a reader may be comparing across the two codebases.
STATUS_OK = "ok"
STATUS_EXHAUSTED = "exhausted"
STATUS_DEAD = "dead"

AUTH_TYPE_API_KEY = "api_key"
AUTH_TYPE_OAUTH = "oauth"

#: A provider refused for a billing reason that re-authenticating will not fix.
FAILURE_REASON_BILLING_UNVERIFIED = "billing_unverified"


@dataclass
class PooledCredential:
    """One credential handed out by a pool.

    Field names match upstream's so lifted call sites read it unchanged. ``label``
    identifies the credential in logs; ``access_token`` is the secret and is the one
    field that must never be logged.
    """

    provider: str
    id: str
    label: str
    access_token: str
    auth_type: str = "api_key"
    priority: int = 0
    source: str = "credential_source"
    refresh_token: Optional[str] = None
    last_status: Optional[str] = None
    last_status_at: Optional[float] = None
    base_url: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"PooledCredential(provider={self.provider!r}, label={self.label!r})"


class CredentialPool:
    """The pool for one provider, backed by the installed credential source."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self._current: Optional[PooledCredential] = None

    def has_credentials(self) -> bool:
        """Whether the source can produce a credential for this provider.

        Asking is the only reliable test: a source may read the environment, a file,
        or a remote secret manager, and any of those can be empty at this moment.
        """
        try:
            resolve_credentials(self.provider)
        except NoCredentials:
            return False
        return True

    def select(self) -> Optional[PooledCredential]:
        """Pick a credential to use now.

        With a rotating pool this is the least-recently-used key that is not cooling
        down; with a single-key source it is always the same one.
        """
        try:
            credentials: Credentials = resolve_credentials(self.provider)
        except NoCredentials:
            return None
        if not credentials.api_key:
            return None

        self._current = PooledCredential(
            provider=self.provider,
            id=credentials.label or label_for(credentials.api_key),
            label=credentials.label or label_for(credentials.api_key),
            access_token=credentials.api_key,
            base_url=credentials.base_url,
        )
        return self._current

    def try_refresh_current(self) -> bool:
        """Always false: an API key has nothing to refresh.

        Upstream renews an OAuth access token here. Returning false rather than
        raising lets the caller's "refresh then retry" path fall through to its
        ordinary failure handling, which is the correct outcome when the credential
        was never renewable.
        """
        return False

    def report_failure(self, *, rate_limited: bool = False) -> None:
        """Tell the source the current credential failed, so it can rotate away."""
        if self._current is None:
            return
        report_credential_failure(
            self.provider,
            Credentials(api_key=self._current.access_token, label=self._current.label),
            rate_limited=rate_limited,
        )

    def __repr__(self) -> str:
        return f"CredentialPool(provider={self.provider!r})"


def load_pool(provider: str) -> CredentialPool:
    """The pool for *provider*. Cheap: it resolves lazily through the seam."""
    return CredentialPool(provider)


def credential_pool_matches_provider(
    pool_or_provider: Any,
    provider: Optional[str],
    *,
    base_url: Optional[str] = None,
) -> bool:
    """Whether a pool belongs to the provider being resolved.

    Upstream needs this because one pool can serve several aliases of the same
    endpoint. Here a pool is created per provider name, so the comparison is direct.
    """
    if pool_or_provider is None or not provider:
        return False
    name = getattr(pool_or_provider, "provider", pool_or_provider)
    return str(name).strip().lower() == str(provider).strip().lower()


def custom_provider_pool_key_candidates(
    base_url: Optional[str],
    provider_name: Optional[str] = None,
) -> List[str]:
    """Names under which a custom endpoint's credentials might be stored.

    A self-hosted endpoint can be referred to by the name the host gave it or by its
    URL, and configuration is written by people, so both are tried.
    """
    candidates: List[str] = []
    if provider_name:
        candidates.append(str(provider_name).strip().lower())
    if base_url:
        normalized = str(base_url).strip().rstrip("/").lower()
        if normalized:
            candidates.append(normalized)
    seen = set()
    return [c for c in candidates if c and not (c in seen or seen.add(c))]


def resolve_runtime_pool_key(provider: Optional[str], base_url: Optional[str]) -> str:
    """The key a runtime's credentials are pooled under.

    Upstream has to reconcile several historical spellings of the same custom
    endpoint. Here a pool is keyed by provider name, and a custom endpoint with no
    name falls back to its URL so two different endpoints never share a pool.
    """
    name = str(provider or "").strip().lower()
    if name and name != "custom":
        return name
    endpoint = str(base_url or "").strip().rstrip("/").lower()
    return endpoint or name or "default"


def get_env_prefer_dotenv(key: str) -> str:
    """An environment value, preferring the workspace ``.env`` over the process.

    The inverse of the usual precedence, and deliberate: the ``.env`` was written for
    this agent, while the process environment may belong to whatever started it.
    """
    from hermes_core.seams.config import get_env_value, load_env

    return load_env().get(key) or get_env_value(key) or ""


def is_pool_source_installed() -> bool:
    """Whether the installed source actually rotates, for diagnostics."""
    return isinstance(get_credential_source(), RotatingKeyPool)
