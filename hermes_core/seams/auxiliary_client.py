"""The secondary model the core calls on its own behalf.

Context compression is the main caller: to keep a long conversation alive it asks a
model to summarise part of the history. That work is not the user's turn, so it
usually wants a cheaper, faster model -- which is why it goes through a separate
client rather than the one running the conversation.

Upstream implements this in ``agent/auxiliary_client.py``, 7,350 lines covering
mixture-of-agents routing, vision fallbacks, per-task concurrency semaphores,
streaming deadlines, quota probes and eight vendors' OAuth. What the compression code
actually calls is nine names, so this module provides those and resolves the client
through the core's own provider and credential seams.

Configure the auxiliary model under ``auxiliary`` in configuration -- either one
setting for everything::

    {"auxiliary": {"provider": "openai", "model": "gpt-4o-mini"}}

or per task, which is what ``task`` selects::

    {"auxiliary": {"compression": {"model": "gpt-4o-mini"}}}

With nothing configured, the auxiliary client falls back to the main model. That is
slower and dearer than a small model, but it always works -- and a compression pass
that silently does not run is far worse, because the conversation simply dies once it
outgrows the context window.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
from contextvars import ContextVar, Token
from typing import Any, Dict, Iterator, List, Optional, Tuple

from hermes_core.providers.base import OMIT_TEMPERATURE  # noqa: F401 -- re-exported
from hermes_core.seams.config import cfg_get, load_config_readonly

logger = logging.getLogger("hermes_core.auxiliary")

__all__ = [
    "AuxiliaryExplicitCancellation",
    "get_text_auxiliary_client",
    "call_llm",
    "aux_interrupt_protection",
    "aux_progress_hook",
    "aux_stream_deadline",
]


class AuxiliaryExplicitCancellation(Exception):
    """An auxiliary request was deliberately cancelled.

    Distinct from a failure: the caller stopped it, usually because the user
    interrupted the turn, and it must not be retried or reported as an error.
    """


def _auxiliary_config(task: Optional[str] = None) -> Dict[str, Any]:
    """Settings for one auxiliary task, falling back to the shared block."""
    auxiliary = cfg_get(load_config_readonly(), "auxiliary", default={}) or {}
    if not isinstance(auxiliary, dict):
        return {}
    shared = {k: v for k, v in auxiliary.items() if not isinstance(v, dict)}
    if task:
        per_task = auxiliary.get(task)
        if isinstance(per_task, dict):
            return {**shared, **per_task}
    return shared


def _get_auxiliary_task_config(task: Optional[str] = None) -> Dict[str, Any]:
    """Upstream's private name for the same lookup."""
    return _auxiliary_config(task)


def _coerce_positive_timeout(value: Any, default: Optional[float] = None) -> Optional[float]:
    """A positive timeout, or *default* for anything else.

    Zero and negatives are rejected rather than passed through: most HTTP clients read
    them as "no timeout", turning a misconfiguration into a request that hangs.
    """
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return default
    return timeout if timeout > 0 else default


def _fallback_entry_api_key(*_args: Any, **_kwargs: Any) -> Optional[str]:
    """Upstream digs a key out of its pool entry as a last resort.

    Credentials come from the credential seam here, so there is no second place to
    look and nothing to fall back to.
    """
    return None


def _resolve_auxiliary_target(task: Optional[str]) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """``(provider, model, base_url, api_key)`` for an auxiliary task."""
    from hermes_core.runtime.auth import (
        AuthError,
        resolve_api_key_provider_credentials,
        resolve_provider,
    )

    settings = _auxiliary_config(task)
    config = load_config_readonly()

    requested = str(settings.get("provider") or "").strip() or None
    provider = resolve_provider(requested)

    model = str(settings.get("model") or "").strip() or None
    if not model:
        # No auxiliary model configured: use the main one. Dearer, but it runs.
        model = str(cfg_get(config, "model", "default", default="") or "").strip() or None

    try:
        credentials = resolve_api_key_provider_credentials(provider)
    except AuthError:
        return provider, model, None, None
    return provider, model, credentials.get("base_url"), credentials.get("api_key")


def get_text_auxiliary_client(
    task: str = "", *, main_runtime: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[Any], Optional[str]]:
    """``(client, model)`` for a text-only auxiliary task.

    Returns ``(None, None)`` rather than raising when nothing is configured: the
    callers treat a missing auxiliary client as "skip this optimisation", which is a
    better outcome than failing the user's turn over a summarisation that could not
    run.
    """
    try:
        from openai import OpenAI
    except ImportError:  # pragma: no cover - openai is a declared dependency
        logger.debug("openai is not installed; auxiliary requests are unavailable")
        return None, None

    try:
        _provider, model, base_url, api_key = _resolve_auxiliary_target(task or None)
    except Exception as exc:
        logger.debug("no auxiliary client for task %r: %s", task, exc)
        return None, None

    if not api_key:
        logger.debug("no auxiliary credentials for task %r", task)
        return None, None

    client = OpenAI(api_key=api_key, base_url=base_url or None)
    return client, model


def call_llm(
    task: Optional[str] = None,
    *,
    messages: List[Dict[str, Any]],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    timeout: Optional[float] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    reasoning_config: Optional[Dict[str, Any]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    api_mode: Optional[str] = None,
    stream: bool = False,
    stream_options: Optional[Dict[str, Any]] = None,
    route_info: Optional[Dict[str, str]] = None,
    latency_info: Optional[Dict[str, int]] = None,
) -> Any:
    """Run one auxiliary request.

    The signature matches upstream's so lifted callers pass their keyword arguments
    unchanged; the parameters this core has no equivalent for are accepted and
    ignored rather than removed, which would break those call sites.
    """
    client, resolved_model = get_text_auxiliary_client(task or "", main_runtime=main_runtime)
    if client is None:
        raise RuntimeError(
            "no auxiliary model is available. Configure `auxiliary.provider` and "
            "`auxiliary.model`, or install a credential source."
        )

    request: Dict[str, Any] = {
        "model": model or resolved_model,
        "messages": messages,
    }
    if temperature is not None:
        request["temperature"] = temperature
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    if tools:
        request["tools"] = tools
    if stream:
        request["stream"] = True
        if stream_options:
            request["stream_options"] = stream_options
    if extra_body:
        request["extra_body"] = extra_body
    if extra_headers:
        request["extra_headers"] = extra_headers

    effective_timeout = _coerce_positive_timeout(timeout)
    if effective_timeout is not None:
        request["timeout"] = effective_timeout

    if route_info is not None:
        route_info["provider"] = provider or ""
        route_info["model"] = str(request["model"] or "")

    return client.chat.completions.create(**request)


_runtime_main: ContextVar[Dict[str, Any]] = ContextVar("hermes_core_runtime_main", default={})


def set_runtime_main(
    provider: str, model: str, *, requested_provider: str = "", base_url: str = "",
    api_key: Any = "", api_mode: str = "", auth_mode: str = "", session_id: str = "",
    cache_scope: str = "",
) -> Token:
    """Record the main conversation's runtime so auxiliary work can match it.

    Context-local rather than global: a host running several agents at once would
    otherwise have one turn overwrite another's routing. Returns a token to restore
    the previous value.
    """
    return _runtime_main.set({
        "provider": provider, "model": model, "requested_provider": requested_provider,
        "base_url": base_url, "api_key": api_key, "api_mode": api_mode,
        "auth_mode": auth_mode, "session_id": session_id, "cache_scope": cache_scope,
    })


def get_runtime_main() -> Dict[str, Any]:
    return _runtime_main.get()


def _resolve_task_provider_model(task: Optional[str] = None) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    """``(provider, model, base_url, api_key, api_mode)`` for an auxiliary task."""
    provider, model, base_url, api_key = _resolve_auxiliary_target(task)
    return provider, model, base_url, api_key, None


def _try_configured_fallback_for_unavailable_client(
    *_args: Any, **_kwargs: Any
) -> Tuple[None, None, None]:
    """Upstream walks a configured chain of fallback providers.

    The core resolves one provider through the credential seam; a host that wants a
    chain implements it in its own ``CredentialSource``.

    Three values, not two: the sole call site unpacks ``fb_client, fb_model, fb_label``.
    Returning a pair raised ``ValueError`` there, and it landed inside an ``except
    ValueError: raise`` written for an unrelated deliberate hard failure -- so instead
    of degrading to "no fallback available" it took the whole turn down.
    """
    return None, None, None


def _validate_proxy_env_urls() -> None:
    """Reject a malformed proxy variable before it reaches the HTTP client.

    A shell typo such as ``:6153export`` otherwise surfaces much later as a cryptic
    "Invalid port" from deep inside httpx.
    """
    from urllib.parse import urlparse

    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"):
        value = str(os.environ.get(key) or "").strip()
        if not value:
            continue
        try:
            parsed = urlparse(value)
            if parsed.scheme in {"http", "https", "socks5", "socks5h"}:
                _ = parsed.port  # raises on a malformed port
        except ValueError as exc:
            raise ValueError(f"{key} is not a usable proxy URL: {value!r} ({exc})") from exc


def _validate_base_url(base_url: str) -> None:
    """Reject an obviously broken endpoint URL before the HTTP client sees it."""
    from urllib.parse import urlparse

    candidate = str(base_url or "").strip()
    if not candidate or candidate.startswith("acp://"):
        return
    try:
        parsed = urlparse(candidate)
        if parsed.scheme in {"http", "https"}:
            _ = parsed.port
            if not parsed.netloc:
                raise ValueError("missing host")
    except ValueError as exc:
        raise ValueError(f"invalid base_url {candidate!r}: {exc}") from exc


def _coerce_llm_message(response: Any) -> Any:
    """Accept a full response or a bare message, and return the message."""
    choices = getattr(response, "choices", None)
    if choices:
        return getattr(choices[0], "message", None)
    return response if hasattr(response, "content") or isinstance(response, dict) else None


def _field(message: Any, name: str) -> Any:
    if isinstance(message, dict):
        return message.get(name)
    return getattr(message, name, None)


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_content_or_reasoning(response: Any, *, max_reasoning_chars: Optional[int] = None) -> str:
    """The text of a response, falling back to its reasoning when there is no content.

    Order: ``content`` with inline think-blocks stripped, then ``reasoning`` or
    ``reasoning_content``, then OpenRouter's ``reasoning_details`` array. Some models
    put a summary entirely in a reasoning field and leave content empty; without the
    fallback a compaction pass would silently produce nothing and the conversation
    would be truncated instead of summarised.

    ``max_reasoning_chars`` bounds the fallback: unbounded chain-of-thought must not
    become the summary that replaces the history.
    """
    message = _coerce_llm_message(response)
    if message is None:
        return ""

    content = _field(message, "content")
    if isinstance(content, str):
        stripped = _THINK_BLOCK.sub("", content).strip()
        if stripped:
            return stripped

    for name in ("reasoning", "reasoning_content"):
        value = _field(message, name)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            return text[:max_reasoning_chars] if max_reasoning_chars else text

    details = _field(message, "reasoning_details")
    if isinstance(details, (list, tuple)):
        parts = [
            str(_field(item, "text") or _field(item, "summary") or "").strip()
            for item in details
        ]
        text = "\n".join(part for part in parts if part).strip()
        if text:
            return text[:max_reasoning_chars] if max_reasoning_chars else text

    return ""


def _read_main_provider() -> str:
    """The provider the conversation is running on."""
    runtime = _runtime_main.get()
    if runtime.get("provider"):
        return str(runtime["provider"])
    return str(cfg_get(load_config_readonly(), "model", "provider", default="") or "")


def _read_main_model() -> str:
    """The model the conversation is running on."""
    runtime = _runtime_main.get()
    if runtime.get("model"):
        return str(runtime["model"])
    return str(cfg_get(load_config_readonly(), "model", "default", default="") or "")


def _managed_local_netloc() -> str:
    """The host:port of a locally-managed inference server, when one is configured."""
    return str(cfg_get(load_config_readonly(), "local_runtime", "netloc", default="") or "")


def _is_managed_local_endpoint(base_url: Optional[str]) -> bool:
    """Whether a base URL points at a locally-managed inference server we started."""
    netloc = _managed_local_netloc()
    return bool(netloc) and netloc in str(base_url or "")


def _fixed_temperature_for_model(
    model: Optional[str], base_url: Optional[str] = None
) -> Any:
    """A temperature this model requires, ``OMIT_TEMPERATURE``, or ``None``.

    Some models manage sampling server-side and reject the parameter's presence
    outright, which is why omission is a distinct answer from "no preference".
    """
    from hermes_core.providers.base import OMIT_TEMPERATURE

    bare = _bare_model(model) or ""
    if "kimi" in bare or "moonshot" in bare:
        return OMIT_TEMPERATURE
    if _is_arcee_trinity_thinking(model):
        return 0.5
    return None


def _apply_user_default_headers(headers: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Merge the host's configured HTTP headers onto resolved ones.

    Configured headers win. ``model.extra_headers`` is an alias that wins over
    ``model.default_headers`` when both are set. The case this exists for: a
    self-hosted endpoint behind a WAF that rejects the SDK's ``User-Agent`` or
    ``X-Stainless-*`` headers, where the host has to override them.

    Values are never logged.
    """
    try:
        config = load_config_readonly()
        configured = cfg_get(config, "model", "default_headers")
        alias = cfg_get(config, "model", "extra_headers")
        if isinstance(alias, dict) and alias:
            configured = {**(configured if isinstance(configured, dict) else {}), **alias}
    except Exception:
        return headers
    if not isinstance(configured, dict) or not configured:
        return headers
    merged = dict(headers or {})
    merged.update({str(k): str(v) for k, v in configured.items()})
    return merged


def _bare_model(model: Optional[str]) -> Optional[str]:
    """The model slug without a provider prefix, lowercased."""
    if not model:
        return None
    return str(model).strip().lower().rsplit("/", 1)[-1]


def _codex_route_bare_model(model: Optional[str], provider: Optional[str]) -> Optional[str]:
    """The bare slug, but only on the Codex OAuth route where these slugs are unique."""
    return _bare_model(model) if (provider or "").strip().lower() == "openai-codex" else None


def _is_codex_gpt54_or_gpt55(model: Optional[str], provider: Optional[str] = None) -> bool:
    """Whether this is a 272K-capped GPT-5.4/5.5/5.6 on the Codex route.

    Route-scoped on purpose: the same slug reached another way exposes a larger
    window, and raising the compression threshold there would compact far too early.
    """
    bare = _codex_route_bare_model(model, provider)
    if bare is None:
        return False
    return bare == "gpt-daybreak-blue-latest" or any(
        bare == family or bare.startswith(family + "-") or bare.startswith(family + ".")
        for family in ("gpt-5.4", "gpt-5.5", "gpt-5.6")
    )


def _is_codex_spark(model: Optional[str], provider: Optional[str] = None) -> bool:
    """Whether this is ``gpt-5.3-codex-spark``, a slug that exists only on that route."""
    return _codex_route_bare_model(model, provider) == "gpt-5.3-codex-spark"


def _is_arcee_trinity_thinking(model: Optional[str]) -> bool:
    bare = _bare_model(model) or ""
    return "trinity" in bare and "thinking" in bare


#: Fractions of the context window at which compression starts, for models whose
#: usable window differs from what the generic threshold assumes.
_CODEX_GPT54_GPT55_COMPACTION_THRESHOLD = 0.85
_CODEX_SPARK_COMPACTION_THRESHOLD = 0.70


def _compression_threshold_for_model(
    model: Optional[str], provider: Optional[str] = None, *,
    allow_codex_gpt55_autoraise: bool = True,
) -> Optional[float]:
    """A per-model compression threshold, or ``None`` to use the configured one.

    Expressed as a fraction of the context window. A reasoning model gets a lower
    threshold so compaction happens before its reasoning context is squeezed; the
    Codex-route models get a higher one because their usable window is larger than
    the generic default assumes.
    """
    if _is_arcee_trinity_thinking(model):
        return 0.75
    if allow_codex_gpt55_autoraise and _is_codex_gpt54_or_gpt55(model, provider):
        return _CODEX_GPT54_GPT55_COMPACTION_THRESHOLD
    if _is_codex_spark(model, provider):
        return _CODEX_SPARK_COMPACTION_THRESHOLD
    return None


def resolve_provider_client(
    provider: Optional[str] = None,
    *,
    model: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
    explicit_api_key: Optional[str] = None,
    api_mode: Optional[str] = None,
    main_runtime: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """``(client, model)`` for a provider, resolving credentials through the seam."""
    try:
        from openai import OpenAI
    except ImportError:  # pragma: no cover - openai is a declared dependency
        return None, None

    from hermes_core.runtime.auth import AuthError, resolve_api_key_provider_credentials

    api_key, base_url = explicit_api_key, explicit_base_url
    if not api_key and provider:
        try:
            resolved = resolve_api_key_provider_credentials(provider)
        except AuthError:
            return None, None
        api_key = resolved.get("api_key")
        base_url = base_url or resolved.get("base_url")

    if not api_key:
        return None, None

    headers = _apply_user_default_headers(None)
    client = OpenAI(
        api_key=api_key,
        base_url=base_url or None,
        **({"default_headers": headers} if headers else {}),
    )
    return client, model


@contextlib.contextmanager
def scoped_runtime_main(main_runtime: Optional[Dict[str, Any]]) -> Iterator[None]:
    """Make *main_runtime* the current one for the duration of the block."""
    if not main_runtime:
        yield
        return
    token = _runtime_main.set(dict(main_runtime))
    try:
        yield
    finally:
        _runtime_main.reset(token)


def _contains_any(text: str, needles: Tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _is_connection_error(exc: BaseException) -> bool:
    """Whether a failure was the network rather than the API.

    The distinction drives what happens next: a connection failure is worth retrying
    or rerouting, while a 4xx is the request's own fault and retrying it just repeats
    the error. Detection reaches past exception classes into the message because these
    arrive wrapped by several layers of HTTP client, and the class is often generic by
    the time the agent sees it.
    """
    with contextlib.suppress(ImportError):
        from openai import APIConnectionError, APITimeoutError

        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            return True
    if _contains_any(type(exc).__name__, ("Connection", "Timeout", "DNS", "SSL")):
        return True
    return _contains_any(
        str(exc).lower(),
        (
            "connection refused", "name or service not known", "no route to host",
            "network is unreachable", "timed out", "connection reset",
            # A stream closed early. Transient, so worth retrying or rerouting.
            "incomplete chunked read", "peer closed connection",
            "response ended prematurely", "unexpected eof",
            "remoteprotocolerror", "localprotocolerror",
        ),
    )


# -- call-shape preservers ---------------------------------------------------------
#
# Upstream wraps auxiliary work in these to interact with its interrupt handling,
# progress display and streaming watchdog -- concerns owned by a CLI and a gateway
# this core does not have. They stay as no-ops so the lifted call sites keep their
# structure: a host that wants progress or cancellation supplies it through the turn
# callbacks instead.


@contextlib.contextmanager
def aux_interrupt_protection(*_args: Any, **_kwargs: Any) -> Iterator[None]:
    """Upstream shields an auxiliary call from the CLI's interrupt handler."""
    yield


@contextlib.contextmanager
def aux_progress_hook(*_args: Any, **_kwargs: Any) -> Iterator[None]:
    """Upstream routes progress to a terminal spinner."""
    yield


@contextlib.contextmanager
def aux_stream_deadline(*_args: Any, **_kwargs: Any) -> Iterator[None]:
    """Upstream arms a watchdog for a stalled stream.

    The request timeout in :func:`call_llm` covers the same failure for a
    non-streaming call, which is what compression makes.
    """
    yield
