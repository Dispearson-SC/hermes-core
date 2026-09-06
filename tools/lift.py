"""Lift modules out of the upstream Hermes clone into this package.

Roughly a hundred modules move across, so moving them by hand is both slow and
unrepeatable. This script does it from a declared manifest instead, which buys three
things a manual copy cannot:

* **Repeatability.** Upstream keeps moving. Re-running the lift against a newer clone
  shows exactly what changed under us.
* **Honesty about the seam.** After rewriting, it reports every first-party import it
  could *not* resolve inside the core. That list is the remaining work, measured
  rather than guessed.
* **A readable diff against upstream.** Lifted modules keep upstream's layout under
  ``hermes_core/`` -- ``agent/foo.py`` becomes ``hermes_core/agent/foo.py``. Renaming
  them would make the rewrite fragile and would permanently destroy the ability to
  diff a lifted module against the original.

The curated public API lives in ``hermes_core/__init__.py``, so mirroring upstream's
internal layout costs nothing at the surface.

Usage:
    python tools/lift.py [--check]

``--check`` reports what would happen without writing anything.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent
UPSTREAM = CORE.parent / "Hermes"

# Modules to lift, in dependency order. A module is listed only once its own
# dependencies are already here, so a failed import surfaces immediately rather than
# after a hundred files have landed.
MANIFEST: list[str] = [
    # -- no first-party imports at all --------------------------------------
    "agent/message_metadata.py",
    "agent/transports/types.py",
    "agent/transports/base.py",
    "agent/transports/__init__.py",
    # -- provider quirks, each self-contained -------------------------------
    "agent/lmstudio_reasoning.py",
    "agent/reasoning_effort.py",
    "agent/moonshot_schema.py",
    "agent/gemini_schema.py",
    "agent/bounded_response.py",
    # -- vendor adapters the wire formats delegate to ------------------------
    "agent/anthropic_endpoints.py",
    "agent/anthropic_message_convert.py",
    "agent/anthropic_credentials.py",
    "agent/anthropic_adapter.py",
    "agent/gemini_native_adapter.py",
    # -- wire formats --------------------------------------------------------
    "agent/transports/anthropic.py",
    "agent/transports/codex.py",
    "agent/transports/chat_completions.py",
    # -- provider profiles ---------------------------------------------------
    "providers/base.py",
    "providers/__init__.py",
    # -- message handling ----------------------------------------------------
    "agent/message_sanitization.py",
    # -- shared utilities ----------------------------------------------------
    "utils.py",
    "agent/secret_scope.py",
    "agent/retry_utils.py",
    "tools/schema_sanitizer.py",
    "tools/budget_config.py",
    # -- the tool machinery --------------------------------------------------
    "tools/registry.py",
    "tools/arg_coercion.py",
    "toolsets.py",
    "model_tools.py",
    # -- services the CLI package held that the core needs -------------------
    "hermes_cli/colors.py",
    "hermes_cli/cli_output.py",
    "hermes_cli/_subprocess_compat.py",
    "hermes_cli/route_identity.py",
    "hermes_cli/managed_scope.py",
    "hermes_cli/secret_prompt.py",
    "hermes_cli/config_providers.py",
    "hermes_cli/config_defaults.py",
    # -- the agent's own persona defaults ------------------------------------
    "hermes_cli/default_soul.py",
    "hermes_cli/personality.py",
    # -- turn-loop leaves: no unlifted first-party dependencies --------------
    "agent/codex_headers.py",
    "agent/credential_persistence.py",
    "agent/empty_response_guard.py",
    "agent/error_classifier.py",
    "agent/fast_mode.py",
    "agent/file_safety.py",
    "agent/iteration_budget.py",
    "agent/jiter_preload.py",
    "agent/memory_provider.py",
    "agent/message_content.py",
    "agent/portal_tags.py",
    "agent/process_bootstrap.py",
    "agent/prompt_cache_boundary.py",
    "agent/provider_projection.py",
    "agent/repetition_guard.py",
    "agent/runtime_cwd.py",
    "agent/session_activity.py",
    "agent/skill_preprocessing.py",
    "agent/skill_utils.py",
    "agent/thinking_timeout_guidance.py",
    "agent/tool_result_classification.py",
    "agent/trajectory.py",
    "agent/turn_loop_errors.py",
    "agent/turn_retry_state.py",
    "hermes_logging.py",
    "hermes_time.py",
    "tools/skill_provenance.py",
    "tools/threat_patterns.py",
    "tools/todo_tool.py",
    "agent/model_metadata.py",
    "agent/redact.py",
    "agent/usage_pricing.py",
    # -- prompt assembly, turn phases, compression support -------------------
    "agent/errors.py",
    "agent/reasoning_timeouts.py",
    "agent/models_dev.py",
    "agent/bedrock_adapter.py",
    "agent/delegation_context.py",
    "agent/codex_responses_adapter.py",
    "agent/display.py",
    "agent/prompt_caching.py",
    "agent/prompt_builder.py",
    "agent/skill_commands.py",
    "agent/turn_truncation.py",
    "agent/context_engine.py",
    "agent/memory_manager.py",
    "agent/micro_compaction.py",
    "agent/turn_response_intake.py",
    "agent/turn_usage.py",
    "agent/turn_api_request.py",
    "agent/turn_api_call.py",
    "agent/turn_stop_gates.py",
    # -- model metadata, plugin engine, hooks and middleware ------------------
    "hermes_cli/codex_models.py",
    "hermes_cli/models.py",
    "hermes_cli/model_normalize.py",
    "hermes_cli/models_pricing.py",
    "hermes_cli/models_reasoning_caps.py",
    "hermes_cli/models_local.py",
    "hermes_cli/runtime_provider_custom.py",
    "hermes_cli/runtime_provider_backends.py",
    "hermes_cli/runtime_provider.py",
    "hermes_cli/lifecycle.py",
    "hermes_cli/middleware.py",
    "hermes_cli/plugin_capabilities.py",
    "hermes_cli/plugins_manifest.py",
    "hermes_cli/plugins_state.py",
    "hermes_cli/plugins_ledger.py",
    "hermes_cli/plugins_discovery.py",
    "hermes_cli/plugins_loader.py",
    "hermes_cli/plugins_dispatch.py",
    "registration_lifecycle.py",
    "hermes_cli/plugins.py",
    "hermes_cli/timeouts.py",
    "hermes_cli/_early_recovery.py",
    "hermes_cli/env_loader.py",
    "hermes_cli/moa_config.py",
    "hermes_cli/fallback_config.py",
    "hermes_cli/urllib_security.py",
    "hermes_cli/prompt_size.py",
    "hermes_cli/process_identity.py",
    "hermes_cli/mem_trim.py",
    # -- skills, secret sources, catalog, remaining turn support -------------
    "hermes_cli/relay_plugin_cutover.py",
    "hermes_cli/models_catalog_static.py",
    "hermes_cli/model_catalog.py",
    "hermes_cli/providers.py",
    "hermes_cli/agent_plugins.py",
    "hermes_cli/commands.py",
    "hermes_cli/approval_transport.py",
    # 2,243 lines of provider credential resolution. The plan was to replace this
    # with a seam; until that exists, lifting it is honest -- it is what the core
    # has actually been running on, and `runtime_provider.py` imports it at module
    # scope, so nothing here works without it.
    "hermes_cli/auth_constants.py",
    "hermes_cli/auth_model_picker.py",
    "hermes_cli/auth_device_flow.py",
    "hermes_cli/auth_oauth_grants.py",
    "hermes_cli/auth_nous.py",
    "hermes_cli/auth_codex.py",
    "hermes_cli/auth_minimax.py",
    "hermes_cli/auth_xai.py",
    "hermes_cli/auth_spotify.py",
    "hermes_cli/auth_qwen.py",
    "hermes_cli/auth.py",
    "hermes_cli/copilot_auth.py",
    "hermes_cli/skin_engine.py",
    "hermes_cli/goals.py",
    "hermes_cli/sizefmt.py",
    "hermes_cli/mcp_security.py",
    "agent/secret_sources/registry.py",
    "agent/secret_sources/_cache.py",
    "agent/secret_sources/command.py",
    "agent/native_compaction.py",
    "agent/agent_runtime_helpers.py",
    "agent/turn_context.py",
    "tools/approval_context.py",
    "tools/terminal_scope.py",
    "tools/lazy_deps.py",
    "tools/thread_context.py",
    "tools/skill_usage.py",
    "tools/skills_guard.py",
    # The `skill_view` / `skills_list` tool is split across four files upstream; the
    # three siblings are as load-bearing as the entry point and lifting it alone left
    # the tool unimportable.
    "tools/path_security.py",
    "tools/skills_tool_setup.py",
    "tools/skills_tool_dedup.py",
    "tools/skills_tool_plugin.py",
    "tools/skills_tool.py",
    "tools/daemon_pool.py",
    # -- live context compression --------------------------------------------
    "agent/conversation_compression.py",
    "agent/context_compressor.py",
    "agent/turn_context.py",
    "agent/turn_context_compaction.py",
    "agent/turn_overflow.py",
    "agent/turn_recovery.py",
    "agent/turn_response_check.py",
    "agent/turn_api_error.py",
    "agent/turn_iteration_prep.py",
    "agent/turn_preflight.py",
    "agent/turn_request_assembly.py",
    "agent/turn_finalizer.py",
    "agent/turn_preflight_gate.py",
    "agent/turn_tool_round.py",
    # -- the loop itself, and the phases that close a turn --------------------
    "agent/turn_empty_response.py",
    "agent/turn_final_response.py",
    "agent/conversation_loop.py",
    # -- the agent object: composition root and the mixins it is built from ---
    "agent/lazy_forward.py",
    "agent/interrupt_compat.py",
    "agent/interrupt_control.py",
    "tools/interrupt.py",
    "agent/activity_tracking.py",
    "agent/api_error_summary.py",
    "agent/api_request_hooks.py",
    "agent/provider_base.py",
    "agent/provider_registry.py",
    "agent/reasoning_params.py",
    "agent/rate_limit_credits.py",
    "agent/status_output.py",
    "agent/thread_scoped_output.py",
    "agent/transcript_repair.py",
    "agent/turn_explainers.py",
    "agent/vision_message_prep.py",
    "agent/tool_guardrails.py",
    "agent/tool_dispatch_helpers.py",
    "agent/background_review.py",
    "agent/client_lifecycle.py",
    "agent/stream_delivery.py",
    "agent/session_persistence.py",
    "agent/compression_facade.py",
    "agent/turn_facade.py",
    "agent/turn_facade_lease.py",
    "agent/tool_executor.py",
    "agent/agent_init.py",
    "hermes_bootstrap.py",
    "run_agent.py",
    "agent/subdirectory_hints.py",
    "agent/search_policy.py",
    "agent/think_scrubber.py",
    "agent/credits_tracker.py",
    "agent/ssl_guard.py",
    "agent/ssl_verify.py",
    "tools/checkpoint_manager.py",
    "agent/aux_accounting.py",
    "agent/prompt_cache_scope.py",
    "agent/review_idle_queue.py",
    "agent/subagent_lifecycle.py",
    # -- system prompt, registries, MCP ---------------------------------------
    "agent/system_prompt.py",
    "agent/chat_completion_helpers.py",
    "agent/secret_sources/base.py",
    "agent/codex_runtime.py",
    "agent/outbound_webhooks.py",
    "agent/shell_hooks.py",
    # Same shape as the skills tool: the MCP client is a dozen files that import each
    # other, and lifting the leaves without the shared helpers left every one of them
    # unimportable.
    # Queries OSV for known-malicious packages before spawning an MCP server via
    # npx/uvx/pipx. Designed to fail OPEN on timeout -- but a missing module fails
    # CLOSED, which blocked every MCP server. Worth keeping on a path where the core
    # launches arbitrary third-party packages.
    "tools/osv_check.py",
    "tools/ansi_strip.py",
    "tools/mcp_schema_cache.py",
    "tools/mcp_tool_common.py",
    "tools/mcp_tool_errors.py",
    # The only bridge from a sync caller to the background MCP asyncio loop. Missing
    # it blocked MCP discovery, dispatch and shutdown alike -- but only once the `mcp`
    # package was installed, because `find_spec("mcp")` short-circuits before the
    # import otherwise. Two stdlib-only functions.
    "agent/async_utils.py",
    "tools/mcp_tool_loop.py",
    "tools/mcp_tool_sampling.py",
    "tools/mcp_tool_server_run.py",
    "tools/mcp_tool_config.py",
    "tools/mcp_tool_schema.py",
    "tools/mcp_tool_transport.py",
    "tools/mcp_tool_discovery.py",
    "tools/mcp_tool_registration.py",
    "tools/mcp_tool_handlers.py",
    "tools/mcp_tool_health.py",
    "tools/mcp_tool_lifecycle.py",
    "tools/mcp_tool.py",
    "tools/interpreter_shutdown.py",
    "agent/reasoning_summaries.py",
    "agent/stream_single_writer.py",
    "agent/opencode_affinity.py",
    "agent/stream_diag.py",
    "agent/chat_completion_helpers_relay.py",
    "agent/inline_tool_executors.py",
    "tools/tool_result_storage.py",
    "agent/deadline.py",
    "tools/mcp_tool_content.py",
    "hermes_cli/auth_zai_kimi.py",
    # -- tool-call validation: the reason this extraction is worth doing ------
    "agent/turn_tool_validation.py",
]

# Surgery applied after the import rewrite, keyed by manifest path.
#
# Every entry removes something that belongs to a surface we are not taking. They
# live here rather than in the lifted files because the lift is re-runnable: an edit
# made by hand would be silently reverted the next time this script runs.
#
# A patch that no longer applies is a hard error. Upstream drifting out from under a
# removal is exactly the thing that must not pass unnoticed.
PATCHES: dict[str, list[tuple[str, str, str]]] = {
    "tools/registry.py": [
        (
            "drop the gateway multiplex probe",
            """    try:
        from gateway.session_context import get_session_env
        if all(str(get_session_env(k, "") or "").strip() for k in _BROWSER_IDENTITY_KEYS):
            return CHECK_FN_CACHE_BYPASS
    except Exception:
        pass
""",
            """    # Upstream consults the gateway's session context here so one Browser session's
    # live tools cannot leak into another when a gateway multiplexes profiles inside
    # a single process. This core has neither a gateway nor multiplexing, and
    # upstream already treats the probe as optional -- the whole block sits under a
    # bare `except Exception: pass` -- so it is dropped rather than carried.
""",
        ),
    ],
    "tools/daemon_pool.py": [
        (
            "explain why the private worker contract is version-detected",
            """        # Mirrors CPython's implementation (3.8–3.13) with two changes:
        # daemon=True and no _threads_queues registration.
""",
            """        # Mirrors CPython's implementation with two changes: daemon=True and no
        # _threads_queues registration.
        #
        # This overrides a private method and calls a private worker, so the argument
        # tuple is CPython's to change -- and it did, in 3.14. ``_worker`` there takes
        # ``(executor_reference, ctx, work_queue)``, where ``ctx`` comes from a new
        # ``_create_worker_context()`` that absorbed what ``_initializer``/``_initargs``
        # used to pass separately; both attributes are gone. Upstream Hermes pins
        # ``<3.14``, so it never met this: on 3.14 every submit raised
        # ``AttributeError: no attribute '_initializer'``, and because tool dispatch runs
        # through this pool, *every tool call in the core failed*.
        #
        # Detected by capability, not by version number: ``hasattr`` tracks the actual
        # contract, so a backport or a further move is picked up on its own.
""",
        ),
        (
            "build the worker argument tuple for the running interpreter's contract",
            """            # Carry the active profile into the review thread so MEMORY.md / skill review writes land in the
            # right profile (#54937).
            t = threading.Thread(
                name=thread_name, target=_worker, daemon=True,
                args=(weakref.ref(self, weakref_cb), self._work_queue, self._initializer, self._initargs),
            )
""",
            """            executor_ref = weakref.ref(self, weakref_cb)
            if hasattr(self, "_create_worker_context"):  # CPython 3.14+
                worker_args = (executor_ref, self._create_worker_context(), self._work_queue)
            else:  # CPython 3.8-3.13
                worker_args = (executor_ref, self._work_queue, self._initializer, self._initargs)
            # Carry the active profile into the review thread so MEMORY.md / skill review writes land in the
            # right profile (#54937).
            t = threading.Thread(
                name=thread_name, target=_worker, daemon=True, args=worker_args,
            )
""",
        ),
    ],
    "agent/anthropic_adapter.py": [
        (
            "report this core's version, not the CLI package's",
            "from hermes_core.runtime import __version__ as _HERMES_VERSION",
            "from hermes_core import __version__ as _HERMES_VERSION",
        ),
    ],
    "hermes_cli/plugins_ledger.py": [
        (
            "guard the gateway sweep the way its own sibling nine lines below is guarded",
            """        from gateway.platform_registry import platform_registry
        for platform_name in tuple(self._plugin_platform_names):
            platform_registry.unregister(platform_name)
""",
            """        # This import is unguarded upstream, where a gateway always exists. Here it
        # raises ModuleNotFoundError on *any* unload-all -- a bare `unload_plugins()`
        # or `discover_plugins(force=True)` -- even with no platform plugin ever
        # loaded. And it raises mid-teardown: registrations are already disposed, but
        # the `self._discovered` reset on the next line never runs, so afterwards
        # `force=False` silently finds nothing and `force=True` raises again, for the
        # life of the process. Since force=True is the only rescan primitive, that
        # makes reloading a plugin during development impossible without a restart.
        # Guarded exactly like the `tools.registry` import eight lines below, which
        # upstream already wraps.
        try:
            from gateway.platform_registry import platform_registry
        except ImportError:
            platform_registry = None
        if platform_registry is not None:
            for platform_name in tuple(self._plugin_platform_names):
                platform_registry.unregister(platform_name)
""",
        ),
    ],
    "agent/session_persistence.py": [
        (
            "divert to JSONL on any unrecoverable write failure, not just two SQLite ones",
            """    if isinstance(e, (StateDbReplacedError, StateDbCorruptError)):
        # A replaced/quarantined handle will not take this batch again — keep it on disk.
        try:
            divert_session_transcript_jsonl(getattr(agent, "session_id", "") or "", batch_rows)
        except Exception:
            logger.warning("JSONL divert failed after state.db %s for %s",
                           agent._last_persistence_error_cause, getattr(agent, "session_id", None), exc_info=True)
""",
            """    # Upstream diverts only for a replaced or quarantined SQLite handle, because those
    # are the two errors its own store will never accept the batch after -- anything
    # else is lock contention that a later flush retries against the same file.
    #
    # That reasoning does not survive the extraction. Here the store is whatever the
    # host installed and can fail for any reason at all, and a failure on the turn's
    # last flush has no later flush to be retried by. The batch is dropped either way
    # once this returns False, so the only question is whether it is dropped onto disk
    # or into nothing. A compression rotation is the one genuine exception: that batch
    # is about to be retried against the successor session just below, so diverting it
    # would duplicate the messages rather than rescue them.
    #
    # Getting this wrong is expensive and invisible: a turn whose write failed still
    # returns completed=True with the model's answer shown, while the exchange is gone
    # from every durable store and the only trace is `result["cleanup_errors"]`, which
    # hosts do not surface.
    if not isinstance(e, CompressionSessionClosedError):
        try:
            divert_session_transcript_jsonl(getattr(agent, "session_id", "") or "", batch_rows)
        except Exception:
            logger.warning("JSONL divert failed after session store %s for %s",
                           agent._last_persistence_error_cause, getattr(agent, "session_id", None), exc_info=True)
""",
        ),
    ],
    "agent/stream_delivery.py": [
        (
            "record streamed text even when nobody is listening to it",
            """        delivered = self._deliver_to_stream_callbacks(text)
        self._enqueue_stream_hook("on_stream_delta", delta=text, kind="text")
        if delivered:
            self._record_streamed_assistant_text(text)
""",
            """        self._deliver_to_stream_callbacks(text)
        self._enqueue_stream_hook("on_stream_delta", delta=text, kind="text")
        # Recording what the model sent and delivering it to a listener are different
        # jobs, and upstream conflates them: it records only when delivery succeeded.
        # That works there because something is always watching a stream.
        #
        # Here nothing need be. An embedded host that only reads what
        # `run_conversation()` returns registers no callback -- and then a *raised*
        # mid-stream error discards every delta that had already arrived, because
        # `_partial_stream_stub` recovers the partial answer from exactly this record.
        # A silent stop is fine; a transport error loses the text and the turn still
        # reports `completed: True`. Bookkeeping the loop depends on must not be
        # conditional on whether anyone was listening.
        self._record_streamed_assistant_text(text)
""",
        ),
    ],
    "agent/tool_guardrails.py": [
        (
            "an unset platform is unattended here -- every attended one was excluded",
            '''def _is_non_interactive_platform(platform: str | None) -> bool:
    """True for gateway/cron sessions where tool loops are unattended."""
    if not isinstance(platform, str) or not platform.strip():
        return False
    return platform.strip().lower() not in _ATTENDED_PLATFORMS
''',
            '''def _is_non_interactive_platform(platform: str | None) -> bool:
    """True where a tool loop runs with nobody watching -- which here is the default.

    Upstream returns False for an unset platform, reading "no platform" as "someone is
    at a terminal". That is right for upstream and wrong by construction here: every
    platform it counts as attended -- cli, tui, desktop, acp, subagent, api_server -- is
    a surface this core deliberately does not carry. An embedded agent has no console
    and nobody to press Ctrl-C.

    The default mattered more than it looks. `hard_stop_enabled` follows this, so with
    no platform set a model could call the same tool with the same arguments
    indefinitely and only the iteration budget would end it -- measured at seven
    identical executions with no guardrail entry, against five and a clean
    `guardrail_halt` once a platform was named. On a metered API that is a bill, and
    the host that hits it is the one that never thought about platform strings.
    """
    if not isinstance(platform, str) or not platform.strip():
        return True
    return platform.strip().lower() not in _ATTENDED_PLATFORMS
''',
        ),
    ],
    "agent/conversation_compression.py": [
        (
            "detect a missing optional lock method by looking, not by importing",
            """    try:
        from hermes_core.seams.session_state import SessionDB
        missing = object()
        return (
            type(lock_db) is SessionDB
            and inspect.getattr_static(SessionDB, "try_acquire_compression_lock", missing) is missing
        )
    except Exception:
        return False
""",
            """    # Upstream asks "is this the exact old SessionDB class from before locks
    # existed?", which only makes sense against its own single concrete class. Here
    # any host may bring its own store, and `try_acquire_compression_lock` is an
    # optional part of the SessionStore protocol that neither shipped store
    # implements.
    #
    # Asking by import got the answer backwards. The import raised, `except
    # Exception: return False` reported "the lock API is NOT absent", the caller then
    # touched the attribute, got AttributeError, and classified that as "lookup
    # failed" rather than "not implemented" -- and a lookup failure means sit out. Net
    # effect: automatic context compaction never ran, on any store, on any cycle.
    #
    # `turn_facade_lease.py` handles the sibling optional methods correctly and this
    # now matches it: ask the type whether the method is there.
    return not callable(getattr(type(lock_db), "try_acquire_compression_lock", None))
""",
        ),
    ],
    "agent/transports/__init__.py": [
        (
            "point transport discovery at the in-core package",
            'importlib.import_module(f"agent.transports.{name}")',
            'importlib.import_module(f"hermes_core.agent.transports.{name}")',
        ),
    ],
    "providers/__init__.py": [
        (
            "point provider discovery at the in-core package",
            'importlib.import_module(f"providers.{modname}")',
            'importlib.import_module(f"hermes_core.providers.{modname}")',
        ),
    ],
    "toolsets.py": [
        (
            "drop the gateway platform-bundle lookup",
            """    try:
        from gateway.platform_registry import platform_registry
        if not platform_registry.is_registered(platform_name):
            return []
    except Exception:
        return []
""",
            """    # Upstream asks the gateway's platform registry whether `<platform>` names a
    # registered messaging plugin, and builds an implicit `hermes-<platform>` bundle
    # if so. This core has no gateway and no messaging platforms, so that bundle is
    # always empty. Upstream already returns [] whenever the lookup fails, so this is
    # the answer it would arrive at anyway -- stated outright rather than reached
    # through a swallowed ImportError.
    return []
""",
        ),
    ],
    "agent/secret_sources/registry.py": [
        (
            "drop the password-manager secret sources",
            '''_BUILTIN_SOURCES = (
    ("agent.secret_sources.bitwarden", "BitwardenSource", "Bitwarden"),
    ("agent.secret_sources.onepassword", "OnePasswordSource", "1Password"),
    ("hermes_core.agent.secret_sources.command", "CommandSource", "command"),
)''',
            '''# Upstream also ships Bitwarden and 1Password sources, which shell out to those
# vendors' CLIs. An embedded core gets its credentials through the credential seam --
# where a host plugs in whatever secret manager it already uses -- so carrying two
# specific ones would be both redundant and a pair of subprocess dependencies.
#
# They were failing harmlessly (the loop below guards each source) but logging a
# warning on every start, which is how a real problem gets missed later.
_BUILTIN_SOURCES = (
    ("hermes_core.agent.secret_sources.command", "CommandSource", "command"),
)''',
        ),
    ],
    "agent/tool_executor.py": [
        (
            "no sandboxed terminal, so no active environment",
            "from hermes_core.tools.terminal_tool_lifecycle import get_active_env\n",
            '''# `get_active_env` reports which sandboxed terminal a turn is attached to, so tool
# results can be scoped and budgeted per environment. That pack is not part of this
# core, and the executor only uses the answer as context -- `None` means "no
# environment", which is both true and a shape the callers already handle.
def get_active_env(_task_id=None):
    """No sandboxed terminal, so no environment is attached to this turn."""
    return None

''',
        ),
    ],
    "tools/thread_context.py": [
        (
            "no shell tool, so no shell approvals to propagate",
            "    from hermes_core.tools import terminal_tool as tt\n\n"
            "    return (tt._get_approval_callback, tt._get_sudo_password_callback,\n"
            "            tt.set_approval_callback, tt.set_sudo_password_callback)\n",
            '''    # These callbacks belong to the shell tool: they ask a human to approve a
    # dangerous command, and to supply a sudo password. The shell tool is part of the
    # sandboxed-terminal capability pack, which this core leaves behind, so there is
    # no command that could need approving.
    #
    # Returning inert callbacks rather than letting the import fail keeps the
    # fail-closed contract intact: nothing to approve, nothing approved.
    def _no_callback():
        return None

    def _ignore_callback(_callback):
        return None

    return (_no_callback, _no_callback, _ignore_callback, _ignore_callback)
''',
        ),
    ],
    "agent/chat_completion_helpers.py": [
        (
            "drop the sandboxed-terminal probe",
            "from hermes_core.tools.terminal_tool_lifecycle import is_persistent_env\n",
            '''# Part of the sandboxed-terminal capability pack, which this core leaves behind.
# Upstream asks whether the turn is attached to a persistent sandbox so it can keep
# the session warm between turns. There is no sandbox here, so the answer is constant.
def is_persistent_env(*_args, **_kwargs):
    """No sandboxed environment, so nothing persists between turns."""
    return False

''',
        ),
    ],
    "run_agent.py": [
        (
            "only ask about delegation when the model asked to delegate",
            """        from hermes_core.tools.delegate_tool import _get_max_concurrent_children
        max_children = _get_max_concurrent_children()
        delegate_count = sum(1 for tc in tool_calls if tc.function.name == "delegate_task")
        if delegate_count <= max_children:
            return tool_calls
""",
            """        # Upstream imports the delegation tool before counting, so every turn pays for
        # it -- and in this core, where subagent delegation is deliberately absent, every
        # turn failed on the import even with no delegation in sight.
        #
        # Counting first is both cheaper and correct: with no delegate_task call there
        # is no cap to apply and nothing to ask.
        delegate_count = sum(1 for tc in tool_calls if tc.function.name == "delegate_task")
        if not delegate_count:
            return tool_calls
        from hermes_core.tools.delegate_tool import _get_max_concurrent_children
        max_children = _get_max_concurrent_children()
        if delegate_count <= max_children:
            return tool_calls
""",
        ),
        (
            "drop the sandboxed-terminal capability pack",
            "from hermes_core.tools.terminal_tool_lifecycle import cleanup_vm, get_active_env\n",
            '''# Sandboxed terminal backends -- Docker, SSH, Modal, Singularity, local -- are a
# capability pack rather than agent intelligence. With browser automation they account
# for 34 of the modules upstream loads eagerly, and they carry their own SDKs. A host
# that wants the agent to run shell commands registers those tools itself.
#
# The teardown hooks stay, as no-ops, so the turn's cleanup path keeps its shape.
def cleanup_vm(task_id=None):
    """No sandboxed terminal to tear down."""
    return None


def get_active_env(task_id=None):
    """No sandboxed terminal environment is active."""
    return None

''',
        ),
        (
            "drop the browser capability pack",
            "from hermes_core.tools.browser_tool_lifecycle import cleanup_browser\n",
            '''def cleanup_browser(task_id=None):
    """No browser session to close."""
    return None

''',
        ),
    ],
    "agent/conversation_loop.py": [
        (
            "drop the vendor entitlement lookup",
            """    try:
        from hermes_core.runtime.nous_account import (
            format_nous_portal_entitlement_message,
            get_nous_portal_account_info,
        )
        account_info = get_nous_portal_account_info(force_fresh=True)
        return format_nous_portal_entitlement_message(account_info, capability=capability) or ""
    except Exception:
        return ""
""",
            """    # Upstream calls one vendor's account service here to explain, in that vendor's
    # words, which plan a capability needs. That is commercial copy for one product,
    # inside what is otherwise generic orchestration -- and it is why extracting this
    # loop needed surgery rather than a copy.
    #
    # The function stays, because the surrounding error paths call it. Upstream
    # already returns "" whenever the lookup fails, and callers treat an empty string
    # as "no extra guidance", so this is a shape they already handle.
    return ""
""",
        ),
        (
            "route billing guidance through the generic path",
            """    if _is_nous_inference_route(provider, base_url):
        return _nous_entitlement_message(capability)

""",
            """    # The vendor-specific branch is gone; every provider now gets the generic
    # explanation below, which reads a quota or permission refusal off the response
    # rather than off one vendor's account API.

""",
        ),
    ],
    "model_tools.py": [
        (
            "let the active AgentContext hand host dependencies to every tool handler",
            """    dispatch_kwargs: Dict[str, Any] = {"task_id": ids.task_id, "session_id": ids.session_id}
""",
            """    # Upstream builds this dict from a fixed set of its own values, so the only way a
    # tool could reach a host's repository, notifier or acting user was to import them --
    # coupling the tool to one application. `AgentContext.extras` opens the same path to
    # the host: whatever it put there arrives as keyword arguments, for the whole time
    # its context is active.
    #
    # Host keys go in first and core keys overwrite them, so a collision can never cost
    # a handler the real `session_id`. That ordering is a backstop, not the check --
    # `AgentContext` rejects the reserved names at construction, where the error can
    # still name the mistake.
    from hermes_core.seams import _active as _hermes_core_active
    dispatch_kwargs: Dict[str, Any] = _hermes_core_active.get_tool_extras()
    dispatch_kwargs.update({"task_id": ids.task_id, "session_id": ids.session_id})
""",
        ),
        (
            "do not auto-discover the bundled tool catalog",
            "discover_builtin_tools()\n",
            """# Upstream calls discover_builtin_tools() here, which AST-scans and imports every
# `tools/*.py` -- about 156 modules. Three of them (async_delegation, desktop_ui,
# react_to_message_tool) import the gateway and two more import cron, which is how a
# headless `import run_agent` ends up loading 396 first-party modules and the very
# surfaces this extraction leaves behind.
#
# So the core registers nothing on its own. A host calls registry.register() for the
# tools it actually wants, and gets exactly those. discover_builtin_tools remains
# importable for anyone who deliberately wants catalog-style discovery over their own
# directory.
""",
        ),
    ],
    "agent/turn_tool_validation.py": [
        (
            "point the invalid-name helper at its extracted home",
            "    from hermes_core.agent.conversation_loop import _invalid_tool_name_error_content",
            "    from hermes_core.agent.tool_name_errors import _invalid_tool_name_error_content",
        ),
    ],
    "agent/agent_init.py": [
        (
            "default _session_db to the in-memory SessionStore seam instead of None",
            "    agent._session_db = session_db  # optional SQLite store (CLI/gateway-provided)",
            """    # A host-supplied store (SQLite, Postgres, an existing session backend, ...) wins.
    # Otherwise every agent gets its own private in-memory store, so the persistence-gated
    # logic scattered through this core (create_session/append_messages_batch/get_session_title/
    # queue_token_counts/the session_search recall tool -- all guarded by
    # `if not agent._session_db: return`) actually runs, instead of `session_db=None` making it a
    # silent no-op the way plain `None` used to. In-memory rather than the seam's disk-backed
    # SqliteSessionStore: this constructor has no signal that the host wants (or has a writable
    # place for) durable storage across process restarts -- passing `session_db=` explicitly is
    # how a host opts into that. See hermes_core.seams.session_store.
    from hermes_core.seams.session_store import InMemorySessionStore
    agent._session_db = session_db if session_db is not None else InMemorySessionStore()""",
        ),
    ],
}

# Top-level functions removed from lifted modules, by name.
#
# Removing a whole function with a text patch is brittle -- reindent it upstream and
# the patch stops matching. These are cut by name through the syntax tree instead, and
# a name that no longer exists is a hard error.
DROPS: dict[str, list[tuple[str, str]]] = {
    "hermes_cli/personality.py": [
        (
            "persist_personality",
            "writes the chosen persona back into the user's config.yaml. Configuration "
            "belongs to the host here -- the core reads it and never edits it -- and this "
            "was the last caller of the comment-preserving YAML writer dropped from "
            "utils.py. Reading and normalising a persona still works; a host that wants "
            "to remember a choice stores it wherever it keeps its own settings.",
        ),
    ],
    "utils.py": [
        (
            "atomic_roundtrip_yaml_update",
            "rewrites one key in the user's config.yaml, preserving comments -- this is "
            "`hermes config set`, a CLI concern. The host owns its configuration here; "
            "the core only reads it. Dropping it also keeps ruamel.yaml out of the "
            "dependency set.",
        ),
        (
            "atomic_roundtrip_yaml_save",
            "persists a whole config-state dict back to the user's file, same CLI "
            "concern and same ruamel dependency. Nothing in the core calls it.",
        ),
    ],
}

# Individual functions pulled out of modules we are not lifting whole.
#
# Some small, self-contained helpers are buried inside modules that carry a great
# deal we are not taking. Lifting the whole module to reach one pure function would
# drag its baggage in; copying the function by hand would lose its provenance and
# drift silently. Extracting it by name keeps both: the source is verbatim, and this
# entry records where it came from and why.
EXTRACTS: list[dict] = [
    {
        "src": "agent/relay_llm.py",
        "names": [
            "_ANTHROPIC_APPEND_DELTAS", "_namespace", "_jsonable", "_jsonable_dict",
            "AnthropicStreamAccumulator",
        ],
        "imports": ["import contextlib", "import json", "from typing import Any"],
        "dst": "agent/anthropic_stream_accumulator.py",
        "doc": (
            "Rebuild an Anthropic message from its stream of server-sent events. "
            "It lives in upstream's relay module but is not relay machinery: any "
            "consumer of an Anthropic stream has to reassemble the message from its "
            "deltas, and getting the block indexing wrong silently drops content. "
            "Extracted so the core keeps the real implementation, not a reinvented one."
        ),
    },
    {
        "src": "hermes_cli/config.py",
        "names": ["_deep_merge", "_normalize_root_model_keys", "split_model_config_default"],
        "dst": "seams/_config_semantics.py",
        "doc": (
            "Configuration semantics taken verbatim from upstream.\n\n"
            "The configuration *source* is ours -- a host hands the core a dict rather than\n"
            "pointing it at ``~/.hermes/config.yaml``. What that dict then *means* is not,\n"
            "and these three carry rules learned from real misconfigurations:\n\n"
            "* ``_deep_merge`` recurses dict-over-dict so overriding one leaf keeps its\n"
            "  sibling defaults, and ignores ``None`` over a dict -- an empty YAML section\n"
            "  (``terminal:`` with no value) would otherwise blank the whole default.\n"
            "* ``_normalize_root_model_keys`` canonicalises the ``model`` section: it aliases\n"
            "  ``api_base`` to ``base_url`` (the name OpenAI-SDK users reach for, which the\n"
            "  runtime does not read), flattens a dict-valued model id, and settles on\n"
            "  ``model.default`` as the one key readers use.\n"
            "* ``split_model_config_default`` turns that value into ``(model, provider)``.\n\n"
            "Reimplementing them would mean rediscovering the same bugs."
        ),
    },
    {
        "src": "agent/prompt_builder.py",
        "names": ["DEVELOPER_ROLE_MODELS"],
        "dst": "agent/prompt_roles.py",
        "doc": (
            "Models that expect the system turn under the ``developer`` role.\n\n"
            "A one-line constant that upstream keeps in ``agent/prompt_builder.py``, a\n"
            "module carrying the whole system-prompt assembly. The chat-completions\n"
            "transport needs only this, so it moves alone."
        ),
    },
    {
        "src": "agent/conversation_loop.py",
        "names": ["_invalid_tool_name_error_content"],
        "dst": "agent/tool_name_errors.py",
        "doc": (
            "Error text for tool calls naming a tool that does not exist.\n\n"
            "Lifted out of upstream's ``agent/conversation_loop.py``, which is 1600 lines\n"
            "and carries the vendor's billing and entitlement logic inline. This helper is\n"
            "pure and has no dependencies, so it moves on its own rather than dragging that\n"
            "module across to reach it."
        ),
    },
]

# Upstream module prefixes rewritten to their in-core equivalent. Order matters:
# the first match wins, so seams (which redirect elsewhere) precede the generic
# package rewrites.
REWRITES: list[tuple[str, str]] = [
    # Seams: upstream reaches into the CLI package for services the core needs.
    # These redirect to our own implementations, which keep the same public names
    # and signatures so call sites need no edits.
    ("hermes_constants", "hermes_core.seams.paths"),
    ("hermes_cli.config", "hermes_core.seams.config"),
    # Upstream's SQLite session store, replaced wholesale by the SessionStore protocol.
    # Eight lifted modules still reach for it lazily from inside error handlers, so
    # without this the *failure* paths failed -- see the seam's docstring for how that
    # lost a conversation without reporting anything.
    ("hermes_state_errors", "hermes_core.seams.session_state"),
    ("hermes_state_common", "hermes_core.seams.session_state"),
    ("hermes_state_registry", "hermes_core.seams.session_state"),
    ("hermes_state", "hermes_core.seams.session_state"),
    # Upstream's credential pool is 2,853 lines of vendor OAuth machinery. The core
    # keeps the names and backs them with a plain rotating key pool instead.
    ("agent.credential_pool", "hermes_core.seams.credential_pool"),
    # Upstream's auxiliary client is 7,350 lines. Compression calls nine names from
    # it; the seam provides those over the core's own provider resolution.
    ("agent.auxiliary_client", "hermes_core.seams.auxiliary_client"),
    # The managed multi-tenant relay: a vendor service this core does not talk to.
    ("hermes_cli.observability.relay_shared_metrics", "hermes_core.seams.relay_shared_metrics"),
    ("agent.relay_runtime", "hermes_core.seams.relay_runtime"),
    ("agent.relay_llm", "hermes_core.seams.relay_llm"),
    ("agent.relay_tools", "hermes_core.seams.relay_tools"),
    # Everything else that lived in the CLI package and the core genuinely needs.
    ("hermes_cli", "hermes_core.runtime"),
    # Straight moves.
    ("agent", "hermes_core.agent"),
    ("tools", "hermes_core.tools"),
    ("providers", "hermes_core.providers"),
    ("utils", "hermes_core.utils"),
    ("registration_lifecycle", "hermes_core.registration_lifecycle"),
    ("hermes_logging", "hermes_core.hermes_logging"),
    ("hermes_time", "hermes_core.hermes_time"),
    ("hermes_bootstrap", "hermes_core.hermes_bootstrap"),
    ("run_agent", "hermes_core.run_agent"),
    ("toolsets", "hermes_core.toolsets"),
    ("model_tools", "hermes_core.model_tools"),
]

# A module is added to REWRITES only once it is in MANIFEST. Rewriting a path before
# the module is lifted turns a visible gap into an ImportError raised at call time,
# deep inside a lazy import; leaving it unrewritten keeps it in the unresolved-imports
# report, which is this extraction's to-do list.

# First-party roots. Anything importing one of these that is not rewritten is a
# dangling reference and gets reported.
FIRST_PARTY = {
    "agent", "tools", "providers", "plugins", "hermes_cli", "gateway", "cron",
    "tui_gateway", "acp_adapter", "model_tools", "toolsets", "utils",
    "hermes_constants", "run_agent", "hermes_bootstrap", "registration_lifecycle",
}


# Where a lifted module lands, when that is not the path it came from.
#
# `hermes_cli` is misnamed upstream: the actual REPL is `cli.py` plus
# `hermes_cli/main.py`, while the package also holds configuration, provider
# metadata, the plugin engine and the agent's own persona defaults -- services the
# core legitimately needs. Carrying a package called `hermes_cli` into a library that
# has no CLI would preserve that confusion, so those modules land under `runtime/`.
DESTINATION_PREFIXES: list[tuple[str, str]] = [
    ("hermes_cli/", "runtime/"),
]


def destination_for(rel: str) -> str:
    for old, new in DESTINATION_PREFIXES:
        if rel.startswith(old):
            return new + rel[len(old):]
    return rel


#: Never pruned. These are written by hand, not lifted, and nothing upstream
#: corresponds to them.
HANDWRITTEN_DIRS: tuple[str, ...] = ("seams", "testing")


def prune_orphans(*, dry_run: bool = False) -> list[str]:
    """Delete previously-lifted files that the manifest no longer names.

    Without this the lift only ever adds. Drop an entry from ``MANIFEST`` and its file
    stays on disk, still importable, still shipped -- indistinguishable from a module
    that is meant to be here. That is how fourteen modules nothing could import
    survived being deliberately removed.

    A lifted file is identified by having an upstream counterpart, which works because
    lifted modules deliberately mirror upstream's layout byte for byte. Hand-written
    files have no counterpart to map back to, and the two hand-written packages are
    excluded outright rather than relying on that.
    """
    # Everything this script legitimately produces: modules copied wholesale, plus the
    # single-function files EXTRACTS carves out of modules that are not lifted whole.
    # Missing the latter is how a first attempt at this deleted `runtime/config.py`
    # and took the core down with it.
    wanted = {destination_for(rel) for rel in MANIFEST}
    wanted |= {destination_for(spec["dst"]) for spec in EXTRACTS}
    removed: list[str] = []

    for path in sorted((CORE / "hermes_core").rglob("*.py")):
        rel = path.relative_to(CORE / "hermes_core").as_posix()
        if rel.split("/", 1)[0] in HANDWRITTEN_DIRS:
            continue
        if rel in wanted:
            continue
        # `ensure_packages` writes these to make a mirrored path importable. Upstream
        # has its own `__init__.py` files, so they map back and would otherwise look
        # like orphans -- deleting them breaks every package under them.
        if path.name == "__init__.py":
            continue
        # Map the in-core path back to the upstream path it would have come from. A
        # file with no upstream counterpart was not lifted -- a generated package
        # `__init__.py`, or something written by hand -- and is left alone.
        candidates = [rel]
        for old, new in DESTINATION_PREFIXES:
            if rel.startswith(new):
                candidates.append(old + rel[len(new):])
        if not any((UPSTREAM / c).exists() for c in candidates):
            continue

        removed.append(rel)
        if not dry_run:
            path.unlink()

    return removed


def rewrite_imports(source: str) -> str:
    """Rewrite first-party module paths to their in-core equivalents.

    Matches both ``from X import y`` and ``import X``, at any indentation, so the
    lazy imports Hermes puts inside function bodies are rewritten too.
    """
    for old, new in REWRITES:
        escaped = re.escape(old)
        source = re.sub(rf"(^\s*from\s+){escaped}(\s|\.)", rf"\1{new}\2", source, flags=re.M)
        # A plain `import run_agent` binds the name `run_agent`. Rewritten naively to
        # `import hermes_core.run_agent` it binds `hermes_core` instead, and every
        # later use of the old name raises NameError -- at call time, far from here.
        # An explicit alias keeps the bound name what the module's code expects.
        # (Dotted bare imports upstream already carry an `as` alias, so they are
        # safe under the general rule below.)
        source = re.sub(
            rf"(^\s*)import\s+{escaped}[ \t]*$", rf"\1import {new} as {old}", source, flags=re.M
        )
        source = re.sub(rf"(^\s*import\s+){escaped}(\s|\.|$)", rf"\1{new}\2", source, flags=re.M)

    # `from agent import relay_runtime` names the submodule in the import list, not in
    # the module path, so the rules above never see `agent.relay_runtime` and a module
    # redirected to a seam keeps resolving to its old home. Handle that shape for every
    # dotted rule whose leaf name is unchanged.
    for old, new in REWRITES:
        if "." not in old or old.rsplit(".", 1)[1] != new.rsplit(".", 1)[1]:
            continue
        old_package, leaf = old.rsplit(".", 1)
        new_package = new.rsplit(".", 1)[0]
        for candidate in {old_package, rewrite_package(old_package)}:
            source = re.sub(
                rf"(^\s*from\s+){re.escape(candidate)}(\s+import\s+{re.escape(leaf)}\b)",
                rf"\1{new_package}\2", source, flags=re.M,
            )
    return source


def rewrite_package(dotted: str) -> str:
    """Apply the module rewrites to a bare package name."""
    for old, new in REWRITES:
        if dotted == old:
            return new
        if dotted.startswith(old + "."):
            return new + dotted[len(old):]
    return dotted


def _manifest_module_map() -> dict[str, str]:
    """Upstream dotted module name -> its dotted name inside the core."""
    mapping: dict[str, str] = {}
    for rel in MANIFEST:
        upstream_dotted = rel[:-3].replace("/", ".").removesuffix(".__init__")
        core_dotted = "hermes_core." + destination_for(rel)[:-3].replace("/", ".").removesuffix(".__init__")
        mapping[upstream_dotted] = core_dotted
    # Seams replace their upstream module wholesale.
    for old, new in REWRITES:
        mapping.setdefault(old, new)
    return mapping


def rewrite_module_strings(source: str) -> str:
    """Rewrite module names that appear as string literals.

    Hermes forwards many agent methods lazily: a mixin holds
    ``importlib.import_module("agent.prompt_caching")`` and resolves the real function
    on first use. Those names never appear in an import statement, so rewriting
    imports alone leaves them pointing at packages that do not exist here -- and
    because they resolve at call time, the failure surfaces deep inside an unrelated
    operation.

    Two guards against corrupting an ordinary string. A candidate must name a module
    actually in the manifest, and it must carry at least one dot: a bare package name
    like "agent" or "tools" is far more often an English word or a config key than a
    module reference. Dropping the second guard silently rewrote
    if p != "agent" -- a parameter-name comparison inside the turn loop's phase
    runner -- into a comparison against a module path, which broke every iteration
    while still importing cleanly.
    """
    mapping = _manifest_module_map()
    if not mapping:
        return source

    pattern = re.compile(r"(['\"])((?:agent|tools|providers|plugins|hermes_cli|model_tools|toolsets|utils|hermes_logging|hermes_constants|registration_lifecycle|run_agent|hermes_bootstrap)(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\1")

    def replace(match: re.Match) -> str:
        quote, dotted = match.group(1), match.group(2)
        target = mapping.get(dotted)
        return f"{quote}{target}{quote}" if target else match.group(0)

    return pattern.sub(replace, source)


def dangling_imports(source: str, path: str) -> list[str]:
    """First-party imports left unresolved after rewriting.

    Reported rather than raised: a dangling import is normal mid-extraction and names
    the next module to lift. Silence here would be the actual problem.
    """
    try:
        tree = ast.parse(source, path)
    except SyntaxError as exc:
        return [f"<syntax error: {exc}>"]

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            root = node.module.split(".")[0]
            if root in FIRST_PARTY:
                found.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FIRST_PARTY:
                    found.add(alias.name)
    return sorted(found)


def drop_functions(source: str, rel: str) -> tuple[str, list[str]]:
    """Remove named top-level functions, leaving a note where each one stood.

    Returns the new source and the names that were not found, so a caller can fail on
    upstream drift rather than silently keeping code we meant to cut.
    """
    wanted = {name: why for name, why in DROPS.get(rel, [])}
    if not wanted:
        return source, []

    tree = ast.parse(source, rel)
    lines = source.splitlines(keepends=True)
    cuts: list[tuple[int, int, str, str]] = []

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in wanted:
            continue
        # Decorators sit above `lineno`, so start from the earliest of them.
        start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        cuts.append((start, node.end_lineno, node.name, wanted.pop(node.name)))

    for start, end, name, why in sorted(cuts, reverse=True):
        note = f"# Removed during extraction -- {name}(): {why}\n"
        lines[start:end] = [note]

    return "".join(lines), sorted(wanted)


def _module_path(dotted: str) -> Path | None:
    """Resolve ``hermes_core.a.b`` to the file backing it, if one exists here."""
    rel = Path(*dotted.split("."))
    for candidate in (CORE / rel.with_suffix(".py"), CORE / rel / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _exported_names(path: Path) -> set[str]:
    """Top-level names a module defines: functions, classes, assignments, imports."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    except SyntaxError:
        return set()

    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    return names


def import_every_module() -> tuple[int, dict[str, str]]:
    """Import every lifted module in a subprocess and report which ones fail.

    The AST check below infers; this measures. Inference over-reports (a star import
    resolves nothing statically) and under-reports (a module can import cleanly and
    still explode on a name it builds at runtime), and the number that actually
    matters -- can a host import this module at all -- is answerable directly.

    Runs in a subprocess because importing ~280 modules has side effects, and because
    a module that hard-exits or segfaults must not take the lift down with it.
    """
    script = textwrap.dedent(
        """
        import importlib, json, pkgutil, sys, tempfile
        import hermes_core.seams  # installs the plugin_compat stand-in
        from hermes_core.seams.paths import DirectoryWorkspace, set_workspace
        set_workspace(DirectoryWorkspace(tempfile.mkdtemp()))
        import hermes_core

        ok, bad = 0, {}
        for mod in pkgutil.walk_packages(hermes_core.__path__, "hermes_core."):
            try:
                importlib.import_module(mod.name)
                ok += 1
            except Exception as exc:
                bad[mod.name] = f"{type(exc).__name__}: {exc}"
        print("---RESULT---")
        print(json.dumps({"ok": ok, "bad": bad}))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=CORE, capture_output=True, text=True,
    )
    _, _, tail = proc.stdout.partition("---RESULT---")
    if not tail.strip():
        return 0, {"<probe>": f"probe did not run: {proc.stderr.strip()[-500:]}"}
    payload = json.loads(tail)
    return payload["ok"], payload["bad"]


def _iter_imports(tree: ast.AST):
    """Yield ``(node, eager)`` for every import, where *eager* means it runs on import.

    Scope is the whole point of this walk. An unresolved import at module scope means
    the module cannot be imported at all; the same import inside a function body means
    one branch is unavailable and everything else still works. Those are different
    severities and conflating them is what let three real gaps hide in a report nobody
    read. ``ast.walk`` flattens the tree and cannot tell them apart, so this descends
    explicitly.

    Class bodies and module-level ``if``/``try`` blocks count as eager: they execute
    during import just as the top level does.
    """
    def walk(node: ast.AST, eager: bool):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                yield child, eager
            # Only a function body defers execution; a class body or a module-level
            # `try` still runs at import time.
            inside = eager and not isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef)
            )
            yield from walk(child, inside)

    yield from walk(tree, True)


def validate_core() -> tuple[list[str], list[str]]:
    """Report in-core references that do not resolve, split by severity.

    Returns ``(blocking, deferred)``.

    * **blocking** -- unresolved at module scope, plus every syntax error and every
      missing symbol. The module cannot be imported. This is a broken lift and the
      script exits non-zero on it.
    * **deferred** -- unresolved inside a function body. Almost always a lazy import
      of a capability pack this core deliberately does not lift (the terminal, vision
      and web tools, the sandboxed environments, the vendor's billing). Reported for
      the record, and not a failure.

    Rewriting is what makes this check necessary. Once ``agent.secret_scope`` becomes
    ``hermes_core.agent.secret_scope`` it no longer looks like a first-party import, so
    the unresolved-upstream report goes quiet even though the module was never lifted.

    Symbols are checked too, and always count as blocking. The seams are written by
    hand and deliberately expose a smaller surface than the upstream modules they
    replace, so lifted code reaching for a helper that was not reproduced is exactly
    the kind of gap worth failing on immediately.
    """
    blocking: list[str] = []
    deferred: list[str] = []
    for path in sorted((CORE / "hermes_core").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        except SyntaxError as exc:
            blocking.append(f"{path.relative_to(CORE)}: syntax error: {exc}")
            continue

        rel = path.relative_to(CORE)
        for node, eager in _iter_imports(tree):
            targets: list[tuple[str, list[str]]] = []
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                targets.append((node.module, [a.name for a in node.names]))
            elif isinstance(node, ast.Import):
                targets.extend((a.name, []) for a in node.names)

            for dotted, imported in targets:
                if not dotted.startswith("hermes_core"):
                    continue
                resolved = _module_path(dotted)
                if resolved is None:
                    where = blocking if eager else deferred
                    where.append(f"{rel}:{node.lineno}: no module {dotted}")
                    continue
                exported = _exported_names(resolved)
                for name in imported:
                    # `from pkg import submodule` is legal and resolves as a module.
                    if name in exported or _module_path(f"{dotted}.{name}"):
                        continue
                    blocking.append(f"{rel}:{node.lineno}: {dotted} has no {name!r}")
    return blocking, deferred


def ensure_packages(path: Path) -> None:
    """Create ``__init__.py`` up the tree so a mirrored path is importable."""
    for parent in path.parents:
        if parent == CORE:
            break
        init = parent / "__init__.py"
        if not init.exists():
            init.write_text("", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="report without writing")
    args = parser.parse_args()

    if not UPSTREAM.is_dir():
        print(f"upstream clone not found at {UPSTREAM}", file=sys.stderr)
        return 2

    dangling: dict[str, list[str]] = {}
    written = 0

    for rel in MANIFEST:
        src = UPSTREAM / rel
        if not src.is_file():
            print(f"MISSING  {rel}", file=sys.stderr)
            return 2

        original = src.read_text(encoding="utf-8")
        rewritten = rewrite_module_strings(rewrite_imports(original))

        for description, old, new in PATCHES.get(rel, []):
            if old not in rewritten:
                print(
                    f"PATCH NO LONGER APPLIES  {rel}: {description}\n"
                    f"  Upstream changed under this removal. Re-read the module and "
                    f"update the patch; do not skip it.",
                    file=sys.stderr,
                )
                return 3
            rewritten = rewritten.replace(old, new, 1)
            print(f"    patched  {rel}: {description}")

        rewritten, missing = drop_functions(rewritten, rel)
        if missing:
            print(
                f"DROP TARGET NOT FOUND  {rel}: {', '.join(missing)}\n"
                f"  Upstream renamed or moved it. Re-read the module and update DROPS.",
                file=sys.stderr,
            )
            return 3
        for name, _why in DROPS.get(rel, []):
            print(f"    dropped  {rel}: {name}()")

        dst = CORE / "hermes_core" / destination_for(rel)

        left = dangling_imports(rewritten, rel)
        if left:
            dangling[rel] = left

        changed = "rewritten" if rewritten != original else "verbatim "
        lines = len(rewritten.splitlines())
        print(f"{'would lift' if args.check else 'lifted'}  {changed}  {lines:>5}  {rel}")

        if not args.check:
            dst.parent.mkdir(parents=True, exist_ok=True)
            ensure_packages(dst)
            dst.write_text(rewritten, encoding="utf-8")
            written += 1

    for extract in EXTRACTS:
        src = UPSTREAM / extract["src"]
        if not src.is_file():
            print(f"MISSING  {extract['src']}", file=sys.stderr)
            return 2

        source = src.read_text(encoding="utf-8")
        tree = ast.parse(source, extract["src"])
        wanted = list(extract["names"])
        segments: list[str] = []

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined = [node.name] if node.name in wanted else []
            elif isinstance(node, ast.Assign):
                # Module-level constants count too: some are a single line living in
                # a module we have no reason to lift whole.
                defined = [t.id for t in node.targets if isinstance(t, ast.Name) and t.id in wanted]
            else:
                continue

            if not defined:
                continue
            segment = ast.get_source_segment(source, node)
            if segment is None:
                print(f"could not read source for {', '.join(defined)}", file=sys.stderr)
                return 3
            segments.append(rewrite_imports(segment))
            for name in defined:
                wanted.remove(name)

        if wanted:
            print(
                f"EXTRACT NOT FOUND  {extract['src']}: {', '.join(wanted)}\n"
                f"  Upstream moved or renamed it. Re-read the module and update EXTRACTS.",
                file=sys.stderr,
            )
            return 3

        # Extracted code keeps whatever the original relied on at module scope, so an
        # entry names the imports its segments need. Without them the module still
        # imports cleanly and fails later, at the first call -- which is exactly the
        # kind of delayed failure this extraction keeps running into.
        extra_imports = "\n".join(extract.get("imports") or ())
        header = (
            f'"""{extract["doc"]}\n\n'
            f'Extracted verbatim from upstream ``{extract["src"]}``; edit there and re-run\n'
            f'the lift rather than editing this file.\n"""\n\n'
            f"from __future__ import annotations\n"
            + (f"\n{extra_imports}\n" if extra_imports else "")
        )
        body = header + "\n\n" + "\n\n\n".join(segments) + "\n"

        print(f"{'would extract' if args.check else 'extracted'}  {', '.join(extract['names'])}"
              f"  ->  {extract['dst']}")

        if not args.check:
            dst = CORE / "hermes_core" / extract["dst"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            ensure_packages(dst)
            dst.write_text(body, encoding="utf-8")
            written += 1

    print(f"\n{written} module(s) written" if not args.check else "\n(check only)")

    # Seven lifted modules do `from hermes_core.runtime import __version__` -- upstream
    # keeps its version in the CLI package's `__init__`, and `hermes_cli/` becomes
    # `runtime/` here. `ensure_packages` writes empty `__init__.py` files, so without
    # this the name is simply absent and those modules fail to import. Re-exported from
    # the real one rather than duplicated, so there is a single source of truth.
    runtime_init = CORE / "hermes_core" / "runtime" / "__init__.py"
    if not args.check and runtime_init.exists():
        runtime_init.write_text(
            '"""Services that lived in upstream\'s CLI package and the core genuinely needs.\n\n'
            "Written by tools/lift.py, not lifted: upstream's `hermes_cli/__init__.py` carries\n"
            "the package version, and seven lifted modules import it from here.\n"
            '"""\n\n'
            "from hermes_core import __version__ as __version__\n",
            encoding="utf-8",
        )

    orphans = prune_orphans(dry_run=args.check)
    if orphans:
        verb = "would remove" if args.check else "removed"
        print(f"\n{verb} {len(orphans)} module(s) no longer in the manifest:")
        for rel in orphans:
            print(f"  {rel}")

    if dangling:
        print("\nunresolved first-party imports -- these name the next modules to lift:")
        for rel, names in dangling.items():
            print(f"  {rel}")
            for name in names:
                print(f"      {name}")
    else:
        print("\nno unresolved upstream imports")

    if args.check:
        return 0

    _, deferred = validate_core()
    if deferred:
        # Lazy imports of capability packs this core does not lift on purpose. Worth
        # printing so the list stays visible, but not a failure: the branch is simply
        # unavailable, which is how an excluded tool is meant to degrade.
        print(f"\nunavailable inside function bodies ({len(deferred)}) -- lazy imports "
              "of modules not lifted; the branch degrades, the module still imports:")
        for problem in deferred:
            print(f"  {problem}")

    importable, failures = import_every_module()
    print(f"\nimport check: {importable} module(s) import cleanly, {len(failures)} fail")
    if failures:
        # Exiting 0 here is exactly how three subsystems -- skills, MCP and plugins --
        # each shipped unimportable without anyone noticing. A lift that leaves a
        # module a host cannot import has not succeeded.
        print("\nBROKEN -- these lifted modules cannot be imported:")
        for name, error in sorted(failures.items()):
            print(f"  {name}\n      {error}")
        print("\nEither add the missing modules to MANIFEST (tools/close_manifest.py "
              "automates this against a probe), or -- if the module belongs to a "
              "capability pack this core deliberately excludes -- remove it from "
              "MANIFEST rather than shipping it broken.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
