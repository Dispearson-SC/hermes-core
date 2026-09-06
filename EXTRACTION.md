# Extracting the Hermes agent core

Working notes for lifting the agent intelligence out of NousResearch/hermes-agent
(clone at `../Hermes`, pinned at `245e48008f`) into a reusable library.

Every number here was measured against that clone, not estimated. Where a
measurement contradicted an earlier assumption, the correction is recorded rather
than the assumption quietly dropped.

## What we are taking

The reasoning loop, provider transports, the tool registry and its tool-call
validation, the MCP client and server, skills, system-prompt assembly, the plugin
engine, and session state plus live context compression behind an interface we own.

## What we are leaving

The gateway and every platform connector, the CLI REPL, the TUI, the desktop app,
the dashboard, the docs site, cron, and all Nous Portal commercial logic.

---

## Measurement 1 — static import closure

Starting from `run_agent` and following only statically-written imports:

| | modules |
|---|---|
| first-party modules in the repo | 1508 |
| reachable at import time (eager) | 142 |
| reachable only from inside function bodies (lazy) | 150 |

Of the 142 eager modules, 39 are capability packs rather than agent intelligence:
browser automation (`tools/browser_tool_*`, `agent/browser_*`, `plugins/browser/*`)
and terminal/sandbox backends (`tools/environments/*` — Docker, SSH, Modal,
Singularity — plus `tools/terminal_tool_*`). `run_agent.py` imports the browser
cluster at module level directly.

**Excluding those packs, the intelligence core is 103 modules — 6.8% of the repo:**

| package | modules |
|---|---|
| `agent/` | 53 |
| `hermes_cli/` (the seam) | 36 |
| root modules | 7 |
| `tools/` | 5 |
| `providers/` | 2 |

`tools/` collapsing from 37 to 5 (`registry`, `arg_coercion`, `interrupt`,
`threat_patterns`, `todo_tool`) is the clearest signal in the whole measurement: the
tool *machinery* is tiny. Everything else under `tools/` is bundled tools.

## Measurement 2 — runtime, and a correction

The static result above said `gateway`, `cron`, `cli`, `tui_gateway` and
`acp_adapter` were absent from the eager closure. Running it for real
(`uv run python -c "from run_agent import AIAgent"`, Python 3.13) **contradicts that
in part** and the earlier claim is withdrawn:

- 396 first-party modules load, not 142.
- `gateway` (19 modules) and `cron` (8 modules) **do** load.
- `cli`, `tui_gateway` and `acp_adapter` do not, at either level.

The gap is dynamic imports, which no static pass can see: `discover_builtin_tools()`
AST-scans `tools/*.py` and imports every match, pulling in ~156 tool modules.

Tracing `__import__` identifies exactly what reaches for the surfaces:

| surface | imported by |
|---|---|
| `cron` | `tools.cronjob_tools`, `tools.cronjob_job_args` |
| `gateway` | `tools.async_delegation`, `tools.desktop_ui`, `tools.react_to_message_tool`, `hermes_cli.plugins_loader`, bundled plugin `hermes_plugins.a2a_platform.*` |

So the architectural conclusion holds — **the agent loop itself never imports the
gateway; the bundled tool catalog does** — but it holds for a different reason than
first claimed, and it is conditional on one design rule rather than being free.

### Design rule this forces

**The core must not auto-discover the bundled tool catalog.** Tool registration is
explicit and opt-in. This was previously a preference; it is now a requirement with
evidence behind it. It is also what an embedding application wants anyway: a host
integrating this core wants to register its own handful of tools, not inherit 156.

`hermes_cli.plugins_loader` reaching into `gateway` is a separate thread to pull when
the plugin engine seam is cut.

---

## The seams

The core reaches into `hermes_cli` 331 times across 99 files. Upstream's own
`AGENTS.md` describes a "narrow waist" with the core sitting below the CLI; that is
aspirational, not enforced. But the 331 imports collapse into a handful of clusters,
and `hermes_cli` turns out to be misnamed — the actual REPL is `cli.py` +
`hermes_cli/main.py`, while the package also holds config, auth, the plugin engine
and the model registry, which the core legitimately needs.

By weight:

| seam | imports | disposition |
|---|---|---|
| `hermes_cli.config` | 176 | Replaced. See below. |
| `hermes_cli.plugin_compat` | 56 | Dropped — an expiring shim, deleted upstream 2026-09-14. |
| `hermes_cli.auth*` (12 modules) | 46 | Behind a credential-source protocol. Per-vendor OAuth (including `auth_spotify`) does not belong in an agent core. |
| `hermes_cli.plugins*` (8 modules) | 39 | Moved into the core. The engine already forbids touching core; it was filed in the wrong package. |
| `hermes_cli.{models,providers,profiles,runtime_provider}` | ~57 | Model/provider metadata, moved next to `providers/`. |
| `hermes_cli._subprocess_compat` | 28 | Vendored as a core utility. No CLI semantics. |
| `hermes_cli.{cli_output,colors}` | — | Presentation. Becomes callbacks. |

### The config seam, and why it costs nothing

`hermes_cli/config.py` is 3912 lines, but 84% of the core's use of it is four names:
`load_config` (63), `load_config_readonly` (48), `cfg_get` (22), `read_raw_config`
(14). All four answer one question: *give me the configuration dict*. `cfg_get` is a
pure ten-line dict traversal with no dependencies at all.

`hermes_core/seams/config.py` keeps those four names with their exact signatures and
makes the source injectable behind a `ConfigSource` protocol, defaulting to a plain
in-memory dict. Because the names and signatures are preserved, **the 176 call sites
need an import-path rewrite and no edits.**

---

## Gaps upstream does not fill

1. **No reusable fake model provider.** Across ~3,821 test files there is none —
   every test hand-rolls `MagicMock()` over `agent.client`. This has to be built, and
   built first: without it the extracted core cannot be tested at all.
2. **No stable public API.** Upstream explicitly disclaims its internal import paths
   as API. This core needs its own documented, versioned surface.

## Acceptance

Integration into Campus-Alert (FastAPI + hexagonal, Python 3.12+), replacing its
LangGraph/OpenRouter agent. Anything that must be patched or reached into to make
that work is a leak in this core's API and is fixed here, not worked around there.

---

## Where this ended up

The extraction is done and runs outside Hermes.

| | |
|---|---|
| lifted from Hermes | 245 modules, 112,973 lines |
| seams written here | 9 modules, 2,252 lines |
| test doubles | 3 modules, 656 lines |
| tests | 8 files, 1,945 lines, **139 passing** |
| extraction tooling | 3 files, 1,280 lines |

By package: `agent/` 146, `runtime/` 56, `tools/` 34, root 8, `providers/` 1.

Hard dependencies: `openai`, `pyyaml`, `requests`, `python-dotenv`,
`concurrent-log-handler`. Nothing else.

### The test that matters

`tests/test_agent_turn.py` runs a real turn with only the model replaced:

    the model asks for get_weather({"city": "Rosario"})
      -> the tool runs with exactly those arguments
      -> {"temp_c": 21, "city": "Rosario"} goes back as a tool-role message
      -> the model answers "Hacen 21 grados en Rosario."
      -> completed: True, 2 API calls

Prompt assembly, tool-call validation, dispatch, result plumbing and the stop
decision are all lifted Hermes code with no Hermes around it.

## The seams, as built

Nine modules stand where upstream had ~30,000 lines of CLI, vendor OAuth and managed
infrastructure:

| seam | replaces | why it is a seam |
|---|---|---|
| `config` | `hermes_cli/config.py` (3,912 lines) | The host owns configuration; the core reads a dict. Upstream's merge and `model.*` canonicalisation are extracted verbatim, because those rules were learned from real misconfigurations. |
| `paths` | `hermes_constants.py` (1,172) | Where state lives. Baking `~/.hermes` into a library hands every host another product's directory convention. |
| `credentials` | — | New. A `CredentialSource` protocol plus `RotatingKeyPool`: rotate keys least-recently-used, bench one on a rate limit, still answer when all are cooling down. |
| `credential_pool` | `agent/credential_pool.py` (2,853) | Upstream's pool names specific vendors 152 times and is welded to their OAuth. Same names, backed by the rotating pool. |
| `auxiliary_client` | `agent/auxiliary_client.py` (7,350) | Compression calls a cheaper model to summarise. It needs nine names, not seven thousand lines. |
| `relay_runtime`, `relay_llm`, `relay_tools` | the vendor's managed relay | Inert. Every entry point is the unmanaged path upstream already takes when no relay is attached. |
| `runtime/auth` | `hermes_cli/auth*.py` (8,184 across 12 modules) | Provider metadata and key resolution. Interactive OAuth is absent by design: a service has no terminal to show a device code to. |

## What was cut, and how

Surgery lives in `tools/lift.py` rather than in the lifted files, so re-running the
lift cannot silently revert it, and a patch that stops applying is a hard error.

- `model_tools.py` line 147, `discover_builtin_tools()` — the single line that
  imports ~156 tool modules and, through three of them, the gateway. After the patch,
  importing `model_tools` registers **zero** tools.
- Gateway probes in `tools/registry.py` and `toolsets.py`.
- The vendor entitlement lookup inside `agent/conversation_loop.py`, and the
  vendor-specific branch of its billing guidance. The loop is now free of it.
- Config *writing* from `utils.py` and `hermes_cli/personality.py` — the host owns
  its configuration, and dropping it also keeps `ruamel.yaml` out of the dependencies.
- Browser automation and sandboxed terminal backends, at their call sites in
  `run_agent.py`, `agent/tool_executor.py`, `agent/chat_completion_helpers.py` and
  `tools/thread_context.py`.
- One upstream inefficiency fixed on the way past: `_cap_delegate_task_calls`
  imported the delegation tool before checking whether the model had asked to
  delegate, so every turn paid for it.

## Three bugs this process caught in its own tooling

Worth recording, because all three were silent.

1. **The import rewriter corrupted a string literal.** Its pattern allowed a bare
   package name, so `if p != "agent"` -- a parameter-name comparison inside the turn
   loop's phase runner -- became a comparison against a module path. The module still
   imported cleanly and every iteration was broken. Fixed by requiring at least one
   dot: a bare `"agent"` is far more often an English word than a module.
2. **A bare `import X` rewrite changed the bound name.** `import run_agent` became
   `import hermes_core.run_agent`, which binds `hermes_core`, so every later use of
   the old name raised at call time. Fixed by emitting an explicit alias.
3. **The lift reported its own failures and then exited 0.** See below -- this one
   cost the most, because it hid three separate broken subsystems at once.

The first two were found by running a turn, not by reading. The third was found by
three independent agents each tripping over the same class of breakage.

### The validator nobody read

`validate_core()` walked every lifted file, correctly found every unresolved
`hermes_core.*` import, printed the list -- and `main()` returned 0 anyway. A
validator whose output nothing acts on is not a validator.

It hid a real and repeating gap. `MANIFEST` lifted a *leaf* module without the
siblings it imports, leaving the leaf unimportable:

| subsystem | lifted | missing |
|---|---|---|
| skills | `tools/skills_tool.py` | `skills_tool_setup`, `skills_tool_dedup`, `skills_tool_plugin`, `path_security` |
| MCP | nine `tools/mcp_tool_*.py` | `mcp_tool_common`, `mcp_tool_errors`, `mcp_tool_loop`, `mcp_tool_sampling`, `mcp_tool_server_run`, `mcp_tool_content`, `ansi_strip` |
| plugins | `hermes_cli/plugins_loader.py` | `plugin_compat` (stood in for by a seam -- see below) |

The root cause is worth stating plainly, because it will recur: the manifest was
closed against a probe that imports `run_agent` and runs a turn. That path never
imports `skills_tool` or `mcp_tool`, so `tools/close_manifest.py` never chased them.
**Import closure has to be probed per subsystem, not just along the turn path.**

Two changes came out of it:

- The check now **measures instead of inferring**. `import_every_module()` imports
  every lifted module in a subprocess and reports which ones fail. Static inference
  both over-reported (a star import resolves nothing statically -- it claimed 37) and
  could under-report (a module can import cleanly and fail on a name built at
  runtime). Importing answers the question a host actually asks. It found 14.
- **The lift exits non-zero when any lifted module cannot be imported.** Unresolved
  imports inside *function bodies* stay a warning, and correctly so: those are lazy
  imports of capability packs this core excludes on purpose, and an unavailable
  branch is how an excluded tool is meant to degrade. An unimportable module is not.

## Verified against a real provider

`tools/live_minimax.py` drives the core against live MiniMax models -- real network,
streaming on, the model deciding for itself whether to call a tool. Both of MiniMax's
protocols, all six models, two API calls each:

| model | OpenAI-compatible `/v1` | Anthropic-compatible `/anthropic` |
|---|---|---|
| MiniMax-M2 | turn ok, tool used | turn ok, tool used |
| MiniMax-M2.1 | turn ok, tool used | turn ok, tool used |
| MiniMax-M2.5 | turn ok, tool used | turn ok, tool used |
| MiniMax-M2.7 | turn ok, tool used | turn ok, tool used |
| MiniMax-M2.7-highspeed | turn ok, tool used | turn ok, tool used |
| MiniMax-M3 | turn ok, tool used | turn ok, tool used |

The model asks for `get_weather`, the tool runs, the model answers from its result.

The script is deliberately not part of the test suite: it costs money, and a suite
that quietly bills whoever runs it is a suite people stop running. It needs
`MINIMAX_API_KEY` in `Core/.env`, which `.gitignore` keeps out of version control.

### The bug only a real provider could find

The streaming path was broken, and the scripted tests could not see it: the fake's own
streaming was disabled, so the symptom got written off as the fake's fidelity rather
than the core's behaviour. Against MiniMax it showed on the first call -- every model
"stalled mid tool-call" and the action was never executed.

The cause was this repository's own `relay_llm` seam. Upstream returns its own
iterator wrapper **even with no relay attached**, and the turn loop reads two things
off it: `final_response` (a whole response, for providers that ignore `stream=True`)
and `on_stream_created` (how the live stream reaches the abort machinery). The seam
returned the SDK's raw stream instead. The loop's read of `stream.final_response`
raised `AttributeError` mid-response, which was classified as a stream that died mid
tool-call, and the complete tool call the model had already sent was dropped -- with
nothing in the logs pointing at the cause.

`UnmanagedLlmStream` now reproduces upstream's unmanaged path. The same fix made the
fake's streaming work, so `tests/test_agent_turn.py` covers streamed turns as well.

The lesson generalises: a seam that looks like a passthrough usually is not, and the
only way to find out is to run the thing against something real.

## Known defects, not fixed

Two are known, measured, and deliberately left alone: both change the shape of what
`run_conversation()` returns, and that is a contract decision rather than a bug fix.
Recorded here in enough detail to act on later without re-deriving anything.

### 1. `api_calls` under-reports on an exhausted turn

**What happens.** When a turn runs out of its iteration budget, the finaliser asks the
model for a summary. Those calls are real, they are billed, and they are not counted:

| `max_iterations` | `result["api_calls"]` | calls the provider actually received |
|---|---|---|
| 0 | 0 | 2 |
| 2 | 2 | 4 |

Measured with a scripted provider counting its own invocations, so the second column is
the ground truth rather than an estimate.

The `max_iterations=0` row is the sharp one: the loop body never executes at all, so
`api_calls` is honestly 0 by its own logic — and yet a model was called twice and its
text became `final_response`. A caller cannot tell "we never tried" from "we tried and
were cut off".

**Origin.** `handle_max_iterations` in `hermes_core/agent/chat_completion_helpers.py`,
reached from `turn_finalizer.py`. The summary call sits outside the budget by design;
what is not by design is that it never reaches the counter.

**Who it hurts.** Anyone metering spend per turn, and anyone gating retries or alerts on
"did we even reach the model". In a multi-tenant deployment that bills per call, this
under-counts, silently, and only on the turns that cost the most.

**Why it is still here.** The fix is to count those calls, which changes `api_calls` for
every existing caller of an exhausted turn. That is a deliberate contract change and
wants to be made once, on purpose.

**Workaround today.** Do not meter on `api_calls`. Count at the transport, or read the
provider's own usage reporting.

### 2. A fast non-retryable failure carries fewer keys than a slow one

**What happens.** Two failure results describe the same class of event with different
shapes:

```python
# after retries were exhausted -- max_retries_exhausted_result
{"completed": False, "failed": True, "error": "...", "final_response": "...",
 "failure_reason": "server_error", "failure_retryable": True,
 "billing_unverified": False, "billing_block": None}

# immediate abort on a 401 -- nonretryable_client_error_result
{"completed": False, "failed": True, "error": "...", "final_response": "...",
 "api_calls": 1}
```

**Origin.** `nonretryable_client_error_result` in `hermes_core/agent/turn_recovery.py`
adds `failure_reason` / `failure_retryable` / `billing_unverified` / `billing_block`
only for the `billing` and `content_policy_blocked` reasons. Every other non-retryable
reason -- `auth` among them -- falls through to a bare `_failed_turn_result()`.
`max_retries_exhausted_result` adds them unconditionally.

**Who it hurts.** A host branching on `"failure_reason" in result` to route an auth
failure differently from a generic one. It works when the failure arrives slowly and
stops working when the same failure arrives fast — which is the harder bug to find,
because the code is right and the timing decides.

**Why it is still here.** Adding keys to a returned dict is cheap; agreeing on which
keys every failure carries is the actual work, and it should be settled once for all
failure shapes rather than patched per call site.

**Workaround today.** Branch on `result.get("error")` text or on `failed is True`, not
on the presence of `failure_reason`.

## Still open

- Session state now sits behind a `SessionStore` protocol
  (`hermes_core/seams/session_store.py`, tests in `tests/test_session_store.py`) with
  an in-memory default and an optional SQLite-backed one. The compaction/rotation
  methods (`archive_and_compact`, `publish_compression_child`, `get_compression_tip`)
  are honest simplifications of upstream's watermark/lock-holder-fenced version --
  correct for one writer, not for several OS processes sharing one file. The
  cross-process turn-lease methods (`acquire_session_turn_lease` and friends) are
  deliberately not part of the protocol; every lifted call site duck-types them and
  skips the whole path when they are absent, same as upstream does for a store that
  doesn't support it.
- **`hermes_cli/auth.py` and its ten siblings are lifted, not seamed.** The plan called
  for credential resolution behind a `CredentialSource` protocol, and that is still the
  right shape: this is 2,243 lines plus per-vendor OAuth for Nous, Codex, MiniMax, xAI,
  Qwen, Z.ai/Kimi -- and Spotify. It is here because `runtime_provider.py` imports it at
  module scope, so nothing works without it, and because it is what the core has
  actually been running on. The real debt is the OAuth flows, not the resolution logic.
- **Three bugs found by adversarial testing are fixed; each was invisible in normal use.**
  Automatic context compaction never ran on either store, because the check for an
  optional lock method asked by importing a module that does not exist, and read the
  failure backwards. A failing store lost the conversation silently -- `completed: True`,
  answer shown, exchange gone -- because the classification path raised before upstream's
  JSONL fallback could run, and that fallback only covered two SQLite-specific errors
  anyway. And `unload_plugins()` bricked the plugin engine for the life of the process,
  because an unguarded gateway import raised mid-teardown and left the manager believing
  it was still discovered. All three are `tools/lift.py` patches; none is a hand edit.
- Skills, MCP, plugins and the session store each have tests plus an adversarial pass.
  288 pass with no failures; 292 with the `mcp` SDK installed, which adds a real
  filesystem MCP server and a full agent turn driven through it.
- Only MiniMax has been exercised live. The provider registry lists sixteen others,
  none of them tested against a real endpoint.
- The `post_tool_call` hook payload is shared, not copied. `invoke_hook` isolates
  exceptions per callback but not data, so a plugin that mutates a nested value in the
  payload changes what every later hook sees. It is the one place the property the
  plugin design rests on -- one plugin cannot damage another -- does not hold.
