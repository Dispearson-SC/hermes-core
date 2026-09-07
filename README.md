# hermes-core

The agent intelligence of [Hermes Agent](https://github.com/NousResearch/hermes-agent),
extracted as an embeddable library: the reasoning loop, provider transports, the tool
registry and its tool-call validation, MCP, skills, system-prompt assembly, and context
management — without the CLI, the TUI, the gateway, the desktop app, or any one vendor's
commercial logic.

This is the integration guide. For what came out of upstream, what stayed behind, and
why each cut was made, see [EXTRACTION.md](EXTRACTION.md).

---

## Install

```bash
uv pip install -e path/to/Core
```

Python 3.11 – 3.14. Five hard dependencies (`openai`, `pyyaml`,
`concurrent-log-handler`, `python-dotenv`, `requests`); everything provider- and
protocol-specific is an extra:

| Extra | Brings | Needed when |
|---|---|---|
| `anthropic` | `anthropic` SDK | Talking to Anthropic's own wire format |
| `mcp` | `mcp` | Using MCP servers — without it, MCP discovery is skipped, not broken |
| `dev` | `pytest`, `pytest-asyncio` | Running this repository's tests |

### Development and containers are different installs

An editable path install is right while you are integrating: anything you have to patch
or reach around to make the core work is a leak in *its* API and comes back as a fix
here, and a pinned tag turns each of those into a release during the phase where they
are expected.

A container has no such path, and pinning a branch is not pinning. Declare both, and let
the build pick:

```toml
# pyproject.toml
dependencies = ["hermes-core @ git+https://github.com/…/hermes-core@v0.0.2"]

[tool.uv.sources]
hermes-core = { path = "../path/to/Core", editable = true }   # development only
```

```dockerfile
RUN uv pip install --no-sources -r pyproject.toml
```

`--no-sources` ignores the `[tool.uv.sources]` block and resolves the git URL, so
development keeps the editable checkout and the image resolves a fixed ref — one
declaration, no vendoring, no second file. (Pattern contributed by the first host to
containerise this core.)

Pin a **tag**, not a branch, and not a commit if you can help it: a SHA is reproducible
but says nothing in a diff about what it contains. If the repository is private, the
build needs a credential — that is a deployment decision, not a packaging one.

## The shortest thing that works

```python
from hermes_core import AIAgent, registry, tool_result

registry.register(
    name="get_weather",
    toolset="demo",
    schema={
        "name": "get_weather",
        "description": "Look up the weather in a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    handler=lambda args, **_core: tool_result(temp_c=21, city=args["city"]),
)

agent = AIAgent(
    base_url="https://api.example.com/v1",
    api_key="sk-...",
    model="some-model",
    enabled_toolsets=["demo"],
)
result = agent.run_conversation("¿qué temperatura hace en Rosario?")
print(result["final_response"])
```

`AIAgent.__init__` carries about eighty keyword parameters inherited from upstream. The
ones above, plus `provider`, `max_iterations`, `session_db`, `enabled_toolsets`,
`quiet_mode` and `system_message`/`conversation_history` on `run_conversation`, are what
this core documents. The rest exist and are not part of its contract.

**The core registers no tools of its own.** Upstream imports every bundled tool at
startup — about 156 modules, three of which reach into the gateway. Here a host gets
exactly the tools it registers.

## The public surface

`hermes_core` exports its names lazily (PEP 562), so importing the package costs almost
nothing and a cycle inside a lifted module cannot turn `import hermes_core` into an
error. Everything below is the contract; reaching past it into a lifted module works and
may stop working after any re-lift, with no warning.

```python
from hermes_core import (
    # ports a host may implement
    Workspace, ConfigSource, CredentialSource, SessionStore, SessionReader, AgentMailbox,
    # adapters that ship, so nothing has to be implemented to start
    DirectoryWorkspace, DictConfigSource, StaticCredentials, EnvCredentials,
    RotatingKeyPool, NoCredentials, Credentials,
    InMemorySessionStore, SqliteSessionStore, SessionStoreBase,
    InMemoryMailbox, SqliteMailbox, MailboxMessage, MailboxState,
    # binding them
    AgentContext, activate, get_active_context,
    set_workspace, set_config_source, set_credential_source,
    # running a turn
    AIAgent, run_conversation_async, call_host_async, bind_host_loop, get_host_loop,
    HostLoopUnavailable,
    # tools
    registry, ToolRegistry, tool_result, tool_error,
    # testing your own integration
    FakeProvider, Script, StreamDrop, install_fake_client,
)
```

## Tools

A tool is a plain function, a JSON Schema, and one `register` call.

```python
def report_incident(args, *, repo, actor, **_core):
    incident = repo.create(title=args["title"], reported_by=actor)
    return tool_result(id=incident.id, status=incident.status)
```

Three rules:

- **Return `tool_result(...)` or `tool_error(...)`.** Both produce the JSON string the
  loop feeds back to the model. `tool_error` bounds its message, so an enormous exception
  string cannot bloat the conversation on every retry.
- **Keep a `**kwargs` catch-all.** The core passes `task_id`, `session_id` and
  `user_task`; a handler that names only its own arguments breaks on a core it did not
  expect.
- **Raising is allowed.** An exception becomes `{"error": ...}` and goes back to the
  model, which gets a chance to correct itself. It does not abort the turn.

A tool whose backing service may be down takes `check_fn=lambda: ...`: when it returns
false the tool is not offered to the model at all, which beats offering it and failing
the call.

### Getting host dependencies into a handler

The `repo` and `actor` above come from the active context, not from an import:

```python
from hermes_core import AgentContext, DirectoryWorkspace, DictConfigSource, StaticCredentials

tenant = AgentContext(
    workspace=DirectoryWorkspace("/var/lib/app/tenants/acme"),
    config=DictConfigSource({"model": {"default": "some-model"}}),
    credentials=StaticCredentials(api_key),
    session_store=SqliteSessionStore("/var/lib/app/tenants/acme/sessions.db"),
)

# per request
with tenant.derive(extras={"repo": incident_repo, "actor": user.id}).activate():
    agent.run_conversation(text)
```

Whatever is in `extras` arrives as keyword arguments on every tool call for as long as
that context is active. This is what keeps a tool from importing one application's
modules — the same tool, registered once, serves every tenant.

`extras` may not use `task_id`, `session_id`, `user_task` or `enabled_tools`; the core
passes those itself, and a context that shadows one is rejected when it is built rather
than silently handing a handler the wrong value mid-turn.

## The four ports

Each is a `@runtime_checkable` `Protocol`. Implement it structurally — matching methods
are enough, no inheritance, no registration.

| Port | Answers | Ships |
|---|---|---|
| `Workspace` | Where does state go? | `DirectoryWorkspace` |
| `ConfigSource` | What are the settings? | `DictConfigSource` |
| `CredentialSource` | Which key, for which provider? | `StaticCredentials`, `EnvCredentials`, `RotatingKeyPool`, `NoCredentials` |
| `SessionStore` | Where does the transcript live? | `InMemorySessionStore`, `SqliteSessionStore` |

They can be installed process-wide (`set_workspace(...)`) for a single-agent process, or
bundled into an `AgentContext` and activated per request. For two tenants in one worker,
use the context: the process-wide setters are service locators, so the second tenant
configured silently replaces the first, API key included.

**Every `AgentContext` field is optional**, and an unset port means *fall through to the
process-wide default* — not "no workspace". So both shapes work, and a single-tenant host
does not have to restate its startup configuration on every request just to pass
`extras`:

```python
# multi-tenant: everything varies per tenant
AgentContext(workspace=..., config=..., credentials=..., session_store=...)

# single-tenant: ports set once at startup, only dependencies vary per request
set_workspace(...); set_config_source(...); set_credential_source(...)
with AgentContext(extras={"repo": repo, "actor": user.id}).activate():
    agent.run_conversation(text)
```

A context is an **isolation** boundary, not a **security** boundary. It stops tenants
tripping over each other. It does not contain code that goes looking. Untrusted tenants
need a process boundary.

### Implementing `SessionStore` over your own database

Sixteen methods, every one of them backed by a real call site in the lifted core (the
module docstring in `hermes_core/seams/session_store.py` lists the grep). You do not have
to write sixteen: `SessionStoreBase` implements all of them in terms of **six storage
primitives**, so subclassing it and writing those six gets you a working store.

```python
from hermes_core import SessionStoreBase

class PostgresSessionStore(SessionStoreBase):
    def _read_session(self, session_id): ...
    def _write_session(self, session_id, row): ...
    def _read_messages(self, session_id): ...
    def _append_rows(self, session_id, rows): ...
    def _replace_rows(self, session_id, rows): ...
    def _all_sessions(self): ...
```

`SqliteSessionStore` in the same file is a complete worked example. Two things to know
before you start:

- **The protocol is synchronous.** The turn loop calls it inline, many times per turn. An
  async ORM needs either a synchronous engine against the same database or a bridge —
  and a bridge on this path is chatty enough to be a poor trade. A synchronous engine is
  usually the right answer.
- **Rows carry ISO-8601 UTC timestamps with an offset.** If your columns are naive, do
  the conversion inside the six primitives.

Cross-process turn leases (`acquire_session_turn_lease` and friends) are deliberately
*not* in the protocol. Every call site probes for them and skips the whole
serialization path when absent, so a host running several processes against one store can
add them to its own class without this protocol changing.

### Reading conversations back — `SessionReader`

Separate from `SessionStore`, and separate on purpose: an agent always knows its own
session id and never enumerates; a dashboard, an audit or a support view needs exactly
the opposite. Folding it in would force every host that only wants an agent to implement
a method for a front end it may not have.

```python
for row in store.list_sessions(limit=20):
    print(row["session_id"], row["message_count"], row["started_at"])
    for message in store.get_messages_as_conversation(row["session_id"]):
        ...
```

Both shipped stores implement it, so a host using either gets it for free.

## Async hosts

`run_conversation` blocks, and it is not going to stop blocking: it is lifted from
upstream, where the agent runs on threads, and re-lifting keeps it that way.

Calling it from a coroutine freezes the whole server for the length of the turn — every
request, not just that one. Two helpers make that unnecessary:

```python
from hermes_core import run_conversation_async, call_host_async

# calling in: the turn runs on a worker thread, the loop keeps serving.
# This binds the host loop for the whole turn, so handlers can use call_host_async
# with no further setup -- you do not wrap this in bind_host_loop.
result = await run_conversation_async(agent, "hola", timeout=120)

# calling out: a sync handler reaching the host's async services
def report_incident(args, *, repo, **_core):
    return tool_result(id=call_host_async(repo.create(args["title"])))
```

`call_host_async` runs the coroutine on **the host's own loop**. That distinction is the
entire point:

- `asyncio.run(coro)` and `registry.register(..., is_async=True)` each run it on a
  brand-new loop in a fresh thread. Anything bound to the host's loop — an asyncpg pool,
  an httpx client, most async database sessions — is then being used from the wrong loop.
  It does not fail immediately, which is what makes it expensive.
- `asyncio.run_coroutine_threadsafe` is correct, but only with the real loop, and it
  deadlocks instantly if reached from the loop thread. `call_host_async` raises
  `HostLoopUnavailable` there instead of hanging.

`bind_host_loop` is for the *other* case only: a host that drives the worker thread
itself (Starlette's `run_in_threadpool`, an existing pool). It must bind **before** the
work leaves the loop thread — a `ContextVar` is copied into a thread when the thread
starts, so a binding made inside the worker never reaches the handler:

```python
with bind_host_loop():
    return await run_in_threadpool(agent.run_conversation, text)
```

**Cancellation.** A cancelled `await` does not stop the turn; Python cannot interrupt an
arbitrary thread. Pass `timeout=` (which calls `agent.interrupt()` and lets the turn
unwind) or call `agent.interrupt()` yourself.

## Agent-to-agent and agent-to-human messaging

`AgentMailbox` is a durable queue for the escalation case: a sales agent that hits a
question it cannot answer hands it to another agent, or to a person, and waits for the
answer to come back.

```python
from hermes_core import MailboxMessage, SqliteMailbox

mailbox = SqliteMailbox("/var/lib/app/mailbox.db")

request_id = mailbox.send(MailboxMessage(
    org="acme",
    sender="ventas",
    recipient="personal",
    subject="¿Queda stock del producto X?",
    context_id="whatsapp:+54911...",      # routes the answer back to the right customer
    payload={"sku": "X-100"},
))

# the personal agent picks it up, decides it needs a human, and asks
mailbox.poll("acme", "personal")
mailbox.mark_input_required("acme", request_id, asked="¿Queda stock del X-100?")

# ... whenever the owner answers, from wherever they answered ...
mailbox.answer("acme", request_id, "Sí, quedan 4.", answered_by="dueño")

# the sales agent collects what came back
for answered in mailbox.answers_for("acme", "ventas"):
    ...
```

Every method takes `org` and is scoped by it. A read outside your organisation reports
the message as **absent**, never as forbidden — the difference matters, because "you may
not see this" still confirms it exists.

State moves through `PENDING → WORKING → INPUT_REQUIRED → COMPLETED` (or `FAILED` /
`CANCELED`). Durable rather than a blocking wait, so the process holding the question can
restart without losing it.

## Testing your integration

The core ships its test doubles as public API, because a host has to be able to test its
own tools and prompts without reaching a real model.

```python
from hermes_core import Script, install_fake_client

client = install_fake_client(
    agent,
    Script().calls(("get_weather", {"city": "Rosario"})).text("Hacen 21 grados."),
)
result = agent.run_conversation("¿qué temperatura hace?")

assert client.call_count == 2
assert client.last_request.tool_names == ["get_weather"]
```

- **Use `install_fake_client`, not `agent.client = ...`.** The loop builds a *per-request*
  client for most providers, so patching the attribute quietly reaches the network — the
  exact failure the helper exists to prevent.
- **Exhausting a script is an error, never a default answer.** If the loop asks for one
  more turn than scripted, it did not stop when you expected, and a fake that answered
  anyway would hide that.
- Pass `drop=StreamDrop(after_chunks=n)` to `text()` or `calls()`, with
  `install_fake_client(agent, script, stream=True)`, to cut a stream short — the shape of
  a connection that dies mid-response.

## Regenerating from upstream

Most of this package is lifted, not written here. `tools/lift.py` holds the manifest, the
import rewrites and the patches, and reproduces the whole core from a fresh upstream
clone:

```bash
python tools/lift.py
```

It fails loudly rather than quietly: a patch whose anchor upstream has changed exits 3
rather than skipping, and it imports every module in a subprocess afterwards, so a
manifest gap is measured rather than inferred. Anything hand-written lives in
`hermes_core/seams/` and `hermes_core/testing/`, which the lift never touches.

If you patch a lifted file directly, add the change to `PATCHES` in `tools/lift.py` in
the same commit — otherwise the next re-lift silently reverts it.
