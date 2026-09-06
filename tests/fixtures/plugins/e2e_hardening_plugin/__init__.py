"""Adversarial fixture for a real end-to-end turn:

* ``fails_tool`` always raises inside its handler.
* ``returns_bad_shape_tool`` returns a raw Python dict instead of the JSON-string
  contract ``tool_result()`` produces -- the "non-serializable to the tool-result
  contract" shape a careless plugin author would ship.
* A ``post_tool_call`` hook records every call it observes (name, status, error
  info) into durable state, so the test can confirm the hook still sees a failed
  plugin tool call without the turn going down.
"""


def _fails_tool(args, **_kwargs):
    raise RuntimeError("fails_tool deliberately raises")


def _returns_bad_shape_tool(args, **_kwargs):
    # Not a str, not the multimodal envelope -- violates the tool-result contract.
    return {"this": "is not JSON-encoded, and not the multimodal envelope either"}


def register(ctx):
    ctx.register_tool(
        name="fails_tool",
        toolset="e2e_hardening_tools",
        schema={
            "name": "fails_tool",
            "description": "Always raises.",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=_fails_tool,
    )
    ctx.register_tool(
        name="returns_bad_shape_tool",
        toolset="e2e_hardening_tools",
        schema={
            "name": "returns_bad_shape_tool",
            "description": "Returns a raw dict instead of a JSON string.",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=_returns_bad_shape_tool,
    )

    def _on_post_tool_call(**kwargs):
        observed = ctx.state.get("observed", [])
        observed.append({
            "tool_name": kwargs.get("tool_name"),
            "status": kwargs.get("status"),
            "error_type": kwargs.get("error_type"),
            "result": kwargs.get("result"),
        })
        ctx.state.set("observed", observed)

    ctx.register_hook("post_tool_call", _on_post_tool_call)
