"""Adversarial fixture: registers a real tool, then raises before ``register()``
returns. Proves the half-finished registration is unwound rather than surviving as a
zombie in the ordinary tool registry."""

from hermes_core.tools.registry import tool_result


def _zombie(args, **_kwargs):
    return tool_result(should_never_be_reachable=True)


def register(ctx):
    ctx.register_tool(
        name="zombie_tool",
        toolset="zombie_tool_tools",
        schema={
            "name": "zombie_tool",
            "description": "Registered just before register() blows up.",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=_zombie,
    )
    raise RuntimeError("zombie_tool_plugin deliberately fails after registering a tool")
