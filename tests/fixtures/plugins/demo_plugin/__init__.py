"""Test-fixture plugin: proves both halves of the hermes_core plugin contract.

Registers one tool (``demo_greet``) and one ``post_tool_call`` hook that records
every call it observes into this plugin's durable state, so a test running in a
separate ``PluginState`` instance can read back what the hook actually saw.
"""

from hermes_core.tools.registry import tool_result


def _demo_greet(args, **_kwargs):
    name = args.get("name", "world")
    return tool_result(greeting=f"Hello, {name}!")


def register(ctx):
    ctx.register_tool(
        name="demo_greet",
        toolset="demo_plugin_tools",
        schema={
            "name": "demo_greet",
            "description": "Greet someone by name.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
        handler=_demo_greet,
    )

    def _on_post_tool_call(**kwargs):
        observed = ctx.state.get("observed_calls", [])
        observed.append({"tool_name": kwargs.get("tool_name"), "args": kwargs.get("args")})
        ctx.state.set("observed_calls", observed)

    ctx.register_hook("post_tool_call", _on_post_tool_call)
