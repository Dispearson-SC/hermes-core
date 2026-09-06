from hermes_core.tools.registry import tool_result


def _shared(args, **_kwargs):
    return tool_result(owner="dup_tool_plugin_a")


def register(ctx):
    ctx.register_tool(
        name="shared_tool",
        toolset="dup_tool_a_tools",
        schema={
            "name": "shared_tool",
            "description": "Claimed by dup_tool_plugin_a.",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=_shared,
    )
