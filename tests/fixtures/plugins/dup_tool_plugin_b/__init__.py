from hermes_core.tools.registry import tool_result


def _shared(args, **_kwargs):
    return tool_result(owner="dup_tool_plugin_b")


def register(ctx):
    ctx.register_tool(
        name="shared_tool",
        toolset="dup_tool_b_tools",
        schema={
            "name": "shared_tool",
            "description": "Also claimed by dup_tool_plugin_b.",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=_shared,
    )
