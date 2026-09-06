from hermes_core.tools.registry import tool_result


def _probe(args, **_kwargs):
    return tool_result(ok=True)


def register(ctx):
    ctx.register_tool(
        name="mismatch_probe",
        toolset="mismatch_tools",
        schema={
            "name": "mismatch_probe",
            "description": "Proves the plugin loaded despite the name/directory mismatch.",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=_probe,
    )
