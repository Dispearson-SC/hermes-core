"""Adversarial fixture: registers two ``post_tool_call`` hooks in order. The first
mutates the ``args`` dict it was handed in place; the second (a distinct callback,
registered after) records exactly what it received. If the mutation is visible to
the second callback, the shared mutable payload lets one observer's hook corrupt
what every other observer of the same event sees."""


def register(ctx):
    def _mutator(**kwargs):
        args = kwargs.get("args")
        if isinstance(args, dict):
            args["tampered_by"] = "hook_mutator_plugin"

    def _observer(**kwargs):
        ctx.state.set("observed_args", kwargs.get("args"))

    ctx.register_hook("post_tool_call", _mutator)
    ctx.register_hook("post_tool_call", _observer)
