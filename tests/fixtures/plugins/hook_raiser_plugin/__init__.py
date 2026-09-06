"""Adversarial fixture: a ``post_tool_call`` hook that records it ran, then always
raises. Proves one hook callback raising cannot take a turn -- or a sibling
callback -- down with it."""


def register(ctx):
    def _on_post_tool_call(**kwargs):
        seen = ctx.state.get("saw_calls", 0)
        ctx.state.set("saw_calls", seen + 1)
        raise RuntimeError("hook_raiser_plugin deliberately raises from post_tool_call")

    ctx.register_hook("post_tool_call", _on_post_tool_call)
