"""Test-fixture plugin: raises during register() to prove one bad plugin cannot
stop discovery of the others."""


def register(ctx):
    raise RuntimeError("broken_plugin deliberately fails to load")
