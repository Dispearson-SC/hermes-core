"""Never reached: plugin.yaml fails to parse before this module is imported."""


def register(ctx):
    raise AssertionError("malformed_plugin's register() must never run")
