"""Shared fixtures.

Deliberately small. Upstream's root conftest runs to roughly 1800 lines, most of it
undoing side effects the code has on import -- redirecting a home directory,
neutralising a keychain, guarding against writes to the user's real profile. A core
that reads its configuration from an injected object and registers nothing on import
needs almost none of that, and the size of this file is one way to notice if that
stops being true.
"""

from types import SimpleNamespace

import pytest


@pytest.fixture
def openai_response():
    """Build a response shaped like the OpenAI SDK's, for transport normalisation.

    A namespace tree rather than the SDK's models: the transport reads attributes, so
    this exercises the same path while keeping the test readable and independent of
    the SDK's constructor requirements.
    """

    def build(
        *,
        content=None,
        tool_calls=None,
        finish_reason=None,
        prompt_tokens=0,
        completion_tokens=0,
    ):
        calls = [
            SimpleNamespace(
                id=call_id,
                type="function",
                function=SimpleNamespace(name=name, arguments=arguments),
            )
            for call_id, name, arguments in (tool_calls or [])
        ]
        message = SimpleNamespace(
            role="assistant",
            content=content,
            tool_calls=calls or None,
            reasoning_content=None,
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=message,
                    finish_reason=finish_reason or ("tool_calls" if calls else "stop"),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            model="test-model",
        )

    return build
