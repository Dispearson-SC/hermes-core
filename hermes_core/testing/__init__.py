"""Test doubles shipped with the core.

These are part of the public surface, not test-only helpers kept behind the package
boundary. A host embedding this core needs to test its own tools and prompts without
reaching a real model, and it should not have to build the double itself -- having to
do exactly that, roughly 3,800 times, is upstream's most costly testing gap.
"""

from hermes_core.testing.fake_client import FakeOpenAIClient, ScriptedStream, install_fake_client
from hermes_core.testing.fake_host import FakeAssistantMessage, FakeTurnHost
from hermes_core.testing.fake_provider import (
    FakeProvider,
    RecordedRequest,
    Script,
    ScriptExhausted,
    StreamDrop,
)

__all__ = [
    "Script",
    "FakeProvider",
    "FakeOpenAIClient",
    "ScriptedStream",
    "install_fake_client",
    "FakeTurnHost",
    "FakeAssistantMessage",
    "RecordedRequest",
    "ScriptExhausted",
    "StreamDrop",
]
