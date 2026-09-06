"""Models that expect the system turn under the ``developer`` role.

A one-line constant that upstream keeps in ``agent/prompt_builder.py``, a
module carrying the whole system-prompt assembly. The chat-completions
transport needs only this, so it moves alone.

Extracted verbatim from upstream ``agent/prompt_builder.py``; edit there and re-run
the lift rather than editing this file.
"""

from __future__ import annotations


DEVELOPER_ROLE_MODELS = ("gpt-5", "codex")
