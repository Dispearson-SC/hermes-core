"""Close the manifest by following what the core actually fails on.

The static analysis got the extraction most of the way, but the last stretch is
dominated by imports written inside function bodies, which no static pass sees until
that branch runs. Chasing them one error at a time is slow and, worse, it is a
judgement-free process being done by hand.

So this drives it: run a probe, read the missing module out of the traceback, add it
to the manifest, re-lift, repeat. It stops on its own when the probe succeeds or when
a round adds nothing.

Two things it will not do. It never adds a capability pack -- browser automation,
sandboxed terminal backends -- because those are deliberately out of the core and a
missing one means a call site needs a patch, not a module. And it never adds anything
outside the packages the extraction covers.

Usage:
    python tools/close_manifest.py <probe.py> [--max-rounds N]
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

CORE = pathlib.Path(__file__).resolve().parent.parent
UPSTREAM = CORE.parent / "Hermes"

_MISSING = re.compile(r"No module named '([\w.]+)'")

# Where a core module came from. Longest prefix first.
_ORIGINS = [
    ("hermes_core.runtime.", "hermes_cli/"),
    ("hermes_core.agent.", "agent/"),
    ("hermes_core.tools.", "tools/"),
    ("hermes_core.providers.", "providers/"),
    ("hermes_core.", ""),
]

# Deliberately excluded from the core: capability packs, and the surfaces this
# extraction leaves behind. A hit here is reported rather than added.
_REFUSED = (
    "hermes_core.tools.browser_tool", "hermes_core.tools.environments",
    "hermes_core.tools.terminal_tool", "hermes_core.agent.browser_",
    "hermes_core.plugins.browser", "hermes_core.agent.auxiliary_client",
    "hermes_core.agent.credential_pool",
)


def upstream_path(dotted: str) -> pathlib.Path | None:
    # A bare first-party root module: the rewrite missed it, usually because it is
    # reached through a string rather than an import statement.
    if "." not in dotted and (UPSTREAM / f"{dotted}.py").is_file():
        return pathlib.Path(f"{dotted}.py")
    if not dotted.startswith("hermes_core"):
        return None
    for prefix, origin in _ORIGINS:
        if dotted.startswith(prefix):
            tail = dotted[len(prefix):].replace(".", "/")
            for candidate in (f"{origin}{tail}.py", f"{origin}{tail}/__init__.py"):
                if (UPSTREAM / candidate).is_file():
                    return pathlib.Path(candidate)
            return None
    return None


def add_to_manifest(rel: str) -> bool:
    lift = CORE / "tools" / "lift.py"
    text = lift.read_text(encoding="utf-8")
    entry = f'    "{rel}",\n'
    if entry in text:
        return False
    anchor = "    # -- tool-call validation"
    lift.write_text(text.replace(anchor, entry + anchor), encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("probe")
    parser.add_argument("--max-rounds", type=int, default=60)
    args = parser.parse_args()

    for round_number in range(1, args.max_rounds + 1):
        result = subprocess.run(
            [sys.executable, args.probe], capture_output=True, text=True, cwd=CORE,
            encoding="utf-8", errors="replace",
        )
        if result.returncode == 0:
            print(result.stdout)
            print(f"probe passes after {round_number - 1} added module(s)")
            return 0

        output = result.stderr + result.stdout
        # The last match, not the first: discovery code suppresses ImportErrors and
        # logs them, so earlier occurrences come from a module that recovered. Only the
        # final one belongs to the traceback that actually stopped the probe. Bare
        # package names are dropped for the same reason -- they are the tail of a
        # suppressed dynamic import, never a module to lift.
        candidates = [
            name for name in _MISSING.findall(output)
            if name not in ("agent", "tools", "providers", "plugins", "hermes_cli")
        ]
        if not candidates:
            print(output[-3000:], file=sys.stderr)
            print("\nfailure is not a missing module -- stopping for a human", file=sys.stderr)
            return 1

        dotted = candidates[-1]
        if dotted.startswith(_REFUSED):
            print(output[-2000:], file=sys.stderr)
            print(
                f"\n{dotted} is deliberately outside the core. The call site needs a "
                f"patch in lift.py, not a module.",
                file=sys.stderr,
            )
            return 2

        rel = upstream_path(dotted)
        if rel is None:
            print(f"no upstream file for {dotted}", file=sys.stderr)
            return 3

        if not add_to_manifest(rel.as_posix()):
            print(f"{rel} is already in the manifest but still missing -- stopping", file=sys.stderr)
            return 4

        print(f"round {round_number}: + {rel.as_posix()}")
        lift = subprocess.run(
            [sys.executable, str(CORE / "tools" / "lift.py")],
            capture_output=True, text=True, cwd=CORE,
            encoding="utf-8", errors="replace",
        )
        if lift.returncode != 0:
            print(lift.stdout[-2000:], lift.stderr[-2000:], file=sys.stderr)
            return 5

    print(f"still failing after {args.max_rounds} rounds", file=sys.stderr)
    return 6


if __name__ == "__main__":
    raise SystemExit(main())
