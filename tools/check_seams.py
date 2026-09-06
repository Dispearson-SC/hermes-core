"""Report names the lifted code imports from a seam that the seam does not define.

The seams are written by hand and deliberately expose a smaller surface than the
upstream modules they replace. Most of the difference is intentional; the rest is a
gap, and the difference between the two is only visible once something asks.

Finding them one failed run at a time is slow, because each missing name hides the
next. This asks the question for every name at once.
"""

from __future__ import annotations

import importlib
import pathlib
import re
import sys

CORE = pathlib.Path(__file__).resolve().parent.parent
_IMPORT = re.compile(
    r"from (hermes_core\.(?:seams|runtime)\.[A-Za-z_][\w.]*) import (\([^)]*\)|[^\n(]*)"
)


def main() -> int:
    wanted: dict[str, set[str]] = {}
    for path in (CORE / "hermes_core").rglob("*.py"):
        source = path.read_text(encoding="utf-8", errors="replace")
        for match in _IMPORT.finditer(source):
            module, names = match.group(1), match.group(2)
            for raw in names.strip("()").replace("\n", " ").split(","):
                name = raw.strip().split(" as ")[0].strip().strip("\\").strip()
                if name and not name.startswith("#") and name != "*":
                    wanted.setdefault(module, set()).add(name)

    missing: list[str] = []
    for module, names in sorted(wanted.items()):
        try:
            loaded = importlib.import_module(module)
        except Exception as exc:
            print(f"could not import {module}: {exc}", file=sys.stderr)
            continue
        for name in sorted(names):
            if not hasattr(loaded, name):
                missing.append(f"{module}.{name}")

    if not missing:
        print("every name the lifted code imports from a seam exists")
        return 0

    print(f"missing from the seams ({len(missing)}):")
    for entry in missing:
        print(f"  {entry}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
