"""Stand-in for ``hermes_core.runtime.plugin_compat``, a module ``tools/lift.py``
has not lifted yet.

Upstream's ``hermes_cli/plugin_compat.py`` is absent from ``tools/lift.py``'s
``MANIFEST`` (checked directly against the script). But the lifted
``hermes_core/runtime/plugins_loader.py`` calls it unconditionally, for every
non-portable plugin load::

    from hermes_core.runtime.plugin_compat import disable_reason
    reason = disable_reason(manifest)

Without this module present under exactly that name, no directory or entry-point
plugin can load at all -- a ``ModuleNotFoundError`` on the very first one. This is a
lift-manifest gap, not something to patch in the lifted loader itself (see
``hermes_core/runtime/plugins_loader.py`` around line 283, and the project rule
against hand-editing lifted files while ``tools/lift.py`` is owned elsewhere).

Upstream's real ``disable_reason()`` flags plugins that still import pre-decomposition
paths from before a large internal refactor -- a migration concern with no "before"
in a freshly extracted core, so this always returns ``None`` (never disable). A
plugin that is otherwise broken still gets caught and isolated the normal way, by
``register()`` raising.

``install_if_missing()`` is called once from ``hermes_core/seams/__init__.py`` (which
every lifted plugin module reaches before this matters, via
``from hermes_core.seams.paths import ...``). It only installs this shim into
``sys.modules`` when the real module does not exist on disk, so a future lift that
adds ``hermes_core/runtime/plugin_compat.py`` for real is picked up unshadowed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Optional

_MODULE_NAME = "hermes_core.runtime.plugin_compat"


def disable_reason(_manifest: Any) -> Optional[str]:
    """Never disables a plugin here -- see module docstring."""
    return None


def install_if_missing() -> None:
    """Register this shim as ``hermes_core.runtime.plugin_compat`` unless the real
    file already exists (a future lift landed it) or something already imported it."""
    if _MODULE_NAME in sys.modules:
        return
    real_path = Path(__file__).resolve().parent.parent / "runtime" / "plugin_compat.py"
    if real_path.exists():
        return
    shim = types.ModuleType(_MODULE_NAME)
    shim.disable_reason = disable_reason
    sys.modules[_MODULE_NAME] = shim
