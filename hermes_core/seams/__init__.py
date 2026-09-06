"""Package init for the seams -- kept minimal, but this one side effect is load-bearing.

``hermes_core.runtime.plugin_compat`` is referenced by lifted plugin-loading code but
was never included in ``tools/lift.py``'s manifest (see ``plugin_compat.py`` in this
package for the full story); without a stand-in, no plugin can load at all. Every
lifted plugin module reaches this package via ``from hermes_core.seams.paths import
...`` before it needs that name, so installing the stand-in here -- once, and only
when the real module is not already present -- is enough.
"""

from hermes_core.seams import plugin_compat as _plugin_compat

_plugin_compat.install_if_missing()
