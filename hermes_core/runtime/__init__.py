"""Services that lived in upstream's CLI package and the core genuinely needs.

Written by tools/lift.py, not lifted: upstream's `hermes_cli/__init__.py` carries
the package version, and seven lifted modules import it from here.
"""

from hermes_core import __version__ as __version__
