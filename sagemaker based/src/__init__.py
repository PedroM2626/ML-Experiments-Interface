"""AutoML Studio src package (namespace-merged).

See MLine/src/__init__.py: both projects expose a top-level `src` package,
so both __init__ files use pkgutil.extend_path to merge instead of shadow.
"""
from pkgutil import extend_path as _extend_path

__path__ = _extend_path(__path__, __name__)
