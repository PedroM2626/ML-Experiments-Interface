"""MLine src package (namespace-merged).

Both MLine/ and `sagemaker based/` historically expose a top-level `src`
package. To allow running the full test-suite in a single pytest process
(from the repo root), both `src/__init__.py` files extend __path__ so the
two directories merge instead of the first one on sys.path shadowing the
other. Each app still runs standalone (single dir on path), so behaviour
there is unchanged.
"""
from pkgutil import extend_path as _extend_path

__path__ = _extend_path(__path__, __name__)
