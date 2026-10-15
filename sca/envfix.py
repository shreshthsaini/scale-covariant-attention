"""In-memory stand-ins for zero-byte modules missing from a scratch-purged site-packages.

The uv env lives on scratch, and its empty files (for example torch/_dynamo/backends/__init__.py,
torch/_thread_safe_fork.py, triton/backends/nvidia/__init__.py) have been removed. Plain imports
still work, but torch.compile does not: it imports the empty modules and reads package __file__.
install() registers a meta-path finder that serves every missing zero-byte .py file of the given
distributions (listed in their RECORD) as an empty module with its original path, and repairs
namespace packages that were imported before it ran. Nothing is written to disk; on an intact
install the table is empty and this does nothing. The real fix is reinstalling torch and triton.
"""

import importlib.abc
import importlib.machinery
from importlib.metadata import PackageNotFoundError, distribution
import os
import sys

_STATE = {}


class _EmptyLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        pass


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, table):
        self.table = table

    def find_spec(self, name, path=None, target=None):
        hit = self.table.get(name)
        if hit is None:
            return None
        file, is_pkg = hit
        spec = importlib.machinery.ModuleSpec(name, _EmptyLoader(), origin=file, is_package=is_pkg)
        if is_pkg:
            spec.submodule_search_locations = [os.path.dirname(file)]
        spec.has_location = True
        return spec


def missing_empty_modules(dists=("torch", "triton")):
    """{module name: (path, is_package)} for zero-byte .py files listed in RECORD but absent."""
    table = {}
    for d in dists:
        try:
            files = distribution(d).files or []
        except PackageNotFoundError:
            continue
        for f in files:
            if f.suffix != ".py" or f.size != 0:
                continue
            path = str(f.locate())
            if os.path.exists(path):
                continue
            parts = list(f.with_suffix("").parts)
            is_pkg = parts[-1] == "__init__"
            if is_pkg:
                parts = parts[:-1]
            table[".".join(parts)] = (path, is_pkg)
    return table


def install(dists=("torch", "triton")):
    if "table" in _STATE:
        return _STATE["table"]
    table = missing_empty_modules(dists)
    _STATE["table"] = table
    if not table:
        return table
    for name, (path, is_pkg) in table.items():
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "__file__", None) is None:
            mod.__file__ = path  # namespace package imported before the finder existed
    sys.meta_path.insert(0, _Finder(table))
    return table
