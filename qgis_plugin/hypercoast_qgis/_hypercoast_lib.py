"""
Helper to import the *external* `hypercoast` Python library from within the QGIS plugin.

Why this exists:
- When the plugin folder is renamed to "hypercoast" by the official QGIS plugin repository,
  importing `hypercoast` from plugin code can resolve to the plugin package itself,
  shadowing the external `hypercoast` library.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Optional

import importlib
import importlib.metadata
import importlib.util
import sys


_CACHED: Optional[ModuleType] = None


def _is_module_from_dir(mod: ModuleType, directory: Path) -> bool:
    """
    Returns True if the given module's __file__ is located within the specified directory.
    """
    try:
        mod_file = getattr(mod, "__file__", None)
        if not mod_file:
            return False
        return Path(mod_file).resolve().is_relative_to(directory.resolve())
    except (ValueError, OSError):
        return False


def _import_hypercoast_without_plugin_shadow(plugin_pkg_dir: Path) -> Optional[ModuleType]:
    """
    Try to import `hypercoast` while temporarily removing the plugin path from sys.path.
    """
    plugin_parent = plugin_pkg_dir.parent
    orig_sys_path = list(sys.path)
    orig_mod = sys.modules.get("hypercoast")

    try:
        sys.path = [p for p in sys.path if Path(p).resolve() != plugin_parent.resolve()]
        if "hypercoast" in sys.modules:
            del sys.modules["hypercoast"]

        imported = importlib.import_module("hypercoast")
        if _is_module_from_dir(imported, plugin_pkg_dir):
            if orig_mod is not None:
                sys.modules["hypercoast"] = orig_mod
            return None
        return imported
    except (ImportError, ModuleNotFoundError):
        return None
    finally:
        sys.path = orig_sys_path
        if orig_mod is not None:
            sys.modules["hypercoast"] = orig_mod


def _load_external_hypercoast_from_dist(dist_name: str) -> Optional[ModuleType]:
    """
    Load the external hypercoast package from an installed distribution.

    We load it under an alias module name (hypercoast_external) to avoid conflicting with
    the plugin package (which may also be named `hypercoast`).
    """
    try:
        dist = importlib.metadata.distribution(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None

    files = list(dist.files or [])
    init_rel = None
    for f in files:
        if str(f).replace("\\", "/").endswith("hypercoast/__init__.py"):
            init_rel = f
            break
    if init_rel is None:
        return None

    init_path = Path(dist.locate_file(init_rel)).resolve()
    pkg_dir = init_path.parent

    alias_name = "hypercoast_external"
    existing = sys.modules.get(alias_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(
        alias_name,
        str(init_path),
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[alias_name] = module
    spec.loader.exec_module(module)
    return module


def get_hypercoast() -> ModuleType:
    """
    Return the external `hypercoast` library module.
    """
    global _CACHED
    if _CACHED is not None:
        return _CACHED

    plugin_dir = Path(__file__).resolve().parent

    # Normal import first (works when plugin package isn't named `hypercoast`)
    try:
        imported = importlib.import_module("hypercoast")
        if not _is_module_from_dir(imported, plugin_dir):
            _CACHED = imported
            return imported
    except (ImportError, ModuleNotFoundError, AttributeError):
        pass

    imported = _import_hypercoast_without_plugin_shadow(plugin_dir)
    if imported is not None:
        _CACHED = imported
        return imported

    # Fallback: load from installed distribution metadata
    for dist_name in ("hypercoast",):
        ext = _load_external_hypercoast_from_dist(dist_name)
        if ext is not None:
            _CACHED = ext
            return ext

    py = getattr(sys, "executable", "python")
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    raise ImportError(
        "HyperCoast plugin could not import the external 'hypercoast' library because the plugin "
        "package name shadows it.\n\n"
        f"QGIS Python:\n  executable: {py}\n  version: {ver}\n\n"
        "Fix: install the HyperCoast Python package into *this same Python environment*.\n"
        "In QGIS, open Python Console and run:\n"
        "  import sys, subprocess\n"
        "  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'hypercoast'])\n"
    )

