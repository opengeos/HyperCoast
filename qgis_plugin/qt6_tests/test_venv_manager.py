"""Tests for QGIS plugin virtual environment runtime integration."""

import sys
import types

from hypercoast_qgis.core import venv_manager


def test_ensure_venv_packages_available_refreshes_matplotlib(monkeypatch, tmp_path):
    """Venv activation should clear stale Matplotlib modules."""
    site_packages = str(tmp_path / "site-packages")
    refreshed_modules = []

    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(venv_manager, "using_conda_env_with_deps", lambda _: False)
    monkeypatch.setattr(venv_manager, "venv_exists", lambda: True)
    monkeypatch.setattr(venv_manager, "get_venv_site_packages", lambda: site_packages)
    monkeypatch.setattr(venv_manager, "_configure_windows_dll_paths", lambda _: None)
    monkeypatch.setattr(venv_manager, "_preload_venv_dlls", lambda _: None)
    monkeypatch.setattr(venv_manager, "_configure_proj_data", lambda _: None)
    monkeypatch.setattr(venv_manager, "_refresh_module", refreshed_modules.append)
    monkeypatch.setattr(venv_manager.platform, "system", lambda: "Linux")

    def fake_import_module(module_name):
        """Return a minimal imported module for runtime diagnostics."""
        module = types.ModuleType(module_name)
        module.__version__ = "1.0"
        module.__file__ = f"{site_packages}/{module_name}/__init__.py"
        return module

    monkeypatch.setattr(venv_manager.importlib, "import_module", fake_import_module)

    assert venv_manager.ensure_venv_packages_available("/tmp/plugin")
    assert "matplotlib" in refreshed_modules
