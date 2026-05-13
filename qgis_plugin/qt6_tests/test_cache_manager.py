"""Tests for generated raster cache path helpers."""

import os

from hypercoast_qgis.cache_manager import (
    create_generated_raster_path,
    generated_raster_cache_dir,
)


class _Project:
    """Tiny QgsProject-like object for cache path tests."""

    def __init__(self, filename):
        """Initialize the fake project.

        Args:
            filename: Project filename returned by ``fileName``.
        """
        self._filename = filename

    def fileName(self):
        """Return the fake project filename."""
        return self._filename


def test_saved_project_uses_user_cache_directory(tmp_path, monkeypatch):
    """Saved projects should use the user-level HyperCoast cache."""
    monkeypatch.setenv("HOME", str(tmp_path))
    project_path = tmp_path / "example.qgz"
    project_path.write_text("", encoding="utf-8")
    project = _Project(str(project_path))

    cache_dir = generated_raster_cache_dir(project)
    raster_path = create_generated_raster_path("Layer One", "rgb", project)

    assert cache_dir == str(tmp_path / ".qgis_hypercoast" / "cache")
    assert os.path.isdir(cache_dir)
    assert os.path.dirname(raster_path) == cache_dir
    assert os.path.basename(raster_path).startswith("Layer_One_rgb_")


def test_unsaved_project_uses_user_cache_directory(tmp_path, monkeypatch):
    """Unsaved projects should use the user-level HyperCoast cache."""
    monkeypatch.setenv("HOME", str(tmp_path))

    cache_dir = generated_raster_cache_dir(_Project(""))

    assert cache_dir == str(tmp_path / ".qgis_hypercoast" / "cache")
    assert os.path.isdir(cache_dir)
