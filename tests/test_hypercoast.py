#!/usr/bin/env python

"""Tests for `hypercoast` package."""

import sys
import types
import unittest

import hypercoast
import numpy as np
import pytest


class TestHypercoast(unittest.TestCase):
    """Tests for `hypercoast` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""


class _FakeImageData:
    """Small PyVista ImageData test double."""

    def __init__(self, dimensions=None):
        """Initialize image data.

        Args:
            dimensions: Optional grid dimensions.
        """
        self.dimensions = dimensions or (1, 1, 1)
        self.origin = (0, 0, 0)
        self.spacing = (1, 1, 1)
        self.point_data = {}

    @property
    def bounds(self):
        """Return PyVista-like grid bounds."""
        x_max = self.origin[0] + (self.dimensions[0] - 1) * self.spacing[0]
        y_max = self.origin[1] + (self.dimensions[1] - 1) * self.spacing[1]
        z_max = self.origin[2] + (self.dimensions[2] - 1) * self.spacing[2]
        return (self.origin[0], x_max, self.origin[1], y_max, self.origin[2], z_max)


class _FakePlotter:
    """Small PyVista Plotter test double."""

    def __init__(self, **kwargs):
        """Initialize the plotter.

        Args:
            **kwargs: Ignored plotter keyword arguments.
        """
        self.meshes = []

    def add_mesh(self, mesh, **kwargs):
        """Record a mesh.

        Args:
            mesh: Mesh added to the plotter.
            **kwargs: Ignored mesh keyword arguments.
        """
        self.meshes.append(mesh)

    def show_axes(self):
        """Ignore axes display calls."""


def test_image_cube_offsets_rgb_overlay(monkeypatch):
    """RGB overlay should be slightly above the cube top face."""
    xr = pytest.importorskip("xarray")
    fake_pyvista = types.SimpleNamespace(ImageData=_FakeImageData, Plotter=_FakePlotter)
    monkeypatch.setitem(sys.modules, "pyvista", fake_pyvista)
    dataset = xr.Dataset(
        {
            "reflectance": (
                ("y", "x", "wavelength"),
                np.ones((4, 5, 4), dtype="float32"),
            )
        },
        coords={"wavelength": [450.0, 550.0, 650.0, 750.0]},
    )

    plotter = hypercoast.image_cube(
        dataset,
        variable="reflectance",
        rgb_wavelengths=[650.0, 550.0, 450.0],
    )

    cube_mesh, rgb_mesh = plotter.meshes
    assert rgb_mesh.origin[2] > cube_mesh.bounds[5]
