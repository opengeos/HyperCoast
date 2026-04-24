"""Tests for :func:`hypercoast.read_tanager` using fabricated HDF5 fixtures.

These tests do not require a real Planet Tanager file and do not touch the
network. They build small HDF5 files that mirror the layout of the four
Planet Tanager product variants and assert that :func:`read_tanager` recovers
the correct canonical data variable name and wavelength values.
"""

import os
import tempfile
import unittest
import warnings
from unittest import mock

import h5py
import numpy as np

import hypercoast


def _write_hdfeos_cube(
    path,
    cube_name,
    n_bands=200,
    height=10,
    width=8,
    include_wavelength=True,
    scale_factor=None,
    add_offset=None,
):
    """Create a minimal HDF5 file that mimics a Tanager HDFEOS SWATHS product.

    Args:
        path (str): Destination path for the HDF5 file.
        cube_name (str): Name of the data cube (for example ``toa_radiance`` or
            ``surface_reflectance``).
        n_bands (int): Number of spectral bands to fabricate.
        height (int): Number of along-track rows.
        width (int): Number of cross-track columns.
        include_wavelength (bool): Whether to write a ``Wavelength`` dataset.
        scale_factor (float, optional): If given, stored as a ``scale_factor``
            attribute on the cube.
        add_offset (float, optional): If given, stored as an ``add_offset``
            attribute on the cube.
    """
    rng = np.random.default_rng(42)
    with h5py.File(path, "w") as f:
        df = f.create_group("HDFEOS/SWATHS/HYP/Data Fields")
        cube = df.create_dataset(
            cube_name,
            data=rng.random((n_bands, height, width), dtype=np.float32),
        )
        if scale_factor is not None:
            cube.attrs["scale_factor"] = scale_factor
        if add_offset is not None:
            cube.attrs["add_offset"] = add_offset
        if include_wavelength:
            df.create_dataset(
                "Wavelength",
                data=np.linspace(400.0, 2500.0, n_bands, dtype=np.float32),
            )
            df.create_dataset("FWHM", data=np.full(n_bands, 10.0, dtype=np.float32))
        gf = f.create_group("HDFEOS/SWATHS/HYP/Geolocation Fields")
        lat = np.tile(np.linspace(0.0, 1.0, height, dtype=np.float32), (width, 1)).T
        lon = np.tile(np.linspace(10.0, 11.0, width, dtype=np.float32), (height, 1))
        gf.create_dataset("Latitude", data=lat)
        gf.create_dataset("Longitude", data=lon)


class TestReadTanager(unittest.TestCase):
    """Exercise the Tanager reader against fabricated HDF5 fixtures."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="hc_tanager_test_")

    def tearDown(self):
        for name in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, name))
        os.rmdir(self.tmpdir)

    def test_basic_radiance_product(self):
        path = os.path.join(self.tmpdir, "basic_radiance.h5")
        _write_hdfeos_cube(path, "toa_radiance", n_bands=200)

        with mock.patch("hypercoast.tanager.requests.get") as mocked_get:
            ds = hypercoast.read_tanager(path)

        mocked_get.assert_not_called()
        self.assertIn("toa_radiance", ds.data_vars)
        self.assertEqual(ds.attrs["product"], "basic_radiance")
        self.assertEqual(ds.attrs["data_var"], "toa_radiance")
        wl = ds.coords["wavelength"].values
        self.assertEqual(wl.size, 200)
        self.assertAlmostEqual(float(wl[0]), 400.0, places=2)
        self.assertAlmostEqual(float(wl[-1]), 2500.0, places=2)

    def test_basic_sr_product_has_alias(self):
        path = os.path.join(self.tmpdir, "basic_sr.h5")
        _write_hdfeos_cube(
            path, "surface_reflectance", n_bands=180, scale_factor=0.0001
        )

        with mock.patch("hypercoast.tanager.requests.get") as mocked_get:
            ds = hypercoast.read_tanager(path)

        mocked_get.assert_not_called()
        self.assertIn("surface_reflectance", ds.data_vars)
        self.assertIn("toa_radiance", ds.data_vars)
        np.testing.assert_array_equal(
            ds["surface_reflectance"].values, ds["toa_radiance"].values
        )
        self.assertEqual(ds.attrs["product"], "basic_sr")
        self.assertEqual(ds.attrs["data_var"], "surface_reflectance")

    def test_ortho_sr_product(self):
        path = os.path.join(self.tmpdir, "ortho_sr.h5")
        _write_hdfeos_cube(path, "ortho_surface_reflectance", n_bands=150)

        with mock.patch("hypercoast.tanager.requests.get") as mocked_get:
            ds = hypercoast.read_tanager(path)

        mocked_get.assert_not_called()
        self.assertEqual(ds.attrs["product"], "ortho_sr")
        self.assertIn("surface_reflectance", ds.data_vars)
        self.assertIn("toa_radiance", ds.data_vars)

    def test_missing_wavelength_warns_and_indexes(self):
        path = os.path.join(self.tmpdir, "no_wavelength.h5")
        _write_hdfeos_cube(path, "toa_radiance", n_bands=120, include_wavelength=False)

        with mock.patch("hypercoast.tanager.requests.get") as mocked_get:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ds = hypercoast.read_tanager(path)

        mocked_get.assert_not_called()
        self.assertTrue(
            any(issubclass(w.category, UserWarning) for w in caught),
            "Expected UserWarning when no wavelength metadata is available",
        )
        wl = ds.coords["wavelength"].values
        np.testing.assert_array_equal(wl, np.arange(120, dtype=float))

    def test_explicit_wavelengths_kwarg_skips_network(self):
        path = os.path.join(self.tmpdir, "no_wavelength.h5")
        _write_hdfeos_cube(path, "toa_radiance", n_bands=100, include_wavelength=False)
        supplied = np.linspace(450.0, 2400.0, 100)

        with mock.patch("hypercoast.tanager.requests.get") as mocked_get:
            ds = hypercoast.read_tanager(path, wavelengths=supplied)

        mocked_get.assert_not_called()
        np.testing.assert_allclose(ds.coords["wavelength"].values, supplied)


if __name__ == "__main__":
    unittest.main()
