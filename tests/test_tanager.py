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
import pytest

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

    def test_fwhm_attribute_in_nm_is_not_scaled(self):
        path = os.path.join(self.tmpdir, "fwhm_attr.h5")
        _write_hdfeos_cube(path, "toa_radiance", n_bands=100, include_wavelength=True)
        with h5py.File(path, "a") as f:
            cube = f["HDFEOS/SWATHS/HYP/Data Fields/toa_radiance"]
            del f["HDFEOS/SWATHS/HYP/Data Fields/FWHM"]
            cube.attrs["fwhm"] = np.full(100, 6.5, dtype=np.float32)
            cube.attrs["fwhm_units"] = "nm"

        ds = hypercoast.read_tanager(path)

        np.testing.assert_allclose(ds.coords["fwhm"].values, np.full(100, 6.5))

    def test_ortho_grid_product_builds_lat_lon(self):
        pytest.importorskip("pyproj")

        path = os.path.join(self.tmpdir, "ortho_grid.h5")
        n_bands = 100
        height = 4
        width = 3
        with h5py.File(path, "w") as f:
            df = f.create_group("HDFEOS/GRIDS/HYP/Data Fields")
            cube = df.create_dataset(
                "toa_radiance",
                data=np.ones((n_bands, height, width), dtype=np.float32),
            )
            cube.attrs["wavelengths"] = np.linspace(400, 900, n_bands)
            cube.attrs["fwhm"] = np.full(n_bands, 6.0, dtype=np.float32)
            cube.attrs["fwhm_units"] = "nm"
            grid = f["HDFEOS/GRIDS/HYP"]
            grid.attrs["epsg_code"] = 32610
            info = f.create_group("HDFEOS INFORMATION")
            info.create_dataset(
                "StructMetadata.0",
                data=(
                    b"GROUP=GridStructure\n"
                    b"GROUP=GRID_1\n"
                    b"XDim=3\n"
                    b"YDim=4\n"
                    b"UpperLeftPointMtrs=(548580.00,4207350.00)\n"
                    b"LowerRightMtrs=(574140.00,4181340.00)\n"
                    b"Projection=HE5_GCTP_UTM\n"
                    b"ZoneCode=10\n"
                    b"END_GROUP=GRID_1\n"
                    b"END_GROUP=GridStructure\n"
                ),
            )

        ds = hypercoast.read_tanager(path, bands=[0, 1, 2])

        self.assertEqual(ds.attrs["product"], "ortho_radiance")
        self.assertEqual(ds.latitude.shape, (height, width))
        self.assertEqual(ds.longitude.shape, (height, width))
        self.assertTrue(np.isfinite(ds.latitude.values).all())
        self.assertTrue(np.isfinite(ds.longitude.values).all())
        np.testing.assert_allclose(ds.coords["fwhm"].values, np.full(3, 6.0))


class _MockResponse:
    """Minimal requests response for mocked STAC calls."""

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class TestTanagerStac(unittest.TestCase):
    """Exercise Tanager STAC search and download helpers without network I/O."""

    def test_stac_wavelengths_ignore_non_spectral_bands(self):
        item = {
            "assets": {
                "basic_radiance_hdf5": {
                    "eo:bands": [
                        {"name": "beta_cloud_mask"},
                        {
                            "name": "toa_radiance_B000",
                            "center_wavelength": 0.4,
                            "full_width_half_max": 0.006,
                        },
                        {
                            "name": "toa_radiance_B001",
                            "center_wavelength": 0.405,
                            "full_width_half_max": 0.0061,
                        },
                        {"name": "Latitude"},
                    ]
                }
            }
        }

        with mock.patch(
            "hypercoast.tanager.requests.get", return_value=_MockResponse(item)
        ):
            wl, fwhm = hypercoast.tanager._read_wavelengths_from_stac(
                "https://example.com/item.json", "basic_radiance_hdf5"
            )

        np.testing.assert_allclose(wl, [400.0, 405.0])
        np.testing.assert_allclose(fwhm, [6.0, 6.1])

    def test_search_tanager_filters_catalog_items(self):
        catalog = {
            "links": [
                {
                    "rel": "child",
                    "href": "https://www.planet.com/data/stac/tanager-core-imagery/coastal-water-bodies/collection.json",
                }
            ]
        }
        collection = {
            "id": "coastal-water-bodies",
            "title": "Coastal and Water Bodies",
            "links": [
                {"rel": "item", "href": "https://example.com/item-1.json"},
                {"rel": "item", "href": "https://example.com/item-2.json"},
            ],
        }
        item_1 = {
            "id": "item-1",
            "bbox": [0, 0, 1, 1],
            "properties": {
                "datetime": "2025-01-15T00:00:00Z",
                "title": "Cloudy Bay scene",
                "cloud_percent": 80,
            },
        }
        item_2 = {
            "id": "item-2",
            "bbox": [0, 0, 1, 1],
            "properties": {
                "datetime": "2025-01-20T00:00:00Z",
                "title": "Clear Bay scene",
                "cloud_percent": 5,
            },
        }

        responses = [
            _MockResponse(obj) for obj in (catalog, collection, item_1, item_2)
        ]
        with mock.patch("hypercoast.tanager.requests.get", side_effect=responses):
            results = hypercoast.search_tanager(
                bbox=[0.5, 0.5, 2, 2],
                temporal=("2025-01-01", "2025-01-31"),
                collections="coastal-water-bodies",
                query="Bay",
                cloud_percent=10,
                count=1,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "item-2")
        self.assertEqual(results[0]["_stac_url"], "https://example.com/item-2.json")
        self.assertEqual(results[0]["_collection_title"], "Coastal and Water Bodies")

    def test_download_tanager_downloads_selected_asset(self):
        item = {
            "id": "item-1",
            "assets": {
                "ortho_radiance_hdf5": {
                    "href": "https://example.com/20250101_ortho_radiance_hdf5.h5"
                }
            },
        }

        with mock.patch(
            "hypercoast.tanager.download_file", return_value="/tmp/out.h5"
        ) as mocked_download:
            result = hypercoast.download_tanager(
                item, out_dir="/tmp", quiet=True, overwrite=True
            )

        self.assertEqual(result, ["/tmp/out.h5"])
        mocked_download.assert_called_once_with(
            "https://example.com/20250101_ortho_radiance_hdf5.h5",
            output="/tmp/20250101_ortho_radiance_hdf5.h5",
            quiet=True,
            overwrite=True,
            unzip=False,
        )

    def test_tanager_footprints_deduplicates_collections(self):
        pytest.importorskip("geopandas")
        catalog = {
            "links": [
                {"rel": "child", "href": "https://example.com/a/collection.json"},
                {"rel": "child", "href": "https://example.com/b/collection.json"},
            ]
        }
        collection_a = {
            "id": "a",
            "title": "Collection A",
            "links": [{"rel": "item", "href": "https://example.com/a/scene.json"}],
        }
        collection_b = {
            "id": "b",
            "title": "Collection B",
            "links": [{"rel": "item", "href": "https://example.com/b/scene.json"}],
        }
        item = {
            "id": "scene",
            "type": "Feature",
            "bbox": [0, 0, 1, 1],
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]],
                ],
            },
            "properties": {
                "datetime": "2025-01-15T00:00:00Z",
                "title": "Shared scene",
            },
            "assets": {
                "ortho_visual": {"href": "https://example.com/visual.tif"},
                "ortho_radiance_hdf5": {"href": "https://example.com/rad.h5"},
            },
        }

        responses = [
            _MockResponse(obj)
            for obj in (catalog, collection_a, item, collection_b, item)
        ]
        with mock.patch("hypercoast.tanager.requests.get", side_effect=responses):
            gdf = hypercoast.tanager_footprints()

        self.assertEqual(len(gdf), 1)
        self.assertEqual(gdf.iloc[0]["id"], "scene")
        self.assertEqual(gdf.iloc[0]["collections"], "a, b")
        self.assertEqual(gdf.iloc[0]["collection_titles"], "Collection A, Collection B")
        self.assertEqual(
            gdf.iloc[0]["ortho_visual_url"], "https://example.com/visual.tif"
        )

    def test_read_tanager_stac_does_not_force_asset_product(self):
        item = {
            "id": "item-1",
            "_stac_url": "https://example.com/item-1.json",
            "assets": {
                "ortho_radiance_hdf5": {
                    "href": "https://example.com/ortho.h5",
                    "eo:bands": [
                        {
                            "center_wavelength": 0.4,
                            "full_width_half_max": 0.006,
                            "name": "toa_radiance_B000",
                        }
                    ],
                }
            },
        }
        fake_ds = mock.MagicMock()
        fake_ds.attrs = {}

        with (
            mock.patch(
                "hypercoast.tanager.download_tanager", return_value=["/tmp/ortho.h5"]
            ),
            mock.patch(
                "hypercoast.tanager.read_tanager", return_value=fake_ds
            ) as mocked,
        ):
            result = hypercoast.read_tanager_stac(item)

        self.assertIs(result, fake_ds)
        _, kwargs = mocked.call_args
        self.assertIsNone(kwargs["product"])
        np.testing.assert_allclose(kwargs["wavelengths"], [400.0])
        np.testing.assert_allclose(kwargs["fwhm"], [6.0])
        self.assertEqual(result.attrs["product"], "ortho_radiance")

    def test_search_tanager_with_count_zero_returns_empty(self):
        with mock.patch("hypercoast.tanager.requests.get") as mocked_get:
            results = hypercoast.search_tanager(count=0)

        self.assertEqual(results, [])
        mocked_get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
