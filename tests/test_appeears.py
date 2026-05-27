# SPDX-FileCopyrightText: 2026 Qiusheng Wu <giswqs@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for AppEEARS helpers."""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

import hypercoast
from hypercoast import appeears


class MockResponse:
    """Minimal requests response mock."""

    def __init__(self, payload=None, content=None):
        """Create a mock response.

        Args:
            payload: JSON payload returned by ``json``.
            content: Binary content returned by ``iter_content``.
        """
        self.payload = payload
        self.content = content or b""
        self.text = ""

    def json(self):
        """Return the mocked JSON payload."""
        return self.payload

    def raise_for_status(self):
        """Mock a successful HTTP response."""

    def iter_content(self, chunk_size=8192):
        """Yield mocked response content.

        Args:
            chunk_size: Maximum chunk size.

        Yields:
            Binary chunks.
        """
        for index in range(0, len(self.content), chunk_size):
            yield self.content[index : index + chunk_size]


class MockSession:
    """Requests-compatible session mock."""

    def __init__(self):
        """Create a mock session."""
        self.calls = []

    def request(self, method, url, **kwargs):
        """Record the request and return a fixture response.

        Args:
            method: HTTP method.
            url: Request URL.
            **kwargs: Request options.

        Returns:
            MockResponse: Fixture response.
        """
        self.calls.append((method, url, kwargs))
        if url.endswith("/login"):
            return MockResponse({"token": "abc"})
        if url.endswith("/product/EMIT_L2A_RFL.001"):
            reflectance = "Reflectance for Wavelength: {} nm, FWHM: 8.415 nm"
            flag = "Usable wavelengths flag for Wavelength: 388.409 nm, FWHM: 8.415 nm"
            return MockResponse(
                {
                    "B001": {
                        "Description": reflectance.format("381.006"),
                        "Units": "spectral reflectance",
                        "DataType": "float32",
                    },
                    "B002": {
                        "Description": reflectance.format("388.409"),
                        "Units": "spectral reflectance",
                        "DataType": "float32",
                    },
                    "B002_GW": {
                        "Description": flag,
                        "Units": "binary flag",
                        "DataType": "uint8",
                    },
                }
            )
        if url.endswith("/bundle/task-1"):
            return MockResponse(
                {
                    "files": [
                        {
                            "file_id": "file-1",
                            "file_name": "folder/EMIT_L2A_RFL_001_B001_doy.tif",
                            "file_type": "tif",
                        },
                        {
                            "file_id": "file-2",
                            "file_name": "readme.txt",
                            "file_type": "txt",
                        },
                    ]
                }
            )
        if url.endswith("/bundle/task-1/file-1"):
            return MockResponse(content=b"data")
        return MockResponse({})


class TestAppEEARS(unittest.TestCase):
    """Test AppEEARS helper behavior."""

    def test_emit_layers_selects_nearest_wavelength(self):
        session = MockSession()
        client = appeears.AppEEARSClient(token="abc", session=session)

        layers = hypercoast.appeears_emit_layers(
            wavelengths=[389],
            client=client,
            return_metadata=True,
        )

        self.assertEqual(layers[0]["layer"], "B002")
        self.assertEqual(layers[0]["wavelength"], 388.409)

    def test_point_task_formats_dates_and_coordinates(self):
        task = hypercoast.appeears_point_task(
            task_name="point",
            coordinates=(-118.3, 34.1),
            layers=[{"product": "EMIT_L2A_RFL.001", "layer": "B001"}],
            start_date="2024-04-01",
            end_date="2024-04-30",
        )

        self.assertEqual(task["task_type"], "point")
        self.assertEqual(task["params"]["dates"][0]["startDate"], "04-01-2024")
        self.assertEqual(
            task["params"]["coordinates"][0],
            {"longitude": -118.3, "latitude": 34.1, "id": "point-1"},
        )

    def test_area_task_accepts_bbox(self):
        task = hypercoast.appeears_area_task(
            task_name="area",
            geometry=[-118.5, 33.8, -118.2, 34.1],
            layers=[{"product": "EMIT_L2A_RFL.001", "layer": "B001"}],
            start_date="04-01-2024",
            end_date="04-30-2024",
        )

        geo = task["params"]["geo"]
        self.assertEqual(task["task_type"], "area")
        self.assertEqual(geo["type"], "FeatureCollection")
        self.assertEqual(geo["features"][0]["geometry"]["type"], "Polygon")
        self.assertEqual(task["params"]["output"]["format"]["type"], "netcdf4")
        self.assertEqual(
            task["params"]["output"]["additionalOptions"],
            {"orthorectify": True},
        )

    def test_download_bundle_filters_file_types(self):
        session = MockSession()
        client = appeears.AppEEARSClient(token="abc", session=session)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = client.download_bundle("task-1", out_dir=tmpdir, file_types=["tif"])

            self.assertEqual(len(paths), 1)
            self.assertEqual(Path(paths[0]).name, "EMIT_L2A_RFL_001_B001_doy.tif")
            self.assertTrue(os.path.exists(paths[0]))
            self.assertEqual(Path(paths[0]).read_bytes(), b"data")

    def test_read_appeears_stacks_netcdf_band_variables(self):
        layers = [
            {"product": "EMIT_L2A_RFL.001", "layer": "B001", "wavelength": 381.006},
            {"product": "EMIT_L2A_RFL.001", "layer": "B002", "wavelength": 388.409},
        ]
        source = xr.Dataset(
            {
                "EMIT_L2A_RFL_001_B002": (("y", "x"), np.full((2, 2), 2.0)),
                "EMIT_L2A_RFL_001_B001": (("y", "x"), np.full((2, 2), 1.0)),
            },
            coords={"y": [0, 1], "x": [10, 11]},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "appeears.nc"
            source.to_netcdf(filepath)

            dataset = hypercoast.read_appeears(str(filepath), layers=layers)

        self.assertEqual(dataset["reflectance"].dims, ("wavelength", "y", "x"))
        self.assertEqual(dataset.sizes["wavelength"], 2)
        self.assertEqual(dataset["wavelength"].values.tolist(), [381.006, 388.409])
        self.assertEqual(dataset["reflectance"].isel(wavelength=0).values[0, 0], 1.0)


if __name__ == "__main__":
    unittest.main()
