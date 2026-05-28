# SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for dataset summary helpers."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hypercoast.summary import (
    extract_spectra_to_csv,
    subset_dataset,
    summarize_dataset,
)


def _dataset():
    """Return a small rectilinear hyperspectral dataset."""
    return xr.Dataset(
        {
            "reflectance": (
                ("wavelength", "y", "x"),
                np.arange(27, dtype="float32").reshape(3, 3, 3),
            )
        },
        coords={
            "wavelength": [450.0, 550.0, 650.0],
            "y": [10.0, 11.0, 12.0],
            "x": [100.0, 101.0, 102.0],
        },
        attrs={"crs": "EPSG:4326"},
    )


def test_summarize_dataset_reports_core_metadata(tmp_path):
    """Dataset summaries should include variables, dimensions, and wavelengths."""
    path = tmp_path / "cube.nc"
    _dataset().to_netcdf(path)

    summary = summarize_dataset(path, variable="reflectance")

    assert summary.exists is True
    assert summary.variables == ["reflectance"]
    assert summary.selected_variable == "reflectance"
    assert summary.crs == "EPSG:4326"
    assert summary.wavelength_count == 3
    assert summary.wavelength_min == 450.0
    assert summary.wavelength_max == 650.0
    assert summary.as_dict()["dimensions"]["x"] == 3


def test_summarize_dataset_missing_file_returns_warning(tmp_path):
    """Missing paths should return a warning instead of raising."""
    summary = summarize_dataset(tmp_path / "missing.nc")

    assert summary.exists is False
    assert summary.warnings


def test_summarize_dataset_rejects_unknown_sensor(tmp_path):
    """Unknown sensors should still raise registry lookup errors."""
    path = tmp_path / "cube.nc"
    _dataset().to_netcdf(path)

    with pytest.raises(KeyError):
        summarize_dataset(path, sensor="unknown")


def test_subset_dataset_writes_bbox_subset(tmp_path):
    """Subset helper should write a NetCDF clipped by x/y coordinates."""
    input_path = tmp_path / "cube.nc"
    output_path = tmp_path / "subset.nc"
    _dataset().to_netcdf(input_path)

    subset_dataset(input_path, output_path, bbox=(100.5, 10.5, 102.0, 12.0))
    subset = xr.open_dataset(output_path)

    assert subset.sizes["x"] == 2
    assert subset.sizes["y"] == 2


def test_extract_spectra_to_csv_writes_long_form_rows(tmp_path, monkeypatch):
    """Batch spectral extraction should write one row per wavelength."""
    import hypercoast.registry as registry
    import hypercoast.summary as summary_module

    points = tmp_path / "points.csv"
    output = tmp_path / "spectra.csv"
    pd.DataFrame({"x": [1.0], "y": [2.0]}).to_csv(points, index=False)
    dataset = xr.Dataset()
    spectrum = xr.DataArray(
        np.array([0.1, 0.2]),
        dims=("wavelength",),
        coords={"wavelength": [500.0, 600.0]},
        name="reflectance",
    )

    monkeypatch.setattr(summary_module, "read_sensor", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(registry, "extract_sensor", lambda *args, **kwargs: spectrum)

    extract_spectra_to_csv("pace", "cube.nc", points, output)
    rows = pd.read_csv(output)

    assert list(rows["wavelength"]) == [500.0, 600.0]
    assert list(rows["value"]) == [0.1, 0.2]
    assert rows.loc[0, "variable"] == "reflectance"
