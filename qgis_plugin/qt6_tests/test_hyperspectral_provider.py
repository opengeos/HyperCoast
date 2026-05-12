"""Tests for hyperspectral dataset variable selection."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from hypercoast_qgis.hyperspectral_provider import HyperspectralDataset


def test_pace_bgc_prefers_raster_product_over_tilt():
    """PACE BGC products should select a 2D geophysical raster variable."""
    dataset = xr.Dataset(
        {
            "tilt": (("latitude",), np.array([1.0, 2.0])),
            "chlor_a": (
                ("latitude", "longitude"),
                np.array([[0.1, 0.2], [0.3, 0.4]], dtype="float32"),
            ),
            "poc": (
                ("latitude", "longitude"),
                np.array([[10.0, 20.0], [30.0, 40.0]], dtype="float32"),
            ),
            "l2_flags": (
                ("latitude", "longitude"),
                np.array([[1, 2], [3, 4]], dtype="int32"),
            ),
        },
        coords={
            "latitude": np.array([29.0, 30.0]),
            "longitude": np.array([-91.0, -90.0]),
        },
    )
    provider = HyperspectralDataset("PACE_OCI_test.nc", "PACE")
    provider.dataset = dataset

    data_var = provider.get_data_variable()

    assert data_var.name == "chlor_a"
    assert data_var.ndim == 2
    assert provider.list_data_variables() == ["chlor_a", "poc", "l2_flags"]


def test_selected_variable_overrides_default_product():
    """A selected export variable should override the provider default."""
    dataset = xr.Dataset(
        {
            "chlor_a": (
                ("latitude", "longitude"),
                np.array([[0.1, 0.2], [0.3, 0.4]], dtype="float32"),
            ),
            "poc": (
                ("latitude", "longitude"),
                np.array([[10.0, 20.0], [30.0, 40.0]], dtype="float32"),
            ),
        }
    )
    provider = HyperspectralDataset("PACE_OCI_test.nc", "PACE")
    provider.dataset = dataset
    provider.set_selected_variable("poc")

    data_var = provider.get_data_variable()

    assert data_var.name == "poc"


def test_pace_integer_variable_uses_nearest_interpolation():
    """Integer PACE variables should not be linearly interpolated."""
    dataset = xr.Dataset(
        {
            "l2_flags": (
                ("latitude", "longitude"),
                np.array([[1, 2], [3, 4]], dtype="int32"),
            )
        }
    )
    provider = HyperspectralDataset("PACE_OCI_test.nc", "PACE")
    provider.dataset = dataset

    method = provider._pace_bgc_interpolation_method("l2_flags", "linear")

    assert method == "nearest"


def test_north_up_orientation_flips_ascending_latitude():
    """Ascending latitude rows must be flipped before GeoTIFF writing."""
    provider = HyperspectralDataset("PACE_OCI_test.nc", "PACE")
    data = np.array([[1, 2], [3, 4]], dtype="float32")
    y_values = np.array([10.0, 20.0])

    oriented = provider._orient_array_north_up(data, y_values, y_axis=0)

    np.testing.assert_array_equal(oriented, np.array([[3, 4], [1, 2]]))
