# SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for coastal workflow presets."""

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

from hypercoast.workflows import apply_workflow, list_workflows, spectral_anomaly


def _cube():
    """Return a small hyperspectral test cube."""
    return xr.Dataset(
        {
            "reflectance": (
                ("wavelength", "y", "x"),
                np.array(
                    [
                        [[0.6, 0.8], [0.4, 0.2]],
                        [[0.2, 0.2], [0.1, 0.1]],
                    ],
                    dtype="float32",
                ),
            )
        },
        coords={"wavelength": [560.0, 860.0], "y": [0, 1], "x": [0, 1]},
    )


def test_list_workflows_contains_coastal_presets():
    """Workflow metadata should include coastal presets."""
    workflows = list_workflows()

    assert "ndwi" in workflows
    assert "chlorophyll" in workflows
    assert workflows["ndwi"]["kind"] == "normalized_difference"


def test_apply_workflow_ndwi():
    """NDWI should calculate a normalized difference from nearest bands."""
    result = apply_workflow(_cube(), "ndwi")

    expected = np.array([[0.5, 0.6], [0.6, 1 / 3]], dtype="float32")
    np.testing.assert_allclose(result.values, expected, rtol=1e-6)


def test_spectral_anomaly_removes_wavelength_dimension():
    """Spectral anomaly output should be spatial."""
    result = spectral_anomaly(_cube())

    assert result.dims == ("y", "x")
    assert result.shape == (2, 2)
