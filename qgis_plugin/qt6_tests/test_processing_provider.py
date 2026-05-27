"""Tests for HyperCoast Processing helper behavior."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from hypercoast_qgis.processing_provider import (
    BaseHyperCoastAlgorithm,
    QgsProcessingException,
    HyperCoastProcessingProvider,
    RGBCompositeAlgorithm,
    WaterQualityWorkflowAlgorithm,
)
from hypercoast_qgis.hyperspectral_provider import HyperspectralDataset


def _loaded_dataset():
    """Return a loaded provider with multiple candidate variables."""
    dataset = xr.Dataset(
        {
            "reflectance": (
                ("wavelength", "y", "x"),
                np.ones((2, 2, 2), dtype="float32"),
            ),
            "quality": (("y", "x"), np.ones((2, 2), dtype="float32")),
            "metadata": (("record",), np.ones(2, dtype="float32")),
        },
        coords={
            "wavelength": np.array([500.0, 600.0]),
            "y": np.array([0.0, 1.0]),
            "x": np.array([0.0, 1.0]),
        },
    )
    provider = HyperspectralDataset("generic.nc", "Generic")
    provider.dataset = dataset
    return provider


def test_blank_processing_variable_uses_default_selection():
    """Blank Processing variable input should behave like no override."""
    algorithm = RGBCompositeAlgorithm()

    assert (
        algorithm._selected_variable({BaseHyperCoastAlgorithm.VARIABLE: ""}, None)
        is None
    )


def test_valid_processing_variable_is_accepted():
    """A raster-like selected variable should pass validation."""
    algorithm = RGBCompositeAlgorithm()
    dataset = _loaded_dataset()
    dataset.set_selected_variable("quality")

    algorithm._validate_selected_variable(dataset)

    assert dataset.get_data_variable().name == "quality"


def test_invalid_processing_variable_raises_clear_exception():
    """Missing or non-raster variables should raise Processing exceptions."""
    algorithm = RGBCompositeAlgorithm()
    dataset = _loaded_dataset()
    dataset.set_selected_variable("missing")

    with pytest.raises(QgsProcessingException, match="Data variable not found"):
        algorithm._validate_selected_variable(dataset)

    dataset.set_selected_variable("metadata")
    with pytest.raises(QgsProcessingException, match="not raster-like"):
        algorithm._validate_selected_variable(dataset)


def test_processing_provider_registers_water_quality_workflow():
    """Processing provider should include the coastal workflow algorithm."""
    provider = HyperCoastProcessingProvider()

    provider.loadAlgorithms()

    names = [algorithm.name() for algorithm in provider.algorithms]
    assert "water_quality_workflow" in names


def test_water_quality_workflow_selects_nearest_band():
    """Workflow helpers should select nearest spectral bands."""
    algorithm = WaterQualityWorkflowAlgorithm()
    dataset = _loaded_dataset()
    band = algorithm._select_nearest_band(dataset.dataset["reflectance"], 590.0)

    np.testing.assert_array_equal(band, np.ones((2, 2), dtype="float32"))
