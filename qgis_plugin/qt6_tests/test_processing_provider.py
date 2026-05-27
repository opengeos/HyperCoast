"""Tests for HyperCoast Processing helper behavior."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from hypercoast_qgis.hyperspectral_provider import HyperspectralDataset
from hypercoast_qgis.processing_provider import (
    BaseHyperCoastAlgorithm,
    BatchExtractSpectraAlgorithm,
    HyperCoastProcessingProvider,
    QgsProcessingException,
    RGBCompositeAlgorithm,
    SummarizeDatasetAlgorithm,
    WaterQualityWorkflowAlgorithm,
)


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
    assert "summarize_dataset" in names
    assert "batch_extract_spectra" in names


def test_water_quality_workflow_selects_nearest_band():
    """Workflow helpers should select nearest spectral bands."""
    algorithm = WaterQualityWorkflowAlgorithm()
    dataset = _loaded_dataset()
    band = algorithm._select_nearest_band(dataset.dataset["reflectance"], 590.0)

    np.testing.assert_array_equal(band, np.ones((2, 2), dtype="float32"))


def test_summarize_dataset_algorithm_writes_json(tmp_path, monkeypatch):
    """Summarize Dataset should write a JSON metadata file."""
    algorithm = SummarizeDatasetAlgorithm()
    output = tmp_path / "summary.json"

    monkeypatch.setattr(algorithm, "_load_dataset", lambda *args: _loaded_dataset())

    result = algorithm.processAlgorithm(
        {
            BaseHyperCoastAlgorithm.INPUT: "generic.nc",
            BaseHyperCoastAlgorithm.DATA_TYPE: 0,
            BaseHyperCoastAlgorithm.VARIABLE: "",
            algorithm.OUTPUT_JSON: str(output),
        },
        None,
        None,
    )

    assert result[algorithm.OUTPUT_JSON] == str(output)
    assert '"variables"' in output.read_text(encoding="utf-8")


def test_batch_extract_spectra_algorithm_writes_long_csv(tmp_path, monkeypatch):
    """Batch Extract Spectra should write long-form CSV output."""
    algorithm = BatchExtractSpectraAlgorithm()
    points = tmp_path / "points.csv"
    output = tmp_path / "spectra.csv"
    points.write_text("x,y\n0,0\n", encoding="utf-8")
    dataset = _loaded_dataset()

    monkeypatch.setattr(algorithm, "_load_dataset", lambda *args: dataset)
    monkeypatch.setattr(
        dataset,
        "extract_spectral_signature",
        lambda *args, **kwargs: (
            np.array([500.0, 600.0]),
            np.array([0.1, 0.2]),
        ),
    )

    result = algorithm.processAlgorithm(
        {
            BaseHyperCoastAlgorithm.INPUT: "generic.nc",
            BaseHyperCoastAlgorithm.DATA_TYPE: 0,
            BaseHyperCoastAlgorithm.VARIABLE: "",
            algorithm.POINTS: str(points),
            algorithm.X_COLUMN: "x",
            algorithm.Y_COLUMN: "y",
            algorithm.CRS: "EPSG:4326",
            algorithm.OUTPUT_CSV: str(output),
        },
        None,
        None,
    )

    assert result[algorithm.OUTPUT_CSV] == str(output)
    assert "wavelength,value" in output.read_text(encoding="utf-8")
