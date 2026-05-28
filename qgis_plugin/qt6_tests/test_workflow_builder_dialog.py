"""Tests for the Workflow Builder dock."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import hypercoast_qgis.dialogs.workflow_builder_dialog as dialog_module
import numpy as np
import pytest
from hypercoast_qgis.dialogs.workflow_builder_dialog import WorkflowBuilderDialog
from qgis.PyQt.QtWidgets import QApplication

xr = pytest.importorskip("xarray")


@pytest.fixture(scope="module")
def qapp():
    """Return a QApplication for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _Iface:
    """Small QGIS iface-like object."""

    def mainWindow(self):
        """Return no parent window."""
        return None


class _Layer:
    """Small layer object."""

    def name(self):
        """Return a display name."""
        return "Layer One"


class _Project:
    """Small project object."""

    def mapLayer(self, layer_id):
        """Return a layer for known IDs."""
        return _Layer() if layer_id == "layer-1" else None

    def addMapLayer(self, layer):
        """Accept added layers."""
        self.added_layer = layer


class _DatasetWrapper:
    """Small HyperspectralDataset-like wrapper."""

    def __init__(self):
        """Initialize the wrapper."""
        self.dataset = xr.Dataset(
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
        self.bounds = (0, 0, 2, 2)
        self.crs = "EPSG:4326"
        self.selected_variable = None

    def list_data_variables(self):
        """Return exportable variables."""
        return ["reflectance"]

    def set_selected_variable(self, variable_name):
        """Set the selected variable."""
        self.selected_variable = variable_name


class _Plugin:
    """Small plugin object."""

    def __init__(self):
        """Initialize the plugin."""
        self.wrapper = _DatasetWrapper()

    def get_all_hyperspectral_layers(self):
        """Return layer metadata."""
        return {"layer-1": {"selected_variable": "reflectance"}}

    def ensure_hyperspectral_dataset(self, layer_id):
        """Return a dataset wrapper."""
        return self.wrapper if layer_id == "layer-1" else None

    def get_hyperspectral_data(self, layer_id):
        """Return layer metadata."""
        return {"selected_variable": "reflectance"} if layer_id == "layer-1" else None


def test_workflow_builder_populates_layers_and_variables(qapp, monkeypatch):
    """Workflow Builder should list registered layers and variables."""
    project = _Project()
    monkeypatch.setattr(dialog_module.QgsProject, "instance", lambda: project)

    dialog = WorkflowBuilderDialog(_Iface(), _Plugin())

    assert dialog.layer_combo.currentData() == "layer-1"
    assert dialog.layer_combo.currentText() == "Layer One"
    assert dialog.variable_combo.currentText() == "reflectance"


def test_workflow_builder_runs_ndwi(qapp, monkeypatch, tmp_path):
    """Workflow Builder should run a workflow and write output."""
    project = _Project()
    written = {}
    monkeypatch.setattr(dialog_module.QgsProject, "instance", lambda: project)
    monkeypatch.setattr(dialog_module, "HAS_RASTERIO", True)
    monkeypatch.setattr(
        dialog_module,
        "QgsRasterLayer",
        lambda *args, **kwargs: type("Layer", (), {"isValid": lambda self: True})(),
    )

    dialog = WorkflowBuilderDialog(_Iface(), _Plugin())
    output = tmp_path / "ndwi.tif"
    dialog.output_edit.setText(str(output))
    monkeypatch.setattr(
        dialog,
        "_write_workflow_raster",
        lambda dataset, result, output_path: written.update(
            {"name": result.name, "output": output_path}
        ),
    )

    dialog.run_workflow()

    assert written["name"] == "nd_560_860"
    assert written["output"] == str(output)
