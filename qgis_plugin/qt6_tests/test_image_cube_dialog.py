"""Tests for the 3D image cube dock helpers."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import hypercoast_qgis.dialogs.image_cube_dialog as dialog_module
import numpy as np
import pytest
from hypercoast_qgis.dialogs.image_cube_dialog import ImageCubeDialog
from qgis.PyQt.QtWidgets import QApplication


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
    """Small QGIS layer-like object."""

    def name(self):
        """Return a layer name."""
        return "Layer One"


class _DatasetWrapper:
    """Small HyperCoast dataset wrapper."""

    def __init__(self, dataset):
        """Initialize the wrapper.

        Args:
            dataset: Xarray dataset.
        """
        self.dataset = dataset

    def get_data_variable(self):
        """Return the default data variable."""
        return self.dataset["reflectance"]


class _Plugin:
    """Small plugin-like object for image cube tests."""

    def __init__(self, dataset):
        """Initialize the plugin.

        Args:
            dataset: Xarray dataset.
        """
        self.wrapper = _DatasetWrapper(dataset)
        self.data_info = {
            "dataset": self.wrapper,
            "rgb_wavelengths": [650, 550, 450],
            "selected_variable": "reflectance",
        }

    def get_all_hyperspectral_layers(self):
        """Return registered layer metadata."""
        return {"layer-1": self.data_info}

    def get_hyperspectral_data(self, layer_id):
        """Return metadata for a layer.

        Args:
            layer_id: QGIS layer ID.

        Returns:
            Layer metadata.
        """
        return self.data_info if layer_id == "layer-1" else None

    def ensure_hyperspectral_dataset(self, layer_id):
        """Return a loaded dataset wrapper.

        Args:
            layer_id: QGIS layer ID.

        Returns:
            Dataset wrapper.
        """
        return self.wrapper if layer_id == "layer-1" else None


def _dataset():
    """Return a small xarray hyperspectral cube."""
    xr = pytest.importorskip("xarray")
    return xr.Dataset(
        {
            "reflectance": (
                ("y", "x", "wavelength"),
                np.ones((8, 10, 4), dtype="float32"),
            )
        },
        coords={"wavelength": [450.0, 550.0, 650.0, 750.0]},
    )


def _spectral_first_dataset():
    """Return a Tanager-like cube with the spectral axis first."""
    xr = pytest.importorskip("xarray")
    return xr.Dataset(
        {
            "surface_reflectance": (
                ("wavelength", "y", "x"),
                np.ones((4, 8, 10), dtype="float32"),
            )
        },
        coords={"wavelength": [450.0, 550.0, 650.0, 750.0]},
    )


def _latlon_dataset():
    """Return a small cube with 2D latitude and longitude coordinates."""
    xr = pytest.importorskip("xarray")
    y = np.linspace(36.0, 39.0, 8)
    x = np.linspace(-123.0, -120.0, 10)
    lon, lat = np.meshgrid(x, y)
    return xr.Dataset(
        {
            "reflectance": (
                ("y", "x", "wavelength"),
                np.ones((8, 10, 4), dtype="float32"),
            )
        },
        coords={
            "wavelength": [450.0, 550.0, 650.0, 750.0],
            "latitude": (("y", "x"), lat),
            "longitude": (("y", "x"), lon),
        },
    )


def test_image_cube_dialog_populates_layers_and_variables(qapp, monkeypatch):
    """The image cube dock should list HyperCoast layers and variables."""
    dialog = ImageCubeDialog(_Iface(), _Plugin(_dataset()))
    monkeypatch.setattr(dialog, "_project_layer", lambda layer_id: _Layer())

    dialog.refresh_layers()

    assert dialog.layer_combo.currentData() == "layer-1"
    assert dialog.layer_combo.currentText() == "Layer One"
    assert dialog.variable_combo.currentData() == "reflectance"
    assert dialog.red_spin.value() == 650.0


def test_image_cube_dialog_downsamples_spatial_and_spectral_axes(qapp):
    """Downsampling should stride spatial and spectral dimensions."""
    dialog = ImageCubeDialog(_Iface(), _Plugin(_dataset()))

    downsampled = dialog._downsample_dataset(_dataset(), 2, 2)

    assert downsampled.sizes["y"] == 4
    assert downsampled.sizes["x"] == 5
    assert downsampled.sizes["wavelength"] == 2


def test_image_cube_dialog_subsets_2d_latlon_bbox(qapp):
    """Drawn WGS84 bboxes should subset 2D latitude/longitude cubes."""
    dataset = _latlon_dataset()
    dialog = ImageCubeDialog(_Iface(), _Plugin(dataset))

    subset = dialog._subset_dataset(
        dataset, "reflectance", [-122.5, 36.5, -121.0, 38.0]
    )

    assert subset.sizes["y"] < dataset.sizes["y"]
    assert subset.sizes["x"] < dataset.sizes["x"]
    assert float(subset["longitude"].min()) >= -123.0
    assert float(subset["latitude"].max()) <= 39.0


def test_image_cube_dialog_stores_drawn_subset_bbox(qapp):
    """Drawn subset bboxes should be stored and displayed."""
    dialog = ImageCubeDialog(_Iface(), _Plugin(_dataset()))

    dialog._on_subset_bbox_drawn([-122.5, 36.5, -121.0, 38.0])

    assert dialog._subset_bbox == [-122.5, 36.5, -121.0, 38.0]
    assert (
        dialog.subset_bbox_edit.text()
        == "-122.50000000, 36.50000000, -121.00000000, 38.00000000"
    )


def test_image_cube_dialog_blocks_oversized_cube(qapp, monkeypatch):
    """Oversized cubes should be blocked before PyVista rendering."""
    dialog = ImageCubeDialog(_Iface(), _Plugin(_dataset()))
    monkeypatch.setattr(dialog_module, "MAX_CUBE_POINTS", 1)
    warnings = []
    monkeypatch.setattr(
        dialog_module.QMessageBox,
        "warning",
        staticmethod(lambda *args: warnings.append(args)),
    )

    assert not dialog._validate_cube_size(_dataset(), "reflectance")
    assert warnings
    assert dialog.status_label.text() == "3D image cube too large"


def test_image_cube_dialog_moves_spectral_axis_last(qapp):
    """Tanager-style spectral-first cubes should render RGB overlays correctly."""
    dataset = _spectral_first_dataset()
    dialog = ImageCubeDialog(_Iface(), _Plugin(dataset))

    prepared = dialog._prepare_image_cube_dataset(dataset, "surface_reflectance")

    assert prepared["surface_reflectance"].dims == ("y", "x", "wavelength")


def test_image_cube_dialog_launches_image_cube_viewer(qapp, monkeypatch):
    """Create Cube should launch the external image cube viewer."""
    plugin = _Plugin(_dataset())
    dialog = ImageCubeDialog(_Iface(), plugin)
    monkeypatch.setattr(dialog, "_project_layer", lambda layer_id: _Layer())
    dialog.refresh_layers()
    dialog.spatial_stride_spin.setValue(2)
    dialog.widget_combo.setCurrentIndex(dialog.widget_combo.findData("slice"))
    recorded = {}

    def _record_start(dataset, kwargs):
        """Record external viewer worker arguments."""
        recorded["sizes"] = dict(dataset.sizes)
        recorded["kwargs"] = kwargs
        dialog._set_cube_busy(False, "3D image cube viewer launched")

    monkeypatch.setattr(dialog, "_start_image_cube_worker", _record_start)

    dialog.create_cube()

    assert recorded["sizes"]["y"] == 4
    assert recorded["kwargs"]["variable"] == "reflectance"
    assert recorded["kwargs"]["widget"] == "slice"
    assert recorded["kwargs"]["rgb_wavelengths"] == [650.0, 550.0, 450.0]
    assert dialog.status_label.text() == "3D image cube viewer launched"


def test_image_cube_dialog_launches_viewer_with_spectral_axis_last(qapp, monkeypatch):
    """Create Cube should normalize Tanager-like cubes before launch."""
    dataset = _spectral_first_dataset()
    plugin = _Plugin(dataset)
    plugin.data_info["selected_variable"] = "surface_reflectance"
    dialog = ImageCubeDialog(_Iface(), plugin)
    dialog.variable_combo.setCurrentIndex(
        dialog.variable_combo.findData("surface_reflectance")
    )
    recorded = {}

    def _record_start(dataset, kwargs):
        """Record external viewer worker arguments."""
        variable = kwargs["variable"]
        recorded["dims"] = dataset[variable].dims
        recorded["kwargs"] = kwargs

    monkeypatch.setattr(dialog, "_start_image_cube_worker", _record_start)

    dialog.create_cube()

    assert recorded["dims"] == ("y", "x", "wavelength")
    assert recorded["kwargs"]["variable"] == "surface_reflectance"
    assert recorded["kwargs"]["rgb_wavelengths"] == [650.0, 550.0, 450.0]


def test_image_cube_dialog_launch_writes_files_and_starts_process(
    qapp, monkeypatch, tmp_path
):
    """External viewer launch should write cube files and start Python."""
    dialog = ImageCubeDialog(_Iface(), _Plugin(_dataset()))
    monkeypatch.setattr(dialog, "_image_cube_cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(dialog, "_viewer_python_path", lambda: "/usr/bin/python")
    monkeypatch.setattr(dialog, "_viewer_popen_kwargs", lambda: {})
    recorded = {}

    class _Process:
        """Small subprocess-like object."""

    def _popen(cmd, cwd, env, stdout, stderr, **kwargs):
        """Record subprocess arguments."""
        recorded["cmd"] = cmd
        recorded["cwd"] = cwd
        recorded["env"] = env
        recorded["kwargs"] = kwargs
        recorded["stdout"] = stdout.name
        recorded["stderr"] = stderr.name
        return _Process()

    monkeypatch.setattr(dialog_module.subprocess, "Popen", _popen)

    dialog._launch_image_cube_viewer(
        _dataset(),
        {
            "variable": "reflectance",
            "clim": (0.0, 0.5),
            "widget": None,
            "rgb_wavelengths": [650.0, 550.0, 450.0],
        },
    )

    assert recorded["cmd"][0] == "/usr/bin/python"
    assert recorded["cwd"] == str(tmp_path)
    assert os.path.exists(recorded["cmd"][1])
    assert os.path.exists(recorded["cmd"][2])
    assert os.path.exists(recorded["cmd"][3])
    assert recorded["env"]["PYTHONPATH"].endswith("HyperCoast")
    assert dialog._viewer_processes


def test_image_cube_dialog_busy_progress_state(qapp):
    """Busy state should show progress and disable cube creation."""
    dialog = ImageCubeDialog(_Iface(), _Plugin(_dataset()))

    dialog._set_cube_busy(True, "Creating 3D image cube...")

    assert not dialog.progress_bar.isHidden()
    assert not dialog.create_btn.isEnabled()
    assert dialog.status_label.text() == "Creating 3D image cube..."

    dialog._set_cube_busy(False, "3D image cube viewer launched")

    assert dialog.progress_bar.isHidden()
    assert dialog.create_btn.isEnabled()
    assert dialog.status_label.text() == "3D image cube viewer launched"
