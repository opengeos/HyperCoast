"""Smoke tests for the dockable HyperCoast panel classes."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import hypercoast_qgis.dialogs.load_data_dialog as load_data_dialog_module
import pytest
from hypercoast_qgis.dialogs.about_dialog import AboutDialog
from hypercoast_qgis.dialogs.band_combination_dialog import BandCombinationDialog
from hypercoast_qgis.dialogs.dependency_installer import DependencyInstallerDialog
from hypercoast_qgis.dialogs.image_cube_dialog import ImageCubeDialog
from hypercoast_qgis.dialogs.load_data_dialog import LoadDataDialog
from hypercoast_qgis.dialogs.settings_dock import SettingsDockWidget
from hypercoast_qgis.dialogs.spectral_plot_dialog import SpectralPlotDialog
from hypercoast_qgis.dialogs.tanager_search_dialog import TanagerSearchDialog
from hypercoast_qgis.dialogs.update_checker import UpdateCheckerDialog
from qgis.PyQt.QtWidgets import QApplication, QDockWidget


@pytest.fixture(scope="module")
def qapp():
    """Return a QApplication for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_plugin_windows_are_dock_widgets():
    """All user-facing plugin windows should be dockable panels."""
    dock_classes = [
        AboutDialog,
        BandCombinationDialog,
        DependencyInstallerDialog,
        ImageCubeDialog,
        LoadDataDialog,
        SettingsDockWidget,
        SpectralPlotDialog,
        TanagerSearchDialog,
        UpdateCheckerDialog,
    ]

    for dock_class in dock_classes:
        assert issubclass(dock_class, QDockWidget)


def test_tanager_auto_detect_sets_reflectance_value_range(qapp, tmp_path):
    """Auto-detected Tanager files should use the default reflectance max."""
    h5py = pytest.importorskip("h5py")
    filepath = tmp_path / "custom_name.h5"
    with h5py.File(filepath, "w") as h5_file:
        h5_file.create_dataset("product/toa_radiance", shape=(2, 3, 4))

    class _Iface:
        """Small iface-like object."""

        def mainWindow(self):
            """Return no parent window."""
            return None

    plugin = type("Plugin", (), {})()
    dialog = LoadDataDialog(_Iface(), plugin)
    dialog.file_path_edit.setText(str(filepath))

    dialog._clear_dataset_preview()

    assert dialog.vmax_spin.value() == 0.5


def test_load_data_browse_starts_in_user_directory(qapp, monkeypatch):
    """Load Data Browse should open from the user home directory."""

    class _Iface:
        """Small iface-like object."""

        def mainWindow(self):
            """Return no parent window."""
            return None

    recorded = {}
    monkeypatch.setattr(
        load_data_dialog_module.os.path, "expanduser", lambda path: "/home/tester"
    )
    monkeypatch.setattr(
        load_data_dialog_module.QFileDialog,
        "getOpenFileName",
        staticmethod(
            lambda parent, title, directory, file_filter: recorded.update(
                directory=directory
            )
            or ("", "")
        ),
    )

    dialog = LoadDataDialog(_Iface(), object())
    dialog.browse_file()

    assert recorded["directory"] == "/home/tester"


def test_spectral_plot_uses_tanager_reflectance_defaults(qapp):
    """Tanager spectra should keep the plot label and y-range on reflectance."""
    pytest.importorskip("matplotlib")

    class _Iface:
        """Small iface-like object."""

        def mainWindow(self):
            """Return no parent window."""
            return None

    class _Plugin:
        """Small plugin-like object."""

        spectral_tool = None

        def get_all_hyperspectral_layers(self):
            """Return no registered layers."""
            return {}

        def get_hyperspectral_data(self, layer_id):
            """Return no layer metadata."""
            return None

    dialog = SpectralPlotDialog(_Iface(), _Plugin())

    dialog.add_spectrum(
        30.0,
        -90.0,
        [500.0, 600.0],
        [12.0, 24.0],
        "Tanager Radiance",
        data_type="Tanager",
    )

    assert dialog.ylabel_combo.currentText() == "Reflectance"
    assert dialog.ymax_spin.value() == 0.5
