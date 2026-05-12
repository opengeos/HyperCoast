"""Smoke tests for the dockable HyperCoast panel classes."""

from qgis.PyQt.QtWidgets import QDockWidget

from hypercoast_qgis.dialogs.about_dialog import AboutDialog
from hypercoast_qgis.dialogs.band_combination_dialog import BandCombinationDialog
from hypercoast_qgis.dialogs.dependency_installer import DependencyInstallerDialog
from hypercoast_qgis.dialogs.load_data_dialog import LoadDataDialog
from hypercoast_qgis.dialogs.settings_dock import SettingsDockWidget
from hypercoast_qgis.dialogs.spectral_plot_dialog import SpectralPlotDialog
from hypercoast_qgis.dialogs.update_checker import UpdateCheckerDialog


def test_plugin_windows_are_dock_widgets():
    """All user-facing plugin windows should be dockable panels."""
    dock_classes = [
        AboutDialog,
        BandCombinationDialog,
        DependencyInstallerDialog,
        LoadDataDialog,
        SettingsDockWidget,
        SpectralPlotDialog,
        UpdateCheckerDialog,
    ]

    for dock_class in dock_classes:
        assert issubclass(dock_class, QDockWidget)
