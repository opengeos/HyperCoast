# -*- coding: utf-8 -*-
"""
HyperCoast QGIS Plugin - Main Plugin Class

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

import os

from qgis.PyQt.QtCore import QCoreApplication, QTimer
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.core import QgsMessageLog, QgsProject, Qgis

# Dependency installer and update checker have no heavy deps, always available
from .dialogs.dependency_installer import DependencyInstallerDialog
from .dialogs.about_dialog import AboutDialog
from .dialogs.update_checker import UpdateCheckerDialog

# Data-related dialogs depend on numpy (via hyperspectral_provider),
# so they may fail to import if dependencies are not installed yet.
_DATA_DIALOGS_AVAILABLE = False
try:
    from .dialogs.load_data_dialog import LoadDataDialog
    from .dialogs.band_combination_dialog import BandCombinationDialog
    from .dialogs.spectral_inspector_tool import SpectralInspectorTool
    from .dialogs.spectral_plot_dialog import SpectralPlotDialog

    _DATA_DIALOGS_AVAILABLE = True
except ImportError:
    pass


class HyperCoastPlugin:
    """Main HyperCoast QGIS Plugin class."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.actions = []
        self.menu = self.tr("&HyperCoast")
        self.toolbar = self.iface.addToolBar("HyperCoast")
        self.toolbar.setObjectName("HyperCoastToolbar")

        # Store reference to dialogs
        self.load_dialog = None
        self.band_dialog = None
        self.spectral_tool = None
        self.spectral_plot_dialog = None
        self.about_dialog = None
        self.update_dialog = None
        self.deps_dialog = None

        # Track whether dependencies are available
        self._deps_available = False

        # Store hyperspectral datasets
        self.hyperspectral_data = {}

    def tr(self, message):
        """Get the translation for a string using Qt translation API."""
        return QCoreApplication.translate("HyperCoast", message)

    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None,
        checkable=False,
    ):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action.
        :param text: Text that shows in the menu.
        :param callback: Function to be called when the action is triggered.
        :param enabled_flag: A flag indicating if the action should be enabled
            by default.
        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu.
        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar.
        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :param parent: Parent widget for the new action.
        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.
        :param checkable: Whether the action is checkable.

        :returns: The action that was created.
        :rtype: QAction
        """
        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)
        action.setCheckable(checkable)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToRasterMenu(self.menu, action)

        self.actions.append(action)
        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icons_dir = os.path.join(self.plugin_dir, "icons")

        # Load Hyperspectral Data action
        self.add_action(
            os.path.join(icons_dir, "load_data.png"),
            text=self.tr("Load Hyperspectral Data"),
            callback=self.show_load_dialog,
            parent=self.iface.mainWindow(),
            status_tip=self.tr("Load hyperspectral data (EMIT, PACE, DESIS, etc.)"),
        )

        # Band Combination action
        self.add_action(
            os.path.join(icons_dir, "band_combination.png"),
            text=self.tr("Band Combination"),
            callback=self.show_band_dialog,
            parent=self.iface.mainWindow(),
            status_tip=self.tr("Change band combination for visualization"),
        )

        # Spectral Inspector action
        self.add_action(
            os.path.join(icons_dir, "hypercoast.png"),
            text=self.tr("Spectral Inspector"),
            callback=self.toggle_spectral_inspector,
            parent=self.iface.mainWindow(),
            status_tip=self.tr("Inspect spectral signatures interactively"),
            checkable=True,
        )

        # Install Dependencies action (menu only, no toolbar)
        self.add_action(
            ":/images/themes/default/mIconPythonFile.svg",
            text=self.tr("Install Dependencies..."),
            callback=self.show_dependency_installer,
            parent=self.iface.mainWindow(),
            status_tip=self.tr("Install required Python packages"),
            add_to_toolbar=False,
        )

        # Check for Updates action (menu only, no toolbar)
        self.add_action(
            ":/images/themes/default/mActionRefresh.svg",
            text=self.tr("Check for Updates..."),
            callback=self.show_update_checker,
            parent=self.iface.mainWindow(),
            status_tip=self.tr("Check for plugin updates from GitHub"),
            add_to_toolbar=False,
        )

        # About action
        self.add_action(
            os.path.join(icons_dir, "about.svg"),
            text=self.tr("About"),
            callback=self.show_about_dialog,
            parent=self.iface.mainWindow(),
            status_tip=self.tr("About HyperCoast plugin"),
        )

        # Auto-check dependencies after GUI is initialized
        QTimer.singleShot(1000, self._check_dependencies_on_startup)

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginRasterMenu(self.tr("&HyperCoast"), action)
            self.iface.removeToolBarIcon(action)

        # Remove the toolbar
        del self.toolbar

        # Clean up tools
        if self.spectral_tool:
            self.spectral_tool.deactivate()
            self.spectral_tool = None

        # Close dialogs
        if self.spectral_plot_dialog:
            self.spectral_plot_dialog.close()
            self.spectral_plot_dialog = None

    def _check_dependencies_on_startup(self):
        """Check if required dependencies are installed on plugin startup."""
        try:
            from . import venv_manager

            is_ready, message = venv_manager.get_venv_status(self.plugin_dir)

            if is_ready:
                QgsMessageLog.logMessage(
                    "Dependencies ready, activating venv packages...",
                    "HyperCoast",
                    Qgis.Info,
                )
                venv_manager.ensure_venv_packages_available()
                self._try_enable_data_dialogs()
            else:
                QgsMessageLog.logMessage(
                    f"Dependencies not ready: {message}",
                    "HyperCoast",
                    Qgis.Info,
                )
                self._deps_available = False

        except Exception as e:
            QgsMessageLog.logMessage(
                f"Error checking dependencies: {e}",
                "HyperCoast",
                Qgis.Warning,
            )

    def _try_enable_data_dialogs(self):
        """Try to import data dialogs and enable data actions.

        After venv activation, dialog modules that were already imported
        may have cached HAS_MATPLOTLIB=False etc. Force-reload them so
        the try/except import blocks inside those modules run again.
        """
        import importlib

        global _DATA_DIALOGS_AVAILABLE

        # Force-reload modules that cache HAS_* flags from try/except imports
        # at module level. Without this, xarray, matplotlib, rasterio etc.
        # remain marked as unavailable even after venv activation.
        # Order matters: reload hyperspectral_provider first (has HAS_XARRAY etc.),
        # then dialogs that import from it.
        try:
            from . import hyperspectral_provider
            from .dialogs import (
                load_data_dialog,
                band_combination_dialog,
                spectral_plot_dialog,
            )

            importlib.reload(hyperspectral_provider)
            importlib.reload(load_data_dialog)
            importlib.reload(band_combination_dialog)
            importlib.reload(spectral_plot_dialog)
        except Exception:
            pass

        # Reset cached dialog instances so they are re-created
        # with the reloaded module classes.
        self.load_dialog = None
        self.band_dialog = None
        self.spectral_tool = None
        self.spectral_plot_dialog = None

        if not _DATA_DIALOGS_AVAILABLE:
            # Try importing again now that venv packages are on sys.path
            try:
                from .dialogs.load_data_dialog import LoadDataDialog  # noqa: F811
                from .dialogs.band_combination_dialog import (  # noqa: F811
                    BandCombinationDialog,
                )
                from .dialogs.spectral_inspector_tool import (  # noqa: F811
                    SpectralInspectorTool,
                )
                from .dialogs.spectral_plot_dialog import (  # noqa: F811
                    SpectralPlotDialog,
                )

                _DATA_DIALOGS_AVAILABLE = True
            except ImportError as e:
                QgsMessageLog.logMessage(
                    f"Data dialogs still unavailable after venv activation: {e}",
                    "HyperCoast",
                    Qgis.Warning,
                )

        if _DATA_DIALOGS_AVAILABLE:
            self._deps_available = True
            self._set_data_actions_enabled(True)
            QgsMessageLog.logMessage(
                "All data features enabled",
                "HyperCoast",
                Qgis.Success,
            )
        else:
            self._deps_available = False
            self._set_data_actions_enabled(False)

    def _set_data_actions_enabled(self, enabled):
        """Enable or disable data-related actions.

        The first 3 actions are data-dependent:
        Load Data, Band Combination, Spectral Inspector.

        :param enabled: Whether to enable the actions.
        """
        for i, action in enumerate(self.actions):
            if i < 3:
                action.setEnabled(enabled)

    def show_dependency_installer(self):
        """Show the dependency installer dialog."""
        if self.deps_dialog is None:
            self.deps_dialog = DependencyInstallerDialog(
                self.plugin_dir, self.iface.mainWindow()
            )
            self.deps_dialog.deps_installed.connect(self._on_deps_installed)
        self.deps_dialog.show()
        self.deps_dialog.raise_()
        self.deps_dialog.activateWindow()

    def _on_deps_installed(self):
        """Handle successful dependency installation."""
        QgsMessageLog.logMessage(
            "Dependencies installed, activating...",
            "HyperCoast",
            Qgis.Info,
        )
        self._try_enable_data_dialogs()

    def show_load_dialog(self):
        """Show the load hyperspectral data dialog."""
        if not self._deps_available:
            self._show_deps_required_warning()
            return
        from .dialogs.load_data_dialog import LoadDataDialog

        if self.load_dialog is None:
            self.load_dialog = LoadDataDialog(self.iface, self)
        self.load_dialog.show()
        self.load_dialog.raise_()
        self.load_dialog.activateWindow()

    def show_band_dialog(self):
        """Show the band combination dialog."""
        if not self._deps_available:
            self._show_deps_required_warning()
            return
        from .dialogs.band_combination_dialog import BandCombinationDialog

        if self.band_dialog is None:
            self.band_dialog = BandCombinationDialog(self.iface, self)
        self.band_dialog.refresh_layers()
        self.band_dialog.show()
        self.band_dialog.raise_()
        self.band_dialog.activateWindow()

    def toggle_spectral_inspector(self):
        """Toggle the spectral inspector tool."""
        if not self._deps_available:
            self._show_deps_required_warning()
            return
        from .dialogs.spectral_inspector_tool import SpectralInspectorTool

        if self.spectral_tool is None:
            self.spectral_tool = SpectralInspectorTool(self.iface.mapCanvas(), self)

        if self.spectral_tool.is_active:
            self.spectral_tool.deactivate()
            self.iface.mapCanvas().unsetMapTool(self.spectral_tool)
        else:
            self.spectral_tool.activate()
            self.iface.mapCanvas().setMapTool(self.spectral_tool)
            # Show the spectral plot dialog
            self.show_spectral_plot()

    def show_spectral_plot(self):
        """Show the spectral plot dialog."""
        if not self._deps_available:
            self._show_deps_required_warning()
            return
        from .dialogs.spectral_plot_dialog import SpectralPlotDialog

        if self.spectral_plot_dialog is None:
            self.spectral_plot_dialog = SpectralPlotDialog(self.iface, self)
        self.spectral_plot_dialog.show()
        self.spectral_plot_dialog.raise_()
        self.spectral_plot_dialog.activateWindow()

    def _show_deps_required_warning(self):
        """Show a warning that dependencies need to be installed."""
        QMessageBox.warning(
            self.iface.mainWindow(),
            "Dependencies Required",
            "Required Python packages are not installed.\n\n"
            "Please use 'Install Dependencies' from the HyperCoast menu first.",
        )
        self.show_dependency_installer()

    def show_about_dialog(self):
        """Show the About dialog."""
        if self.about_dialog is None:
            self.about_dialog = AboutDialog(self.iface.mainWindow())
        self.about_dialog.show()
        self.about_dialog.raise_()
        self.about_dialog.activateWindow()

    def show_update_checker(self):
        """Show the update checker dialog."""
        if self.update_dialog is None:
            self.update_dialog = UpdateCheckerDialog(
                self.plugin_dir, self.iface.mainWindow()
            )
        self.update_dialog.show()
        self.update_dialog.raise_()
        self.update_dialog.activateWindow()

    def register_hyperspectral_layer(self, layer_id, data_info):
        """Register a hyperspectral layer with its metadata.

        :param layer_id: The QGIS layer ID
        :param data_info: Dictionary containing dataset info (xarray dataset, type, wavelengths, etc.)
        """
        self.hyperspectral_data[layer_id] = data_info

    def get_hyperspectral_data(self, layer_id):
        """Get hyperspectral data for a layer.

        :param layer_id: The QGIS layer ID
        :returns: Data info dictionary or None if not found
        """
        return self.hyperspectral_data.get(layer_id)

    def get_all_hyperspectral_layers(self):
        """Get all registered hyperspectral layers.

        :returns: Dictionary of all hyperspectral data
        """
        # Clean up layers that no longer exist
        project = QgsProject.instance()
        valid_layers = {}
        for layer_id, data_info in self.hyperspectral_data.items():
            if project.mapLayer(layer_id):
                valid_layers[layer_id] = data_info
        self.hyperspectral_data = valid_layers
        return self.hyperspectral_data
