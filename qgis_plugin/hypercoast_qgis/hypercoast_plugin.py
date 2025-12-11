# -*- coding: utf-8 -*-
"""
HyperCoast QGIS Plugin - Main Plugin Class

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

import os
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMenu, QToolBar
from qgis.core import QgsProject

# Import dialogs
from .dialogs.load_data_dialog import LoadDataDialog
from .dialogs.band_combination_dialog import BandCombinationDialog
from .dialogs.spectral_inspector_tool import SpectralInspectorTool
from .dialogs.spectral_plot_dialog import SpectralPlotDialog


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

        icon_path = os.path.join(self.plugin_dir, "icons", "hypercoast.png")

        # Load Hyperspectral Data action
        self.add_action(
            icon_path,
            text=self.tr("Load Hyperspectral Data"),
            callback=self.show_load_dialog,
            parent=self.iface.mainWindow(),
            status_tip=self.tr("Load hyperspectral data (EMIT, PACE, DESIS, etc.)"),
        )

        # Band Combination action
        self.add_action(
            icon_path,
            text=self.tr("Band Combination"),
            callback=self.show_band_dialog,
            parent=self.iface.mainWindow(),
            status_tip=self.tr("Change band combination for visualization"),
        )

        # Spectral Inspector action
        self.add_action(
            icon_path,
            text=self.tr("Spectral Inspector"),
            callback=self.toggle_spectral_inspector,
            parent=self.iface.mainWindow(),
            status_tip=self.tr("Inspect spectral signatures interactively"),
            checkable=True,
        )

        # Separator
        self.toolbar.addSeparator()

        # Show Spectral Plot Window
        self.add_action(
            icon_path,
            text=self.tr("Show Spectral Plot"),
            callback=self.show_spectral_plot,
            parent=self.iface.mainWindow(),
            status_tip=self.tr("Show/hide spectral plot window"),
            add_to_toolbar=False,
        )

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

    def show_load_dialog(self):
        """Show the load hyperspectral data dialog."""
        if self.load_dialog is None:
            self.load_dialog = LoadDataDialog(self.iface, self)
        self.load_dialog.show()
        self.load_dialog.raise_()
        self.load_dialog.activateWindow()

    def show_band_dialog(self):
        """Show the band combination dialog."""
        if self.band_dialog is None:
            self.band_dialog = BandCombinationDialog(self.iface, self)
        self.band_dialog.refresh_layers()
        self.band_dialog.show()
        self.band_dialog.raise_()
        self.band_dialog.activateWindow()

    def toggle_spectral_inspector(self):
        """Toggle the spectral inspector tool."""
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
        if self.spectral_plot_dialog is None:
            self.spectral_plot_dialog = SpectralPlotDialog(self.iface, self)
        self.spectral_plot_dialog.show()
        self.spectral_plot_dialog.raise_()
        self.spectral_plot_dialog.activateWindow()

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
