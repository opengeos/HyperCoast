# -*- coding: utf-8 -*-
"""
HyperCoast QGIS Plugin - Main Plugin Class

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

import os

from qgis.PyQt.QtCore import QCoreApplication, Qt, QTimer
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QDockWidget, QMessageBox, QToolBar
from qgis.core import QgsApplication, QgsMessageLog, QgsProject, Qgis

# About and update checker have no heavy deps, always available
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


TOOLBAR_OBJECT_NAME = "HyperCoastToolbar"
MENU_TITLE = "&HyperCoast"
DOCK_OBJECT_NAMES = {
    "HyperCoastSettingsDock",
    "HyperCoastLoadDataDock",
    "HyperCoastBandCombinationDock",
    "HyperCoastSpectralPlotDock",
    "HyperCoastAboutDock",
    "HyperCoastUpdateDock",
    "HyperCoastDependencyInstallerDock",
}


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
        self._remove_toolbars_by_object_name()
        self._remove_menus_by_title()
        self._remove_docks_by_object_name()
        self.toolbar = self.iface.addToolBar("HyperCoast")
        self.toolbar.setObjectName(TOOLBAR_OBJECT_NAME)

        # Store reference to dialogs and dock widgets
        self.load_dialog = None
        self.band_dialog = None
        self.spectral_tool = None
        self.spectral_plot_dialog = None
        self.about_dialog = None
        self.update_dialog = None
        self._settings_dock = None
        self._dock_actions = {}
        self._registered_docks = []
        self.processing_provider = None

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
        self.load_action = self.add_action(
            os.path.join(icons_dir, "load_data.png"),
            text=self.tr("Load Hyperspectral Data"),
            callback=self.show_load_dialog,
            parent=self.iface.mainWindow(),
            status_tip=self.tr("Load hyperspectral data (EMIT, PACE, DESIS, etc.)"),
            checkable=True,
        )

        # Band Combination action
        self.band_action = self.add_action(
            os.path.join(icons_dir, "band_combination.png"),
            text=self.tr("Band Combination"),
            callback=self.show_band_dialog,
            parent=self.iface.mainWindow(),
            status_tip=self.tr("Change band combination for visualization"),
            checkable=True,
        )

        # Spectral Inspector action
        self.spectral_action = self.add_action(
            os.path.join(icons_dir, "hypercoast.png"),
            text=self.tr("Spectral Inspector"),
            callback=self.toggle_spectral_inspector,
            parent=self.iface.mainWindow(),
            status_tip=self.tr("Inspect spectral signatures interactively"),
            checkable=True,
        )

        # Install Dependencies action (checkable for dock toggle)
        self.settings_action = self.add_action(
            ":/images/themes/default/mIconPythonFile.svg",
            text=self.tr("Install Dependencies..."),
            callback=self.toggle_settings_dock,
            parent=self.iface.mainWindow(),
            status_tip=self.tr("Install required Python packages"),
            add_to_toolbar=False,
            checkable=True,
        )

        # Check for Updates action (menu only, no toolbar)
        self.update_action = self.add_action(
            ":/images/themes/default/mActionRefresh.svg",
            text=self.tr("Check for Updates..."),
            callback=self.show_update_checker,
            parent=self.iface.mainWindow(),
            status_tip=self.tr("Check for plugin updates from GitHub"),
            add_to_toolbar=False,
            checkable=True,
        )

        # About action
        self.about_action = self.add_action(
            os.path.join(icons_dir, "about.svg"),
            text=self.tr("About"),
            callback=self.show_about_dialog,
            parent=self.iface.mainWindow(),
            status_tip=self.tr("About HyperCoast plugin"),
            checkable=True,
        )

        # Auto-check dependencies after GUI is initialized
        QTimer.singleShot(1000, self._check_dependencies_on_startup)

    def _remove_toolbar(self, toolbar):
        """Detach and schedule deletion of a plugin toolbar widget."""
        if toolbar is None:
            return

        main_window = self.iface.mainWindow()
        actions = []
        try:
            actions = list(toolbar.actions())
        except Exception:
            pass  # nosec B110
        try:
            toolbar.clear()
        except Exception:
            pass  # nosec B110
        for action in actions:
            try:
                action.deleteLater()
            except Exception:
                pass  # nosec B110
        try:
            main_window.removeToolBar(toolbar)
        except Exception:
            pass  # nosec B110
        try:
            toolbar.hide()
        except Exception:
            pass  # nosec B110
        try:
            toolbar.setParent(None)
        except Exception:
            pass  # nosec B110
        try:
            toolbar.deleteLater()
        except Exception:
            pass  # nosec B110

    def _remove_toolbars_by_object_name(self):
        """Remove current or stale plugin toolbars from QGIS."""
        main_window = self.iface.mainWindow()
        for toolbar in main_window.findChildren(QToolBar, TOOLBAR_OBJECT_NAME):
            self._remove_toolbar(toolbar)

    def _plugin_menu_titles(self):
        """Return possible translated and untranslated plugin menu titles."""
        titles = {MENU_TITLE}
        translator = getattr(self, "tr", None)
        if callable(translator):
            try:
                titles.add(translator(MENU_TITLE))
            except Exception:
                pass  # nosec B110
        return titles

    def _remove_menu(self, menu):
        """Detach and schedule deletion of a plugin menu."""
        if menu is None:
            return

        main_window = self.iface.mainWindow()
        try:
            menu.clear()
        except Exception:
            pass  # nosec B110
        try:
            main_window.menuBar().removeAction(menu.menuAction())
        except Exception:
            pass  # nosec B110
        try:
            menu.setParent(None)
        except Exception:
            pass  # nosec B110
        try:
            menu.deleteLater()
        except Exception:
            pass  # nosec B110

    def _remove_menus_by_title(self):
        """Remove current or stale plugin menus from QGIS."""
        menu_bar = self.iface.mainWindow().menuBar()
        titles = self._plugin_menu_titles()
        for action in menu_bar.actions():
            menu = action.menu()
            if menu is not None and menu.title() in titles:
                self._remove_menu(menu)

    def _remove_dock(self, dock):
        """Detach and schedule deletion of a plugin dock widget."""
        if dock is None:
            return
        try:
            self.iface.removeDockWidget(dock)
        except Exception:
            pass  # nosec B110
        try:
            dock.hide()
        except Exception:
            pass  # nosec B110
        try:
            dock.setParent(None)
        except Exception:
            pass  # nosec B110
        try:
            dock.deleteLater()
        except Exception:
            pass  # nosec B110

    def _remove_docks_by_object_name(self):
        """Remove current or stale plugin dock widgets from QGIS."""
        main_window = self.iface.mainWindow()
        for dock in main_window.findChildren(QDockWidget):
            if dock.objectName() in DOCK_OBJECT_NAMES:
                self._remove_dock(dock)

    def _register_dock(self, dock, action):
        """Register a dock widget and sync it with a checkable action.

        Args:
            dock: Dock widget to manage.
            action: QAction that toggles dock visibility.
        """
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        dock.visibilityChanged.connect(
            lambda visible, dock=dock, action=action: self._on_dock_visibility_changed(
                dock, action, visible
            )
        )
        self.iface.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        for existing in self._registered_docks:
            if existing is not dock:
                try:
                    self.iface.mainWindow().tabifyDockWidget(existing, dock)
                    break
                except Exception:
                    pass  # nosec B110
        self._registered_docks.append(dock)
        self._dock_actions[dock.objectName()] = action

    def _on_dock_visibility_changed(self, dock, action, visible):
        """Keep dock action checked state aligned with dock visibility.

        Args:
            dock: Dock widget whose visibility changed.
            action: QAction associated with the dock.
            visible: Whether the dock is visible.
        """
        action.blockSignals(True)
        action.setChecked(visible)
        action.blockSignals(False)

    def _show_dock(self, attr_name, factory, action, before_show=None):
        """Create if needed, then show and raise a dock widget.

        Args:
            attr_name: Plugin attribute that stores the dock instance.
            factory: Callable that creates the dock widget.
            action: QAction associated with the dock.
            before_show: Optional callable invoked before showing the dock.

        Returns:
            The dock widget.
        """
        dock = getattr(self, attr_name, None)
        if dock is None:
            dock = factory()
            setattr(self, attr_name, dock)
            self._register_dock(dock, action)
        if before_show is not None:
            before_show(dock)
        dock.show()
        dock.raise_()
        action.setChecked(True)
        return dock

    def _toggle_dock(self, attr_name, factory, action, before_show=None):
        """Toggle a dock widget's visibility.

        Args:
            attr_name: Plugin attribute that stores the dock instance.
            factory: Callable that creates the dock widget.
            action: QAction associated with the dock.
            before_show: Optional callable invoked before showing the dock.

        Returns:
            The dock widget.
        """
        dock = getattr(self, attr_name, None)
        if dock is not None and dock.isVisible():
            dock.hide()
            return dock
        return self._show_dock(attr_name, factory, action, before_show)

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginRasterMenu(self.tr("&HyperCoast"), action)
            self.iface.removeToolBarIcon(action)

        # Remove the toolbar
        del self.toolbar

        self._remove_processing_provider()

        # Clean up dock widgets
        for attr_name in (
            "_settings_dock",
            "load_dialog",
            "band_dialog",
            "spectral_plot_dialog",
            "about_dialog",
            "update_dialog",
        ):
            dock = getattr(self, attr_name, None)
            if dock is not None:
                self._remove_dock(dock)
                setattr(self, attr_name, None)
        self._registered_docks.clear()
        self._dock_actions.clear()

        # Clean up tools
        if self.spectral_tool:
            self.spectral_tool.deactivate()
            self.spectral_tool = None

        # Close dialogs
        if self.spectral_plot_dialog:
            self.spectral_plot_dialog.close()
            self.spectral_plot_dialog = None

        self._remove_toolbars_by_object_name()
        self._remove_menus_by_title()
        self._remove_docks_by_object_name()

    def _check_dependencies_on_startup(self):
        """Check if required dependencies are installed on plugin startup."""
        try:
            from .core import venv_manager

            # Clean up old versioned venv directories from previous layout
            venv_manager.cleanup_old_venv_directories()

            is_ready, message = venv_manager.get_venv_status(self.plugin_dir)

            if is_ready:
                if venv_manager.using_conda_env_with_deps(self.plugin_dir):
                    startup_msg = (
                        "Conda environment detected; using Conda-provided "
                        "packages (skipping venv activation)"
                    )
                else:
                    startup_msg = "Dependencies ready, activating venv packages..."
                QgsMessageLog.logMessage(
                    startup_msg,
                    "HyperCoast",
                    Qgis.MessageLevel.Info,
                )
                venv_manager.ensure_venv_packages_available(self.plugin_dir)
                self._try_enable_data_dialogs()
            else:
                QgsMessageLog.logMessage(
                    f"Dependencies not ready: {message}",
                    "HyperCoast",
                    Qgis.MessageLevel.Info,
                )
                self._deps_available = False

        except Exception as e:
            QgsMessageLog.logMessage(
                f"Error checking dependencies: {e}",
                "HyperCoast",
                Qgis.MessageLevel.Warning,
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

        # Clear the hypercoast library cache so get_hypercoast() re-imports
        # with venv packages on sys.path (instead of returning a stale ref).
        try:
            from . import _hypercoast_lib

            _hypercoast_lib._CACHED = None
        except Exception:
            pass

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
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Error reloading data modules: {e}",
                "HyperCoast",
                Qgis.MessageLevel.Warning,
            )

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
                    Qgis.MessageLevel.Warning,
                )

        if _DATA_DIALOGS_AVAILABLE:
            self._deps_available = True
            self._set_data_actions_enabled(True)
            self._ensure_processing_provider()
            QgsMessageLog.logMessage(
                "All data features enabled",
                "HyperCoast",
                Qgis.MessageLevel.Success,
            )
        else:
            self._deps_available = False
            self._set_data_actions_enabled(False)
            self._remove_processing_provider()

    def _ensure_processing_provider(self):
        """Register the HyperCoast Processing provider if available."""
        if self.processing_provider is not None:
            return
        try:
            from .processing_provider import HyperCoastProcessingProvider

            self.processing_provider = HyperCoastProcessingProvider(self.plugin_dir)
            QgsApplication.processingRegistry().addProvider(self.processing_provider)
            QgsMessageLog.logMessage(
                "HyperCoast Processing provider registered",
                "HyperCoast",
                Qgis.MessageLevel.Info,
            )
        except Exception as e:
            self.processing_provider = None
            QgsMessageLog.logMessage(
                f"Could not register Processing provider: {e}",
                "HyperCoast",
                Qgis.MessageLevel.Warning,
            )

    def _remove_processing_provider(self):
        """Remove the HyperCoast Processing provider if registered."""
        if self.processing_provider is None:
            return
        try:
            QgsApplication.processingRegistry().removeProvider(self.processing_provider)
        except Exception:
            pass  # nosec B110
        self.processing_provider = None

    def _set_data_actions_enabled(self, enabled):
        """Enable or disable data-related actions.

        The first 3 actions are data-dependent:
        Load Data, Band Combination, Spectral Inspector.

        :param enabled: Whether to enable the actions.
        """
        for i, action in enumerate(self.actions):
            if i < 3:
                action.setEnabled(enabled)

    def toggle_settings_dock(self):
        """Toggle the Settings dock widget visibility."""
        try:
            from .dialogs.settings_dock import SettingsDockWidget

            self._toggle_dock(
                "_settings_dock",
                lambda: self._create_settings_dock(SettingsDockWidget),
                self.settings_action,
            )
        except Exception as e:
            QMessageBox.critical(
                self.iface.mainWindow(),
                "Error",
                f"Failed to create Settings panel:\n{str(e)}",
            )
            self.settings_action.setChecked(False)

    def _create_settings_dock(self, dock_class):
        """Create the Settings dock and connect dependency signals.

        Args:
            dock_class: Settings dock widget class.

        Returns:
            Settings dock widget.
        """
        dock = dock_class(self.plugin_dir, self.iface, self.iface.mainWindow())
        dock.setObjectName("HyperCoastSettingsDock")
        dock.deps_installed.connect(self._on_deps_installed)
        return dock

    def _on_settings_visibility_changed(self, visible):
        """Handle Settings dock visibility change.

        Args:
            visible: Whether the dock is now visible.
        """
        self.settings_action.setChecked(visible)

    def _on_deps_installed(self):
        """Handle successful dependency installation."""
        QgsMessageLog.logMessage(
            "Dependencies installed, activating...",
            "HyperCoast",
            Qgis.MessageLevel.Info,
        )
        self._try_enable_data_dialogs()

    def show_load_dialog(self):
        """Show the load hyperspectral data dialog."""
        if not self._deps_available:
            self._show_deps_required_warning()
            self.load_action.setChecked(False)
            return
        from .dialogs.load_data_dialog import LoadDataDialog

        self._toggle_dock(
            "load_dialog",
            lambda: LoadDataDialog(self.iface, self, self.iface.mainWindow()),
            self.load_action,
        )

    def show_band_dialog(self):
        """Show the band combination dialog."""
        if not self._deps_available:
            self._show_deps_required_warning()
            self.band_action.setChecked(False)
            return
        from .dialogs.band_combination_dialog import BandCombinationDialog

        self._toggle_dock(
            "band_dialog",
            lambda: BandCombinationDialog(self.iface, self, self.iface.mainWindow()),
            self.band_action,
            before_show=lambda dock: dock.refresh_layers(),
        )

    def toggle_spectral_inspector(self):
        """Toggle the spectral inspector tool."""
        if not self._deps_available:
            self._show_deps_required_warning()
            self.spectral_action.setChecked(False)
            return
        from .dialogs.spectral_inspector_tool import SpectralInspectorTool

        if self.spectral_tool is None:
            self.spectral_tool = SpectralInspectorTool(self.iface.mapCanvas(), self)

        if self.spectral_tool.is_active:
            self.spectral_tool.deactivate()
            self.iface.mapCanvas().unsetMapTool(self.spectral_tool)
            if self.spectral_plot_dialog is not None:
                self.spectral_plot_dialog.hide()
        else:
            self.spectral_tool.activate()
            self.iface.mapCanvas().setMapTool(self.spectral_tool)
            # Show the spectral plot dialog
            self.show_spectral_plot()

    def show_spectral_plot(self):
        """Show the spectral plot dialog."""
        if not self._deps_available:
            self._show_deps_required_warning()
            self.spectral_action.setChecked(False)
            return
        from .dialogs.spectral_plot_dialog import SpectralPlotDialog

        self._show_dock(
            "spectral_plot_dialog",
            lambda: SpectralPlotDialog(self.iface, self, self.iface.mainWindow()),
            self.spectral_action,
        )

    def _show_deps_required_warning(self):
        """Show a warning that dependencies need to be installed."""
        reply = QMessageBox.warning(
            self.iface.mainWindow(),
            "Dependencies Required",
            "Required Python packages are not installed.\n\n"
            "Would you like to open Settings to install them?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._open_settings_deps_tab()

    def _open_settings_deps_tab(self):
        """Open the Settings dock and refresh dependency status."""
        if self._settings_dock is None:
            try:
                from .dialogs.settings_dock import SettingsDockWidget

                self._settings_dock = self._create_settings_dock(SettingsDockWidget)
                self._register_dock(self._settings_dock, self.settings_action)
            except Exception as e:
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Error",
                    f"Failed to create Settings panel:\n{str(e)}",
                )
                return

        self._show_dock(
            "_settings_dock",
            lambda: self._settings_dock,
            self.settings_action,
        )
        self._settings_dock.show_dependencies_tab()

    def show_about_dialog(self):
        """Show the About dialog."""
        self._toggle_dock(
            "about_dialog",
            lambda: AboutDialog(self.iface.mainWindow()),
            self.about_action,
        )

    def show_update_checker(self):
        """Show the update checker dialog."""
        self._toggle_dock(
            "update_dialog",
            lambda: UpdateCheckerDialog(self.plugin_dir, self.iface.mainWindow()),
            self.update_action,
        )

    def register_hyperspectral_layer(self, layer_id, data_info):
        """Register a hyperspectral layer with its metadata.

        :param layer_id: The QGIS layer ID
        :param data_info: Dictionary containing dataset info.
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
