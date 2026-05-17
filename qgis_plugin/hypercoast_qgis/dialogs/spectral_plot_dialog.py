# -*- coding: utf-8 -*-
"""
HyperCoast QGIS Plugin - Spectral Plot Dialog

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

import csv
import numpy as np
from qgis.PyQt.QtCore import Qt, QTimer
from qgis.PyQt.QtWidgets import (
    QDockWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QGroupBox,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QSizePolicy,
    QFileDialog,
    QMessageBox,
    QSplitter,
    QWidget,
    QTabWidget,
)

try:
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qtagg import (
            NavigationToolbar2QT as NavigationToolbar,
        )
    except ImportError:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import (
            NavigationToolbar2QT as NavigationToolbar,
        )
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

DEFAULT_Y_RANGE = (0.0, 0.5)
TANAGER_Y_RANGE = (0.0, 100.0)


class SpectralPlotDialog(QDockWidget):
    """Dockable panel for displaying spectral plots."""

    def __init__(self, iface, plugin, parent=None):
        """Initialize the dialog.

        :param iface: QGIS interface
        :param plugin: Reference to the main plugin
        :param parent: Parent widget
        """
        super().__init__(parent or iface.mainWindow())
        self.iface = iface
        self.plugin = plugin

        self.setWindowTitle("Spectral Plot")
        self.setObjectName("HyperCoastSpectralPlotDock")
        self.setMinimumWidth(560)
        self.setMinimumHeight(480)

        # Store spectra
        self.spectra = []
        self.colors = plt.cm.tab10.colors if HAS_MATPLOTLIB else []

        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        content = QWidget(self)
        self.setWidget(content)
        layout = QVBoxLayout(content)

        if not HAS_MATPLOTLIB:
            layout.addWidget(
                QLabel(
                    "Matplotlib is required for spectral plots.\n"
                    "Please install matplotlib: pip install matplotlib"
                )
            )
            return

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Top: Plot area
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)

        # Create matplotlib figure
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # Navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        splitter.addWidget(plot_widget)

        # Bottom: Controls and data table
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)

        # Tab widget for controls and data
        tab_widget = QTabWidget()

        # Tab 1: Plot options
        options_tab = QWidget()
        options_layout = QVBoxLayout(options_tab)
        policy = getattr(QSizePolicy, "Policy", QSizePolicy)

        # Sampling options
        sampling_group = QGroupBox("Sampling")
        sampling_group.setSizePolicy(policy.Expanding, policy.Maximum)
        sampling_layout = QHBoxLayout()
        sampling_layout.setContentsMargins(8, 6, 8, 6)
        sampling_layout.setSpacing(6)

        self.layer_combo = QComboBox()
        self.layer_combo.setMinimumWidth(180)
        self.layer_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.layer_combo.setSizePolicy(policy.Expanding, policy.Fixed)
        self.layer_combo.currentIndexChanged.connect(self._on_selected_layer_changed)
        sampling_layout.addWidget(QLabel("Layer:"))
        sampling_layout.addWidget(self.layer_combo, 1)

        refresh_layers_btn = QPushButton("Refresh Layers")
        refresh_layers_btn.clicked.connect(self.refresh_layer_combo)
        sampling_layout.addWidget(refresh_layers_btn)

        sampling_group.setLayout(sampling_layout)
        sampling_group.setMaximumHeight(sampling_group.sizeHint().height())
        options_layout.addWidget(sampling_group)

        option_groups_layout = QHBoxLayout()

        # Plot options
        options_group = QGroupBox("Plot Options")
        options_form = QFormLayout()

        self.stack_check = QCheckBox("Stack Spectra")
        self.stack_check.setChecked(False)
        self.stack_check.stateChanged.connect(self.update_plot)
        options_form.addRow("", self.stack_check)

        self.legend_check = QCheckBox("Show Legend")
        self.legend_check.setChecked(True)
        self.legend_check.stateChanged.connect(self.update_plot)
        options_form.addRow("", self.legend_check)

        self.grid_check = QCheckBox("Show Grid")
        self.grid_check.setChecked(True)
        self.grid_check.stateChanged.connect(self.update_plot)
        options_form.addRow("", self.grid_check)

        options_group.setLayout(options_form)
        option_groups_layout.addWidget(options_group, 1)

        # Axis options
        axis_group = QGroupBox("Axis Range")
        axis_form = QFormLayout()

        xlim_layout = QHBoxLayout()
        self.xmin_spin = QDoubleSpinBox()
        self.xmin_spin.setRange(0, 5000)
        self.xmin_spin.setValue(400)
        xlim_layout.addWidget(self.xmin_spin)
        xlim_layout.addWidget(QLabel("-"))
        self.xmax_spin = QDoubleSpinBox()
        self.xmax_spin.setRange(0, 5000)
        self.xmax_spin.setValue(2500)
        xlim_layout.addWidget(self.xmax_spin)
        xlim_layout.addWidget(QLabel("nm"))
        axis_form.addRow("X Range:", xlim_layout)

        ylim_layout = QHBoxLayout()
        self.ymin_spin = QDoubleSpinBox()
        self.ymin_spin.setRange(-1000, 10000)
        self.ymin_spin.setValue(DEFAULT_Y_RANGE[0])
        self.ymin_spin.setDecimals(4)
        ylim_layout.addWidget(self.ymin_spin)
        ylim_layout.addWidget(QLabel("-"))
        self.ymax_spin = QDoubleSpinBox()
        self.ymax_spin.setRange(-1000, 10000)
        self.ymax_spin.setValue(DEFAULT_Y_RANGE[1])
        self.ymax_spin.setDecimals(4)
        ylim_layout.addWidget(self.ymax_spin)
        axis_form.addRow("Y Range:", ylim_layout)

        self.auto_scale = QCheckBox("Auto Scale")
        self.auto_scale.setChecked(True)
        axis_form.addRow("", self.auto_scale)

        apply_range_btn = QPushButton("Apply Range")
        apply_range_btn.clicked.connect(self.apply_axis_range)
        axis_form.addRow("", apply_range_btn)

        axis_group.setLayout(axis_form)
        option_groups_layout.addWidget(axis_group, 2)

        # Labels
        labels_group = QGroupBox("Labels")
        labels_form = QFormLayout()

        self.xlabel_combo = QComboBox()
        self.xlabel_combo.addItems(
            ["Wavelength (nm)", "Band Number", "Frequency (THz)"]
        )
        self.xlabel_combo.currentIndexChanged.connect(self.update_plot)
        labels_form.addRow("X Label:", self.xlabel_combo)

        self.ylabel_combo = QComboBox()
        self.ylabel_combo.addItems(
            [
                "Reflectance",
                "Radiance",
                "Value",
                "Normalized Reflectance",
                "Transmittance",
            ]
        )
        self.ylabel_combo.currentIndexChanged.connect(self.update_plot)
        labels_form.addRow("Y Label:", self.ylabel_combo)

        labels_group.setLayout(labels_form)
        option_groups_layout.addWidget(labels_group, 2)
        options_layout.addLayout(option_groups_layout)

        tab_widget.addTab(options_tab, "Options")

        # Tab 2: Data table
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)

        self.data_table = QTableWidget()
        self.data_table.setColumnCount(4)
        self.data_table.setHorizontalHeaderLabels(
            ["Latitude", "Longitude", "Layer", "# Bands"]
        )
        self.data_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.data_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        data_layout.addWidget(self.data_table)

        table_buttons = QHBoxLayout()

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_selected_spectrum)
        table_buttons.addWidget(remove_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_all_spectra)
        table_buttons.addWidget(clear_btn)

        data_layout.addLayout(table_buttons)

        tab_widget.addTab(data_tab, "Data")

        bottom_layout.addWidget(tab_widget)

        # Export buttons
        export_layout = QHBoxLayout()

        export_csv_btn = QPushButton("Export to CSV")
        export_csv_btn.clicked.connect(self.export_to_csv)
        export_layout.addWidget(export_csv_btn)

        export_plot_btn = QPushButton("Export Plot")
        export_plot_btn.clicked.connect(self.export_plot)
        export_layout.addWidget(export_plot_btn)

        export_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        export_layout.addWidget(close_btn)

        bottom_layout.addLayout(export_layout)

        splitter.addWidget(bottom_widget)

        # Set splitter sizes
        splitter.setSizes([400, 200])

        layout.addWidget(splitter)

        self.refresh_layer_combo()

        # Defer the first draw until Qt has assigned the dock/canvas geometry.
        QTimer.singleShot(0, self.update_plot)

    def refresh_layer_combo(self):
        """Refresh the spectral sampling layer selector."""
        if not hasattr(self, "layer_combo"):
            return

        current_layer_id = self.selected_layer_id()
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        self.layer_combo.addItem("All HyperCoast layers", None)

        for layer_id in self.plugin.get_all_hyperspectral_layers():
            layer = self._project_layer(layer_id)
            if layer is not None:
                self.layer_combo.addItem(layer.name(), layer_id)

        if current_layer_id:
            index = self.layer_combo.findData(current_layer_id)
            if index >= 0:
                self.layer_combo.setCurrentIndex(index)

        self.layer_combo.blockSignals(False)
        self._apply_data_type_plot_defaults(self._selected_layer_data_type())

    def selected_layer_id(self):
        """Return the selected sampling layer ID.

        Returns:
            QGIS layer ID, or None to sample every HyperCoast layer.
        """
        if not hasattr(self, "layer_combo"):
            return None
        return self.layer_combo.currentData()

    def _on_selected_layer_changed(self, *args):
        """Apply plot defaults when the selected sampling layer changes."""
        self._apply_data_type_plot_defaults(self._selected_layer_data_type())

    def _selected_layer_data_type(self):
        """Return the data type represented by the selected sampling layer.

        Returns:
            HyperCoast data type string, or ``None`` for mixed/unknown layers.
        """
        layer_id = self.selected_layer_id()
        if layer_id:
            data_info = self.plugin.get_hyperspectral_data(layer_id) or {}
            return data_info.get("data_type")

        data_types = {
            (self.plugin.get_hyperspectral_data(layer_id) or {}).get("data_type")
            for layer_id in self.plugin.get_all_hyperspectral_layers()
        }
        data_types.discard(None)
        if len(data_types) == 1:
            return next(iter(data_types))
        return None

    def _apply_data_type_plot_defaults(self, data_type):
        """Apply axis label and range defaults for a data type.

        Args:
            data_type: HyperCoast data type string.
        """
        if data_type != "Tanager":
            return

        index = self.ylabel_combo.findText("Radiance")
        if index >= 0:
            self.ylabel_combo.setCurrentIndex(index)
        self.ymin_spin.setValue(TANAGER_Y_RANGE[0])
        self.ymax_spin.setValue(TANAGER_Y_RANGE[1])

    def _project_layer(self, layer_id):
        """Return a QGIS project layer by ID.

        Args:
            layer_id: QGIS layer ID.

        Returns:
            QGIS layer object, or None.
        """
        try:
            from qgis.core import QgsProject

            return QgsProject.instance().mapLayer(layer_id)
        except Exception:
            return None

    def add_spectrum(self, lat, lon, wavelengths, values, layer_name, data_type=None):
        """Add a new spectrum to the plot.

        :param lat: Latitude
        :param lon: Longitude
        :param wavelengths: Array of wavelengths
        :param values: Array of spectral values
        :param layer_name: Name of the source layer
        :param data_type: Optional HyperCoast data type.
        """
        self._apply_data_type_plot_defaults(data_type)
        spectrum = {
            "lat": lat,
            "lon": lon,
            "wavelengths": np.array(wavelengths),
            "values": np.array(values),
            "layer": layer_name,
            "label": f"({lat:.4f}, {lon:.4f})",
            "data_type": data_type,
        }

        self.spectra.append(spectrum)

        # Add to table
        row = self.data_table.rowCount()
        self.data_table.insertRow(row)
        self.data_table.setItem(row, 0, QTableWidgetItem(f"{lat:.6f}"))
        self.data_table.setItem(row, 1, QTableWidgetItem(f"{lon:.6f}"))
        self.data_table.setItem(row, 2, QTableWidgetItem(layer_name))
        self.data_table.setItem(row, 3, QTableWidgetItem(str(len(wavelengths))))

        # Update plot
        self.update_plot()

    def update_plot(self):
        """Update the spectral plot."""
        if not HAS_MATPLOTLIB:
            return

        self.ax.clear()

        if not self.spectra:
            self.ax.set_xlabel(self.xlabel_combo.currentText())
            self.ax.set_ylabel(self.ylabel_combo.currentText())
            self.ax.set_title("Spectral Signatures")
            self.ax.text(
                0.5,
                0.5,
                "Click on map to extract spectra",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
                fontsize=12,
                color="gray",
            )
            self._draw_canvas()
            return

        # Plot each spectrum
        for i, spectrum in enumerate(self.spectra):
            color = self.colors[i % len(self.colors)]

            wavelengths = spectrum["wavelengths"]
            values = self._transform_values(spectrum["values"])
            x_values = self._transform_x_values(wavelengths)

            # Filter negative values (often used as nodata)
            mask = ~np.isnan(values) & (values > -0.1)
            if np.any(mask):
                plot_values = values[mask]
                if self.stack_check.isChecked():
                    span = np.nanmax(plot_values) - np.nanmin(plot_values)
                    offset = (span if span > 0 else 0.1) * i
                    plot_values = plot_values + offset
                self.ax.plot(
                    x_values[mask],
                    plot_values,
                    color=color,
                    label=spectrum["label"],
                    linewidth=1.5,
                )

        # Set labels
        self.ax.set_xlabel(self.xlabel_combo.currentText())
        ylabel = self.ylabel_combo.currentText()
        title = "Spectral Signatures"
        if self.stack_check.isChecked():
            ylabel = f"{ylabel} + display offset"
            title = "Spectral Signatures (Stacked)"
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)

        # Grid
        if self.grid_check.isChecked():
            self.ax.grid(True, alpha=0.3)

        # Legend
        if self.legend_check.isChecked() and self.spectra:
            self.ax.legend(loc="upper right", fontsize=8)

        # Axis range
        if not self.auto_scale.isChecked():
            self.ax.set_xlim(self.xmin_spin.value(), self.xmax_spin.value())
            self.ax.set_ylim(self.ymin_spin.value(), self.ymax_spin.value())

        self.figure.tight_layout()
        self._draw_canvas()

    def apply_axis_range(self):
        """Apply custom axis range."""
        self.auto_scale.setChecked(False)
        self.update_plot()

    def _transform_x_values(self, wavelengths):
        """Return x-axis values for the selected axis mode.

        Args:
            wavelengths: Wavelength array in nanometers.

        Returns:
            Numpy array for plotting.
        """
        mode = self.xlabel_combo.currentText()
        wavelengths = np.array(wavelengths, dtype=float)
        if mode == "Band Number":
            return np.arange(1, len(wavelengths) + 1)
        if mode == "Frequency (THz)":
            with np.errstate(divide="ignore", invalid="ignore"):
                return 299792.458 / wavelengths
        return wavelengths

    def _draw_canvas(self):
        """Draw the matplotlib canvas when its Qt geometry is valid."""
        if not HAS_MATPLOTLIB:
            return
        if self.canvas.width() <= 0 or self.canvas.height() <= 0:
            QTimer.singleShot(50, self.update_plot)
            return
        try:
            self.canvas.draw_idle()
        except ValueError as exc:
            if "Invalid affine transformation matrix" not in str(exc):
                raise
            QTimer.singleShot(50, self.update_plot)

    def _transform_values(self, values):
        """Return y-axis values for the selected value mode.

        Args:
            values: Raw spectral values.

        Returns:
            Numpy array for plotting.
        """
        values = np.array(values, dtype=float)
        if self.ylabel_combo.currentText() != "Normalized Reflectance":
            return values
        valid = values[~np.isnan(values)]
        if valid.size == 0:
            return values
        vmin = np.nanmin(valid)
        vmax = np.nanmax(valid)
        if vmax <= vmin:
            return values * 0
        return (values - vmin) / (vmax - vmin)

    def remove_selected_spectrum(self):
        """Remove the selected spectrum from the plot."""
        selected_rows = set()
        for item in self.data_table.selectedItems():
            selected_rows.add(item.row())

        # Remove in reverse order to maintain correct indices
        for row in sorted(selected_rows, reverse=True):
            if row < len(self.spectra):
                del self.spectra[row]
                self.data_table.removeRow(row)

        self.update_plot()

    def clear_all_spectra(self):
        """Clear all spectra from the plot."""
        self.spectra.clear()
        self.data_table.setRowCount(0)

        # Clear markers from inspector tool
        if self.plugin.spectral_tool:
            self.plugin.spectral_tool.clear_points()

        self.update_plot()

    def export_to_csv(self):
        """Export spectra to CSV file."""
        if not self.spectra:
            QMessageBox.warning(self, "Warning", "No spectra to export.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Spectra to CSV", "", "CSV Files (*.csv);;All Files (*.*)"
        )

        if not filepath:
            return

        try:
            # Find the union of all wavelengths
            all_wavelengths = set()
            for spectrum in self.spectra:
                all_wavelengths.update(spectrum["wavelengths"].tolist())
            wavelengths = sorted(all_wavelengths)

            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)

                # Header
                header = ["wavelength"]
                for spectrum in self.spectra:
                    header.append(f"({spectrum['lat']:.4f} {spectrum['lon']:.4f})")
                writer.writerow(header)

                # Data rows
                for wl in wavelengths:
                    row = [wl]
                    for spectrum in self.spectra:
                        idx = np.argmin(np.abs(spectrum["wavelengths"] - wl))
                        if np.abs(spectrum["wavelengths"][idx] - wl) < 1:
                            row.append(spectrum["values"][idx])
                        else:
                            row.append("")
                    writer.writerow(row)

            QMessageBox.information(self, "Success", f"Spectra exported to {filepath}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")

    def export_plot(self):
        """Export the plot as an image."""
        if not HAS_MATPLOTLIB:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Plot",
            "",
            "PNG Image (*.png);;PDF Document (*.pdf);;SVG Image (*.svg);;All Files (*.*)",
        )

        if not filepath:
            return

        try:
            self.figure.savefig(filepath, dpi=300, bbox_inches="tight")
            QMessageBox.information(self, "Success", f"Plot exported to {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export plot: {str(e)}")

    def closeEvent(self, event):
        """Handle dialog close event."""
        # Deactivate spectral tool if active
        if self.plugin.spectral_tool and self.plugin.spectral_tool.is_active:
            self.plugin.spectral_tool.deactivate()
            self.iface.mapCanvas().unsetMapTool(self.plugin.spectral_tool)

            # Uncheck the action
            for action in self.plugin.actions:
                if action.text() == self.plugin.tr("Spectral Inspector"):
                    action.setChecked(False)

        event.accept()
