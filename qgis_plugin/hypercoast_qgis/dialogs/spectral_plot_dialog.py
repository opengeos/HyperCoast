# -*- coding: utf-8 -*-
"""
HyperCoast QGIS Plugin - Spectral Plot Dialog

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

import csv
import numpy as np
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QGroupBox,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QFormLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFileDialog,
    QMessageBox,
    QSplitter,
    QWidget,
    QTabWidget,
)

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import (
        NavigationToolbar2QT as NavigationToolbar,
    )
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class SpectralPlotDialog(QDialog):
    """Dialog for displaying spectral plots."""

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
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)

        # Store spectra
        self.spectra = []
        self.colors = plt.cm.tab10.colors if HAS_MATPLOTLIB else []

        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        if not HAS_MATPLOTLIB:
            layout.addWidget(
                QLabel(
                    "Matplotlib is required for spectral plots.\n"
                    "Please install matplotlib: pip install matplotlib"
                )
            )
            return

        # Main splitter
        splitter = QSplitter(Qt.Vertical)

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
        options_layout = QHBoxLayout(options_tab)

        # Plot options
        options_group = QGroupBox("Plot Options")
        options_form = QFormLayout()

        self.stack_check = QCheckBox("Stack Spectra")
        self.stack_check.setChecked(True)
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
        options_layout.addWidget(options_group)

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
        self.ymin_spin.setRange(-1, 10)
        self.ymin_spin.setValue(0)
        self.ymin_spin.setDecimals(4)
        ylim_layout.addWidget(self.ymin_spin)
        ylim_layout.addWidget(QLabel("-"))
        self.ymax_spin = QDoubleSpinBox()
        self.ymax_spin.setRange(-1, 10)
        self.ymax_spin.setValue(0.5)
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
        options_layout.addWidget(axis_group)

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
        options_layout.addWidget(labels_group)

        tab_widget.addTab(options_tab, "Options")

        # Tab 2: Data table
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)

        self.data_table = QTableWidget()
        self.data_table.setColumnCount(4)
        self.data_table.setHorizontalHeaderLabels(
            ["Latitude", "Longitude", "Layer", "# Bands"]
        )
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.data_table.setSelectionBehavior(QTableWidget.SelectRows)
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

        # Initialize empty plot
        self.update_plot()

    def add_spectrum(self, lat, lon, wavelengths, values, layer_name):
        """Add a new spectrum to the plot.

        :param lat: Latitude
        :param lon: Longitude
        :param wavelengths: Array of wavelengths
        :param values: Array of spectral values
        :param layer_name: Name of the source layer
        """
        spectrum = {
            "lat": lat,
            "lon": lon,
            "wavelengths": np.array(wavelengths),
            "values": np.array(values),
            "layer": layer_name,
            "label": f"({lat:.4f}, {lon:.4f})",
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
            self.canvas.draw()
            return

        # Plot each spectrum
        for i, spectrum in enumerate(self.spectra):
            color = self.colors[i % len(self.colors)]

            wavelengths = spectrum["wavelengths"]
            values = spectrum["values"]

            # Filter negative values (often used as nodata)
            mask = ~np.isnan(values) & (values > -0.1)
            if np.any(mask):
                self.ax.plot(
                    wavelengths[mask],
                    values[mask],
                    color=color,
                    label=spectrum["label"],
                    linewidth=1.5,
                )

        # Set labels
        self.ax.set_xlabel(self.xlabel_combo.currentText())
        self.ax.set_ylabel(self.ylabel_combo.currentText())
        self.ax.set_title("Spectral Signatures")

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
        self.canvas.draw()

    def apply_axis_range(self):
        """Apply custom axis range."""
        self.auto_scale.setChecked(False)
        self.update_plot()

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
