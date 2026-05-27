# -*- coding: utf-8 -*-
"""
HyperCoast QGIS Plugin - Workflow Builder Dialog

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

import os

import numpy as np
from qgis.core import QgsProject, QgsRasterLayer
from qgis.PyQt.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QWidget,
)

from ..cache_manager import create_generated_raster_path
from ..hyperspectral_provider import HAS_RASTERIO

WORKFLOW_LABELS = {
    "NDWI water mask": "ndwi",
    "Chlorophyll-a proxy": "chlorophyll",
    "Turbidity proxy": "turbidity",
    "CDOM proxy": "cdom",
    "Cyanobacteria proxy": "cyanobacteria",
    "Spectral anomaly": "anomaly",
}


class WorkflowBuilderDialog(QDockWidget):
    """Dockable panel for running workflow presets on loaded layers."""

    def __init__(self, iface, plugin, parent=None):
        """Initialize the workflow builder dialog.

        Args:
            iface: QGIS interface.
            plugin: HyperCoast plugin instance.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.iface = iface
        self.plugin = plugin
        self.setWindowTitle("Workflow Builder")
        self.setObjectName("HyperCoastWorkflowBuilderDock")

        widget = QWidget()
        layout = QFormLayout(widget)

        self.layer_combo = QComboBox()
        self.variable_combo = QComboBox()
        self.workflow_combo = QComboBox()
        self.workflow_combo.addItems(list(WORKFLOW_LABELS))
        self.first_wavelength_edit = QLineEdit()
        self.second_wavelength_edit = QLineEdit()
        self.output_edit = QLineEdit()

        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit)
        browse_btn = QPushButton("...")
        browse_btn.clicked.connect(self.browse_output)
        output_row.addWidget(browse_btn)

        self.run_btn = QPushButton("Run Workflow")
        self.run_btn.clicked.connect(self.run_workflow)
        self.status_label = QLabel("")

        layout.addRow("Layer:", self.layer_combo)
        layout.addRow("Variable:", self.variable_combo)
        layout.addRow("Workflow:", self.workflow_combo)
        layout.addRow("First wavelength:", self.first_wavelength_edit)
        layout.addRow("Second wavelength:", self.second_wavelength_edit)
        layout.addRow("Output GeoTIFF:", output_row)
        layout.addRow("", self.run_btn)
        layout.addRow("", self.status_label)

        self.setWidget(widget)
        self.layer_combo.currentIndexChanged.connect(self.populate_variables)
        self.workflow_combo.currentTextChanged.connect(self._set_default_wavelengths)
        self.refresh_layers()
        self._set_default_wavelengths(self.workflow_combo.currentText())

    def refresh_layers(self):
        """Refresh the loaded HyperCoast layer list."""
        current = self.layer_combo.currentData()
        self.layer_combo.clear()
        for layer_id, data_info in self.plugin.get_all_hyperspectral_layers().items():
            layer = QgsProject.instance().mapLayer(layer_id)
            name = (
                layer.name()
                if layer is not None and hasattr(layer, "name")
                else layer_id
            )
            self.layer_combo.addItem(name, layer_id)
        if current:
            index = self.layer_combo.findData(current)
            if index >= 0:
                self.layer_combo.setCurrentIndex(index)
        self.populate_variables()

    def populate_variables(self):
        """Populate data variables for the selected layer."""
        self.variable_combo.clear()
        layer_id = self.layer_combo.currentData()
        if not layer_id:
            return
        dataset = self.plugin.ensure_hyperspectral_dataset(layer_id)
        if dataset is None:
            return
        variables = dataset.list_data_variables()
        self.variable_combo.addItems(variables)
        data_info = self.plugin.get_hyperspectral_data(layer_id) or {}
        selected = data_info.get("selected_variable")
        if selected:
            index = self.variable_combo.findText(selected)
            if index >= 0:
                self.variable_combo.setCurrentIndex(index)

    def browse_output(self):
        """Choose an output GeoTIFF path."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Workflow Raster",
            "",
            "GeoTIFF Files (*.tif *.tiff);;All Files (*.*)",
        )
        if path:
            self.output_edit.setText(path)

    def run_workflow(self):
        """Run the selected workflow and add the output layer."""
        if not HAS_RASTERIO:
            QMessageBox.critical(self, "HyperCoast", "rasterio is required.")
            return
        layer_id = self.layer_combo.currentData()
        if not layer_id:
            QMessageBox.warning(self, "HyperCoast", "Select a HyperCoast layer.")
            return
        dataset = self.plugin.ensure_hyperspectral_dataset(layer_id)
        if dataset is None or dataset.dataset is None:
            QMessageBox.warning(
                self, "HyperCoast", "Could not load the selected dataset."
            )
            return

        variable = self.variable_combo.currentText() or None
        dataset.set_selected_variable(variable)
        workflow = WORKFLOW_LABELS[self.workflow_combo.currentText()]
        output_path = self.output_edit.text().strip() or self._default_output_path()
        self.output_edit.setText(output_path)

        try:
            from hypercoast.workflows import apply_workflow

            overrides = self._wavelength_overrides()
            result = apply_workflow(
                dataset.dataset,
                workflow,
                variable=variable,
                wavelengths=overrides,
            )
            self._write_workflow_raster(dataset, result, output_path)
            raster_layer = QgsRasterLayer(output_path, os.path.basename(output_path))
            if raster_layer.isValid():
                QgsProject.instance().addMapLayer(raster_layer)
            self.status_label.setText("Workflow completed")
        except Exception as exc:
            QMessageBox.critical(self, "HyperCoast", f"Workflow failed: {exc}")

    def _set_default_wavelengths(self, label):
        """Populate default wavelength override fields."""
        defaults = {
            "NDWI water mask": ("560", "860"),
            "Chlorophyll-a proxy": ("560", "665"),
            "Turbidity proxy": ("665", "860"),
            "CDOM proxy": ("412", "555"),
            "Cyanobacteria proxy": ("709", "665"),
            "Spectral anomaly": ("", ""),
        }
        first, second = defaults.get(label, ("", ""))
        self.first_wavelength_edit.setText(first)
        self.second_wavelength_edit.setText(second)

    def _wavelength_overrides(self):
        """Return optional workflow wavelength overrides."""
        if self.workflow_combo.currentText() == "Spectral anomaly":
            return {}
        first = self.first_wavelength_edit.text().strip()
        second = self.second_wavelength_edit.text().strip()
        overrides = {}
        if first:
            overrides["a"] = float(first)
        if second:
            overrides["b"] = float(second)
        return overrides

    def _default_output_path(self):
        """Return a project cache output path."""
        layer_name = self.layer_combo.currentText() or "workflow"
        suffix = WORKFLOW_LABELS[self.workflow_combo.currentText()]
        return create_generated_raster_path(
            layer_name,
            suffix,
            project=QgsProject.instance(),
        )

    def _write_workflow_raster(self, dataset, result, output_path):
        """Write a workflow DataArray to GeoTIFF."""
        import rasterio
        from rasterio.transform import from_bounds

        arr = np.asarray(result.values, dtype="float32")
        if arr.ndim != 2:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D workflow output, got shape {arr.shape}")
        height, width = arr.shape
        bounds = dataset.bounds or (0, 0, width, height)
        transform = from_bounds(*bounds, width, height)
        nodata = -9999.0
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs=dataset.crs or "EPSG:4326",
            transform=transform,
            nodata=nodata,
            tiled=True,
            compress="deflate",
        ) as dst:
            dst.write(np.nan_to_num(arr, nan=nodata).astype("float32"), 1)
            dst.set_band_description(1, result.name or "workflow")

    def showEvent(self, event):
        """Refresh layers when the dock is shown."""
        super().showEvent(event)
        self.refresh_layers()
