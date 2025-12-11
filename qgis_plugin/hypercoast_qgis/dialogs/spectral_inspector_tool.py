# -*- coding: utf-8 -*-
"""
HyperCoast QGIS Plugin - Spectral Inspector Tool

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

import numpy as np
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QCursor
from qgis.gui import QgsMapToolEmitPoint, QgsRubberBand
from qgis.core import (
    QgsProject,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsWkbTypes,
    QgsMessageLog,
    Qgis,
)


class SpectralInspectorTool(QgsMapToolEmitPoint):
    """Map tool for inspecting spectral signatures by clicking on the map."""

    # Signal emitted when a spectral signature is extracted
    spectral_extracted = pyqtSignal(float, float, object, object, str)

    def __init__(self, canvas, plugin):
        """Initialize the spectral inspector tool.

        :param canvas: The map canvas
        :param plugin: Reference to the main plugin
        """
        super().__init__(canvas)
        self.canvas = canvas
        self.plugin = plugin
        self.is_active = False

        # Store extracted points for visualization
        self.points = []
        self.rubber_bands = []

        # Set cursor
        self.setCursor(QCursor(Qt.CrossCursor))

    def activate(self):
        """Activate the tool."""
        self.is_active = True
        QgsMessageLog.logMessage(
            "Spectral Inspector activated - click on map to extract spectra",
            "HyperCoast",
            Qgis.Info,
        )

    def deactivate(self):
        """Deactivate the tool."""
        self.is_active = False
        # Clear rubber bands
        for rb in self.rubber_bands:
            self.canvas.scene().removeItem(rb)
        self.rubber_bands.clear()
        self.points.clear()

    def canvasPressEvent(self, event):
        """Handle mouse press event on canvas."""
        if not self.is_active:
            return

        # Get clicked point in map coordinates
        point = self.toMapCoordinates(event.pos())

        self.extract_spectrum_at_point(point)

    def extract_spectrum_at_point(self, point):
        """Extract spectral signature at the given point.

        :param point: QgsPointXY in map coordinates
        """
        # Get all hyperspectral layers
        hyper_layers = self.plugin.get_all_hyperspectral_layers()

        if not hyper_layers:
            QgsMessageLog.logMessage(
                "No hyperspectral layers loaded", "HyperCoast", Qgis.Warning
            )
            return

        # Transform point to WGS84 for extraction
        map_crs = self.canvas.mapSettings().destinationCrs()
        wgs84_crs = QgsCoordinateReferenceSystem("EPSG:4326")
        transform = QgsCoordinateTransform(map_crs, wgs84_crs, QgsProject.instance())

        try:
            wgs84_point = transform.transform(point)
            lon = wgs84_point.x()
            lat = wgs84_point.y()
        except Exception:
            lon = point.x()
            lat = point.y()

        # Add visual marker
        self._add_point_marker(point)

        # Extract from each layer
        for layer_id, data_info in hyper_layers.items():
            layer = QgsProject.instance().mapLayer(layer_id)
            if not layer:
                continue

            dataset = data_info.get("dataset")
            if dataset is None:
                continue

            # Check if point is within layer extent
            layer_extent = layer.extent()

            # Transform point to layer CRS for extent check
            layer_crs = layer.crs()
            to_layer_transform = QgsCoordinateTransform(
                map_crs, layer_crs, QgsProject.instance()
            )

            try:
                layer_point = to_layer_transform.transform(point)
            except Exception:
                layer_point = point

            if not layer_extent.contains(layer_point):
                continue

            try:
                # Extract spectral signature
                wavelengths, values = dataset.extract_spectral_signature(lon, lat)

                if wavelengths is not None and values is not None:
                    # Emit signal for plot update
                    self.spectral_extracted.emit(
                        lat, lon, wavelengths, values, layer.name()
                    )

                    # Store point
                    self.points.append(
                        {
                            "lat": lat,
                            "lon": lon,
                            "wavelengths": wavelengths,
                            "values": values,
                            "layer": layer.name(),
                        }
                    )

                    # Update spectral plot dialog if open
                    if self.plugin.spectral_plot_dialog:
                        self.plugin.spectral_plot_dialog.add_spectrum(
                            lat, lon, wavelengths, values, layer.name()
                        )

                    QgsMessageLog.logMessage(
                        f"Extracted spectrum at ({lat:.4f}, {lon:.4f}) from {layer.name()}",
                        "HyperCoast",
                        Qgis.Info,
                    )

            except Exception as e:
                QgsMessageLog.logMessage(
                    f"Error extracting spectrum: {str(e)}", "HyperCoast", Qgis.Warning
                )

    def _add_point_marker(self, point):
        """Add a visual marker at the clicked point.

        :param point: QgsPointXY
        """
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtGui import QColor

        rb = QgsRubberBand(self.canvas, QgsWkbTypes.PointGeometry)
        rb.setColor(QColor(255, 0, 0, 200))
        rb.setWidth(3)
        rb.setIcon(QgsRubberBand.ICON_CIRCLE)
        rb.setIconSize(10)

        rb.addPoint(point, True)

        self.rubber_bands.append(rb)

    def clear_points(self):
        """Clear all extracted points and markers."""
        for rb in self.rubber_bands:
            self.canvas.scene().removeItem(rb)
        self.rubber_bands.clear()
        self.points.clear()

    def get_extracted_points(self):
        """Get all extracted spectral points.

        :returns: List of point dictionaries
        """
        return self.points
