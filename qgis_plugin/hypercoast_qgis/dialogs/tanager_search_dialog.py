# -*- coding: utf-8 -*-
"""
HyperCoast QGIS Plugin - Tanager Search Dialog

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

import os
import ast
import json
from urllib.parse import unquote, urlparse
import urllib.request

from qgis.PyQt.QtCore import QObject, Qt, QThread, QUrl, pyqtSignal
from qgis.PyQt.QtGui import QColor, QCursor, QDesktopServices
from qgis.gui import QgsMapTool as _QgsMapTool, QgsRubberBand

try:
    from qgis.PyQt.QtCore import QVariant
except ImportError:  # pragma: no cover - PyQt6 test shim

    class QVariant:
        """Small QVariant fallback for Qt6-only test environments."""

        String = str
        Double = float


from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsFillSymbol,
    QgsGeometry,
    QgsMessageLog,
    QgsPointXY,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
    QgsVectorLayer,
    QgsWkbTypes,
    Qgis,
)

if isinstance(_QgsMapTool, type):
    QgsMapTool = _QgsMapTool
else:  # pragma: no cover - used by lightweight Qt-only test stubs

    class QgsMapTool(QObject):
        """Minimal QgsMapTool fallback for tests without QGIS GUI classes."""

        def __init__(self, canvas=None):
            """Initialize the fallback map tool.

            Args:
                canvas: Optional map canvas.
            """
            super().__init__()
            self.canvas = canvas

        def setCursor(self, cursor):
            """Store the requested cursor.

            Args:
                cursor: Cursor object.
            """
            self.cursor = cursor

        def toMapCoordinates(self, position):
            """Return the provided position as a map coordinate.

            Args:
                position: Position-like object.

            Returns:
                The input position.
            """
            return position

        def deactivate(self):
            """Deactivate the fallback map tool."""


TANAGER_HDF5_ASSET = "ortho_radiance_hdf5"
TANAGER_HDF5_ASSETS = (
    ("Ortho Radiance HDF5", "ortho_radiance_hdf5"),
    ("Basic Radiance HDF5", "basic_radiance_hdf5"),
    ("Ortho Surface Reflectance HDF5", "ortho_sr_hdf5"),
    ("Basic Surface Reflectance HDF5", "basic_sr_hdf5"),
)
TANAGER_VISUAL_ASSET = "ortho_visual"
TANAGER_GLOBAL_FOOTPRINTS_URL = (
    "https://raw.githubusercontent.com/opengeos/planet-open-data/refs/heads/main/"
    "data/planet-tanager-sample-products/footprints.geojson"
)


def _asset_url(item, asset_key):
    """Return a STAC asset URL from an item.

    Args:
        item: STAC item dictionary.
        asset_key: STAC asset key.

    Returns:
        Asset URL string, or an empty string when unavailable.
    """
    assets = item.get("assets", {}) if isinstance(item, dict) else {}
    asset = assets.get(asset_key, {})
    return asset.get("href", "") or ""


def _asset_filename(url, fallback="tanager.h5"):
    """Return a readable filename from an asset URL.

    Args:
        url: Asset URL.
        fallback: Filename to use when the URL has no basename.

    Returns:
        Filename string.
    """
    path = urlparse(url).path if url else ""
    filename = os.path.basename(unquote(path))
    return filename or fallback


def _item_title(item):
    """Return a readable Tanager item title.

    Args:
        item: STAC item dictionary.

    Returns:
        Item title or ID.
    """
    properties = item.get("properties", {}) if isinstance(item, dict) else {}
    return properties.get("title") or item.get("id") or "Tanager"


def _stac_browser_url(stac_url):
    """Return the Planet STAC browser URL for an item URL.

    Args:
        stac_url: Raw or browser STAC item URL.

    Returns:
        Browser URL string.
    """
    if not stac_url:
        return ""
    if "/data/stac/browser/" in stac_url:
        return stac_url
    return stac_url.replace("/data/stac/", "/data/stac/browser/", 1)


def _extent_to_bbox(extent):
    """Convert a QGIS extent-like object to a bbox list.

    Args:
        extent: Object exposing QGIS extent methods.

    Returns:
        Bounding box as ``[xmin, ymin, xmax, ymax]``.
    """
    return [
        float(extent.xMinimum()),
        float(extent.yMinimum()),
        float(extent.xMaximum()),
        float(extent.yMaximum()),
    ]


def tanager_download_dir(project=None):
    """Return the persistent Tanager download directory.

    Args:
        project: Optional QgsProject-like object. Kept for backward
            compatibility and ignored.

    Returns:
        Absolute directory path.
    """
    out_dir = os.path.join(os.path.expanduser("~"), "Downloads")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


class TanagerBboxMapTool(QgsMapTool):
    """Map tool for drawing a Tanager search bounding box."""

    bbox_drawn = pyqtSignal(list)
    canceled = pyqtSignal()

    def __init__(self, canvas, parent=None):
        """Initialize the map tool.

        Args:
            canvas: QGIS map canvas.
            parent: Optional QObject parent.
        """
        super().__init__(canvas)
        self.canvas = canvas
        self.parent = parent
        self._start_point = None
        self._rubber_band = None
        try:
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        except AttributeError:
            self.setCursor(QCursor(Qt.CrossCursor))
        except Exception:
            pass  # nosec B110

    def canvasPressEvent(self, event):
        """Start drawing the bounding box.

        Args:
            event: QGIS map mouse event.
        """
        if not self._is_left_button(event):
            return
        self._start_point = self.toMapCoordinates(event.pos())
        self._ensure_rubber_band()

    def canvasMoveEvent(self, event):
        """Update the live bounding box preview.

        Args:
            event: QGIS map mouse event.
        """
        if self._start_point is None:
            return
        self._update_rubber_band(self._start_point, self.toMapCoordinates(event.pos()))

    def canvasReleaseEvent(self, event):
        """Finish drawing and emit the WGS84 bbox.

        Args:
            event: QGIS map mouse event.
        """
        if self._start_point is None or not self._is_left_button(event):
            return
        end_point = self.toMapCoordinates(event.pos())
        bbox = self._bbox_from_points(self._start_point, end_point)
        self._start_point = None
        self._remove_rubber_band()
        if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
            self.canceled.emit()
            return
        self.bbox_drawn.emit(self._transform_bbox_to_wgs84(bbox))

    def keyPressEvent(self, event):
        """Cancel drawing when Escape is pressed.

        Args:
            event: QGIS key event.
        """
        escape_key = getattr(getattr(Qt, "Key", Qt), "Key_Escape", None)
        if escape_key is None:
            escape_key = getattr(Qt, "Key_Escape", None)
        if escape_key is not None and event.key() == escape_key:
            self._start_point = None
            self._remove_rubber_band()
            self.canceled.emit()

    def deactivate(self):
        """Remove the live preview when the map tool is deactivated."""
        self._start_point = None
        self._remove_rubber_band()
        super_deactivate = getattr(super(), "deactivate", None)
        if callable(super_deactivate):
            super_deactivate()

    def _is_left_button(self, event):
        """Return whether the event used the left mouse button.

        Args:
            event: QGIS map mouse event.

        Returns:
            True when the left mouse button was used.
        """
        left_button = getattr(getattr(Qt, "MouseButton", Qt), "LeftButton", None)
        if left_button is None:
            left_button = getattr(Qt, "LeftButton", None)
        button = getattr(event, "button", None)
        if callable(button) and left_button is not None:
            return button() == left_button
        return True

    def _ensure_rubber_band(self):
        """Create the preview rubber band if needed."""
        if self._rubber_band is not None:
            return
        self._rubber_band = QgsRubberBand(self.canvas, self._polygon_geometry_type())
        try:
            self._rubber_band.setColor(QColor(0, 102, 204, 90))
            self._rubber_band.setStrokeColor(QColor(0, 102, 204, 220))
            self._rubber_band.setFillColor(QColor(173, 216, 230, 55))
            self._rubber_band.setWidth(2)
        except Exception:
            pass  # nosec B110

    def _update_rubber_band(self, start_point, end_point):
        """Update the preview rectangle.

        Args:
            start_point: First corner in map coordinates.
            end_point: Opposite corner in map coordinates.
        """
        self._ensure_rubber_band()
        points = self._rectangle_points(start_point, end_point)
        try:
            self._rubber_band.reset(self._polygon_geometry_type())
            for point in points:
                self._rubber_band.addPoint(point, False)
            self._rubber_band.addPoint(points[0], True)
        except Exception:
            pass  # nosec B110

    def _remove_rubber_band(self):
        """Remove the preview rubber band from the canvas."""
        if self._rubber_band is None:
            return
        try:
            self.canvas.scene().removeItem(self._rubber_band)
        except Exception:
            pass  # nosec B110
        self._rubber_band = None

    def _transform_bbox_to_wgs84(self, bbox):
        """Transform a map-canvas bbox to EPSG:4326.

        Args:
            bbox: Bounding box in the map canvas CRS.

        Returns:
            Bounding box list in EPSG:4326.
        """
        try:
            source_crs = self.canvas.mapSettings().destinationCrs()
            target_crs = QgsCoordinateReferenceSystem("EPSG:4326")
            if (
                source_crs
                and target_crs.isValid()
                and source_crs.isValid()
                and source_crs != target_crs
            ):
                transform = QgsCoordinateTransform(
                    source_crs, target_crs, QgsProject.instance()
                )
                extent = transform.transformBoundingBox(QgsRectangle(*bbox))
                return _extent_to_bbox(extent)
        except Exception as exc:
            QgsMessageLog.logMessage(
                f"Could not transform drawn Tanager bbox: {exc}",
                "HyperCoast",
                Qgis.MessageLevel.Warning,
            )
        return bbox

    def _bbox_from_points(self, start_point, end_point):
        """Return a normalized bbox from two map points.

        Args:
            start_point: First corner point.
            end_point: Opposite corner point.

        Returns:
            Bounding box list.
        """
        x_values = [float(start_point.x()), float(end_point.x())]
        y_values = [float(start_point.y()), float(end_point.y())]
        return [min(x_values), min(y_values), max(x_values), max(y_values)]

    def _rectangle_points(self, start_point, end_point):
        """Return rectangle corner points.

        Args:
            start_point: First corner point.
            end_point: Opposite corner point.

        Returns:
            List of QGIS points in map coordinates.
        """
        xmin, ymin, xmax, ymax = self._bbox_from_points(start_point, end_point)
        return [
            QgsPointXY(xmin, ymin),
            QgsPointXY(xmax, ymin),
            QgsPointXY(xmax, ymax),
            QgsPointXY(xmin, ymax),
        ]

    def _polygon_geometry_type(self):
        """Return the QGIS polygon geometry type enum.

        Returns:
            Polygon geometry enum value.
        """
        geometry_type = getattr(QgsWkbTypes, "GeometryType", QgsWkbTypes)
        return getattr(
            geometry_type,
            "PolygonGeometry",
            getattr(QgsWkbTypes, "PolygonGeometry", 2),
        )


class TanagerSearchWorker(QThread):
    """Worker thread for Tanager STAC searches."""

    progress = pyqtSignal(int)
    finished = pyqtSignal(object, object, str)

    def __init__(self, search_params, parent=None):
        """Initialize the worker.

        Args:
            search_params: Keyword arguments for ``hypercoast.search_tanager``.
            parent: Optional parent QObject.
        """
        super().__init__(parent)
        self.search_params = dict(search_params)

    def run(self):
        """Run the Tanager search."""
        try:
            from .._hypercoast_lib import get_hypercoast

            self.progress.emit(20)
            hypercoast = get_hypercoast()
            params = dict(self.search_params)
            params["return_gdf"] = False
            self.progress.emit(50)
            items = hypercoast.search_tanager(**params)
            self.progress.emit(90)
            self.finished.emit(items, None, "")
        except Exception as exc:
            self.finished.emit([], None, str(exc))


class TanagerDownloadWorker(QThread):
    """Worker thread for downloading a Tanager HDF5 asset."""

    progress = pyqtSignal(int)
    finished = pyqtSignal(str, str)

    def __init__(self, item, asset_key, output_path, parent=None):
        """Initialize the worker.

        Args:
            item: STAC item dictionary.
            asset_key: STAC asset key to download.
            output_path: Destination HDF5 path.
            parent: Optional parent QObject.
        """
        super().__init__(parent)
        self.item = item
        self.asset_key = asset_key
        self.output_path = output_path

    def run(self):
        """Download the HDF5 asset."""
        try:
            from .._hypercoast_lib import get_hypercoast

            self.progress.emit(20)
            hypercoast = get_hypercoast()
            url = _asset_url(self.item, self.asset_key)
            if not url:
                raise ValueError(f"The selected asset '{self.asset_key}' has no URL.")
            filepath = hypercoast.download_file(
                url,
                output=self.output_path,
                quiet=True,
                overwrite=True,
                unzip=False,
            )
            self.progress.emit(90)
            self.finished.emit(filepath, "")
        except Exception as exc:
            self.finished.emit("", str(exc))


class TanagerSearchDialog(QDockWidget):
    """Dockable panel for searching and visualizing Tanager data."""

    def __init__(self, iface, plugin, parent=None):
        """Initialize the dialog.

        Args:
            iface: QGIS interface.
            plugin: Reference to the main plugin.
            parent: Optional parent widget.
        """
        super().__init__(parent or iface.mainWindow())
        self.iface = iface
        self.plugin = plugin
        self.items = []
        self.gdf = None
        self._search_worker = None
        self._load_worker = None
        self._pending_load_context = None
        self._connected_global_footprint_layers = set()
        self._result_footprints_layer = None
        self._bbox_map_tool = None
        self._previous_map_tool = None

        self.setWindowTitle("Search Tanager Data")
        self.setObjectName("HyperCoastTanagerSearchDock")
        self.setMinimumWidth(520)
        self.setMinimumHeight(560)

        self.setup_ui()
        self.refresh_extent()
        self.add_global_footprints_layer()

    def setup_ui(self):
        """Set up the user interface."""
        content = QWidget(self)
        self.setWidget(content)
        layout = QVBoxLayout(content)

        search_group = QGroupBox("Search")
        form = QFormLayout(search_group)

        extent_row = QHBoxLayout()
        self.use_extent_check = QCheckBox("Use current map extent")
        self.use_extent_check.setChecked(True)
        extent_row.addWidget(self.use_extent_check)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_extent)
        extent_row.addWidget(refresh_btn)
        draw_btn = QPushButton("Draw BBox")
        draw_btn.clicked.connect(self.start_bbox_drawing)
        extent_row.addWidget(draw_btn)
        global_btn = QPushButton("Global")
        global_btn.clicked.connect(self.clear_bbox)
        extent_row.addWidget(global_btn)
        form.addRow("Area:", extent_row)

        self.bbox_edit = QLineEdit()
        self.bbox_edit.setPlaceholderText("xmin, ymin, xmax, ymax")
        form.addRow("BBox:", self.bbox_edit)

        date_row = QHBoxLayout()
        self.start_date_edit = QLineEdit()
        self.start_date_edit.setPlaceholderText("YYYY-MM-DD")
        date_row.addWidget(QLabel("Start:"))
        date_row.addWidget(self.start_date_edit)
        self.end_date_edit = QLineEdit()
        self.end_date_edit.setPlaceholderText("YYYY-MM-DD")
        date_row.addWidget(QLabel("End:"))
        date_row.addWidget(self.end_date_edit)
        form.addRow("Dates:", date_row)

        self.collection_edit = QLineEdit()
        self.collection_edit.setPlaceholderText("Collection ID or title")
        form.addRow("Collection:", self.collection_edit)

        self.query_edit = QLineEdit()
        self.query_edit.setPlaceholderText("Scene text")
        form.addRow("Query:", self.query_edit)

        filter_row = QHBoxLayout()
        self.cloud_spin = QDoubleSpinBox()
        self.cloud_spin.setRange(0, 100)
        self.cloud_spin.setValue(100)
        self.cloud_spin.setSuffix(" %")
        filter_row.addWidget(QLabel("Max cloud:"))
        filter_row.addWidget(self.cloud_spin)
        self.count_spin = QSpinBox()
        self.count_spin.setRange(-1, 10000)
        self.count_spin.setValue(100)
        filter_row.addWidget(QLabel("Count:"))
        filter_row.addWidget(self.count_spin)
        form.addRow("Filters:", filter_row)

        layout.addWidget(search_group)

        self.results_table = QTableWidget(0, 6)
        self.results_table.setHorizontalHeaderLabels(
            ["ID", "Datetime", "Collection", "Cloud", "Visual", "Radiance HDF5"]
        )
        self._stretch_result_columns()
        try:
            self.results_table.setSelectionBehavior(
                QAbstractItemView.SelectionBehavior.SelectRows
            )
            self.results_table.setSelectionMode(
                QAbstractItemView.SelectionMode.ExtendedSelection
            )
        except AttributeError:
            self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.results_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        try:
            self.results_table.setEditTriggers(
                QAbstractItemView.EditTrigger.NoEditTriggers
            )
        except AttributeError:
            self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.itemSelectionChanged.connect(
            self._on_result_selection_changed
        )
        layout.addWidget(self.results_table)

        download_group = QGroupBox("Download")
        download_form = QFormLayout(download_group)
        self.asset_combo = QComboBox()
        self._populate_asset_combo()
        download_form.addRow("Asset:", self.asset_combo)

        download_dir_row = QHBoxLayout()
        self.download_dir_edit = QLineEdit()
        self.download_dir_edit.setText(tanager_download_dir(QgsProject.instance()))
        download_dir_row.addWidget(self.download_dir_edit)
        browse_dir_btn = QPushButton("Browse...")
        browse_dir_btn.clicked.connect(self.browse_download_dir)
        download_dir_row.addWidget(browse_dir_btn)
        download_form.addRow("Folder:", download_dir_row)
        layout.addWidget(download_group)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        button_layout = QGridLayout()
        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self.search)
        button_layout.addWidget(self.search_btn, 0, 0)

        self.selected_footprints_btn = QPushButton("Use Selected Footprints")
        self.selected_footprints_btn.clicked.connect(
            lambda: self.populate_from_selected_footprints()
        )
        button_layout.addWidget(self.selected_footprints_btn, 0, 1)

        self.footprints_btn = QPushButton("Add Footprints")
        self.footprints_btn.clicked.connect(lambda: self.add_footprints_layer())
        button_layout.addWidget(self.footprints_btn, 0, 2)

        self.visual_btn = QPushButton("Open Visual")
        self.visual_btn.clicked.connect(self.open_visual_layer)
        button_layout.addWidget(self.visual_btn, 1, 0)

        self.stac_btn = QPushButton("Open STAC")
        self.stac_btn.clicked.connect(self.open_stac_item)
        button_layout.addWidget(self.stac_btn, 1, 1)

        self.download_btn = QPushButton("Download HDF5")
        self.download_btn.clicked.connect(self.download_hdf5)
        button_layout.addWidget(self.download_btn, 1, 2)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn, 1, 3)
        layout.addLayout(button_layout)
        self._sync_button_state()

    def refresh_extent(self):
        """Refresh the bbox field from the current QGIS map extent."""
        self._restore_previous_map_tool()
        bbox = self.current_canvas_bbox()
        if bbox:
            self.bbox_edit.setText(", ".join(f"{value:.8f}" for value in bbox))
            self.use_extent_check.setChecked(True)

    def start_bbox_drawing(self):
        """Activate a map tool for drawing the Tanager search bbox."""
        try:
            canvas = self.iface.mapCanvas()
        except Exception as exc:
            QMessageBox.warning(self, "Draw BBox", f"Could not access map: {exc}")
            return

        if self._bbox_map_tool is None:
            self._bbox_map_tool = TanagerBboxMapTool(canvas, self)
            self._bbox_map_tool.bbox_drawn.connect(self._on_bbox_drawn)
            self._bbox_map_tool.canceled.connect(self._on_bbox_drawing_canceled)

        map_tool = getattr(canvas, "mapTool", None)
        if callable(map_tool):
            try:
                current_tool = map_tool()
                if current_tool is not self._bbox_map_tool:
                    self._previous_map_tool = current_tool
            except Exception:
                self._previous_map_tool = None

        try:
            canvas.setMapTool(self._bbox_map_tool)
            self.use_extent_check.setChecked(False)
            self.status_label.setText("Draw a rectangle on the map for the search bbox")
        except Exception as exc:
            QMessageBox.warning(
                self, "Draw BBox", f"Could not activate map tool: {exc}"
            )

    def _on_bbox_drawn(self, bbox):
        """Use a drawn bbox as manual search input.

        Args:
            bbox: Bounding box list in EPSG:4326.
        """
        self.use_extent_check.setChecked(False)
        self.bbox_edit.setText(", ".join(f"{value:.8f}" for value in bbox))
        self.status_label.setText("Drawn bbox ready for Tanager search")
        self._restore_previous_map_tool()

    def _on_bbox_drawing_canceled(self):
        """Handle bbox drawing cancellation."""
        self.status_label.setText("BBox drawing canceled")
        self._restore_previous_map_tool()

    def _restore_previous_map_tool(self):
        """Restore the previous QGIS map tool after bbox drawing."""
        if self._bbox_map_tool is None:
            return
        try:
            canvas = self.iface.mapCanvas()
            map_tool = getattr(canvas, "mapTool", None)
            current_tool = map_tool() if callable(map_tool) else None
            if current_tool is not self._bbox_map_tool:
                return
            if self._previous_map_tool is not None:
                canvas.setMapTool(self._previous_map_tool)
            else:
                unset_map_tool = getattr(canvas, "unsetMapTool", None)
                if callable(unset_map_tool):
                    unset_map_tool(self._bbox_map_tool)
        except Exception:
            pass  # nosec B110
        finally:
            self._previous_map_tool = None

    def clear_bbox(self):
        """Clear the bbox filter for a global Tanager search."""
        self._restore_previous_map_tool()
        self.use_extent_check.setChecked(False)
        self.bbox_edit.clear()

    def closeEvent(self, event):
        """Restore the previous map tool before closing.

        Args:
            event: QGIS close event.
        """
        self._restore_previous_map_tool()
        super().closeEvent(event)

    def browse_download_dir(self):
        """Select a Tanager HDF5 download directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Tanager Download Folder",
            self.download_dir_edit.text()
            or tanager_download_dir(QgsProject.instance()),
        )
        if directory:
            self.download_dir_edit.setText(directory)

    def add_global_footprints_layer(self):
        """Add the global Tanager sample footprints layer to the project.

        Returns:
            The existing or newly added global footprint layer, or ``None``.
        """
        try:
            existing = self._global_footprints_layer()
            if existing is not None:
                self._connect_global_footprint_selection(existing)
                self.populate_from_selected_footprints(show_message=False)
                return existing

            layer = None
            for source in (
                TANAGER_GLOBAL_FOOTPRINTS_URL,
                f"/vsicurl/{TANAGER_GLOBAL_FOOTPRINTS_URL}",
            ):
                candidate = QgsVectorLayer(source, "Tanager Sample Footprints", "ogr")
                if candidate.isValid():
                    layer = candidate
                    break

            if layer is None:
                QgsMessageLog.logMessage(
                    "Could not load global Tanager footprints GeoJSON",
                    "HyperCoast",
                    Qgis.MessageLevel.Warning,
                )
                return None

            layer.setCustomProperty("hypercoast/tanager_global_footprints", "true")
            layer.setCustomProperty(
                "hypercoast/tanager_footprints_url", TANAGER_GLOBAL_FOOTPRINTS_URL
            )
            self._style_footprint_layer(layer)
            QgsProject.instance().addMapLayer(layer)
            self._connect_global_footprint_selection(layer)
            return layer
        except Exception as exc:
            QgsMessageLog.logMessage(
                f"Could not add global Tanager footprints: {exc}",
                "HyperCoast",
                Qgis.MessageLevel.Warning,
            )
            return None

    def _global_footprints_layer(self):
        """Return an existing global Tanager footprints layer.

        Returns:
            Existing layer, or ``None``.
        """
        try:
            layers = QgsProject.instance().mapLayers()
        except Exception:
            return None
        if not isinstance(layers, dict):
            return None
        for layer in layers.values():
            custom_property = getattr(layer, "customProperty", None)
            if not callable(custom_property):
                continue
            if custom_property("hypercoast/tanager_global_footprints", "") == "true":
                return layer
        return None

    def _connect_global_footprint_selection(self, layer):
        """Connect global footprint selection changes to the results table.

        Args:
            layer: Global Tanager footprint layer.
        """
        layer_key = self._layer_key(layer)
        if layer_key in self._connected_global_footprint_layers:
            return
        signal = getattr(layer, "selectionChanged", None)
        if signal is None or not hasattr(signal, "connect"):
            return
        try:
            signal.connect(
                lambda *args: self.populate_from_selected_footprints(show_message=False)
            )
            self._connected_global_footprint_layers.add(layer_key)
        except Exception:
            pass  # nosec B110

    def _layer_key(self, layer):
        """Return a stable key for a QGIS layer object.

        Args:
            layer: QGIS layer-like object.

        Returns:
            Layer ID when available, otherwise the Python object ID.
        """
        layer_id = getattr(layer, "id", None)
        if callable(layer_id):
            try:
                return layer_id()
            except Exception:
                pass  # nosec B110
        return id(layer)

    def populate_from_selected_footprints(self, show_message=True):
        """Populate results from selected global Tanager footprints.

        Args:
            show_message: Whether to show an information dialog when no
                footprints are selected.
        """
        layer = self._global_footprints_layer()
        selected_features = []
        if layer is not None:
            selected = getattr(layer, "selectedFeatures", None)
            if callable(selected):
                try:
                    selected_features = list(selected())
                except Exception:
                    selected_features = []

        if not selected_features:
            if show_message:
                QMessageBox.information(
                    self,
                    "Tanager Footprints",
                    "Select one or more global Tanager footprints on the map first.",
                )
            return

        items = [
            self._item_from_global_footprint_feature(feature)
            for feature in selected_features
        ]
        self.items = items
        self.gdf = None
        self.populate_results(items)
        self._select_result_rows(range(len(items)))
        self.status_label.setText(
            f"{len(items)} selected Tanager footprint scenes loaded"
        )

    def _select_result_rows(self, rows):
        """Select result table rows.

        Args:
            rows: Iterable of row indexes.
        """
        self.results_table.clearSelection()
        for row in rows:
            self.results_table.selectRow(row)
        self._sync_button_state()

    def _item_from_global_footprint_feature(self, feature):
        """Create a lightweight STAC item from a footprint feature.

        Args:
            feature: QGIS feature from the global footprint GeoJSON.

        Returns:
            STAC-like item dictionary.
        """
        asset_keys = self._asset_keys_from_value(
            self._feature_attr(feature, "asset_keys")
        )
        return {
            "id": self._feature_attr(feature, "id"),
            "collection": self._feature_attr(feature, "collection"),
            "_collection_title": self._feature_attr(feature, "collection_title"),
            "_stac_url": self._feature_attr(feature, "stac_href"),
            "properties": {
                "title": self._feature_attr(feature, "title"),
                "datetime": self._feature_attr(feature, "datetime"),
                "collection_title": self._feature_attr(feature, "collection_title"),
            },
            "assets": {asset_key: {} for asset_key in asset_keys},
        }

    def _feature_attr(self, feature, name, default=""):
        """Read a feature attribute by name.

        Args:
            feature: QGIS feature-like object.
            name: Attribute name.
            default: Fallback value.

        Returns:
            Attribute value or ``default``.
        """
        attribute = getattr(feature, "attribute", None)
        if callable(attribute):
            try:
                value = attribute(name)
                return self._plain_attr_value(value, default)
            except Exception:
                pass  # nosec B110
        try:
            value = feature[name]
            return self._plain_attr_value(value, default)
        except Exception:
            return default

    def _plain_attr_value(self, value, default=""):
        """Convert a QGIS attribute value to a plain Python value.

        Args:
            value: Raw feature attribute value.
            default: Fallback value.

        Returns:
            Plain attribute value suitable for display and STAC metadata.
        """
        if value in (None, ""):
            return default
        to_py_datetime = getattr(value, "toPyDateTime", None)
        if callable(to_py_datetime):
            try:
                return to_py_datetime().isoformat()
            except Exception:
                pass  # nosec B110
        to_string = getattr(value, "toString", None)
        if callable(to_string):
            for date_format in (
                getattr(getattr(Qt, "DateFormat", object()), "ISODateWithMs", None),
                getattr(getattr(Qt, "DateFormat", object()), "ISODate", None),
                getattr(Qt, "ISODate", None),
            ):
                if date_format is None:
                    continue
                try:
                    text = to_string(date_format)
                    if text:
                        return text
                except Exception:
                    pass  # nosec B110
            try:
                text = to_string()
                if text:
                    return text
            except Exception:
                pass  # nosec B110
        return value

    def _asset_keys_from_value(self, value):
        """Parse a footprint ``asset_keys`` attribute.

        Args:
            value: Attribute value from the footprint layer.

        Returns:
            List of STAC asset keys.
        """
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        if value in (None, ""):
            return []
        text = str(value).strip()
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, (list, tuple)):
                    return [str(item) for item in parsed]
            except Exception:
                pass  # nosec B110
        return [part.strip() for part in text.split(",") if part.strip()]

    def current_canvas_bbox(self):
        """Return the current map canvas extent in EPSG:4326.

        Returns:
            Bounding box list, or ``None`` when unavailable.
        """
        try:
            canvas = self.iface.mapCanvas()
            extent = canvas.extent()
            source_crs = canvas.mapSettings().destinationCrs()
            target_crs = QgsCoordinateReferenceSystem("EPSG:4326")
            if (
                source_crs
                and target_crs.isValid()
                and source_crs.isValid()
                and source_crs != target_crs
            ):
                transform = QgsCoordinateTransform(
                    source_crs, target_crs, QgsProject.instance()
                )
                extent = transform.transformBoundingBox(extent)
            return _extent_to_bbox(extent)
        except Exception as exc:
            QgsMessageLog.logMessage(
                f"Could not read map extent for Tanager search: {exc}",
                "HyperCoast",
                Qgis.MessageLevel.Warning,
            )
            return None

    def search_params(self):
        """Return validated Tanager search parameters.

        Returns:
            Dictionary of search keyword arguments.

        Raises:
            ValueError: If the date or bbox input is invalid.
        """
        params = {
            "count": self.count_spin.value(),
            "cloud_percent": self.cloud_spin.value(),
        }
        collection = self.collection_edit.text().strip()
        if collection:
            params["collections"] = collection
        query = self.query_edit.text().strip()
        if query:
            params["query"] = query

        start = self.start_date_edit.text().strip()
        end = self.end_date_edit.text().strip()
        if start or end:
            if not (start and end):
                raise ValueError("Both start and end dates are required.")
            params["temporal"] = (start, end)

        if self.use_extent_check.isChecked():
            bbox = self.current_canvas_bbox()
        else:
            bbox = self._parse_bbox(self.bbox_edit.text())
        if bbox:
            params["bbox"] = bbox

        return params

    def _parse_bbox(self, value):
        """Parse bbox text.

        Args:
            value: Text in ``xmin, ymin, xmax, ymax`` format.

        Returns:
            Bounding box list, or ``None`` for blank text.

        Raises:
            ValueError: If the text is not a valid four-number bbox.
        """
        value = value.strip()
        if not value:
            return None
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 4:
            raise ValueError("BBox must contain xmin, ymin, xmax, ymax.")
        try:
            bbox = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError("BBox values must be numbers.") from exc
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError("BBox minimum values must be less than maximum values.")
        return bbox

    def search(self):
        """Search Tanager STAC data."""
        if self._search_worker is not None and self._search_worker.isRunning():
            return
        try:
            params = self.search_params()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Search", str(exc))
            return

        self._set_busy(True)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(10)
        self.status_label.setText("Searching Tanager STAC...")
        self._search_worker = TanagerSearchWorker(params, self)
        self._search_worker.progress.connect(self.progress_bar.setValue)
        self._search_worker.finished.connect(self._on_search_finished)
        self._search_worker.start()

    def _on_search_finished(self, items, gdf, error_detail):
        """Handle search completion.

        Args:
            items: STAC item dictionaries.
            gdf: GeoDataFrame of item footprints.
            error_detail: Error message when search failed.
        """
        try:
            if error_detail:
                raise ValueError(error_detail)
            self.items = list(items or [])
            self.gdf = gdf
            self.populate_results(self.items)
            self.add_footprints_layer(show_message=False, zoom=True, replace=True)
            self.progress_bar.setValue(100)
            self.status_label.setText(f"{len(self.items)} Tanager scenes found")
        except Exception as exc:
            self.items = []
            self.gdf = None
            self.populate_results([])
            self.status_label.setText("Search failed")
            QMessageBox.warning(self, "Tanager Search", str(exc))
            QgsMessageLog.logMessage(
                f"Tanager search failed: {exc}",
                "HyperCoast",
                Qgis.MessageLevel.Warning,
            )
        finally:
            self.progress_bar.setRange(0, 100)
            self._set_busy(False)

    def populate_results(self, items):
        """Populate the result table.

        Args:
            items: STAC item dictionaries.
        """
        self.results_table.setRowCount(0)
        for row, item in enumerate(items):
            properties = item.get("properties", {})
            self.results_table.insertRow(row)
            values = [
                item.get("id", ""),
                properties.get("datetime", ""),
                item.get("_collection_title")
                or properties.get("collection_title", "")
                or item.get("collection", ""),
                properties.get("cloud_percent", ""),
                "yes" if self._has_asset(item, TANAGER_VISUAL_ASSET) else "",
                "yes" if self._has_hdf5_asset(item) else "",
            ]
            for col, value in enumerate(values):
                table_item = QTableWidgetItem(str(value))
                if col == 0:
                    table_item.setData(Qt.ItemDataRole.UserRole, item)
                self.results_table.setItem(row, col, table_item)
        self.results_table.resizeColumnsToContents()
        self._stretch_result_columns()
        self._sync_button_state()

    def selected_item(self):
        """Return the currently selected STAC item.

        Returns:
            STAC item dictionary, or ``None``.
        """
        rows = self.results_table.selectionModel().selectedRows()
        row = rows[0].row() if rows else self.results_table.currentRow()
        if row < 0:
            return None
        table_item = self.results_table.item(row, 0)
        return table_item.data(Qt.ItemDataRole.UserRole) if table_item else None

    def hydrated_selected_item(self):
        """Return the selected item with full STAC asset metadata.

        Returns:
            Hydrated STAC item dictionary, or ``None``.
        """
        row = self._selected_table_row()
        if row < 0:
            return None
        item = self.selected_item()
        if item is None:
            return None
        hydrated = self._hydrate_stac_item(item)
        table_item = self.results_table.item(row, 0)
        if table_item is not None:
            table_item.setData(Qt.ItemDataRole.UserRole, hydrated)
        self._populate_asset_combo(hydrated)
        return hydrated

    def _selected_table_row(self):
        """Return the selected result table row.

        Returns:
            Selected row index, or ``-1``.
        """
        rows = self.results_table.selectionModel().selectedRows()
        return rows[0].row() if rows else self.results_table.currentRow()

    def _hydrate_stac_item(self, item):
        """Load full STAC item metadata for a lightweight footprint row.

        Args:
            item: STAC-like item dictionary.

        Returns:
            Full STAC item when available, otherwise the input item.
        """
        if self._has_asset_url(item):
            return item
        stac_url = item.get("_stac_url") or item.get("properties", {}).get("stac_href")
        if not stac_url:
            return item
        try:
            with urllib.request.urlopen(stac_url, timeout=30) as response:  # nosec B310
                loaded = json.load(response)
            loaded = dict(loaded)
            loaded["_stac_url"] = stac_url
            if item.get("_collection_title"):
                loaded["_collection_title"] = item["_collection_title"]
            return loaded
        except Exception as exc:
            QgsMessageLog.logMessage(
                f"Could not load Tanager STAC item {stac_url}: {exc}",
                "HyperCoast",
                Qgis.MessageLevel.Warning,
            )
            return item

    def _has_asset(self, item, asset_key):
        """Return whether an item advertises an asset.

        Args:
            item: STAC item dictionary.
            asset_key: STAC asset key.

        Returns:
            True when the asset key exists.
        """
        assets = item.get("assets", {}) if isinstance(item, dict) else {}
        return asset_key in assets

    def _has_hdf5_asset(self, item):
        """Return whether an item advertises any known HDF5 asset.

        Args:
            item: STAC item dictionary.

        Returns:
            True when at least one HDF5 asset key exists.
        """
        return any(
            self._has_asset(item, asset_key) for _, asset_key in TANAGER_HDF5_ASSETS
        )

    def _has_asset_url(self, item):
        """Return whether any STAC asset has an href.

        Args:
            item: STAC item dictionary.

        Returns:
            True when at least one asset has a URL.
        """
        assets = item.get("assets", {}) if isinstance(item, dict) else {}
        return any(asset.get("href") for asset in assets.values())

    def _item_geometry(self, item):
        """Return a QGIS geometry from a STAC item geometry.

        Args:
            item: STAC item dictionary.

        Returns:
            QgsGeometry, or ``None`` when geometry is unavailable.
        """
        geometry = item.get("geometry", {}) if isinstance(item, dict) else {}
        geom_type = geometry.get("type")
        coordinates = geometry.get("coordinates")
        if not geom_type or not coordinates:
            return None

        def ring_points(ring):
            """Return QGIS points for a GeoJSON ring."""
            return [QgsPointXY(float(x), float(y)) for x, y, *rest in ring]

        try:
            if geom_type == "Polygon":
                return QgsGeometry.fromPolygonXY(
                    [ring_points(ring) for ring in coordinates]
                )
            if geom_type == "MultiPolygon":
                return QgsGeometry.fromMultiPolygonXY(
                    [[ring_points(ring) for ring in polygon] for polygon in coordinates]
                )
        except Exception:
            return None
        return None

    def _item_stac_url(self, item):
        """Return the best STAC URL for an item.

        Args:
            item: STAC item dictionary.

        Returns:
            STAC item URL string.
        """
        return item.get("_stac_url") or item.get("properties", {}).get("stac_href", "")

    def _remove_result_footprints_layer(self):
        """Remove the current search result footprint layer, if present."""
        layer = self._result_footprints_layer
        self._result_footprints_layer = None
        if layer is None:
            return
        try:
            QgsProject.instance().removeMapLayer(layer.id())
        except Exception:
            pass  # nosec B110

    def _result_item_ids(self):
        """Return selected result table item IDs.

        Returns:
            Set of selected STAC item IDs.
        """
        rows = self.results_table.selectionModel().selectedRows()
        if not rows and self.results_table.currentRow() >= 0:
            rows = [
                self.results_table.model().index(self.results_table.currentRow(), 0)
            ]

        item_ids = set()
        for row_index in rows:
            table_item = self.results_table.item(row_index.row(), 0)
            item = table_item.data(Qt.ItemDataRole.UserRole) if table_item else None
            if isinstance(item, dict) and item.get("id"):
                item_ids.add(item["id"])
        return item_ids

    def _select_result_footprints(self):
        """Select result footprint features matching selected table rows."""
        layer = self._result_footprints_layer
        if layer is None:
            return
        item_ids = self._result_item_ids()
        if not item_ids:
            try:
                layer.removeSelection()
            except Exception:
                pass  # nosec B110
            return

        feature_ids = []
        get_features = getattr(layer, "getFeatures", None)
        if callable(get_features):
            try:
                for feature in get_features():
                    if self._feature_attr(feature, "id") in item_ids:
                        feature_ids.append(feature.id())
            except Exception:
                feature_ids = []

        try:
            layer.selectByIds(feature_ids)
        except Exception:
            pass  # nosec B110

    def _has_item_footprints(self):
        """Return whether the current items include STAC geometries."""
        for item in self.items:
            geometry = item.get("geometry", {}) if isinstance(item, dict) else {}
            if geometry.get("type") in {"Polygon", "MultiPolygon"}:
                if geometry.get("coordinates"):
                    return True
        return False

    def add_footprints_layer(self, show_message=True, zoom=True, replace=False):
        """Add Tanager result footprints as an in-memory vector layer."""
        if len(self.items) == 0:
            if show_message:
                QMessageBox.information(
                    self, "Tanager Footprints", "No results to add."
                )
            return

        try:
            if replace:
                self._remove_result_footprints_layer()
            layer = QgsVectorLayer(
                "Polygon?crs=EPSG:4326",
                "Tanager Footprints",
                "memory",
            )
            provider = layer.dataProvider()
            fields = [
                QgsField("id", QVariant.String),
                QgsField("datetime", QVariant.String),
                QgsField("collection", QVariant.String),
                QgsField("cloud_pct", QVariant.Double),
                QgsField("stac_url", QVariant.String),
                QgsField("visual_url", QVariant.String),
                QgsField("hdf5_url", QVariant.String),
            ]
            provider.addAttributes(fields)
            layer.updateFields()

            features = []
            for item in self.items:
                geometry = self._item_geometry(item)
                if geometry is None:
                    continue
                properties = item.get("properties", {})
                feature = QgsFeature(layer.fields())
                feature.setGeometry(geometry)
                feature.setAttributes(
                    [
                        item.get("id", ""),
                        properties.get("datetime", ""),
                        item.get("_collection_title") or item.get("collection", ""),
                        properties.get("cloud_percent"),
                        self._item_stac_url(item),
                        _asset_url(item, TANAGER_VISUAL_ASSET),
                        _asset_url(item, TANAGER_HDF5_ASSET),
                    ]
                )
                features.append(feature)

            if not features:
                if show_message:
                    QMessageBox.information(
                        self,
                        "Tanager Footprints",
                        "The search results do not include footprint geometries.",
                    )
                return

            provider.addFeatures(features)
            layer.updateExtents()
            layer.setCustomProperty(
                "hypercoast/tanager_stac_url",
                ",".join(self._item_stac_url(item) for item in self.items),
            )
            layer.setCustomProperty("hypercoast/tanager_result_footprints", "true")
            self._style_footprint_layer(layer)
            QgsProject.instance().addMapLayer(layer)
            self._result_footprints_layer = layer
            self._select_result_footprints()
            if zoom:
                self._zoom_to_layer(layer)
        except Exception as exc:
            if show_message:
                QMessageBox.warning(
                    self, "Tanager Footprints", f"Could not add footprints: {exc}"
                )

    def open_visual_layer(self):
        """Open the selected Tanager visual asset as a QGIS raster layer."""
        item = self.hydrated_selected_item()
        if item is None:
            return
        url = _asset_url(item, TANAGER_VISUAL_ASSET)
        if not url:
            QMessageBox.warning(
                self,
                "Tanager Visual",
                "The selected scene does not include an ortho visual asset.",
            )
            return

        layer_name = f"Tanager Visual - {_item_title(item)}"
        raster_layer = QgsRasterLayer(url, layer_name, "gdal")
        if not raster_layer.isValid():
            raster_layer = QgsRasterLayer(f"/vsicurl/{url}", layer_name, "gdal")
        if not raster_layer.isValid():
            QMessageBox.warning(
                self,
                "Tanager Visual",
                "QGIS could not open the ortho visual asset.",
            )
            return

        raster_layer.setCustomProperty("hypercoast/source_path", url)
        raster_layer.setCustomProperty("hypercoast/data_type", "Tanager Visual")
        raster_layer.setCustomProperty(
            "hypercoast/tanager_stac_url", item.get("_stac_url", "")
        )
        QgsProject.instance().addMapLayer(raster_layer)
        self.iface.setActiveLayer(raster_layer)
        self._zoom_to_layer(raster_layer)

    def open_stac_item(self):
        """Open the selected Tanager STAC item in the Planet STAC browser."""
        item = self.selected_item()
        if item is None:
            return
        stac_url = item.get("_stac_url") or item.get("properties", {}).get("stac_href")
        browser_url = _stac_browser_url(stac_url)
        if not browser_url:
            QMessageBox.warning(
                self,
                "Tanager STAC",
                "The selected scene does not include a STAC item URL.",
            )
            return
        QDesktopServices.openUrl(QUrl(browser_url))

    def download_hdf5(self):
        """Download the selected Tanager HDF5 asset."""
        item = self.hydrated_selected_item()
        if item is None:
            return
        asset_key = self.asset_combo.currentData()
        if not asset_key:
            QMessageBox.warning(
                self,
                "Tanager HDF5",
                "The selected scene does not include downloadable HDF5 assets.",
            )
            return
        if not _asset_url(item, asset_key):
            QMessageBox.warning(
                self,
                "Tanager HDF5",
                f"The selected scene does not include asset '{asset_key}'.",
            )
            return
        if self._load_worker is not None and self._load_worker.isRunning():
            return

        try:
            asset_url = _asset_url(item, asset_key)
            out_dir = self.download_dir_edit.text().strip() or tanager_download_dir(
                QgsProject.instance()
            )
            os.makedirs(out_dir, exist_ok=True)
            default_path = os.path.join(
                out_dir,
                _asset_filename(asset_url, fallback=f"{_item_title(item)}.h5"),
            )
            output_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Tanager HDF5",
                default_path,
                "HDF5 files (*.h5 *.hdf5);;All files (*)",
            )
            if not output_path:
                return
            if not os.path.splitext(output_path)[1]:
                output_path = f"{output_path}.h5"
            output_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(output_dir, exist_ok=True)
            self.download_dir_edit.setText(output_dir)

            self._set_busy(True)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(20)
            self.status_label.setText("Downloading Tanager HDF5...")
            self._load_worker = TanagerDownloadWorker(
                item,
                asset_key,
                output_path,
                parent=self,
            )
            self._load_worker.progress.connect(self.progress_bar.setValue)
            self._load_worker.finished.connect(self._on_download_finished)
            self._load_worker.start()
        except Exception as exc:
            self._set_busy(False)
            QMessageBox.warning(self, "Tanager HDF5", str(exc))

    def _on_download_finished(self, filepath, error_detail):
        """Handle Tanager download completion.

        Args:
            filepath: Downloaded HDF5 path.
            error_detail: Error message when downloading failed.
        """
        try:
            if error_detail:
                raise ValueError(error_detail)
            self.progress_bar.setValue(100)
            self.status_label.setText(
                f"Tanager HDF5 saved: {os.path.basename(filepath)}"
            )
        except Exception as exc:
            QMessageBox.warning(self, "Tanager HDF5", str(exc))
            QgsMessageLog.logMessage(
                f"Error downloading Tanager HDF5: {exc}",
                "HyperCoast",
                Qgis.MessageLevel.Warning,
            )
        finally:
            self._set_busy(False)

    def _style_footprint_layer(self, layer):
        """Apply the default Tanager footprint style.

        Args:
            layer: QGIS vector layer.
        """
        try:
            symbol = QgsFillSymbol.createSimple(
                {
                    "color": "173,216,230,80",
                    "outline_color": "0,102,204,255",
                    "outline_width": "0.6",
                }
            )
            layer.renderer().setSymbol(symbol)
            layer.triggerRepaint()
        except Exception:
            pass  # nosec B110

    def _zoom_to_layer(self, layer):
        """Zoom the map canvas to a layer extent.

        Args:
            layer: QGIS map layer.
        """
        try:
            canvas = self.iface.mapCanvas()
            extent = layer.extent()
            layer_crs = layer.crs()
            canvas_crs = canvas.mapSettings().destinationCrs()
            if layer_crs.isValid() and canvas_crs.isValid() and layer_crs != canvas_crs:
                transform = QgsCoordinateTransform(
                    layer_crs, canvas_crs, QgsProject.instance()
                )
                extent = transform.transformBoundingBox(extent)
            extent.scale(1.05)
            canvas.setExtent(extent)
            canvas.refresh()
        except Exception:
            pass  # nosec B110

    def _sync_button_state(self):
        """Enable or disable result actions based on current state."""
        has_results = self.results_table.rowCount() > 0
        has_result_footprints = has_results and self._has_item_footprints()
        has_selection = self.selected_item() is not None
        self.footprints_btn.setEnabled(has_result_footprints)
        self.visual_btn.setEnabled(has_selection)
        self.stac_btn.setEnabled(has_selection)
        self.download_btn.setEnabled(has_selection)

    def _on_result_selection_changed(self):
        """Refresh asset options after the selected result changes."""
        self._populate_asset_combo(self.selected_item())
        self._select_result_footprints()
        self._sync_button_state()

    def _populate_asset_combo(self, item=None):
        """Populate the HDF5 asset combo.

        Args:
            item: Optional selected STAC item used to filter available assets.
        """
        current = (
            self.asset_combo.currentData() if hasattr(self, "asset_combo") else None
        )
        available = item.get("assets", {}) if isinstance(item, dict) else None
        self.asset_combo.blockSignals(True)
        self.asset_combo.clear()
        for label, asset_key in TANAGER_HDF5_ASSETS:
            if available is None or asset_key in available:
                self.asset_combo.addItem(label, asset_key)
        if self.asset_combo.count() == 0:
            self.asset_combo.addItem("No HDF5 assets available", "")
        elif current:
            index = self.asset_combo.findData(current)
            if index >= 0:
                self.asset_combo.setCurrentIndex(index)
        self.asset_combo.blockSignals(False)

    def _stretch_result_columns(self):
        """Make result table columns use the full table width."""
        try:
            header = self.results_table.horizontalHeader()
            header.setStretchLastSection(True)
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        except AttributeError:
            header = self.results_table.horizontalHeader()
            header.setStretchLastSection(True)
            header.setSectionResizeMode(QHeaderView.Stretch)

    def _set_busy(self, busy):
        """Toggle controls while background work runs.

        Args:
            busy: Whether the dialog is busy.
        """
        self.progress_bar.setVisible(busy)
        self.search_btn.setEnabled(not busy)
        self.selected_footprints_btn.setEnabled(not busy)
        self.footprints_btn.setEnabled(
            (not busy)
            and self.results_table.rowCount() > 0
            and self._has_item_footprints()
        )
        self.visual_btn.setEnabled((not busy) and self.selected_item() is not None)
        self.stac_btn.setEnabled((not busy) and self.selected_item() is not None)
        self.download_btn.setEnabled((not busy) and self.selected_item() is not None)

    def _stop_worker(self, worker):
        """Request a worker thread to stop.

        Args:
            worker: QThread instance, or None.
        """
        if worker is None:
            return
        try:
            if worker.isRunning():
                worker.requestInterruption()
                worker.quit()
                worker.wait(5000)
        except RuntimeError:
            pass  # nosec B110

    def closeEvent(self, event):
        """Stop background workers before the dock closes.

        Args:
            event: Qt close event.
        """
        self._stop_worker(self._search_worker)
        self._stop_worker(self._load_worker)
        super().closeEvent(event)
