# -*- coding: utf-8 -*-
"""
HyperCoast QGIS Plugin - Tanager Search Dialog

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

import os
import ast
import json
import urllib.request

from qgis.PyQt.QtCore import Qt, QThread, QUrl, pyqtSignal
from qgis.PyQt.QtGui import QDesktopServices

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
    QgsContrastEnhancement,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsFillSymbol,
    QgsGeometry,
    QgsMessageLog,
    QgsMultiBandColorRenderer,
    QgsProject,
    QgsRasterLayer,
    QgsSingleBandGrayRenderer,
    QgsVectorLayer,
    Qgis,
)

from ..cache_manager import create_generated_raster_path, generated_raster_cache_dir
from ..hyperspectral_provider import HyperspectralDataset

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
        project: Optional QgsProject-like object.

    Returns:
        Absolute directory path.
    """
    out_dir = os.path.join(generated_raster_cache_dir(project), "tanager")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


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
            params["return_gdf"] = True
            self.progress.emit(50)
            items, gdf = hypercoast.search_tanager(**params)
            self.progress.emit(90)
            self.finished.emit(items, gdf, "")
        except Exception as exc:
            self.finished.emit([], None, str(exc))


class TanagerDownloadLoadWorker(QThread):
    """Worker thread for downloading and loading a Tanager HDF5 asset."""

    progress = pyqtSignal(int)
    finished = pyqtSignal(object, str, str, str)

    def __init__(
        self,
        item,
        asset_key,
        out_dir,
        output_path,
        wavelengths,
        parent=None,
    ):
        """Initialize the worker.

        Args:
            item: STAC item dictionary.
            asset_key: STAC asset key to download.
            out_dir: Download directory.
            output_path: GeoTIFF output path.
            wavelengths: RGB wavelengths.
            parent: Optional parent QObject.
        """
        super().__init__(parent)
        self.item = item
        self.asset_key = asset_key
        self.out_dir = out_dir
        self.output_path = output_path
        self.wavelengths = wavelengths

    def run(self):
        """Download the HDF5 asset and export an RGB raster."""
        try:
            from .._hypercoast_lib import get_hypercoast

            self.progress.emit(15)
            hypercoast = get_hypercoast()
            paths = hypercoast.download_tanager(
                self.item,
                asset=self.asset_key,
                out_dir=self.out_dir,
                quiet=True,
                overwrite=False,
            )
            if not paths:
                raise ValueError("Tanager download produced no file.")

            filepath = paths[0]
            self.progress.emit(45)
            dataset = HyperspectralDataset(filepath, "Tanager")
            result = dataset.load_and_export(
                self.output_path, wavelengths=self.wavelengths
            )
            if result is None:
                detail = getattr(dataset, "last_error", None) or "Unknown error"
                self.finished.emit(None, filepath, "", detail)
                return

            self.progress.emit(90)
            self.finished.emit(dataset, filepath, self.output_path, "")
        except Exception as exc:
            self.finished.emit(None, "", "", str(exc))


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

        self.setWindowTitle("Search Tanager Data")
        self.setObjectName("HyperCoastTanagerSearchDock")
        self.setMinimumWidth(680)
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

        button_layout = QHBoxLayout()
        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self.search)
        button_layout.addWidget(self.search_btn)

        self.selected_footprints_btn = QPushButton("Use Selected Footprints")
        self.selected_footprints_btn.clicked.connect(
            self.populate_from_selected_footprints
        )
        button_layout.addWidget(self.selected_footprints_btn)

        self.footprints_btn = QPushButton("Add Footprints")
        self.footprints_btn.clicked.connect(self.add_footprints_layer)
        button_layout.addWidget(self.footprints_btn)

        self.visual_btn = QPushButton("Open Visual")
        self.visual_btn.clicked.connect(self.open_visual_layer)
        button_layout.addWidget(self.visual_btn)

        self.stac_btn = QPushButton("Open STAC")
        self.stac_btn.clicked.connect(self.open_stac_item)
        button_layout.addWidget(self.stac_btn)

        self.download_btn = QPushButton("Download HDF5")
        self.download_btn.clicked.connect(self.download_hdf5)
        button_layout.addWidget(self.download_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
        self._sync_button_state()

    def refresh_extent(self):
        """Refresh the bbox field from the current QGIS map extent."""
        bbox = self.current_canvas_bbox()
        if bbox:
            self.bbox_edit.setText(", ".join(f"{value:.8f}" for value in bbox))
            self.use_extent_check.setChecked(True)

    def clear_bbox(self):
        """Clear the bbox filter for a global Tanager search."""
        self.use_extent_check.setChecked(False)
        self.bbox_edit.clear()

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

    def add_footprints_layer(self):
        """Add Tanager result footprints as an in-memory vector layer."""
        if self.gdf is None or len(self.items) == 0:
            QMessageBox.information(self, "Tanager Footprints", "No results to add.")
            return

        try:
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
            for item, (_, row) in zip(self.items, self.gdf.iterrows()):
                geometry = row.geometry
                if geometry is None or geometry.is_empty:
                    continue
                properties = item.get("properties", {})
                feature = QgsFeature(layer.fields())
                feature.setGeometry(QgsGeometry.fromWkt(geometry.wkt))
                feature.setAttributes(
                    [
                        item.get("id", ""),
                        properties.get("datetime", ""),
                        item.get("_collection_title") or item.get("collection", ""),
                        properties.get("cloud_percent"),
                        item.get("_stac_url", ""),
                        _asset_url(item, TANAGER_VISUAL_ASSET),
                        _asset_url(item, TANAGER_HDF5_ASSET),
                    ]
                )
                features.append(feature)

            provider.addFeatures(features)
            layer.updateExtents()
            layer.setCustomProperty(
                "hypercoast/tanager_stac_url",
                ",".join(item.get("_stac_url", "") for item in self.items),
            )
            self._style_footprint_layer(layer)
            QgsProject.instance().addMapLayer(layer)
            self._zoom_to_layer(layer)
        except Exception as exc:
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
        """Download and load the selected Tanager radiance HDF5 asset."""
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
            out_dir = self.download_dir_edit.text().strip()
            if not out_dir:
                raise ValueError("Please select a download folder.")
            os.makedirs(out_dir, exist_ok=True)
            asset_label = self.asset_combo.currentText() or asset_key
            layer_name = f"Tanager {asset_label} - {_item_title(item)}"
            output_path = create_generated_raster_path(
                layer_name, "rgb", project=QgsProject.instance()
            )
            wavelengths = [650, 550, 450]
            self._pending_load_context = {
                "layer_name": layer_name,
                "wavelengths": wavelengths,
                "stac_url": item.get("_stac_url", ""),
            }
            self._set_busy(True)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(20)
            self.status_label.setText("Downloading Tanager HDF5...")
            self._load_worker = TanagerDownloadLoadWorker(
                item,
                asset_key,
                out_dir,
                output_path,
                wavelengths,
                parent=self,
            )
            self._load_worker.progress.connect(self.progress_bar.setValue)
            self._load_worker.finished.connect(self._on_download_load_finished)
            self._load_worker.start()
        except Exception as exc:
            self._pending_load_context = None
            self._set_busy(False)
            QMessageBox.warning(self, "Tanager HDF5", str(exc))

    def _on_download_load_finished(self, dataset, filepath, temp_path, error_detail):
        """Handle Tanager download and load worker completion.

        Args:
            dataset: Loaded HyperspectralDataset, or None on failure.
            filepath: Downloaded HDF5 path.
            temp_path: Exported GeoTIFF path.
            error_detail: Error message when loading failed.
        """
        if self._pending_load_context is not None and filepath:
            self._pending_load_context["filepath"] = filepath
        self._on_load_finished(dataset, temp_path, error_detail)

    def _on_load_finished(self, dataset, temp_path, error_detail):
        """Add the loaded Tanager HDF5 raster to QGIS.

        Args:
            dataset: Loaded HyperspectralDataset, or None on failure.
            temp_path: Exported GeoTIFF path.
            error_detail: Error message when loading failed.
        """
        try:
            context = self._pending_load_context or {}
            if dataset is None:
                raise ValueError(error_detail or "Failed to load Tanager HDF5.")

            layer_name = context.get("layer_name", "Tanager Radiance")
            raster_layer = QgsRasterLayer(temp_path, layer_name)
            if not raster_layer.isValid():
                raise ValueError("Created Tanager raster layer is not valid.")

            if (not raster_layer.crs().isValid()) and dataset.crs:
                known_crs = QgsCoordinateReferenceSystem(str(dataset.crs))
                if known_crs.isValid():
                    raster_layer.setCrs(known_crs)

            self._style_raster_layer(raster_layer)
            self._set_hyperspectral_properties(raster_layer, dataset, context)
            QgsProject.instance().addMapLayer(raster_layer)

            selected_data_var = dataset.get_data_variable()
            selected_variable = getattr(selected_data_var, "name", None)
            self.plugin.register_hyperspectral_layer(
                raster_layer.id(),
                {
                    "dataset": dataset,
                    "filepath": context.get("filepath", ""),
                    "data_type": dataset.data_type,
                    "wavelengths": dataset.wavelengths,
                    "rgb_wavelengths": context.get("wavelengths", []),
                    "selected_variable": selected_variable,
                    "bounds": dataset.bounds,
                    "crs": dataset.crs,
                },
            )
            spectral_plot = getattr(self.plugin, "spectral_plot_dialog", None)
            if spectral_plot is not None:
                spectral_plot.clear_all_spectra()

            self.iface.setActiveLayer(raster_layer)
            self._zoom_to_layer(raster_layer)
            self.progress_bar.setValue(100)
            self.status_label.setText("Tanager HDF5 loaded")
        except Exception as exc:
            QMessageBox.warning(self, "Tanager HDF5", str(exc))
            QgsMessageLog.logMessage(
                f"Error loading Tanager HDF5: {exc}",
                "HyperCoast",
                Qgis.MessageLevel.Warning,
            )
        finally:
            self._pending_load_context = None
            self._set_busy(False)

    def _style_raster_layer(self, raster_layer):
        """Apply a default renderer to a generated Tanager raster.

        Args:
            raster_layer: QGIS raster layer.
        """
        provider = raster_layer.dataProvider()
        if provider is None:
            return
        for band_idx in range(1, raster_layer.bandCount() + 1):
            try:
                provider.setNoDataValue(band_idx, -9999.0)
            except Exception:
                pass  # nosec B110

        if raster_layer.bandCount() >= 3:
            renderer = QgsMultiBandColorRenderer(provider, 1, 2, 3)
            renderer_bands = [1, 2, 3]
        else:
            renderer = QgsSingleBandGrayRenderer(provider, 1)
            renderer_bands = [1]

        for band_idx in renderer_bands:
            ce = QgsContrastEnhancement(provider.dataType(band_idx))
            ce.setContrastEnhancementAlgorithm(
                QgsContrastEnhancement.StretchToMinimumMaximum
            )
            ce.setMinimumValue(0)
            ce.setMaximumValue(0.5)
            if raster_layer.bandCount() < 3:
                renderer.setContrastEnhancement(ce)
            elif band_idx == 1:
                renderer.setRedContrastEnhancement(ce)
            elif band_idx == 2:
                renderer.setGreenContrastEnhancement(ce)
            else:
                renderer.setBlueContrastEnhancement(ce)
        raster_layer.setRenderer(renderer)

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

    def _set_hyperspectral_properties(self, raster_layer, dataset, context):
        """Persist HyperCoast metadata on a Tanager raster layer.

        Args:
            raster_layer: Layer receiving custom properties.
            dataset: Loaded Tanager dataset provider.
            context: Pending load context.
        """
        raster_layer.setCustomProperty(
            "hypercoast/source_path", context.get("filepath", "")
        )
        raster_layer.setCustomProperty("hypercoast/data_type", dataset.data_type)
        raster_layer.setCustomProperty(
            "hypercoast/rgb_wavelengths",
            ",".join(str(value) for value in context.get("wavelengths", [])),
        )
        raster_layer.setCustomProperty(
            "hypercoast/tanager_stac_url", context.get("stac_url", "")
        )
        if dataset.crs:
            raster_layer.setCustomProperty("hypercoast/crs", str(dataset.crs))

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
        has_result_footprints = has_results and self.gdf is not None
        has_selection = self.selected_item() is not None
        self.footprints_btn.setEnabled(has_result_footprints)
        self.visual_btn.setEnabled(has_selection)
        self.stac_btn.setEnabled(has_selection)
        self.download_btn.setEnabled(has_selection)

    def _on_result_selection_changed(self):
        """Refresh asset options after the selected result changes."""
        self._populate_asset_combo(self.selected_item())
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
            (not busy) and self.results_table.rowCount() > 0 and self.gdf is not None
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
