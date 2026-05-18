"""Tests for the Tanager search dock helpers."""

import os
from datetime import datetime

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from qgis.PyQt.QtWidgets import QApplication, QHeaderView

import hypercoast_qgis.hypercoast_plugin as plugin_module
import hypercoast_qgis.dialogs.tanager_search_dialog as dialog_module
from hypercoast_qgis.dialogs.tanager_search_dialog import (
    TANAGER_HDF5_ASSET,
    TANAGER_VISUAL_ASSET,
    TanagerSearchDialog,
    TanagerSearchWorker,
    _asset_filename,
    _asset_url,
    _stac_browser_url,
)
from hypercoast_qgis.hypercoast_plugin import HyperCoastPlugin


@pytest.fixture(scope="module")
def qapp():
    """Return a QApplication for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _Extent:
    """Small extent-like object."""

    def xMinimum(self):
        """Return xmin."""
        return -122.0

    def yMinimum(self):
        """Return ymin."""
        return 37.0

    def xMaximum(self):
        """Return xmax."""
        return -121.0

    def yMaximum(self):
        """Return ymax."""
        return 38.0


class _MapSettings:
    """Small map settings-like object."""

    def destinationCrs(self):
        """Return no CRS so no transform is attempted."""
        return None


class _Canvas:
    """Small map canvas-like object."""

    def extent(self):
        """Return current extent."""
        return _Extent()

    def mapSettings(self):
        """Return map settings."""
        return _MapSettings()


class _Iface:
    """Small QGIS iface-like object."""

    def mainWindow(self):
        """Return no parent window."""
        return None

    def mapCanvas(self):
        """Return a fake map canvas."""
        return _Canvas()


def _dialog(qapp, monkeypatch=None, tmp_path=None, skip_global=True):
    """Return an initialized Tanager dialog."""
    if monkeypatch is not None and skip_global:
        monkeypatch.setattr(
            dialog_module.TanagerSearchDialog,
            "add_global_footprints_layer",
            lambda self: None,
        )
    if tmp_path is not None:
        monkeypatch.setattr(
            dialog_module,
            "tanager_download_dir",
            lambda project=None: str(tmp_path),
        )
    plugin = HyperCoastPlugin.__new__(HyperCoastPlugin)
    plugin.register_hyperspectral_layer = lambda *args, **kwargs: None
    return TanagerSearchDialog(_Iface(), plugin)


def test_canvas_bbox_uses_map_extent():
    """Canvas bbox extraction should read the current map extent."""
    dialog = TanagerSearchDialog.__new__(TanagerSearchDialog)
    dialog.iface = _Iface()

    assert dialog.current_canvas_bbox() == [-122.0, 37.0, -121.0, 38.0]


def test_dialog_minimum_width_is_compact(qapp, monkeypatch, tmp_path):
    """The Tanager dock should fit into a compact side panel."""
    dialog = _dialog(qapp, monkeypatch=monkeypatch, tmp_path=tmp_path)

    assert dialog.minimumWidth() == 520


def test_populate_results_stores_items_and_asset_flags(qapp):
    """Search results should populate rows with stored STAC items."""
    dialog = _dialog(qapp)
    item = {
        "id": "scene-1",
        "_collection_title": "Coastal",
        "properties": {"datetime": "2025-01-15T00:00:00Z", "cloud_percent": 5},
        "assets": {
            TANAGER_VISUAL_ASSET: {"href": "https://example.com/visual.tif"},
            TANAGER_HDF5_ASSET: {"href": "https://example.com/radiance.h5"},
        },
    }

    dialog.populate_results([item])

    assert dialog.results_table.rowCount() == 1
    assert dialog.results_table.item(0, 0).text() == "scene-1"
    assert dialog.results_table.item(0, 4).text() == "yes"
    assert dialog.results_table.item(0, 5).text() == "yes"
    assert dialog.results_table.item(0, 0).data(256) == item
    assert (
        dialog.results_table.horizontalHeader().sectionResizeMode(0)
        == QHeaderView.ResizeMode.Stretch
    )


def test_search_finished_adds_footprints_without_gdf(qapp, monkeypatch, tmp_path):
    """STAC search results should add footprints without GeoDataFrame output."""
    dialog = _dialog(qapp, monkeypatch=monkeypatch, tmp_path=tmp_path)
    recorded = {}

    def _add_footprints_layer(**kwargs):
        """Record footprint layer creation arguments."""
        recorded.update(kwargs)

    monkeypatch.setattr(dialog, "add_footprints_layer", _add_footprints_layer)

    dialog._on_search_finished([{"id": "scene-1", "properties": {}}], None, "")

    assert dialog.gdf is None
    assert recorded == {"show_message": False, "zoom": True, "replace": True}


def test_table_selection_selects_matching_result_footprint(qapp, monkeypatch, tmp_path):
    """Selecting a result row should select the matching map footprint."""
    dialog = _dialog(qapp, monkeypatch=monkeypatch, tmp_path=tmp_path)
    selected = []

    class _Feature:
        """Small feature-like object."""

        def __init__(self, feature_id, item_id):
            """Initialize feature identifiers."""
            self.feature_id = feature_id
            self.item_id = item_id

        def id(self):
            """Return the feature id."""
            return self.feature_id

        def attribute(self, name):
            """Return the requested attribute."""
            if name == "id":
                return self.item_id
            return None

    class _Layer:
        """Small vector layer-like object."""

        def getFeatures(self):
            """Return footprint features."""
            return [_Feature(11, "scene-1"), _Feature(22, "scene-2")]

        def selectByIds(self, feature_ids):
            """Record selected feature ids."""
            selected[:] = feature_ids

    dialog._result_footprints_layer = _Layer()
    dialog.populate_results(
        [
            {"id": "scene-1", "properties": {}, "geometry": {"type": "Point"}},
            {"id": "scene-2", "properties": {}, "geometry": {"type": "Point"}},
        ]
    )

    dialog.results_table.selectRow(1)

    assert selected == [22]


def test_global_footprints_layer_is_added_and_styled(qapp, monkeypatch, tmp_path):
    """Opening the Tanager dock should add styled global footprints."""
    added = []
    symbols = []

    class _Provider:
        """Small provider-like object."""

        def addAttributes(self, fields):
            """Ignore fields."""

        def addFeatures(self, features):
            """Ignore features."""

    class _Renderer:
        """Small renderer-like object."""

        def setSymbol(self, symbol):
            """Record the assigned symbol."""
            symbols.append(symbol)

    class _Layer:
        """Small vector layer-like object."""

        def __init__(self, source, name, provider):
            """Initialize layer metadata."""
            self.source = source
            self.name = name
            self.provider = provider
            self.properties = {}
            self._renderer = _Renderer()

        def isValid(self):
            """Pretend the layer is valid."""
            return True

        def setCustomProperty(self, key, value):
            """Record custom properties."""
            self.properties[key] = value

        def customProperty(self, key, default=None):
            """Return custom properties."""
            return self.properties.get(key, default)

        def renderer(self):
            """Return the renderer."""
            return self._renderer

        def triggerRepaint(self):
            """Pretend to repaint."""

    class _Project:
        """Small project-like object."""

        def mapLayers(self):
            """Return no existing layers."""
            return {}

        def addMapLayer(self, layer):
            """Record the added layer."""
            added.append(layer)

    monkeypatch.setattr(dialog_module, "QgsVectorLayer", _Layer)
    monkeypatch.setattr(
        dialog_module.QgsFillSymbol,
        "createSimple",
        staticmethod(lambda spec: spec),
    )
    monkeypatch.setattr(
        dialog_module.QgsProject, "instance", staticmethod(lambda: _Project())
    )
    _dialog(qapp, tmp_path=tmp_path, monkeypatch=monkeypatch, skip_global=False)

    assert len(added) == 1
    assert added[0].properties["hypercoast/tanager_global_footprints"] == "true"
    assert symbols[0]["outline_color"] == "0,102,204,255"
    assert symbols[0]["color"] == "173,216,230,80"


def test_asset_url_returns_empty_for_missing_asset():
    """Missing Tanager assets should return an empty URL."""
    assert _asset_url({"assets": {}}, TANAGER_VISUAL_ASSET) == ""


def test_asset_combo_filters_to_selected_item_assets(qapp, monkeypatch, tmp_path):
    """Selecting a result should show only HDF5 assets available on that item."""
    dialog = _dialog(qapp, monkeypatch=monkeypatch, tmp_path=tmp_path)
    item = {
        "id": "scene-1",
        "properties": {},
        "assets": {"ortho_sr_hdf5": {"href": "https://example.com/sr.h5"}},
    }

    dialog.populate_results([item])
    dialog.results_table.selectRow(0)

    assert dialog.asset_combo.count() == 1
    assert dialog.asset_combo.currentData() == "ortho_sr_hdf5"


def test_selected_global_footprints_populate_results(qapp, monkeypatch, tmp_path):
    """Selected global footprint features should become table results."""
    dialog = _dialog(qapp, monkeypatch=monkeypatch, tmp_path=tmp_path)

    class _DateTime:
        """Small Qt date/time-like object."""

        def toPyDateTime(self):
            """Return a Python datetime."""
            return datetime(2025, 1, 15, 12, 30, 0)

    class _Feature:
        """Small selected feature-like object."""

        def __init__(self):
            """Initialize feature attributes."""
            self.attrs = {
                "id": "scene-1",
                "title": "Scene One",
                "datetime": _DateTime(),
                "collection": "coastal-water-bodies",
                "collection_title": "Coastal",
                "stac_href": "https://example.com/item.json",
                "asset_keys": "['ortho_visual', 'ortho_radiance_hdf5']",
            }

        def attribute(self, name):
            """Return a feature attribute."""
            return self.attrs.get(name)

    class _Layer:
        """Small global footprint layer-like object."""

        def customProperty(self, key, default=None):
            """Return global footprint marker."""
            if key == "hypercoast/tanager_global_footprints":
                return "true"
            return default

        def selectedFeatures(self):
            """Return selected features."""
            return [_Feature()]

    class _Project:
        """Small project-like object."""

        def mapLayers(self):
            """Return a global footprint layer."""
            return {"layer-1": _Layer()}

    monkeypatch.setattr(
        dialog_module.QgsProject, "instance", staticmethod(lambda: _Project())
    )

    dialog.populate_from_selected_footprints()

    item = dialog.results_table.item(0, 0).data(256)
    assert dialog.results_table.rowCount() == 1
    assert dialog.results_table.item(0, 0).text() == "scene-1"
    assert dialog.results_table.item(0, 1).text() == "2025-01-15T12:30:00"
    assert dialog.results_table.item(0, 4).text() == "yes"
    assert dialog.results_table.item(0, 5).text() == "yes"
    assert dialog.results_table.selectionModel().selectedRows()[0].row() == 0
    assert item["_stac_url"] == "https://example.com/item.json"
    assert "ortho_radiance_hdf5" in item["assets"]


def test_hydrated_selected_item_fetches_stac_asset_urls(qapp, monkeypatch, tmp_path):
    """Lightweight footprint rows should hydrate one STAC item on demand."""
    dialog = _dialog(qapp, monkeypatch=monkeypatch, tmp_path=tmp_path)
    item = {
        "id": "scene-1",
        "_stac_url": "https://example.com/item.json",
        "properties": {},
        "assets": {"ortho_visual": {}, "ortho_radiance_hdf5": {}},
    }
    dialog.populate_results([item])
    dialog.results_table.selectRow(0)

    class _Response:
        """Small HTTP response-like object."""

        def __enter__(self):
            """Return response context."""
            return self

        def __exit__(self, exc_type, exc, tb):
            """Exit response context."""

        def read(self):
            """Return STAC JSON bytes."""
            return (
                b'{"id": "scene-1", "assets": {'
                b'"ortho_visual": {"href": "https://example.com/visual.tif"},'
                b'"ortho_radiance_hdf5": {"href": "https://example.com/rad.h5"}}}'
            )

    monkeypatch.setattr(
        dialog_module.urllib.request,
        "urlopen",
        lambda url, timeout=30: _Response(),
    )

    hydrated = dialog.hydrated_selected_item()

    assert hydrated["assets"]["ortho_visual"]["href"].endswith("visual.tif")
    assert hydrated["_stac_url"] == "https://example.com/item.json"


def test_open_stac_item_uses_planet_browser_url(qapp, monkeypatch, tmp_path):
    """Open STAC should convert raw STAC URLs to browser URLs."""
    dialog = _dialog(qapp, monkeypatch=monkeypatch, tmp_path=tmp_path)
    item = {
        "id": "scene-1",
        "_stac_url": (
            "https://www.planet.com/data/stac/tanager-core-imagery/agriculture/"
            "20250223_165538_00_4001/20250223_165538_00_4001.json"
        ),
        "properties": {},
        "assets": {},
    }
    opened = []
    monkeypatch.setattr(
        dialog_module.QDesktopServices,
        "openUrl",
        lambda url: opened.append(url.toString()),
    )
    dialog.populate_results([item])
    dialog.results_table.selectRow(0)

    dialog.open_stac_item()

    assert opened == [
        "https://www.planet.com/data/stac/browser/tanager-core-imagery/"
        "agriculture/20250223_165538_00_4001/"
        "20250223_165538_00_4001.json"
    ]


def test_stac_browser_url_keeps_existing_browser_url():
    """Existing browser STAC URLs should be preserved."""
    url = "https://www.planet.com/data/stac/browser/a/b/item.json"

    assert _stac_browser_url(url) == url


def test_asset_filename_ignores_query_string():
    """Asset filenames should be parsed from URL paths."""
    url = "https://example.com/data/scene_ortho_radiance_hdf5.h5?token=abc"

    assert _asset_filename(url) == "scene_ortho_radiance_hdf5.h5"


def test_download_prompts_for_save_path_and_asset(qapp, monkeypatch, tmp_path):
    """Download should prompt for a destination HDF5 path."""
    dialog = _dialog(qapp, monkeypatch=monkeypatch, tmp_path=tmp_path)
    item = {
        "id": "scene-1",
        "properties": {"title": "Scene One"},
        "assets": {"basic_sr_hdf5": {"href": "https://example.com/sr.h5"}},
    }
    dialog.populate_results([item])
    dialog.results_table.selectRow(0)
    dialog.download_dir_edit.setText(str(tmp_path))
    save_path = str(tmp_path / "custom.h5")
    monkeypatch.setattr(
        dialog_module.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: (save_path, "")),
    )
    recorded = {}

    class _Signal:
        """Small Qt signal-like object."""

        def connect(self, callback):
            """Ignore signal connections."""

    class _Worker:
        """Small worker-like object."""

        def __init__(self, item, asset_key, output_path, parent):
            """Record worker arguments."""
            recorded.update(
                {
                    "item": item,
                    "asset_key": asset_key,
                    "output_path": output_path,
                    "parent": parent,
                }
            )
            self.progress = _Signal()
            self.finished = _Signal()

        def isRunning(self):
            """Return that the worker is idle."""
            return False

        def start(self):
            """Record that the worker started."""
            recorded["started"] = True

    monkeypatch.setattr(dialog_module, "TanagerDownloadWorker", _Worker)

    dialog.download_hdf5()

    assert recorded["asset_key"] == "basic_sr_hdf5"
    assert recorded["output_path"] == save_path
    assert dialog.download_dir_edit.text() == str(tmp_path)
    assert recorded["started"]


def test_search_worker_reports_errors(qapp, monkeypatch):
    """Search worker should emit an error instead of raising."""
    import hypercoast_qgis._hypercoast_lib as lib

    def _raise():
        """Raise a fake dependency error."""
        raise RuntimeError("missing dependency")

    monkeypatch.setattr(lib, "get_hypercoast", _raise)
    worker = TanagerSearchWorker({})
    emitted = []
    worker.finished.connect(lambda items, gdf, error: emitted.append(error))

    worker.run()

    assert emitted == ["missing dependency"]


def test_search_worker_requests_items_without_geodataframe(qapp, monkeypatch):
    """Search worker should avoid GeoDataFrame creation for faster searches."""
    import hypercoast_qgis._hypercoast_lib as lib

    recorded = {}

    class _HyperCoast:
        """Fake HyperCoast module."""

        def search_tanager(self, **params):
            """Record search parameters and return items."""
            recorded.update(params)
            return [{"id": "scene-1"}]

    monkeypatch.setattr(lib, "get_hypercoast", lambda: _HyperCoast())
    worker = TanagerSearchWorker({"count": 1})
    emitted = []
    worker.finished.connect(
        lambda items, gdf, error: emitted.append((items, gdf, error))
    )

    worker.run()

    assert recorded["return_gdf"] is False
    assert emitted == [([{"id": "scene-1"}], None, "")]


def test_tanager_action_is_dependency_gated():
    """The Tanager dock action should respect plugin dependency gating."""
    plugin = HyperCoastPlugin.__new__(HyperCoastPlugin)
    plugin._deps_available = False
    plugin.warning_shown = False
    plugin._show_deps_required_warning = lambda: setattr(plugin, "warning_shown", True)

    class _Action:
        """Small action-like object."""

        def __init__(self):
            """Initialize checked state."""
            self.checked = True

        def setChecked(self, checked):
            """Store checked state."""
            self.checked = checked

    plugin.tanager_search_action = _Action()

    plugin.show_tanager_search_dialog()

    assert plugin.warning_shown
    assert not plugin.tanager_search_action.checked


def test_tanager_action_is_first_toolbar_icon(qapp, monkeypatch):
    """The Tanager search action should be first in the toolbar."""
    added_actions = []

    class _Toolbar:
        """Small toolbar-like object."""

        def addAction(self, action):
            """Record added toolbar action."""
            added_actions.append(action)

    class _Iface:
        """Small QGIS iface-like object."""

        def mainWindow(self):
            """Return no parent window."""
            return None

        def addPluginToRasterMenu(self, menu, action):
            """Ignore menu registration."""

    plugin = HyperCoastPlugin.__new__(HyperCoastPlugin)
    plugin.iface = _Iface()
    plugin.plugin_dir = os.path.dirname(os.path.dirname(dialog_module.__file__))
    plugin.actions = []
    plugin.menu = plugin.tr("&HyperCoast")
    plugin.toolbar = _Toolbar()
    monkeypatch.setattr(plugin_module.QTimer, "singleShot", lambda *args: None)
    monkeypatch.setattr(
        plugin,
        "_connect_project_signals",
        lambda: None,
    )

    plugin.initGui()

    assert added_actions[0].text() == "Search Tanager Data"
