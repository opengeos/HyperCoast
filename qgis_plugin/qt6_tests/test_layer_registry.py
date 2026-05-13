"""Tests for HyperCoast layer metadata rehydration."""

from hypercoast_qgis.hypercoast_plugin import HyperCoastPlugin


class _Layer:
    """Small QGIS layer-like object with custom properties."""

    def __init__(self, properties):
        """Initialize the fake layer.

        Args:
            properties: Custom property mapping.
        """
        self._properties = properties

    def customProperty(self, key, default=None):
        """Return a fake custom property."""
        return self._properties.get(key, default)


class _Project:
    """Small QGIS project-like object."""

    def __init__(self, layers):
        """Initialize the fake project.

        Args:
            layers: Mapping of layer IDs to layer-like objects.
        """
        self._layers = layers

    def mapLayers(self):
        """Return fake project layers."""
        return self._layers


def _plugin():
    """Return a HyperCoastPlugin instance without constructing QGIS UI."""
    plugin = HyperCoastPlugin.__new__(HyperCoastPlugin)
    plugin.hyperspectral_data = {}
    return plugin


def test_rehydrate_parses_custom_properties(tmp_path, monkeypatch):
    """Custom properties should rebuild HyperCoast layer metadata."""
    source = tmp_path / "source.nc"
    source.write_text("", encoding="utf-8")
    layer = _Layer(
        {
            "hypercoast/source_path": str(source),
            "hypercoast/data_type": "PACE",
            "hypercoast/selected_variable": "chlor_a",
            "hypercoast/rgb_wavelengths": "650,550,450",
            "hypercoast/crs": "EPSG:4326",
        }
    )
    project = _Project({"layer-1": layer})
    import hypercoast_qgis.hypercoast_plugin as plugin_module

    monkeypatch.setattr(
        plugin_module.QgsProject, "instance", staticmethod(lambda: project)
    )
    plugin = _plugin()

    plugin.rehydrate_hyperspectral_layers()

    data_info = plugin.hyperspectral_data["layer-1"]
    assert data_info["dataset"] is None
    assert data_info["filepath"] == str(source)
    assert data_info["data_type"] == "PACE"
    assert data_info["selected_variable"] == "chlor_a"
    assert data_info["rgb_wavelengths"] == [650.0, 550.0, 450.0]
    assert data_info["crs"] == "EPSG:4326"


def test_rehydrate_skips_missing_source_files(monkeypatch):
    """Missing source files should not crash project metadata scanning."""
    layer = _Layer({"hypercoast/source_path": "/missing/source.nc"})
    project = _Project({"layer-1": layer})
    import hypercoast_qgis.hypercoast_plugin as plugin_module

    monkeypatch.setattr(
        plugin_module.QgsProject, "instance", staticmethod(lambda: project)
    )
    plugin = _plugin()

    plugin.rehydrate_hyperspectral_layers()

    assert plugin.hyperspectral_data == {}


def test_lazy_load_preserves_selected_variable(tmp_path, monkeypatch):
    """Lazy loading should pass the rehydrated selected variable to the provider."""
    source = tmp_path / "source.nc"
    source.write_text("", encoding="utf-8")
    plugin = _plugin()
    plugin.hyperspectral_data["layer-1"] = {
        "dataset": None,
        "filepath": str(source),
        "data_type": "PACE",
        "selected_variable": "chlor_a",
    }
    selected_variables = []

    class _Dataset:
        """Fake HyperspectralDataset that records selected variables."""

        def __init__(self, filepath, data_type):
            """Initialize fake dataset metadata."""
            self.filepath = filepath
            self.data_type = data_type
            self.dataset = object()
            self.wavelengths = [1, 2, 3]
            self.bounds = (0, 0, 1, 1)
            self.crs = "EPSG:4326"
            self.last_error = None

        def set_selected_variable(self, variable_name):
            """Record the selected variable."""
            selected_variables.append(variable_name)

        def load(self):
            """Pretend the dataset loaded successfully."""
            return True

    import hypercoast_qgis.hyperspectral_provider as provider_module

    monkeypatch.setattr(provider_module, "HyperspectralDataset", _Dataset)

    dataset = plugin.ensure_hyperspectral_dataset("layer-1")

    assert dataset is not None
    assert selected_variables == ["chlor_a"]
    assert plugin.hyperspectral_data["layer-1"]["dataset"] is dataset
