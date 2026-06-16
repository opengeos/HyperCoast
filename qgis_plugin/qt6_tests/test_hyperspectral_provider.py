"""Tests for hyperspectral dataset variable selection."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from hypercoast_qgis.hyperspectral_provider import (
    DATA_TYPES,
    READER_DATA_TYPES,
    HyperspectralDataset,
)
from hypercoast_qgis.provider.detection import detect_data_type


def _spectral_dataset(variable="reflectance"):
    """Return a minimal spectral xarray dataset for loader/export tests."""
    return xr.Dataset(
        {
            variable: (
                ("wavelength", "y", "x"),
                np.ones((2, 2, 2), dtype="float32"),
            )
        },
        coords={"wavelength": np.array([500.0, 600.0])},
    )


def test_data_types_include_registry_metadata():
    """QGIS data types should expose registry defaults when available."""
    assert DATA_TYPES["PACE"]["variable"] == "Rrs"
    assert DATA_TYPES["Wyvern"]["default_rgb"] == [799, 679, 570]


def test_reader_data_types_exclude_generic():
    """Reader-backed data types should cover sensors but not the fallback."""
    assert "Generic" not in READER_DATA_TYPES
    assert "EMIT" in READER_DATA_TYPES
    assert "Wyvern" in READER_DATA_TYPES


def test_detect_data_type_matches_registry_sensor_by_name():
    """A newly registered sensor should auto-detect from its name alone."""
    data_types = {"EMIT": {}, "PACE": {}, "HYPERION": {}, "Generic": {}}

    assert detect_data_type("scene_HYPERION_001.tif", data_types=data_types) == (
        "HYPERION"
    )
    # Established alias tokens still resolve to their sensor.
    assert detect_data_type("PACE_OCI_2024.nc", data_types=data_types) == "PACE"
    assert detect_data_type("random_scene.tif", data_types=data_types) == "Generic"


def test_load_with_hypercoast_dispatches_through_read_sensor(monkeypatch):
    """Loading should route every sensor through the registry read_sensor."""
    import hypercoast_qgis.hyperspectral_provider as provider_module

    calls = {}

    class _HyperCoast:
        """Fake HyperCoast module recording the requested sensor."""

        def read_sensor(self, sensor, source, **kwargs):
            """Record the sensor name and return a spectral dataset."""
            calls["sensor"] = sensor
            calls["source"] = source
            return _spectral_dataset()

    monkeypatch.setattr(provider_module, "HAS_HYPERCOAST", True)
    monkeypatch.setattr(provider_module, "hypercoast", _HyperCoast(), raising=False)

    provider = HyperspectralDataset("AVIRIS_scene.nc", "AVIRIS")

    assert provider.load() is True
    assert calls["sensor"] == "AVIRIS"
    assert provider.dataset is not None


def test_export_with_hypercoast_dispatches_through_sensor_to_image(
    tmp_path, monkeypatch
):
    """Export should route reader-backed sensors through sensor_to_image."""
    import hypercoast_qgis.hyperspectral_provider as provider_module
    import leafmap

    calls = {}

    class _HyperCoast:
        """Fake HyperCoast module recording the converted sensor."""

        def sensor_to_image(self, sensor, data, **kwargs):
            """Record the sensor name and return a placeholder image."""
            calls["sensor"] = sensor
            return "IMAGE"

    def _fake_write(image, output, dtype=None):
        """Write a placeholder GeoTIFF so the export path can succeed."""
        with open(output, "wb") as handle:
            handle.write(b"fake geotiff")

    monkeypatch.setattr(provider_module, "HAS_HYPERCOAST", True)
    monkeypatch.setattr(provider_module, "hypercoast", _HyperCoast(), raising=False)
    monkeypatch.setattr(leafmap, "image_to_geotiff", _fake_write)

    provider = HyperspectralDataset("WYVERN_scene.tif", "Wyvern")
    provider.dataset = _spectral_dataset()
    output_path = tmp_path / "wyvern.tif"

    result = provider._export_with_hypercoast(str(output_path), [650, 550, 450])

    assert result == str(output_path)
    assert calls["sensor"] == "Wyvern"


def test_extract_metadata_returns_false_for_empty_pace():
    """Empty PACE datasets should fail metadata extraction."""
    provider = HyperspectralDataset("PACE_OCI_test.nc", "PACE")
    provider.dataset = xr.Dataset()

    assert provider._extract_metadata() is False


def test_tanager_hdf5_asset_filename_auto_detects_tanager():
    """Tanager asset filenames should not fall through to Generic."""
    provider = HyperspectralDataset("20240925_185509_74_4001_ortho_radiance_hdf5.h5")

    assert provider.data_type == "Tanager"


def test_tanager_hdf5_content_auto_detects_custom_filename(tmp_path):
    """Renamed Tanager HDF5 files should still auto-detect by content."""
    h5py = pytest.importorskip("h5py")
    filepath = tmp_path / "custom_name.h5"
    with h5py.File(filepath, "w") as h5_file:
        h5_file.create_dataset("product/toa_radiance", shape=(2, 3, 4))

    provider = HyperspectralDataset(str(filepath))

    assert provider.data_type == "Tanager"


def test_pace_bgc_prefers_raster_product_over_tilt():
    """PACE BGC products should select a 2D geophysical raster variable."""
    dataset = xr.Dataset(
        {
            "tilt": (("latitude",), np.array([1.0, 2.0])),
            "chlor_a": (
                ("latitude", "longitude"),
                np.array([[0.1, 0.2], [0.3, 0.4]], dtype="float32"),
            ),
            "poc": (
                ("latitude", "longitude"),
                np.array([[10.0, 20.0], [30.0, 40.0]], dtype="float32"),
            ),
            "l2_flags": (
                ("latitude", "longitude"),
                np.array([[1, 2], [3, 4]], dtype="int32"),
            ),
        },
        coords={
            "latitude": np.array([29.0, 30.0]),
            "longitude": np.array([-91.0, -90.0]),
        },
    )
    provider = HyperspectralDataset("PACE_OCI_test.nc", "PACE")
    provider.dataset = dataset

    data_var = provider.get_data_variable()

    assert data_var.name == "chlor_a"
    assert data_var.ndim == 2
    assert provider.list_data_variables() == ["chlor_a", "poc", "l2_flags"]


def test_selected_variable_overrides_default_product():
    """A selected export variable should override the provider default."""
    dataset = xr.Dataset(
        {
            "chlor_a": (
                ("latitude", "longitude"),
                np.array([[0.1, 0.2], [0.3, 0.4]], dtype="float32"),
            ),
            "poc": (
                ("latitude", "longitude"),
                np.array([[10.0, 20.0], [30.0, 40.0]], dtype="float32"),
            ),
        }
    )
    provider = HyperspectralDataset("PACE_OCI_test.nc", "PACE")
    provider.dataset = dataset
    provider.set_selected_variable("poc")

    data_var = provider.get_data_variable()

    assert data_var.name == "poc"


def test_pace_integer_variable_uses_nearest_interpolation():
    """Integer PACE variables should not be linearly interpolated."""
    dataset = xr.Dataset(
        {
            "l2_flags": (
                ("latitude", "longitude"),
                np.array([[1, 2], [3, 4]], dtype="int32"),
            )
        }
    )
    provider = HyperspectralDataset("PACE_OCI_test.nc", "PACE")
    provider.dataset = dataset

    method = provider._pace_bgc_interpolation_method("l2_flags", "linear")

    assert method == "nearest"


def test_north_up_orientation_flips_ascending_latitude():
    """Ascending latitude rows must be flipped before GeoTIFF writing."""
    provider = HyperspectralDataset("PACE_OCI_test.nc", "PACE")
    data = np.array([[1, 2], [3, 4]], dtype="float32")
    y_values = np.array([10.0, 20.0])

    oriented = provider._orient_array_north_up(data, y_values, y_axis=0)

    np.testing.assert_array_equal(oriented, np.array([[3, 4], [1, 2]]))


def test_projected_xy_spectral_extraction_uses_dataset_crs(monkeypatch):
    """Projected x/y datasets should use projected click coordinates."""
    import hypercoast_qgis.hyperspectral_provider as provider_module

    monkeypatch.setattr(provider_module, "HAS_HYPERCOAST", False)
    dataset = xr.Dataset(
        {
            "reflectance": (
                ("wavelength", "y", "x"),
                np.array(
                    [
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[10.0, 20.0], [30.0, 40.0]],
                    ],
                    dtype="float32",
                ),
            )
        },
        coords={
            "wavelength": np.array([500.0, 600.0]),
            "y": np.array([0.0, 100.0]),
            "x": np.array([0.0, 1000.0]),
        },
    )
    provider = HyperspectralDataset("generic.tif", "Generic")
    provider.dataset = dataset
    provider.wavelengths = dataset.coords["wavelength"].values
    provider.crs = "EPSG:32618"

    wavelengths, values = provider.extract_spectral_signature(
        510.0, 90.0, crs="EPSG:32618"
    )

    np.testing.assert_array_equal(wavelengths, np.array([500.0, 600.0]))
    np.testing.assert_array_equal(values, np.array([4.0, 40.0], dtype="float32"))


def test_lat_lon_spectral_extraction_uses_geographic_coordinates(monkeypatch):
    """Latitude/longitude datasets should still extract with EPSG:4326 input."""
    import hypercoast_qgis.hyperspectral_provider as provider_module

    monkeypatch.setattr(provider_module, "HAS_HYPERCOAST", False)
    dataset = xr.Dataset(
        {
            "reflectance": (
                ("wavelength", "latitude", "longitude"),
                np.array(
                    [
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[10.0, 20.0], [30.0, 40.0]],
                    ],
                    dtype="float32",
                ),
            )
        },
        coords={
            "wavelength": np.array([500.0, 600.0]),
            "latitude": np.array([30.0, 31.0]),
            "longitude": np.array([-91.0, -90.0]),
        },
    )
    provider = HyperspectralDataset("generic.nc", "Generic")
    provider.dataset = dataset
    provider.wavelengths = dataset.coords["wavelength"].values
    provider.crs = "EPSG:4326"

    wavelengths, values = provider.extract_spectral_signature(
        -90.9, 30.1, crs="EPSG:4326"
    )

    np.testing.assert_array_equal(wavelengths, np.array([500.0, 600.0]))
    np.testing.assert_array_equal(values, np.array([1.0, 10.0], dtype="float32"))


def test_tanager_2d_geolocation_spectral_extraction(monkeypatch):
    """Tanager y/x cubes should extract using 2D lat/lon geolocation."""
    import hypercoast_qgis.hyperspectral_provider as provider_module

    monkeypatch.setattr(provider_module, "HAS_HYPERCOAST", False)
    dataset = xr.Dataset(
        {
            "toa_radiance": (
                ("wavelength", "y", "x"),
                np.array(
                    [
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[10.0, 20.0], [30.0, 40.0]],
                    ],
                    dtype="float32",
                ),
            )
        },
        coords={
            "wavelength": np.array([500.0, 600.0]),
            "latitude": (
                ("y", "x"),
                np.array([[30.0, 30.0], [31.0, 31.0]]),
            ),
            "longitude": (
                ("y", "x"),
                np.array([[-91.0, -90.0], [-91.0, -90.0]]),
            ),
        },
    )
    provider = HyperspectralDataset("scene_ortho_radiance_hdf5.h5", "Tanager")
    provider.dataset = dataset
    provider.wavelengths = dataset.coords["wavelength"].values
    provider.crs = "EPSG:4326"

    wavelengths, values = provider.extract_spectral_signature(
        -90.1, 30.9, crs="EPSG:4326"
    )

    np.testing.assert_array_equal(wavelengths, np.array([500.0, 600.0]))
    np.testing.assert_array_equal(values, np.array([4.0, 40.0], dtype="float32"))


def test_emit_runtime_export_accepts_written_file_when_return_is_none(
    tmp_path, monkeypatch
):
    """HyperCoast writers may signal success by writing the requested file."""
    import hypercoast_qgis.hyperspectral_provider as provider_module

    output_path = tmp_path / "emit.tif"

    class _HyperCoast:
        """Fake HyperCoast module."""

        def emit_to_image(self, filepath, wavelengths, output):
            """Write the output file and return None."""
            with open(output, "wb") as fp:
                fp.write(b"fake geotiff")
            return None

    monkeypatch.setattr(provider_module, "HAS_HYPERCOAST", True)
    monkeypatch.setattr(provider_module, "hypercoast", _HyperCoast(), raising=False)
    provider = HyperspectralDataset("EMIT_test.nc", "EMIT")

    result = provider._export_emit_with_runtime_hypercoast(
        str(output_path), [650, 550, 450]
    )

    assert result == str(output_path)


def test_per_pixel_geolocation_requires_geolocation_export():
    """Datasets with 2D lat/lon coordinates should not use bbox fallback."""
    dataset = xr.Dataset(
        {
            "reflectance": (
                ("wavelength", "downtrack", "crosstrack"),
                np.ones((2, 2, 2), dtype="float32"),
            ),
            "latitude": (
                ("downtrack", "crosstrack"),
                np.array([[30.0, 30.1], [31.0, 31.1]]),
            ),
            "longitude": (
                ("downtrack", "crosstrack"),
                np.array([[-91.0, -90.9], [-91.1, -90.8]]),
            ),
        },
        coords={"wavelength": np.array([500.0, 600.0])},
    )
    provider = HyperspectralDataset("EMIT_test.nc", "EMIT")
    provider.dataset = dataset

    assert provider._requires_geolocation_export()
