import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "hypercoast" / "cesl.py"
SPEC = importlib.util.spec_from_file_location("hypercoast_cesl", MODULE_PATH)
cesl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cesl)

try:
    import pandas  # noqa: F401

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib  # noqa: F401

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class TestCesl(unittest.TestCase):
    def _mock_response(self, payload):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = payload
        return response

    @patch.object(cesl.requests, "get")
    def test_search_cesl_formats_parameters(self, mock_get):
        mock_get.return_value = self._mock_response({"catalog": {"ids": [1, 2, 3]}})

        ids = cesl.search_cesl(
            bbox=(44.0, 42.0, -70.0, -72.0),
            biomass=True,
            coverage=80,
            taxonomy="Fucus/vesiculosus",
        )

        self.assertEqual(ids, [1, 2, 3])
        mock_get.assert_called_once()
        _, kwargs = mock_get.call_args
        self.assertEqual(kwargs["params"]["format"], "json")
        self.assertEqual(kwargs["params"]["bbox"], "(44.0,42.0,-70.0,-72.0)")
        self.assertEqual(kwargs["params"]["biomass"], "true")
        self.assertEqual(kwargs["params"]["coverage"], 80)
        self.assertEqual(kwargs["params"]["taxonomy"], "Fucus/vesiculosus")

    @patch.object(cesl.requests, "get")
    def test_get_cesl_metadata_normalizes_values(self, mock_get):
        mock_get.return_value = self._mock_response(
            {
                "sample-metadata": {
                    "location_lat": {"units": "degrees", "value": 43.8},
                    "location_lon": {"units": "degrees", "value": -69.8},
                    "site_name": {"units": None, "value": "TNC Basin"},
                }
            }
        )

        metadata = cesl.get_cesl_metadata(216)

        self.assertEqual(metadata["location_lat"], 43.8)
        self.assertEqual(metadata["location_lon"], -69.8)
        self.assertEqual(metadata["site_name"], "TNC Basin")

    @unittest.skipUnless(HAS_PANDAS, "pandas is required for spectrum tables")
    @patch.object(cesl.requests, "get")
    def test_get_cesl_spectrum_resolves_case_insensitive_key(self, mock_get):
        mock_get.return_value = self._mock_response(
            {
                "sample-data": {
                    "wavelength": [400, 500, 600],
                    "Reflectance": [0.1, 0.2, 0.3],
                }
            }
        )

        spectrum = cesl.get_cesl_spectrum(216, spectrum_key="reflectance")

        self.assertEqual(list(spectrum.columns), ["wavelength", "Reflectance"])
        self.assertEqual(spectrum.attrs["sample_id"], 216)
        self.assertEqual(spectrum.attrs["spectrum_key"], "Reflectance")
        self.assertEqual(spectrum["Reflectance"].tolist(), [0.1, 0.2, 0.3])

    @unittest.skipUnless(
        HAS_PANDAS and HAS_MATPLOTLIB,
        "pandas and matplotlib are required for CESL plotting tests",
    )
    @patch.object(cesl.requests, "get")
    def test_plot_cesl_spectrum_returns_axes(self, mock_get):
        mock_get.return_value = self._mock_response(
            {
                "sample-data": {
                    "wavelength": [400, 500, 600],
                    "Reflectance": [0.1, 0.2, 0.3],
                }
            }
        )

        ax = cesl.plot_cesl_spectrum(216, color="red")

        self.assertEqual(ax.get_title(), "CESL Sample 216")
        self.assertEqual(ax.get_xlabel(), "Wavelength (nm)")
        self.assertEqual(ax.get_ylabel(), "Reflectance")
        self.assertEqual(len(ax.lines), 1)
        self.assertEqual(tuple(ax.figure.get_size_inches()), (6.4, 4.8))

    @unittest.skipUnless(
        HAS_PANDAS and HAS_MATPLOTLIB,
        "pandas and matplotlib are required for CESL plotting tests",
    )
    @patch.object(cesl.requests, "get")
    def test_plot_cesl_spectrum_applies_figsize(self, mock_get):
        mock_get.return_value = self._mock_response(
            {
                "sample-data": {
                    "wavelength": [400, 500, 600],
                    "Reflectance": [0.1, 0.2, 0.3],
                }
            }
        )

        ax = cesl.plot_cesl_spectrum(216, figsize=(10, 4))

        self.assertEqual(tuple(ax.figure.get_size_inches()), (10.0, 4.0))

    @unittest.skipUnless(
        HAS_PANDAS and HAS_MATPLOTLIB,
        "pandas and matplotlib are required for CESL plotting tests",
    )
    @patch.object(cesl.requests, "get")
    def test_plot_cesl_spectrum_applies_axis_ranges(self, mock_get):
        mock_get.return_value = self._mock_response(
            {
                "sample-data": {
                    "wavelength": [400, 500, 600],
                    "Reflectance": [0.1, 5.0, 0.3],
                }
            }
        )

        ax = cesl.plot_cesl_spectrum(
            216,
            x_range=(425, 575),
            y_range=(0.0, 1.0),
        )

        self.assertEqual(ax.get_xlim(), (425.0, 575.0))
        self.assertEqual(ax.get_ylim(), (0.0, 1.0))

    @patch.object(cesl, "get_cesl_metadata")
    def test_cesl_to_geojson_writes_output(self, mock_get_metadata):
        metadata_by_id = {
            101: {
                "location_lat": 43.8,
                "location_lon": -69.8,
                "site_name": "Site A",
            },
            102: {
                "location_lat": 43.9,
                "location_lon": -69.7,
                "site_name": "Site B",
            },
        }

        mock_get_metadata.side_effect = lambda sample_id, **_: metadata_by_id[sample_id]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = f"{tmp_dir}/cesl_sites.geojson"
            geojson = cesl.cesl_to_geojson(
                output=output,
                sample_ids=[101, 102],
                max_workers=2,
            )

            self.assertEqual(geojson["type"], "FeatureCollection")
            self.assertEqual(len(geojson["features"]), 2)
            self.assertTrue(
                all("sample_id" in item["properties"] for item in geojson["features"])
            )

            with open(output, encoding="utf-8") as file:
                saved = json.load(file)

            self.assertEqual(saved["type"], "FeatureCollection")
            self.assertEqual(saved["features"][0]["geometry"]["type"], "Point")

    @patch.object(cesl, "get_cesl_metadata")
    def test_get_cesl_sites_supports_unit_wrapped_coordinates(self, mock_get_metadata):
        mock_get_metadata.return_value = {
            "location_lat": {"value": 43.8, "units": "degrees"},
            "location_lon": {"value": -69.8, "units": "degrees"},
            "site_name": {"value": "Site A", "units": None},
        }

        records = cesl.get_cesl_sites(sample_ids=[101], include_units=True)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["latitude"], 43.8)
        self.assertEqual(records[0]["longitude"], -69.8)


if __name__ == "__main__":
    unittest.main()
