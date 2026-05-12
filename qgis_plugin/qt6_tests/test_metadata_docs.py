"""Tests for QGIS plugin metadata and docs consistency."""

import configparser


def _metadata_path():
    """Return the plugin metadata path."""
    return __import__("pathlib").Path(__file__).resolve().parents[1] / (
        "hypercoast_qgis/metadata.txt"
    )


def test_readme_qgis_minimum_matches_metadata():
    """README should advertise the same QGIS minimum version as metadata."""
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    parser = configparser.ConfigParser()
    parser.read(_metadata_path(), encoding="utf-8")
    minimum = parser["general"]["qgisMinimumVersion"]

    readme = (root / "README.md").read_text(encoding="utf-8")
    assert f"QGIS {minimum} or later" in readme


def test_processing_provider_is_advertised():
    """The metadata should expose the implemented Processing provider."""
    parser = configparser.ConfigParser()
    parser.read(_metadata_path(), encoding="utf-8")
    assert parser["general"]["hasProcessingProvider"] == "yes"
