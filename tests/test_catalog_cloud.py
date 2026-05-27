# SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for catalog workspaces and cloud helpers."""

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

from hypercoast.catalog import SearchResult, load_search_result
from hypercoast.cloud import suggest_chunks


def test_search_result_json_csv_and_filter(tmp_path):
    """SearchResult should save, load, tabulate, and filter items."""
    result = SearchResult(
        items=[
            {"id": "a", "properties": {"datetime": "2024-01-01", "cloud": 10}},
            {"id": "b", "properties": {"datetime": "2024-01-02", "cloud": 80}},
        ],
        sensor="pace",
        query={"count": 2},
    )
    json_path = tmp_path / "search.json"
    csv_path = tmp_path / "search.csv"

    result.to_json(json_path)
    loaded = load_search_result(json_path)
    loaded.to_csv(csv_path)
    filtered = loaded.filter(cloud=10)

    assert loaded.sensor == "pace"
    assert csv_path.exists()
    assert len(filtered.items) == 1
    assert filtered.items[0]["id"] == "a"


def test_search_result_from_single_dict_item():
    """Dictionary items should not be expanded into keys."""
    result = SearchResult.from_result({"id": "scene-1"}, sensor="pace")

    assert result.items == [{"id": "scene-1"}]


def test_suggest_chunks_prioritizes_spectral_dimension():
    """Chunk suggestions should keep spectral chunks smaller than spatial ones."""
    data = xr.DataArray(
        np.zeros((64, 100, 200), dtype="float32"),
        dims=("wavelength", "y", "x"),
    )

    chunks = suggest_chunks(data, target_pixels=10_000)

    assert chunks["wavelength"] == 32
    assert chunks["y"] == 100
    assert chunks["x"] == 100
