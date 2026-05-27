# SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Search-result workspace helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .registry import download_sensor


@dataclass
class SearchResult:
    """Represent reusable search results.

    Args:
        items: Search result items.
        sensor: Optional sensor name.
        query: Optional query metadata.
        gdf: Optional GeoDataFrame returned by search backends.
    """

    items: List[Any]
    sensor: Optional[str] = None
    query: Dict[str, Any] = field(default_factory=dict)
    gdf: Any = None

    @classmethod
    def from_result(
        cls,
        result: Any,
        sensor: Optional[str] = None,
        query: Optional[Dict[str, Any]] = None,
    ) -> "SearchResult":
        """Create a workspace from a backend search result.

        Args:
            result: List of items or ``(items, gdf)`` tuple.
            sensor: Optional sensor name.
            query: Optional query metadata.

        Returns:
            SearchResult: Search result workspace.
        """
        gdf = None
        items = result
        if isinstance(result, tuple) and len(result) >= 2:
            items, gdf = result[0], result[1]
        if isinstance(items, dict):
            items = items.get("items", [items])
        return cls(list(items or []), sensor=sensor, query=query or {}, gdf=gdf)

    @classmethod
    def from_json(cls, path: str | Path) -> "SearchResult":
        """Load search results from JSON.

        Args:
            path: JSON file path.

        Returns:
            SearchResult: Loaded search result workspace.
        """
        with open(path, encoding="utf-8") as src:
            payload = json.load(src)
        return cls(
            items=payload.get("items", []),
            sensor=payload.get("sensor"),
            query=payload.get("query", {}),
        )

    def to_json(self, path: str | Path) -> str:
        """Save search results to JSON.

        Args:
            path: Output JSON path.

        Returns:
            str: Output path.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sensor": self.sensor,
            "query": self.query,
            "items": self.items,
        }
        with open(output, "w", encoding="utf-8") as dst:
            json.dump(payload, dst, default=str, indent=2)
        return str(output)

    def to_dataframe(self) -> pd.DataFrame:
        """Return search items as a pandas DataFrame.

        Returns:
            pd.DataFrame: Tabular search results.
        """
        if self.gdf is not None:
            return pd.DataFrame(self.gdf)
        rows = [_item_summary(item) for item in self.items]
        return pd.DataFrame(rows)

    def to_csv(self, path: str | Path) -> str:
        """Save search results to CSV.

        Args:
            path: Output CSV path.

        Returns:
            str: Output path.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(output, index=False)
        return str(output)

    def to_file(self, path: str | Path, driver: Optional[str] = None) -> str:
        """Save geospatial search results when GeoDataFrame output is available.

        Args:
            path: Output file path.
            driver: Optional GeoPandas output driver.

        Returns:
            str: Output path.
        """
        if self.gdf is None:
            suffix = Path(path).suffix.lower()
            if suffix == ".csv":
                return self.to_csv(path)
            return self.to_json(path)
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        kwargs = {"driver": driver} if driver else {}
        self.gdf.to_file(output, **kwargs)
        return str(output)

    def filter(self, **properties: Any) -> "SearchResult":
        """Filter items by exact top-level or property values.

        Args:
            **properties: Values to match.

        Returns:
            SearchResult: Filtered result workspace.
        """
        items = [
            item
            for item in self.items
            if all(_item_value(item, key) == value for key, value in properties.items())
        ]
        return SearchResult(items=items, sensor=self.sensor, query=self.query)

    def download(self, out_dir: Optional[str] = None, **kwargs: Any) -> Any:
        """Download items with the registered sensor downloader.

        Args:
            out_dir: Optional output directory.
            **kwargs: Downloader keyword arguments.

        Returns:
            Any: Downloader result.
        """
        if not self.sensor:
            raise ValueError("SearchResult.sensor is required for download.")
        return download_sensor(self.sensor, self.items, out_dir=out_dir, **kwargs)


def load_search_result(path: str | Path) -> SearchResult:
    """Load a search-result workspace from JSON.

    Args:
        path: JSON file path.

    Returns:
        SearchResult: Loaded search result workspace.
    """
    return SearchResult.from_json(path)


def _item_summary(item: Any) -> Dict[str, Any]:
    """Return common summary fields for a search item."""
    if isinstance(item, dict):
        properties = item.get("properties", {})
        return {
            "id": item.get("id") or item.get("granule_ur"),
            "title": item.get("title") or properties.get("title"),
            "collection": item.get("collection") or item.get("collection_id"),
            "datetime": properties.get("datetime") or item.get("time_start"),
            "url": item.get("_stac_url") or item.get("href"),
        }
    return {"item": str(item)}


def _item_value(item: Any, key: str) -> Any:
    """Return a value from an item or its properties."""
    if not isinstance(item, dict):
        return None
    if key in item:
        return item[key]
    return item.get("properties", {}).get(key)


__all__ = ["SearchResult", "load_search_result"]
