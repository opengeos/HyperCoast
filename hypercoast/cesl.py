# SPDX-FileCopyrightText: 2026 Qiusheng Wu <giswqs@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Utilities for working with the Coastal Ecosystem Spectral Library (CESL)."""

from __future__ import annotations

import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence

import requests

if TYPE_CHECKING:
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import pandas as pd

CESL_API_BASE_URL = "https://speclib.nearview.net/api"
_CATALOG_KEY_CANDIDATES = ("catalog",)
_METADATA_KEY_CANDIDATES = ("sample-metadata", "sample/metadata")
_DATA_KEY_CANDIDATES = ("sample-data", "sample/data")
_LATITUDE_KEYS = ("location_lat", "latitude", "lat", "site_lat", "site_latitude")
_LONGITUDE_KEYS = (
    "location_lon",
    "longitude",
    "lon",
    "lng",
    "site_lon",
    "site_longitude",
)


def _get_payload(data: Dict[str, Any], candidates: Sequence[str]) -> Dict[str, Any]:
    """Return the first matching payload from a CESL API response."""

    for key in candidates:
        if key in data:
            return data[key]

    raise KeyError(f"Could not find any of {candidates} in the CESL response.")


def _format_catalog_param(
    value: Optional[Sequence[float]], expected_length: int, name: str
) -> Optional[str]:
    """Format tuple-based CESL catalog parameters like bbox and circle."""

    if value is None:
        return None

    if len(value) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} values.")

    return f"({','.join(str(item) for item in value)})"


def _normalize_cesl_value(value: Any, include_units: bool = False) -> Any:
    """Flatten CESL metadata fields to JSON-friendly values."""

    if not isinstance(value, dict):
        return value

    if "value" not in value:
        return value

    normalized = value.get("value")
    units = value.get("units")

    if include_units:
        return {"value": normalized, "units": units}

    return normalized


def _normalize_cesl_metadata(
    metadata: Dict[str, Any], include_units: bool = False
) -> Dict[str, Any]:
    """Normalize the CESL metadata payload."""

    return {
        key: _normalize_cesl_value(value, include_units=include_units)
        for key, value in metadata.items()
    }


class _MissingCoordinateError(ValueError):
    """Raised when a required coordinate cannot be found in CESL metadata."""


def _extract_coordinate(
    metadata: Dict[str, Any], candidates: Sequence[str], label: str
) -> float:
    """Extract a coordinate from normalized CESL metadata."""

    for key in candidates:
        value = metadata.get(key)
        if isinstance(value, dict) and "value" in value:
            value = value["value"]
        if value not in (None, ""):
            return float(value)

    raise _MissingCoordinateError(f"Could not find {label} in CESL metadata.")


def _build_feature(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a normalized CESL record to a GeoJSON feature."""

    properties = {
        key: value
        for key, value in record.items()
        if key not in {"latitude", "longitude"}
    }

    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [record["longitude"], record["latitude"]],
        },
        "properties": properties,
    }


def _request_cesl(
    endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30
) -> Dict[str, Any]:
    """Send a request to the CESL API and return the parsed JSON payload."""

    url = f"{CESL_API_BASE_URL}/{endpoint.lstrip('/')}"
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    try:
        data = response.json()
    except Exception as exc:
        raise ValueError(
            f"CESL API returned non-JSON response for {url!r}: {response.text[:200]!r}"
        ) from exc

    if "error" in data:
        error = data["error"]
        message = error.get("message", "Unknown CESL API error.")
        code = error.get("code")
        raise ValueError(f"CESL API error {code}: {message}")

    return data


def search_cesl(
    bbox: Optional[Sequence[float]] = None,
    circle: Optional[Sequence[float]] = None,
    publish_date_start: Optional[str] = None,
    publish_date_end: Optional[str] = None,
    taxonomy: Optional[str] = None,
    biomass: Optional[bool] = None,
    coverage: Optional[float] = None,
    timeout: int = 30,
) -> List[int]:
    """Search the CESL catalog and return matching sample IDs.

    Args:
        bbox (Sequence[float], optional): Bounding box formatted as
            ``(north, south, east, west)``.
        circle (Sequence[float], optional): Circular search formatted as
            ``(latitude, longitude, radius_km)``.
        publish_date_start (str, optional): ISO8601 start date for published
            samples.
        publish_date_end (str, optional): ISO8601 end date for published
            samples.
        taxonomy (str, optional): Binomial or taxonomy path filter.
        biomass (bool, optional): Whether biomass measurements are required.
        coverage (float, optional): Minimum percentage coverage threshold.
        timeout (int, optional): Request timeout in seconds. Defaults to 30.

    Returns:
        list[int]: Matching CESL sample IDs.
    """

    params: Dict[str, Any] = {"format": "json"}
    formatted_bbox = _format_catalog_param(bbox, 4, "bbox")
    formatted_circle = _format_catalog_param(circle, 3, "circle")

    if formatted_bbox is not None:
        params["bbox"] = formatted_bbox
    if formatted_circle is not None:
        params["circle"] = formatted_circle
    if publish_date_start is not None:
        params["publish_date_start"] = publish_date_start
    if publish_date_end is not None:
        params["publish_date_end"] = publish_date_end
    if taxonomy is not None:
        params["taxonomy"] = taxonomy
    if biomass is not None:
        params["biomass"] = str(biomass).lower()
    if coverage is not None:
        params["coverage"] = coverage

    data = _request_cesl("catalog", params=params, timeout=timeout)
    payload = _get_payload(data, _CATALOG_KEY_CANDIDATES)
    return payload.get("ids", [])


def get_cesl_metadata(
    sample_id: int,
    include_units: bool = False,
    timeout: int = 30,
    crosswalk: str = "speclib",
) -> Dict[str, Any]:
    """Retrieve metadata for a CESL sample.

    Args:
        sample_id (int): CESL sample ID.
        include_units (bool, optional): Whether to preserve CESL units metadata.
            Defaults to False.
        timeout (int, optional): Request timeout in seconds. Defaults to 30.
        crosswalk (str, optional): Metadata crosswalk type. Defaults to
            ``"speclib"``.

    Returns:
        dict[str, Any]: Normalized metadata for the sample.
    """

    data = _request_cesl(
        f"sample/{sample_id}/metadata",
        params={"format": "json", "crosswalk": crosswalk},
        timeout=timeout,
    )
    payload = _get_payload(data, _METADATA_KEY_CANDIDATES)
    return _normalize_cesl_metadata(payload, include_units=include_units)


def get_cesl_spectrum(
    sample_id: int, spectrum_key: Optional[str] = None, timeout: int = 30
) -> pd.DataFrame:
    """Retrieve the wavelength and spectrum values for a CESL sample.

    Args:
        sample_id (int): CESL sample ID.
        spectrum_key (str, optional): Name of the spectrum field to extract.
            Defaults to the first non-``wavelength`` field returned by the API.
        timeout (int, optional): Request timeout in seconds. Defaults to 30.

    Returns:
        pandas.DataFrame: A DataFrame with ``wavelength`` and spectrum columns.
    """

    import pandas as pd

    data = _request_cesl(
        f"sample/{sample_id}/data", params={"format": "json"}, timeout=timeout
    )
    payload = _get_payload(data, _DATA_KEY_CANDIDATES)

    if "wavelength" not in payload:
        raise KeyError("CESL spectrum response does not contain 'wavelength'.")

    if spectrum_key is None:
        spectrum_keys = [key for key in payload if key.lower() != "wavelength"]
        if not spectrum_keys:
            raise KeyError("CESL spectrum response does not contain any spectral data.")
        spectrum_key = spectrum_keys[0]
    else:
        matching_keys = {
            key.lower(): key for key in payload if key.lower() != "wavelength"
        }
        resolved_key = matching_keys.get(spectrum_key.lower())
        if resolved_key is None:
            raise KeyError(f"Could not find spectrum key '{spectrum_key}'.")
        spectrum_key = resolved_key

    spectrum = pd.DataFrame(
        {"wavelength": payload["wavelength"], spectrum_key: payload[spectrum_key]}
    )
    spectrum.attrs["sample_id"] = sample_id
    spectrum.attrs["spectrum_key"] = spectrum_key
    return spectrum


def plot_cesl_spectrum(
    sample_id: int,
    spectrum_key: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xlabel: str = "Wavelength (nm)",
    ylabel: Optional[str] = None,
    figsize: Optional[Sequence[float]] = None,
    x_range: Optional[Sequence[float]] = None,
    y_range: Optional[Sequence[float]] = None,
    timeout: int = 30,
    **kwargs: Any,
) -> plt.Axes:
    """Plot a CESL spectrum for a selected sample.

    Args:
        sample_id (int): CESL sample ID.
        spectrum_key (str, optional): Name of the spectrum field to plot.
        ax (matplotlib.axes.Axes, optional): Existing axes to plot on.
        title (str, optional): Plot title. Defaults to ``CESL Sample <id>``.
        xlabel (str, optional): X-axis label. Defaults to ``Wavelength (nm)``.
        ylabel (str, optional): Y-axis label. Defaults to the selected spectrum
            key.
        figsize (Sequence[float], optional): Figure size passed to
            ``matplotlib.pyplot.subplots`` when ``ax`` is not provided.
        x_range (Sequence[float], optional): Two-element x-axis range used to
            exclude wavelength outliers from the visible plot extent.
        y_range (Sequence[float], optional): Two-element y-axis range used to
            exclude reflectance outliers from the visible plot extent.
        timeout (int, optional): Request timeout in seconds. Defaults to 30.
        **kwargs: Additional keyword arguments passed to ``Axes.plot``.

    Returns:
        matplotlib.axes.Axes: The axes containing the plot.
    """

    import matplotlib.pyplot as plt

    spectrum = get_cesl_spectrum(
        sample_id=sample_id, spectrum_key=spectrum_key, timeout=timeout
    )
    spectrum_key = spectrum.attrs["spectrum_key"]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    label = kwargs.pop("label", f"Sample {sample_id}")
    ax.plot(spectrum["wavelength"], spectrum[spectrum_key], label=label, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or spectrum_key)
    ax.set_title(title or f"CESL Sample {sample_id}")

    if x_range is not None:
        ax.set_xlim(x_range)
    if y_range is not None:
        ax.set_ylim(y_range)

    if label is not None:
        ax.legend()

    return ax


def get_cesl_sites(
    sample_ids: Optional[Iterable[int]] = None,
    include_units: bool = False,
    max_workers: int = 8,
    timeout: int = 30,
    skip_missing_coordinates: bool = True,
    skip_errors: bool = False,
    **search_kwargs: Any,
) -> List[Dict[str, Any]]:
    """Retrieve CESL site metadata for a set of sample IDs.

    Args:
        sample_ids (Iterable[int], optional): CESL sample IDs to retrieve.
            Defaults to the full CESL catalog or a filtered catalog search.
        include_units (bool, optional): Whether to preserve units in metadata.
            Defaults to False.
        max_workers (int, optional): Number of worker threads used to fetch
            metadata. Defaults to 8.
        timeout (int, optional): Request timeout in seconds. Defaults to 30.
        skip_missing_coordinates (bool, optional): Whether to skip samples
            without coordinates. Defaults to True.
        skip_errors (bool, optional): Whether to skip samples that fail for any
            reason (e.g. non-JSON API responses). A warning is emitted for each
            skipped sample. Defaults to False.
        **search_kwargs: Additional arguments passed to :func:`search_cesl` when
            ``sample_ids`` is not provided.

    Returns:
        list[dict[str, Any]]: Normalized site records including coordinates.
    """

    if sample_ids is None:
        sample_ids = search_cesl(timeout=timeout, **search_kwargs)

    sample_ids = list(sample_ids)

    def fetch_site(sample_id: int) -> Dict[str, Any]:
        metadata = get_cesl_metadata(
            sample_id=sample_id, include_units=include_units, timeout=timeout
        )
        latitude = _extract_coordinate(metadata, _LATITUDE_KEYS, "latitude")
        longitude = _extract_coordinate(metadata, _LONGITUDE_KEYS, "longitude")

        return {
            **metadata,
            "sample_id": sample_id,
            "latitude": latitude,
            "longitude": longitude,
        }

    _BATCH_SIZE = 50
    records: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch_start in range(0, len(sample_ids), _BATCH_SIZE):
            batch = sample_ids[batch_start : batch_start + _BATCH_SIZE]
            futures = {
                executor.submit(fetch_site, sample_id): sample_id for sample_id in batch
            }
            for future in as_completed(futures):
                sample_id = futures[future]
                try:
                    records.append(future.result())
                except _MissingCoordinateError:
                    if not skip_missing_coordinates:
                        raise
                except Exception as exc:
                    if skip_errors:
                        warnings.warn(
                            f"Skipping sample {sample_id}: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    else:
                        raise RuntimeError(
                            f"Failed to retrieve CESL metadata for sample {sample_id}."
                        ) from exc

    records.sort(key=lambda record: record["sample_id"])
    return records


def cesl_to_gdf(
    sample_ids: Optional[Iterable[int]] = None,
    include_units: bool = False,
    max_workers: int = 8,
    timeout: int = 30,
    skip_missing_coordinates: bool = True,
    skip_errors: bool = False,
    **search_kwargs: Any,
) -> gpd.GeoDataFrame:
    """Convert CESL site metadata to a GeoPandas GeoDataFrame."""

    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "geopandas is required to convert CESL sites to a GeoDataFrame."
        ) from exc
    import pandas as pd

    records = get_cesl_sites(
        sample_ids=sample_ids,
        include_units=include_units,
        max_workers=max_workers,
        timeout=timeout,
        skip_missing_coordinates=skip_missing_coordinates,
        skip_errors=skip_errors,
        **search_kwargs,
    )

    if not records:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    frame = pd.DataFrame(records)
    return gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame["longitude"], frame["latitude"]),
        crs="EPSG:4326",
    )


def cesl_to_geojson(
    output: Optional[str] = None,
    sample_ids: Optional[Iterable[int]] = None,
    include_units: bool = False,
    max_workers: int = 8,
    timeout: int = 30,
    skip_missing_coordinates: bool = True,
    skip_errors: bool = False,
    **search_kwargs: Any,
) -> Dict[str, Any]:
    """Create a GeoJSON feature collection for CESL sites.

    Args:
        output (str, optional): Output GeoJSON path. If provided, the GeoJSON is
            written to disk.
        sample_ids (Iterable[int], optional): CESL sample IDs to export.
        include_units (bool, optional): Whether to preserve units in properties.
            Defaults to False.
        max_workers (int, optional): Number of worker threads used to fetch
            metadata. Defaults to 8.
        timeout (int, optional): Request timeout in seconds. Defaults to 30.
        skip_missing_coordinates (bool, optional): Whether to skip samples
            without coordinates. Defaults to True.
        skip_errors (bool, optional): Whether to skip samples that fail for any
            reason (e.g. non-JSON API responses). Defaults to False.
        **search_kwargs: Additional arguments passed to :func:`search_cesl` when
            ``sample_ids`` is not provided.

    Returns:
        dict[str, Any]: A GeoJSON FeatureCollection.
    """

    records = get_cesl_sites(
        sample_ids=sample_ids,
        include_units=include_units,
        max_workers=max_workers,
        timeout=timeout,
        skip_missing_coordinates=skip_missing_coordinates,
        skip_errors=skip_errors,
        **search_kwargs,
    )
    feature_collection = {
        "type": "FeatureCollection",
        "features": [_build_feature(record) for record in records],
    }

    if output is not None:
        output = os.path.abspath(output)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w", encoding="utf-8") as file:
            json.dump(feature_collection, file, indent=2)

    return feature_collection
