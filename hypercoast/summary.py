# SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Dataset summary and batch extraction helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import xarray as xr

from .registry import get_sensor, read_sensor

SPECTRAL_DIMS = ("wavelength", "wavelengths", "band", "bands")


@dataclass
class DatasetSummary:
    """Describe a local hyperspectral dataset.

    Args:
        path: Dataset path.
        exists: Whether the path exists.
        sensor: Sensor name used to read the dataset.
        variables: Data variable names.
        selected_variable: Selected data variable.
        crs: Dataset CRS when discoverable.
        bounds: Dataset bounds when discoverable.
        wavelength_count: Number of spectral coordinates.
        wavelength_min: Minimum wavelength or band coordinate.
        wavelength_max: Maximum wavelength or band coordinate.
        dimensions: Dataset dimensions.
        default_rgb: Default RGB wavelengths from the sensor registry.
        warnings: Non-fatal summary warnings.
    """

    path: str
    exists: bool
    sensor: Optional[str] = None
    variables: list[str] = field(default_factory=list)
    selected_variable: Optional[str] = None
    crs: Optional[str] = None
    bounds: Optional[tuple[float, float, float, float]] = None
    wavelength_count: int = 0
    wavelength_min: Optional[float] = None
    wavelength_max: Optional[float] = None
    dimensions: dict[str, int] = field(default_factory=dict)
    default_rgb: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary dictionary.

        Returns:
            dict: Summary fields.
        """
        return asdict(self)


def summarize_dataset(
    path: str | Path,
    sensor: Optional[str] = None,
    variable: Optional[str] = None,
) -> DatasetSummary:
    """Summarize a local dataset.

    Args:
        path: Local dataset path.
        sensor: Optional sensor name or alias.
        variable: Optional selected data variable.

    Returns:
        DatasetSummary: Dataset metadata and warnings.
    """
    dataset_path = Path(path)
    summary = DatasetSummary(path=str(dataset_path), exists=dataset_path.exists())
    if not dataset_path.exists():
        summary.warnings.append(f"File not found: {dataset_path}")
        return summary

    handler = None
    if sensor:
        handler = get_sensor(sensor)
        summary.sensor = handler.name
        summary.default_rgb = [float(value) for value in handler.default_rgb]

    try:
        ds = (
            read_sensor(sensor, dataset_path)
            if sensor
            else xr.open_dataset(dataset_path)
        )
    except Exception as exc:
        summary.warnings.append(f"Could not open dataset: {exc}")
        return summary

    try:
        summary.variables = list(ds.data_vars)
        summary.dimensions = {name: int(size) for name, size in ds.sizes.items()}
        summary.selected_variable = _select_variable(ds, variable)
        summary.crs = _dataset_crs(ds)
        summary.bounds = _dataset_bounds(ds)
        wavelengths = _wavelength_values(ds, summary.selected_variable)
        if wavelengths is not None and wavelengths.size > 0:
            finite = wavelengths[np.isfinite(wavelengths)]
            summary.wavelength_count = int(wavelengths.size)
            if finite.size > 0:
                summary.wavelength_min = float(np.nanmin(finite))
                summary.wavelength_max = float(np.nanmax(finite))
    finally:
        close = getattr(ds, "close", None)
        if callable(close):
            close()

    if variable and summary.selected_variable != variable:
        summary.warnings.append(f"Variable not found or not selectable: {variable}")
    return summary


def subset_dataset(
    path: str | Path,
    output: str | Path,
    bbox: tuple[float, float, float, float],
    sensor: Optional[str] = None,
    variable: Optional[str] = None,
) -> str:
    """Subset a rectilinear local dataset by bounding box and write NetCDF.

    Args:
        path: Input dataset path.
        output: Output NetCDF path.
        bbox: Bounding box as ``(xmin, ymin, xmax, ymax)``.
        sensor: Optional sensor name or alias.
        variable: Optional data variable to keep.

    Returns:
        str: Output path.
    """
    ds = read_sensor(sensor, path) if sensor else xr.open_dataset(path)
    try:
        selected = _select_variable(ds, variable)
        if variable and selected:
            ds = ds[[selected]]
        xmin, ymin, xmax, ymax = bbox
        ds = _subset_by_bbox(ds, xmin, ymin, xmax, ymax)
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(output_path)
        return str(output_path)
    finally:
        close = getattr(ds, "close", None)
        if callable(close):
            close()


def extract_spectra_to_csv(
    sensor: str,
    path: str | Path,
    points_csv: str | Path,
    output: str | Path,
    x_column: str = "x",
    y_column: str = "y",
    crs: str = "EPSG:4326",
) -> str:
    """Extract spectra for CSV point coordinates.

    Args:
        sensor: Sensor name or alias.
        path: Input dataset path.
        points_csv: CSV file containing point coordinates.
        output: Output long-form CSV path.
        x_column: X or longitude column.
        y_column: Y or latitude column.
        crs: Coordinate reference system for point coordinates.

    Returns:
        str: Output CSV path.
    """
    from .registry import extract_sensor

    points = pd.read_csv(points_csv)
    if x_column not in points.columns or y_column not in points.columns:
        raise ValueError(f"Point CSV must contain '{x_column}' and '{y_column}'.")

    dataset = read_sensor(sensor, path)
    rows: list[dict[str, Any]] = []
    try:
        for feature_id, row in points.iterrows():
            lon, lat = _to_lon_lat(float(row[x_column]), float(row[y_column]), crs)
            spectrum = extract_sensor(sensor, dataset, lat=lat, lon=lon)
            wavelengths, values = _spectrum_values(spectrum)
            for wavelength, value in zip(wavelengths, values):
                rows.append(
                    {
                        "feature_id": feature_id,
                        "x": row[x_column],
                        "y": row[y_column],
                        "crs": crs,
                        "wavelength": wavelength,
                        "value": value,
                        "layer": str(path),
                        "variable": getattr(spectrum, "name", None),
                    }
                )
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return str(output_path)


def _select_variable(ds: xr.Dataset, variable: Optional[str]) -> Optional[str]:
    """Return a selected or inferred data variable name."""
    if variable and variable in ds.data_vars:
        return variable
    for candidate in ("reflectance", "Rrs", "toa_radiance", "surface_reflectance"):
        if candidate in ds.data_vars:
            return candidate
    return next(iter(ds.data_vars), None)


def _wavelength_values(
    ds: xr.Dataset,
    variable: Optional[str],
) -> Optional[np.ndarray]:
    """Return spectral coordinate values when available."""
    candidates: list[str] = []
    if variable and variable in ds:
        candidates.extend(dim for dim in ds[variable].dims if dim in SPECTRAL_DIMS)
    candidates.extend(name for name in SPECTRAL_DIMS if name in ds.coords)
    for name in candidates:
        if name in ds.coords:
            return np.asarray(ds.coords[name].values, dtype=float)
    return None


def _dataset_crs(ds: xr.Dataset) -> Optional[str]:
    """Return dataset CRS when discoverable."""
    try:
        rio = getattr(ds, "rio", None)
        crs = getattr(rio, "crs", None)
        if crs:
            return str(crs)
    except Exception:
        pass
    crs = ds.attrs.get("crs") or ds.attrs.get("spatial_ref")
    return str(crs) if crs else None


def _dataset_bounds(ds: xr.Dataset) -> Optional[tuple[float, float, float, float]]:
    """Return dataset bounds from common coordinate names."""
    try:
        rio = getattr(ds, "rio", None)
        bounds = rio.bounds() if rio is not None else None
        if bounds:
            return tuple(float(value) for value in bounds)
    except Exception:
        pass
    x_coord = _first_coord(ds, ("x", "longitude", "lon"))
    y_coord = _first_coord(ds, ("y", "latitude", "lat"))
    if x_coord is None or y_coord is None:
        return None
    x_values = np.asarray(x_coord.values, dtype=float)
    y_values = np.asarray(y_coord.values, dtype=float)
    if x_values.size == 0 or y_values.size == 0:
        return None
    return (
        float(np.nanmin(x_values)),
        float(np.nanmin(y_values)),
        float(np.nanmax(x_values)),
        float(np.nanmax(y_values)),
    )


def _first_coord(ds: xr.Dataset, names: tuple[str, ...]) -> Optional[xr.DataArray]:
    """Return the first present coordinate."""
    for name in names:
        if name in ds.coords:
            return ds.coords[name]
    return None


def _subset_by_bbox(
    ds: xr.Dataset,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> xr.Dataset:
    """Subset a dataset using common rectilinear coordinates."""
    x_name = (
        "x" if "x" in ds.coords else "longitude" if "longitude" in ds.coords else None
    )
    y_name = (
        "y" if "y" in ds.coords else "latitude" if "latitude" in ds.coords else None
    )
    if x_name is None or y_name is None:
        raise ValueError("Dataset must contain x/y or longitude/latitude coordinates.")
    return ds.sel(
        {
            x_name: _coord_slice(ds.coords[x_name], xmin, xmax),
            y_name: _coord_slice(ds.coords[y_name], ymin, ymax),
        }
    )


def _coord_slice(coord: xr.DataArray, low: float, high: float) -> slice:
    """Return an orientation-aware coordinate slice."""
    values = np.asarray(coord.values)
    if values.size > 1 and values[0] > values[-1]:
        return slice(high, low)
    return slice(low, high)


def _to_lon_lat(x: float, y: float, crs: str) -> tuple[float, float]:
    """Return coordinates as lon/lat."""
    if crs.upper() in {"EPSG:4326", "CRS84", "OGC:CRS84"}:
        return x, y
    try:
        from pyproj import Transformer

        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)
        return float(lon), float(lat)
    except Exception as exc:
        raise ValueError(f"Could not transform points from {crs}: {exc}") from exc


def _spectrum_values(spectrum: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return wavelength and value arrays from an extracted spectrum."""
    values = np.asarray(getattr(spectrum, "values", spectrum), dtype=float).ravel()
    coords = getattr(spectrum, "coords", {})
    for name in SPECTRAL_DIMS:
        if name in coords:
            return np.asarray(coords[name].values, dtype=float).ravel(), values
    return np.arange(values.size, dtype=float), values


__all__ = [
    "DatasetSummary",
    "extract_spectra_to_csv",
    "subset_dataset",
    "summarize_dataset",
]
