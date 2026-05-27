# SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Coastal workflow presets for hyperspectral datasets."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class WorkflowPreset:
    """Describe a coastal workflow preset.

    Args:
        name: Workflow identifier.
        description: Human-readable workflow description.
        kind: Workflow algorithm kind.
        wavelengths: Named wavelength parameters in nanometers.
        variable: Optional default data variable.
    """

    name: str
    description: str
    kind: str
    wavelengths: Dict[str, float]
    variable: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """Return serializable workflow metadata."""
        return asdict(self)


WORKFLOW_PRESETS: Dict[str, WorkflowPreset] = {
    "ndwi": WorkflowPreset(
        name="ndwi",
        description="Water mask index using green and near-infrared bands.",
        kind="normalized_difference",
        wavelengths={"a": 560.0, "b": 860.0},
    ),
    "chlorophyll": WorkflowPreset(
        name="chlorophyll",
        description="Simple green/red ratio proxy for chlorophyll-a screening.",
        kind="ratio",
        wavelengths={"a": 560.0, "b": 665.0},
    ),
    "turbidity": WorkflowPreset(
        name="turbidity",
        description="Red/NIR ratio proxy for suspended sediment or turbidity.",
        kind="ratio",
        wavelengths={"a": 665.0, "b": 860.0},
    ),
    "cdom": WorkflowPreset(
        name="cdom",
        description="Blue/green ratio proxy for colored dissolved organic matter.",
        kind="ratio",
        wavelengths={"a": 412.0, "b": 555.0},
    ),
    "cyanobacteria": WorkflowPreset(
        name="cyanobacteria",
        description="Red-edge contrast proxy for cyanobacteria screening.",
        kind="normalized_difference",
        wavelengths={"a": 709.0, "b": 665.0},
    ),
    "anomaly": WorkflowPreset(
        name="anomaly",
        description="Maximum absolute spectral z-score anomaly.",
        kind="spectral_anomaly",
        wavelengths={},
    ),
}


def list_workflows() -> Dict[str, Dict[str, Any]]:
    """Return serializable workflow preset metadata.

    Returns:
        dict: Mapping of workflow names to metadata.
    """
    return {name: preset.as_dict() for name, preset in WORKFLOW_PRESETS.items()}


def apply_workflow(
    data: xr.Dataset | xr.DataArray,
    workflow: str,
    variable: Optional[str] = None,
    **kwargs: Any,
) -> xr.DataArray:
    """Apply a named coastal workflow to hyperspectral data.

    Args:
        data: Input dataset or data array.
        workflow: Workflow preset name.
        variable: Optional data variable for xarray datasets.
        **kwargs: Optional workflow overrides.

    Returns:
        xr.DataArray: Workflow output.
    """
    key = workflow.lower().strip()
    if key not in WORKFLOW_PRESETS:
        available = ", ".join(sorted(WORKFLOW_PRESETS))
        raise KeyError(
            f"Unknown workflow '{workflow}'. Available workflows: {available}"
        )

    preset = WORKFLOW_PRESETS[key]
    da = _as_data_array(data, variable or preset.variable)
    wavelengths = dict(preset.wavelengths)
    wavelengths.update(kwargs.pop("wavelengths", {}) or {})

    if preset.kind == "normalized_difference":
        return normalized_difference(da, wavelengths["a"], wavelengths["b"])
    if preset.kind == "ratio":
        return band_ratio(da, wavelengths["a"], wavelengths["b"])
    if preset.kind == "spectral_anomaly":
        return spectral_anomaly(da)
    raise ValueError(f"Unsupported workflow kind: {preset.kind}")


def normalized_difference(
    data: xr.Dataset | xr.DataArray,
    wavelength_a: float,
    wavelength_b: float,
    variable: Optional[str] = None,
) -> xr.DataArray:
    """Calculate a normalized difference between two wavelengths.

    Args:
        data: Input dataset or data array.
        wavelength_a: First wavelength in nanometers.
        wavelength_b: Second wavelength in nanometers.
        variable: Optional data variable for xarray datasets.

    Returns:
        xr.DataArray: Normalized difference output.
    """
    da = _as_data_array(data, variable)
    a = _select_wavelength(da, wavelength_a)
    b = _select_wavelength(da, wavelength_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (a - b) / (a + b)
    result.name = f"nd_{int(wavelength_a)}_{int(wavelength_b)}"
    return result


def band_ratio(
    data: xr.Dataset | xr.DataArray,
    numerator: float,
    denominator: float,
    variable: Optional[str] = None,
) -> xr.DataArray:
    """Calculate a ratio between two wavelengths.

    Args:
        data: Input dataset or data array.
        numerator: Numerator wavelength in nanometers.
        denominator: Denominator wavelength in nanometers.
        variable: Optional data variable for xarray datasets.

    Returns:
        xr.DataArray: Ratio output.
    """
    da = _as_data_array(data, variable)
    a = _select_wavelength(da, numerator)
    b = _select_wavelength(da, denominator)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = a / b
    result.name = f"ratio_{int(numerator)}_{int(denominator)}"
    return result


def spectral_anomaly(
    data: xr.Dataset | xr.DataArray,
    variable: Optional[str] = None,
) -> xr.DataArray:
    """Calculate the maximum absolute spectral z-score per pixel.

    Args:
        data: Input dataset or data array.
        variable: Optional data variable for xarray datasets.

    Returns:
        xr.DataArray: Maximum absolute spectral anomaly score.
    """
    da = _as_data_array(data, variable)
    dim = _wavelength_dim(da)
    mean = da.mean(dim=dim, skipna=True)
    std = da.std(dim=dim, skipna=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = (da - mean) / std
    result = np.abs(zscore).max(dim=dim, skipna=True)
    result.name = "spectral_anomaly"
    return result


def _as_data_array(
    data: xr.Dataset | xr.DataArray,
    variable: Optional[str] = None,
) -> xr.DataArray:
    """Return the selected data array."""
    if isinstance(data, xr.DataArray):
        return data
    if variable is not None:
        if variable not in data.data_vars:
            raise KeyError(f"Variable not found: {variable}")
        return data[variable]
    for candidate in ("reflectance", "Rrs", "toa_radiance", "surface_reflectance"):
        if candidate in data.data_vars:
            return data[candidate]
    for name, da in data.data_vars.items():
        if _has_wavelength_dim(da):
            return da
    raise ValueError("No wavelength-bearing data variable found.")


def _select_wavelength(da: xr.DataArray, wavelength: float) -> xr.DataArray:
    """Select the nearest wavelength from a data array."""
    dim = _wavelength_dim(da)
    return da.sel({dim: wavelength}, method="nearest")


def _has_wavelength_dim(da: xr.DataArray) -> bool:
    """Return whether a data array has a spectral dimension."""
    return any(dim in da.dims for dim in ("wavelength", "wavelengths", "band"))


def _wavelength_dim(da: xr.DataArray) -> str:
    """Return the spectral dimension name."""
    for dim in ("wavelength", "wavelengths", "band"):
        if dim in da.dims:
            return dim
    raise ValueError("Data array has no wavelength dimension.")


__all__ = [
    "WORKFLOW_PRESETS",
    "WorkflowPreset",
    "apply_workflow",
    "band_ratio",
    "list_workflows",
    "normalized_difference",
    "spectral_anomaly",
]
