# SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Cloud-native helpers for hyperspectral datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import xarray as xr


def suggest_chunks(
    dataset: xr.Dataset | xr.DataArray,
    target_pixels: int = 2_000_000,
) -> Dict[str, int]:
    """Suggest Dask chunks for a hyperspectral dataset.

    Args:
        dataset: Input dataset or data array.
        target_pixels: Approximate target pixels per spatial chunk.

    Returns:
        dict: Suggested chunk sizes by dimension.
    """
    dims = dict(dataset.sizes)
    chunks: Dict[str, int] = {}
    for dim, size in dims.items():
        lower = dim.lower()
        if lower in ("wavelength", "wavelengths", "band"):
            chunks[dim] = min(size, 32)
        elif lower in ("x", "y", "latitude", "longitude"):
            chunks[dim] = max(1, min(size, int(target_pixels**0.5)))
        else:
            chunks[dim] = size
    return chunks


def open_cloud_dataset(
    path: str,
    chunks: Optional[Dict[str, int]] = None,
    **kwargs: Any,
):
    """Open a cloud-friendly xarray dataset.

    Args:
        path: Dataset path or URL.
        chunks: Optional Dask chunks. If None, chunks are inferred lazily by xarray.
        **kwargs: Additional ``xarray.open_dataset`` keyword arguments.

    Returns:
        xr.Dataset: Opened dataset.
    """
    if chunks is not None:
        kwargs["chunks"] = chunks
    return xr.open_dataset(path, **kwargs)


def to_zarr(
    dataset: xr.Dataset | xr.DataArray,
    output: str | Path,
    chunks: Optional[Dict[str, int]] = None,
    **kwargs: Any,
) -> str:
    """Write a dataset to Zarr.

    Args:
        dataset: Input dataset or data array.
        output: Output Zarr path.
        chunks: Optional chunks to apply before writing.
        **kwargs: Additional ``to_zarr`` keyword arguments.

    Returns:
        str: Output path.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(dataset, xr.DataArray):
        obj = dataset.to_dataset(name=dataset.name or "data")
    else:
        obj = dataset
    if chunks:
        obj = obj.chunk(chunks)
    obj.to_zarr(output, **kwargs)
    return str(output)


def to_cog(
    data: xr.Dataset | xr.DataArray,
    output: str | Path,
    variable: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Write a data array to a Cloud Optimized GeoTIFF.

    Args:
        data: Input dataset or data array.
        output: Output COG path.
        variable: Optional dataset variable.
        **kwargs: Additional ``rioxarray.to_raster`` keyword arguments.

    Returns:
        str: Output path.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, xr.Dataset):
        if variable is None:
            variable = next(iter(data.data_vars))
        arr = data[variable]
    else:
        arr = data
    arr.rio.to_raster(output, driver="COG", **kwargs)
    return str(output)


__all__ = ["open_cloud_dataset", "suggest_chunks", "to_cog", "to_zarr"]
