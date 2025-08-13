# SPDX-FileCopyrightText: 2025 Advait Dhamorikar <advaitdhamorikar[at]gmail.com>
#
# SPDX-License-Identifier: MIT

import numpy as np
import xarray as xr
from typing import List, Optional, Any, Union, Tuple
from affine import Affine
import h5py
import pyproj
from leafmap import array_to_image
from .common import convert_coords


def read_prisma(
    filepath: str,
    wavelengths: Optional[List[float]] = None,
    method: str = "nearest",
    **kwargs: Any,
) -> Tuple[xr.Dataset, str, Affine, Tuple[float, float]]:
    """
    Reads PRISMA hyperspectral Level-2 .he5 data and returns an xarray dataset with
    reflectance values, associated wavelengths, and geospatial metadata.

    This function loads both VNIR and SWIR spectral cubes, scales the raw integer
    values to physical reflectance units, merges them into a single spectral
    dataset, sorts wavelengths in ascending order, and assigns spatial coordinates
    derived from product corner coordinates. It also writes the CRS and affine
    transform to the dataset.

    Args:
        filepath (str): Path to the PRISMA .he5 file.
        wavelengths (List[float], optional): Wavelengths to select. If None,
            all available wavelengths are included.
        method (str, optional): Method used when selecting wavelengths (e.g.,
            "nearest"). Defaults to "nearest".
        **kwargs (Any): Additional keyword arguments passed to wavelength
            selection.

    Returns:
        Tuple[xr.Dataset, str, Affine, Tuple[float, float]]:
            - An xarray.Dataset containing reflectance data with coordinates.
            - The CRS as an EPSG string.
            - The affine transform describing spatial referencing.
            - The cell size as a tuple (x_res, y_res).
    """
    with h5py.File(filepath, "r") as f:
        vnir_cube_path = "HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube"
        swir_cube_path = "HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube"
        vnir_cube_data = f[vnir_cube_path][()]
        swir_cube_data = f[swir_cube_path][()]
        vnir_wavelengths = f.attrs["List_Cw_Vnir"][()]
        swir_wavelengths = f.attrs["List_Cw_Swir"][()]
        l2_scale_vnir_min = f.attrs["L2ScaleVnirMin"][()]
        l2_scale_vnir_max = f.attrs["L2ScaleVnirMax"][()]
        l2_scale_swir_min = f.attrs["L2ScaleSwirMin"][()]
        l2_scale_swir_max = f.attrs["L2ScaleSwirMax"][()]
        epsg_code = f.attrs["Epsg_Code"][()]
        ul_easting = f.attrs["Product_ULcorner_easting"][()]
        ul_northing = f.attrs["Product_ULcorner_northing"][()]
        lr_easting = f.attrs["Product_LRcorner_easting"][()]
        lr_northing = f.attrs["Product_LRcorner_northing"][()]

    fill_value = -9999
    max_data_value = 65535

    vnir_cube_data = l2_scale_vnir_min + (
        vnir_cube_data.astype(np.float32) / max_data_value
    ) * (l2_scale_vnir_max - l2_scale_vnir_min)
    swir_cube_data = l2_scale_swir_min + (
        swir_cube_data.astype(np.float32) / max_data_value
    ) * (l2_scale_swir_max - l2_scale_swir_min)

    vnir_cube_data[vnir_cube_data == fill_value] = np.nan
    swir_cube_data[swir_cube_data == fill_value] = np.nan

    combined_reflectance = np.concatenate((vnir_cube_data, swir_cube_data), axis=1)
    combined_wavelengths = np.concatenate((vnir_wavelengths, swir_wavelengths))

    valid_indices = combined_wavelengths > 0
    combined_wavelengths = combined_wavelengths[valid_indices]
    combined_reflectance = combined_reflectance[:, valid_indices, :]
    sort_indices = np.argsort(combined_wavelengths)
    combined_wavelengths = combined_wavelengths[sort_indices]
    combined_reflectance = combined_reflectance[:, sort_indices, :]

    rows = combined_reflectance.shape[0]
    cols = combined_reflectance.shape[2]
    x_res = (lr_easting - ul_easting) / cols
    y_res = (lr_northing - ul_northing) / rows

    transform = Affine.translation(ul_easting, ul_northing) * Affine.scale(x_res, y_res)
    x_coords = np.array([transform * (i, 0) for i in range(cols)])[:, 0]
    y_coords = np.array([transform * (0, j) for j in range(rows)])[:, 1]

    ds = xr.Dataset(
        data_vars=dict(
            reflectance=(
                ["y", "wavelength", "x"],
                combined_reflectance,
                dict(
                    units="unitless",
                    _FillValue=np.nan,
                    standard_name="reflectance",
                    long_name="Combined atmospherically corrected surface reflectance",
                ),
            ),
        ),
        coords=dict(
            wavelength=(
                ["wavelength"],
                combined_wavelengths,
                dict(long_name="center wavelength", units="nm"),
            ),
            y=(["y"], y_coords, dict(units="m")),
            x=(["x"], x_coords, dict(units="m")),
        ),
    )

    ds["reflectance"] = ds.reflectance.transpose("y", "x", "wavelength")

    transform = Affine.translation(ul_easting, ul_northing) * Affine.scale(x_res, y_res)
    crs = f"EPSG:{epsg_code}"
    if crs is None:
        raise ValueError(
            "Dataset has no CRS. Please ensure read_prisma writes CRS before returning."
        )
    ds.rio.write_crs(crs, inplace=True)
    ds.rio.write_transform(transform, inplace=True)

    global_atts = ds.attrs
    global_atts["Conventions"] = "CF-1.6"
    ds.attrs = dict(
        units="unitless",
        _FillValue=-9999,
        grid_mapping="crs",
        standard_name="reflectance",
        long_name="atmospherically corrected surface reflectance",
        crs=ds.rio.crs.to_string(),
    )
    ds.attrs.update(global_atts)

    return ds


def prisma_to_image(
    dataset: Union[xr.Dataset, str],
    wavelengths: Optional[np.ndarray] = None,
    method: str = "nearest",
    output: Optional[str] = None,
    **kwargs: Any,
):
    """
    Converts a PRISMA hyperspectral dataset to a georeferenced image.

    If given a file path, the dataset is read using `read_prisma` and converted
    into a spatially referenced raster image. Optionally, a subset of wavelengths
    can be selected, and values are scaled or clipped before writing to an image.

    Args:
        dataset (Union[xr.Dataset, str]): The PRISMA dataset or the path to the
            dataset file (.he5).
        wavelengths (np.ndarray, optional): Wavelengths to select from the dataset.
            If None, all wavelengths are included. Defaults to None.
        method (str, optional): Method to use for wavelength selection (e.g.,
            "nearest"). Defaults to "nearest".
        output (str, optional): File path to save the output raster. If None, a
            raster object will be returned instead of being saved. Defaults to None.
        **kwargs (Any): Additional arguments passed to `leafmap.array_to_image`,
            including data type (`dtype`), compression, and colormap.

    Returns:
        Optional[rasterio.Dataset]: If `output` is None, returns the in-memory
        raster object created from the dataset. If `output` is provided, saves
        the raster to disk and returns the output file path.
    """
    crs = dataset.rio.crs
    transform = dataset.rio.transform()

    if isinstance(dataset, str):
        dataset = read_prisma(dataset)

    if wavelengths is not None:
        dataset = dataset.sel(wavelength=wavelengths, method=method)

    resolution = (transform.a, transform.e)

    output_array = dataset["reflectance"].values
    if not np.any(np.isfinite(output_array)):
        print("Warning: All reflectance values are NaN. Output image will be blank.")
        return None

    output_dtype = kwargs.get("dtype", np.float32)

    vmin, vmax = np.nanpercentile(output_array, (2, 98))
    output_array = np.clip(output_array, vmin, vmax)

    if output_dtype == np.uint8:
        output_array = ((output_array - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        kwargs["dtype"] = output_dtype

    result = array_to_image(
        output_array,
        output=output,
        transpose=False,
        crs=crs,
        transform=transform,
        cellsize=resolution,
        **kwargs,
    )

    return output if output is not None else result


def extract_prisma(
    dataset: xr.Dataset,
    lat: float,
    lon: float,
    offset: float = 15.0,
) -> xr.DataArray:
    """
    Extracts an averaged reflectance spectrum from a PRISMA hyperspectral dataset.

    A square spatial window is centered at the specified latitude and longitude,
    and the reflectance values within that window are averaged across the spatial
    dimensions to produce a single spectrum.

    Args:
        dataset (xarray.Dataset): The PRISMA dataset containing reflectance data,
            with valid CRS information.
        lat (float): Latitude of the center point.
        lon (float): Longitude of the center points.
        offset (float, optional): Half-size of the square window for extraction,
            expressed in the dataset's projected coordinate units (e.g., meters).
            Defaults to 15.0.

    Returns:
        xarray.DataArray: A 1D array containing the averaged reflectance values
        across wavelengths. If no matching pixels are found, returns NaN values.
    """
    if dataset.rio.crs is None:
        raise ValueError("Dataset CRS not set. Please provide dataset with CRS info.")

    crs = dataset.rio.crs.to_string()

    # Convert lat/lon to projected coords
    x_proj, y_proj = convert_coords([(lat, lon)], "epsg:4326", crs)[0]

    da = dataset["reflectance"]
    x_con = (da["x"] > x_proj - offset) & (da["x"] < x_proj + offset)
    y_con = (da["y"] > y_proj - offset) & (da["y"] < y_proj + offset)

    try:
        data = da.where(x_con & y_con, drop=True)
        data = data.mean(dim=["x", "y"], skipna=True)
    except ValueError:
        # No matching pixels
        data = np.full(da.sizes["wavelength"], np.nan)

    return xr.DataArray(
        data,
        dims=["wavelength"],
        coords={"wavelength": dataset.coords["wavelength"]},
    )
