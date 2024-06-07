"""This module contains functions to read and process NASA AVIRIS hyperspectral data.
More info about the data can be found at https://aviris.jpl.nasa.gov.
The source code is adapted from https://bit.ly/4bRCgqs. Credit goes to the
original authors.
"""

import rioxarray
import numpy as np
import xarray as xr
from typing import List, Union, Dict, Optional, Tuple, Any
from .common import convert_coords


def read_aviris(
    filepath: str,
    wavelengths: Optional[List[float]] = None,
    method: str = "nearest",
    **kwargs: Any,
) -> xr.Dataset:
    """
    Reads NASA AVIRIS hyperspectral data and returns an xarray dataset.

    Args:
        filepath (str): The path to the AVIRIS data.
        wavelengths (List[float], optional): The wavelengths to select. If None,
            all wavelengths are selected. Defaults to None.
        method (str, optional): The method to use for selection. Defaults to
            "nearest".
        **kwargs (Any): Additional arguments to pass to the selection method.

    Returns:
        xr.Dataset: The dataset containing the reflectance data.
    """

    if filepath.endswith(".hdr"):
        filepath = filepath.replace(".hdr", "")

    ds = xr.open_dataset(filepath, engine="rasterio")

    wavelength = ds["wavelength"].values.tolist()
    wavelength = [round(num, 2) for num in wavelength]

    cols = ds.x.size
    rows = ds.y.size

    geo_transform = ds.rio.transform()
    geo_transform = list(geo_transform)[:6]

    # get the raster geotransform as its component parts
    xres, xrot, xmin, yrot, yres, ymax = geo_transform

    # generate coordinate arrays
    xarr = np.array([xmin + i * xres for i in range(0, cols)])
    yarr = np.array([ymax + i * yres for i in range(0, rows)])

    ds["y"] = xr.DataArray(
        data=yarr,
        dims=("y"),
        name="y",
        attrs=dict(
            units="m",
            standard_name="projection_y_coordinate",
            long_name="y coordinate of projection",
        ),
    )

    ds["x"] = xr.DataArray(
        data=xarr,
        dims=("x"),
        name="x",
        attrs=dict(
            units="m",
            standard_name="projection_x_coordinate",
            long_name="x coordinate of projection",
        ),
    )

    global_atts = ds.attrs
    global_atts["Conventions"] = "CF-1.6"
    ds.attrs = dict(
        units="unitless",
        _FillValue=-9999,
        grid_mapping="crs",
        standard_name="reflectance",
        long_name="atmospherically corrected surface reflectance",
    )
    ds.attrs.update(global_atts)

    ds = ds.transpose("y", "x", "band")
    ds = ds.drop_vars(["wavelength", "xc", "yc"])
    ds = ds.rename({"band": "wavelength", "band_data": "reflectance"})
    ds.coords["wavelength"] = wavelength
    ds.attrs["crs"] = ds.rio.crs.to_string()

    if wavelengths is not None:
        ds = ds.sel(wavelength=wavelengths, method=method, **kwargs)
    return ds


def aviris_to_image(
    dataset: Union[xr.Dataset, str],
    wavelengths: Optional[np.ndarray] = None,
    method: str = "nearest",
    output: Optional[str] = None,
    **kwargs: Any,
):
    """
    Converts an AVIRIS dataset to an image.

    Args:
        dataset (Union[xr.Dataset, str]): The dataset containing the AVIRIS data
            or the file path to the dataset.
        wavelengths (np.ndarray, optional): The specific wavelengths to select. If None, all
            wavelengths are selected. Defaults to None.
        method (str, optional): The method to use for data interpolation.
            Defaults to "nearest".
        output (str, optional): The file path where the image will be saved. If
            None, the image will be returned as a PIL Image object. Defaults to None.
        **kwargs (Any): Additional keyword arguments to be passed to
            `leafmap.array_to_image`.

    Returns:
        Optional[rasterio.Dataset]: The image converted from the dataset. If
            `output` is provided, the image will be saved to the specified file
            and the function will return None.
    """
    from leafmap import array_to_image

    if isinstance(dataset, str):
        dataset = read_aviris(dataset, method=method)

    if wavelengths is not None:
        dataset = dataset.sel(wavelength=wavelengths, method=method)

    return array_to_image(
        dataset["reflectance"],
        output=output,
        transpose=False,
        dtype=np.float32,
        **kwargs,
    )


def extract_aviris(ds, lat, lon):
    """
    Extracts NEON AOP data from a given xarray Dataset.

    Args:
        ds (xarray.Dataset): The dataset containing the NEON AOP data.
        lat (float): The latitude of the point to extract.
        lon (float): The longitude of the point to extract.

    Returns:
        xarray.DataArray: The extracted data.
    """

    crs = ds.attrs["crs"]

    x, y = convert_coords([[lat, lon]], "epsg:4326", crs)[0]

    values = ds.sel(x=x, y=y, method="nearest")["reflectance"].values

    da = xr.DataArray(
        values, dims=["wavelength"], coords={"wavelength": ds.coords["wavelength"]}
    )

    return da
