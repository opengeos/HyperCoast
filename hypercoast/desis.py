"""
This Module has the functions related to working with a DESIS dataset.
"""

import rioxarray
import numpy as np
import xarray as xr
import pandas as pd
from .common import convert_coords


def read_desis(filepath, wavelengths=None, method="nearest", **kwargs):
    """
    Reads DESIS data from a given file and returns an xarray Dataset.

    Args:
        filepath (str): Path to the file to read.
        wavelengths (array-like, optional): Specific wavelengths to select. If
            None, all wavelengths are selected.
        method (str, optional): Method to use for selection when wavelengths is not
            None. Defaults to "nearest".
        **kwargs: Additional keyword arguments to pass to the `sel` method when
            bands is not None.

    Returns:
        xr.Dataset: An xarray Dataset containing the DESIS data.
    """

    url = "https://github.com/opengeos/datasets/releases/download/hypercoast/desis_wavelengths.csv"
    df = pd.read_csv(url)
    dataset = xr.open_dataset(filepath)
    dataset = dataset.rename(
        {"band": "wavelength", "band_data": "reflectance"}
    ).transpose("y", "x", "wavelength")
    dataset["wavelength"] = df["wavelength"].tolist()

    if wavelengths is not None:
        dataset = dataset.sel(wavelength=wavelengths, method=method, **kwargs)

    dataset.attrs["crs"] = dataset.rio.crs.to_string()

    return dataset


def desis_to_image(dataset, wavelengths=None, method="nearest", output=None, **kwargs):
    """
    Converts an DESIS dataset to an image.

    Args:
        dataset (xarray.Dataset or str): The dataset containing the DESIS data
            or the file path to the dataset.
        wavelengths (array-like, optional): The specific wavelengths to select.
            If None, all wavelengths are selected. Defaults to None.
        method (str, optional): The method to use for data interpolation.
            Defaults to "nearest".
        output (str, optional): The file path where the image will be saved. If
            None, the image will be returned as a PIL Image object. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to
            `leafmap.array_to_image`.

    Returns:
        rasterio.Dataset or None: The image converted from the dataset. If
            `output` is provided, the image will be saved to the specified file
            and the function will return None.
    """
    from leafmap import array_to_image

    if isinstance(dataset, str):
        dataset = read_desis(dataset, method=method)

    if wavelengths is not None:
        dataset = dataset.sel(wavelength=wavelengths, method=method)

    return array_to_image(
        dataset["reflectance"], output=output, transpose=False, **kwargs
    )


def extract_desis(ds, lat, lon):
    """
    Extracts DESIS data from a given xarray Dataset.

    Args:
        ds (xarray.Dataset): The dataset containing the DESIS data.
        lat (float): The latitude of the point to extract.
        lon (float): The longitude of the point to extract.

    Returns:
        xarray.DataArray: The extracted data.
    """

    crs = ds.attrs["crs"]

    x, y = convert_coords([[lat, lon]], "epsg:4326", crs)[0]

    values = ds.sel(x=x, y=y, method="nearest")["reflectance"].values / 10000

    da = xr.DataArray(
        values, dims=["wavelength"], coords={"wavelength": ds.coords["wavelength"]}
    )

    return da


def filter_desis(dataset, lat, lon, return_plot=False, **kwargs):
    """
    Filters a DESIS dataset based on latitude and longitude.

    Args:
        dataset (xr.Dataset): The DESIS dataset to filter.
        lat (float or tuple): The latitude to filter by. If a tuple or list,
            it represents a range.
        lon (float or tuple): The longitude to filter by. If a tuple or
            list, it represents a range.

    Returns:
        xr.DataArray: The filtered DESIS data.
    """

    if isinstance(lat, list) or isinstance(lat, tuple):
        min_lat = min(lat)
        max_lat = max(lat)
    else:
        min_lat = lat
        max_lat = lat

    if isinstance(lon, list) or isinstance(lon, tuple):
        min_lon = min(lon)
        max_lon = max(lon)
    else:
        min_lon = lon
        max_lon = lon

    if min_lat == max_lat and min_lon == max_lon:
        coords = [[min_lat, min_lon]]
    else:
        coords = [[min_lat, min_lon], [max_lat, max_lon]]
    coords = convert_coords(coords, "epsg:4326", dataset.rio.crs.to_string())

    if len(coords) == 1:
        x, y = coords[0]
        da = dataset.sel(x=x, y=y, method="nearest")["reflectance"]
    else:
        x_min, y_min = coords[0]
        x_max, y_max = coords[1]
        print(x_min, y_min, x_max, y_max)
        da = dataset.sel(x=slice(x_min, x_max), y=slice(y_min, y_max))["reflectance"]

    if return_plot:
        rrs_stack = da.stack(
            {"pixel": ["latitude", "longitude"]},
            create_index=False,
        )
        rrs_stack.plot.line(hue="pixel", **kwargs)
    else:
        return da
