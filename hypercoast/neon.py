"""This module contains functions to read and process NEON AOP hyperspectral data.
More info about the data can be found at https://bit.ly/3Rfszdc.
The source code is adapted from https://bit.ly/3KwyZkn. Credit goes to the
original authors.
"""

import h5py
import rioxarray
import numpy as np
import xarray as xr
from typing import List, Union, Dict, Optional, Tuple, Any
from .common import convert_coords


def list_neon_datasets(filepath: str, print_node: bool = False) -> None:
    """
    Lists all the datasets in an HDF5 file.

    Args:
        filepath (str): The path to the HDF5 file.
        print_node (bool, optional): If True, prints the node object of each dataset.
            If False, prints the name of each dataset. Defaults to False.
    """

    f = h5py.File(filepath, "r")

    if print_node:

        def list_dataset(_, node):
            if isinstance(node, h5py.Dataset):
                print(node)

    else:

        def list_dataset(name, node):
            if isinstance(node, h5py.Dataset):
                print(name)

    f.visititems(list_dataset)


def read_neon(
    filepath: str,
    wavelengths: Optional[List[float]] = None,
    method: str = "nearest",
    **kwargs: Any,
) -> xr.Dataset:
    """
    Reads NEON AOP hyperspectral hdf5 files and returns an xarray dataset.

    Args:
        filepath (str): The path to the hdf5 file.
        wavelengths (List[float], optional): The wavelengths to select. If None,
            all wavelengths are selected. Defaults to None.
        method (str, optional): The method to use for selection. Defaults to
            "nearest".
        **kwargs (Any): Additional arguments to pass to the selection method.

    Returns:
        xr.Dataset: The dataset containing the reflectance data.
    """
    f = h5py.File(filepath, "r")

    serc_refl = f["SERC"]["Reflectance"]
    wavelengths = serc_refl["Metadata"]["Spectral_Data"]["Wavelength"][()].tolist()
    wavelengths = [round(num, 2) for num in wavelengths]

    epsg_code = serc_refl["Metadata"]["Coordinate_System"]["EPSG Code"][()]
    epsg_code_number = int(epsg_code.decode("utf-8"))

    serc_mapInfo = serc_refl["Metadata"]["Coordinate_System"]["Map_Info"]
    mapInfo_string = serc_mapInfo[()].decode("utf-8")  # read in as string
    mapInfo_split = mapInfo_string.split(",")

    res = float(mapInfo_split[5]), float(mapInfo_split[6])

    serc_reflArray = serc_refl["Reflectance_Data"]
    refl_shape = serc_reflArray.shape

    # Extract the upper left-hand corner coordinates from mapInfo
    xMin = float(mapInfo_split[3])
    yMax = float(mapInfo_split[4])

    # Calculate the xMax and yMin values from the dimensions
    xMax = xMin + (
        refl_shape[1] * res[0]
    )  # xMax = left edge + (# of columns * x pixel resolution)
    yMin = yMax - (
        refl_shape[0] * res[1]
    )  # yMin = top edge - (# of rows * y pixel resolution)

    # serc_ext = (xMin, xMax, yMin, yMax)

    # View and apply scale factor and data ignore value
    scaleFactor = serc_reflArray.attrs["Scale_Factor"]
    noDataValue = serc_reflArray.attrs["Data_Ignore_Value"]

    da = serc_reflArray[:, :, :].astype(float)
    da[da == int(noDataValue)] = np.nan
    da[da < 0] = np.nan
    da[da > 10000] = np.nan
    da = da / scaleFactor

    coords = {
        "y": np.linspace(yMax, yMin, da.shape[0]),
        "x": np.linspace(xMin, xMax, da.shape[1]),
        "wavelength": wavelengths,
    }

    xda = xr.DataArray(
        da,
        coords=coords,
        dims=["y", "x", "wavelength"],
        attrs={
            "scale_factor": scaleFactor,
            "no_data_value": noDataValue,
            "crs": f"EPSG:{epsg_code_number}",
            "transform": (res[0], 0.0, xMin, 0.0, -res[1], yMax),
        },
    )

    if wavelengths is not None:
        xda = xda.sel(wavelength=wavelengths, method=method, **kwargs)

    dataset = xda.to_dataset(name="reflectance")
    dataset.attrs = dataset["reflectance"].attrs

    return dataset


def neon_to_image(
    dataset: Union[xr.Dataset, str],
    wavelengths: Optional[np.ndarray] = None,
    method: str = "nearest",
    output: Optional[str] = None,
    **kwargs: Any,
):
    """
    Converts an NEON dataset to an image.

    Args:
        dataset (Union[xr.Dataset, str]): The dataset containing the NEON data
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
        dataset = read_neon(dataset, method=method)

    if wavelengths is not None:
        dataset = dataset.sel(wavelength=wavelengths, method=method)

    return array_to_image(
        dataset["reflectance"],
        output=output,
        transpose=False,
        dtype=np.float32,
        **kwargs,
    )


def extract_neon(ds, lat, lon):
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
