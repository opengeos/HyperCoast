import rioxarray
import numpy as np
import xarray as xr


def read_desis(filepath, bands=None, method="nearest", **kwargs):
    """
    Reads DESIS data from a given file and returns an xarray Dataset.

    Args:
        filepath (str): Path to the file to read.
        bands (array-like, optional): Specific bands to select. If None, all
            bands are selected.
        method (str, optional): Method to use for selection when bands is not
            None. Defaults to "nearest".
        **kwargs: Additional keyword arguments to pass to the `sel` method when
            bands is not None.

    Returns:
        xr.Dataset: An xarray Dataset containing the DESIS data.
    """

    dataset = xr.open_dataset(filepath)

    if bands is not None:
        dataset = dataset.sel(band=bands, method=method, **kwargs)

    dataset = dataset.rename({"band_data": "reflectance"})

    return dataset


def desis_to_image(dataset, bands=None, method="nearest", output=None, **kwargs):
    """
    Converts an DESIS dataset to an image.

    Args:
        dataset (xarray.Dataset or str): The dataset containing the DESIS data
            or the file path to the dataset.
        bands (array-like, optional): The specific bands to select. If None, all
            bands are selected. Defaults to None.
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

    if bands is not None:
        dataset = dataset.sel(band=bands, method=method)

    return array_to_image(dataset["reflectance"], output=output, **kwargs)
