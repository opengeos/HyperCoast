"""This module contains functions to read and process PACE data.
"""

import xarray as xr


def read_pace(filepath, wavelengths=None, method="nearest", **kwargs):
    """
    Reads PACE data from a given file and returns an xarray Dataset.

    Args:
        filepath (str): Path to the file to read.
        wavelengths (array-like, optional): Specific wavelengths to select. If None, all wavelengths are selected.
        method (str, optional): Method to use for selection when wavelengths is not None. Defaults to "nearest".
        **kwargs: Additional keyword arguments to pass to the `sel` method when wavelengths is not None.

    Returns:
        xr.Dataset: An xarray Dataset containing the PACE data.
    """

    rrs = xr.open_dataset(filepath, group="geophysical_data")["Rrs"]
    wvl = xr.open_dataset(filepath, group="sensor_band_parameters")
    dataset = xr.open_dataset(filepath, group="navigation_data")
    dataset = dataset.set_coords(("longitude", "latitude"))
    dataset = dataset.rename({"pixel_control_points": "pixels_per_line"})
    dataset = xr.merge((rrs, dataset.coords))
    dataset.coords["wavelength_3d"] = wvl.coords["wavelength_3d"]
    dataset = dataset.rename(
        {
            "number_of_lines": "latitude",
            "pixels_per_line": "longitude",
            "wavelength_3d": "wavelength",
        }
    )

    if wavelengths is not None:
        dataset = dataset.sel(wavelength=wavelengths, method=method, **kwargs)

    return dataset


def plot_pace(
    dataset,
    wavelengths=None,
    method="nearest",
    cmap="jet",
    vmin=0,
    vmax=0.02,
    ncols=1,
    **kwargs,
):
    """
    Plots PACE data from a given xarray Dataset.

    Args:
        dataset (xr.Dataset): An xarray Dataset containing the PACE data.
        wavelengths (array-like, optional): Specific wavelengths to select. If None, all wavelengths are selected.
        method (str, optional): Method to use for selection when wavelengths is not None. Defaults to "nearest".
        cmap (str, optional): Colormap to use. Defaults to "jet".
        vmin (float, optional): Minimum value for the colormap. Defaults to 0.
        vmax (float, optional): Maximum value for the colormap. Defaults to 0.02.
        ncols (int, optional): Number of columns in the plot. Defaults to 1.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import math

    if isinstance(dataset, str):
        dataset = read_pace(dataset, wavelengths, method)

    if wavelengths is not None:
        if not isinstance(wavelengths, list):
            wavelengths = [wavelengths]
        dataset = dataset.sel(wavelength=wavelengths, method=method)
    else:
        wavelengths = dataset.coords["wavelength"][0].values.tolist()

    lat = dataset.coords["latitude"]
    lon = dataset.coords["longitude"]

    nrows = math.ceil(len(wavelengths) / ncols)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(6.4 * ncols, 4.8 * nrows), **kwargs
    )

    for i in range(nrows):
        for j in range(ncols):
            index = i * ncols + j
            if index < len(wavelengths):
                wavelength = wavelengths[index]
                data = dataset.sel(wavelength=wavelength, method=method)["Rrs"]

                if min(nrows, ncols) == 1:
                    ax = axes[index]
                else:
                    ax = axes[i, j]
                im = ax.pcolormesh(
                    lon, lat, np.squeeze(data), cmap=cmap, vmin=vmin, vmax=vmax
                )
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                ax.set_title(
                    f"wavelength = {dataset.coords['wavelength'].values[index]} [nm]"
                )
                fig.colorbar(im, ax=ax, label="Reflectance")

    plt.tight_layout()
    plt.show()
