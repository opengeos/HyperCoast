import xarray as xr
import holoviews as hv
import hvplot.xarray
import numpy as np


def read_emit(fp, **kwargs):
    """
    Opens an EMIT dataset from a file path and assigns new coordinates to it.

    Args:
        fp (str): The file path to the EMIT dataset.
        **kwargs: Additional keyword arguments to be passed to `xr.open_dataset`.

    Returns:
        xarray.Dataset: The dataset with new coordinates assigned.

    """
    ds = xr.open_dataset(fp, **kwargs)
    wvl = xr.open_dataset(fp, group="sensor_band_parameters")
    loc = xr.open_dataset(fp, group="location")
    ds = ds.assign_coords(
        {
            "downtrack": (["downtrack"], ds.downtrack.data),
            "crosstrack": (["crosstrack"], ds.crosstrack.data),
            **wvl.variables,
            **loc.variables,
        }
    )
    ds = ds.swap_dims({"bands": "wavelengths"})
    del wvl
    del loc
    return ds


def plot_emit(
    ds,
    downtrack,
    crosstrack,
    remove_nans=True,
    x="wavelengths",
    y="reflectance",
    color="black",
    frame_height=400,
    frame_width=600,
    **kwargs,
):
    """
    Plots a line graph of the reflectance data from a given dataset.

    Args:
        ds (xarray.Dataset): The dataset containing the reflectance data.
        downtrack (int): The downtrack coordinate to select.
        crosstrack (int): The crosstrack coordinate to select.
        remove_nans (bool, optional): If True, replace non-good wavelengths with NaN. Defaults to True.
        x (str, optional): The x-axis label. Defaults to "wavelengths".
        y (str, optional): The y-axis label. Defaults to "reflectance".
        color (str, optional): The color of the line. Defaults to "black".
        frame_height (int, optional): The height of the frame. Defaults to 400.
        frame_width (int, optional): The width of the frame. Defaults to 600.
        **kwargs: Additional keyword arguments to be passed to `hvplot.line`.

    Returns:
        hvplot.Plot: The line plot of the reflectance data.
    """
    if remove_nans:
        ds["reflectance"].data[:, :, ds["good_wavelengths"].data == 0] = np.nan
    example = ds["reflectance"].sel(downtrack=downtrack, crosstrack=crosstrack)
    return example.hvplot.line(
        y=y,
        x=x,
        color=color,
        frame_height=frame_height,
        frame_width=frame_width,
        **kwargs,
    )


def viz_emit(
    ds,
    wavelength,
    cmap="viridis",
    aspect="equal",
    frame_width=720,
    method="nearest",
    **kwargs,
):
    """
    Visualizes the reflectance data from a given dataset at a specific wavelength.

    Args:
        ds (xarray.Dataset): The dataset containing the reflectance data.
        wavelength (float): The specific wavelength to visualize.
        cmap (str, optional): The colormap to use. Defaults to "viridis".
        aspect (str, optional): The aspect ratio of the plot. Defaults to "equal".
        frame_width (int, optional): The width of the frame. Defaults to 720.
        method (str, optional): The method to use for data selection. Defaults to "nearest".
        **kwargs: Additional keyword arguments to be passed to `hvplot.image`.

    Returns:
        hvplot.Plot: The image plot of the reflectance data at the specified wavelength.
    """
    example = ds.sel(wavelengths=wavelength, method=method)
    return example.hvplot.image(cmap=cmap, aspect=aspect, frame_width=frame_width).opts(
        title=f"{example.wavelengths.values:.3f} {example.wavelengths.units}", **kwargs
    )
