"""This module contains functions to read and process PACE data.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Optional, Any
from .common import extract_date_from_filename


def read_pace(
    filepath, wavelengths=None, method="nearest", engine="h5netcdf", **kwargs
):
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

    rrs = xr.open_dataset(filepath, engine=engine, group="geophysical_data")["Rrs"]
    wvl = xr.open_dataset(filepath, engine=engine, group="sensor_band_parameters")
    dataset = xr.open_dataset(filepath, engine=engine, group="navigation_data")
    dataset = dataset.set_coords(("longitude", "latitude"))
    dataset = dataset.rename({"pixel_control_points": "pixels_per_line"})
    dataset = xr.merge([rrs, dataset.coords.to_dataset()])
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


def read_pace_chla(
    filepaths: Union[str, List[str]], engine: str = "h5netcdf", **kwargs
) -> xr.DataArray:
    """
    Reads chlorophyll-a data from PACE files and applies a logarithmic transformation.

    This function supports reading from a single file or multiple files. For multiple files,
    it combines them into a single dataset. It then extracts the chlorophyll-a variable,
    applies a logarithmic transformation, and sets the coordinate reference system to EPSG:4326.

    Args:
        filepaths: A string or a list of strings containing the file path(s) to the PACE chlorophyll-a data files.
        engine: The backend engine to use for reading files. Defaults to "h5netcdf".
        **kwargs: Additional keyword arguments to pass to `xr.open_dataset` or `xr.open_mfdataset`.

    Returns:
        An xarray DataArray containing the logarithmically transformed chlorophyll-a data with updated attributes.

    Examples:
        Read chlorophyll-a data from a single file:
        >>> chla_data = read_pace_chla('path/to/single/file.nc')

        Read and combine chlorophyll-a data from multiple files:
        >>> chla_data = read_pace_chla(['path/to/file1.nc', 'path/to/file2.nc'], combine='by_coords')
    """

    import os
    import glob
    import rioxarray

    date = None
    if os.path.isfile(filepaths):
        filepaths = [filepaths]
    if "combine" not in kwargs:
        kwargs["combine"] = "nested"
    if "concat_dim" not in kwargs:
        kwargs["concat_dim"] = "date"
    dataset = xr.open_mfdataset(filepaths, engine=engine, **kwargs)
    if not isinstance(filepaths, list):
        filepaths = glob.glob(filepaths)
        filepaths.sort()

    dates = [extract_date_from_filename(f) for f in filepaths]
    date = [timestamp.strftime("%Y-%m-%d") for timestamp in dates]
    dataset = dataset.assign_coords(date=("date", date))

    chla = np.log10(dataset["chlor_a"])
    chla.attrs.update(
        {
            "units": f'lg({dataset["chlor_a"].attrs["units"]})',
        }
    )

    if date is not None:
        chla.attrs["date"] = date

    chla = chla.transpose("lat", "lon", "date")

    chla.rio.write_crs("EPSG:4326", inplace=True)

    return chla


def viz_pace(
    dataset: Union[xr.Dataset, str],
    wavelengths: Optional[Union[List[float], float]] = None,
    method: str = "nearest",
    figsize: Tuple[float, float] = (6.4, 4.8),
    cmap: str = "jet",
    vmin: float = 0,
    vmax: float = 0.02,
    ncols: int = 1,
    crs: Optional[str] = None,
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    **kwargs,
):
    """
    Plots PACE data from a given xarray Dataset.

    Args:
        dataset (xr.Dataset): An xarray Dataset containing the PACE data.
        wavelengths (array-like, optional): Specific wavelengths to select. If None, all wavelengths are selected.
        method (str, optional): Method to use for selection when wavelengths is not None. Defaults to "nearest".
        figsize (tuple, optional): Figure size. Defaults to (6.4, 4.8).
        cmap (str, optional): Colormap to use. Defaults to "jet".
        vmin (float, optional): Minimum value for the colormap. Defaults to 0.
        vmax (float, optional): Maximum value for the colormap. Defaults to 0.02.
        ncols (int, optional): Number of columns in the plot. Defaults to 1.
        crs (str or cartopy.crs.CRS, optional): Coordinate reference system to use. If None, a simple plot is created. Defaults to None.
            See https://scitools.org.uk/cartopy/docs/latest/reference/projections.html
        xlim (array-like, optional): Limits for the x-axis. Defaults to None.
        ylim (array-like, optional): Limits for the y-axis. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the `plt.subplots` function.
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

    if crs is None:

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(figsize[0] * ncols, figsize[1] * nrows),
            **kwargs,
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

    else:

        import cartopy
        from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

        if crs == "default":
            crs = cartopy.crs.PlateCarree()

        if xlim is None:
            xlim = [math.floor(lon.min()), math.ceil(lon.max())]

        if ylim is None:
            ylim = [math.floor(lat.min()), math.ceil(lat.max())]

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(figsize[0] * ncols, figsize[1] * nrows),
            subplot_kw={"projection": cartopy.crs.PlateCarree()},
            **kwargs,
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
                    im = ax.pcolormesh(lon, lat, data, cmap="jet", vmin=0, vmax=0.02)
                    ax.coastlines()
                    ax.add_feature(cartopy.feature.STATES, linewidth=0.5)
                    ax.set_xticks(np.linspace(xlim[0], xlim[1], 5), crs=crs)
                    ax.set_yticks(np.linspace(ylim[0], ylim[1], 5), crs=crs)
                    lon_formatter = LongitudeFormatter(zero_direction_label=True)
                    lat_formatter = LatitudeFormatter()
                    ax.xaxis.set_major_formatter(lon_formatter)
                    ax.yaxis.set_major_formatter(lat_formatter)
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")
                    ax.set_title(
                        f"wavelength = {dataset.coords['wavelength'].values[index]} [nm]"
                    )
                    plt.colorbar(im, label="Reflectance")

        plt.tight_layout()
        plt.show()


def viz_pace_chla(
    data: Union[str, xr.DataArray],
    date: Optional[str] = None,
    aspect: float = 2,
    cmap: str = "jet",
    size: int = 6,
    **kwargs: Any,
) -> xr.plot.facetgrid.FacetGrid:
    """
    Visualizes PACE chlorophyll-a data using an xarray DataArray.

    This function supports loading data from a file path (str) or directly using an xarray DataArray.
    It allows for selection of a specific date for visualization or averages over all dates if none is specified.

    Args:
        data (Union[str, xr.DataArray]): The chlorophyll-a data to visualize. Can be a file path or an xarray DataArray.
        date (Optional[str], optional): Specific date to visualize. If None, averages over all dates. Defaults to None.
        aspect (float, optional): Aspect ratio of the plot. Defaults to 2.
        cmap (str, optional): Colormap for the plot. Defaults to "jet".
        size (int, optional): Size of the plot. Defaults to 6.
        **kwargs (Any): Additional keyword arguments to pass to `xarray.plot`.

    Returns:
        xr.plot.facetgrid.FacetGrid: The plot generated from the chlorophyll-a data.

    Raises:
        ValueError: If `data` is not a file path (str) or an xarray DataArray.
    """
    if isinstance(data, str):
        data = read_pace_chla(data)
    elif not isinstance(data, xr.DataArray):
        raise ValueError("data must be an xarray DataArray")

    if date is not None:
        data = data.sel(date=date)
    else:
        if "date" in data.coords:
            data = data.mean(dim="date")

    return data.plot(aspect=aspect, cmap=cmap, size=size, **kwargs)


def filter_pace(dataset, latitude, longitude, drop=True, return_plot=False, **kwargs):
    """
    Filters a PACE dataset based on latitude and longitude.

    Args:
        dataset (xr.Dataset): The PACE dataset to filter.
        latitude (float or tuple): The latitude to filter by. If a tuple or list, it represents a range.
        longitude (float or tuple): The longitude to filter by. If a tuple or list, it represents a range.
        drop (bool, optional): Whether to drop the filtered out data. Defaults to True.

    Returns:
        xr.DataArray: The filtered PACE data.
    """
    if isinstance(latitude, list) or isinstance(latitude, tuple):
        lat_con = (dataset["latitude"] > latitude[0]) & (
            dataset["latitude"] < latitude[1]
        )
    else:
        lat_con = dataset["latitude"] == latitude

    if isinstance(longitude, list) or isinstance(longitude, tuple):
        lon_con = (dataset["longitude"] > longitude[0]) & (
            dataset["longitude"] < longitude[1]
        )
    else:
        lon_con = dataset["longitude"] == longitude

    da = dataset["Rrs"].where(lat_con & lon_con, drop=drop, **kwargs)
    da_filtered = da.dropna(dim="latitude", how="all")
    da_filtered = da_filtered.dropna(dim="longitude", how="all")

    if return_plot:
        rrs_stack = da_filtered.stack(
            {"pixel": ["latitude", "longitude"]},
            create_index=False,
        )
        rrs_stack.plot.line(hue="pixel")
    else:
        return da_filtered


def extract_pace(
    dataset: Union[xr.Dataset, str],
    latitude: Union[float, Tuple[float, float]],
    longitude: Union[float, Tuple[float, float]],
    delta: float = 0.01,
    return_plot: bool = False,
    **kwargs,
) -> Union[xr.DataArray, plt.Figure]:
    """
    Extracts data from a PACE dataset for a given latitude and longitude range
        and calculates the mean over these dimensions.

    Args:
        dataset (Union[xr.Dataset, str]): The PACE dataset or path to the dataset file.
        latitude (Union[float, Tuple[float, float]]): The latitude or range of
            latitudes to extract data for.
        longitude (Union[float, Tuple[float, float]]): The longitude or range of
            longitudes to extract data for.
        delta (float, optional): The range to add/subtract to the latitude and
            longitude if they are not ranges. Defaults to 0.01.
        return_plot (bool, optional): Whether to return a plot of the data. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the plot function.

    Returns:
        Union[xr.DataArray, plt.figure.Figure]: The mean data over the latitude
            and longitude dimensions, or a plot of this data if return_plot is True.
    """
    if isinstance(latitude, list) or isinstance(latitude, tuple):
        pass
    else:
        latitude = (latitude - delta, latitude + delta)

    if isinstance(longitude, list) or isinstance(longitude, tuple):
        pass
    else:
        longitude = (longitude - delta, longitude + delta)

    ds = filter_pace(dataset, latitude, longitude, return_plot=False)
    data = ds.mean(dim=["latitude", "longitude"])
    if return_plot:
        return data.plot.line(**kwargs)
    else:
        return data


def grid_pace(dataset, wavelengths=None, method="nearest", **kwargs):
    """
    Grids a PACE dataset based on latitude and longitude.

    Args:
        dataset (xr.Dataset): The PACE dataset to grid.
        wavelengths (float or int): The wavelength to select.
        method (str, optional): The method to use for griddata interpolation.
            Defaults to "nearest".
        **kwargs: Additional keyword arguments to pass to the xr.Dataset constructor.

    Returns:
        xr.DataArray: The gridded PACE data.
    """
    from scipy.interpolate import griddata

    if wavelengths is None:
        wavelengths = dataset.coords["wavelength"].values[0]

    # Ensure wavelengths is a list
    if not isinstance(wavelengths, list):
        wavelengths = [wavelengths]

    lat = dataset.latitude
    lon = dataset.longitude

    grid_lat = np.linspace(lat.min(), lat.max(), lat.shape[0])
    grid_lon = np.linspace(lon.min(), lon.max(), lon.shape[1])
    grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lon, grid_lat)

    gridded_data_dict = {}
    for wavelength in wavelengths:
        data = dataset.sel(wavelength=wavelength, method=method)["Rrs"]
        gridded_data = griddata(
            (lat.data.flatten(), lon.data.flatten()),
            data.data.flatten(),
            (grid_lat_2d, grid_lon_2d),
            method=method,
        )
        gridded_data_dict[wavelength] = gridded_data

    # Create a 3D array with dimensions latitude, longitude, and wavelength
    gridded_data_3d = np.dstack(list(gridded_data_dict.values()))

    dataset2 = xr.Dataset(
        {"Rrs": (("latitude", "longitude", "wavelength"), gridded_data_3d)},
        coords={
            "latitude": ("latitude", grid_lat),
            "longitude": ("longitude", grid_lon),
            "wavelength": ("wavelength", list(gridded_data_dict.keys())),
        },
        **kwargs,
    )

    dataset2["Rrs"].rio.write_crs("EPSG:4326", inplace=True)

    return dataset2


def pace_to_image(
    dataset, wavelengths=None, method="nearest", gridded=False, output=None, **kwargs
):
    """
    Converts an PACE dataset to an image.

    Args:
        dataset (xarray.Dataset or str): The dataset containing the EMIT data or the file path to the dataset.
        wavelengths (array-like, optional): The specific wavelengths to select. If None, all wavelengths are selected. Defaults to None.
        method (str, optional): The method to use for data interpolation. Defaults to "nearest".
        gridded (bool, optional): Whether the dataset is a gridded dataset. Defaults to False,
        output (str, optional): The file path where the image will be saved. If None, the image will be returned as a PIL Image object. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to `leafmap.array_to_image`.

    Returns:
        rasterio.Dataset or None: The image converted from the dataset. If `output` is provided, the image will be saved to the specified file and the function will return None.
    """
    from leafmap import array_to_image

    if isinstance(dataset, str):
        dataset = read_pace(dataset, wavelengths=wavelengths, method="nearest")

    if wavelengths is not None:
        dataset = dataset.sel(wavelength=wavelengths, method="nearest")

    if not gridded:
        grid = grid_pace(dataset, wavelengths=wavelengths, method=method)
    else:
        grid = dataset
    data = grid["Rrs"]
    data.rio.write_crs("EPSG:4326", inplace=True)

    return array_to_image(data, transpose=False, output=output, **kwargs)


def pace_chla_to_image(data, output=None, **kwargs):
    """
    Converts PACE chlorophyll-a data to an image.

    Args:
        data (xr.DataArray or str): The chlorophyll-a data or the file path to the data.
        output (str, optional): The file path where the image will be saved. If None, the image will be returned as a PIL Image object. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to `leafmap.array_to_image`.

    Returns:
        rasterio.Dataset or None: The image converted from the data. If `output` is provided, the image will be saved to the specified file and the function will return None.
    """
    from leafmap import array_to_image

    if isinstance(data, str):
        data = read_pace_chla(data)
    elif not isinstance(data, xr.DataArray):
        raise ValueError("data must be an xarray DataArray")

    return array_to_image(data, transpose=False, output=output, **kwargs)
