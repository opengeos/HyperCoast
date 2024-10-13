# SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
#
# SPDX-License-Identifier: MIT

"""This module contains functions to read and process PACE data.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Optional, Any, Callable
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
    if "pixel_control_points" in dataset.dims:
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


def read_pace_aop(filepath, engine="h5netcdf", **kwargs):
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

    rrs = xr.open_dataset(filepath, engine=engine, group="geophysical_data", **kwargs)[
        "Rrs"
    ]
    wvl = xr.open_dataset(
        filepath, engine=engine, group="sensor_band_parameters", **kwargs
    )
    dataset = xr.open_dataset(
        filepath, engine=engine, group="navigation_data", **kwargs
    )
    dataset = dataset.set_coords(("longitude", "latitude"))
    if "pixel_control_points" in dataset.dims:
        dataset = dataset.rename({"pixel_control_points": "pixels_per_line"})
    dataset = xr.merge([rrs, dataset.coords.to_dataset()])
    dataset.coords["wavelength_3d"] = wvl.coords["wavelength_3d"]

    return dataset


def read_pace_bgc(
    filepath: str,
    variable: Optional[str] = None,
    engine: str = "h5netcdf",
    **kwargs: Any,
) -> xr.Dataset:
    """
    Reads PACE BGC data from a specified file and returns an xarray Dataset.

    This function opens a dataset from a file using the specified engine,
    optionally selects a single variable, merges geophysical and navigation data,
    sets appropriate coordinates, and renames dimensions for easier use.

    Args:
        filepath (str): The path to the file containing the PACE BGC data.
        variable (Optional[str], optional): The specific variable to extract
            from the geophysical_data group. If None, all variables are read. Defaults to None.
        engine (str, optional): The engine to use for reading the file. Defaults to "h5netcdf".
        **kwargs (Any): Additional keyword arguments to pass to `xr.open_dataset`.

    Returns:
        xr.Dataset: An xarray Dataset containing the requested PACE BGC data,
        with merged geophysical and navigation data, set coordinates, and renamed dimensions.

    Example:
        >>> dataset = read_pace_bgc("path/to/your/datafile.h5", variable="chlor_a")
        >>> print(dataset)
    """

    ds = xr.open_dataset(filepath, engine=engine, group="geophysical_data", **kwargs)
    if variable is not None:
        ds = ds[variable]
    dataset = xr.open_dataset(
        filepath, engine=engine, group="navigation_data", **kwargs
    )
    dataset = dataset.set_coords(("longitude", "latitude"))
    if "pixel_control_points" in dataset.dims:
        dataset = dataset.rename({"pixel_control_points": "pixels_per_line"})
    dataset = xr.merge([ds, dataset.coords.to_dataset()])
    dataset = dataset.rename(
        {
            "number_of_lines": "latitude",
            "pixels_per_line": "longitude",
        }
    )
    attrs = xr.open_dataset(filepath, engine=engine, **kwargs).attrs
    dataset.attrs.update(attrs)

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
    if isinstance(filepaths, str) and os.path.isfile(filepaths):
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


def view_pace_pixel_locations(
    filepath: str, step: int = 20, figsize: Tuple[float, float] = (8, 6), **kwargs: Any
) -> plt.Figure:
    """
    Visualizes a subset of PACE pixel locations on a scatter plot.

    This function reads PACE AOP data from a specified file, subsamples the data according to a step size,
    and plots the longitude and latitude of the selected pixels using a scatter plot.

    Args:
        filepath (str): The path to the file containing the PACE AOP data.
        step (int, optional): The step size for subsampling the data. A smaller step size results in more
            data points being plotted. Defaults to 20.
        **kwargs (Any): Additional keyword arguments to pass to the `plot.scatter` method.

    Returns:
        plt.Figure: A matplotlib figure object containing the scatter plot.

    Example:
        >>> plot = view_pace_pixel_locations("path/to/your/datafile.h5", step=10)
        >>> plt.show()
    """

    # Create a new figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create the plot
    dataset = read_pace_aop(filepath)
    number_of_lines = dataset.sizes["number_of_lines"]
    pixels_per_line = dataset.sizes["pixels_per_line"]

    ax.scatter(
        dataset.sel(
            {
                "number_of_lines": slice(None, None, number_of_lines // step),
                "pixels_per_line": slice(None, None, pixels_per_line // step),
            }
        ).longitude,
        dataset.sel(
            {
                "number_of_lines": slice(None, None, number_of_lines // step),
                "pixels_per_line": slice(None, None, pixels_per_line // step),
            }
        ).latitude,
        **kwargs,
    )

    # Set labels and title
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("PACE Pixel Locations")

    return fig


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

    grid_lat = np.linspace(lat.min().values, lat.max().values, lat.shape[0])
    grid_lon = np.linspace(lon.min().values, lon.max().values, lon.shape[1])
    grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lon, grid_lat)

    gridded_data_dict = {}
    for wavelength in wavelengths:
        data = dataset.sel(wavelength=wavelength, method="nearest")["Rrs"]
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


def grid_pace_bgc(
    dataset: xr.Dataset,
    variable: str = "chlor_a",
    method: str = "nearest",
    **kwargs: Any,
) -> xr.DataArray:
    """
    Grids PACE BGC data using specified interpolation method.

    This function takes an xarray Dataset containing PACE BGC data, interpolates it onto a regular grid
    using the specified method, and returns the gridded data as an xarray DataArray with the specified
    variable.

    Args:
        dataset (xr.Dataset): The input dataset containing PACE BGC data with latitude and longitude coordinates.
        variable (str, optional): The variable within the dataset to grid. Can be
            one of chlor_a, carbon_phyto, poc, chlor_a_unc, carbon_phyto_unc, and l2_flags.
            Defaults to "chlor_a".
        method (str, optional): The interpolation method to use. Options include "nearest", "linear", and "cubic".
            Defaults to "nearest".
        **kwargs (Any): Additional keyword arguments to pass to the xr.Dataset creation.

    Returns:
        xr.DataArray: The gridded data as an xarray DataArray, with the specified variable and EPSG:4326 CRS.

    Example:
        >>> dataset = hypercoast.read_pace_bgc("path_to_your_dataset.nc")
        >>> gridded_data = grid_pace_bgc(dataset, variable="chlor_a", method="nearest")
        >>> print(gridded_data)
    """
    import rioxarray
    from scipy.interpolate import griddata

    lat = dataset.latitude
    lon = dataset.longitude

    grid_lat = np.linspace(lat.min().values, lat.max().values, lat.shape[0])
    grid_lon = np.linspace(lon.min().values, lon.max().values, lon.shape[1])
    grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lon, grid_lat)

    data = dataset[variable]
    gridded_data = griddata(
        (lat.data.flatten(), lon.data.flatten()),
        data.data.flatten(),
        (grid_lat_2d, grid_lon_2d),
        method=method,
    )

    dataset2 = xr.Dataset(
        {variable: (("latitude", "longitude"), gridded_data)},
        coords={
            "latitude": ("latitude", grid_lat),
            "longitude": ("longitude", grid_lon),
        },
        **kwargs,
    )

    dataset2 = dataset2[variable].rio.write_crs("EPSG:4326")

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
    from leafmap import array_to_image, image_to_geotiff

    if isinstance(data, str):
        data = read_pace_chla(data)
    elif not isinstance(data, xr.DataArray):
        raise ValueError("data must be an xarray DataArray")

    image = array_to_image(data, transpose=False, output=None, **kwargs)

    if output is not None:
        image_to_geotiff(image, output, dtype="float32")

    return image


def cyano_band_ratios(
    dataset: Union[xr.Dataset, str],
    plot: bool = True,
    extent: List[float] = None,
    figsize: tuple[int, int] = (12, 6),
    **kwargs,
) -> xr.DataArray:
    """
    Calculates cyanobacteria band ratios from PACE data.

    Args:
        dataset (xr.Dataset or str): The dataset containing the PACE data or the file path to the dataset.
        plot (bool, optional): Whether to plot the data. Defaults to True.
        extent (list, optional): The extent of the plot. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to (12, 6).
        **kwargs: Additional keyword arguments to pass to the `plt.subplots` function.

    Returns:
        xr.DataArray: The cyanobacteria band ratios.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    if isinstance(dataset, str):
        dataset = read_pace(dataset)
    elif not isinstance(dataset, xr.Dataset):
        raise ValueError("dataset must be an xarray Dataset")

    da = dataset["Rrs"]
    data = (
        (da.sel(wavelength=650) > da.sel(wavelength=620))
        & (da.sel(wavelength=701) > da.sel(wavelength=681))
        & (da.sel(wavelength=701) > da.sel(wavelength=450))
    )

    if plot:
        # Create a plot
        _, ax = plt.subplots(
            figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()}, **kwargs
        )

        if extent is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Plot the data
        data.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap="coolwarm",
            cbar_kwargs={"label": "Cyano"},
        )

        # Add coastlines
        ax.coastlines()

        # Add state boundaries
        states_provinces = cfeature.NaturalEarthFeature(
            category="cultural",
            name="admin_1_states_provinces_lines",
            scale="50m",
            facecolor="none",
        )

        ax.add_feature(states_provinces, edgecolor="gray")

        # Optionally, add gridlines, labels, etc.
        ax.gridlines(draw_labels=True)
        plt.show()

    return data


def apply_kmeans(
    dataset: Union[xr.Dataset, str],
    n_clusters: int = 6,
    filter_condition: Optional[Callable[[xr.DataArray], xr.DataArray]] = None,
    plot: bool = True,
    figsize: tuple[int, int] = (8, 6),
    colors: list[str] = None,
    extent: list[float] = None,
    title: str = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies K-means clustering to the dataset and optionally plots the results.

    Args:
        dataset (xr.Dataset | str): The dataset containing the PACE data or the file path to the dataset.
        n_clusters (int, optional): Number of clusters for K-means. Defaults to 6.
        plot (bool, optional): Whether to plot the data. Defaults to True.
        figsize (tuple[int, int], optional): Figure size for the plot. Defaults to (8, 6).
        colors (list[str], optional): List of colors to use for the clusters. Defaults to None.
        extent (list[float] | None, optional): The extent to zoom in to the specified region. Defaults to None.
        title (str | None, optional): Title for the plot. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the `plt.subplots` function.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The cluster labels, latitudes, and longitudes.
    """

    import numpy as np
    from sklearn.cluster import KMeans

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    if isinstance(dataset, str):
        dataset = read_pace(dataset)
    elif isinstance(dataset, xr.DataArray):
        dataset = dataset.to_dataset()
    elif not isinstance(dataset, xr.Dataset):
        raise ValueError("dataset must be an xarray Dataset")

    if title is None:
        title = f"K-means Clustering with {n_clusters} Clusters"

    da = dataset["Rrs"]

    reshaped_data = da.values.reshape(-1, da.shape[-1])
    reshaped_data_no_nan = reshaped_data[~np.isnan(reshaped_data).any(axis=1)]

    # Apply K-means clustering to classify into 5-6 water types.
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(reshaped_data_no_nan)

    # Initialize an array for cluster labels with NaN
    labels = np.full(reshaped_data.shape[0], np.nan)

    # Assign the computed cluster labels to the non-NaN positions
    labels[~np.isnan(reshaped_data).any(axis=1)] = kmeans.labels_

    # Reshape the labels back to the original spatial dimensions
    cluster_labels = labels.reshape(da.shape[:-1])

    if filter_condition is not None:
        cluster_labels = np.where(filter_condition, cluster_labels, np.nan)

    latitudes = da.coords["latitude"].values
    longitudes = da.coords["longitude"].values

    if plot:

        # Create a custom discrete color map for K-means clusters
        if colors is None:
            colors = ["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628", "#984ea3"]
        cmap = mcolors.ListedColormap(colors)
        bounds = np.arange(-0.5, n_clusters, 1)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Create a figure and axis with the correct map projection

        if "dpi" not in kwargs:
            kwargs["dpi"] = 100

        if "subplot_kw" not in kwargs:
            kwargs["subplot_kw"] = {"projection": ccrs.PlateCarree()}

        fig, ax = plt.subplots(
            figsize=figsize,
            **kwargs,
        )

        # Plot the K-means classification results on the map
        im = ax.pcolormesh(
            longitudes,
            latitudes,
            cluster_labels,
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
        )

        # Add geographic features for context
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.STATES, linestyle="--")

        # Add gridlines
        ax.gridlines(draw_labels=True)

        # Set the extent to zoom in to the specified region
        if extent is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Add color bar with labels
        cbar = plt.colorbar(
            im,
            ax=ax,
            orientation="vertical",
            # pad=0.02,
            fraction=0.05,
            ticks=np.arange(n_clusters),
        )
        cbar.ax.set_yticklabels([f"Class {i+1}" for i in range(n_clusters)])
        cbar.set_label("Water Types", rotation=270, labelpad=10)

        # Add title
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Show the plot
        plt.show()

    return cluster_labels, latitudes, longitudes


def apply_pca(
    dataset: Union[xr.Dataset, str],
    n_components: int = 3,
    plot: bool = True,
    figsize: tuple[int, int] = (8, 6),
    x_component: int = 0,
    y_component: int = 1,
    color: str = "blue",
    title: str = "PCA of Spectral Data",
    **kwargs,
) -> np.ndarray:
    """
    Applies Principal Component Analysis (PCA) to the dataset and optionally plots the results.

    Args:
        dataset (xr.Dataset | str): The dataset containing the PACE data or the file path to the dataset.
        n_components (int, optional): Number of principal components to compute. Defaults to 3.
        plot (bool, optional): Whether to plot the data. Defaults to True.
        figsize (tuple[int, int], optional): Figure size for the plot. Defaults to (8, 6).
        x_component (int, optional): The principal component to plot on the x-axis. Defaults to 0.
        y_component (int, optional): The principal component to plot on the y-axis. Defaults to 1.
        color (str, optional): Color of the scatter plot points. Defaults to "blue".
        title (str, optional): Title for the plot. Defaults to "PCA of Spectral Data".
        **kwargs: Additional keyword arguments to pass to the `plt.scatter` function.

    Returns:
        np.ndarray: The PCA-transformed data.
    """
    from sklearn.decomposition import PCA

    if isinstance(dataset, str):
        dataset = read_pace(dataset)
    elif not isinstance(dataset, xr.Dataset):
        raise ValueError("dataset must be an xarray Dataset")

    da = dataset["Rrs"]

    # Reshape data to (n_pixels, n_bands)
    reshaped_data = da.values.reshape(-1, da.shape[-1])

    # Handle NaNs by removing them
    reshaped_data_no_nan = reshaped_data[~np.isnan(reshaped_data).any(axis=1)]

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(reshaped_data_no_nan)

    if plot:
        plt.figure(figsize=figsize)
        if "s" not in kwargs:
            kwargs["s"] = 1
        plt.scatter(
            pca_data[:, x_component], pca_data[:, y_component], c=color, **kwargs
        )
        plt.title(title)
        plt.xlabel(f"Principal Component {x_component + 1}")
        plt.ylabel(f"Principal Component {y_component + 1}")
        plt.show()

    return pca_data


def apply_sam(
    dataset: Union[xr.Dataset, str],
    n_components: int = 3,
    n_clusters: int = 6,
    random_state: int = 0,
    plot: bool = True,
    figsize: tuple[int, int] = (8, 6),
    extent: list[float] = None,
    colors: list[str] = None,
    title: str = "Spectral Angle Mapper",
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies Spectral Angle Mapper (SAM) to the dataset and optionally plots the results.

    Args:
        dataset (Union[xr.Dataset, str]): The dataset containing the PACE data or the file path to the dataset.
        n_components (int, optional): Number of principal components to compute. Defaults to 3.
        n_clusters (int, optional): Number of clusters for K-means. Defaults to 6.
        random_state (int, optional): Random state for K-means. Defaults to 0.
        plot (bool, optional): Whether to plot the data. Defaults to True.
        figsize (Tuple[int, int], optional): Figure size for the plot. Defaults to (8, 6).
        extent (List[float], optional): The extent to zoom in to the specified region. Defaults to None.
        colors (List[str], optional): Colors for the clusters. Defaults to None.
        title (str, optional): Title for the plot. Defaults to "Spectral Angle Mapper".
        **kwargs: Additional keyword arguments to pass to the `plt.subplots` function.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The best match classification, latitudes, and longitudes.
    """
    from sklearn.cluster import KMeans
    import matplotlib.colors as mcolors
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from sklearn.decomposition import PCA

    if isinstance(dataset, str):
        dataset = read_pace(dataset)
    elif isinstance(dataset, xr.DataArray):
        dataset = dataset.to_dataset()
    elif not isinstance(dataset, xr.Dataset):
        raise ValueError("dataset must be an xarray Dataset")

    da = dataset["Rrs"]

    # Reshape data to (n_pixels, n_bands)
    reshaped_data = da.values.reshape(-1, da.shape[-1])

    # Handle NaNs by removing them
    reshaped_data_no_nan = reshaped_data[~np.isnan(reshaped_data).any(axis=1)]

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(reshaped_data_no_nan)

    # Apply K-means to find clusters representing endmembers
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(pca_data)

    # The cluster centers in the original spectral space are your endmembers
    endmembers = pca.inverse_transform(kmeans.cluster_centers_)

    def spectral_angle_mapper(pixel, reference):
        norm_pixel = np.linalg.norm(pixel)
        norm_reference = np.linalg.norm(reference)
        cos_theta = np.dot(pixel, reference) / (norm_pixel * norm_reference)
        angle = np.arccos(np.clip(cos_theta, -1, 1))
        return angle

    # Apply SAM for each pixel and each endmember
    angles = np.zeros((reshaped_data_no_nan.shape[0], endmembers.shape[0]))

    for i in range(reshaped_data_no_nan.shape[0]):
        for j in range(endmembers.shape[0]):
            angles[i, j] = spectral_angle_mapper(
                reshaped_data_no_nan[i, :], endmembers[j, :]
            )

    # Find the minimum angle (best match) for each pixel
    best_match = np.argmin(angles, axis=1)

    # Reshape best_match back to the original spatial dimensions
    original_shape = da.shape[:-1]  # Get the spatial dimensions
    best_match_full = np.full(reshaped_data.shape[0], np.nan)
    best_match_full[~np.isnan(reshaped_data).any(axis=1)] = best_match
    best_match_full = best_match_full.reshape(original_shape)

    latitudes = da.coords["latitude"].values
    longitudes = da.coords["longitude"].values

    if plot:

        if colors is None:
            colors = ["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628", "#984ea3"]
        # Create a custom discrete color map
        cmap = mcolors.ListedColormap(colors)
        bounds = np.arange(-0.5, n_clusters, 1)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Create a figure and axis with the correct map projection
        fig, ax = plt.subplots(
            figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()}, **kwargs
        )

        # Plot the SAM classification results
        im = ax.pcolormesh(
            longitudes,
            latitudes,
            best_match_full,
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
        )

        # Add geographic features for context
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.STATES, linestyle="--")

        # Add gridlines
        ax.gridlines(draw_labels=True)

        # Set the extent to zoom in to the specified region
        if extent is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Add color bar with labels
        cbar = plt.colorbar(
            im,
            ax=ax,
            orientation="vertical",
            # pad=0.02,
            fraction=0.05,
            ticks=np.arange(n_clusters),
        )
        cbar.ax.set_yticklabels([f"Class {i+1}" for i in range(n_clusters)])
        cbar.set_label("Water Types", rotation=270, labelpad=20)

        # Add title
        ax.set_title(title, fontsize=14)

        # Show the plot
        plt.show()

    return best_match_full, latitudes, longitudes
