import h5py
import xarray as xr
import numpy as np
import requests
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Optional, Any, Callable
from .common import download_file


def read_tanager(filepath, bands=None, stac_url=None, **kwargs):
    """
    Read Planet Tanager HDF5 hyperspectral data and return an xarray.Dataset.

    Parameters:
        filepath (str or Path): Local file path or HTTPS URL to the .h5 file.
        bands (list or slice, optional): Indices of spectral bands to read.
        stac_url (str, optional): STAC item URL containing wavelength metadata.
        **kwargs: Additional arguments (reserved for future use).

    Returns:
        xr.Dataset: Dataset with georeferenced radiance and band metadata.
    """
    if isinstance(filepath, str) and filepath.startswith("https://"):
        filepath = download_file(filepath)  # You must define this if needed

    if stac_url is None:
        # Example static fallback STAC URL; update as needed
        stac_url = (
            "https://www.planet.com/data/stac/tanager-core-imagery/coastal-water-bodies/"
            "20250514_193937_64_4001/20250514_193937_64_4001.json"
        )

    # Parse STAC metadata
    stac_item = requests.get(stac_url, timeout=10).json()
    bands_meta = stac_item["assets"]["basic_radiance_hdf5"]["eo:bands"]

    wavelengths = np.array([b["center_wavelength"] for b in bands_meta])
    fwhm = np.array([b.get("full_width_half_max", np.nan) for b in bands_meta])

    if bands is not None:
        wavelengths = wavelengths[bands]
        fwhm = fwhm[bands]

    with h5py.File(filepath, "r") as f:
        data = f["HDFEOS/SWATHS/HYP/Data Fields/toa_radiance"][()]
        lat = f["HDFEOS/SWATHS/HYP/Geolocation Fields/Latitude"][()]
        lon = f["HDFEOS/SWATHS/HYP/Geolocation Fields/Longitude"][()]

    if bands is not None:
        data = data[bands]

    coords = {
        "band": np.arange(data.shape[0]),
        "wavelength": ("band", wavelengths),
        "fwhm": ("band", fwhm),
        "latitude": (("y", "x"), lat),
        "longitude": (("y", "x"), lon),
    }

    da = xr.DataArray(data, dims=("band", "y", "x"), coords=coords, name="toa_radiance")

    ds = xr.Dataset(
        data_vars={"toa_radiance": da},
        coords={
            "band": da.band,
            "wavelength": ("band", wavelengths),
            "fwhm": ("band", fwhm),
            "latitude": (("y", "x"), lat),
            "longitude": (("y", "x"), lon),
        },
        attrs={
            "source": "Planet Tanager HDF5",
            "stac_item": stac_url,
        },
        **kwargs,
    )

    return ds


def grid_tanager(dataset, bands=None, method="nearest", **kwargs):
    """
    Grids a Tanager dataset based on latitude and longitude.

    Args:
        dataset (xr.Dataset): The Tanager dataset to grid.
        bands (list): The bands to select.
        method (str, optional): The method to use for griddata interpolation.
            Defaults to "nearest".
        **kwargs: Additional keyword arguments to pass to the xr.Dataset constructor.

    Returns:
        xr.DataArray: The gridded Tanager data.
    """
    from scipy.interpolate import griddata
    from scipy.spatial import ConvexHull

    if bands is None:
        bands = dataset.coords["band"].values[0]

    # Ensure wavelengths is a list
    if not isinstance(bands, list):
        bands = [bands]

    lat = dataset.latitude
    lon = dataset.longitude

    # Find valid data points for any band to define spatial mask
    first_band_data = dataset.sel(band=bands[0], method="nearest")["toa_radiance"]
    overall_valid_mask = ~np.isnan(first_band_data.data) & (first_band_data.data > 0)

    if not np.any(overall_valid_mask):
        # No valid data, return empty grid
        grid_lat = np.linspace(lat.min().values, lat.max().values, lat.shape[0])
        grid_lon = np.linspace(lon.min().values, lon.max().values, lon.shape[1])
        grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lon, grid_lat)
        gridded_data_dict = {band: np.full_like(grid_lat_2d, np.nan) for band in bands}
    else:
        # Get valid coordinates for spatial masking
        valid_lat = lat.data[overall_valid_mask]
        valid_lon = lon.data[overall_valid_mask]

        grid_lat = np.linspace(lat.min().values, lat.max().values, lat.shape[0])
        grid_lon = np.linspace(lon.min().values, lon.max().values, lon.shape[1])
        grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lon, grid_lat)

        # Create convex hull mask to limit interpolation to data coverage area
        try:
            hull = ConvexHull(np.column_stack([valid_lon, valid_lat]))
            from matplotlib.path import Path

            hull_path = Path(
                np.column_stack([valid_lon[hull.vertices], valid_lat[hull.vertices]])
            )
            grid_points = np.column_stack(
                [grid_lon_2d.flatten(), grid_lat_2d.flatten()]
            )
            inside_hull = hull_path.contains_points(grid_points).reshape(
                grid_lat_2d.shape
            )
        except:
            # Fallback: use simple bounding box
            inside_hull = (
                (grid_lat_2d >= valid_lat.min())
                & (grid_lat_2d <= valid_lat.max())
                & (grid_lon_2d >= valid_lon.min())
                & (grid_lon_2d <= valid_lon.max())
            )

        gridded_data_dict = {}
        for band in bands:
            data = dataset.sel(band=band, method="nearest")["toa_radiance"]

            # Mask nodata values (both NaN and zero values)
            data_flat = data.data.flatten()
            valid_mask = ~np.isnan(data_flat) & (data_flat > 0)
            if not np.any(valid_mask):
                gridded_data = np.full_like(grid_lat_2d, np.nan)
            else:
                gridded_data = griddata(
                    (lat.data.flatten()[valid_mask], lon.data.flatten()[valid_mask]),
                    data_flat[valid_mask],
                    (grid_lat_2d, grid_lon_2d),
                    method=method,
                    fill_value=np.nan,
                )
                # Apply spatial mask to prevent edge interpolation
                gridded_data[~inside_hull] = np.nan
            gridded_data_dict[band] = gridded_data

    wavelengths = dataset.wavelength.values[bands]
    # Create a 3D array with dimensions latitude, longitude, and wavelength
    gridded_data_3d = np.dstack(list(gridded_data_dict.values()))

    dataset2 = xr.Dataset(
        {"toa_radiance": (("latitude", "longitude", "band"), gridded_data_3d)},
        coords={
            "latitude": ("latitude", grid_lat),
            "longitude": ("longitude", grid_lon),
            "band": ("band", list(gridded_data_dict.keys())),
            "wavelength": ("band", wavelengths),
        },
        **kwargs,
    )

    dataset2["toa_radiance"].rio.write_crs("EPSG:4326", inplace=True)

    return dataset2


def tanager_to_image(dataset, bands=None, method="nearest", output=None, **kwargs):
    """
    Converts an Tanager dataset to an image.

    Args:
        dataset (xarray.Dataset or str): The dataset containing the EMIT data or the file path to the dataset.
        bands (array-like, optional): The specific bands to select. If None, all bands are selected. Defaults to None.
        method (str, optional): The method to use for data interpolation. Defaults to "nearest".
        output (str, optional): The file path where the image will be saved. If None, the image will be returned as a PIL Image object. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to `leafmap.array_to_image`.

    Returns:
        rasterio.Dataset or None: The image converted from the dataset. If `output` is provided, the image will be saved to the specified file and the function will return None.
    """
    from leafmap import array_to_image

    if isinstance(dataset, str):
        dataset = read_tanager(dataset, bands=bands)

    grid = grid_tanager(dataset, bands=bands, method=method)

    data = grid["toa_radiance"]
    data.rio.write_crs("EPSG:4326", inplace=True)

    return array_to_image(data, transpose=False, output=output, **kwargs)


def filter_tanager(
    dataset, latitude, longitude, drop=True, return_plot=False, **kwargs
):
    """
    Filters a Tanager dataset based on latitude and longitude.

    Args:
        dataset (xr.Dataset): The Tanager dataset to filter.
        latitude (float or tuple): The latitude to filter by. If a tuple or list, it represents a range.
        longitude (float or tuple): The longitude to filter by. If a tuple or list, it represents a range.
        drop (bool, optional): Whether to drop the filtered out data. Defaults to True.

    Returns:
        xr.DataArray: The filtered Tanager data.
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

    da = dataset["toa_radiance"].where(lat_con & lon_con, drop=drop, **kwargs)
    da_filtered = da.dropna(dim="y", how="all")
    da_filtered = da_filtered.dropna(dim="x", how="all")

    if return_plot:
        rrs_stack = da_filtered.stack(
            {"pixel": ["y", "x"]},
            create_index=False,
        )
        rrs_stack.plot.line(hue="pixel")
    else:
        return da_filtered


def extract_tanager(
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

    ds = filter_tanager(dataset, latitude, longitude, return_plot=False)
    data = ds.mean(dim=["y", "x"])
    if return_plot:
        return data.plot.line(**kwargs)
    else:
        return data
