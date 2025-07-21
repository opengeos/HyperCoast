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

    wavelengths = np.array([b["center_wavelength"] * 1000 for b in bands_meta])
    fwhm = np.array([b.get("full_width_half_max", np.nan) * 1000 for b in bands_meta])

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
        "wavelength": wavelengths,
        "fwhm": ("wavelength", fwhm),
        "latitude": (("y", "x"), lat),
        "longitude": (("y", "x"), lon),
    }

    da = xr.DataArray(
        data, dims=("wavelength", "y", "x"), coords=coords, name="toa_radiance"
    )

    ds = xr.Dataset(
        data_vars={"toa_radiance": da},
        coords={
            "wavelength": da.wavelength,
            "fwhm": ("wavelength", fwhm),
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


def grid_tanager(
    dataset,
    bands=None,
    wavelengths=None,
    method="nearest",
    row_range=None,
    col_range=None,
    **kwargs,
):
    """
    Grids a Tanager dataset based on latitude and longitude.

    Args:
        dataset (xr.Dataset): The Tanager dataset to grid.
        bands (list, optional): The band indices to select. Defaults to None.
        wavelengths (list, optional): The wavelength values to select. Takes priority over bands. Defaults to None.
        method (str, optional): The method to use for griddata interpolation.
            Defaults to "nearest".
        row_range (tuple, optional): Row range (start_row, end_row) to subset the data. Defaults to None.
        col_range (tuple, optional): Column range (start_col, end_col) to subset the data. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the xr.Dataset constructor.

    Returns:
        xr.DataArray: The gridded Tanager data.
    """
    from scipy.interpolate import griddata
    from scipy.spatial import ConvexHull

    # Priority: wavelengths > bands > default
    if wavelengths is not None:
        # Use wavelengths directly
        if not isinstance(wavelengths, list):
            wavelengths = [wavelengths]
        selected_wavelengths = wavelengths
    elif bands is not None:
        # Convert bands to wavelengths
        if not isinstance(bands, list):
            bands = [bands]

        selected_wavelengths = []
        for band in bands:
            if isinstance(band, (int, np.integer)) or (
                isinstance(band, float) and band < 500
            ):
                # Treat as band index
                selected_wavelengths.append(
                    dataset.coords["wavelength"].values[int(band)]
                )
            else:
                # Treat as wavelength value
                selected_wavelengths.append(band)
    else:
        # Default to first wavelength
        selected_wavelengths = dataset.coords["wavelength"].values

    # Apply spatial subset filtering if ranges are provided
    if row_range is not None or col_range is not None:
        # Get original array dimensions
        y_size, x_size = dataset.latitude.shape

        # Determine row and column indices
        start_row = row_range[0] if row_range is not None else 0
        end_row = row_range[1] if row_range is not None else y_size
        start_col = col_range[0] if col_range is not None else 0
        end_col = col_range[1] if col_range is not None else x_size

        # Ensure indices are within bounds
        start_row = max(0, min(start_row, y_size))
        end_row = max(start_row, min(end_row, y_size))
        start_col = max(0, min(start_col, x_size))
        end_col = max(start_col, min(end_col, x_size))

        # Subset the dataset using isel for y and x dimensions
        dataset_subset = dataset.isel(
            y=slice(start_row, end_row), x=slice(start_col, end_col)
        )

        # For subsets, return the data directly without interpolation to avoid artifacts
        selected_data_list = []
        for wl in selected_wavelengths:
            data = dataset_subset.sel(wavelength=wl, method="nearest")["toa_radiance"]
            selected_data_list.append(data.values)

        # Stack wavelengths as the last dimension
        gridded_data_3d = np.stack(selected_data_list, axis=-1)

        # Create output dataset with proper coordinates
        lat_subset = dataset_subset.latitude
        lon_subset = dataset_subset.longitude

        # Create coordinate arrays for the subset
        y_coords = np.arange(gridded_data_3d.shape[0])
        x_coords = np.arange(gridded_data_3d.shape[1])

        dataset2 = xr.Dataset(
            {"toa_radiance": (("y", "x", "wavelength"), gridded_data_3d)},
            coords={
                "y": ("y", y_coords),
                "x": ("x", x_coords),
                "wavelength": ("wavelength", selected_wavelengths),
                "latitude": (("y", "x"), lat_subset.values),
                "longitude": (("y", "x"), lon_subset.values),
            },
            **kwargs,
        )

        dataset2["toa_radiance"].rio.write_crs("EPSG:4326", inplace=True)
        return dataset2

    lat = dataset.latitude
    lon = dataset.longitude

    # Find valid data points for any wavelength to define spatial mask
    first_wavelength_data = dataset.sel(
        wavelength=selected_wavelengths[0], method="nearest"
    )["toa_radiance"]
    overall_valid_mask = ~np.isnan(first_wavelength_data.data) & (
        first_wavelength_data.data > 0
    )

    if not np.any(overall_valid_mask):
        # No valid data, return empty grid using valid lat/lon bounds
        valid_lat_data = lat.data[~np.isnan(lat.data)]
        valid_lon_data = lon.data[~np.isnan(lon.data)]

        if len(valid_lat_data) == 0 or len(valid_lon_data) == 0:
            # Fallback to original bounds if no valid subset data
            grid_lat = np.linspace(lat.min().values, lat.max().values, lat.shape[0])
            grid_lon = np.linspace(lon.min().values, lon.max().values, lon.shape[1])
        else:
            grid_lat = np.linspace(
                valid_lat_data.min(), valid_lat_data.max(), lat.shape[0]
            )
            grid_lon = np.linspace(
                valid_lon_data.min(), valid_lon_data.max(), lon.shape[1]
            )

        grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lon, grid_lat)
        gridded_data_dict = {
            wl: np.full_like(grid_lat_2d, np.nan) for wl in selected_wavelengths
        }
    else:
        # Get valid coordinates for spatial masking
        valid_lat = lat.data[overall_valid_mask]
        valid_lon = lon.data[overall_valid_mask]

        # Create grid based on valid data bounds (considering subset if applied)
        grid_lat = np.linspace(valid_lat.min(), valid_lat.max(), lat.shape[0])
        grid_lon = np.linspace(valid_lon.min(), valid_lon.max(), lon.shape[1])
        grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lon, grid_lat)

        # For subsets, use simple bounding box instead of convex hull to avoid over-masking
        if row_range is not None or col_range is not None:
            # For subsets, just use bounding box
            inside_hull = (
                (grid_lat_2d >= valid_lat.min())
                & (grid_lat_2d <= valid_lat.max())
                & (grid_lon_2d >= valid_lon.min())
                & (grid_lon_2d <= valid_lon.max())
            )
        else:
            # For full dataset, use convex hull for better edge handling
            try:
                hull = ConvexHull(np.column_stack([valid_lon, valid_lat]))
                from matplotlib.path import Path

                hull_path = Path(
                    np.column_stack(
                        [valid_lon[hull.vertices], valid_lat[hull.vertices]]
                    )
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
        for wl in selected_wavelengths:
            data = dataset.sel(wavelength=wl, method="nearest")["toa_radiance"]

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
                # Apply spatial mask to prevent edge interpolation (only for full dataset)
                if row_range is None and col_range is None:
                    gridded_data[~inside_hull] = np.nan
            gridded_data_dict[wl] = gridded_data

    selected_wavelengths = list(gridded_data_dict.keys())
    # Create a 3D array with dimensions latitude, longitude, and wavelength
    gridded_data_3d = np.dstack(list(gridded_data_dict.values()))

    dataset2 = xr.Dataset(
        {"toa_radiance": (("latitude", "longitude", "wavelength"), gridded_data_3d)},
        coords={
            "latitude": ("latitude", grid_lat),
            "longitude": ("longitude", grid_lon),
            "wavelength": ("wavelength", selected_wavelengths),
        },
        **kwargs,
    )

    dataset2["toa_radiance"].rio.write_crs("EPSG:4326", inplace=True)

    return dataset2


def tanager_to_image(
    dataset,
    bands=None,
    wavelengths=None,
    method="nearest",
    row_range=None,
    col_range=None,
    output=None,
    **kwargs,
):
    """
    Converts an Tanager dataset to an image.

    Args:
        dataset (xarray.Dataset or str): The dataset containing the EMIT data or the file path to the dataset.
        bands (array-like, optional): The specific band indices to select. Defaults to None.
        wavelengths (array-like, optional): The specific wavelength values to select. Takes priority over bands. Defaults to None.
        method (str, optional): The method to use for data interpolation. Defaults to "nearest".
        row_range (tuple, optional): Row range (start_row, end_row) to subset the data. Defaults to None.
        col_range (tuple, optional): Column range (start_col, end_col) to subset the data. Defaults to None.
        output (str, optional): The file path where the image will be saved. If None, the image will be returned as a PIL Image object. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to `leafmap.array_to_image`.

    Returns:
        rasterio.Dataset or None: The image converted from the dataset. If `output` is provided, the image will be saved to the specified file and the function will return None.
    """
    from leafmap import array_to_image

    if isinstance(dataset, str):
        dataset = read_tanager(dataset, bands=bands)

    grid = grid_tanager(
        dataset,
        bands=bands,
        wavelengths=wavelengths,
        method=method,
        row_range=row_range,
        col_range=col_range,
    )

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
