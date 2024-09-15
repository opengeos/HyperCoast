# SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
#
# SPDX-License-Identifier: MIT

"""The common module contains common functions and classes used by the other modules.
"""

import os
import leafmap
import xarray as xr
from typing import List, Union, Dict, Optional, Tuple, Any


def github_raw_url(url: str) -> str:
    """Get the raw URL for a GitHub file.

    Args:
        url (str): The GitHub URL.
    Returns:
        str: The raw URL.
    """
    if isinstance(url, str) and url.startswith("https://github.com/") and "blob" in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace(
            "blob/", ""
        )
    return url


def download_file(
    url: Optional[str] = None,
    output: Optional[str] = None,
    quiet: Optional[bool] = True,
    proxy: Optional[str] = None,
    speed: Optional[float] = None,
    use_cookies: Optional[bool] = True,
    verify: Optional[bool] = True,
    uid: Optional[str] = None,
    fuzzy: Optional[bool] = False,
    resume: Optional[bool] = False,
    unzip: Optional[bool] = True,
    overwrite: Optional[bool] = False,
    subfolder: Optional[bool] = False,
) -> str:
    """Download a file from URL, including Google Drive shared URL.

    Args:
        url (str, optional): Google Drive URL is also supported. Defaults to None.
        output (str, optional): Output filename. Default is basename of URL.
        quiet (bool, optional): Suppress terminal output. Default is True.
        proxy (str, optional): Proxy. Defaults to None.
        speed (float, optional): Download byte size per second (e.g., 256KB/s = 256 * 1024). Defaults to None.
        use_cookies (bool, optional): Flag to use cookies. Defaults to True.
        verify (bool | str, optional): Either a bool, in which case it controls whether the server's TLS certificate is verified, or a string,
            in which case it must be a path to a CA bundle to use. Default is True.. Defaults to True.
        uid (str, optional): Google Drive's file ID. Defaults to None.
        fuzzy (bool, optional): Fuzzy extraction of Google Drive's file Id. Defaults to False.
        resume (bool, optional): Resume the download from existing tmp file if possible. Defaults to False.
        unzip (bool, optional): Unzip the file. Defaults to True.
        overwrite (bool, optional): Overwrite the file if it already exists. Defaults to False.
        subfolder (bool, optional): Create a subfolder with the same name as the file. Defaults to False.

    Returns:
        str: The output file path.
    """
    import zipfile
    import tarfile
    import gdown

    if output is None:
        if isinstance(url, str) and url.startswith("http"):
            output = os.path.basename(url)

    out_dir = os.path.abspath(os.path.dirname(output))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if isinstance(url, str):
        if os.path.exists(os.path.abspath(output)) and (not overwrite):
            print(
                f"{output} already exists. Skip downloading. Set overwrite=True to overwrite."
            )
            return os.path.abspath(output)
        else:
            url = github_raw_url(url)

    if "https://drive.google.com/file/d/" in url:
        fuzzy = True

    output = gdown.download(
        url, output, quiet, proxy, speed, use_cookies, verify, uid, fuzzy, resume
    )

    if unzip:
        if output.endswith(".zip"):
            with zipfile.ZipFile(output, "r") as zip_ref:
                if not quiet:
                    print("Extracting files...")
                if subfolder:
                    basename = os.path.splitext(os.path.basename(output))[0]

                    output = os.path.join(out_dir, basename)
                    if not os.path.exists(output):
                        os.makedirs(output)
                    zip_ref.extractall(output)
                else:
                    zip_ref.extractall(os.path.dirname(output))
        elif output.endswith(".tar.gz") or output.endswith(".tar"):
            if output.endswith(".tar.gz"):
                mode = "r:gz"
            else:
                mode = "r"

            with tarfile.open(output, mode) as tar_ref:
                if not quiet:
                    print("Extracting files...")
                if subfolder:
                    basename = os.path.splitext(os.path.basename(output))[0]
                    output = os.path.join(out_dir, basename)
                    if not os.path.exists(output):
                        os.makedirs(output)
                    tar_ref.extractall(output)
                else:
                    tar_ref.extractall(os.path.dirname(output))

    return os.path.abspath(output)


def netcdf_groups(filepath: str) -> List[str]:
    """
    Get the list of groups in a NetCDF file.

    Args:
        filepath (str): The path to the NetCDF file.

    Returns:
        list: A list of group names in the NetCDF file.

    Example:
        >>> netcdf_groups('path/to/netcdf/file')
        ['group1', 'group2', 'group3']
    """
    import h5netcdf

    with h5netcdf.File(filepath) as file:
        groups = list(file)
    return groups


def search_datasets(count: int = -1, **kwargs: Any) -> List[Dict[str, Any]]:
    """
    Searches for datasets using the EarthAccess API with optional filters.

    This function wraps the `earthaccess.search_datasets` function, allowing for
    customized search queries based on a count limit and additional keyword arguments
    which serve as filters for the search.

    Args:
        count (int, optional): The maximum number of datasets to return. A value of -1
            indicates no limit. Defaults to -1.
        **kwargs (Any): Additional keyword arguments to pass as search filters to the
            EarthAccess API.
            keyword: case-insensitive and supports wildcards ? and *
            short_name: e.g. ATL08
            doi: DOI for a dataset
            daac: e.g. NSIDC or PODAAC
            provider: particular to each DAAC, e.g. POCLOUD, LPDAAC etc.
            temporal: a tuple representing temporal bounds in the form (date_from, date_to)
            bounding_box: a tuple representing spatial bounds in the form
            (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat)


    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary contains
            information about a dataset found in the search.

    Example:
        >>> results = search_datasets(count=5, keyword='temperature')
        >>> print(results)
    """

    import earthaccess

    return earthaccess.search_datasets(count=count, **kwargs)


def search_nasa_data(
    count: int = -1,
    short_name: Optional[str] = None,
    bbox: Optional[List[float]] = None,
    temporal: Optional[str] = None,
    version: Optional[str] = None,
    doi: Optional[str] = None,
    daac: Optional[str] = None,
    provider: Optional[str] = None,
    output: Optional[str] = None,
    crs: str = "EPSG:4326",
    return_gdf: bool = False,
    **kwargs,
) -> Union[List[dict], tuple]:
    """Searches for NASA Earthdata granules.

    Args:
        count (int, optional): The number of granules to retrieve. Defaults to -1 (retrieve all).
        short_name (str, optional): The short name of the dataset.
        bbox (List[float], optional): The bounding box coordinates [xmin, ymin, xmax, ymax].
        temporal (str, optional): The temporal extent of the data.
        version (str, optional): The version of the dataset.
        doi (str, optional): The Digital Object Identifier (DOI) of the dataset.
        daac (str, optional): The Distributed Active Archive Center (DAAC) of the dataset.
        provider (str, optional): The provider of the dataset.
        output (str, optional): The output file path to save the GeoDataFrame as a file.
        crs (str, optional): The coordinate reference system (CRS) of the GeoDataFrame. Defaults to "EPSG:4326".
        return_gdf (bool, optional): Whether to return the GeoDataFrame in addition to the granules. Defaults to False.
        **kwargs: Additional keyword arguments for the earthaccess.search_data() function.

    Returns:
        Union[List[dict], tuple]: The retrieved granules. If return_gdf is True, also returns the resulting GeoDataFrame.
    """

    if isinstance(bbox, list):
        bbox = tuple(bbox)

    return leafmap.nasa_data_search(
        count=count,
        short_name=short_name,
        bbox=bbox,
        temporal=temporal,
        version=version,
        doi=doi,
        daac=daac,
        provider=provider,
        output=output,
        crs=crs,
        return_gdf=return_gdf,
        **kwargs,
    )


def download_nasa_data(
    granules: List[dict],
    out_dir: Optional[str] = None,
    provider: Optional[str] = None,
    threads: int = 8,
) -> None:
    """Downloads NASA Earthdata granules.

    Args:
        granules (List[dict]): The granules to download.
        out_dir (str, optional): The output directory where the granules will be downloaded. Defaults to None (current directory).
        provider (str, optional): The provider of the granules.
        threads (int, optional): The number of threads to use for downloading. Defaults to 8.
    """

    leafmap.nasa_data_download(
        granules=granules, out_dir=out_dir, provider=provider, threads=threads
    )


def search_pace(
    bbox: Optional[List[float]] = None,
    temporal: Optional[str] = None,
    count: int = -1,
    short_name: Optional[str] = "PACE_OCI_L2_AOP_NRT",
    output: Optional[str] = None,
    crs: str = "EPSG:4326",
    return_gdf: bool = False,
    **kwargs,
) -> Union[List[dict], tuple]:
    """Searches for NASA PACE granules.

    Args:
        bbox (List[float], optional): The bounding box coordinates [xmin, ymin, xmax, ymax].
        temporal (str, optional): The temporal extent of the data.
        count (int, optional): The number of granules to retrieve. Defaults to -1 (retrieve all).
        short_name (str, optional): The short name of the dataset. Defaults to "PACE_OCI_L2_AOP_NRT".
        output (str, optional): The output file path to save the GeoDataFrame as a file.
        crs (str, optional): The coordinate reference system (CRS) of the GeoDataFrame. Defaults to "EPSG:4326".
        return_gdf (bool, optional): Whether to return the GeoDataFrame in addition to the granules. Defaults to False.
        **kwargs: Additional keyword arguments for the earthaccess.search_data() function.

    Returns:
        Union[List[dict], tuple]: The retrieved granules. If return_gdf is True, also returns the resulting GeoDataFrame.
    """

    return search_nasa_data(
        count=count,
        short_name=short_name,
        bbox=bbox,
        temporal=temporal,
        output=output,
        crs=crs,
        return_gdf=return_gdf,
        **kwargs,
    )


def search_pace_chla(
    bbox: Optional[List[float]] = None,
    temporal: Optional[str] = None,
    count: int = -1,
    short_name: Optional[str] = "PACE_OCI_L3M_CHL_NRT",
    granule_name: Optional[str] = "*.DAY.*.0p1deg.*",
    output: Optional[str] = None,
    crs: str = "EPSG:4326",
    return_gdf: bool = False,
    **kwargs,
) -> Union[List[dict], tuple]:
    """Searches for NASA PACE Chlorophyll granules.

    Args:
        bbox (List[float], optional): The bounding box coordinates [xmin, ymin, xmax, ymax].
        temporal (str, optional): The temporal extent of the data.
        count (int, optional): The number of granules to retrieve. Defaults to -1 (retrieve all).
        short_name (str, optional): The short name of the dataset. Defaults to "PACE_OCI_L3M_CHL_NRT".
        output (str, optional): The output file path to save the GeoDataFrame as a file.
        crs (str, optional): The coordinate reference system (CRS) of the GeoDataFrame. Defaults to "EPSG:4326".
        return_gdf (bool, optional): Whether to return the GeoDataFrame in addition to the granules. Defaults to False.
        **kwargs: Additional keyword arguments for the earthaccess.search_data() function.

    Returns:
        Union[List[dict], tuple]: The retrieved granules. If return_gdf is True, also returns the resulting GeoDataFrame.
    """

    return search_nasa_data(
        count=count,
        short_name=short_name,
        bbox=bbox,
        temporal=temporal,
        granule_name=granule_name,
        output=output,
        crs=crs,
        return_gdf=return_gdf,
        **kwargs,
    )


def search_emit(
    bbox: Optional[List[float]] = None,
    temporal: Optional[str] = None,
    count: int = -1,
    short_name: Optional[str] = "EMITL2ARFL",
    output: Optional[str] = None,
    crs: str = "EPSG:4326",
    return_gdf: bool = False,
    **kwargs,
) -> Union[List[dict], tuple]:
    """Searches for NASA EMIT granules.

    Args:
        bbox (List[float], optional): The bounding box coordinates [xmin, ymin, xmax, ymax].
        temporal (str, optional): The temporal extent of the data.
        count (int, optional): The number of granules to retrieve. Defaults to -1 (retrieve all).
        short_name (str, optional): The short name of the dataset. Defaults to "EMITL2ARFL".
        output (str, optional): The output file path to save the GeoDataFrame as a file.
        crs (str, optional): The coordinate reference system (CRS) of the GeoDataFrame. Defaults to "EPSG:4326".
        return_gdf (bool, optional): Whether to return the GeoDataFrame in addition to the granules. Defaults to False.
        **kwargs: Additional keyword arguments for the earthaccess.search_data() function.

    Returns:
        Union[List[dict], tuple]: The retrieved granules. If return_gdf is True, also returns the resulting GeoDataFrame.
    """

    return search_nasa_data(
        count=count,
        short_name=short_name,
        bbox=bbox,
        temporal=temporal,
        output=output,
        crs=crs,
        return_gdf=return_gdf,
        **kwargs,
    )


def search_ecostress(
    bbox: Optional[List[float]] = None,
    temporal: Optional[str] = None,
    count: int = -1,
    short_name: Optional[str] = "ECO_L2T_LSTE",
    output: Optional[str] = None,
    crs: str = "EPSG:4326",
    return_gdf: bool = False,
    **kwargs,
) -> Union[List[dict], tuple]:
    """Searches for NASA ECOSTRESS granules.

    Args:
        bbox (List[float], optional): The bounding box coordinates [xmin, ymin, xmax, ymax].
        temporal (str, optional): The temporal extent of the data.
        count (int, optional): The number of granules to retrieve. Defaults to -1 (retrieve all).
        short_name (str, optional): The short name of the dataset. Defaults to "ECO_L2T_LSTE".
        output (str, optional): The output file path to save the GeoDataFrame as a file.
        crs (str, optional): The coordinate reference system (CRS) of the GeoDataFrame. Defaults to "EPSG:4326".
        return_gdf (bool, optional): Whether to return the GeoDataFrame in addition to the granules. Defaults to False.
        **kwargs: Additional keyword arguments for the earthaccess.search_data() function.

    Returns:
        Union[List[dict], tuple]: The retrieved granules. If return_gdf is True, also returns the resulting GeoDataFrame.
    """

    return search_nasa_data(
        count=count,
        short_name=short_name,
        bbox=bbox,
        temporal=temporal,
        output=output,
        crs=crs,
        return_gdf=return_gdf,
        **kwargs,
    )


def download_pace(
    granules: List[dict],
    out_dir: Optional[str] = None,
    threads: int = 8,
) -> None:
    """Downloads NASA PACE granules.

    Args:
        granules (List[dict]): The granules to download.
        out_dir (str, optional): The output directory where the granules will be
            downloaded. Defaults to None (current directory).
        threads (int, optional): The number of threads to use for downloading.
            Defaults to 8.
    """

    download_nasa_data(granules=granules, out_dir=out_dir, threads=threads)


def download_emit(
    granules: List[dict],
    out_dir: Optional[str] = None,
    threads: int = 8,
) -> None:
    """Downloads NASA EMIT granules.

    Args:
        granules (List[dict]): The granules to download.
        out_dir (str, optional): The output directory where the granules will be
            downloaded. Defaults to None (current directory).
        threads (int, optional): The number of threads to use for downloading.
            Defaults to 8.
    """

    download_nasa_data(granules=granules, out_dir=out_dir, threads=threads)


def download_ecostress(
    granules: List[dict],
    out_dir: Optional[str] = None,
    threads: int = 8,
) -> None:
    """Downloads NASA ECOSTRESS granules.

    Args:
        granules (List[dict]): The granules to download.
        out_dir (str, optional): The output directory where the granules will be
            downloaded. Defaults to None (current directory).
        threads (int, optional): The number of threads to use for downloading.
            Defaults to 8.
    """

    download_nasa_data(granules=granules, out_dir=out_dir, threads=threads)


def nasa_earth_login(strategy: str = "all", persist: bool = True, **kwargs) -> None:
    """Logs in to NASA Earthdata.

    Args:
        strategy (str, optional): The login strategy. Defaults to "all".
        persist (bool, optional): Whether to persist the login. Defaults to True.
    """
    from leafmap import get_api_key
    import earthaccess

    USERNAME = get_api_key("EARTHDATA_USERNAME")
    PASSWORD = get_api_key("EARTHDATA_PASSWORD")
    if (USERNAME is not None) and (PASSWORD is not None):
        strategy = "environment"
    earthaccess.login(strategy=strategy, persist=persist, **kwargs)


def convert_coords(
    coords: List[Tuple[float, float]], from_epsg: str, to_epsg: str
) -> List[Tuple[float, float]]:
    """
    Convert a list of coordinates from one EPSG to another.

    Args:
        coords: List of tuples containing coordinates in the format (latitude, longitude).
        from_epsg: Source EPSG code (default is "epsg:4326").
        to_epsg: Target EPSG code (default is "epsg:32615").

    Returns:
        List of tuples containing converted coordinates in the format (x, y).
    """
    import pyproj

    # Define the coordinate transformation
    transformer = pyproj.Transformer.from_crs(from_epsg, to_epsg, always_xy=True)

    # Convert each coordinate
    converted_coords = [transformer.transform(lon, lat) for lat, lon in coords]

    return converted_coords


def image_cube(
    dataset,
    variable: str = "reflectance",
    cmap: str = "jet",
    clim: Tuple[float, float] = (0, 0.5),
    title: str = "Reflectance",
    rgb_bands: Optional[List[int]] = None,
    rgb_wavelengths: Optional[List[float]] = None,
    rgb_gamma: float = 1.0,
    rgb_cmap: Optional[str] = None,
    rgb_clim: Optional[Tuple[float, float]] = None,
    rgb_args: Dict[str, Any] = None,
    widget=None,
    plotter_args: Dict[str, Any] = None,
    show_axes: bool = True,
    grid_origin=(0, 0, 0),
    grid_spacing=(1, 1, 1),
    **kwargs: Any,
):
    """
    Creates an image cube from a dataset and plots it using PyVista.

    Args:
        dataset (Union[str, xr.Dataset]): The dataset to plot. Can be a path to
            a NetCDF file or an xarray Dataset.
        variable (str, optional): The variable to plot. Defaults to "reflectance".
        cmap (str, optional): The colormap to use. Defaults to "jet".
        clim (Tuple[float, float], optional): The color limits. Defaults to (0, 0.5).
        title (str, optional): The title for the scalar bar. Defaults to "Reflectance".
        rgb_bands (Optional[List[int]], optional): The bands to use for the RGB
            image. Defaults to None.
        rgb_wavelengths (Optional[List[float]], optional): The wavelengths to
            use for the RGB image. Defaults to None.
        rgb_gamma (float, optional): The gamma correction for the RGB image.
            Defaults to 1.
        rgb_cmap (Optional[str], optional): The colormap to use for the RGB image.
            Defaults to None.
        rgb_clim (Optional[Tuple[float, float]], optional): The color limits for
            the RGB image. Defaults to None.
        rgb_args (Dict[str, Any], optional): Additional arguments for the
            `add_mesh` method for the RGB image. Defaults to {}.
        widget (Optional[str], optional): The widget to use for the image cube.
            Can be one of the following: "box", "plane", "slice", "orthogonal",
            and "threshold". Defaults to None.
        plotter_args (Dict[str, Any], optional): Additional arguments for the
            `pv.Plotter` constructor. Defaults to {}.
        show_axes (bool, optional): Whether to show the axes. Defaults to True.
        grid_origin (Tuple[float, float, float], optional): The origin of the grid.
            Defaults to (0, 0, 0).
        grid_spacing (Tuple[float, float, float], optional): The spacing of the grid.
        **kwargs (Dict[str, Any], optional): Additional arguments for the
            `add_mesh` method. Defaults to {}.

    Returns:
        pv.Plotter: The PyVista Plotter with the image cube added.
    """

    import pyvista as pv

    if rgb_args is None:
        rgb_args = {}

    if plotter_args is None:
        plotter_args = {}

    allowed_widgets = ["box", "plane", "slice", "orthogonal", "threshold"]

    if widget is not None:
        if widget not in allowed_widgets:
            raise ValueError(f"widget must be one of the following: {allowed_widgets}")

    if isinstance(dataset, str):
        dataset = xr.open_dataset(dataset)

    da = dataset[variable]  # xarray DataArray
    values = da.to_numpy()

    # Create the spatial reference for the image cube
    grid = pv.ImageData()

    # Set the grid dimensions: shape because we want to inject our values on the POINT data
    grid.dimensions = values.shape

    # Edit the spatial reference
    grid.origin = grid_origin  # The bottom left corner of the data set
    grid.spacing = grid_spacing  # These are the cell sizes along each axis

    # Add the data values to the cell data
    grid.point_data["values"] = values.flatten(order="F")  # Flatten the array

    # Plot the image cube with the RGB image overlay
    p = pv.Plotter(**plotter_args)

    if "scalar_bar_args" not in kwargs:
        kwargs["scalar_bar_args"] = {"title": title}
    else:
        kwargs["scalar_bar_args"]["title"] = title

    if "show_edges" not in kwargs:
        kwargs["show_edges"] = False

    if widget == "box":
        p.add_mesh_clip_box(grid, cmap=cmap, clim=clim, **kwargs)
    elif widget == "plane":
        if "normal" not in kwargs:
            kwargs["normal"] = (0, 0, 1)
        if "invert" not in kwargs:
            kwargs["invert"] = True
        if "normal_rotation" not in kwargs:
            kwargs["normal_rotation"] = False
        p.add_mesh_clip_plane(grid, cmap=cmap, clim=clim, **kwargs)
    elif widget == "slice":
        if "normal" not in kwargs:
            kwargs["normal"] = (0, 0, 1)
        if "normal_rotation" not in kwargs:
            kwargs["normal_rotation"] = False
        p.add_mesh_slice(grid, cmap=cmap, clim=clim, **kwargs)
    elif widget == "orthogonal":
        p.add_mesh_slice_orthogonal(grid, cmap=cmap, clim=clim, **kwargs)
    elif widget == "threshold":
        p.add_mesh_threshold(grid, cmap=cmap, clim=clim, **kwargs)
    else:
        p.add_mesh(grid, cmap=cmap, clim=clim, **kwargs)

    if rgb_bands is not None or rgb_wavelengths is not None:

        if rgb_bands is not None:
            rgb_image = dataset.isel(wavelength=rgb_bands, method="nearest")[
                variable
            ].to_numpy()
        elif rgb_wavelengths is not None:
            rgb_image = dataset.sel(wavelength=rgb_wavelengths, method="nearest")[
                variable
            ].to_numpy()

        x_dim, y_dim = rgb_image.shape[0], rgb_image.shape[1]
        z_dim = 1
        im = pv.ImageData(dimensions=(x_dim, y_dim, z_dim))

        # Add scalar data, you may also need to flatten this
        im.point_data["rgb_image"] = (
            rgb_image.reshape(-1, rgb_image.shape[2], order="F") * rgb_gamma
        )

        grid_z_max = grid.bounds[5]
        im.origin = (0, 0, grid_z_max)

        if rgb_image.shape[2] < 3:
            if rgb_cmap is None:
                rgb_cmap = cmap
            if rgb_clim is None:
                rgb_clim = clim

            if "cmap" not in rgb_args:
                rgb_args["cmap"] = rgb_cmap
            if "clim" not in rgb_args:
                rgb_args["clim"] = rgb_clim
        else:
            if "rgb" not in rgb_args:
                rgb_args["rgb"] = True

        if "show_scalar_bar" not in rgb_args:
            rgb_args["show_scalar_bar"] = False
        if "show_edges" not in rgb_args:
            rgb_args["show_edges"] = False

        p.add_mesh(im, **rgb_args)

    if show_axes:
        p.show_axes()

    return p


def open_dataset(
    filename: str,
    engine: Optional[str] = None,
    chunks: Optional[Dict[str, int]] = None,
    **kwargs: Any,
) -> xr.Dataset:
    """
    Opens and returns an xarray Dataset from a file.

    This function is a wrapper around `xarray.open_dataset` that allows for additional
    customization through keyword arguments.

    Args:
        filename (str): Path to the file to open.
        engine (Optional[str]): Name of the engine to use for reading the file. If None, xarray's
            default engine is used. Examples include 'netcdf4', 'h5netcdf', 'zarr', etc.
        chunks (Optional[Dict[str, int]]): Dictionary specifying how to chunk the dataset along each dimension.
            For example, `{'time': 1}` would load the dataset in single-time-step chunks. If None,
            the dataset is not chunked.
        **kwargs: Additional keyword arguments passed to `xarray.open_dataset`.

    Returns:
        xr.Dataset: The opened dataset.

    Examples:
        Open a NetCDF file without chunking:
        >>> dataset = open_dataset('path/to/file.nc')

        Open a Zarr dataset, chunking along the 'time' dimension:
        >>> dataset = open_dataset('path/to/dataset.zarr', engine='zarr', chunks={'time': 10})
    """

    try:
        dataset = xr.open_dataset(filename, engine=engine, chunks=chunks, **kwargs)
    except OSError:
        dataset = xr.open_dataset(filename, engine="h5netcdf", chunks=chunks, **kwargs)

    return dataset


def extract_date_from_filename(filename: str):
    """
    Extracts a date from a filename assuming the date is in 'YYYYMMDD' format.

    This function searches the filename for a sequence of 8 digits that represent a date in
    'YYYYMMDD' format. If such a sequence is found, it converts the sequence into a pandas
    Timestamp object. If no such sequence is found, the function returns None.

    Args:
        filename (str): The filename from which to extract the date.

    Returns:
        Optional[pd.Timestamp]: A pandas Timestamp object representing the date found in the filename,
        or None if no date in 'YYYYMMDD' format is found.

    Examples:
        >>> extract_date_from_filename("example_20230101.txt")
        Timestamp('2023-01-01 00:00:00')

        >>> extract_date_from_filename("no_date_in_this_filename.txt")
        None
    """
    import re
    import pandas as pd

    # Assuming the date format in filename is 'YYYYMMDD'
    date_match = re.search(r"\d{8}", filename)
    if date_match:
        return pd.to_datetime(date_match.group(), format="%Y%m%d")
    else:
        return None


def extract_spectral(
    ds: xr.Dataset, lat: float, lon: float, name: str = "data"
) -> xr.DataArray:
    """
    Extracts spectral signature from a given xarray Dataset.

    Args:
        ds (xarray.Dataset): The dataset containing the spectral data.
        lat (float): The latitude of the point to extract.
        lon (float): The longitude of the point to extract.

    Returns:
        xarray.DataArray: The extracted data.
    """

    crs = ds.rio.crs

    x, y = convert_coords([[lat, lon]], "epsg:4326", crs)[0]

    values = ds.sel(x=x, y=y, method="nearest")[name].values

    da = xr.DataArray(values, dims=["band"], coords={"band": ds.coords["band"]})

    return da


def download_acolite(outdir: str = ".", platform: Optional[str] = None) -> str:
    """
    Downloads the Acolite release based on the OS platform and extracts it to the specified directory.
    For more information, see the Acolite manual https://github.com/acolite/acolite/releases.

    Args:
        outdir (str): The output directory where the file will be Acolite and extracted.
        platform (Optional[str]): The platform for which to download acolite. If None, the current system platform is used.
                                  Valid values are 'linux', 'darwin', and 'windows'.

    Returns:
        str: The path to the extracted Acolite directory.

    Raises:
        Exception: If the platform is unsupported or the download fails.
    """
    import platform as pf
    import requests
    import tarfile
    from tqdm import tqdm

    base_url = "https://github.com/acolite/acolite/releases/download/20231023.0/"

    if platform is None:
        platform = pf.system().lower()
    else:
        platform = platform.lower()

    if platform == "linux":
        download_url = base_url + "acolite_py_linux_20231023.0.tar.gz"
        root_dir = "acolite_py_linux"
    elif platform == "darwin":
        download_url = base_url + "acolite_py_mac_20231023.0.tar.gz"
        root_dir = "acolite_py_mac"
    elif platform == "windows":
        download_url = base_url + "acolite_py_win_20231023.0.tar.gz"
        root_dir = "acolite_py_win"
    else:
        print(f"Unsupported OS platform: {platform}")
        return

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    extracted_path = os.path.join(outdir, root_dir)
    file_name = os.path.join(outdir, download_url.split("/")[-1])

    if os.path.exists(file_name):
        print(f"{file_name} already exists. Skip downloading.")
        return extracted_path

    response = requests.get(download_url, stream=True, timeout=60)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192

    if response.status_code == 200:
        with open(file_name, "wb") as file, tqdm(
            desc=file_name,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))
        print(f"Downloaded {file_name}")
    else:
        print(f"Failed to download file from {download_url}")
        return

    # Unzip the file
    with tarfile.open(file_name, "r:gz") as tar:
        tar.extractall(path=outdir)

    print(f"Extracted to {extracted_path}")
    return extracted_path


def run_acolite(
    acolite_dir: str,
    settings_file: Optional[str] = None,
    input_file: Optional[str] = None,
    out_dir: Optional[str] = None,
    polygon: Optional[str] = None,
    l2w_parameters: Optional[str] = None,
    rgb_rhot: bool = True,
    rgb_rhos: bool = True,
    map_l2w: bool = True,
    verbose: bool = True,
    **kwargs: Any,
) -> None:
    """
    Runs the Acolite software for atmospheric correction and water quality retrieval.
    For more information, see the Acolite manual https://github.com/acolite/acolite/releases

    This function constructs and executes a command to run the Acolite software with the specified
    parameters. It supports running Acolite with a settings file or with individual parameters
    specified directly. Additional parameters can be passed as keyword arguments.

    Args:
        acolite_dir (str): The directory where Acolite is installed.
        settings_file (Optional[str], optional): The path to the Acolite settings file. If provided,
            other parameters except `verbose` are ignored. Defaults to None.
        input_file (Optional[str], optional): The path to the input file for processing. Defaults to None.
        out_dir (Optional[str], optional): The directory where output files will be saved. Defaults to None.
        polygon (Optional[str], optional): The path to a polygon file for spatial subset. Defaults to None.
        l2w_parameters (Optional[str], optional): Parameters for L2W processing. Defaults to None.
        rgb_rhot (bool, optional): Flag to generate RGB images using rhot. Defaults to True.
        rgb_rhos (bool, optional): Flag to generate RGB images using rhos. Defaults to True.
        map_l2w (bool, optional): Flag to map L2W products. Defaults to True.
        verbose (bool, optional): If True, prints the command output; otherwise, suppresses it. Defaults to True.
        **kwargs (Any): Additional command line arguments to pass to acolite. Such as
            --l2w_export_geotiff, --merge_tiles, etc.

    Returns:
        None: This function does not return a value. It executes the Acolite software.

    Example:
        >>> run_acolite("/path/to/acolite", input_file="/path/to/inputfile", output="/path/to/output")
    """

    import subprocess
    from datetime import datetime

    def get_formatted_current_time(format_str="%Y-%m-%d %H:%M:%S"):
        current_time = datetime.now()
        formatted_time = current_time.strftime(format_str)
        return formatted_time

    acolite_dir_name = os.path.split(acolite_dir)[-1]
    acolite_exe = "acolite"
    if acolite_dir_name.endswith("win"):
        acolite_exe += ".exe"

    if isinstance(input_file, list):
        input_file = ",".join(input_file)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    acolite_exe_path = os.path.join(acolite_dir, "dist", "acolite", acolite_exe)
    acolite_exe_path = acolite_exe_path.replace("\\", "/")

    acolite_cmd = [acolite_exe_path, "--cli"]

    if settings_file is not None:
        acolite_cmd.extend(["--settings", settings_file])
    else:
        lines = []
        lines.append("## ACOLITE settings")
        lines.append(f"## Written at {get_formatted_current_time()}")
        if input_file is not None:
            input_file = input_file.replace("\\", "/")
            lines.append(f"inputfile={input_file}")
        if out_dir is not None:
            out_dir = out_dir.replace("\\", "/")
            lines.append(f"output={out_dir}")
        if polygon is not None:
            lines.append(f"polygon={polygon}")
        else:
            lines.append("polygon=None")
        if l2w_parameters is not None:
            lines.append(f"l2w_parameters={l2w_parameters}")
        if rgb_rhot:
            lines.append("rgb_rhot=True")
        else:
            lines.append("rgb_rhot=False")
        if rgb_rhos:
            lines.append("rgb_rhos=True")
        else:
            lines.append("rgb_rhos=False")
        if map_l2w:
            lines.append("map_l2w=True")
        else:
            lines.append("map_l2w=False")

        for key, value in kwargs.items():
            lines.append(f"{key}={value}")

        lines.append(f"runid={get_formatted_current_time('%Y%m%d_%H%M%S')}")
        settings_filename = f"acolite_run_{get_formatted_current_time('%Y%m%d_%H%M%S')}_settings_user.txt"
        settings_file = os.path.join(out_dir, settings_filename).replace("\\", "/")
        with open(settings_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        acolite_cmd.extend(["--settings", settings_file])

    if acolite_dir_name.endswith("win"):
        acolite_cmd = " ".join(acolite_cmd)

    if verbose:
        subprocess.run(acolite_cmd, check=True)
    else:
        subprocess.run(
            acolite_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )


def pca(input_file, output_file, n_components=3, **kwargs):
    """
    Performs Principal Component Analysis (PCA) on a dataset.

    Args:
        input_file (str): The input file containing the data to analyze.
        output_file (str): The output file to save the PCA results.
        n_components (int, optional): The number of principal components to compute. Defaults to 3.
        **kwargs: Additional keyword arguments to pass to the scikit-learn PCA function.
            For more info, see https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html.

    Returns:
        None: This function does not return a value. It saves the PCA results to the output file.
    """

    import rasterio
    from sklearn.decomposition import PCA

    # Function to load the GeoTIFF image
    def load_geotiff(file_path):
        with rasterio.open(file_path) as src:
            image = src.read()
            profile = src.profile
        return image, profile

    # Function to perform PCA
    def perform_pca(image, n_components=3, **kwargs):
        # Reshape the image to [n_bands, n_pixels]
        n_bands, width, height = image.shape
        image_reshaped = image.reshape(n_bands, width * height).T

        # Perform PCA
        model = PCA(n_components=n_components, **kwargs)
        principal_components = model.fit_transform(image_reshaped)

        # Reshape the principal components back to image dimensions
        pca_image = principal_components.T.reshape(n_components, width, height)
        return pca_image

    # Function to save the PCA-transformed image
    def save_geotiff(file_path, image, profile):
        profile.update(count=image.shape[0])
        with rasterio.open(file_path, "w", **profile) as dst:
            dst.write(image)

    image, profile = load_geotiff(input_file)
    pca_image = perform_pca(image, n_components, **kwargs)
    save_geotiff(output_file, pca_image, profile)


def show_field_data(
    data: Union[str],
    x_col: str = "wavelength",
    y_col_prefix: str = "(",
    x_label: str = "Wavelengths (nm)",
    y_label: str = "Reflectance",
    use_marker_cluster: bool = True,
    min_width: int = 400,
    max_width: int = 600,
    min_height: int = 200,
    max_height: int = 250,
    layer_name: str = "Marker Cluster",
    m: object = None,
    center: Tuple[float, float] = (20, 0),
    zoom: int = 2,
):
    """
    Displays field data on a map with interactive markers and popups showing time series data.

    Args:
        data (Union[str, pd.DataFrame]): Path to the CSV file or a pandas DataFrame containing the data.
        x_col (str): Column name to use for the x-axis of the charts. Default is "wavelength".
        y_col_prefix (str): Prefix to identify the columns that contain the location-specific data. Default is "(".
        x_label (str): Label for the x-axis of the charts. Default is "Wavelengths (nm)".
        y_label (str): Label for the y-axis of the charts. Default is "Reflectance".
        use_marker_cluster (bool): Whether to use marker clustering. Default is True.
        min_width (int): Minimum width of the popup. Default is 400.
        max_width (int): Maximum width of the popup. Default is 600.
        min_height (int): Minimum height of the popup. Default is 200.
        max_height (int): Maximum height of the popup. Default is 250.
        layer_name (str): Name of the marker cluster layer. Default is "Marker Cluster".
        m (Map, optional): An ipyleaflet Map instance to add the markers to. Default is None.
        center (Tuple[float, float]): Center of the map as a tuple of (latitude, longitude). Default is (20, 0).
        zoom (int): Zoom level of the map. Default is 2.

    Returns:
        Map: An ipyleaflet Map with the added markers and popups.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from ipyleaflet import Map, Marker, Popup, MarkerCluster
    from ipywidgets import Output, VBox

    # Read the CSV file
    if isinstance(data, str):
        data = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        pass
    else:
        raise ValueError("data must be a path to a CSV file or a pandas DataFrame")

    # Extract locations from columns
    locations = [col for col in data.columns if col.startswith(y_col_prefix)]
    coordinates = [tuple(map(float, loc.strip("()").split())) for loc in locations]

    # Create the map
    if m is None:
        m = Map(center=center, zoom=zoom)

    # Function to create the chart
    def create_chart(data, title):
        _, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size here
        ax.plot(data[x_col], data["values"])
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        output = Output()  # Adjust the output widget size here
        with output:
            plt.show()
        return output

    # Define a callback function to create and show the popup
    def callback_with_popup_creation(location, values):
        def f(**kwargs):
            marker_center = kwargs["coordinates"]
            output = create_chart(values, f"Location: {location}")
            popup = Popup(
                location=marker_center,
                child=VBox([output]),
                min_width=min_width,
                max_width=max_width,
                min_height=min_height,
                max_height=max_height,
            )
            m.add_layer(popup)

        return f

    markers = []

    # Add points to the map
    for i, coord in enumerate(coordinates):
        location = f"{coord}"
        values = pd.DataFrame({x_col: data[x_col], "values": data[locations[i]]})
        marker = Marker(location=coord, title=location, name=f"Marker {i + 1}")
        marker.on_click(callback_with_popup_creation(location, values))
        markers.append(marker)

    if use_marker_cluster:
        marker_cluster = MarkerCluster(markers=markers, name=layer_name)
        m.add_layer(marker_cluster)
    else:
        for marker in markers:
            m.add_layer(marker)

    return m
