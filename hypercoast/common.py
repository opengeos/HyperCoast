"""The common module contains common functions and classes used by the other modules.
"""

import os
import leafmap
import xarray as xr
from typing import List, Union, Dict, Optional, Tuple, Any


def github_raw_url(url):
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
    url=None,
    output=None,
    quiet=False,
    proxy=None,
    speed=None,
    use_cookies=True,
    verify=True,
    id=None,
    fuzzy=False,
    resume=False,
    unzip=True,
    overwrite=False,
    subfolder=False,
):
    """Download a file from URL, including Google Drive shared URL.

    Args:
        url (str, optional): Google Drive URL is also supported. Defaults to None.
        output (str, optional): Output filename. Default is basename of URL.
        quiet (bool, optional): Suppress terminal output. Default is False.
        proxy (str, optional): Proxy. Defaults to None.
        speed (float, optional): Download byte size per second (e.g., 256KB/s = 256 * 1024). Defaults to None.
        use_cookies (bool, optional): Flag to use cookies. Defaults to True.
        verify (bool | str, optional): Either a bool, in which case it controls whether the server's TLS certificate is verified, or a string,
            in which case it must be a path to a CA bundle to use. Default is True.. Defaults to True.
        id (str, optional): Google Drive's file ID. Defaults to None.
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
        url, output, quiet, proxy, speed, use_cookies, verify, id, fuzzy, resume
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

    leafmap.nasa_data_login(strategy=strategy, persist=persist, **kwargs)


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
    rgb_args: Dict[str, Any] = {},
    widget=None,
    plotter_args: Dict[str, Any] = {},
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
    import xarray as xr

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
    except Exception as e:
        dataset = xr.open_dataset(filename, engine="h5netcdf", chunks=chunks, **kwargs)

    return dataset
