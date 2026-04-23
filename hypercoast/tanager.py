"""Reader and helpers for Planet Tanager hyperspectral HDF5 products.

Supports all four published product variants via :func:`read_tanager`:
``basic_radiance_hdf5``, ``ortho_radiance_hdf5``, ``basic_sr_hdf5``, and
``ortho_sr_hdf5``. The reader auto-detects the HDF5 layout and sources
wavelength metadata from inside the file when available, falling back to a
caller-supplied STAC URL only when necessary.
"""

import warnings

import h5py
import xarray as xr
import numpy as np
import requests
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Optional, Any, Callable
from .common import download_file

# Recognized Planet Tanager products and their canonical output var names.
_PRODUCT_VAR_NAME = {
    "basic_radiance": "toa_radiance",
    "ortho_radiance": "toa_radiance",
    "basic_sr": "surface_reflectance",
    "ortho_sr": "surface_reflectance",
}

_PRODUCT_STAC_ASSET = {
    "basic_radiance": "basic_radiance_hdf5",
    "ortho_radiance": "ortho_radiance_hdf5",
    "basic_sr": "basic_sr_hdf5",
    "ortho_sr": "ortho_sr_hdf5",
}


def _ensure_nm(values):
    """Return a wavelength array in nanometers.

    Planet STAC expresses center wavelengths in micrometers while Tanager HDF5
    files typically store them in nanometers. This helper uses a magnitude
    heuristic: values with a maximum below 10 are assumed to be in micrometers
    and are scaled by 1000.

    Args:
        values (array-like): Wavelength values in either nm or um.

    Returns:
        numpy.ndarray: Wavelength values in nanometers.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size and np.nanmax(arr) < 10:
        arr = arr * 1000.0
    return arr


def _cube_name_to_product(name):
    """Map an HDF5 cube dataset name to a Tanager product identifier.

    Args:
        name (str): HDF5 dataset name such as ``toa_radiance`` or
            ``ortho_surface_reflectance``.

    Returns:
        str or None: One of ``basic_radiance``, ``ortho_radiance``,
        ``basic_sr``, ``ortho_sr``, or ``None`` if the name is not recognized.
    """
    lname = name.lower()
    if "ortho" in lname and ("reflect" in lname or "_sr" in lname):
        return "ortho_sr"
    if "ortho" in lname and "radiance" in lname:
        return "ortho_radiance"
    if "reflect" in lname or lname.endswith("_sr"):
        return "basic_sr"
    if "radiance" in lname:
        return "basic_radiance"
    return None


def _discover_tanager_layout(h5_file, product=None):
    """Inspect a Tanager HDF5 file and return its layout.

    Detection cascades through (1) the canonical HDFEOS SWATHS layout used by
    the ``basic_radiance_hdf5`` and ``basic_sr_hdf5`` products, and (2) a
    generic sweep with ``h5py.visititems`` that picks the largest 3-D float
    cube whose band axis length falls in the Tanager-plausible range. Latitude
    and longitude are resolved similarly, with a preference for
    ``Geolocation Fields``.

    Args:
        h5_file (h5py.File): An open HDF5 file handle.
        product (str, optional): If supplied, forces the product variant to one
            of ``basic_radiance``, ``ortho_radiance``, ``basic_sr``, or
            ``ortho_sr``. When forced, the corresponding data-field name
            (``toa_radiance`` or ``surface_reflectance``) is required at the
            canonical HDFEOS path.

    Returns:
        dict: A layout descriptor with keys ``product``, ``data_path``,
        ``data_var_name``, ``lat_path``, ``lon_path``, ``wavelength_path``,
        ``fwhm_path``, ``scale_factor``, ``add_offset``, ``fill_value``,
        ``band_axis``, and ``stac_asset_key``.

    Raises:
        ValueError: If no 3-D data cube can be located in the file.
    """
    hdfeos_root = "HDFEOS/SWATHS/HYP"
    df_root = f"{hdfeos_root}/Data Fields"
    gf_root = f"{hdfeos_root}/Geolocation Fields"

    data_path = None
    cube_name = None

    candidate_names_by_product = {
        "basic_radiance": ["toa_radiance"],
        "ortho_radiance": ["toa_radiance", "ortho_radiance"],
        "basic_sr": ["surface_reflectance"],
        "ortho_sr": ["surface_reflectance", "ortho_surface_reflectance"],
    }

    if product is not None:
        if product not in _PRODUCT_VAR_NAME:
            raise ValueError(
                f"Unknown Tanager product '{product}'. Expected one of "
                f"{sorted(_PRODUCT_VAR_NAME)}."
            )
        for candidate in candidate_names_by_product[product]:
            path = f"{df_root}/{candidate}"
            if path in h5_file:
                data_path = path
                cube_name = candidate
                break

    if data_path is None:
        for candidate in (
            "toa_radiance",
            "surface_reflectance",
            "ortho_radiance",
            "ortho_surface_reflectance",
        ):
            path = f"{df_root}/{candidate}"
            if path in h5_file:
                data_path = path
                cube_name = candidate
                break

    if data_path is None:
        best = {"size": -1, "path": None, "name": None, "shape": None}

        def _visitor(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            if obj.ndim != 3 or not np.issubdtype(obj.dtype, np.floating):
                return
            band_axis = int(np.argmin(obj.shape))
            n_bands = obj.shape[band_axis]
            if not (100 <= n_bands <= 600):
                return
            if obj.size > best["size"]:
                best["size"] = obj.size
                best["path"] = name
                best["name"] = name.rsplit("/", 1)[-1]
                best["shape"] = obj.shape

        h5_file.visititems(_visitor)
        if best["path"] is None:
            raise ValueError(
                "No 3-D hyperspectral cube found in the Tanager HDF5 file."
            )
        data_path = best["path"]
        cube_name = best["name"]

    if product is None:
        product = _cube_name_to_product(cube_name) or "basic_radiance"

    data_var_name = _PRODUCT_VAR_NAME[product]

    cube = h5_file[data_path]
    if data_path.startswith(f"{hdfeos_root}/"):
        # HDFEOS SWATHS products store cubes in (band, y, x) order.
        band_axis = 0
    else:
        band_axis = int(np.argmin(cube.shape))

    lat_path = f"{gf_root}/Latitude" if f"{gf_root}/Latitude" in h5_file else None
    lon_path = f"{gf_root}/Longitude" if f"{gf_root}/Longitude" in h5_file else None
    if lat_path is None or lon_path is None:
        found_lat = []
        found_lon = []

        def _geo_visitor(name, obj):
            if not isinstance(obj, h5py.Dataset) or obj.ndim != 2:
                return
            leaf = name.rsplit("/", 1)[-1].lower()
            if leaf in ("latitude", "lat"):
                found_lat.append(name)
            elif leaf in ("longitude", "lon"):
                found_lon.append(name)

        h5_file.visititems(_geo_visitor)
        lat_path = lat_path or (found_lat[0] if found_lat else None)
        lon_path = lon_path or (found_lon[0] if found_lon else None)

    wl_path = None
    fwhm_path = None
    for candidate in (
        f"{df_root}/Wavelength",
        f"{df_root}/Wavelengths",
        f"{df_root}/wavelength",
        f"{df_root}/wavelengths",
        f"{hdfeos_root}/Wavelength",
        "Wavelengths",
        "Wavelength",
    ):
        if candidate in h5_file:
            wl_path = candidate
            break
    for candidate in (
        f"{df_root}/FWHM",
        f"{df_root}/fwhm",
        f"{hdfeos_root}/FWHM",
        "FWHM",
    ):
        if candidate in h5_file:
            fwhm_path = candidate
            break

    def _attr(obj, *names, default=None):
        for n in names:
            if n in obj.attrs:
                return obj.attrs[n]
        return default

    scale_factor = float(_attr(cube, "scale_factor", default=1.0))
    add_offset = float(_attr(cube, "add_offset", default=0.0))
    fill_value = _attr(cube, "_FillValue", "FillValue", "missing_value", default=None)
    if fill_value is not None:
        try:
            fill_value = float(fill_value)
        except (TypeError, ValueError):
            fill_value = None

    return {
        "product": product,
        "data_path": data_path,
        "data_var_name": data_var_name,
        "lat_path": lat_path,
        "lon_path": lon_path,
        "wavelength_path": wl_path,
        "fwhm_path": fwhm_path,
        "scale_factor": scale_factor,
        "add_offset": add_offset,
        "fill_value": fill_value,
        "band_axis": band_axis,
        "stac_asset_key": _PRODUCT_STAC_ASSET[product],
    }


def _read_wavelengths_from_hdf5(h5_file, layout, n_bands):
    """Read wavelengths and FWHM from inside the HDF5 file.

    Args:
        h5_file (h5py.File): An open HDF5 file handle.
        layout (dict): Layout descriptor from ``_discover_tanager_layout``.
        n_bands (int): Expected number of bands; a match is required.

    Returns:
        tuple: ``(wavelengths_nm, fwhm_nm)`` as numpy arrays, or ``(None, None)``
        if no matching in-file metadata is found.
    """
    wl_nm = None
    fwhm_nm = None

    wl_path = layout.get("wavelength_path")
    if wl_path and wl_path in h5_file:
        vals = np.asarray(h5_file[wl_path][()], dtype=float).ravel()
        if vals.size == n_bands:
            wl_nm = _ensure_nm(vals)

    if wl_nm is None:
        cube = h5_file[layout["data_path"]]
        for attr_name in (
            "wavelengths",
            "center_wavelengths",
            "band_center_wavelengths",
        ):
            if attr_name in cube.attrs:
                vals = np.asarray(cube.attrs[attr_name], dtype=float).ravel()
                if vals.size == n_bands:
                    wl_nm = _ensure_nm(vals)
                    break

    fwhm_path = layout.get("fwhm_path")
    if fwhm_path and fwhm_path in h5_file:
        vals = np.asarray(h5_file[fwhm_path][()], dtype=float).ravel()
        if vals.size == n_bands:
            fwhm_nm = _ensure_nm(vals)
    if fwhm_nm is None:
        cube = h5_file[layout["data_path"]]
        for attr_name in ("fwhm", "full_width_half_max"):
            if attr_name in cube.attrs:
                vals = np.asarray(cube.attrs[attr_name], dtype=float).ravel()
                if vals.size == n_bands:
                    fwhm_nm = _ensure_nm(vals)
                    break

    return wl_nm, fwhm_nm


def _read_wavelengths_from_stac(stac_url, asset_key):
    """Fetch wavelengths and FWHM from a Planet STAC item.

    Args:
        stac_url (str): URL to the STAC item JSON.
        asset_key (str): Asset key whose ``eo:bands`` describes the product,
            e.g. ``basic_radiance_hdf5`` or ``ortho_sr_hdf5``.

    Returns:
        tuple: ``(wavelengths_nm, fwhm_nm)`` as numpy arrays.

    Raises:
        KeyError: If the asset key or ``eo:bands`` block is absent.
    """
    stac_item = requests.get(stac_url, timeout=10).json()
    assets = stac_item.get("assets", {})
    if asset_key not in assets:
        available = sorted(assets.keys())
        raise KeyError(
            f"STAC item has no asset '{asset_key}'. Available assets: {available}"
        )
    bands_meta = assets[asset_key]["eo:bands"]
    wl = np.array([b["center_wavelength"] for b in bands_meta], dtype=float)
    fwhm = np.array(
        [b.get("full_width_half_max", np.nan) for b in bands_meta], dtype=float
    )
    return _ensure_nm(wl), _ensure_nm(fwhm)


def read_tanager(
    filepath,
    bands=None,
    stac_url=None,
    wavelengths=None,
    product=None,
    **kwargs,
):
    """Read Planet Tanager HDF5 hyperspectral data and return an xarray.Dataset.

    Auto-detects the Tanager product variant from the file contents and sources
    wavelength metadata from inside the file when available. Supports all four
    Planet Tanager product variants: ``basic_radiance``, ``ortho_radiance``,
    ``basic_sr``, and ``ortho_sr``. Surface reflectance products expose their
    data as ``surface_reflectance`` with a ``toa_radiance`` alias retained for
    backward compatibility with the rest of the HyperCoast Tanager helpers;
    the alias may be removed in a future major release.

    Wavelengths are sourced in this precedence: (1) the ``wavelengths`` kwarg,
    (2) a wavelength dataset or attribute inside the HDF5 file, (3) the
    ``stac_url`` kwarg parsed for ``eo:bands`` metadata, (4) a synthesized
    integer index with a ``UserWarning``. No hardcoded STAC URL is used.

    Args:
        filepath (str or os.PathLike): Local file path or HTTPS URL to the
            Tanager ``.h5`` file.
        bands (array-like, optional): Indices of spectral bands to keep.
        stac_url (str, optional): STAC item URL to source wavelength metadata
            from when the file does not contain it.
        wavelengths (array-like, optional): Wavelengths in nanometers to use
            directly; must match the number of bands being read.
        product (str, optional): Force a specific product variant. One of
            ``basic_radiance``, ``ortho_radiance``, ``basic_sr``, ``ortho_sr``.
        **kwargs: Extra keyword arguments forwarded to ``xr.Dataset``.

    Returns:
        xr.Dataset: Dataset with dims ``(wavelength, y, x)``, a canonical data
        variable (``toa_radiance`` for radiance products, ``surface_reflectance``
        for SR products, plus a ``toa_radiance`` alias), and ``latitude`` /
        ``longitude`` coordinates on ``(y, x)``.

    Raises:
        ValueError: If no 3-D hyperspectral cube can be located in the file.
    """
    if isinstance(filepath, str) and filepath.startswith("https://"):
        filepath = download_file(filepath)

    with h5py.File(filepath, "r") as f:
        layout = _discover_tanager_layout(f, product=product)

        cube_shape = f[layout["data_path"]].shape
        band_axis = layout["band_axis"]
        n_bands_total = cube_shape[band_axis]

        data = f[layout["data_path"]][()]
        if band_axis != 0:
            data = np.moveaxis(data, band_axis, 0)

        lat_path = layout["lat_path"]
        lon_path = layout["lon_path"]
        if lat_path is None or lon_path is None:
            raise ValueError(
                "Could not locate Latitude/Longitude datasets in the Tanager HDF5 file."
            )
        lat = f[lat_path][()]
        lon = f[lon_path][()]

        wl_nm, fwhm_nm = _read_wavelengths_from_hdf5(f, layout, n_bands_total)

    if layout["fill_value"] is not None:
        data = np.where(data == layout["fill_value"], np.nan, data.astype(float))
    if layout["scale_factor"] != 1.0 or layout["add_offset"] != 0.0:
        data = data.astype(float) * layout["scale_factor"] + layout["add_offset"]

    if wavelengths is not None:
        wl_nm = np.asarray(wavelengths, dtype=float).ravel()
        if wl_nm.size != n_bands_total:
            raise ValueError(
                f"`wavelengths` has length {wl_nm.size} but the data cube has "
                f"{n_bands_total} bands."
            )
        fwhm_nm = fwhm_nm if fwhm_nm is not None else np.full(n_bands_total, np.nan)

    if wl_nm is None and stac_url is not None:
        wl_nm, fwhm_nm = _read_wavelengths_from_stac(stac_url, layout["stac_asset_key"])
        if wl_nm.size != n_bands_total:
            raise ValueError(
                f"STAC item reports {wl_nm.size} bands but the data cube has "
                f"{n_bands_total} bands."
            )

    if wl_nm is None:
        warnings.warn(
            "No wavelength metadata found in the Tanager HDF5 file and no "
            "`stac_url` or `wavelengths` supplied; falling back to integer "
            "band indices. Pass `wavelengths` or `stac_url` for physical nm "
            "values.",
            UserWarning,
            stacklevel=2,
        )
        wl_nm = np.arange(n_bands_total, dtype=float)
        fwhm_nm = np.full(n_bands_total, np.nan)

    if fwhm_nm is None:
        fwhm_nm = np.full(n_bands_total, np.nan)

    if bands is not None:
        data = data[bands]
        wl_nm = wl_nm[bands]
        fwhm_nm = fwhm_nm[bands]

    data_var_name = layout["data_var_name"]

    coords = {
        "wavelength": wl_nm,
        "fwhm": ("wavelength", fwhm_nm),
        "latitude": (("y", "x"), lat),
        "longitude": (("y", "x"), lon),
    }

    da = xr.DataArray(
        data, dims=("wavelength", "y", "x"), coords=coords, name=data_var_name
    )

    ds = xr.Dataset(
        data_vars={data_var_name: da},
        coords={
            "wavelength": da.wavelength,
            "fwhm": ("wavelength", fwhm_nm),
            "latitude": (("y", "x"), lat),
            "longitude": (("y", "x"), lon),
        },
        attrs={
            "source": "Planet Tanager HDF5",
            "product": layout["product"],
            "stac_item": stac_url or "",
            "data_var": data_var_name,
        },
        **kwargs,
    )

    if data_var_name == "surface_reflectance" and "toa_radiance" not in ds.data_vars:
        ds["toa_radiance"] = ds["surface_reflectance"]

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
            except Exception:
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
