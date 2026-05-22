# SPDX-FileCopyrightText: 2024 Jack McNelis <jjmcne@gmail.com>
# SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

"""This module contains functions to read and process NASA AVIRIS hyperspectral data.
More info about the data can be found at https://aviris.jpl.nasa.gov.
A portion of the source code is adapted from the jjmcnelis/aviris-ng-notebooks repository
available at https://bit.ly/4bRCgqs. It is licensed under the MIT License. Credit goes to the
original author Jack McNelis.

SPDX-FileCopyrightText = [
    "2024 Jack McNelis <jjmcne@gmail.com>",
]
SPDX-License-Identifier = "MIT"
"""

import os

import rioxarray
import numpy as np
import xarray as xr
from typing import List, Union, Optional, Any
from .common import convert_coords

AVIRIS_NETCDF_SUFFIXES = (".nc", ".nc4", ".cdf")
AVIRIS_ASSET_ALIASES = {
    "reflectance": ("_RFL_ORT.nc",),
    "rfl": ("_RFL_ORT.nc",),
    "rfl_ort": ("_RFL_ORT.nc",),
    "uncertainty": ("_UNC_ORT.nc",),
    "unc": ("_UNC_ORT.nc",),
    "unc_ort": ("_UNC_ORT.nc",),
    "quicklook": ("_RFL_ORT_QL.tif", "_QL.tif"),
    "ql": ("_RFL_ORT_QL.tif", "_QL.tif"),
    "rfl_ort_ql": ("_RFL_ORT_QL.tif", "_QL.tif"),
    "browse": ("_BROWSE.jpg",),
}


def _is_remote_path(path: str) -> bool:
    """Return True for paths that should be opened through fsspec."""
    return isinstance(path, str) and path.startswith(("http://", "https://", "s3://"))


def _is_netcdf_path(path: str) -> bool:
    """Return True when a path looks like an AVIRIS NetCDF source."""
    if not isinstance(path, str):
        return False
    return path.split("?", 1)[0].lower().endswith(AVIRIS_NETCDF_SUFFIXES)


def _source_name(source: Any) -> Optional[str]:
    """Return a source name or URL for path-like and file-like objects."""
    if isinstance(source, (str, os.PathLike)):
        return os.fspath(source)
    for attr in ("full_name", "path", "url", "name"):
        value = getattr(source, attr, None)
        if value:
            return str(value)
    return None


def _open_aviris_netcdf(
    filepath: Any,
    chunks: Any = "auto",
    engine: Optional[str] = None,
    storage_options: Optional[dict] = None,
    group: Optional[str] = "reflectance",
    **kwargs: Any,
) -> xr.Dataset:
    """Open a local or remote AVIRIS NetCDF dataset."""
    open_kwargs = {"decode_coords": "all", **kwargs}
    if chunks is not None:
        open_kwargs["chunks"] = chunks

    if engine is None and (_is_remote_path(filepath) or group is not None):
        engine = "h5netcdf"

    if engine is not None:
        open_kwargs["engine"] = engine

    def open_dataset(source, group_name=None):
        kwargs = dict(open_kwargs)
        if group_name is not None:
            kwargs["group"] = group_name
        return xr.open_dataset(source, **kwargs)

    def seek_start(source):
        seek = getattr(source, "seek", None)
        if callable(seek):
            try:
                seek(0)
            except Exception:
                pass

    def assign_root_coords(dataset, root_dataset):
        if root_dataset is None:
            return dataset
        coords = {}
        for coord in ("easting", "northing", "x", "y"):
            if coord in root_dataset and coord in dataset.dims:
                coords[coord] = np.asarray(root_dataset[coord].values)
        if coords:
            dataset = dataset.assign_coords(coords)
        for var_name in root_dataset.variables:
            if (
                var_name not in dataset.variables
                and root_dataset[var_name].shape == ()
                and (
                    "grid_mapping_name" in root_dataset[var_name].attrs
                    or "spatial_ref" in root_dataset[var_name].attrs
                    or "crs_wkt" in root_dataset[var_name].attrs
                )
            ):
                dataset[var_name] = root_dataset[var_name]
        return dataset

    source = filepath

    if _is_remote_path(filepath):
        import fsspec

        if storage_options is None:
            storage_options = {}
        source = fsspec.open(filepath, mode="rb", **storage_options).open()

    root_ds = None
    if group is not None:
        try:
            root_ds = open_dataset(source)
            seek_start(source)
            ds = open_dataset(source, group)
            ds = assign_root_coords(ds, root_ds)
            root_ds.close()
            return ds
        except Exception:
            seek_start(source)
            if root_ds is not None:
                root_ds.close()

    return open_dataset(source)


def _normalize_aviris_netcdf(
    ds: xr.Dataset,
    wavelengths: Optional[List[float]] = None,
    method: str = "nearest",
    variable: str = "reflectance",
    **kwargs: Any,
) -> xr.Dataset:
    """Normalize AVIRIS-3/5 orthocorrected NetCDF dimensions and metadata."""
    if variable not in ds.data_vars:
        candidates = [
            name
            for name in ds.data_vars
            if name.lower() in ("reflectance", "rfl", "surface_reflectance")
        ]
        if not candidates:
            raise ValueError(f"Could not find AVIRIS data variable: {variable}")
        variable = candidates[0]

    rename = {}
    for old, new in (
        ("easting", "x"),
        ("projection_x_coordinate", "x"),
        ("northing", "y"),
        ("projection_y_coordinate", "y"),
        ("bands", "wavelength"),
        ("band", "wavelength"),
    ):
        if old in ds.dims or old in ds.coords:
            rename[old] = new

    if variable != "reflectance":
        rename[variable] = "reflectance"
        variable = "reflectance"

    if rename:
        ds = ds.rename(rename)

    if "wavelength" not in ds.coords and "wavelength" in ds.data_vars:
        ds = ds.set_coords("wavelength")

    if "wavelength" in ds.coords:
        wavelength = [round(float(num), 2) for num in ds["wavelength"].values]
        ds = ds.assign_coords(wavelength=wavelength)

    dims = [dim for dim in ("y", "x", "wavelength") if dim in ds[variable].dims]
    if len(dims) == 3:
        ds = ds.transpose(*dims, ...)

    global_atts = dict(ds.attrs)
    global_atts.setdefault("Conventions", "CF-1.6")
    fill_value = ds[variable].attrs.get(
        "_FillValue", ds[variable].encoding.get("_FillValue", -9999)
    )
    try:
        if np.isnan(fill_value):
            fill_value = -9999
    except TypeError:
        pass
    ds.attrs = {
        "units": ds[variable].attrs.get("units", "unitless"),
        "_FillValue": fill_value,
        "grid_mapping": ds[variable].attrs.get("grid_mapping", "crs"),
        "standard_name": ds[variable].attrs.get("standard_name", "reflectance"),
        "long_name": ds[variable].attrs.get(
            "long_name", "atmospherically corrected surface reflectance"
        ),
    }
    ds.attrs.update(global_atts)

    try:
        crs = ds.rio.crs
    except Exception:
        crs = None

    grid_mapping = ds[variable].attrs.get("grid_mapping")
    if crs is None and grid_mapping in ds.variables:
        try:
            from rasterio.crs import CRS

            crs = CRS.from_cf(ds[grid_mapping].attrs)
        except Exception:
            crs = None

    if crs is not None:
        ds.attrs["crs"] = crs.to_string()
        try:
            ds = ds.rio.write_crs(crs)
        except Exception:
            pass

    try:
        ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
        ds = ds.rio.write_transform()
    except Exception:
        pass

    if wavelengths is not None:
        ds = ds.sel(wavelength=wavelengths, method=method, **kwargs)

    return ds


def _decode_attr(value: Any) -> Any:
    """Decode byte-string metadata values read from NetCDF/HDF5."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_attr(value.item())
    return value


def _dataset_crs(dataset: xr.Dataset) -> str:
    """Return CRS for an AVIRIS dataset from attrs, rio, or grid mapping."""
    if dataset.attrs.get("crs"):
        return dataset.attrs["crs"]

    try:
        crs = dataset.rio.crs
        if crs is not None:
            return crs.to_string()
    except Exception:
        pass

    mapping_names = []
    if "reflectance" in dataset:
        mapping = dataset["reflectance"].attrs.get("grid_mapping")
        if mapping:
            mapping_names.append(_decode_attr(mapping))
    mapping_names.extend(["crs", "spatial_ref", "transverse_mercator"])

    for name in mapping_names:
        if name not in dataset:
            continue
        attrs = dataset[name].attrs
        for attr in ("spatial_ref", "crs_wkt"):
            if attrs.get(attr):
                return _decode_attr(attrs[attr])
        try:
            from rasterio.crs import CRS

            crs = CRS.from_cf(
                {key: _decode_attr(value) for key, value in attrs.items()}
            )
            if crs is not None:
                return crs.to_string()
        except Exception:
            continue

    raise KeyError("Could not determine the AVIRIS dataset CRS.")


def _coerce_link_dict(link: Any) -> dict:
    """Return a plain dictionary for common CMR or earthaccess link objects."""
    if isinstance(link, dict):
        output = dict(link)
        if "href" not in output and "URL" in output:
            output["href"] = output["URL"]
        if "title" not in output and "Description" in output:
            output["title"] = output["Description"]
        return output
    output = {}
    for key in ("href", "title", "rel", "type"):
        value = getattr(link, key, None)
        if value is not None:
            output[key] = value
    return output


def _granule_links(granule: Any) -> List[dict]:
    """Extract downloadable links from a CMR/earthaccess granule-like object."""
    links = []

    data_links = getattr(granule, "data_links", None)
    if callable(data_links):
        for access in (None, "external", "direct"):
            try:
                urls = data_links() if access is None else data_links(access)
            except TypeError:
                continue
            for url in urls:
                links.append(
                    {
                        "href": url,
                        "rel": "data_links" if access is None else access,
                    }
                )

    s3_links = getattr(granule, "s3_links", None)
    if callable(s3_links):
        try:
            links.extend({"href": url, "rel": "s3_links"} for url in s3_links())
        except TypeError:
            pass

    if isinstance(granule, dict):
        links.extend(granule.get("links", []))
        links.extend(granule.get("RelatedUrls", []))
        umm = granule.get("umm")
        if isinstance(umm, dict):
            links.extend(umm.get("RelatedUrls", []))
        assets = granule.get("assets", {})
        for key, asset in assets.items():
            if isinstance(asset, dict) and asset.get("href"):
                links.append({"href": asset["href"], "title": key, "rel": key})
    else:
        umm = getattr(granule, "umm", None)
        if isinstance(umm, dict):
            links.extend(umm.get("RelatedUrls", []))

    unique_links = []
    seen = set()
    for link in links:
        coerced = _coerce_link_dict(link)
        href = coerced.get("href")
        if href and href not in seen:
            unique_links.append(coerced)
            seen.add(href)

    return unique_links


def get_aviris_granule_ur(granule: Any) -> Optional[str]:
    """Get the CMR GranuleUR from an AVIRIS granule-like object."""
    if isinstance(granule, dict):
        umm = granule.get("umm")
        if isinstance(umm, dict) and umm.get("GranuleUR"):
            return umm["GranuleUR"]
        for key in ("GranuleUR", "producer_granule_id", "title"):
            if granule.get(key):
                return granule[key]
    return None


def get_aviris_collection_concept_id(granule: Any) -> Optional[str]:
    """Get the CMR collection concept ID from an AVIRIS granule-like object."""
    if isinstance(granule, dict):
        meta = granule.get("meta")
        if isinstance(meta, dict) and meta.get("collection-concept-id"):
            return meta["collection-concept-id"]
        for key in ("collection-concept-id", "collection_concept_id"):
            if granule.get(key):
                return granule[key]
    return None


def get_aviris_bounds(granule: Any) -> Optional[List[float]]:
    """Get AVIRIS granule bounds as [min_lon, min_lat, max_lon, max_lat]."""
    if not isinstance(granule, dict):
        return None

    spatial = None
    umm = granule.get("umm")
    if isinstance(umm, dict):
        spatial = umm.get("SpatialExtent")
    if spatial is None:
        spatial = granule.get("SpatialExtent")

    try:
        polygons = spatial["HorizontalSpatialDomain"]["Geometry"]["GPolygons"]
    except (TypeError, KeyError):
        return None

    lons = []
    lats = []
    for polygon in polygons:
        points = polygon.get("Boundary", {}).get("Points", [])
        for point in points:
            if "Longitude" in point and "Latitude" in point:
                lons.append(float(point["Longitude"]))
                lats.append(float(point["Latitude"]))

    if not lons or not lats:
        return None

    return [min(lons), min(lats), max(lons), max(lats)]


def get_aviris_asset_url(
    granule: Any,
    asset: str = "RFL_ORT",
    prefer_s3: bool = False,
) -> str:
    """Get an AVIRIS asset URL from a CMR/earthaccess granule.

    Args:
        granule (Any): A CMR granule dictionary, an earthaccess granule, or a
            STAC-like item dictionary.
        asset (str, optional): Asset alias or filename fragment. Common aliases
            include ``"RFL_ORT"``, ``"UNC_ORT"``, ``"RFL_ORT_QL"``, and
            ``"BROWSE"``. Defaults to ``"RFL_ORT"``.
        prefer_s3 (bool, optional): Prefer ``s3://`` links when available.
            Defaults to False.

    Returns:
        str: The matching asset URL.
    """
    asset_key = asset.lower()
    fragments = AVIRIS_ASSET_ALIASES.get(asset_key, (asset,))
    links = _granule_links(granule)
    if not links:
        raise ValueError("No links found in the AVIRIS granule.")

    def score(link: dict) -> int:
        href = str(link.get("href", ""))
        is_s3 = href.startswith("s3://")
        if prefer_s3 and is_s3:
            return 0
        if not prefer_s3 and not is_s3:
            return 0
        return 1

    for link in sorted(links, key=score):
        href = str(link.get("href", ""))
        title = str(link.get("title", ""))
        haystack = f"{href} {title}".lower()
        if any(fragment.lower() in haystack for fragment in fragments):
            return href

    raise ValueError(f"Could not find AVIRIS asset matching {asset!r}.")


def read_aviris(
    filepath: str,
    wavelengths: Optional[List[float]] = None,
    method: str = "nearest",
    chunks: Any = "auto",
    engine: Optional[str] = None,
    variable: str = "reflectance",
    storage_options: Optional[dict] = None,
    group: Optional[str] = "reflectance",
    **kwargs: Any,
) -> xr.Dataset:
    """
    Reads NASA AVIRIS hyperspectral data and returns an xarray dataset.

    Args:
        filepath (str): The path or URL to the AVIRIS data.
        wavelengths (List[float], optional): The wavelengths to select. If None,
            all wavelengths are selected. Defaults to None.
        method (str, optional): The method to use for selection. Defaults to
            "nearest".
        chunks (Any, optional): Chunking passed to ``xarray.open_dataset`` for
            NetCDF inputs. Defaults to ``"auto"``.
        engine (str, optional): Xarray backend engine for NetCDF inputs. Remote
            NetCDF sources default to ``"h5netcdf"``.
        variable (str, optional): Data variable to read from NetCDF inputs.
            Defaults to ``"reflectance"``.
        storage_options (dict, optional): Options passed to ``fsspec.open`` for
            remote NetCDF inputs, such as S3 or HTTP credentials.
        group (str, optional): NetCDF group containing the reflectance variable.
            Defaults to ``"reflectance"`` for AVIRIS-3/5 products. If the group
            is not present, the root dataset is opened.
        **kwargs (Any): Additional arguments to pass to the selection method.

    Returns:
        xr.Dataset: The dataset containing the reflectance data.
    """
    if not isinstance(filepath, (str, os.PathLike)) and not hasattr(filepath, "read"):
        filepath = get_aviris_asset_url(filepath, asset="RFL_ORT")

    if isinstance(filepath, os.PathLike):
        filepath = os.fspath(filepath)

    file_name = _source_name(filepath)
    if _is_netcdf_path(filepath) or _is_netcdf_path(file_name):
        ds = _open_aviris_netcdf(
            filepath,
            chunks=chunks,
            engine=engine,
            storage_options=storage_options,
            group=group,
        )
        ds = _normalize_aviris_netcdf(
            ds, wavelengths=wavelengths, method=method, variable=variable, **kwargs
        )
        if file_name:
            ds.attrs["_source"] = file_name
            ds.attrs["_source_is_remote"] = _is_remote_path(file_name)
        return ds

    if filepath.endswith(".hdr"):
        filepath = filepath.replace(".hdr", "")

    ds = xr.open_dataset(filepath, engine="rasterio")

    wavelength = ds["wavelength"].values.tolist()
    wavelength = [round(num, 2) for num in wavelength]

    cols = ds.x.size
    rows = ds.y.size

    rio_transform = ds.rio.transform()
    geo_transform = list(rio_transform)[:6]

    # get the raster geotransform as its component parts
    xres, _, xmin, _, yres, ymax = geo_transform

    # generate coordinate arrays
    xarr = np.array([xmin + i * xres for i in range(0, cols)])
    yarr = np.array([ymax + i * yres for i in range(0, rows)])

    ds["y"] = xr.DataArray(
        data=yarr,
        dims=("y"),
        name="y",
        attrs=dict(
            units="m",
            standard_name="projection_y_coordinate",
            long_name="y coordinate of projection",
        ),
    )

    ds["x"] = xr.DataArray(
        data=xarr,
        dims=("x"),
        name="x",
        attrs=dict(
            units="m",
            standard_name="projection_x_coordinate",
            long_name="x coordinate of projection",
        ),
    )

    global_atts = ds.attrs
    global_atts["Conventions"] = "CF-1.6"
    ds.attrs = dict(
        units="unitless",
        _FillValue=-9999,
        grid_mapping="crs",
        standard_name="reflectance",
        long_name="atmospherically corrected surface reflectance",
    )
    ds.attrs.update(global_atts)

    ds = ds.transpose("y", "x", "band")
    ds = ds.drop_vars(["wavelength"])
    ds = ds.rename({"band": "wavelength", "band_data": "reflectance"})
    ds.coords["wavelength"] = wavelength
    ds.attrs["crs"] = ds.rio.crs.to_string()
    ds.rio.write_transform(rio_transform)

    if wavelengths is not None:
        ds = ds.sel(wavelength=wavelengths, method=method, **kwargs)
    return ds


def aviris_to_image(
    dataset: Union[xr.Dataset, str],
    wavelengths: Optional[np.ndarray] = None,
    method: str = "nearest",
    output: Optional[str] = None,
    **kwargs: Any,
):
    """
    Converts an AVIRIS dataset to an image.

    Args:
        dataset (Union[xr.Dataset, str]): The dataset containing the AVIRIS data
            or the file path to the dataset.
        wavelengths (np.ndarray, optional): The specific wavelengths to select. If None, all
            wavelengths are selected. Defaults to None.
        method (str, optional): The method to use for data interpolation.
            Defaults to "nearest".
        output (str, optional): The file path where the image will be saved. If
            None, the image will be returned as a PIL Image object. Defaults to None.
        **kwargs (Any): Additional keyword arguments to be passed to
            `leafmap.array_to_image`.

    Returns:
        Optional[rasterio.Dataset]: The image converted from the dataset. If
            `output` is provided, the image will be saved to the specified file
            and the function will return None.
    """
    from leafmap import array_to_image

    if isinstance(dataset, str):
        dataset = read_aviris(dataset, method=method)

    if wavelengths is not None:
        dataset = dataset.sel(wavelength=wavelengths, method=method)

    return array_to_image(
        dataset["reflectance"],
        output=output,
        transpose=False,
        dtype=np.float32,
        **kwargs,
    )


def extract_aviris(
    dataset: xr.Dataset,
    lat: float,
    lon: float,
    offset: float = 2.0,
    allow_remote: bool = False,
) -> xr.DataArray:
    """
    Extracts AVIRIS data from a given xarray Dataset.

    Args:
        dataset (xarray.Dataset): The dataset containing the AVIRIS data.
        lat (float): The latitude of the point to extract.
        lon (float): The longitude of the point to extract.
        offset (float, optional): The offset from the point to extract. Defaults to 2.0.
        allow_remote (bool, optional): Allow extraction from a remote NetCDF
            source. Defaults to False because point reads from AVIRIS NetCDF
            cubes can be very slow over HTTPS/S3.

    Returns:
        xarray.DataArray: The extracted data.
    """

    if dataset.attrs.get("_source_is_remote") and not allow_remote:
        source = dataset.attrs.get("_source", "remote AVIRIS NetCDF")
        raise RuntimeError(
            "Interactive spectral extraction from a remote AVIRIS NetCDF can be "
            "very slow because the reflectance cube is chunked in large spatial "
            f"blocks. Download the reflectance NetCDF locally and reopen it with "
            f"read_aviris() before using the spectral widget. Source: {source}"
        )

    crs = _dataset_crs(dataset)

    x, y = convert_coords([[lat, lon]], "epsg:4326", crs)[0]

    da = dataset["reflectance"]

    if "xc" in da.coords and "yc" in da.coords:
        x_con = (da["xc"] > x - offset) & (da["xc"] < x + offset)
        y_con = (da["yc"] > y - offset) & (da["yc"] < y + offset)

        try:
            data = da.where(x_con & y_con, drop=True)
            data = data.mean(dim=["x", "y"])
        except ValueError:
            data = np.nan * np.ones(da.sizes["wavelength"])
    else:
        try:
            data = da.sel(x=x, y=y, method="nearest")
        except (KeyError, ValueError):
            data = np.nan * np.ones(da.sizes["wavelength"])

    da = xr.DataArray(
        data, dims=["wavelength"], coords={"wavelength": dataset.coords["wavelength"]}
    )

    return da
