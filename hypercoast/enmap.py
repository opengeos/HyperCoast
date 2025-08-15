# SPDX-FileCopyrightText: 2025 Advait Dhamorikar <advaitdhamorikar[at]gmail.com>
#
# SPDX-License-Identifier: MIT

from typing import List, Union, Optional, Any
import xarray as xr
import numpy as np
import os
import glob
import xml.etree.ElementTree as ET
from leafmap import array_to_image
from .common import convert_coords


def _extract_enmap_wavelengths_from_xml(metadata_xml_path: str) -> List[float]:
    """
    Extracts EnMAP spectral center wavelengths (in nm) from the metadata XML file.
    Looks for <wavelengthCenterOfBand> inside each <bandID> block.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(metadata_xml_path)
    root = tree.getroot()

    wavelengths = []
    for band_elem in root.findall(".//bandID"):
        wl_elem = band_elem.find("wavelengthCenterOfBand")
        if wl_elem is not None and wl_elem.text:
            wavelengths.append(float(wl_elem.text))

    if not wavelengths:
        raise ValueError(
            f"No wavelengthCenterOfBand elements found in XML: {metadata_xml_path}"
        )

    return wavelengths


def read_enmap(
    filepath: str,
    wavelengths: Optional[List[float]] = None,
    method: str = "nearest",
    **kwargs: Any,
) -> xr.Dataset:
    """
    Reads EnMAP hyperspectral GeoTIFF (COG) and returns an xarray dataset.
    If wavelengths are not in the file, they will be extracted from the
    associated Metadata XML file in the same directory.
    Masks NoData values as NaN.
    """
    ds = xr.open_dataset(filepath, engine="rasterio")

    if wavelengths is None:
        if "wavelength" in ds:
            wavelength = ds["wavelength"].values.tolist()
        elif "wavelength" in ds.attrs:
            wavelength = [
                float(w) for w in str(ds.attrs["wavelength"]).replace(",", " ").split()
            ]
        else:
            folder = os.path.dirname(filepath)
            pattern_lower = os.path.join(folder, "*METADATA.xml")
            pattern_upper = os.path.join(folder, "*METADATA.XML")

            matches = glob.glob(pattern_lower) + glob.glob(pattern_upper)
            if not matches:
                raise ValueError(
                    f"No wavelength information in GeoTIFF and no matching METADATA XML found in {folder}"
                )
            xml_path = matches[0]
            wavelength = _extract_enmap_wavelengths_from_xml(xml_path)

    wavelength = [round(float(num), 2) for num in wavelength]

    cols, rows = ds.x.size, ds.y.size
    rio_transform = ds.rio.transform()
    xres, _, xmin, _, yres, ymax = list(rio_transform)[:6]
    xarr = np.array([xmin + i * xres for i in range(cols)])
    yarr = np.array([ymax + i * yres for i in range(rows)])

    ds["y"] = xr.DataArray(
        yarr,
        dims=("y",),
        attrs={"units": "m", "standard_name": "projection_y_coordinate"},
    )
    ds["x"] = xr.DataArray(
        xarr,
        dims=("x",),
        attrs={"units": "m", "standard_name": "projection_x_coordinate"},
    )

    # Dataset attributes
    global_atts = ds.attrs
    global_atts["Conventions"] = "CF-1.6"
    ds.attrs = {
        "units": "unitless",
        "_FillValue": 0,
        "grid_mapping": "crs",
        "standard_name": "reflectance",
        "long_name": "surface reflectance",
    }
    ds.attrs.update(global_atts)

    ds = ds.transpose("y", "x", "band")
    ds = ds.rename({"band": "wavelength", "band_data": "reflectance"})
    ds.coords["wavelength"] = wavelength

    if ds.rio.crs:
        ds.attrs["crs"] = ds.rio.crs.to_string()
    ds.rio.write_transform(rio_transform)

    if wavelengths is not None:
        ds = ds.sel(wavelength=wavelengths, method=method, **kwargs)

    return ds


def enmap_to_image(
    dataset: Union[xr.Dataset, str],
    wavelengths: Optional[np.ndarray] = None,
    method: str = "nearest",
    output: Optional[str] = None,
    scale_factor: float = 10000.0,
    rgb_wavelengths: Optional[List[float]] = None,
    **kwargs: Any,
):
    """
    Converts an EnMAP dataset to an RGB image.
    Automatically masks NoData and scales reflectance if needed.

    Args:
        dataset (Union[xr.Dataset, str]): Path or xarray Dataset.
        wavelengths (np.ndarray, optional): Wavelengths to select.
        method (str): Selection method ("nearest" by default).
        output (str, optional): File path to save image. If None, returns PIL.Image.
        scale_factor (float): Scale factor to convert integer reflectance to 0–1.
        rgb_wavelengths (list, optional): Wavelengths (nm) to use for RGB, e.g., [660, 560, 480].
        **kwargs: Passed to leafmap.array_to_image().
    """

    if isinstance(dataset, str):
        dataset = read_enmap(dataset, method=method)

    if wavelengths is not None:
        dataset = dataset.sel(wavelength=wavelengths, method=method)

    # Pick default RGB if not provided
    if rgb_wavelengths is None and wavelengths is None:
        rgb_wavelengths = [660, 560, 480]  # Red, Green, Blue in nm
        dataset = dataset.sel(wavelength=rgb_wavelengths, method="nearest")

    arr = dataset["reflectance"]

    if "_FillValue" in dataset.attrs:
        nodata = dataset.attrs["_FillValue"]
        arr = arr.where(arr != nodata)

    if scale_factor and arr.max() > 1.5:
        arr = arr / scale_factor

    return array_to_image(
        arr,
        output=output,
        transpose=False,
        dtype=np.float32,
        **kwargs,
    )


def extract_enmap(ds: xr.Dataset, lat: float, lon: float) -> xr.DataArray:
    """
    Extracts EnMAP reflectance spectrum from a given xarray Dataset.

    Args:
        ds (xarray.Dataset): The dataset containing the EnMAP data.
        lat (float): The latitude of the point to extract.
        lon (float): The longitude of the point to extract.

    Returns:
        xarray.DataArray: The extracted spectrum with wavelength as coordinate.
    """
    if "crs" not in ds.attrs:
        raise ValueError("Dataset has no CRS in attributes.")

    crs = ds.attrs["crs"]

    x, y = convert_coords([[lat, lon]], "epsg:4326", crs)[0]

    values = ds.sel(x=x, y=y, method="nearest")["reflectance"].values

    da = xr.DataArray(
        values,
        dims=["wavelength"],
        coords={"wavelength": ds.coords["wavelength"]},
        name="reflectance",
    )

    return da
