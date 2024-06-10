"""
This Module has the functions related to working with an EMIT dataset. This includes
doing things like opening and flattening the data to work in xarray, orthorectification,
and visualization.

Some source code is adapted from https://github.com/nasa/EMIT-Data-Resources.
Credits to the original authors, including Erik Bolch, Alex Leigh, and others.

"""

import os
import xarray as xr
import numpy as np
import geopandas as gpd


def read_emit(filepath, ortho=True, wavelengths=None, method="nearest", **kwargs):
    """
    Opens an EMIT dataset from a file path and assigns new coordinates to it.

    Args:
        filepath (str): The file path to the EMIT dataset.
        ortho (bool, optional): If True, the function will return an orthorectified dataset. Defaults to True.
        wavelengths (array-like, optional): The specific wavelengths to select. If None, all wavelengths are selected. Defaults to None.
        method (str, optional): The method to use for data selection. Defaults to "nearest".
        **kwargs: Additional keyword arguments to be passed to `xr.open_dataset`.

    Returns:
        xarray.Dataset: The dataset with new coordinates assigned.

    """

    if ortho == True:
        return emit_xarray(
            filepath, ortho=True, wavelengths=wavelengths, method=method, **kwargs
        )
    else:
        ds = xr.open_dataset(filepath, **kwargs)
        wvl = xr.open_dataset(filepath, group="sensor_band_parameters")
        loc = xr.open_dataset(filepath, group="location")
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

        if wavelengths is not None:
            ds = ds.sel(wavelengths=wavelengths, method=method)

        ds = ds.rename({"wavelengths": "wavelength"})
        return ds


def plot_emit(
    ds,
    longitude=None,
    latitude=None,
    downtrack=None,
    crosstrack=None,
    remove_nans=True,
    x="wavelengths",
    y="reflectance",
    color="black",
    frame_height=400,
    frame_width=600,
    title=None,
    method="nearest",
    ortho=True,
    options={},
    **kwargs,
):
    """
    Plots a line graph of the reflectance data from a given dataset.

    Args:
        ds (xarray.Dataset or str): The dataset containing the reflectance data or the file path to the dataset.
        longitude (float, optional): The longitude coordinate to select for orthorectified data. Defaults to None.
        latitude (float, optional): The latitude coordinate to select for orthorectified data. Defaults to None.
        downtrack (int, optional): The downtrack coordinate to select for non-orthorectified data. Defaults to None.
        crosstrack (int, optional): The crosstrack coordinate to select for non-orthorectified data. Defaults to None.
        remove_nans (bool, optional): If True, replace non-good wavelengths with NaN. Defaults to True.
        x (str, optional): The x-axis label. Defaults to "wavelengths".
        y (str, optional): The y-axis label. Defaults to "reflectance".
        color (str, optional): The color of the line. Defaults to "black".
        frame_height (int, optional): The height of the frame. Defaults to 400.
        frame_width (int, optional): The width of the frame. Defaults to 600.
        title (str, optional): The title of the plot. If None, a default title will be generated. Defaults to None.
        method (str, optional): The method to use for data selection. Defaults to "nearest".
        ortho (bool, optional): If True, the function will use longitude and latitude for data selection. Defaults to True.
        options (dict, optional): Additional options to be passed to `hvplot.line`. Defaults to {}.
        **kwargs: Additional keyword arguments to be passed to `hvplot.line`.

    Returns:
        hvplot.Plot: The line plot of the reflectance data.
    """

    import hvplot.xarray

    if ortho == True:
        if longitude is None or latitude is None:
            raise ValueError(
                "Longitude and Latitude must be provided for orthorectified data."
            )
    else:
        if downtrack is None or crosstrack is None:
            raise ValueError(
                "Downtrack and Crosstrack must be provided for non-orthorectified data."
            )

    if longitude is not None and latitude is not None:
        ortho = True

    if downtrack is not None and crosstrack is not None:
        ortho = False

    if isinstance(ds, str):
        ds = read_emit(ds, ortho=ortho)

    if remove_nans:
        ds["reflectance"].data[:, :, ds["good_wavelengths"].data == 0] = np.nan

    if ortho:
        example = ds["reflectance"].sel(
            longitude=longitude, latitude=latitude, method=method
        )
        if title is None:
            title = f"Reflectance at longitude={longitude:.3f}, latitude={latitude:.3f}"

    else:
        example = ds["reflectance"].sel(
            downtrack=downtrack, crosstrack=crosstrack, method=method
        )
        if title is None:
            title = f"Reflectance at downtrack={downtrack}, crosstrack={crosstrack}"

    line = example.hvplot.line(
        y=y,
        x=x,
        color=color,
        frame_height=frame_height,
        frame_width=frame_width,
        **kwargs,
    ).opts(title=title, **options)
    return line


def viz_emit(
    ds,
    wavelengths,
    cmap="viridis",
    frame_width=720,
    method="nearest",
    ortho=True,
    aspect="equal",
    tiles="ESRI",
    alpha=0.8,
    title=None,
    options={},
    **kwargs,
):
    """
    Visualizes the reflectance data from a given dataset at specific wavelengths.

    Args:
        ds (xarray.Dataset or str): The dataset containing the reflectance data or the file path to the dataset.
        wavelengths (array-like): The specific wavelengths to visualize.
        cmap (str, optional): The colormap to use. Defaults to "viridis".
        frame_width (int, optional): The width of the frame. Defaults to 720.
        method (str, optional): The method to use for data selection. Defaults to "nearest".
        ortho (bool, optional): If True, the function will return an orthorectified image. Defaults to True.
        aspect (str, optional): The aspect ratio of the plot. Defaults to "equal".
        tiles (str, optional): The tile source to use for the background map. Defaults to "ESRI".
        alpha (float, optional): The alpha value for the image. Defaults to 0.8.
        title (str, optional): The title of the plot. If None, a default title will be generated. Defaults to None.
        options (dict, optional): Additional options to be passed to `hvplot.image`. Defaults to {}.
        **kwargs: Additional keyword arguments to be passed to `hvplot.image`.

    Returns:
        hvplot.Plot: The image plot of the reflectance data at the specified wavelengths.
    """
    import hvplot.xarray

    if isinstance(ds, str):
        ds = read_emit(ds, ortho=ortho)
    example = ds.sel(wavelength=wavelengths, method=method)

    if title is None:
        title = f"Reflectance at {example.wavelengths.values:.3f} {example.wavelengths.units}"

    if ortho:
        image = example.hvplot.image(
            cmap=cmap,
            geo=ortho,
            tiles=tiles,
            alpha=alpha,
            frame_width=frame_width,
            **kwargs,
        ).opts(title=title, **options)
    else:
        image = example.hvplot.image(
            cmap=cmap, aspect=aspect, alpha=alpha, frame_width=frame_width, **kwargs
        ).opts(title=title, **options)

    return image


def emit_to_netcdf(data, output, **kwargs):
    """
    Transposes an EMIT dataset and saves it as a NetCDF file.

    Args:
        data (xarray.Dataset or str): The dataset containing the EMIT data or the file path to the dataset.
        output (str): The file path where the NetCDF file will be saved.
        **kwargs: Additional keyword arguments to be passed to `xarray.Dataset.to_netcdf`.

    """
    if isinstance(data, str):
        data = read_emit(data, ortho=True)

    ds_geo = data.transpose("wavelengths", "latitude", "longitude")
    ds_geo.to_netcdf(output, **kwargs)


def emit_to_image(data, wavelengths=None, method="nearest", output=None, **kwargs):
    """
    Converts an EMIT dataset to an image.

    Args:
        data (xarray.Dataset or str): The dataset containing the EMIT data or the file path to the dataset.
        wavelengths (array-like, optional): The specific wavelengths to select. If None, all wavelengths are selected. Defaults to None.
        method (str, optional): The method to use for data selection. Defaults to "nearest".
        output (str, optional): The file path where the image will be saved. If None, the image will be returned as a PIL Image object. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to `leafmap.array_to_image`.

    Returns:
        rasterio.Dataset or None: The image converted from the dataset. If `output` is provided, the image will be saved to the specified file and the function will return None.
    """
    from leafmap import array_to_image

    if isinstance(data, str):
        data = read_emit(data, ortho=True)

    ds = data["reflectance"]

    if wavelengths is not None:
        ds = ds.sel(wavelength=wavelengths, method=method)
    return array_to_image(ds, transpose=False, output=output, **kwargs)


def emit_xarray(
    filepath,
    ortho=False,
    qmask=None,
    unpacked_bmask=None,
    wavelengths=None,
    method="nearest",
):
    """
    Streamlines opening an EMIT dataset as an xarray.Dataset.

    Args:
        filepath (str): A filepath to an EMIT netCDF file.
        ortho (bool, optional): Whether to orthorectify the dataset or leave in crosstrack/downtrack coordinates. Defaults to False.
        qmask (numpy.ndarray, optional): A numpy array output from the quality_mask function used to mask pixels based on quality flags selected in that function. Any non-orthorectified array with the proper crosstrack and downtrack dimensions can also be used. Defaults to None.
        unpacked_bmask (numpy.ndarray, optional): A numpy array from the band_mask function that can be used to mask band-specific pixels that have been interpolated. Defaults to None.
        wavelengths (array-like, optional): The specific wavelengths to select. If None, all wavelengths are selected. Defaults to None.
        method (str, optional): The method to use for data selection. Defaults to "nearest".

    Returns:
        xarray.Dataset: An xarray.Dataset constructed based on the parameters provided.
    """
    # Grab granule filename to check product
    import s3fs
    from fsspec.implementations.http import HTTPFile

    if type(filepath) == s3fs.core.S3File:
        granule_id = filepath.info()["name"].split("/", -1)[-1].split(".", -1)[0]
    elif type(filepath) == HTTPFile:
        granule_id = filepath.path.split("/", -1)[-1].split(".", -1)[0]
    else:
        granule_id = os.path.splitext(os.path.basename(filepath))[0]

    # Read in Data as Xarray Datasets
    engine, wvl_group = "h5netcdf", None

    ds = xr.open_dataset(filepath, engine=engine)
    loc = xr.open_dataset(filepath, engine=engine, group="location")

    # Check if mineral dataset and read in groups (only ds/loc for minunc)

    if "L2B_MIN_" in granule_id:
        wvl_group = "mineral_metadata"
    elif "L2B_MINUNC" not in granule_id:
        wvl_group = "sensor_band_parameters"

    wvl = None

    if wvl_group:
        wvl = xr.open_dataset(filepath, engine=engine, group=wvl_group)

    # Building Flat Dataset from Components
    data_vars = {**ds.variables}

    # Format xarray coordinates based upon emit product (no wvl for mineral uncertainty)
    coords = {
        "downtrack": (["downtrack"], ds.downtrack.data),
        "crosstrack": (["crosstrack"], ds.crosstrack.data),
        **loc.variables,
    }

    product_band_map = {
        "L2B_MIN_": "name",
        "L2A_MASK_": "mask_bands",
        "L1B_OBS_": "observation_bands",
        "L2A_RFL_": "wavelengths",
        "L1B_RAD_": "wavelengths",
        "L2A_RFLUNCERT_": "wavelengths",
    }

    # if band := product_band_map.get(next((k for k in product_band_map.keys() if k in granule_id), 'unknown'), None):
    # coords['bands'] = wvl[band].data

    if wvl:
        coords = {**coords, **wvl.variables}

    out_xr = xr.Dataset(data_vars=data_vars, coords=coords, attrs=ds.attrs)
    out_xr.attrs["granule_id"] = granule_id

    if band := product_band_map.get(
        next((k for k in product_band_map.keys() if k in granule_id), "unknown"), None
    ):
        if "minerals" in list(out_xr.dims):
            out_xr = out_xr.swap_dims({"minerals": band})
            out_xr = out_xr.rename({band: "mineral_name"})
        else:
            out_xr = out_xr.swap_dims({"bands": band})

    # Apply Quality and Band Masks, set fill values to NaN
    for var in list(ds.data_vars):
        if qmask is not None:
            out_xr[var].data[qmask == 1] = np.nan
        if unpacked_bmask is not None:
            out_xr[var].data[unpacked_bmask == 1] = np.nan
        out_xr[var].data[out_xr[var].data == -9999] = np.nan

    if ortho is True:
        out_xr = ortho_xr(out_xr)
        out_xr.attrs["Orthorectified"] = "True"

    if wavelengths is not None:
        out_xr = out_xr.sel(wavelengths=wavelengths, method=method)

    out_xr = out_xr.rename({"wavelengths": "wavelength"})
    return out_xr


# Function to Calculate the Lat and Lon Vectors/Coordinate Grid
def coord_vects(ds):
    """
    This function calculates the Lat and Lon Coordinate Vectors using the GLT and Metadata from an EMIT dataset read into xarray.

    Parameters:
    ds: an xarray.Dataset containing the root variable and metadata of an EMIT dataset
    loc: an xarray.Dataset containing the 'location' group of an EMIT dataset

    Returns:
    lon, lat (numpy.array): longitute and latitude array grid for the dataset

    """
    # Retrieve Geotransform from Metadata
    GT = ds.geotransform
    # Create Array for Lat and Lon and fill
    dim_x = ds.glt_x.shape[1]
    dim_y = ds.glt_x.shape[0]
    lon = np.zeros(dim_x)
    lat = np.zeros(dim_y)
    # Note: no rotation for EMIT Data
    for x in np.arange(dim_x):
        x_geo = (GT[0] + 0.5 * GT[1]) + x * GT[1]  # Adjust coordinates to pixel-center
        lon[x] = x_geo
    for y in np.arange(dim_y):
        y_geo = (GT[3] + 0.5 * GT[5]) + y * GT[5]
        lat[y] = y_geo
    return lon, lat


def apply_glt(ds_array, glt_array, fill_value=-9999, GLT_NODATA_VALUE=0):
    """
    Applies the GLT array to a numpy array of either 2 or 3 dimensions to orthorectify the data.

    Args:
        ds_array (numpy.ndarray): A numpy array of the desired variable.
        glt_array (numpy.ndarray): A GLT array constructed from EMIT GLT data.
        fill_value (int, optional): The value to fill in the output array where the GLT array has no data. Defaults to -9999.
        GLT_NODATA_VALUE (int, optional): The value in the GLT array that indicates no data. Defaults to 0.

    Returns:
        numpy.ndarray: A numpy array of orthorectified data.
    """

    # Build Output Dataset
    if ds_array.ndim == 2:
        ds_array = ds_array[:, :, np.newaxis]
    out_ds = np.full(
        (glt_array.shape[0], glt_array.shape[1], ds_array.shape[-1]),
        fill_value,
        dtype=np.float32,
    )
    valid_glt = np.all(glt_array != GLT_NODATA_VALUE, axis=-1)

    # Adjust for One based Index - make a copy to prevent decrementing multiple times inside ortho_xr when applying the glt to elev
    glt_array_copy = glt_array.copy()
    glt_array_copy[valid_glt] -= 1
    out_ds[valid_glt, :] = ds_array[
        glt_array_copy[valid_glt, 1], glt_array_copy[valid_glt, 0], :
    ]
    return out_ds


def ortho_xr(ds, GLT_NODATA_VALUE=0, fill_value=-9999):
    """
    Uses `apply_glt` to create an orthorectified xarray dataset.

    Args:
        ds (xarray.Dataset): An xarray dataset produced by emit_xarray.
        GLT_NODATA_VALUE (int, optional): No data value for the GLT tables. Defaults to 0.
        fill_value (int, optional): The fill value for EMIT datasets. Defaults to -9999.

    Returns:
        xarray.Dataset: An orthocorrected xarray dataset.
    """
    # Build glt_ds

    glt_ds = np.nan_to_num(
        np.stack([ds["glt_x"].data, ds["glt_y"].data], axis=-1), nan=GLT_NODATA_VALUE
    ).astype(int)

    # List Variables
    var_list = list(ds.data_vars)

    # Remove flat field from data vars - the flat field is only useful with additional information before orthorectification
    if "flat_field_update" in var_list:
        var_list.remove("flat_field_update")

    # Create empty dictionary for orthocorrected data vars
    data_vars = {}

    # Extract Rawspace Dataset Variable Values (Typically Reflectance)
    for var in var_list:
        raw_ds = ds[var].data
        var_dims = ds[var].dims
        # Apply GLT to dataset
        out_ds = apply_glt(raw_ds, glt_ds, GLT_NODATA_VALUE=GLT_NODATA_VALUE)

        # Mask fill values
        out_ds[out_ds == fill_value] = np.nan

        # Update variables - Only works for 2 or 3 dimensional arays
        if raw_ds.ndim == 2:
            out_ds = out_ds.squeeze()
            data_vars[var] = (["latitude", "longitude"], out_ds)
        else:
            data_vars[var] = (["latitude", "longitude", var_dims[-1]], out_ds)

        del raw_ds

    # Calculate Lat and Lon Vectors
    lon, lat = coord_vects(
        ds
    )  # Reorder this function to make sense in case of multiple variables

    # Apply GLT to elevation
    elev_ds = apply_glt(ds["elev"].data, glt_ds)
    elev_ds[elev_ds == fill_value] = np.nan

    # Delete glt_ds - no longer needed
    del glt_ds

    # Create Coordinate Dictionary
    coords = {
        "latitude": (["latitude"], lat),
        "longitude": (["longitude"], lon),
        **ds.coords,
    }  # unpack to add appropriate coordinates

    # Remove Unnecessary Coords
    for key in ["downtrack", "crosstrack", "lat", "lon", "glt_x", "glt_y", "elev"]:
        del coords[key]

    # Add Orthocorrected Elevation
    coords["elev"] = (["latitude", "longitude"], np.squeeze(elev_ds))

    # Build Output xarray Dataset and assign data_vars array attributes
    out_xr = xr.Dataset(data_vars=data_vars, coords=coords, attrs=ds.attrs)

    del out_ds
    # Assign Attributes from Original Datasets
    for var in var_list:
        out_xr[var].attrs = ds[var].attrs
    out_xr.coords["latitude"].attrs = ds["lat"].attrs
    out_xr.coords["longitude"].attrs = ds["lon"].attrs
    out_xr.coords["elev"].attrs = ds["elev"].attrs

    # Add Spatial Reference in recognizable format
    out_xr.rio.write_crs(ds.spatial_ref, inplace=True)

    return out_xr


def quality_mask(filepath, quality_bands):
    """
    Builds a single layer mask to apply based on the bands selected from an EMIT L2A Mask file.

    Args:
        filepath (str): An EMIT L2A Mask netCDF file.
        quality_bands (list): A list of bands (quality flags only) from the mask file that should be used in creation of mask.

    Returns:
        numpy.ndarray: A numpy array that can be used with the emit_xarray function to apply a quality mask.

    Raises:
        AttributeError: If the selected flags include a data band (5 or 6) not just flag bands.
    """
    # Open Dataset
    mask_ds = xr.open_dataset(filepath, engine="h5netcdf")
    # Open Sensor band Group
    mask_parameters_ds = xr.open_dataset(
        filepath, engine="h5netcdf", group="sensor_band_parameters"
    )
    # Print Flags used
    flags_used = mask_parameters_ds["mask_bands"].data[quality_bands]
    print(f"Flags used: {flags_used}")
    # Check for data bands and build mask
    if any(x in quality_bands for x in [5, 6]):
        err_str = f"Selected flags include a data band (5 or 6) not just flag bands"
        raise AttributeError(err_str)
    else:
        qmask = np.sum(mask_ds["mask"][:, :, quality_bands].values, axis=-1)
        qmask[qmask > 1] = 1
    return qmask


def band_mask(filepath):
    """
    Unpacks the packed band mask to apply to the dataset. Can be used manually or as an input in the emit_xarray() function.

    Args:
        filepath (str): An EMIT L2A Mask netCDF file.

    Returns:
        numpy.ndarray: A numpy array that can be used with the emit_xarray function to apply a band mask.
    """
    # Open Dataset
    mask_ds = xr.open_dataset(filepath, engine="h5netcdf")
    # Open band_mask and convert to uint8
    bmask = mask_ds.band_mask.data.astype("uint8")
    # Print Flags used
    unpacked_bmask = np.unpackbits(bmask, axis=-1)
    # Remove bands > 285
    unpacked_bmask = unpacked_bmask[:, :, 0:285]
    # Check for data bands and build mask
    return unpacked_bmask


def write_envi(
    xr_ds,
    output_dir,
    overwrite=False,
    extension=".img",
    interleave="BIL",
    glt_file=False,
):
    """
    Takes an EMIT dataset read into an xarray dataset using the emit_xarray function and writes an ENVI file and header.

    Args:
        xr_ds (xarray.Dataset): An EMIT dataset read into xarray using the emit_xarray function.
        output_dir (str): Output directory.
        overwrite (bool, optional): Overwrite existing file if True. Defaults to False.
        extension (str, optional): The file extension for the envi formatted file, .img by default. Defaults to ".img".
        interleave (str, optional): The interleave option for the ENVI file. Defaults to "BIL".
        glt_file (bool, optional): Also create a GLT ENVI file for later use to reproject. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - envi_ds (spectral.io.envi.Image): ENVI file in the output directory.
            - glt_ds (spectral.io.envi.Image): GLT file in the output directory.

    Raises:
        Exception: If the data is already orthorectified but a GLT file is still requested.
    """
    from spectral.io import envi

    # Check if xr_ds has been orthorectified, raise exception if it has been but GLT is still requested
    if (
        "Orthorectified" in xr_ds.attrs.keys()
        and xr_ds.attrs["Orthorectified"] == "True"
        and glt_file == True
    ):
        raise Exception("Data is already orthorectified.")

    # Typemap dictionary for ENVI files
    envi_typemap = {
        "uint8": 1,
        "int16": 2,
        "int32": 3,
        "float32": 4,
        "float64": 5,
        "complex64": 6,
        "complex128": 9,
        "uint16": 12,
        "uint32": 13,
        "int64": 14,
        "uint64": 15,
    }

    # Get CRS/geotransform for creation of Orthorectified ENVI file or optional GLT file
    gt = xr_ds.attrs["geotransform"]
    mapinfo = (
        "{Geographic Lat/Lon, 1, 1, "
        + str(gt[0])
        + ", "
        + str(gt[3])
        + ", "
        + str(gt[1])
        + ", "
        + str(gt[5] * -1)
        + ", WGS-84, units=Degrees}"
    )

    # This creates the coordinate system string
    # hard-coded replacement of wkt crs could probably be improved, though should be the same for all EMIT datasets
    csstring = '{ GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]] }'
    # List data variables (typically reflectance/radiance)
    var_names = list(xr_ds.data_vars)

    # Loop through variable names
    for var in var_names:
        # Define output filename
        output_name = os.path.join(output_dir, xr_ds.attrs["granule_id"] + "_" + var)

        nbands = 1
        if len(xr_ds[var].data.shape) > 2:
            nbands = xr_ds[var].data.shape[2]

        # Start building metadata
        metadata = {
            "lines": xr_ds[var].data.shape[0],
            "samples": xr_ds[var].data.shape[1],
            "bands": nbands,
            "interleave": interleave,
            "header offset": 0,
            "file type": "ENVI Standard",
            "data type": envi_typemap[str(xr_ds[var].data.dtype)],
            "byte order": 0,
        }

        for key in list(xr_ds.attrs.keys()):
            if key == "summary":
                metadata["description"] = xr_ds.attrs[key]
            elif key not in ["geotransform", "spatial_ref"]:
                metadata[key] = f"{{ {xr_ds.attrs[key]} }}"

        # List all variables in dataset (including coordinate variables)
        meta_vars = list(xr_ds.variables)

        # Add band parameter information to metadata (ie wavelengths/obs etc.)
        for m in meta_vars:
            if m == "wavelengths" or m == "radiance_wl":
                metadata["wavelength"] = np.array(xr_ds[m].data).astype(str).tolist()
            elif m == "fwhm" or m == "radiance_fwhm":
                metadata["fwhm"] = np.array(xr_ds[m].data).astype(str).tolist()
            elif m == "good_wavelengths":
                metadata["good_wavelengths"] = (
                    np.array(xr_ds[m].data).astype(int).tolist()
                )
            elif m == "observation_bands":
                metadata["band names"] = np.array(xr_ds[m].data).astype(str).tolist()
            elif m == "mask_bands":
                if var == "band_mask":
                    metadata["band names"] = [
                        "packed_bands_" + bn
                        for bn in np.arange(285 / 8).astype(str).tolist()
                    ]
                else:
                    metadata["band names"] = (
                        np.array(xr_ds[m].data).astype(str).tolist()
                    )
            if "wavelength" in list(metadata.keys()) and "band names" not in list(
                metadata.keys()
            ):
                metadata["band names"] = metadata["wavelength"]

        # Add CRS/mapinfo if xarray dataset has been orthorectified
        if (
            "Orthorectified" in xr_ds.attrs.keys()
            and xr_ds.attrs["Orthorectified"] == "True"
        ):
            metadata["coordinate system string"] = csstring
            metadata["map info"] = mapinfo

        # Replace NaN values in each layer with fill_value
        # np.nan_to_num(xr_ds[var].data, copy=False, nan=-9999)

        # Write Variables as ENVI Output
        envi_ds = envi.create_image(
            envi_header(output_name), metadata, ext=extension, force=overwrite
        )
        mm = envi_ds.open_memmap(interleave="bip", writable=True)

        dat = xr_ds[var].data

        if len(dat.shape) == 2:
            dat = dat.reshape((dat.shape[0], dat.shape[1], 1))

        mm[...] = dat

    # Create GLT Metadata/File
    if glt_file == True:
        # Output Name
        glt_output_name = os.path.join(
            output_dir, xr_ds.attrs["granule_id"] + "_" + "glt"
        )

        # Write GLT Metadata
        glt_metadata = metadata

        # Remove Unwanted Metadata
        glt_metadata.pop("wavelength", None)
        glt_metadata.pop("fwhm", None)

        # Replace Metadata
        glt_metadata["lines"] = xr_ds["glt_x"].data.shape[0]
        glt_metadata["samples"] = xr_ds["glt_x"].data.shape[1]
        glt_metadata["bands"] = 2
        glt_metadata["data type"] = envi_typemap["int32"]
        glt_metadata["band names"] = ["glt_x", "glt_y"]
        glt_metadata["coordinate system string"] = csstring
        glt_metadata["map info"] = mapinfo

        # Write GLT Outputs as ENVI File
        glt_ds = envi.create_image(
            envi_header(glt_output_name), glt_metadata, ext=extension, force=overwrite
        )
        mmglt = glt_ds.open_memmap(interleave="bip", writable=True)
        mmglt[...] = np.stack(
            (xr_ds["glt_x"].values, xr_ds["glt_y"].values), axis=-1
        ).astype("int32")


def envi_header(inputpath):
    """
    Convert a ENVI binary/header path to a header, handling extensions.

    Args:
        inputpath (str): Path to ENVI binary file.

    Returns:
        str: The header file associated with the input reference. If the header file does not exist, it returns the expected header file path.
    """
    if (
        os.path.splitext(inputpath)[-1] == ".img"
        or os.path.splitext(inputpath)[-1] == ".dat"
        or os.path.splitext(inputpath)[-1] == ".raw"
    ):
        # headers could be at either filename.img.hdr or filename.hdr.  Check both, return the one that exists if it
        # does, if not return the latter (new file creation presumed).
        hdrfile = os.path.splitext(inputpath)[0] + ".hdr"
        if os.path.isfile(hdrfile):
            return hdrfile
        elif os.path.isfile(inputpath + ".hdr"):
            return inputpath + ".hdr"
        return hdrfile
    elif os.path.splitext(inputpath)[-1] == ".hdr":
        return inputpath
    else:
        return inputpath + ".hdr"


def raw_spatial_crop(ds, shape):
    """
    Use a polygon to clip the file GLT, then a bounding box to crop the spatially raw data. Regions clipped in the GLT are set to 0 so a mask will be applied when
    used to orthorectify the data at a later point in a workflow.

    Args:
        ds (xarray.Dataset): Raw spatial EMIT data (non-orthorectified) opened with the `emit_xarray` function.
        shape (geopandas.GeoDataFrame): A polygon opened with geopandas.

    Returns:
        xarray.Dataset: A clipped GLT and raw spatial data clipped to a bounding box.
    """
    # Reformat the GLT
    lon, lat = coord_vects(ds)
    data_vars = {
        "glt_x": (["latitude", "longitude"], ds.glt_x.data),
        "glt_y": (["latitude", "longitude"], ds.glt_y.data),
    }
    coords = {
        "latitude": (["latitude"], lat),
        "longitude": (["longitude"], lon),
        "ortho_y": (["latitude"], ds.ortho_y.data),
        "ortho_x": (["longitude"], ds.ortho_x.data),
    }
    glt_ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=ds.attrs)
    glt_ds.rio.write_crs(glt_ds.spatial_ref, inplace=True)

    # Clip the emit glt
    clipped = glt_ds.rio.clip(shape.geometry.values, shape.crs, all_touched=True)

    # Pull new geotransform from clipped glt
    clipped_gt = np.array(
        [float(i) for i in clipped["spatial_ref"].GeoTransform.split(" ")]
    )  # THIS GEOTRANSFORM IS OFF BY HALF A PIXEL

    # Create Crosstrack and Downtrack masks for spatially raw dataset -1 is to account for 1 based index. May be a more robust way to do this exists
    crosstrack_mask = (ds.crosstrack >= np.nanmin(clipped.glt_x.data) - 1) & (
        ds.crosstrack <= np.nanmax(clipped.glt_x.data) - 1
    )
    downtrack_mask = (ds.downtrack >= np.nanmin(clipped.glt_y.data) - 1) & (
        ds.downtrack <= np.nanmax(clipped.glt_y.data) - 1
    )

    # Mask Areas outside of crosstrack and downtrack covered by the shape
    clipped_ds = ds.where((crosstrack_mask & downtrack_mask), drop=True)
    # Replace Full dataset geotransform with clipped geotransform
    clipped_ds.attrs["geotransform"] = clipped_gt

    # Drop unnecessary vars from dataset
    clipped_ds = clipped_ds.drop_vars(["glt_x", "glt_y", "downtrack", "crosstrack"])

    # Re-index the GLT to the new array
    glt_x_data = clipped.glt_x.data - np.nanmin(clipped.glt_x)
    glt_y_data = clipped.glt_y.data - np.nanmin(clipped.glt_y)
    clipped_ds = clipped_ds.assign_coords(
        {
            "glt_x": (["ortho_y", "ortho_x"], np.nan_to_num(glt_x_data)),
            "glt_y": (["ortho_y", "ortho_x"], np.nan_to_num(glt_y_data)),
        }
    )
    clipped_ds = clipped_ds.assign_coords(
        {
            "downtrack": (
                ["downtrack"],
                np.arange(0, clipped_ds[list(ds.data_vars.keys())[0]].shape[0]),
            ),
            "crosstrack": (
                ["crosstrack"],
                np.arange(0, clipped_ds[list(ds.data_vars.keys())[0]].shape[1]),
            ),
        }
    )

    return clipped_ds


def is_adjacent(scene: str, same_orbit: list):
    """
    Checks if the scene numbers from the same orbit are adjacent/sequential.

    Args:
        scene (str): The scene number to check.
        same_orbit (list): A list of scene numbers from the same orbit.

    Returns:
        bool: True if the scene numbers are adjacent/sequential, False otherwise.
    """
    scene_nums = [int(scene.split(".")[-2].split("_")[-1]) for scene in same_orbit]
    return all(b - a == 1 for a, b in zip(scene_nums[:-1], scene_nums[1:]))


def merge_emit(datasets: dict, gdf: gpd.GeoDataFrame):
    """
    Merges xarray datasets formatted using emit_xarray. Note: GDF may only work with a single geometry.

    Args:
        datasets (dict): A dictionary of xarray datasets formatted using emit_xarray.
        gdf (gpd.GeoDataFrame): A GeoDataFrame containing the geometry to be used for merging.

    Returns:
        xarray.Dataset: A merged xarray dataset.

    Raises:
        Exception: If there are inconsistencies in the 1D variables across datasets.
    """
    from rioxarray.merge import merge_arrays

    nested_data_arrays = {}
    # loop over datasets
    for dataset in datasets:
        # create dictionary of arrays for each dataset

        # create dictionary of 1D variables, which should be consistent across datasets
        one_d_arrays = {}

        # Dictionary of variables to merge
        data_arrays = {}
        # Loop over variables in dataset including elevation
        for var in list(datasets[dataset].data_vars) + ["elev"]:
            # Get 1D for this variable and add to dictionary
            if not one_d_arrays:
                # These should be an array describing the others (wavelengths, mask_bands, etc.)
                one_dim = [
                    item
                    for item in list(datasets[dataset].coords)
                    if item not in ["latitude", "longitude", "spatial_ref"]
                    and len(datasets[dataset][item].dims) == 1
                ]
                # print(one_dim)
                for od in one_dim:
                    one_d_arrays[od] = datasets[dataset].coords[od].data

                # Update format for merging - This could probably be improved
            da = datasets[dataset][var].reset_coords("elev", drop=False)
            da = da.rename({"latitude": "y", "longitude": "x"})
            if len(da.dims) == 3:
                if any(item in list(da.coords) for item in one_dim):
                    da = da.drop_vars(one_dim)
                da = da.drop_vars("elev")
                da = da.to_array(name=var).squeeze("variable", drop=True)
                da = da.transpose(da.dims[-1], da.dims[0], da.dims[1])
                # print(da.dims)
            if var == "elev":
                da = da.to_array(name=var).squeeze("variable", drop=True)
            data_arrays[var] = da
            nested_data_arrays[dataset] = data_arrays

            # Transpose the nested arrays dict. This is horrible to read, but works to pair up variables (ie mask) from the different granules
    transposed_dict = {
        inner_key: {
            outer_key: inner_dict[inner_key]
            for outer_key, inner_dict in nested_data_arrays.items()
        }
        for inner_key in nested_data_arrays[next(iter(nested_data_arrays))]
    }

    # remove some unused data
    del nested_data_arrays, data_arrays, da

    # Merge the arrays using rioxarray.merge_arrays()
    merged = {}
    for _var in transposed_dict:
        merged[_var] = merge_arrays(
            list(transposed_dict[_var].values()),
            bounds=gdf.unary_union.bounds,
            nodata=np.nan,
        )

    # Create a new xarray dataset from the merged arrays
    # Create Merged Dataset
    merged_ds = xr.Dataset(data_vars=merged, coords=one_d_arrays)
    # Rename x and y to longitude and latitude
    merged_ds = merged_ds.rename({"y": "latitude", "x": "longitude"})
    del transposed_dict, merged
    return merged_ds


def ortho_browse(url, glt, spatial_ref, geotransform, white_background=True):
    """
    Use an EMIT GLT, geotransform, and spatial ref to orthorectify a browse image. (browse images are in native resolution)

    Args:
        url (str): URL of the browse image.
        glt (numpy.ndarray): A GLT array constructed from EMIT GLT data.
        spatial_ref (str): Spatial reference system.
        geotransform (list): A list of six numbers that define the affine transform between pixel coordinates and map coordinates.
        white_background (bool, optional): If True, the fill value for the orthorectified image is white (255). If False, the fill value is black (0). Defaults to True.

    Returns:
        xarray.DataArray: An orthorectified browse image in the form of an xarray DataArray.
    """
    from skimage import io

    # Read Data
    data = io.imread(url)
    # Orthorectify using GLT and transpose so band is first dimension
    if white_background == True:
        fill = 255
    else:
        fill = 0
    ortho_data = apply_glt(data, glt, fill_value=fill).transpose(2, 0, 1)
    coords = {
        "y": (
            ["y"],
            (geotransform[3] + 0.5 * geotransform[5])
            + np.arange(glt.shape[0]) * geotransform[5],
        ),
        "x": (
            ["x"],
            (geotransform[0] + 0.5 * geotransform[1])
            + np.arange(glt.shape[1]) * geotransform[1],
        ),
    }
    ortho_data = ortho_data.astype(int)
    ortho_data[ortho_data == -1] = 0
    # Place in xarray.datarray
    da = xr.DataArray(ortho_data, dims=["band", "y", "x"], coords=coords)
    da.rio.write_crs(spatial_ref, inplace=True)
    return da
