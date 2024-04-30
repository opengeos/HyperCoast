import xarray as xr


def read_pace(filepath, wavelengths=None, method="nearest", **kwargs):

    ds = xr.open_dataset(filepath, group="geophysical_data")
    ds = ds.swap_dims(
        {
            "number_of_lines": "latitude",
            "pixels_per_line": "longitude",
        }
    )
    wvl = xr.open_dataset(filepath, group="sensor_band_parameters")
    loc = xr.open_dataset(filepath, group="navigation_data")

    lat = loc.latitude
    lat = lat.swap_dims(
        {"number_of_lines": "latitude", "pixel_control_points": "longitude"}
    )

    lon = loc.longitude
    wavelengths = wvl.wavelength_3d
    Rrs = ds.Rrs

    dataset = xr.Dataset(
        {"Rrs": (("latitude", "longitude", "wavelengths"), Rrs.data)},
        coords={
            "latitude": (("latitude", "longitude"), lat.data),
            "longitude": (("latitude", "longitude"), lon.data),
            "wavelengths": ("wavelengths", wavelengths.data),
        },
    )

    if wavelengths is not None:
        dataset = dataset.sel(wavelengths=wavelengths, method=method, **kwargs)
    return dataset
