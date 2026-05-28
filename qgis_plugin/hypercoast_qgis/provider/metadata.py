# -*- coding: utf-8 -*-
"""
Metadata extraction helpers for the HyperCoast QGIS provider.

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

import numpy as np


def geolocation_bounds(lon, lat):
    """Return bounds from longitude and latitude arrays.

    Args:
        lon: Longitude array.
        lat: Latitude array.

    Returns:
        Tuple of ``(xmin, ymin, xmax, ymax)``.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    return (
        float(np.nanmin(lon)),
        float(np.nanmin(lat)),
        float(np.nanmax(lon)),
        float(np.nanmax(lat)),
    )


def coordinate_bounds(x, y):
    """Return bounds from rectilinear x and y coordinates.

    Args:
        x: X coordinate array.
        y: Y coordinate array.

    Returns:
        Tuple of ``(xmin, ymin, xmax, ymax)``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return (
        float(np.nanmin(x)),
        float(np.nanmin(y)),
        float(np.nanmax(x)),
        float(np.nanmax(y)),
    )


def spectral_coordinates(dataset):
    """Return the first available spectral coordinate array.

    Args:
        dataset: Xarray dataset.

    Returns:
        NumPy array or None.
    """
    for coord_name in ["wavelength", "wavelengths", "band", "bands"]:
        if coord_name in dataset.coords:
            return np.array(dataset.coords[coord_name].values)
    return None
