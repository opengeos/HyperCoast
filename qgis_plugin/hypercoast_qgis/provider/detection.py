# -*- coding: utf-8 -*-
"""
Dataset type detection helpers for the HyperCoast QGIS provider.

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

import os

# Filename tokens that map to a data type but are not simply the sensor's own
# name. The sensor name itself is always matched automatically, so only the
# non-obvious aliases need to be listed here.
EXTRA_NAME_TOKENS = {
    "PACE": ("OCI",),
    "PRISMA": ("PRS",),
    "Tanager": (
        "ORTHO_RADIANCE_HDF5",
        "BASIC_RADIANCE_HDF5",
        "ORTHO_SR_HDF5",
        "BASIC_SR_HDF5",
    ),
}

# Preferred detection order for the built-in sensors so that more specific
# tokens are matched first. Any additional sensor supplied through ``data_types``
# (for example a newly registered hypercoast sensor) is appended after these.
DEFAULT_DETECTION_ORDER = (
    "EMIT",
    "PACE",
    "DESIS",
    "NEON",
    "AVIRIS",
    "PRISMA",
    "EnMAP",
    "Tanager",
    "Wyvern",
)


def _detection_order(data_types):
    """Return the data types to test, in detection order.

    Args:
        data_types: Optional iterable of available data-type keys (for example
            the keys of the provider ``DATA_TYPES`` table, which is sourced from
            the hypercoast registry). When ``None``, the built-in order is used.

    Returns:
        list[str]: Data-type names to check, excluding ``Generic``.
    """
    if not data_types:
        return list(DEFAULT_DETECTION_ORDER)

    available = [name for name in data_types if name != "Generic"]
    ordered = [name for name in DEFAULT_DETECTION_ORDER if name in available]
    extra = sorted(name for name in available if name not in DEFAULT_DETECTION_ORDER)
    return ordered + extra


def _tokens_for(data_type):
    """Return the uppercase filename tokens that identify a data type.

    Args:
        data_type: Sensor data-type key (for example ``"EnMAP"``).

    Returns:
        list[str]: Uppercase substrings that indicate the data type.
    """
    tokens = [data_type.upper()]
    tokens.extend(EXTRA_NAME_TOKENS.get(data_type, ()))
    return tokens


def detect_data_type(filepath, tanager_hdf5_checker=None, data_types=None):
    """Detect a supported HyperCoast data type from a file path.

    Detection is registry-driven: any sensor present in ``data_types`` (which the
    provider sources from the hypercoast registry) is matched by its name, so new
    sensors added to the library are detected without changing this function.

    Args:
        filepath: Input file path.
        tanager_hdf5_checker: Optional callable used to inspect HDF5 content.
        data_types: Optional iterable of available data-type keys. When ``None``,
            the built-in sensor set is used.

    Returns:
        Detected data type name.
    """
    _, ext = os.path.splitext(str(filepath).lower())
    filename = os.path.basename(str(filepath)).upper()

    for data_type in _detection_order(data_types):
        tokens = _tokens_for(data_type)
        if any(token in filename for token in tokens):
            return data_type

    if ext in [".h5", ".hdf5"] and callable(tanager_hdf5_checker):
        if tanager_hdf5_checker():
            return "Tanager"

    return "Generic"
