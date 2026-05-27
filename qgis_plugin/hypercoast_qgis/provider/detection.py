# -*- coding: utf-8 -*-
"""
Dataset type detection helpers for the HyperCoast QGIS provider.

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

import os


def detect_data_type(filepath, tanager_hdf5_checker=None):
    """Detect a supported HyperCoast data type from a file path.

    Args:
        filepath: Input file path.
        tanager_hdf5_checker: Optional callable used to inspect HDF5 content.

    Returns:
        Detected data type name.
    """
    _, ext = os.path.splitext(str(filepath).lower())
    filename = os.path.basename(str(filepath)).upper()

    if "EMIT" in filename:
        return "EMIT"
    if "PACE" in filename or "OCI" in filename:
        return "PACE"
    if "DESIS" in filename:
        return "DESIS"
    if "NEON" in filename:
        return "NEON"
    if "AVIRIS" in filename:
        return "AVIRIS"
    if "PRISMA" in filename or "PRS" in filename:
        return "PRISMA"
    if "ENMAP" in filename:
        return "EnMAP"
    if (
        "TANAGER" in filename
        or "ORTHO_RADIANCE_HDF5" in filename
        or "BASIC_RADIANCE_HDF5" in filename
        or "ORTHO_SR_HDF5" in filename
        or "BASIC_SR_HDF5" in filename
    ):
        return "Tanager"
    if "WYVERN" in filename:
        return "Wyvern"
    if ext in [".h5", ".hdf5"] and callable(tanager_hdf5_checker):
        if tanager_hdf5_checker():
            return "Tanager"
    return "Generic"
