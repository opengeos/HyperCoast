# -*- coding: utf-8 -*-
"""
Cache helpers for HyperCoast QGIS generated rasters.

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

import os
import uuid


def generated_raster_cache_dir(project=None):
    """Return the persistent cache directory for plugin-generated rasters.

    Args:
        project: Optional QgsProject-like object. This argument is accepted
            for caller compatibility and is not used.

    Returns:
        Absolute cache directory path.
    """
    _ = project
    cache_dir = os.path.join(os.path.expanduser("~"), ".qgis_hypercoast", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def sanitize_cache_name(name):
    """Return a filesystem-safe cache filename component.

    Args:
        name: User-facing layer name.

    Returns:
        Sanitized filename component.
    """
    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
    return safe_name.strip("_") or "hypercoast"


def create_generated_raster_path(layer_name, suffix, project=None):
    """Create a unique persistent path for a generated GeoTIFF.

    Args:
        layer_name: QGIS layer name used as a readable prefix.
        suffix: Short output kind, such as ``"rgb"`` or ``"bands"``.
        project: Optional QgsProject-like object.

    Returns:
        Unique GeoTIFF path in the HyperCoast cache directory.
    """
    cache_dir = generated_raster_cache_dir(project)
    safe_name = sanitize_cache_name(layer_name)
    safe_suffix = sanitize_cache_name(suffix)
    token = uuid.uuid4().hex[:10]
    return os.path.join(cache_dir, f"{safe_name}_{safe_suffix}_{token}.tif")
