# SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Sensor registry for common HyperCoast workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import xarray as xr

from .aviris import aviris_to_image, extract_aviris, read_aviris
from .common import (
    download_aviris,
    download_ecostress,
    download_emit,
    download_pace,
    search_aviris,
    search_ecostress,
    search_emit,
    search_pace,
)
from .desis import desis_to_image, extract_desis, read_desis
from .emit import emit_to_image, read_emit
from .enmap import enmap_to_image, extract_enmap, read_enmap
from .neon import extract_neon, neon_to_image, read_neon
from .pace import extract_pace, pace_to_image, read_pace
from .prisma import extract_prisma, prisma_to_image, read_prisma
from .tanager import (
    download_tanager,
    extract_tanager,
    read_tanager,
    search_tanager,
    tanager_to_image,
)
from .wyvern import extract_wyvern, read_wyvern, wyvern_to_image


@dataclass(frozen=True)
class SensorHandler:
    """Describe a HyperCoast sensor workflow.

    Args:
        name: Canonical sensor name.
        read: Function used to read local sensor data.
        to_image: Function used to convert sensor data to an image.
        extract: Function used to extract a point spectrum.
        search: Function used to search remote sensor data.
        download: Function used to download remote sensor data.
        aliases: Additional case-insensitive names accepted for the sensor.
        description: Short human-readable description.
    """

    name: str
    read: Optional[Callable[..., xr.Dataset]] = None
    to_image: Optional[Callable[..., Any]] = None
    extract: Optional[Callable[..., xr.DataArray]] = None
    search: Optional[Callable[..., Any]] = None
    download: Optional[Callable[..., Any]] = None
    aliases: tuple[str, ...] = ()
    description: str = ""


SENSOR_REGISTRY: Dict[str, SensorHandler] = {}
_ALIASES: Dict[str, str] = {}


def register_sensor(handler: SensorHandler) -> SensorHandler:
    """Register a sensor handler.

    Args:
        handler: The sensor handler to register.

    Returns:
        SensorHandler: The registered handler.
    """
    key = _normalize_name(handler.name)
    SENSOR_REGISTRY[key] = handler
    _ALIASES[key] = key
    for alias in handler.aliases:
        _ALIASES[_normalize_name(alias)] = key
    return handler


def list_sensors() -> List[str]:
    """Return registered sensor names.

    Returns:
        list[str]: Sorted canonical sensor names.
    """
    return sorted(SENSOR_REGISTRY)


def get_sensor(name: str) -> SensorHandler:
    """Return the handler for a sensor name or alias.

    Args:
        name: Sensor name or alias.

    Returns:
        SensorHandler: Matching sensor handler.

    Raises:
        KeyError: If the sensor is not registered.
    """
    key = _normalize_name(name)
    if key not in _ALIASES:
        available = ", ".join(list_sensors())
        raise KeyError(f"Unsupported sensor '{name}'. Available sensors: {available}")
    return SENSOR_REGISTRY[_ALIASES[key]]


def read_sensor(sensor: str, source: Any, **kwargs: Any) -> xr.Dataset:
    """Read sensor data using the registered reader.

    Args:
        sensor: Sensor name or alias.
        source: Local source path or object accepted by the sensor reader.
        **kwargs: Additional reader keyword arguments.

    Returns:
        xr.Dataset: Loaded sensor dataset.

    Raises:
        NotImplementedError: If the sensor has no registered reader.
    """
    handler = get_sensor(sensor)
    if handler.read is None:
        raise NotImplementedError(f"{handler.name} has no registered reader.")
    return handler.read(source, **kwargs)


def sensor_to_image(
    sensor: str,
    data: Any,
    output: Optional[str] = None,
    **kwargs: Any,
):
    """Convert sensor data to an image using the registered converter.

    Args:
        sensor: Sensor name or alias.
        data: Dataset or local path accepted by the image converter.
        output: Optional output image path.
        **kwargs: Additional image converter keyword arguments.

    Returns:
        Any: The sensor converter return value.

    Raises:
        NotImplementedError: If the sensor has no registered image converter.
    """
    handler = get_sensor(sensor)
    if handler.to_image is None:
        raise NotImplementedError(f"{handler.name} has no registered image converter.")
    return handler.to_image(data, output=output, **kwargs)


def extract_sensor(
    sensor: str,
    data: Any,
    lat: float,
    lon: float,
    **kwargs: Any,
) -> xr.DataArray:
    """Extract a point spectrum using the registered extractor.

    Args:
        sensor: Sensor name or alias.
        data: Dataset or local path accepted by the sensor reader.
        lat: Latitude of the point to extract.
        lon: Longitude of the point to extract.
        **kwargs: Additional extractor keyword arguments.

    Returns:
        xr.DataArray: Extracted spectrum.

    Raises:
        NotImplementedError: If the sensor has no registered extractor.
    """
    handler = get_sensor(sensor)
    if handler.extract is None:
        raise NotImplementedError(f"{handler.name} has no registered extractor.")
    if isinstance(data, (str, bytes)):
        data = read_sensor(sensor, data)
    return handler.extract(data, lat=lat, lon=lon, **kwargs)


def search_sensor(sensor: str, **kwargs: Any) -> Any:
    """Search remote data using the registered search function.

    Args:
        sensor: Sensor name or alias.
        **kwargs: Search keyword arguments.

    Returns:
        Any: Search function result.

    Raises:
        NotImplementedError: If the sensor has no registered search function.
    """
    handler = get_sensor(sensor)
    if handler.search is None:
        raise NotImplementedError(f"{handler.name} has no registered search function.")
    return handler.search(**kwargs)


def download_sensor(sensor: str, items: Iterable[Any], **kwargs: Any) -> Any:
    """Download remote data using the registered download function.

    Args:
        sensor: Sensor name or alias.
        items: Granules or STAC items accepted by the download function.
        **kwargs: Download keyword arguments.

    Returns:
        Any: Download function result.

    Raises:
        NotImplementedError: If the sensor has no registered download function.
    """
    handler = get_sensor(sensor)
    if handler.download is None:
        raise NotImplementedError(
            f"{handler.name} has no registered download function."
        )
    return handler.download(items, **kwargs)


def _normalize_name(name: str) -> str:
    """Normalize a sensor name for registry lookup."""
    return str(name).lower().replace("_", "-").strip()


def _register_defaults() -> None:
    """Register the built-in HyperCoast sensors."""
    register_sensor(
        SensorHandler(
            name="aviris",
            read=read_aviris,
            to_image=aviris_to_image,
            extract=extract_aviris,
            search=search_aviris,
            download=download_aviris,
            aliases=("aviris-ng", "aviris3", "aviris5"),
            description="NASA AVIRIS and AVIRIS-NG hyperspectral data.",
        )
    )
    register_sensor(
        SensorHandler(
            name="desis",
            read=read_desis,
            to_image=desis_to_image,
            extract=extract_desis,
            description="DESIS hyperspectral data.",
        )
    )
    register_sensor(
        SensorHandler(
            name="emit",
            read=read_emit,
            to_image=emit_to_image,
            search=search_emit,
            download=download_emit,
            description="NASA EMIT hyperspectral data.",
        )
    )
    register_sensor(
        SensorHandler(
            name="enmap",
            read=read_enmap,
            to_image=enmap_to_image,
            extract=extract_enmap,
            description="DLR EnMAP hyperspectral data.",
        )
    )
    register_sensor(
        SensorHandler(
            name="ecostress",
            search=search_ecostress,
            download=download_ecostress,
            description="NASA ECOSTRESS data discovery and download.",
        )
    )
    register_sensor(
        SensorHandler(
            name="neon",
            read=read_neon,
            to_image=neon_to_image,
            extract=extract_neon,
            aliases=("neon-aop",),
            description="NEON AOP hyperspectral data.",
        )
    )
    register_sensor(
        SensorHandler(
            name="pace",
            read=read_pace,
            to_image=pace_to_image,
            extract=extract_pace,
            search=search_pace,
            download=download_pace,
            description="NASA PACE OCI data.",
        )
    )
    register_sensor(
        SensorHandler(
            name="prisma",
            read=read_prisma,
            to_image=prisma_to_image,
            extract=extract_prisma,
            description="ASI PRISMA hyperspectral data.",
        )
    )
    register_sensor(
        SensorHandler(
            name="tanager",
            read=read_tanager,
            to_image=tanager_to_image,
            extract=extract_tanager,
            search=search_tanager,
            download=download_tanager,
            aliases=("planet-tanager",),
            description="Planet Tanager STAC and HDF5 hyperspectral data.",
        )
    )
    register_sensor(
        SensorHandler(
            name="wyvern",
            read=read_wyvern,
            to_image=wyvern_to_image,
            extract=extract_wyvern,
            description="Wyvern hyperspectral GeoTIFF data.",
        )
    )


_register_defaults()


__all__ = [
    "SENSOR_REGISTRY",
    "SensorHandler",
    "download_sensor",
    "extract_sensor",
    "get_sensor",
    "list_sensors",
    "read_sensor",
    "register_sensor",
    "search_sensor",
    "sensor_to_image",
]
