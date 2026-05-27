# SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Command line interface for HyperCoast."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from . import __version__
from .common import pca
from .registry import (
    download_sensor,
    extract_sensor,
    get_sensor,
    list_sensors,
    search_sensor,
    sensor_to_image,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the HyperCoast command line interface.

    Args:
        argv: Optional argument sequence. If None, ``sys.argv`` is used.

    Returns:
        int: Process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="hypercoast",
        description="Utilities for HyperCoast data discovery and processing.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="Show package and runtime info.")
    info.set_defaults(func=_cmd_info)

    sensors = subparsers.add_parser("sensors", help="List registered sensors.")
    sensors.set_defaults(func=_cmd_sensors)

    search = subparsers.add_parser("search", help="Search remote sensor data.")
    _add_sensor_arg(search)
    search.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
    )
    search.add_argument(
        "--temporal",
        help="Temporal range, for example 2024-01-01/2024-01-31.",
    )
    search.add_argument("--count", type=int, default=10, help="Maximum result count.")
    search.add_argument("--output", help="Optional output file for search results.")
    search.add_argument(
        "--return-gdf",
        action="store_true",
        help="Return and optionally save GeoDataFrame output.",
    )
    search.set_defaults(func=_cmd_search)

    download = subparsers.add_parser("download", help="Download remote sensor data.")
    _add_sensor_arg(download)
    download.add_argument("input", help="JSON file with granules or STAC items.")
    download.add_argument("--out-dir", help="Output directory.")
    download.add_argument("--provider", help="NASA Earthdata provider.")
    download.add_argument("--threads", type=int, default=8, help="Download threads.")
    download.add_argument(
        "--asset",
        default="ortho_radiance_hdf5",
        help="Tanager STAC asset key.",
    )
    download.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files when supported.",
    )
    download.set_defaults(func=_cmd_download)

    rgb = subparsers.add_parser("rgb", help="Convert sensor data to an image.")
    _add_sensor_arg(rgb)
    rgb.add_argument("input", help="Input dataset path.")
    rgb.add_argument("output", help="Output image path.")
    rgb.add_argument("--wavelengths", nargs="+", type=float, help="Wavelengths to use.")
    rgb.add_argument("--method", default="nearest", help="Wavelength selection method.")
    rgb.set_defaults(func=_cmd_rgb)

    extract = subparsers.add_parser(
        "extract-spectrum",
        help="Extract a point spectrum from sensor data.",
    )
    _add_sensor_arg(extract)
    extract.add_argument("input", help="Input dataset path.")
    extract.add_argument("--lat", type=float, required=True, help="Latitude.")
    extract.add_argument("--lon", type=float, required=True, help="Longitude.")
    extract.add_argument("--output", required=True, help="Output CSV path.")
    extract.set_defaults(func=_cmd_extract_spectrum)

    pca_parser = subparsers.add_parser("pca", help="Run PCA on an input image.")
    pca_parser.add_argument("input", help="Input raster path.")
    pca_parser.add_argument("output", help="Output raster path.")
    pca_parser.add_argument(
        "--components",
        type=int,
        default=3,
        help="Number of PCA components.",
    )
    pca_parser.set_defaults(func=_cmd_pca)

    return parser


def _add_sensor_arg(parser: argparse.ArgumentParser) -> None:
    """Add a sensor argument to a command parser."""
    parser.add_argument("sensor", help="Sensor name or alias.")


def _cmd_info(_: argparse.Namespace) -> int:
    """Print package and runtime information."""
    print(f"HyperCoast {__version__}")
    print(f"Python {platform.python_version()}")
    print("Sensors: " + ", ".join(list_sensors()))
    return 0


def _cmd_sensors(_: argparse.Namespace) -> int:
    """Print registered sensor names and descriptions."""
    for name in list_sensors():
        handler = get_sensor(name)
        suffix = f" - {handler.description}" if handler.description else ""
        print(f"{name}{suffix}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    """Run a remote sensor search."""
    result = search_sensor(
        args.sensor,
        bbox=args.bbox,
        temporal=args.temporal,
        count=args.count,
        output=args.output,
        return_gdf=args.return_gdf,
    )
    if args.return_gdf:
        result = result[0]
    print(_json_default(result))
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    """Download remote sensor data from a JSON item file."""
    with open(args.input, encoding="utf-8") as src:
        items = json.load(src)

    if get_sensor(args.sensor).name == "tanager":
        result = download_sensor(
            args.sensor,
            items,
            asset=args.asset,
            out_dir=args.out_dir,
            overwrite=args.overwrite,
        )
    else:
        kwargs = {"out_dir": args.out_dir, "threads": args.threads}
        if args.provider and get_sensor(args.sensor).name == "pace":
            kwargs["provider"] = args.provider
        result = download_sensor(args.sensor, items, **kwargs)
    print(_json_default(result))
    return 0


def _cmd_rgb(args: argparse.Namespace) -> int:
    """Convert sensor data to an image."""
    sensor_to_image(
        args.sensor,
        args.input,
        output=args.output,
        wavelengths=args.wavelengths,
        method=args.method,
    )
    print(args.output)
    return 0


def _cmd_extract_spectrum(args: argparse.Namespace) -> int:
    """Extract a point spectrum to CSV."""
    spectrum = extract_sensor(args.sensor, args.input, lat=args.lat, lon=args.lon)
    _write_spectrum_csv(spectrum, args.output)
    print(args.output)
    return 0


def _cmd_pca(args: argparse.Namespace) -> int:
    """Run PCA on a raster."""
    pca(args.input, args.output, n_components=args.components)
    print(args.output)
    return 0


def _write_spectrum_csv(spectrum: Any, output: str) -> None:
    """Write an extracted spectrum to CSV.

    Args:
        spectrum: xarray DataArray-like spectrum.
        output: Output CSV path.
    """
    import pandas as pd

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(spectrum, "coords") and "wavelength" in spectrum.coords:
        df = pd.DataFrame(
            {
                "wavelength": spectrum.coords["wavelength"].values,
                "value": spectrum.values,
            }
        )
    else:
        df = spectrum.to_dataframe(name="value").reset_index()
    df.to_csv(path, index=False)


def _json_default(value: Any) -> str:
    """Serialize search results to JSON text."""
    return json.dumps(value, default=str, indent=2)


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
