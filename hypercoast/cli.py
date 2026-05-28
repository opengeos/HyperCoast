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
from .catalog import SearchResult
from .common import pca
from .registry import (
    download_sensor,
    extract_sensor,
    get_sensor,
    list_sensors,
    read_sensor,
    registry_as_dict,
    search_sensor,
    sensor_to_image,
)
from .summary import extract_spectra_to_csv, subset_dataset, summarize_dataset
from .workflows import apply_workflow, list_workflows


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

    registry = subparsers.add_parser(
        "registry",
        help="Show sensor registry metadata.",
    )
    registry.add_argument("--json", action="store_true", help="Print JSON output.")
    registry.set_defaults(func=_cmd_registry)

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
        "--workspace-output",
        help="Optional SearchResult workspace JSON output.",
    )
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

    validate = subparsers.add_parser(
        "validate",
        help="Validate a local dataset against registry metadata.",
    )
    _add_sensor_arg(validate)
    validate.add_argument("input", help="Input dataset path.")
    validate.add_argument(
        "--read",
        action="store_true",
        help="Open the dataset with the registered reader.",
    )
    validate.set_defaults(func=_cmd_validate)

    inspect = subparsers.add_parser(
        "inspect",
        help="Inspect a local dataset path and matching sensors.",
    )
    inspect.add_argument("input", help="Input dataset path.")
    inspect.add_argument("--sensor", help="Optional sensor name or alias.")
    inspect.add_argument("--json", action="store_true", help="Print JSON output.")
    inspect.set_defaults(func=_cmd_inspect)

    summarize = subparsers.add_parser(
        "summarize",
        help="Summarize a local hyperspectral dataset.",
    )
    summarize.add_argument("input", help="Input dataset path.")
    summarize.add_argument("--sensor", help="Optional sensor reader to use.")
    summarize.add_argument("--variable", help="Optional data variable.")
    summarize.add_argument("--json", action="store_true", help="Print JSON output.")
    summarize.set_defaults(func=_cmd_summarize)

    subset = subparsers.add_parser(
        "subset",
        help="Subset a local rectilinear dataset by bounding box.",
    )
    subset.add_argument("input", help="Input dataset path.")
    subset.add_argument("output", help="Output NetCDF path.")
    subset.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
    )
    subset.add_argument("--sensor", help="Optional sensor reader to use.")
    subset.add_argument("--variable", help="Optional data variable.")
    subset.set_defaults(func=_cmd_subset)

    workflows = subparsers.add_parser(
        "workflows",
        help="List workflow presets.",
    )
    workflows.add_argument("--json", action="store_true", help="Print JSON output.")
    workflows.set_defaults(func=_cmd_workflows)

    workflow = subparsers.add_parser(
        "workflow",
        help="Run a coastal workflow preset on a local dataset.",
    )
    workflow.add_argument("name", help="Workflow preset name.")
    workflow.add_argument("input", help="Input dataset path.")
    workflow.add_argument("output", help="Output NetCDF or CSV path.")
    workflow.add_argument("--sensor", help="Optional sensor reader to use.")
    workflow.add_argument("--variable", help="Optional data variable.")
    workflow.add_argument(
        "--format",
        choices=("netcdf", "csv"),
        help="Output format. Defaults to the output file extension.",
    )
    workflow.set_defaults(func=_cmd_workflow)

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

    spectra = subparsers.add_parser(
        "spectra",
        help="Extract spectra for point coordinates from a CSV file.",
    )
    _add_sensor_arg(spectra)
    spectra.add_argument("input", help="Input dataset path.")
    spectra.add_argument("--points", required=True, help="Input point CSV path.")
    spectra.add_argument("--output", required=True, help="Output long-form CSV path.")
    spectra.add_argument("--x-column", default="x", help="Point x/lon column.")
    spectra.add_argument("--y-column", default="y", help="Point y/lat column.")
    spectra.add_argument("--crs", default="EPSG:4326", help="Point coordinate CRS.")
    spectra.set_defaults(func=_cmd_spectra)

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


def _cmd_registry(args: argparse.Namespace) -> int:
    """Print registry metadata."""
    registry = registry_as_dict()
    if args.json:
        print(_json_default(registry))
        return 0
    for name, metadata in registry.items():
        extensions = ", ".join(metadata.get("extensions") or ["any"])
        variable = metadata.get("default_variable") or "data"
        rgb = metadata.get("default_rgb") or []
        rgb_text = ", ".join(str(value) for value in rgb) if rgb else "none"
        print(f"{name}: {extensions}; variable={variable}; rgb={rgb_text}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    """Run a remote sensor search."""
    query = {
        "bbox": args.bbox,
        "temporal": args.temporal,
        "count": args.count,
    }
    result = search_sensor(
        args.sensor,
        bbox=args.bbox,
        temporal=args.temporal,
        count=args.count,
        output=args.output,
        return_gdf=args.return_gdf,
    )
    workspace = SearchResult.from_result(result, sensor=args.sensor, query=query)
    if args.workspace_output:
        workspace.to_json(args.workspace_output)
    if args.return_gdf:
        result = result[0]
    print(_json_default(result))
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    """Download remote sensor data from a JSON item file."""
    try:
        with open(args.input, encoding="utf-8") as src:
            items = json.load(src)
    except FileNotFoundError:
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in {args.input}: {exc}", file=sys.stderr)
        return 1
    if isinstance(items, dict) and "items" in items:
        items = items["items"]

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


def _cmd_validate(args: argparse.Namespace) -> int:
    """Validate a local dataset path against a sensor registration."""
    path = Path(args.input)
    try:
        handler = get_sensor(args.sensor)
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1
    suffix = path.suffix.lower()
    extensions = tuple(ext.lower() for ext in handler.extensions)
    if extensions and suffix not in extensions:
        allowed = ", ".join(extensions)
        print(
            f"Error: {handler.name} does not list {suffix} files. "
            f"Supported extensions: {allowed}",
            file=sys.stderr,
        )
        return 1
    if args.read:
        try:
            read_sensor(args.sensor, path)
        except Exception as exc:
            print(f"Error: reader failed for {path}: {exc}", file=sys.stderr)
            return 1
    print(f"{path} is valid for {handler.name}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect a local dataset path."""
    path = Path(args.input)
    suffix = path.suffix.lower()
    matches = [
        name
        for name in list_sensors()
        if suffix in tuple(ext.lower() for ext in get_sensor(name).extensions)
    ]
    payload: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "suffix": suffix,
        "size_bytes": path.stat().st_size if path.exists() else None,
        "matched_sensors": matches,
    }
    if args.sensor:
        try:
            handler = get_sensor(args.sensor)
        except KeyError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        payload["sensor"] = handler.as_dict()
    if args.json:
        print(_json_default(payload))
    else:
        print(f"Path: {payload['path']}")
        print(f"Exists: {payload['exists']}")
        print(f"Suffix: {payload['suffix'] or 'none'}")
        print(f"Size: {payload['size_bytes']}")
        print("Matched sensors: " + (", ".join(matches) or "none"))
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    """Summarize a local dataset."""
    try:
        summary = summarize_dataset(
            args.input,
            sensor=args.sensor,
            variable=args.variable,
        )
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    payload = summary.as_dict()
    if args.json:
        print(_json_default(payload))
    else:
        print(f"Path: {payload['path']}")
        print(f"Exists: {payload['exists']}")
        print(f"Sensor: {payload['sensor'] or 'auto'}")
        print("Variables: " + (", ".join(payload["variables"]) or "none"))
        print(f"Selected variable: {payload['selected_variable'] or 'none'}")
        print(f"CRS: {payload['crs'] or 'unknown'}")
        print(f"Bounds: {payload['bounds'] or 'unknown'}")
        print(f"Wavelengths: {payload['wavelength_count']}")
        if payload["warnings"]:
            print("Warnings: " + "; ".join(payload["warnings"]))
    return 0


def _cmd_subset(args: argparse.Namespace) -> int:
    """Subset a local dataset."""
    try:
        output = subset_dataset(
            args.input,
            args.output,
            bbox=tuple(args.bbox),
            sensor=args.sensor,
            variable=args.variable,
        )
    except Exception as exc:
        print(f"Error: subset failed: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


def _cmd_workflows(args: argparse.Namespace) -> int:
    """Print workflow preset metadata."""
    workflows = list_workflows()
    if args.json:
        print(_json_default(workflows))
        return 0
    for name, metadata in workflows.items():
        print(f"{name}: {metadata['description']}")
    return 0


def _cmd_workflow(args: argparse.Namespace) -> int:
    """Run a workflow preset and write the result."""
    import xarray as xr

    try:
        data = (
            read_sensor(args.sensor, args.input)
            if args.sensor
            else xr.open_dataset(args.input)
        )
        result = apply_workflow(data, args.name, variable=args.variable)
    except Exception as exc:
        print(f"Error: workflow failed: {exc}", file=sys.stderr)
        return 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_format = args.format or (
        "csv" if output.suffix.lower() == ".csv" else "netcdf"
    )
    if output_format == "csv":
        name = result.name or args.name
        result.to_dataframe(name=name).reset_index().to_csv(output, index=False)
    else:
        result.to_netcdf(output)
    print(output)
    return 0


def _cmd_extract_spectrum(args: argparse.Namespace) -> int:
    """Extract a point spectrum to CSV."""
    spectrum = extract_sensor(args.sensor, args.input, lat=args.lat, lon=args.lon)
    _write_spectrum_csv(spectrum, args.output)
    print(args.output)
    return 0


def _cmd_spectra(args: argparse.Namespace) -> int:
    """Extract spectra for CSV point coordinates."""
    try:
        output = extract_spectra_to_csv(
            args.sensor,
            args.input,
            points_csv=args.points,
            output=args.output,
            x_column=args.x_column,
            y_column=args.y_column,
            crs=args.crs,
        )
    except Exception as exc:
        print(f"Error: spectra extraction failed: {exc}", file=sys.stderr)
        return 1
    print(output)
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
