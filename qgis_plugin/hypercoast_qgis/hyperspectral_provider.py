# -*- coding: utf-8 -*-
"""
HyperCoast QGIS Plugin - Hyperspectral Data Provider

This module handles loading and processing various hyperspectral data formats.
It uses the hypercoast library when available for best compatibility.

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

import os
import platform
import sys
import traceback
import numpy as np

try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    import rasterio
    from rasterio.transform import from_bounds

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    from qgis.core import QgsMessageLog, Qgis

    HAS_QGIS = True
    LOG_INFO = Qgis.Info
    LOG_WARNING = Qgis.Warning
    LOG_CRITICAL = Qgis.Critical
except Exception:
    HAS_QGIS = False
    LOG_INFO = None
    LOG_WARNING = None
    LOG_CRITICAL = None

# Check if hypercoast is available
try:
    from ._hypercoast_lib import get_hypercoast

    hypercoast = get_hypercoast()

    HAS_HYPERCOAST = True
except Exception:
    HAS_HYPERCOAST = False


def _log(message, level=LOG_INFO):
    """Write logs to QGIS message panel when available."""
    text = str(message)
    if HAS_QGIS:
        QgsMessageLog.logMessage(
            text, "HyperCoast", level if level is not None else LOG_INFO
        )
    else:
        print(f"[HyperCoast] {text}")


# Supported data types
DATA_TYPES = {
    "EMIT": {
        "extensions": [".nc", ".nc4"],
        "description": "NASA EMIT L2A Reflectance",
        "variable": "reflectance",
    },
    "PACE": {
        "extensions": [".nc", ".nc4"],
        "description": "NASA PACE OCI L2 AOP",
        "variable": "Rrs",
    },
    "DESIS": {
        "extensions": [".nc", ".tif", ".tiff"],
        "description": "DESIS Hyperspectral",
        "variable": "reflectance",
    },
    "NEON": {
        "extensions": [".h5"],
        "description": "NEON AOP Hyperspectral",
        "variable": "reflectance",
    },
    "AVIRIS": {
        "extensions": [".nc", ".img", ".bil"],
        "description": "AVIRIS/AVIRIS-NG",
        "variable": "reflectance",
    },
    "PRISMA": {
        "extensions": [".he5", ".nc"],
        "description": "ASI PRISMA",
        "variable": "reflectance",
    },
    "EnMAP": {
        "extensions": [".nc", ".tif"],
        "description": "DLR EnMAP",
        "variable": "reflectance",
    },
    "Tanager": {
        "extensions": [".h5"],
        "description": "Planet Tanager",
        "variable": "toa_radiance",
    },
    "Wyvern": {
        "extensions": [".tif", ".tiff"],
        "description": "Wyvern Hyperspectral",
        "variable": "reflectance",
    },
    "Generic": {
        "extensions": [".tif", ".tiff", ".nc", ".nc4"],
        "description": "Generic Hyperspectral (GeoTIFF/NetCDF)",
        "variable": "data",
    },
}


class HyperspectralDataset:
    """Class to represent a hyperspectral dataset."""

    def __init__(self, filepath, data_type="auto"):
        """Initialize the hyperspectral dataset.

        :param filepath: Path to the hyperspectral data file
        :param data_type: Type of data ('EMIT', 'PACE', etc.) or 'auto' for auto-detection
        """
        self.filepath = filepath
        self.data_type = data_type if data_type != "auto" else self._detect_type()
        self.dataset = None
        self.wavelengths = None
        self.data_array = None
        self.bounds = None
        self.crs = None
        self.nodata = np.nan
        self._is_swath = False  # True for non-gridded data like PACE
        self.last_error = None

    def _detect_type(self):
        """Auto-detect the data type based on file extension and content."""
        _, ext = os.path.splitext(self.filepath.lower())
        filename = os.path.basename(self.filepath).upper()

        if "EMIT" in filename:
            return "EMIT"
        elif "PACE" in filename or "OCI" in filename:
            return "PACE"
        elif "DESIS" in filename:
            return "DESIS"
        elif "NEON" in filename:
            return "NEON"
        elif "AVIRIS" in filename:
            return "AVIRIS"
        elif "PRISMA" in filename or "PRS" in filename:
            return "PRISMA"
        elif "ENMAP" in filename:
            return "EnMAP"
        elif "TANAGER" in filename:
            return "Tanager"
        elif "WYVERN" in filename:
            return "Wyvern"
        else:
            return "Generic"

    def _log_windows_context(self, stage):
        """Log extra runtime context for Windows debugging."""
        if platform.system() != "Windows":
            return
        xarray_ver = getattr(xr, "__version__", "missing") if HAS_XARRAY else "missing"
        h5py_ver = getattr(h5py, "__version__", "missing") if HAS_H5PY else "missing"
        _log(
            f"[{stage}] Windows context | "
            f"file='{self.filepath}' | data_type='{self.data_type}' | "
            f"python='{sys.executable}' | py_ver='{sys.version.split()[0]}' | "
            f"xarray='{xarray_ver}' | h5py='{h5py_ver}' | "
            f"has_hypercoast={HAS_HYPERCOAST}"
        )

    def load(self):
        """Load the hyperspectral data based on its type."""
        if not HAS_XARRAY:
            raise ImportError("xarray is required to load hyperspectral data")

        self.last_error = None
        _log(
            f"Starting dataset load: type={self.data_type}, file={self.filepath}",
            LOG_INFO,
        )
        self._log_windows_context("load-start")

        # Use hypercoast library if available
        if HAS_HYPERCOAST:
            return self._load_with_hypercoast()

        # On Windows, h5py/hypercoast may be unavailable in-process due to
        # HDF5 DLL conflicts, but they work fine in the venv's own Python.
        # Run the loading in a subprocess and retrieve the result.
        if platform.system() == "Windows" and self.data_type != "Generic":
            if self._load_via_subprocess():
                return True

        # Fallback to manual loading
        loader = {
            "EMIT": self._load_emit_fallback,
            "PACE": self._load_pace_fallback,
            "DESIS": self._load_desis_fallback,
            "NEON": self._load_neon_fallback,
            "Generic": self._load_generic,
        }.get(self.data_type, self._load_generic)

        success = loader()
        if not success and self.last_error is None:
            self.last_error = f"Failed to load {self.data_type} dataset"
            _log(self.last_error, LOG_WARNING)
        elif success:
            _log(
                f"Dataset loaded successfully: type={self.data_type}, "
                f"has_dataset={self.dataset is not None}",
                LOG_INFO,
            )
        return success

    def _load_with_hypercoast(self):
        """Load data using the hypercoast library."""
        try:
            _log(
                f"Trying hypercoast loader: type={self.data_type}, file={self.filepath}",
                LOG_INFO,
            )
            if self.data_type == "EMIT":
                self.dataset = hypercoast.read_emit(self.filepath)
                self._extract_emit_metadata()

            elif self.data_type == "PACE":
                self.dataset = hypercoast.read_pace(self.filepath)
                self._is_swath = True
                self._extract_pace_metadata()

            elif self.data_type == "DESIS":
                self.dataset = hypercoast.read_desis(self.filepath)
                self._extract_desis_metadata()

            elif self.data_type == "NEON":
                self.dataset = hypercoast.read_neon(self.filepath)
                self._extract_neon_metadata()

            elif self.data_type == "AVIRIS":
                self.dataset = hypercoast.read_aviris(self.filepath)
                self._extract_generic_metadata()

            elif self.data_type == "PRISMA":
                self.dataset = hypercoast.read_prisma(self.filepath)
                self._extract_generic_metadata()

            elif self.data_type == "EnMAP":
                self.dataset = hypercoast.read_enmap(self.filepath)
                self._extract_generic_metadata()

            elif self.data_type == "Tanager":
                self.dataset = hypercoast.read_tanager(self.filepath)
                self._extract_tanager_metadata()

            elif self.data_type == "Wyvern":
                self.dataset = hypercoast.read_wyvern(self.filepath)
                self._extract_generic_metadata()

            else:
                return self._load_generic()

            # Guard against readers that return None instead of raising
            if self.dataset is None:
                _log(
                    "Hypercoast reader returned None; switching to generic fallback",
                    LOG_WARNING,
                )
                return self._load_generic()

            _log(
                f"Hypercoast loader succeeded: type={self.data_type}",
                LOG_INFO,
            )
            return True

        except Exception as e:
            self.last_error = f"Error loading with hypercoast: {e}"
            _log(self.last_error, LOG_WARNING)
            _log(traceback.format_exc(limit=2), LOG_WARNING)
            # Try type-specific fallback first, then generic.
            fallback = {
                "EMIT": self._load_emit_fallback,
                "PACE": self._load_pace_fallback,
                "DESIS": self._load_desis_fallback,
                "NEON": self._load_neon_fallback,
            }.get(self.data_type, self._load_generic)

            _log(
                f"Trying fallback loader for type={self.data_type}: {fallback.__name__}",
                LOG_INFO,
            )
            if fallback():
                _log(
                    f"Fallback loader succeeded for type={self.data_type}",
                    LOG_INFO,
                )
                return True
            return self._load_generic()

    def _load_via_subprocess(self):
        """Load a dataset by running hypercoast in the venv subprocess.

        On Windows the venv's h5py works in its own process (no HDF5 DLL
        conflict with QGIS).  This method runs the hypercoast reader in a
        subprocess, saves the result to a temp NetCDF file, then loads it
        back with xarray in the QGIS process.
        """
        from .core.venv_manager import run_in_venv

        reader_map = {
            "EMIT": "read_emit",
            "PACE": "read_pace",
            "DESIS": "read_desis",
            "NEON": "read_neon",
            "AVIRIS": "read_aviris",
            "PRISMA": "read_prisma",
            "EnMAP": "read_enmap",
            "Tanager": "read_tanager",
            "Wyvern": "read_wyvern",
        }
        reader_fn = reader_map.get(self.data_type)
        if reader_fn is None:
            return False

        import tempfile

        # Create a temp file name that does NOT already exist – the venv
        # subprocess will create it.  On Windows, NamedTemporaryFile keeps the
        # file open (and therefore locked) by default, so we use mkstemp,
        # close the fd immediately, delete the placeholder, and let the
        # subprocess create the file fresh.
        fd, tmp_nc = tempfile.mkstemp(suffix=".nc", prefix="_hc_subprocess_")
        os.close(fd)
        os.remove(tmp_nc)  # subprocess will create this path

        # Escape backslashes for Windows paths in the script
        fp = self.filepath.replace("\\", "\\\\")
        out = tmp_nc.replace("\\", "\\\\")

        script = (
            "import hypercoast, sys\n"
            f"ds = hypercoast.{reader_fn}(r'{fp}')\n"
            "if ds is None:\n"
            "    sys.exit(1)\n"
            "ds.load()\n"
            f"ds.to_netcdf(r'{out}')\n"
        )

        _log(
            f"Running hypercoast.{reader_fn} in venv subprocess",
            LOG_INFO,
        )
        rc, stdout, stderr = run_in_venv(script, timeout=120)

        if rc != 0:
            detail = (stderr or stdout or "unknown error").strip()[-300:]
            _log(
                f"Subprocess load failed (rc={rc}): {detail}",
                LOG_WARNING,
            )
            return False

        try:
            # Load into memory so the file handle is released before cleanup
            with xr.open_dataset(tmp_nc) as ds:
                self.dataset = ds.load()
            # Extract metadata using the same helpers as _load_with_hypercoast
            extractor = {
                "EMIT": self._extract_emit_metadata,
                "PACE": self._extract_pace_metadata,
                "DESIS": self._extract_desis_metadata,
                "NEON": self._extract_neon_metadata,
                "Tanager": self._extract_tanager_metadata,
            }.get(self.data_type, self._extract_generic_metadata)
            if self.data_type == "PACE":
                self._is_swath = True
            extractor()
            _log(
                f"Subprocess load succeeded: type={self.data_type}",
                LOG_INFO,
            )
            return True
        except Exception as exc:
            _log(f"Failed to read subprocess output: {exc}", LOG_WARNING)
            return False
        finally:
            try:
                os.remove(tmp_nc)
            except OSError:
                pass

    def _export_via_subprocess(self, output_path, wavelengths):
        """Export to GeoTIFF by running hypercoast in the venv subprocess.

        This avoids h5py DLL conflicts on Windows by performing the
        hypercoast export in a separate process that can load h5py.
        """
        from .core.venv_manager import run_in_venv

        fn_map = {
            "EMIT": "emit_to_image",
            "PACE": "pace_to_image",
            "DESIS": "desis_to_image",
            "NEON": "neon_to_image",
            "Tanager": "tanager_to_image",
        }
        to_image_fn = fn_map.get(self.data_type)
        if to_image_fn is None:
            return None

        reader_map = {
            "EMIT": "read_emit",
            "PACE": "read_pace",
            "DESIS": "read_desis",
            "NEON": "read_neon",
            "Tanager": "read_tanager",
        }
        reader_fn = reader_map.get(self.data_type)
        if reader_fn is None:
            return None

        if wavelengths is None:
            wavelengths = [650, 550, 450]

        fp = self.filepath.replace("\\", "\\\\")
        op = output_path.replace("\\", "\\\\")

        extra_args = ""
        if self.data_type == "PACE":
            extra_args = ", method='nearest'"

        script = (
            "import hypercoast, sys\n"
            "from leafmap import image_to_geotiff\n"
            f"ds = hypercoast.{reader_fn}(r'{fp}')\n"
            "if 'wavelength' not in ds.dims and 'wavelength' not in ds.coords:\n"
            "    print('NO_WAVELENGTH_DIM', file=sys.stderr)\n"
            "    sys.exit(2)\n"
            f"image = hypercoast.{to_image_fn}(ds, wavelengths={wavelengths!r}"
            f"{extra_args})\n"
            f"image_to_geotiff(image, r'{op}', dtype='float32')\n"
        )

        _log(
            f"Running hypercoast export in venv subprocess: {self.data_type}",
            LOG_INFO,
        )
        rc, stdout, stderr = run_in_venv(script, timeout=180)

        if rc != 0:
            detail = (stderr or stdout or "unknown error").strip()[-300:]
            if rc == 2 and "NO_WAVELENGTH_DIM" in (stderr or ""):
                _log(
                    "Subprocess export skipped: dataset has no wavelength dimension",
                    LOG_INFO,
                )
            else:
                _log(f"Subprocess export failed (rc={rc}): {detail}", LOG_WARNING)
            return None

        if os.path.isfile(output_path):
            _log(f"Subprocess export succeeded: {output_path}", LOG_INFO)
            return output_path

        _log("Subprocess export produced no output file", LOG_WARNING)
        return None

    def _extract_emit_metadata(self):
        """Extract metadata from EMIT dataset."""
        ds = self.dataset

        if "wavelength" in ds.coords:
            self.wavelengths = ds.coords["wavelength"].values

        if "latitude" in ds.coords and "longitude" in ds.coords:
            lat = ds.coords["latitude"].values
            lon = ds.coords["longitude"].values
            self.bounds = (
                float(np.nanmin(lon)),
                float(np.nanmin(lat)),
                float(np.nanmax(lon)),
                float(np.nanmax(lat)),
            )

        self.crs = "EPSG:4326"

    def _extract_pace_metadata(self):
        """Extract metadata from PACE dataset."""
        ds = self.dataset

        if "wavelength" in ds.coords:
            self.wavelengths = ds.coords["wavelength"].values

        # PACE has 2D lat/lon arrays
        if "latitude" in ds.coords:
            lat = ds["latitude"].values
            lon = ds["longitude"].values
            self.bounds = (
                float(np.nanmin(lon)),
                float(np.nanmin(lat)),
                float(np.nanmax(lon)),
                float(np.nanmax(lat)),
            )

        self.crs = "EPSG:4326"

    def _extract_desis_metadata(self):
        """Extract metadata from DESIS dataset."""
        ds = self.dataset

        if "wavelength" in ds.coords:
            self.wavelengths = np.array(ds.coords["wavelength"].values)

        try:
            self.crs = ds.attrs.get("crs", "EPSG:4326")
            if hasattr(ds, "rio") and ds.rio.crs:
                bounds = ds.rio.bounds()
                self.bounds = tuple(bounds)
        except Exception:
            self.crs = "EPSG:4326"

    def _extract_neon_metadata(self):
        """Extract metadata from NEON dataset."""
        ds = self.dataset

        if "wavelength" in ds.coords:
            self.wavelengths = np.array(ds.coords["wavelength"].values)

        self.crs = ds.attrs.get("crs", "EPSG:32618")

        if "x" in ds.coords and "y" in ds.coords:
            x = ds.coords["x"].values
            y = ds.coords["y"].values
            self.bounds = (
                float(np.min(x)),
                float(np.min(y)),
                float(np.max(x)),
                float(np.max(y)),
            )

    def _extract_tanager_metadata(self):
        """Extract metadata from Tanager dataset."""
        ds = self.dataset

        if "wavelength" in ds.coords:
            self.wavelengths = ds.coords["wavelength"].values

        if "latitude" in ds.coords:
            lat = ds["latitude"].values
            lon = ds["longitude"].values
            self.bounds = (
                float(np.nanmin(lon)),
                float(np.nanmin(lat)),
                float(np.nanmax(lon)),
                float(np.nanmax(lat)),
            )

        self.crs = "EPSG:4326"
        self._is_swath = True

    def _extract_generic_metadata(self):
        """Extract metadata from generic dataset."""
        ds = self.dataset

        # Try to find wavelength coordinate
        for coord_name in ["wavelength", "wavelengths", "band", "bands"]:
            if coord_name in ds.coords:
                self.wavelengths = np.array(ds.coords[coord_name].values)
                break

        # Try to get CRS and bounds
        try:
            if hasattr(ds, "rio") and ds.rio.crs:
                self.crs = str(ds.rio.crs)
                bounds = ds.rio.bounds()
                self.bounds = tuple(bounds)
            elif "crs" in ds.attrs:
                self.crs = ds.attrs["crs"]
        except Exception:
            self.crs = "EPSG:4326"

    def _load_emit_fallback(self):
        """Fallback EMIT loader without hypercoast."""
        try:
            _log(f"Trying EMIT fallback loader: file={self.filepath}", LOG_INFO)
            engine = self._select_group_engine(
                self.filepath, required_groups=["location", "sensor_band_parameters"]
            )
            _log(f"EMIT fallback selected engine={engine}", LOG_INFO)

            ds = xr.open_dataset(self.filepath, engine=engine)
            wvl = xr.open_dataset(
                self.filepath, group="sensor_band_parameters", engine=engine
            )
            loc = xr.open_dataset(self.filepath, group="location", engine=engine)

            ds = ds.assign_coords(
                {
                    "downtrack": (["downtrack"], ds.downtrack.data),
                    "crosstrack": (["crosstrack"], ds.crosstrack.data),
                    **wvl.variables,
                    **loc.variables,
                }
            )

            if "bands" in ds.dims:
                ds = ds.swap_dims({"bands": "wavelengths"})

            if "wavelengths" in ds.dims:
                ds = ds.rename({"wavelengths": "wavelength"})

            self.dataset = ds

            if "wavelength" in ds.coords:
                self.wavelengths = ds.coords["wavelength"].values
            elif "wavelengths" in ds.coords:
                self.wavelengths = ds.coords["wavelengths"].values

            if "lon" in loc and "lat" in loc:
                lon = loc["lon"].values
                lat = loc["lat"].values
                self.bounds = (
                    float(np.nanmin(lon)),
                    float(np.nanmin(lat)),
                    float(np.nanmax(lon)),
                    float(np.nanmax(lat)),
                )

            self.crs = "EPSG:4326"
            _log("EMIT fallback loader succeeded", LOG_INFO)
            return True

        except Exception as e:
            self.last_error = f"Error in EMIT fallback loader: {e}"
            _log(self.last_error, LOG_WARNING)
            return False

    def _load_pace_fallback(self):
        """Fallback PACE loader without hypercoast."""
        try:
            _log(f"Trying PACE fallback loader: file={self.filepath}", LOG_INFO)
            engine = self._select_group_engine(
                self.filepath,
                required_groups=[
                    "navigation_data",
                    "geophysical_data",
                    "sensor_band_parameters",
                ],
            )
            _log(f"PACE fallback selected engine={engine}", LOG_INFO)

            dataset = xr.open_dataset(
                self.filepath, group="navigation_data", engine=engine
            )
            dataset = dataset.set_coords(["latitude", "longitude"])

            product = xr.open_dataset(
                self.filepath, group="geophysical_data", engine=engine
            )
            band_params = xr.open_dataset(
                self.filepath, group="sensor_band_parameters", engine=engine
            )

            dataset = xr.merge(
                [dataset, product], join="outer", combine_attrs="drop_conflicts"
            )

            if "pixel_control_points" in dataset.dims:
                dataset = dataset.rename({"pixel_control_points": "pixels_per_line"})

            rename_dict = {
                "number_of_lines": "latitude",
                "pixels_per_line": "longitude",
            }
            if "wavelength_3d" in dataset.dims:
                rename_dict["wavelength_3d"] = "wavelength"
            dataset = dataset.rename(rename_dict)

            if "wavelength_3d" in band_params.coords and "wavelength" in dataset.dims:
                dataset = dataset.assign_coords(
                    wavelength=band_params.coords["wavelength_3d"].values
                )

            self.dataset = dataset
            self._is_swath = True

            if "wavelength" in dataset.coords:
                self.wavelengths = dataset.coords["wavelength"].values

            lat = dataset["latitude"].values
            lon = dataset["longitude"].values
            self.bounds = (
                float(np.nanmin(lon)),
                float(np.nanmin(lat)),
                float(np.nanmax(lon)),
                float(np.nanmax(lat)),
            )

            self.crs = "EPSG:4326"
            _log("PACE fallback loader succeeded", LOG_INFO)
            return True

        except Exception as e:
            self.last_error = f"Error in PACE fallback loader: {e}"
            _log(self.last_error, LOG_WARNING)
            return False

    def _load_desis_fallback(self):
        """Fallback DESIS loader."""
        return self._load_generic()

    def _load_neon_fallback(self):
        """Fallback NEON loader without hypercoast."""
        try:
            _log(f"Trying NEON fallback loader: file={self.filepath}", LOG_INFO)
            if not HAS_H5PY:
                raise ImportError("h5py is required to load NEON data")

            with h5py.File(self.filepath, "r") as f:
                site_code = list(f.keys())[0]
                site_refl = f[site_code]["Reflectance"]

                wavelengths_list = site_refl["Metadata"]["Spectral_Data"]["Wavelength"][
                    ()
                ].tolist()
                wavelengths_list = [round(num, 2) for num in wavelengths_list]

                epsg_code = site_refl["Metadata"]["Coordinate_System"]["EPSG Code"][()]
                epsg_code_number = int(epsg_code.decode("utf-8"))

                mapInfo_string = site_refl["Metadata"]["Coordinate_System"]["Map_Info"][
                    ()
                ].decode("utf-8")
                mapInfo_split = mapInfo_string.split(",")

                res = float(mapInfo_split[5]), float(mapInfo_split[6])

                site_reflArray = site_refl["Reflectance_Data"]
                refl_shape = site_reflArray.shape

                xMin = float(mapInfo_split[3])
                yMax = float(mapInfo_split[4])
                xMax = xMin + (refl_shape[1] * res[0])
                yMin = yMax - (refl_shape[0] * res[1])

                scaleFactor = site_reflArray.attrs["Scale_Factor"]
                noDataValue = site_reflArray.attrs["Data_Ignore_Value"]

                da = site_reflArray[:, :, :].astype(float)
                da[da == int(noDataValue)] = np.nan
                da[da < 0] = np.nan
                da[da > 10000] = np.nan
                da = da / scaleFactor

            self._build_neon_dataset(
                da,
                wavelengths_list,
                epsg_code_number,
                mapInfo_split,
                res,
                scaleFactor,
                noDataValue,
            )
            _log("NEON fallback loader succeeded", LOG_INFO)
            return True

        except Exception as e:
            self.last_error = f"Error in NEON fallback loader: {e}"
            _log(self.last_error, LOG_WARNING)
            _log(traceback.format_exc(limit=2), LOG_WARNING)
            return False

    def _build_neon_dataset(
        self,
        da,
        wavelengths_list,
        epsg_code_number,
        mapInfo_split,
        res,
        scaleFactor,
        noDataValue,
    ):
        """Assemble an xarray Dataset from raw NEON arrays."""
        xMin = float(mapInfo_split[3])
        yMax = float(mapInfo_split[4])
        xMax = xMin + (da.shape[1] * res[0])
        yMin = yMax - (da.shape[0] * res[1])

        coords = {
            "y": np.linspace(yMax, yMin, da.shape[0]),
            "x": np.linspace(xMin, xMax, da.shape[1]),
            "wavelength": wavelengths_list,
        }

        xda = xr.DataArray(
            da,
            coords=coords,
            dims=["y", "x", "wavelength"],
            attrs={
                "scale_factor": scaleFactor,
                "no_data_value": noDataValue,
                "crs": f"EPSG:{epsg_code_number}",
                "transform": (res[0], 0.0, xMin, 0.0, -res[1], yMax),
            },
        )

        self.dataset = xda.to_dataset(name="reflectance")
        self.dataset.attrs = xda.attrs
        self.wavelengths = np.array(wavelengths_list)
        self.bounds = (xMin, yMin, xMax, yMax)
        self.crs = f"EPSG:{epsg_code_number}"

    def _load_generic(self):
        """Load generic hyperspectral data (GeoTIFF or NetCDF)."""
        try:
            _log(f"Trying generic loader: file={self.filepath}", LOG_INFO)
            _, ext = os.path.splitext(self.filepath.lower())

            if ext in [".tif", ".tiff"]:
                import rioxarray

                ds = xr.open_dataset(self.filepath, engine="rasterio")

                if "band" in ds.dims:
                    n_bands = ds.dims["band"]
                    wavelengths = np.arange(1, n_bands + 1)
                    ds = ds.assign_coords({"wavelength": ("band", wavelengths)})
                    ds = ds.swap_dims({"band": "wavelength"})

                self.crs = str(ds.rio.crs) if ds.rio.crs else "EPSG:4326"
                bounds = ds.rio.bounds()
                self.bounds = tuple(bounds)

            else:
                ds = self._open_dataset_with_fallback_engines(self.filepath, ext)
                self.crs = "EPSG:4326"

            self.dataset = ds

            for coord_name in ["wavelength", "wavelengths", "band", "bands"]:
                if coord_name in ds.coords:
                    self.wavelengths = np.array(ds.coords[coord_name].values)
                    break

            _log(
                f"Generic loader succeeded: ext={ext}, has_wavelengths={self.wavelengths is not None}",
                LOG_INFO,
            )
            return True

        except Exception as e:
            self.last_error = f"Error loading generic data: {e}"
            _log(self.last_error, LOG_WARNING)
            _log(traceback.format_exc(limit=2), LOG_WARNING)
            return False

    def _open_dataset_with_fallback_engines(self, filepath, ext):
        """Open NetCDF/HDF datasets with multiple backend engines."""
        engines = [None, "h5netcdf", "netcdf4", "scipy"]
        errors = []

        for engine in engines:
            try:
                label = "auto" if engine is None else engine
                _log(
                    f"Trying xarray open_dataset engine={label} for file={filepath}",
                    LOG_INFO,
                )
                if engine is None:
                    return xr.open_dataset(filepath)
                return xr.open_dataset(filepath, engine=engine)
            except Exception as exc:
                errors.append(f"{label}: {exc}")
                _log(
                    f"xarray backend failed engine={label}: {exc}",
                    LOG_WARNING,
                )

        raise ValueError(
            f"Unable to open file '{filepath}' (ext={ext}) with any backend. "
            + " | ".join(errors)
        )

    def _select_group_engine(self, filepath, required_groups):
        """Select a backend engine that can open root + all required groups."""
        engines = ["h5netcdf", "netcdf4"]

        errors = []
        for engine in engines:
            opened = []
            try:
                _log(
                    f"Trying grouped dataset engine={engine} for file={filepath}",
                    LOG_INFO,
                )
                root = xr.open_dataset(filepath, engine=engine)
                opened.append(root)
                for group in required_groups:
                    ds = xr.open_dataset(filepath, group=group, engine=engine)
                    opened.append(ds)

                return engine
            except Exception as exc:
                errors.append(f"{engine}: {exc}")
                _log(
                    f"Grouped dataset engine failed engine={engine}: {exc}",
                    LOG_WARNING,
                )
            finally:
                for ds in opened:
                    try:
                        ds.close()
                    except Exception:
                        pass

        raise ValueError(
            "Unable to open grouped dataset with available backends. "
            + " | ".join(errors)
        )

    def get_data_variable(self):
        """Get the main data variable from the dataset."""
        if self.dataset is None:
            return None

        var_name = DATA_TYPES.get(self.data_type, {}).get("variable", "data")

        for name in [
            var_name,
            "reflectance",
            "Rrs",
            "toa_radiance",
            "data",
            "band_data",
        ]:
            if name in self.dataset.data_vars:
                return self.dataset[name]

        data_vars = list(self.dataset.data_vars)
        if data_vars:
            return self.dataset[data_vars[0]]

        return None

    def extract_spectral_signature(self, x, y, crs="EPSG:4326"):
        """Extract spectral signature at a given location.

        :param x: X coordinate (longitude)
        :param y: Y coordinate (latitude)
        :param crs: Coordinate reference system
        :returns: Tuple of (wavelengths, values) or (None, None)
        """
        if self.dataset is None or self.wavelengths is None:
            return None, None

        try:
            # Use hypercoast extraction functions if available
            if HAS_HYPERCOAST:
                return self._extract_with_hypercoast(x, y)

            # Fallback extraction
            data_var = self.get_data_variable()
            if data_var is None:
                return None, None

            if "latitude" in data_var.dims and "longitude" in data_var.dims:
                values = data_var.sel(latitude=y, longitude=x, method="nearest").values
            elif "y" in data_var.dims and "x" in data_var.dims:
                # May need coordinate transformation
                values = data_var.sel(y=y, x=x, method="nearest").values
            else:
                return None, None

            return self.wavelengths, values

        except Exception as e:
            _log(f"Error extracting spectral signature: {e}", LOG_WARNING)
            return None, None

    def _extract_with_hypercoast(self, lon, lat):
        """Extract spectral signature using hypercoast."""
        try:
            if self.data_type == "EMIT":
                values = (
                    self.dataset["reflectance"]
                    .sel(latitude=lat, longitude=lon, method="nearest")
                    .values
                )
                return self.wavelengths, values

            elif self.data_type == "PACE":
                da = hypercoast.extract_pace(self.dataset, lat, lon)
                return da.coords["wavelength"].values, da.values

            elif self.data_type == "DESIS":
                da = hypercoast.extract_desis(self.dataset, lat, lon)
                return da.coords["wavelength"].values, da.values

            elif self.data_type == "NEON":
                da = hypercoast.extract_neon(self.dataset, lat, lon)
                return da.coords["wavelength"].values, da.values

            else:
                data_var = self.get_data_variable()
                if "y" in data_var.dims and "x" in data_var.dims:
                    values = data_var.sel(y=lat, x=lon, method="nearest").values
                else:
                    values = data_var.sel(
                        latitude=lat, longitude=lon, method="nearest"
                    ).values
                return self.wavelengths, values

        except Exception as e:
            _log(f"Error in hypercoast extraction: {e}", LOG_WARNING)
            return None, None

    @staticmethod
    def _find_and_set_proj_data():
        """Locate proj.db and set PROJ_DATA if not already configured.

        Returns:
            The PROJ_DATA path that was set, or None if not found.
        """
        if os.environ.get("PROJ_DATA") or os.environ.get("PROJ_LIB"):
            return os.environ.get("PROJ_DATA") or os.environ.get("PROJ_LIB")

        candidates = []

        # pyproj bundled data (pip wheel)
        try:
            import pyproj

            candidates.append(
                os.path.join(
                    os.path.dirname(pyproj.__file__), "proj_dir", "share", "proj"
                )
            )
            # pyproj.datadir may know the path
            datadir_fn = getattr(getattr(pyproj, "datadir", None), "get_data_dir", None)
            if datadir_fn:
                try:
                    candidates.append(datadir_fn())
                except Exception:
                    pass
        except Exception:
            pass

        # QGIS / system locations
        if platform.system() == "Windows":
            exe_dir = os.path.dirname(sys.executable)
            candidates.append(os.path.join(os.path.dirname(exe_dir), "share", "proj"))
        else:
            candidates.extend(["/usr/share/proj", "/usr/local/share/proj"])

        for candidate in candidates:
            if candidate and os.path.isfile(os.path.join(candidate, "proj.db")):
                os.environ["PROJ_DATA"] = candidate
                _log(f"Set PROJ_DATA={candidate}", LOG_INFO)
                return candidate

        return None

    def export_to_geotiff(self, output_path, wavelengths=None, bands=None):
        """Export selected bands to a GeoTIFF file.

        :param output_path: Output file path
        :param wavelengths: List of wavelengths to export (takes priority)
        :param bands: List of band indices to export
        :returns: Path to the created file
        """
        if not HAS_RASTERIO:
            raise ImportError("rasterio is required to export GeoTIFF")

        if self.dataset is None:
            raise ValueError("No dataset loaded")

        _log(
            f"Exporting dataset to GeoTIFF: output={output_path}, "
            f"type={self.data_type}, has_hypercoast={HAS_HYPERCOAST}",
            LOG_INFO,
        )

        # Use hypercoast's export functions if available
        if HAS_HYPERCOAST:
            return self._export_with_hypercoast(output_path, wavelengths)

        # On Windows, try running the export in a venv subprocess where
        # h5py / hypercoast are available without DLL conflicts.
        if platform.system() == "Windows":
            result = self._export_via_subprocess(output_path, wavelengths)
            if result:
                return result

        return self._export_fallback(output_path, wavelengths, bands)

    def _export_with_hypercoast(self, output_path, wavelengths):
        """Export using hypercoast library."""
        try:
            from leafmap import array_to_image

            if wavelengths is None:
                wavelengths = [650, 550, 450]  # Default RGB

            if self.data_type == "EMIT":
                image = hypercoast.emit_to_image(self.dataset, wavelengths=wavelengths)

            elif self.data_type == "PACE":
                # PACE needs gridding first
                image = hypercoast.pace_to_image(
                    self.dataset, wavelengths=wavelengths, method="nearest"
                )

            elif self.data_type == "DESIS":
                image = hypercoast.desis_to_image(self.dataset, wavelengths=wavelengths)

            elif self.data_type == "NEON":
                image = hypercoast.neon_to_image(self.dataset, wavelengths=wavelengths)

            elif self.data_type == "Tanager":
                image = hypercoast.tanager_to_image(
                    self.dataset, wavelengths=wavelengths
                )

            else:
                return self._export_fallback(output_path, wavelengths, None)

            # Save the image to the output path
            from leafmap import image_to_geotiff

            image_to_geotiff(image, output_path, dtype="float32")

            return output_path

        except Exception as e:
            msg = str(e)
            # If PROJ can't find its database, try configuring PROJ_DATA and retry
            if "no database context" in msg or "proj_create" in msg:
                proj_data = self._find_and_set_proj_data()
                if proj_data:
                    try:
                        image_to_geotiff(image, output_path, dtype="float32")
                        return output_path
                    except Exception:
                        pass
            _log(f"Error in hypercoast export: {e}", LOG_WARNING)
            return self._export_fallback(output_path, wavelengths, None)

    def _export_fallback(self, output_path, wavelengths, bands):
        """Fallback export without hypercoast."""
        _log("Using fallback GeoTIFF export path", LOG_INFO)
        data_var = self.get_data_variable()
        if data_var is None:
            raise ValueError("No data variable found")

        # Select bands
        if wavelengths is not None and "wavelength" in data_var.dims:
            data = data_var.sel(wavelength=wavelengths, method="nearest")
        elif bands is not None:
            data = data_var.isel(wavelength=bands)
        else:
            # Default to first 3 bands
            if len(data_var.dims) == 3:
                wl_dim = [
                    d
                    for d in data_var.dims
                    if d not in ["x", "y", "latitude", "longitude"]
                ][0]
                data = data_var.isel({wl_dim: slice(0, 3)})
            else:
                data = data_var

        arr = data.values

        # Ensure correct shape (bands, height, width)
        if arr.ndim == 1:
            # 1-D data cannot be exported as a raster
            raise ValueError(
                f"Data variable has only 1 dimension {data.dims}; "
                "cannot export as GeoTIFF"
            )
        elif arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        elif arr.ndim == 3:
            # Find wavelength dimension and move to first
            dims = list(data.dims)
            for i, d in enumerate(dims):
                if d not in ["x", "y", "latitude", "longitude"]:
                    if i != 0:
                        arr = np.moveaxis(arr, i, 0)
                    break

        # Handle NaN values
        arr = np.nan_to_num(arr, nan=0.0)

        if arr.ndim < 3:
            raise ValueError(
                f"Cannot reshape data to (bands, height, width); shape is {arr.shape}"
            )
        n_bands, height, width = arr.shape

        # Create transform
        if self.bounds:
            transform = from_bounds(*self.bounds, width, height)
        else:
            transform = from_bounds(0, 0, width, height, width, height)

        def _write_geotiff(crs_value):
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=n_bands,
                dtype="float32",
                crs=crs_value,
                transform=transform,
            ) as dst:
                dst.write(arr.astype("float32"))

        try:
            _write_geotiff(self.crs or "EPSG:4326")
        except Exception as e:
            msg = str(e)
            proj_db_conflict = (
                "proj_create_from_database" in msg
                or "DATABASE.LAYOUT.VERSION" in msg
                or "It comes from another PROJ installation" in msg
                or "no database context" in msg
            )
            if proj_db_conflict:
                _log(
                    "GeoTIFF export hit PROJ DB mismatch while writing CRS; "
                    "retrying export without CRS metadata.",
                    LOG_WARNING,
                )
                _write_geotiff(None)
            else:
                raise

        _log(f"GeoTIFF export succeeded: {output_path}", LOG_INFO)

        return output_path


def get_supported_formats():
    """Get a list of supported file formats."""
    extensions = set()
    for data_type in DATA_TYPES.values():
        extensions.update(data_type["extensions"])
    return sorted(extensions)


def create_file_filter():
    """Create a file filter string for QFileDialog."""
    formats = get_supported_formats()
    ext_str = " ".join(f"*{ext}" for ext in formats)
    return f"Hyperspectral Files ({ext_str});;All Files (*.*)"


def check_dependencies():
    """Check and report on available dependencies."""
    deps = {
        "xarray": HAS_XARRAY,
        "rasterio": HAS_RASTERIO,
        "h5py": HAS_H5PY,
        "hypercoast": HAS_HYPERCOAST,
    }

    missing = [name for name, available in deps.items() if not available]

    if missing:
        _log(
            f"Warning: Missing optional dependencies: {', '.join(missing)}",
            LOG_WARNING,
        )
        if "hypercoast" in missing:
            _log(
                "Install hypercoast for best compatibility: pip install hypercoast",
                LOG_INFO,
            )

    return deps
