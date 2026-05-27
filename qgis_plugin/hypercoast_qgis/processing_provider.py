# -*- coding: utf-8 -*-
"""
HyperCoast QGIS Processing provider.

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

import os

import numpy as np
from qgis.PyQt.QtGui import QIcon
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFile,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterString,
    QgsProcessingProvider,
)

from .hyperspectral_provider import DATA_TYPES, HyperspectralDataset

DATA_TYPE_OPTIONS = ["auto"] + list(DATA_TYPES.keys())


def _file_behavior_file():
    """Return the QGIS file-parameter enum for file inputs."""
    behavior = getattr(QgsProcessingParameterFile, "Behavior", None)
    if behavior is not None:
        return behavior.File
    return QgsProcessingParameterFile.File


def _number_type(name):
    """Return a QGIS number-parameter enum value by name.

    Args:
        name: Number type name, such as ``"Double"`` or ``"Integer"``.

    Returns:
        QGIS number parameter enum value.
    """
    enum = getattr(QgsProcessingParameterNumber, "Type", None)
    if enum is not None:
        return getattr(enum, name)
    return getattr(QgsProcessingParameterNumber, name)


class HyperCoastProcessingProvider(QgsProcessingProvider):
    """Processing provider for HyperCoast batch algorithms."""

    def __init__(self, plugin_dir=None):
        """Initialize the provider.

        Args:
            plugin_dir: Path to the plugin directory.
        """
        super().__init__()
        self.plugin_dir = plugin_dir or os.path.dirname(__file__)

    def loadAlgorithms(self):
        """Register HyperCoast processing algorithms."""
        self.addAlgorithm(RGBCompositeAlgorithm())
        self.addAlgorithm(SingleBandExportAlgorithm())
        self.addAlgorithm(SpectralIndexAlgorithm())
        self.addAlgorithm(WaterQualityWorkflowAlgorithm())
        self.addAlgorithm(PCAAlgorithm())

    def id(self):
        """Return the provider ID."""
        return "hypercoast"

    def name(self):
        """Return the provider display name."""
        return "HyperCoast"

    def longName(self):
        """Return the provider long display name."""
        return "HyperCoast Hyperspectral Tools"

    def icon(self):
        """Return the provider icon."""
        icon_path = os.path.join(self.plugin_dir, "icons", "hypercoast.png")
        return QIcon(icon_path)


class BaseHyperCoastAlgorithm(QgsProcessingAlgorithm):
    """Base class for HyperCoast file-based processing algorithms."""

    INPUT = "INPUT"
    DATA_TYPE = "DATA_TYPE"
    VARIABLE = "VARIABLE"
    OUTPUT = "OUTPUT"

    def group(self):
        """Return the Processing group name."""
        return "Hyperspectral"

    def groupId(self):
        """Return the Processing group ID."""
        return "hyperspectral"

    def _add_common_inputs(self):
        """Add common input and data-type parameters."""
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT,
                "Input hyperspectral dataset",
                behavior=_file_behavior_file(),
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.DATA_TYPE,
                "Data type",
                options=DATA_TYPE_OPTIONS,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.VARIABLE,
                "Data variable (optional)",
                defaultValue="",
                optional=True,
            )
        )

    def _load_dataset(self, parameters, context):
        """Load the requested HyperCoast dataset.

        Args:
            parameters: Processing parameter dict.
            context: Processing context.

        Returns:
            Loaded HyperspectralDataset.

        Raises:
            QgsProcessingException: If the dataset cannot be loaded.
        """
        input_path = self.parameterAsFile(parameters, self.INPUT, context)
        data_type_index = self.parameterAsEnum(parameters, self.DATA_TYPE, context)
        data_type = DATA_TYPE_OPTIONS[data_type_index]
        dataset = HyperspectralDataset(input_path, data_type)
        dataset.set_selected_variable(self._selected_variable(parameters, context))
        if not dataset.load():
            raise QgsProcessingException(dataset.last_error or "Failed to load dataset")
        self._validate_selected_variable(dataset)
        return dataset

    def _selected_variable(self, parameters, context):
        """Return the optional Processing data variable parameter.

        Args:
            parameters: Processing parameter dict.
            context: Processing context.

        Returns:
            Selected variable name, or None.
        """
        try:
            value = self.parameterAsString(parameters, self.VARIABLE, context)
        except Exception:
            value = parameters.get(self.VARIABLE, "") if parameters else ""
        value = str(value).strip() if value is not None else ""
        return value or None

    def _validate_selected_variable(self, dataset):
        """Validate the explicitly selected data variable.

        Args:
            dataset: Loaded HyperspectralDataset.

        Raises:
            QgsProcessingException: If the requested variable cannot be used
                as raster data.
        """
        variable = dataset.selected_variable
        if not variable:
            return
        if dataset.dataset is None or variable not in dataset.dataset.data_vars:
            raise QgsProcessingException(f"Data variable not found: {variable}")
        if not dataset.is_exportable_variable(variable):
            raise QgsProcessingException(
                f"Data variable is not raster-like: {variable}"
            )

    def _data_cube(self, dataset):
        """Return a data cube arranged as bands, rows, columns.

        Args:
            dataset: Loaded HyperspectralDataset.

        Returns:
            Tuple of array, height, width.

        Raises:
            QgsProcessingException: If no raster-like data variable is available.
        """
        data_var = dataset.get_data_variable()
        if data_var is None:
            raise QgsProcessingException("No data variable found")
        arr = np.asarray(data_var.values, dtype="float32")
        if arr.ndim != 3:
            raise QgsProcessingException(
                f"Expected a 3D hyperspectral cube, got shape {arr.shape}"
            )

        dims = list(data_var.dims)
        band_axis = None
        for i, dim in enumerate(dims):
            if dim not in ("x", "y", "latitude", "longitude"):
                band_axis = i
                break
        if band_axis is None:
            raise QgsProcessingException("Could not identify spectral dimension")
        if band_axis != 0:
            arr = np.moveaxis(arr, band_axis, 0)
        _, height, width = arr.shape
        return arr, height, width

    def _write_raster(self, dataset, output_path, arr):
        """Write a raster array using dataset bounds and CRS.

        Args:
            dataset: Loaded HyperspectralDataset.
            output_path: Output GeoTIFF path.
            arr: Array shaped as bands, rows, columns or rows, columns.

        Returns:
            Output path.
        """
        import rasterio
        from rasterio.transform import from_bounds

        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        count, height, width = arr.shape
        bounds = dataset.bounds or (0, 0, width, height)
        transform = from_bounds(*bounds, width, height)
        nodata_value = -9999.0
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=count,
            dtype="float32",
            crs=dataset.crs or "EPSG:4326",
            transform=transform,
            nodata=nodata_value,
        ) as dst:
            dst.write(np.nan_to_num(arr, nan=nodata_value).astype("float32"))
        return output_path

    def _spectral_dim(self, data_var):
        """Return the spectral dimension name for a data variable.

        Args:
            data_var: xarray DataArray to inspect.

        Returns:
            Spectral dimension name.

        Raises:
            QgsProcessingException: If no spectral dimension is available.
        """
        for dim in ("wavelength", "wavelengths", "band"):
            if dim in data_var.dims:
                return dim
        raise QgsProcessingException("Dataset does not expose a spectral dimension")

    def _select_nearest_band(self, data_var, wavelength):
        """Select the nearest spectral band.

        Args:
            data_var: xarray DataArray with a spectral dimension.
            wavelength: Target wavelength in nanometers.

        Returns:
            Selected band as a NumPy array.
        """
        dim = self._spectral_dim(data_var)
        if dim in data_var.coords:
            return data_var.sel({dim: wavelength}, method="nearest").values
        axis = data_var.get_axis_num(dim)
        index = int(np.clip(round(wavelength), 0, data_var.shape[axis] - 1))
        return data_var.isel({dim: index}).values


class RGBCompositeAlgorithm(BaseHyperCoastAlgorithm):
    """Export a wavelength-based RGB composite."""

    RED = "RED"
    GREEN = "GREEN"
    BLUE = "BLUE"

    def name(self):
        """Return the algorithm ID."""
        return "rgb_composite"

    def displayName(self):
        """Return the algorithm display name."""
        return "Export RGB Composite"

    def shortHelpString(self):
        """Return help text."""
        return "Exports a three-band RGB GeoTIFF from selected wavelengths."

    def initAlgorithm(self, config=None):
        """Define algorithm parameters."""
        self._add_common_inputs()
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RED,
                "Red wavelength (nm)",
                type=_number_type("Double"),
                defaultValue=650,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.GREEN,
                "Green wavelength (nm)",
                type=_number_type("Double"),
                defaultValue=550,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.BLUE,
                "Blue wavelength (nm)",
                type=_number_type("Double"),
                defaultValue=450,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(self.OUTPUT, "Output")
        )

    def processAlgorithm(self, parameters, context, feedback):
        """Run RGB composite export."""
        wavelengths = [
            self.parameterAsDouble(parameters, self.RED, context),
            self.parameterAsDouble(parameters, self.GREEN, context),
            self.parameterAsDouble(parameters, self.BLUE, context),
        ]
        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        dataset = self._load_dataset(parameters, context)
        result = dataset.export_to_geotiff(output_path, wavelengths=wavelengths)
        if result is None:
            raise QgsProcessingException(dataset.last_error or "Failed to export RGB")
        return {self.OUTPUT: result}

    def createInstance(self):
        """Create a new algorithm instance."""
        return RGBCompositeAlgorithm()


class SingleBandExportAlgorithm(BaseHyperCoastAlgorithm):
    """Export one wavelength as a single-band GeoTIFF."""

    WAVELENGTH = "WAVELENGTH"

    def name(self):
        """Return the algorithm ID."""
        return "single_band"

    def displayName(self):
        """Return the algorithm display name."""
        return "Export Single Band"

    def shortHelpString(self):
        """Return help text."""
        return "Exports the nearest band to the requested wavelength."

    def initAlgorithm(self, config=None):
        """Define algorithm parameters."""
        self._add_common_inputs()
        self.addParameter(
            QgsProcessingParameterNumber(
                self.WAVELENGTH,
                "Wavelength (nm)",
                type=_number_type("Double"),
                defaultValue=550,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(self.OUTPUT, "Output")
        )

    def processAlgorithm(self, parameters, context, feedback):
        """Run single-band export."""
        dataset = self._load_dataset(parameters, context)
        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        wavelength = self.parameterAsDouble(parameters, self.WAVELENGTH, context)
        result = dataset.export_to_geotiff(output_path, wavelengths=[wavelength])
        if result is None:
            raise QgsProcessingException("Failed to export selected wavelength")
        return {self.OUTPUT: result}

    def createInstance(self):
        """Create a new algorithm instance."""
        return SingleBandExportAlgorithm()


class SpectralIndexAlgorithm(BaseHyperCoastAlgorithm):
    """Calculate common wavelength-based spectral indices."""

    INDEX = "INDEX"
    RED = "RED"
    GREEN = "GREEN"
    NIR = "NIR"

    INDEX_OPTIONS = ["NDVI", "NDWI"]

    def name(self):
        """Return the algorithm ID."""
        return "spectral_index"

    def displayName(self):
        """Return the algorithm display name."""
        return "Calculate Spectral Index"

    def shortHelpString(self):
        """Return help text."""
        return "Calculates NDVI or NDWI from nearest wavelength bands."

    def initAlgorithm(self, config=None):
        """Define algorithm parameters."""
        self._add_common_inputs()
        self.addParameter(
            QgsProcessingParameterEnum(
                self.INDEX, "Index", options=self.INDEX_OPTIONS, defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RED,
                "Red wavelength (nm)",
                type=_number_type("Double"),
                defaultValue=660,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.GREEN,
                "Green wavelength (nm)",
                type=_number_type("Double"),
                defaultValue=560,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.NIR,
                "NIR wavelength (nm)",
                type=_number_type("Double"),
                defaultValue=850,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(self.OUTPUT, "Output")
        )

    def processAlgorithm(self, parameters, context, feedback):
        """Run spectral index calculation."""
        dataset = self._load_dataset(parameters, context)
        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        index = self.INDEX_OPTIONS[
            self.parameterAsEnum(parameters, self.INDEX, context)
        ]
        data_var = dataset.get_data_variable()
        if data_var is None or "wavelength" not in data_var.dims:
            raise QgsProcessingException(
                "Dataset does not expose a wavelength dimension"
            )

        red = data_var.sel(
            wavelength=self.parameterAsDouble(parameters, self.RED, context),
            method="nearest",
        ).values.astype("float32")
        green = data_var.sel(
            wavelength=self.parameterAsDouble(parameters, self.GREEN, context),
            method="nearest",
        ).values.astype("float32")
        nir = data_var.sel(
            wavelength=self.parameterAsDouble(parameters, self.NIR, context),
            method="nearest",
        ).values.astype("float32")

        with np.errstate(divide="ignore", invalid="ignore"):
            if index == "NDWI":
                output = (green - nir) / (green + nir)
            else:
                output = (nir - red) / (nir + red)

        self._write_raster(dataset, output_path, output)
        return {self.OUTPUT: output_path}

    def createInstance(self):
        """Create a new algorithm instance."""
        return SpectralIndexAlgorithm()


class WaterQualityWorkflowAlgorithm(BaseHyperCoastAlgorithm):
    """Calculate coastal water-quality workflow presets."""

    WORKFLOW = "WORKFLOW"

    WORKFLOW_OPTIONS = [
        "NDWI water mask",
        "Chlorophyll-a proxy",
        "Turbidity proxy",
        "CDOM proxy",
        "Cyanobacteria proxy",
        "Spectral anomaly",
    ]
    WORKFLOW_PRESETS = {
        "NDWI water mask": ("normalized_difference", 560.0, 860.0),
        "Chlorophyll-a proxy": ("ratio", 560.0, 665.0),
        "Turbidity proxy": ("ratio", 665.0, 860.0),
        "CDOM proxy": ("ratio", 412.0, 555.0),
        "Cyanobacteria proxy": ("normalized_difference", 709.0, 665.0),
        "Spectral anomaly": ("anomaly", None, None),
    }

    def name(self):
        """Return the algorithm ID."""
        return "water_quality_workflow"

    def displayName(self):
        """Return the algorithm display name."""
        return "Water Quality Workflow"

    def shortHelpString(self):
        """Return help text."""
        return "Runs registry-aligned coastal water-quality workflow presets."

    def initAlgorithm(self, config=None):
        """Define algorithm parameters."""
        self._add_common_inputs()
        self.addParameter(
            QgsProcessingParameterEnum(
                self.WORKFLOW,
                "Workflow",
                options=self.WORKFLOW_OPTIONS,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(self.OUTPUT, "Output")
        )

    def processAlgorithm(self, parameters, context, feedback):
        """Run a water-quality workflow."""
        dataset = self._load_dataset(parameters, context)
        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        workflow = self.WORKFLOW_OPTIONS[
            self.parameterAsEnum(parameters, self.WORKFLOW, context)
        ]
        kind, first_wavelength, second_wavelength = self.WORKFLOW_PRESETS[workflow]
        data_var = dataset.get_data_variable()
        if data_var is None:
            raise QgsProcessingException("No data variable found")

        if kind == "anomaly":
            dim = self._spectral_dim(data_var)
            arr = data_var.astype("float32")
            mean = arr.mean(dim=dim, skipna=True)
            std = arr.std(dim=dim, skipna=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                output = np.abs((arr - mean) / std).max(dim=dim, skipna=True)
            output = output.values.astype("float32")
        else:
            first = self._select_nearest_band(data_var, first_wavelength).astype(
                "float32"
            )
            second = self._select_nearest_band(data_var, second_wavelength).astype(
                "float32"
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                if kind == "normalized_difference":
                    output = (first - second) / (first + second)
                else:
                    output = first / second

        self._write_raster(dataset, output_path, output)
        if feedback:
            feedback.setProgress(100)
        return {self.OUTPUT: output_path}

    def createInstance(self):
        """Create a new algorithm instance."""
        return WaterQualityWorkflowAlgorithm()


class PCAAlgorithm(BaseHyperCoastAlgorithm):
    """Calculate PCA component rasters from a hyperspectral cube."""

    COMPONENTS = "COMPONENTS"

    def name(self):
        """Return the algorithm ID."""
        return "pca"

    def displayName(self):
        """Return the algorithm display name."""
        return "PCA Components"

    def shortHelpString(self):
        """Return help text."""
        return "Calculates leading PCA component rasters from a hyperspectral cube."

    def initAlgorithm(self, config=None):
        """Define algorithm parameters."""
        self._add_common_inputs()
        self.addParameter(
            QgsProcessingParameterNumber(
                self.COMPONENTS,
                "Number of components",
                type=_number_type("Integer"),
                minValue=1,
                maxValue=20,
                defaultValue=3,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(self.OUTPUT, "Output")
        )

    def processAlgorithm(self, parameters, context, feedback):
        """Run PCA component export."""
        dataset = self._load_dataset(parameters, context)
        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        n_components = self.parameterAsInt(parameters, self.COMPONENTS, context)
        data_var = dataset.get_data_variable()
        if data_var is None:
            raise QgsProcessingException("No data variable found")
        if data_var.ndim != 3:
            raise QgsProcessingException(
                f"Expected a 3D hyperspectral cube, got shape {data_var.shape}"
            )
        dims = list(data_var.dims)
        spatial_axes = [
            i
            for i, dim in enumerate(dims)
            if dim in ("x", "y", "latitude", "longitude")
        ]
        if len(spatial_axes) < 2:
            raise QgsProcessingException("Could not identify spatial dimensions")
        height, width = [data_var.shape[i] for i in spatial_axes[:2]]
        total_pixels = height * width
        max_total_pixels = 50_000_000
        if total_pixels > max_total_pixels:
            raise QgsProcessingException(
                f"Raster has {total_pixels:,} pixels which exceeds the "
                f"PCA limit of {max_total_pixels:,}. Crop or downsample the "
                "input before running PCA."
            )
        if feedback and feedback.isCanceled():
            raise QgsProcessingException("PCA cancelled by user")

        arr, height, width = self._data_cube(dataset)
        bands, _, _ = arr.shape
        n_components = min(n_components, bands)
        if feedback:
            feedback.setProgress(20)

        pixels = arr.reshape(bands, height * width).T
        valid = np.all(np.isfinite(pixels), axis=1)
        n_valid = int(np.count_nonzero(valid))
        if n_valid == 0:
            raise QgsProcessingException("No finite pixels available for PCA")

        # Cap fitting set to keep memory bounded; project full cube in chunks.
        max_fit_pixels = 500_000
        valid_indices = np.flatnonzero(valid)
        if n_valid > max_fit_pixels:
            rng = np.random.default_rng(0)
            fit_indices = rng.choice(valid_indices, size=max_fit_pixels, replace=False)
        else:
            fit_indices = valid_indices
        if feedback and feedback.isCanceled():
            raise QgsProcessingException("PCA cancelled by user")
        if feedback:
            feedback.setProgress(40)

        fit_pixels = pixels[fit_indices].astype(np.float64, copy=False)
        mean = fit_pixels.mean(axis=0)
        fit_centered = fit_pixels - mean
        # Covariance-based PCA: eigendecompose a (bands x bands) matrix.
        cov = np.cov(fit_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        components_matrix = eigenvectors[:, order[:n_components]]
        eig_top = eigenvalues[order[:n_components]]
        scales = np.sqrt(np.maximum(eig_top, 0.0))

        scores = np.zeros((pixels.shape[0], n_components), dtype="float32")
        chunk = 200_000
        for start in range(0, valid_indices.size, chunk):
            if feedback and feedback.isCanceled():
                return {self.OUTPUT: output_path}
            stop = min(start + chunk, valid_indices.size)
            idx = valid_indices[start:stop]
            block = (
                pixels[idx].astype(np.float64, copy=False) - mean
            ) @ components_matrix
            for i, scale in enumerate(scales):
                if scale > 0:
                    block[:, i] /= scale
            scores[idx] = block.astype("float32")
            if feedback:
                progress = 50 + int(40 * stop / max(valid_indices.size, 1))
                feedback.setProgress(progress)

        components = scores.T.reshape(n_components, height, width)
        self._write_raster(dataset, output_path, components)
        if feedback:
            feedback.setProgress(100)
        return {self.OUTPUT: output_path}

    def createInstance(self):
        """Create a new algorithm instance."""
        return PCAAlgorithm()
