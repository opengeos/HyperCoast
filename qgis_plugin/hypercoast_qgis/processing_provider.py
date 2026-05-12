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
        if not dataset.load():
            raise QgsProcessingException(dataset.last_error or "Failed to load dataset")
        return dataset

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
        ) as dst:
            dst.write(np.nan_to_num(arr, nan=0.0).astype("float32"))
        return output_path


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
        input_path = self.parameterAsFile(parameters, self.INPUT, context)
        data_type_index = self.parameterAsEnum(parameters, self.DATA_TYPE, context)
        data_type = DATA_TYPE_OPTIONS[data_type_index]
        wavelengths = [
            self.parameterAsDouble(parameters, self.RED, context),
            self.parameterAsDouble(parameters, self.GREEN, context),
            self.parameterAsDouble(parameters, self.BLUE, context),
        ]
        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        dataset = HyperspectralDataset(input_path, data_type)
        result = dataset.load_and_export(output_path, wavelengths=wavelengths)
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
        arr, height, width = self._data_cube(dataset)
        bands, _, _ = arr.shape
        n_components = min(n_components, bands)

        pixels = arr.reshape(bands, height * width).T
        valid = np.all(np.isfinite(pixels), axis=1)
        if not np.any(valid):
            raise QgsProcessingException("No finite pixels available for PCA")

        x = pixels[valid]
        x = x - np.nanmean(x, axis=0)
        _, singular_values, vt = np.linalg.svd(x, full_matrices=False)
        scores = np.zeros((pixels.shape[0], n_components), dtype="float32")
        scores[valid] = np.dot(x, vt[:n_components].T)
        for i in range(n_components):
            if singular_values[i] > 0:
                scores[valid, i] = scores[valid, i] / singular_values[i]

        components = scores.T.reshape(n_components, height, width)
        self._write_raster(dataset, output_path, components)
        return {self.OUTPUT: output_path}

    def createInstance(self):
        """Create a new algorithm instance."""
        return PCAAlgorithm()
