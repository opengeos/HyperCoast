# -*- coding: utf-8 -*-
"""
HyperCoast QGIS Plugin - Load Data Dialog

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

import os
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QDockWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QLineEdit,
    QFileDialog,
    QGroupBox,
    QDoubleSpinBox,
    QProgressBar,
    QMessageBox,
    QFormLayout,
    QListWidget,
    QListWidgetItem,
    QTextEdit,
    QTabWidget,
    QWidget,
)
from qgis.core import (
    QgsRasterLayer,
    QgsProject,
    QgsMessageLog,
    QgsCoordinateTransform,
    QgsCoordinateReferenceSystem,
    QgsContrastEnhancement,
    QgsMultiBandColorRenderer,
    QgsSingleBandGrayRenderer,
    Qgis,
)

from ..hyperspectral_provider import (
    HyperspectralDataset,
    DATA_TYPES,
    create_file_filter,
)
from ..cache_manager import create_generated_raster_path

DEFAULT_VALUE_RANGE = (0.0, 0.5)
TANAGER_VALUE_RANGE = (0.0, 100.0)


class DatasetPreviewWorker(QThread):
    """Worker thread for loading dataset metadata without blocking QGIS."""

    progress = pyqtSignal(int)
    finished = pyqtSignal(object, str)

    def __init__(self, filepath, data_type, parent=None):
        """Initialize the worker.

        Args:
            filepath: Dataset path to preview.
            data_type: Requested HyperCoast data type.
            parent: Optional parent QObject.
        """
        super().__init__(parent)
        self.filepath = filepath
        self.data_type = data_type

    def run(self):
        """Load dataset metadata in a background thread."""
        try:
            self.progress.emit(50)
            dataset = HyperspectralDataset(self.filepath, self.data_type)
            if dataset.load():
                self.progress.emit(80)
                self.finished.emit(dataset, "")
            else:
                self.finished.emit(None, dataset.last_error or "Unknown error")
        except Exception as exc:
            self.finished.emit(None, str(exc))


class DatasetLoadWorker(QThread):
    """Worker thread for loading and exporting an RGB raster."""

    progress = pyqtSignal(int)
    finished = pyqtSignal(object, str, str)

    def __init__(
        self,
        filepath,
        data_type,
        output_path,
        wavelengths,
        variable_name=None,
        existing_dataset=None,
        parent=None,
    ):
        """Initialize the worker.

        Args:
            filepath: Dataset path to load.
            data_type: Requested HyperCoast data type.
            output_path: GeoTIFF path to create.
            wavelengths: RGB wavelengths.
            variable_name: Data variable to export.
            existing_dataset: Optional preloaded dataset to reuse.
            parent: Optional parent QObject.
        """
        super().__init__(parent)
        self.filepath = filepath
        self.data_type = data_type
        self.output_path = output_path
        self.wavelengths = wavelengths
        self.variable_name = variable_name
        self.existing_dataset = existing_dataset

    def run(self):
        """Load/export the dataset in a background thread."""
        try:
            self.progress.emit(40)
            dataset = self.existing_dataset
            if dataset is None:
                dataset = HyperspectralDataset(self.filepath, self.data_type)
                dataset.set_selected_variable(self.variable_name)
                result = dataset.load_and_export(
                    self.output_path, wavelengths=self.wavelengths
                )
            else:
                dataset.set_selected_variable(self.variable_name)
                self.progress.emit(60)
                result = dataset.export_to_geotiff(
                    self.output_path, wavelengths=self.wavelengths
                )

            if result is None:
                detail = getattr(dataset, "last_error", None) or "Unknown error"
                self.finished.emit(None, "", detail)
                return

            self.progress.emit(80)
            self.finished.emit(dataset, self.output_path, "")
        except Exception as exc:
            self.finished.emit(None, "", str(exc))


class LoadDataDialog(QDockWidget):
    """Dockable panel for loading hyperspectral data."""

    def __init__(self, iface, plugin, parent=None):
        """Initialize the dialog.

        :param iface: QGIS interface
        :param plugin: Reference to the main plugin
        :param parent: Parent widget
        """
        super().__init__(parent or iface.mainWindow())
        self.iface = iface
        self.plugin = plugin
        self.dataset = None
        self._preview_worker = None
        self._load_worker = None
        self._pending_load_context = None

        self.setWindowTitle("Load Hyperspectral Data")
        self.setObjectName("HyperCoastLoadDataDock")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        content = QWidget(self)
        self.setWidget(content)
        layout = QVBoxLayout(content)

        # Tab widget for different loading options
        tab_widget = QTabWidget()

        # Tab 1: Load from file
        file_tab = QWidget()
        file_layout = QVBoxLayout(file_tab)

        # File selection
        file_group = QGroupBox("File Selection")
        file_group_layout = QFormLayout()

        file_row = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select hyperspectral data file...")
        file_row.addWidget(self.file_path_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_file)
        file_row.addWidget(browse_btn)

        file_group_layout.addRow("File:", file_row)

        # Data type selection
        self.data_type_combo = QComboBox()
        self.data_type_combo.addItem("Auto-detect", "auto")
        for dtype, info in DATA_TYPES.items():
            self.data_type_combo.addItem(f"{dtype} - {info['description']}", dtype)
        self.data_type_combo.currentIndexChanged.connect(self._clear_dataset_preview)
        file_group_layout.addRow("Data Type:", self.data_type_combo)

        file_group.setLayout(file_group_layout)
        file_layout.addWidget(file_group)

        # Visualization options
        vis_group = QGroupBox("Visualization Options")
        vis_layout = QFormLayout()

        self.variable_combo = QComboBox()
        self.variable_combo.setEnabled(False)
        self.variable_combo.addItem("Default variable", None)
        vis_layout.addRow("Variable:", self.variable_combo)

        # RGB wavelength selection
        rgb_layout = QHBoxLayout()
        self.red_spin = QDoubleSpinBox()
        self.red_spin.setRange(0, 5000)
        self.red_spin.setValue(650)
        self.red_spin.setSuffix(" nm")
        rgb_layout.addWidget(QLabel("R:"))
        rgb_layout.addWidget(self.red_spin)

        self.green_spin = QDoubleSpinBox()
        self.green_spin.setRange(0, 5000)
        self.green_spin.setValue(550)
        self.green_spin.setSuffix(" nm")
        rgb_layout.addWidget(QLabel("G:"))
        rgb_layout.addWidget(self.green_spin)

        self.blue_spin = QDoubleSpinBox()
        self.blue_spin.setRange(0, 5000)
        self.blue_spin.setValue(450)
        self.blue_spin.setSuffix(" nm")
        rgb_layout.addWidget(QLabel("B:"))
        rgb_layout.addWidget(self.blue_spin)

        vis_layout.addRow("RGB Wavelengths:", rgb_layout)

        # Value range
        range_layout = QHBoxLayout()
        self.vmin_spin = QDoubleSpinBox()
        self.vmin_spin.setRange(-1000, 1000)
        self.vmin_spin.setValue(0)
        self.vmin_spin.setDecimals(4)
        range_layout.addWidget(QLabel("Min:"))
        range_layout.addWidget(self.vmin_spin)

        self.vmax_spin = QDoubleSpinBox()
        self.vmax_spin.setRange(-1000, 1000)
        self.vmax_spin.setValue(DEFAULT_VALUE_RANGE[1])
        self.vmax_spin.setDecimals(4)
        range_layout.addWidget(QLabel("Max:"))
        range_layout.addWidget(self.vmax_spin)

        vis_layout.addRow("Value Range:", range_layout)

        # Layer name
        self.layer_name_edit = QLineEdit()
        self.layer_name_edit.setPlaceholderText("Auto-generated from filename")
        vis_layout.addRow("Layer Name:", self.layer_name_edit)

        vis_group.setLayout(vis_layout)
        file_layout.addWidget(vis_group)

        # Dataset info
        info_group = QGroupBox("Dataset Information")
        info_layout = QVBoxLayout()
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(150)
        info_layout.addWidget(self.info_text)

        self.preview_btn = QPushButton("Preview Dataset")
        self.preview_btn.clicked.connect(self.preview_dataset)
        info_layout.addWidget(self.preview_btn)

        info_group.setLayout(info_layout)
        file_layout.addWidget(info_group)

        file_layout.addStretch()
        tab_widget.addTab(file_tab, "From File")

        # Tab 2: Preset wavelength combinations
        presets_tab = QWidget()
        presets_layout = QVBoxLayout(presets_tab)

        presets_group = QGroupBox("Wavelength Presets")
        presets_group_layout = QVBoxLayout()

        self.presets_list = QListWidget()
        presets = [
            ("True Color (RGB)", 650, 550, 450),
            ("Color Infrared (CIR)", 850, 650, 550),
            ("False Color (Urban)", 2200, 850, 650),
            ("Agriculture", 850, 680, 550),
            ("Vegetation Analysis", 850, 680, 450),
            ("Water Bodies", 550, 480, 450),
            ("Geology", 2200, 1600, 850),
            ("Chlorophyll-a", 700, 680, 550),
        ]

        for name, r, g, b in presets:
            item = QListWidgetItem(f"{name}: R={r}nm, G={g}nm, B={b}nm")
            item.setData(Qt.ItemDataRole.UserRole, (r, g, b))
            self.presets_list.addItem(item)

        self.presets_list.currentItemChanged.connect(self.apply_preset)
        self.presets_list.itemDoubleClicked.connect(self.apply_preset)
        presets_group_layout.addWidget(self.presets_list)

        presets_group.setLayout(presets_group_layout)
        presets_layout.addWidget(presets_group)
        presets_layout.addStretch()

        tab_widget.addTab(presets_tab, "Presets")

        layout.addWidget(tab_widget)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Buttons
        button_layout = QHBoxLayout()

        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_data)
        button_layout.addWidget(self.load_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

    def browse_file(self):
        """Open file browser dialog."""
        file_filter = create_file_filter()
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Hyperspectral Data File", "", file_filter
        )

        if filepath:
            self.file_path_edit.setText(filepath)

            # Always update layer name based on the new filename
            basename = os.path.splitext(os.path.basename(filepath))[0]
            self.layer_name_edit.setText(basename)

            # Clear dataset info since we have a new file
            self.dataset = None
            self.info_text.clear()
            self._populate_variable_combo(None)
            self._apply_data_type_value_range(
                self._resolved_value_range_data_type(filepath)
            )

    def _clear_dataset_preview(self, *args):
        """Clear loaded preview state after the requested data type changes."""
        self.dataset = None
        self.info_text.clear()
        self._populate_variable_combo(None)
        self._apply_data_type_value_range(self._resolved_value_range_data_type())

    def _resolved_value_range_data_type(self, filepath=None):
        """Return the data type to use for visualization range defaults.

        Args:
            filepath: Optional data file path to inspect for auto-detection.

        Returns:
            Selected or auto-detected data type.
        """
        data_type = self.data_type_combo.currentData()
        if data_type != "auto":
            return data_type
        filepath = filepath or self.file_path_edit.text()
        if filepath and os.path.exists(filepath):
            return HyperspectralDataset(filepath, "auto").data_type
        return data_type

    def _apply_data_type_value_range(self, data_type):
        """Apply visualization range defaults for a data type.

        Args:
            data_type: Resolved HyperCoast data type.
        """
        vmin, vmax = (
            TANAGER_VALUE_RANGE if data_type == "Tanager" else DEFAULT_VALUE_RANGE
        )
        self.vmin_spin.setValue(vmin)
        self.vmax_spin.setValue(vmax)

    def preview_dataset(self):
        """Preview the selected dataset."""
        filepath = self.file_path_edit.text()
        if not filepath or not os.path.exists(filepath):
            QMessageBox.warning(self, "Error", "Please select a valid file.")
            return
        if self._preview_worker is not None and self._preview_worker.isRunning():
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(20)
        self.preview_btn.setEnabled(False)

        data_type = self.data_type_combo.currentData()
        QgsMessageLog.logMessage(
            f"Preview dataset requested: file={filepath}, selected_type={data_type}",
            "HyperCoast",
            Qgis.MessageLevel.Info,
        )
        self._preview_worker = DatasetPreviewWorker(filepath, data_type, self)
        self._preview_worker.progress.connect(self.progress_bar.setValue)
        self._preview_worker.finished.connect(self._on_preview_finished)
        self._preview_worker.start()

    def _on_preview_finished(self, dataset, error_detail):
        """Handle dataset preview completion.

        Args:
            dataset: Loaded HyperspectralDataset, or None on failure.
            error_detail: Error message when preview failed.
        """
        self.preview_btn.setEnabled(True)
        try:
            if dataset is not None:
                self.dataset = dataset
                self._apply_data_type_value_range(dataset.data_type)
                self.progress_bar.setValue(80)
                self.info_text.setText(self._format_dataset_info(dataset))
                self._populate_variable_combo(dataset)
                self.progress_bar.setValue(100)
                QgsMessageLog.logMessage(
                    f"Preview dataset succeeded: resolved_type={dataset.data_type}",
                    "HyperCoast",
                    Qgis.MessageLevel.Info,
                )
            else:
                self.dataset = None
                self._populate_variable_combo(None)
                QgsMessageLog.logMessage(
                    f"Preview dataset failed: details={error_detail}",
                    "HyperCoast",
                    Qgis.MessageLevel.Warning,
                )
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Failed to load dataset preview.\n\nDetails:\n{error_detail}",
                )
        finally:
            self.progress_bar.setVisible(False)

    def _format_dataset_info(self, dataset):
        """Return display text for loaded dataset metadata.

        Args:
            dataset: Loaded HyperspectralDataset.

        Returns:
            Formatted metadata text.
        """
        filepath = dataset.filepath
        info_lines = []
        info_lines.append(f"File: {os.path.basename(filepath)}")
        info_lines.append(f"Data Type: {dataset.data_type}")

        if dataset.wavelengths is not None:
            n_bands = len(dataset.wavelengths)
            info_lines.append(f"Bands: {n_bands}")
            try:
                wl_min = float(dataset.wavelengths.min())
                wl_max = float(dataset.wavelengths.max())
                info_lines.append(f"Wavelength Range: {wl_min:.1f} - {wl_max:.1f} nm")
            except Exception:
                info_lines.append("Wavelength Range: unavailable")

        if dataset.bounds:
            info_lines.append(f"Bounds: {dataset.bounds}")

        if dataset.crs:
            info_lines.append(f"CRS: {dataset.crs}")

        if dataset.dataset is not None:
            ds = dataset.dataset
            dims_str = ", ".join(f"{k}: {v}" for k, v in ds.dims.items())
            info_lines.append(f"Dimensions: {dims_str}")
            info_lines.append(f"Variables: {list(ds.data_vars)}")

        return "\n".join(info_lines)

    def _populate_variable_combo(self, dataset):
        """Populate the export variable combo from a loaded dataset.

        Args:
            dataset: Loaded HyperspectralDataset, or None to reset.
        """
        current = self.variable_combo.currentData()
        self.variable_combo.blockSignals(True)
        self.variable_combo.clear()
        self.variable_combo.addItem("Default variable", None)

        variables = dataset.list_data_variables() if dataset is not None else []
        for variable in variables:
            data_var = dataset.dataset[variable]
            dims = " x ".join(str(size) for size in data_var.shape)
            self.variable_combo.addItem(f"{variable} ({dims})", variable)

        if current in variables:
            index = self.variable_combo.findData(current)
            if index >= 0:
                self.variable_combo.setCurrentIndex(index)

        self.variable_combo.setEnabled(bool(variables))
        self.variable_combo.blockSignals(False)

    def _selected_variable_name(self):
        """Return the variable selected for export.

        Returns:
            Selected variable name, or None for the provider default.
        """
        return self.variable_combo.currentData()

    def apply_preset(self, item, previous=None):
        """Apply a wavelength preset."""
        if item is None:
            return
        r, g, b = item.data(Qt.ItemDataRole.UserRole)
        self.red_spin.setValue(r)
        self.green_spin.setValue(g)
        self.blue_spin.setValue(b)

    def load_data(self):
        """Load the hyperspectral data and add to QGIS."""
        filepath = self.file_path_edit.text()
        if not filepath or not os.path.exists(filepath):
            QMessageBox.warning(self, "Error", "Please select a valid file.")
            return
        if self._load_worker is not None and self._load_worker.isRunning():
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)
        self.load_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)

        try:
            requested_type = self.data_type_combo.currentData()
            QgsMessageLog.logMessage(
                "Load dataset requested: "
                f"file={filepath}, selected_type={requested_type}",
                "HyperCoast",
                Qgis.MessageLevel.Info,
            )
            type_changed = (
                self.dataset is not None
                and requested_type != "auto"
                and self.dataset.data_type != requested_type
            )

            # Get wavelengths for RGB composite
            wavelengths = [
                self.red_spin.value(),
                self.green_spin.value(),
                self.blue_spin.value(),
            ]
            variable_name = self._selected_variable_name()

            layer_name = (
                self.layer_name_edit.text()
                or os.path.splitext(os.path.basename(filepath))[0]
            )
            default_layer_name = os.path.splitext(os.path.basename(filepath))[0]
            if variable_name and layer_name == default_layer_name:
                layer_name = f"{layer_name} - {variable_name}"
            temp_path = self._create_temp_raster_path(layer_name, "rgb")

            # Load dataset if not already loaded, file changed, or previous load
            # left an uninitialized dataset object.
            needs_load = (
                self.dataset is None
                or self.dataset.filepath != filepath
                or self.dataset.dataset is None
                or type_changed
            )

            existing_dataset = None
            if not needs_load:
                if self.dataset.dataset is None:
                    raise ValueError(
                        "Dataset object is initialized but data is not loaded"
                    )
                existing_dataset = self.dataset
                existing_dataset.set_selected_variable(variable_name)

            self._pending_load_context = {
                "filepath": filepath,
                "layer_name": layer_name,
                "wavelengths": wavelengths,
                "variable_name": variable_name,
            }
            self._load_worker = DatasetLoadWorker(
                filepath,
                requested_type,
                temp_path,
                wavelengths,
                variable_name=variable_name,
                existing_dataset=existing_dataset,
                parent=self,
            )
            self._load_worker.progress.connect(self.progress_bar.setValue)
            self._load_worker.finished.connect(self._on_load_finished)
            self._load_worker.start()

        except Exception as e:
            self._finish_load_ui()
            QMessageBox.critical(self, "Error", f"Error loading data: {str(e)}")
            QgsMessageLog.logMessage(
                f"Error loading hyperspectral data: {str(e)}",
                "HyperCoast",
                Qgis.MessageLevel.Critical,
            )

    def _on_load_finished(self, dataset, temp_path, error_detail):
        """Add the exported raster to QGIS after the load worker completes.

        Args:
            dataset: Loaded HyperspectralDataset, or None on failure.
            temp_path: Exported GeoTIFF path.
            error_detail: Error message when loading failed.
        """
        try:
            context = self._pending_load_context or {}
            filepath = context.get("filepath", "")
            layer_name = context.get("layer_name", "HyperCoast")
            wavelengths = context.get("wavelengths", [])
            variable_name = context.get("variable_name")

            if dataset is None:
                QgsMessageLog.logMessage(
                    f"Load dataset failed: file={filepath}, details={error_detail}",
                    "HyperCoast",
                    Qgis.MessageLevel.Warning,
                )
                raise ValueError(f"Failed to load dataset. Details: {error_detail}")

            self.dataset = dataset
            self._apply_data_type_value_range(self.dataset.data_type)
            self.progress_bar.setValue(80)
            selected_data_var = self.dataset.get_data_variable()
            selected_variable = variable_name or getattr(
                selected_data_var, "name", None
            )

            raster_layer = QgsRasterLayer(temp_path, layer_name)

            if not raster_layer.isValid():
                raise ValueError("Created layer is not valid")

            # If export omitted CRS due PROJ conflicts, assign known dataset CRS.
            if (
                (not raster_layer.crs().isValid())
                and self.dataset is not None
                and self.dataset.crs
            ):
                known_crs = QgsCoordinateReferenceSystem(str(self.dataset.crs))
                if known_crs.isValid():
                    raster_layer.setCrs(known_crs)
                    QgsMessageLog.logMessage(
                        f"Assigned layer CRS from dataset metadata: {self.dataset.crs}",
                        "HyperCoast",
                        Qgis.MessageLevel.Info,
                    )

            # Set value range for better visualization
            provider = raster_layer.dataProvider()
            if provider:
                self._apply_nodata(provider, raster_layer.bandCount())
                vmin = self.vmin_spin.value()
                vmax = self.vmax_spin.value()

                if raster_layer.bandCount() >= 3:
                    renderer = QgsMultiBandColorRenderer(
                        raster_layer.dataProvider(), 1, 2, 3
                    )
                    renderer_bands = [1, 2, 3]
                else:
                    renderer = QgsSingleBandGrayRenderer(raster_layer.dataProvider(), 1)
                    renderer_bands = [1]

                for band_idx in renderer_bands:
                    ce = QgsContrastEnhancement(provider.dataType(band_idx))
                    ce.setContrastEnhancementAlgorithm(
                        QgsContrastEnhancement.StretchToMinimumMaximum
                    )
                    ce.setMinimumValue(vmin)
                    ce.setMaximumValue(vmax)

                    if raster_layer.bandCount() < 3:
                        renderer.setContrastEnhancement(ce)
                    elif band_idx == 1:
                        renderer.setRedContrastEnhancement(ce)
                    elif band_idx == 2:
                        renderer.setGreenContrastEnhancement(ce)
                    else:
                        renderer.setBlueContrastEnhancement(ce)

                raster_layer.setRenderer(renderer)

            self._set_layer_custom_properties(
                raster_layer, filepath, wavelengths, selected_variable
            )
            QgsProject.instance().addMapLayer(raster_layer)

            self.progress_bar.setValue(90)

            # Register with plugin for spectral analysis
            data_info = {
                "dataset": self.dataset,
                "filepath": filepath,
                "data_type": self.dataset.data_type,
                "wavelengths": self.dataset.wavelengths,
                "rgb_wavelengths": wavelengths,
                "selected_variable": selected_variable,
                "bounds": self.dataset.bounds,
                "crs": self.dataset.crs,
            }
            self.plugin.register_hyperspectral_layer(raster_layer.id(), data_info)

            self.progress_bar.setValue(100)

            # Reset spectral plot when loading a new dataset
            if self.plugin.spectral_plot_dialog is not None:
                self.plugin.spectral_plot_dialog.clear_all_spectra()

            # Zoom to layer with proper CRS transformation
            canvas = self.iface.mapCanvas()
            layer_extent = raster_layer.extent()
            layer_crs = raster_layer.crs()
            canvas_crs = canvas.mapSettings().destinationCrs()

            # Transform extent if CRS differs
            if layer_crs.isValid() and canvas_crs.isValid() and layer_crs != canvas_crs:
                transform = QgsCoordinateTransform(
                    layer_crs, canvas_crs, QgsProject.instance()
                )
                layer_extent = transform.transformBoundingBox(layer_extent)

            # Set extent with a small buffer for better visibility
            layer_extent.scale(1.05)
            canvas.setExtent(layer_extent)
            canvas.refresh()

            # Set the layer as active
            self.iface.setActiveLayer(raster_layer)

            QgsMessageLog.logMessage(
                f"Loaded hyperspectral layer: {layer_name}",
                "HyperCoast",
                Qgis.MessageLevel.Info,
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading data: {str(e)}")
            QgsMessageLog.logMessage(
                f"Error loading hyperspectral data: {str(e)}",
                "HyperCoast",
                Qgis.MessageLevel.Critical,
            )

        finally:
            self._finish_load_ui()

    def _finish_load_ui(self):
        """Restore the load panel controls after background work."""
        self.progress_bar.setVisible(False)
        self.load_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self._pending_load_context = None

    def _create_temp_raster_path(self, layer_name, suffix):
        """Create a unique temporary GeoTIFF path.

        Args:
            layer_name: QGIS layer name used as a readable prefix.
            suffix: Short output kind, such as ``"rgb"``.

        Returns:
            Path to a unique persistent cache GeoTIFF.
        """
        return create_generated_raster_path(
            layer_name, suffix, project=QgsProject.instance()
        )

    def _set_layer_custom_properties(
        self, raster_layer, filepath, wavelengths, selected_variable=None
    ):
        """Persist plugin state on a QGIS layer.

        Args:
            raster_layer: Layer receiving custom properties.
            filepath: Source hyperspectral dataset path.
            wavelengths: Selected RGB wavelengths.
            selected_variable: Data variable exported to the layer.
        """
        raster_layer.setCustomProperty("hypercoast/source_path", filepath)
        raster_layer.setCustomProperty("hypercoast/data_type", self.dataset.data_type)
        if selected_variable:
            raster_layer.setCustomProperty(
                "hypercoast/selected_variable", selected_variable
            )
        raster_layer.setCustomProperty(
            "hypercoast/rgb_wavelengths", ",".join(str(v) for v in wavelengths)
        )
        if self.dataset.crs:
            raster_layer.setCustomProperty("hypercoast/crs", str(self.dataset.crs))

    def _apply_nodata(self, provider, band_count):
        """Mark plugin-exported no-data values on a QGIS raster provider.

        Args:
            provider: QGIS raster data provider.
            band_count: Number of raster bands.
        """
        for band_idx in range(1, band_count + 1):
            try:
                provider.setNoDataValue(band_idx, -9999.0)
            except Exception:
                pass  # nosec B110

    def _stop_worker(self, worker):
        """Request a worker thread to stop and wait for it to finish.

        Args:
            worker: QThread instance, or None.
        """
        if worker is None:
            return
        try:
            if worker.isRunning():
                worker.requestInterruption()
                worker.quit()
                worker.wait(5000)
        except RuntimeError:
            pass  # nosec B110

    def closeEvent(self, event):
        """Stop background workers before the dock closes.

        Args:
            event: Qt close event.
        """
        self._stop_worker(self._preview_worker)
        self._stop_worker(self._load_worker)
        super().closeEvent(event)
