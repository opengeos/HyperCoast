# -*- coding: utf-8 -*-
"""
HyperCoast QGIS Plugin - Load Data Dialog

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

import os
import tempfile
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QLineEdit,
    QFileDialog,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QProgressBar,
    QMessageBox,
    QFormLayout,
    QListWidget,
    QListWidgetItem,
    QSplitter,
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
    Qgis,
)

from ..hyperspectral_provider import (
    HyperspectralDataset,
    DATA_TYPES,
    create_file_filter,
)


class LoadDataDialog(QDialog):
    """Dialog for loading hyperspectral data."""

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

        self.setWindowTitle("Load Hyperspectral Data")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)

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
        file_group_layout.addRow("Data Type:", self.data_type_combo)

        file_group.setLayout(file_group_layout)
        file_layout.addWidget(file_group)

        # Visualization options
        vis_group = QGroupBox("Visualization Options")
        vis_layout = QFormLayout()

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
        self.vmax_spin.setValue(0.5)
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

        preview_btn = QPushButton("Preview Dataset")
        preview_btn.clicked.connect(self.preview_dataset)
        info_layout.addWidget(preview_btn)

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
            item.setData(Qt.UserRole, (r, g, b))
            self.presets_list.addItem(item)

        self.presets_list.itemDoubleClicked.connect(self.apply_preset)
        presets_group_layout.addWidget(self.presets_list)

        apply_preset_btn = QPushButton("Apply Selected Preset")
        apply_preset_btn.clicked.connect(self.apply_selected_preset)
        presets_group_layout.addWidget(apply_preset_btn)

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

        load_btn = QPushButton("Load Data")
        load_btn.clicked.connect(self.load_data)
        button_layout.addWidget(load_btn)

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

    def preview_dataset(self):
        """Preview the selected dataset."""
        filepath = self.file_path_edit.text()
        if not filepath or not os.path.exists(filepath):
            QMessageBox.warning(self, "Error", "Please select a valid file.")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(20)

        try:
            data_type = self.data_type_combo.currentData()
            self.dataset = HyperspectralDataset(filepath, data_type)

            self.progress_bar.setValue(50)

            if self.dataset.load():
                self.progress_bar.setValue(80)

                # Display dataset info
                info_lines = []
                info_lines.append(f"File: {os.path.basename(filepath)}")
                info_lines.append(f"Data Type: {self.dataset.data_type}")

                if self.dataset.wavelengths is not None:
                    n_bands = len(self.dataset.wavelengths)
                    wl_min = self.dataset.wavelengths.min()
                    wl_max = self.dataset.wavelengths.max()
                    info_lines.append(f"Bands: {n_bands}")
                    info_lines.append(
                        f"Wavelength Range: {wl_min:.1f} - {wl_max:.1f} nm"
                    )

                if self.dataset.bounds:
                    info_lines.append(f"Bounds: {self.dataset.bounds}")

                if self.dataset.crs:
                    info_lines.append(f"CRS: {self.dataset.crs}")

                if self.dataset.dataset is not None:
                    ds = self.dataset.dataset
                    dims_str = ", ".join(f"{k}: {v}" for k, v in ds.dims.items())
                    info_lines.append(f"Dimensions: {dims_str}")
                    info_lines.append(f"Variables: {list(ds.data_vars)}")

                self.info_text.setText("\n".join(info_lines))
                self.progress_bar.setValue(100)
            else:
                QMessageBox.warning(self, "Error", "Failed to load dataset preview.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading dataset: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)

    def apply_preset(self, item):
        """Apply a wavelength preset."""
        r, g, b = item.data(Qt.UserRole)
        self.red_spin.setValue(r)
        self.green_spin.setValue(g)
        self.blue_spin.setValue(b)

    def apply_selected_preset(self):
        """Apply the selected preset from the list."""
        item = self.presets_list.currentItem()
        if item:
            self.apply_preset(item)

    def load_data(self):
        """Load the hyperspectral data and add to QGIS."""
        filepath = self.file_path_edit.text()
        if not filepath or not os.path.exists(filepath):
            QMessageBox.warning(self, "Error", "Please select a valid file.")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)

        try:
            # Load dataset if not already loaded
            if self.dataset is None or self.dataset.filepath != filepath:
                data_type = self.data_type_combo.currentData()
                self.dataset = HyperspectralDataset(filepath, data_type)

                if not self.dataset.load():
                    raise ValueError("Failed to load dataset")

            self.progress_bar.setValue(40)

            # Get wavelengths for RGB composite
            wavelengths = [
                self.red_spin.value(),
                self.green_spin.value(),
                self.blue_spin.value(),
            ]

            # Export to temporary GeoTIFF
            temp_dir = tempfile.gettempdir()
            layer_name = (
                self.layer_name_edit.text()
                or os.path.splitext(os.path.basename(filepath))[0]
            )
            temp_path = os.path.join(temp_dir, f"{layer_name}_rgb.tif")

            self.progress_bar.setValue(60)

            self.dataset.export_to_geotiff(temp_path, wavelengths=wavelengths)

            self.progress_bar.setValue(80)

            # Add to QGIS
            raster_layer = QgsRasterLayer(temp_path, layer_name)

            if not raster_layer.isValid():
                raise ValueError("Created layer is not valid")

            # Set value range for better visualization
            provider = raster_layer.dataProvider()
            if provider:
                vmin = self.vmin_spin.value()
                vmax = self.vmax_spin.value()

                # Apply contrast enhancement
                from qgis.core import QgsContrastEnhancement, QgsMultiBandColorRenderer

                renderer = QgsMultiBandColorRenderer(
                    raster_layer.dataProvider(), 1, 2, 3  # R, G, B band indices
                )

                for band_idx in [1, 2, 3]:
                    ce = QgsContrastEnhancement(provider.dataType(band_idx))
                    ce.setContrastEnhancementAlgorithm(
                        QgsContrastEnhancement.StretchToMinimumMaximum
                    )
                    ce.setMinimumValue(vmin)
                    ce.setMaximumValue(vmax)

                    if band_idx == 1:
                        renderer.setRedContrastEnhancement(ce)
                    elif band_idx == 2:
                        renderer.setGreenContrastEnhancement(ce)
                    else:
                        renderer.setBlueContrastEnhancement(ce)

                raster_layer.setRenderer(renderer)

            QgsProject.instance().addMapLayer(raster_layer)

            self.progress_bar.setValue(90)

            # Register with plugin for spectral analysis
            data_info = {
                "dataset": self.dataset,
                "filepath": filepath,
                "data_type": self.dataset.data_type,
                "wavelengths": self.dataset.wavelengths,
                "rgb_wavelengths": wavelengths,
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

            QMessageBox.information(
                self, "Success", f"Layer '{layer_name}' loaded successfully!"
            )

            QgsMessageLog.logMessage(
                f"Loaded hyperspectral layer: {layer_name}", "HyperCoast", Qgis.Info
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading data: {str(e)}")
            QgsMessageLog.logMessage(
                f"Error loading hyperspectral data: {str(e)}",
                "HyperCoast",
                Qgis.Critical,
            )

        finally:
            self.progress_bar.setVisible(False)
