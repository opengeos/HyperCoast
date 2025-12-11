# -*- coding: utf-8 -*-
"""
HyperCoast QGIS Plugin - Band Combination Dialog

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

import os
import tempfile
import numpy as np
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QGroupBox,
    QDoubleSpinBox,
    QSlider,
    QCheckBox,
    QFormLayout,
    QMessageBox,
    QProgressBar,
)
from qgis.core import (
    QgsProject,
    QgsRasterLayer,
    QgsContrastEnhancement,
    QgsMultiBandColorRenderer,
    QgsSingleBandGrayRenderer,
)


class BandCombinationDialog(QDialog):
    """Dialog for changing band combinations of hyperspectral layers."""

    def __init__(self, iface, plugin, parent=None):
        """Initialize the dialog.

        :param iface: QGIS interface
        :param plugin: Reference to the main plugin
        :param parent: Parent widget
        """
        super().__init__(parent or iface.mainWindow())
        self.iface = iface
        self.plugin = plugin

        self.setWindowTitle("Band Combination")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Layer selection
        layer_group = QGroupBox("Layer Selection")
        layer_layout = QFormLayout()

        self.layer_combo = QComboBox()
        self.layer_combo.currentIndexChanged.connect(self.on_layer_changed)
        layer_layout.addRow("Layer:", self.layer_combo)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        layer_layout.addRow("Info:", self.info_label)

        layer_group.setLayout(layer_layout)
        layout.addWidget(layer_group)

        # Band selection mode
        mode_group = QGroupBox("Display Mode")
        mode_layout = QHBoxLayout()

        self.rgb_radio = QRadioButton("RGB Composite")
        self.rgb_radio.setChecked(True)
        self.rgb_radio.toggled.connect(self.on_mode_changed)
        mode_layout.addWidget(self.rgb_radio)

        self.single_radio = QRadioButton("Single Band")
        self.single_radio.toggled.connect(self.on_mode_changed)
        mode_layout.addWidget(self.single_radio)

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # RGB band selection
        self.rgb_group = QGroupBox("RGB Wavelength Selection")
        rgb_layout = QFormLayout()

        # Red band
        red_layout = QHBoxLayout()
        self.red_spin = QDoubleSpinBox()
        self.red_spin.setRange(0, 5000)
        self.red_spin.setValue(650)
        self.red_spin.setSuffix(" nm")
        self.red_spin.valueChanged.connect(self.update_red_slider)
        red_layout.addWidget(self.red_spin)

        self.red_slider = QSlider(Qt.Horizontal)
        self.red_slider.setRange(0, 5000)
        self.red_slider.setValue(650)
        self.red_slider.valueChanged.connect(lambda v: self.red_spin.setValue(v))
        red_layout.addWidget(self.red_slider)

        rgb_layout.addRow("Red:", red_layout)

        # Green band
        green_layout = QHBoxLayout()
        self.green_spin = QDoubleSpinBox()
        self.green_spin.setRange(0, 5000)
        self.green_spin.setValue(550)
        self.green_spin.setSuffix(" nm")
        self.green_spin.valueChanged.connect(self.update_green_slider)
        green_layout.addWidget(self.green_spin)

        self.green_slider = QSlider(Qt.Horizontal)
        self.green_slider.setRange(0, 5000)
        self.green_slider.setValue(550)
        self.green_slider.valueChanged.connect(lambda v: self.green_spin.setValue(v))
        green_layout.addWidget(self.green_slider)

        rgb_layout.addRow("Green:", green_layout)

        # Blue band
        blue_layout = QHBoxLayout()
        self.blue_spin = QDoubleSpinBox()
        self.blue_spin.setRange(0, 5000)
        self.blue_spin.setValue(450)
        self.blue_spin.setSuffix(" nm")
        self.blue_spin.valueChanged.connect(self.update_blue_slider)
        blue_layout.addWidget(self.blue_spin)

        self.blue_slider = QSlider(Qt.Horizontal)
        self.blue_slider.setRange(0, 5000)
        self.blue_slider.setValue(450)
        self.blue_slider.valueChanged.connect(lambda v: self.blue_spin.setValue(v))
        blue_layout.addWidget(self.blue_slider)

        rgb_layout.addRow("Blue:", blue_layout)

        self.rgb_group.setLayout(rgb_layout)
        layout.addWidget(self.rgb_group)

        # Single band selection
        self.single_group = QGroupBox("Single Band Selection")
        single_layout = QFormLayout()

        single_wl_layout = QHBoxLayout()
        self.single_spin = QDoubleSpinBox()
        self.single_spin.setRange(0, 5000)
        self.single_spin.setValue(550)
        self.single_spin.setSuffix(" nm")
        single_wl_layout.addWidget(self.single_spin)

        self.single_slider = QSlider(Qt.Horizontal)
        self.single_slider.setRange(0, 5000)
        self.single_slider.setValue(550)
        self.single_slider.valueChanged.connect(lambda v: self.single_spin.setValue(v))
        single_wl_layout.addWidget(self.single_slider)

        single_layout.addRow("Wavelength:", single_wl_layout)

        # Colormap selection
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(
            [
                "Grayscale",
                "Jet",
                "Viridis",
                "Plasma",
                "Inferno",
                "Magma",
                "Hot",
                "Cool",
                "Rainbow",
                "Spectral",
            ]
        )
        single_layout.addRow("Colormap:", self.colormap_combo)

        self.single_group.setLayout(single_layout)
        self.single_group.setVisible(False)
        layout.addWidget(self.single_group)

        # Value range
        range_group = QGroupBox("Value Range")
        range_layout = QFormLayout()

        range_row = QHBoxLayout()
        self.vmin_spin = QDoubleSpinBox()
        self.vmin_spin.setRange(-1000, 1000)
        self.vmin_spin.setValue(0)
        self.vmin_spin.setDecimals(4)
        range_row.addWidget(QLabel("Min:"))
        range_row.addWidget(self.vmin_spin)

        self.vmax_spin = QDoubleSpinBox()
        self.vmax_spin.setRange(-1000, 1000)
        self.vmax_spin.setValue(0.5)
        self.vmax_spin.setDecimals(4)
        range_row.addWidget(QLabel("Max:"))
        range_row.addWidget(self.vmax_spin)

        range_layout.addRow("Range:", range_row)

        self.auto_stretch = QCheckBox("Auto-stretch to layer statistics")
        range_layout.addRow("", self.auto_stretch)

        range_group.setLayout(range_layout)
        layout.addWidget(range_group)

        # Presets
        presets_group = QGroupBox("Quick Presets")
        presets_layout = QHBoxLayout()

        presets = [
            ("True Color", 650, 550, 450),
            ("CIR", 850, 650, 550),
            ("False Color", 2200, 850, 650),
            ("Vegetation", 850, 680, 450),
        ]

        for name, r, g, b in presets:
            btn = QPushButton(name)
            btn.clicked.connect(
                lambda checked, r=r, g=g, b=b: self.apply_preset(r, g, b)
            )
            presets_layout.addWidget(btn)

        presets_group.setLayout(presets_layout)
        layout.addWidget(presets_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Buttons
        button_layout = QHBoxLayout()

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_band_combination)
        button_layout.addWidget(apply_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

    def refresh_layers(self):
        """Refresh the list of available hyperspectral layers."""
        self.layer_combo.clear()

        hyper_layers = self.plugin.get_all_hyperspectral_layers()

        for layer_id, data_info in hyper_layers.items():
            layer = QgsProject.instance().mapLayer(layer_id)
            if layer:
                self.layer_combo.addItem(layer.name(), layer_id)

        if self.layer_combo.count() == 0:
            self.layer_combo.addItem("No hyperspectral layers loaded", None)

    def on_layer_changed(self, index):
        """Handle layer selection change."""
        layer_id = self.layer_combo.currentData()
        if not layer_id:
            self.info_label.setText("No layer selected")
            return

        data_info = self.plugin.get_hyperspectral_data(layer_id)
        if data_info:
            wavelengths = data_info.get("wavelengths")
            if wavelengths is not None and len(wavelengths) > 0:
                wl_min = float(np.min(wavelengths))
                wl_max = float(np.max(wavelengths))
                n_bands = len(wavelengths)

                self.info_label.setText(
                    f"Type: {data_info.get('data_type', 'Unknown')}\n"
                    f"Bands: {n_bands}\n"
                    f"Range: {wl_min:.1f} - {wl_max:.1f} nm"
                )

                # Update slider ranges
                for slider in [
                    self.red_slider,
                    self.green_slider,
                    self.blue_slider,
                    self.single_slider,
                ]:
                    slider.setRange(int(wl_min), int(wl_max))

                for spin in [
                    self.red_spin,
                    self.green_spin,
                    self.blue_spin,
                    self.single_spin,
                ]:
                    spin.setRange(wl_min, wl_max)
            else:
                self.info_label.setText(
                    f"Type: {data_info.get('data_type', 'Unknown')}"
                )

    def on_mode_changed(self, state):
        """Handle display mode change."""
        # Enforce mutual exclusivity between the two checkboxes
        if self.rgb_radio.isChecked():
            self.single_radio.setChecked(False)
        elif self.single_radio.isChecked():
            self.rgb_radio.setChecked(False)
        else:
            # If both are unchecked, default to RGB mode
            self.rgb_radio.setChecked(True)
            self.single_radio.setChecked(False)
        is_rgb = self.rgb_radio.isChecked()
        self.rgb_group.setVisible(is_rgb)
        self.single_group.setVisible(not is_rgb)

    def update_red_slider(self, value):
        """Update red slider from spinbox."""
        self.red_slider.setValue(int(value))

    def update_green_slider(self, value):
        """Update green slider from spinbox."""
        self.green_slider.setValue(int(value))

    def update_blue_slider(self, value):
        """Update blue slider from spinbox."""
        self.blue_slider.setValue(int(value))

    def apply_preset(self, r, g, b):
        """Apply a wavelength preset."""
        self.rgb_radio.setChecked(True)
        self.red_spin.setValue(r)
        self.green_spin.setValue(g)
        self.blue_spin.setValue(b)

    def apply_band_combination(self):
        """Apply the selected band combination to the layer."""
        layer_id = self.layer_combo.currentData()
        if not layer_id:
            QMessageBox.warning(self, "Error", "Please select a layer.")
            return

        data_info = self.plugin.get_hyperspectral_data(layer_id)
        if not data_info:
            QMessageBox.warning(self, "Error", "Layer data not found.")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(20)

        try:
            dataset = data_info.get("dataset")
            if dataset is None:
                raise ValueError("Dataset not available")

            layer = QgsProject.instance().mapLayer(layer_id)
            if not layer:
                raise ValueError("Layer not found")

            self.progress_bar.setValue(40)

            # Get wavelengths
            if self.rgb_radio.isChecked():
                wavelengths = [
                    self.red_spin.value(),
                    self.green_spin.value(),
                    self.blue_spin.value(),
                ]
            else:
                wavelengths = [self.single_spin.value()]

            # Export new combination
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"{layer.name()}_bands.tif")

            self.progress_bar.setValue(60)

            dataset.export_to_geotiff(temp_path, wavelengths=wavelengths)

            self.progress_bar.setValue(80)

            # Create new layer
            new_layer = QgsRasterLayer(temp_path, layer.name())

            if not new_layer.isValid():
                raise ValueError("Failed to create layer")

            # Apply styling
            vmin = self.vmin_spin.value()
            vmax = self.vmax_spin.value()

            provider = new_layer.dataProvider()

            if self.rgb_radio.isChecked():
                renderer = QgsMultiBandColorRenderer(provider, 1, 2, 3)

                for band_idx in [1, 2, 3]:
                    ce = QgsContrastEnhancement(provider.dataType(band_idx))
                    ce.setContrastEnhancementAlgorithm(
                        QgsContrastEnhancement.StretchToMinimumMaximum
                    )
                    if not self.auto_stretch.isChecked():
                        ce.setMinimumValue(vmin)
                        ce.setMaximumValue(vmax)

                    if band_idx == 1:
                        renderer.setRedContrastEnhancement(ce)
                    elif band_idx == 2:
                        renderer.setGreenContrastEnhancement(ce)
                    else:
                        renderer.setBlueContrastEnhancement(ce)
            else:
                renderer = QgsSingleBandGrayRenderer(provider, 1)
                ce = QgsContrastEnhancement(provider.dataType(1))
                ce.setContrastEnhancementAlgorithm(
                    QgsContrastEnhancement.StretchToMinimumMaximum
                )
                if not self.auto_stretch.isChecked():
                    ce.setMinimumValue(vmin)
                    ce.setMaximumValue(vmax)
                renderer.setContrastEnhancement(ce)

            new_layer.setRenderer(renderer)

            # Replace layer in project
            QgsProject.instance().removeMapLayer(layer_id)
            QgsProject.instance().addMapLayer(new_layer)

            # Update plugin registration
            data_info["rgb_wavelengths"] = wavelengths
            self.plugin.register_hyperspectral_layer(new_layer.id(), data_info)

            self.progress_bar.setValue(100)

            # Refresh layer list
            self.refresh_layers()

            self.iface.mapCanvas().refresh()

            QMessageBox.information(self, "Success", "Band combination applied!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying bands: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)
