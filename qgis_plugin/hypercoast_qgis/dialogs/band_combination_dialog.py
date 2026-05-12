# -*- coding: utf-8 -*-
"""
HyperCoast QGIS Plugin - Band Combination Dialog

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

import os
import tempfile
import numpy as np
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QDockWidget,
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
    QRadioButton,
    QSizePolicy,
    QWidget,
)
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsProject,
    QgsRasterLayer,
    QgsContrastEnhancement,
    QgsMultiBandColorRenderer,
    QgsSingleBandGrayRenderer,
    QgsColorRampShader,
    QgsRasterShader,
    QgsSingleBandPseudoColorRenderer,
)


class BandExportWorker(QThread):
    """Worker thread for exporting selected bands without blocking QGIS."""

    progress = pyqtSignal(int)
    finished = pyqtSignal(str, str)

    def __init__(self, dataset, output_path, wavelengths, parent=None):
        """Initialize the worker.

        Args:
            dataset: Loaded HyperspectralDataset.
            output_path: GeoTIFF path to create.
            wavelengths: Selected wavelengths.
            parent: Optional parent QObject.
        """
        super().__init__(parent)
        self.dataset = dataset
        self.output_path = output_path
        self.wavelengths = wavelengths

    def run(self):
        """Export selected bands in a background thread."""
        try:
            self.progress.emit(60)
            self.dataset.export_to_geotiff(
                self.output_path, wavelengths=self.wavelengths
            )
            self.progress.emit(80)
            self.finished.emit(self.output_path, "")
        except Exception as exc:
            self.finished.emit("", str(exc))


class BandCombinationDialog(QDockWidget):
    """Dockable panel for changing band combinations of hyperspectral layers."""

    def __init__(self, iface, plugin, parent=None):
        """Initialize the dialog.

        :param iface: QGIS interface
        :param plugin: Reference to the main plugin
        :param parent: Parent widget
        """
        super().__init__(parent or iface.mainWindow())
        self.iface = iface
        self.plugin = plugin
        self._export_worker = None
        self._pending_apply_context = None

        self.setWindowTitle("Band Combination")
        self.setObjectName("HyperCoastBandCombinationDock")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        content = QWidget(self)
        self.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Layer selection
        layer_group = QGroupBox("Layer Selection")
        layer_layout = QFormLayout()
        self._compact_form_layout(layer_layout)

        self.layer_combo = QComboBox()
        self.layer_combo.currentIndexChanged.connect(self.on_layer_changed)
        layer_layout.addRow("Layer:", self.layer_combo)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        layer_layout.addRow("Info:", self.info_label)

        layer_group.setLayout(layer_layout)
        self._set_compact_group(layer_group)
        layout.addWidget(layer_group)

        # Band selection mode
        mode_group = QGroupBox("Display Mode")
        mode_layout = QHBoxLayout()
        mode_layout.setContentsMargins(8, 6, 8, 6)
        mode_layout.setSpacing(12)

        self.rgb_radio = QRadioButton("RGB Composite")
        self.rgb_radio.setChecked(True)
        self.rgb_radio.toggled.connect(self.on_mode_changed)
        mode_layout.addWidget(self.rgb_radio)

        self.single_radio = QRadioButton("Single Band")
        self.single_radio.toggled.connect(self.on_mode_changed)
        mode_layout.addWidget(self.single_radio)
        mode_layout.addStretch()

        mode_group.setLayout(mode_layout)
        self._set_compact_group(mode_group)
        layout.addWidget(mode_group)

        # RGB band selection
        self.rgb_group = QGroupBox("RGB Wavelength Selection")
        rgb_layout = QFormLayout()
        self._compact_form_layout(rgb_layout)

        # Red band
        red_layout = QHBoxLayout()
        red_layout.setSpacing(8)
        self.red_spin = QDoubleSpinBox()
        self.red_spin.setRange(0, 5000)
        self.red_spin.setValue(650)
        self.red_spin.setSuffix(" nm")
        self.red_spin.valueChanged.connect(self.update_red_slider)
        red_layout.addWidget(self.red_spin)

        self.red_slider = QSlider(Qt.Orientation.Horizontal)
        self.red_slider.setRange(0, 5000)
        self.red_slider.setValue(650)
        self.red_slider.valueChanged.connect(lambda v: self.red_spin.setValue(v))
        red_layout.addWidget(self.red_slider)

        rgb_layout.addRow("Red:", red_layout)

        # Green band
        green_layout = QHBoxLayout()
        green_layout.setSpacing(8)
        self.green_spin = QDoubleSpinBox()
        self.green_spin.setRange(0, 5000)
        self.green_spin.setValue(550)
        self.green_spin.setSuffix(" nm")
        self.green_spin.valueChanged.connect(self.update_green_slider)
        green_layout.addWidget(self.green_spin)

        self.green_slider = QSlider(Qt.Orientation.Horizontal)
        self.green_slider.setRange(0, 5000)
        self.green_slider.setValue(550)
        self.green_slider.valueChanged.connect(lambda v: self.green_spin.setValue(v))
        green_layout.addWidget(self.green_slider)

        rgb_layout.addRow("Green:", green_layout)

        # Blue band
        blue_layout = QHBoxLayout()
        blue_layout.setSpacing(8)
        self.blue_spin = QDoubleSpinBox()
        self.blue_spin.setRange(0, 5000)
        self.blue_spin.setValue(450)
        self.blue_spin.setSuffix(" nm")
        self.blue_spin.valueChanged.connect(self.update_blue_slider)
        blue_layout.addWidget(self.blue_spin)

        self.blue_slider = QSlider(Qt.Orientation.Horizontal)
        self.blue_slider.setRange(0, 5000)
        self.blue_slider.setValue(450)
        self.blue_slider.valueChanged.connect(lambda v: self.blue_spin.setValue(v))
        blue_layout.addWidget(self.blue_slider)

        rgb_layout.addRow("Blue:", blue_layout)

        self.rgb_group.setLayout(rgb_layout)
        self._set_compact_group(self.rgb_group)
        layout.addWidget(self.rgb_group)

        # Single band selection
        self.single_group = QGroupBox("Single Band Selection")
        single_layout = QFormLayout()
        self._compact_form_layout(single_layout)

        single_wl_layout = QHBoxLayout()
        single_wl_layout.setSpacing(8)
        self.single_spin = QDoubleSpinBox()
        self.single_spin.setRange(0, 5000)
        self.single_spin.setValue(550)
        self.single_spin.setSuffix(" nm")
        single_wl_layout.addWidget(self.single_spin)

        self.single_slider = QSlider(Qt.Orientation.Horizontal)
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
        self._set_compact_group(self.single_group)
        layout.addWidget(self.single_group)

        # Value range
        range_group = QGroupBox("Value Range")
        range_layout = QFormLayout()
        self._compact_form_layout(range_layout)

        range_row = QHBoxLayout()
        range_row.setSpacing(8)
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
        self._set_compact_group(range_group)
        layout.addWidget(range_group)

        # Presets
        presets_group = QGroupBox("Quick Presets")
        presets_layout = QHBoxLayout()
        presets_layout.setContentsMargins(8, 6, 8, 6)
        presets_layout.setSpacing(8)

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
        self._set_compact_group(presets_group)
        layout.addWidget(presets_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Buttons
        button_layout = QHBoxLayout()

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_band_combination)
        button_layout.addWidget(self.apply_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)
        layout.addStretch(1)

    def _set_compact_group(self, group):
        """Prevent group boxes from consuming extra vertical dock space.

        Args:
            group: QGroupBox to constrain.
        """
        policy = getattr(QSizePolicy, "Policy", QSizePolicy)
        group.setSizePolicy(policy.Preferred, policy.Maximum)

    def _compact_form_layout(self, layout):
        """Apply compact spacing to a form layout.

        Args:
            layout: QFormLayout to compact.
        """
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

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
        if self._export_worker is not None and self._export_worker.isRunning():
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(20)
        self.apply_btn.setEnabled(False)

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

            temp_path = self._create_temp_raster_path(layer.name(), "bands")
            self._pending_apply_context = {
                "layer_id": layer_id,
                "data_info": data_info,
                "wavelengths": wavelengths,
                "is_rgb": self.rgb_radio.isChecked(),
                "vmin": self.vmin_spin.value(),
                "vmax": self.vmax_spin.value(),
                "auto_stretch": self.auto_stretch.isChecked(),
                "colormap": self.colormap_combo.currentText(),
            }
            self._export_worker = BandExportWorker(
                dataset, temp_path, wavelengths, self
            )
            self._export_worker.progress.connect(self.progress_bar.setValue)
            self._export_worker.finished.connect(self._on_export_finished)
            self._export_worker.start()

        except Exception as e:
            self._finish_apply_ui()
            QMessageBox.critical(self, "Error", f"Error applying bands: {str(e)}")

    def _on_export_finished(self, temp_path, error_detail):
        """Handle selected-band export completion.

        Args:
            temp_path: Exported GeoTIFF path, or empty string on failure.
            error_detail: Error message when export failed.
        """
        try:
            if not temp_path:
                raise ValueError(error_detail or "Failed to export selected bands")

            context = self._pending_apply_context or {}
            layer_id = context["layer_id"]
            data_info = context["data_info"]
            wavelengths = context["wavelengths"]
            old_layer = QgsProject.instance().mapLayer(layer_id)
            if not old_layer:
                raise ValueError("Layer not found")

            new_layer = QgsRasterLayer(temp_path, old_layer.name())

            if not new_layer.isValid():
                raise ValueError("Failed to create layer")

            # If the GeoTIFF was written without CRS (e.g. due to a PROJ DB
            # conflict on Windows), restore it from the dataset metadata so
            # the layer is placed at the correct location on the map.
            dataset = data_info.get("dataset")
            dataset_crs = data_info.get("crs") or getattr(dataset, "crs", None)
            if not new_layer.crs().isValid() and dataset_crs:
                known_crs = QgsCoordinateReferenceSystem(str(dataset_crs))
                if known_crs.isValid():
                    new_layer.setCrs(known_crs)

            # Apply styling
            vmin = context["vmin"]
            vmax = context["vmax"]

            provider = new_layer.dataProvider()
            self._apply_nodata(provider, new_layer.bandCount())

            if context["is_rgb"]:
                renderer = QgsMultiBandColorRenderer(provider, 1, 2, 3)

                for band_idx in [1, 2, 3]:
                    ce = QgsContrastEnhancement(provider.dataType(band_idx))
                    ce.setContrastEnhancementAlgorithm(
                        QgsContrastEnhancement.StretchToMinimumMaximum
                    )
                    if not context["auto_stretch"]:
                        ce.setMinimumValue(vmin)
                        ce.setMaximumValue(vmax)

                    if band_idx == 1:
                        renderer.setRedContrastEnhancement(ce)
                    elif band_idx == 2:
                        renderer.setGreenContrastEnhancement(ce)
                    else:
                        renderer.setBlueContrastEnhancement(ce)
            else:
                renderer = self._create_single_band_renderer(
                    provider,
                    context["colormap"],
                    vmin,
                    vmax,
                    context["auto_stretch"],
                )

            new_layer.setRenderer(renderer)
            self._set_layer_custom_properties(new_layer, data_info, wavelengths)

            # Replacing the raster is necessary because the rendered GeoTIFF path
            # changes, but persist the HyperCoast metadata on the replacement.
            QgsProject.instance().removeMapLayer(layer_id)
            QgsProject.instance().addMapLayer(new_layer)

            # Update plugin registration
            data_info["rgb_wavelengths"] = wavelengths
            self.plugin.register_hyperspectral_layer(new_layer.id(), data_info)

            self.progress_bar.setValue(100)

            # Refresh layer list
            self.refresh_layers()

            self.iface.mapCanvas().refresh()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying bands: {str(e)}")

        finally:
            self._finish_apply_ui()

    def _finish_apply_ui(self):
        """Restore controls after selected-band export."""
        self.progress_bar.setVisible(False)
        self.apply_btn.setEnabled(True)
        self._pending_apply_context = None

    def _create_temp_raster_path(self, layer_name, suffix):
        """Create a unique temporary GeoTIFF path.

        Args:
            layer_name: QGIS layer name used as a readable prefix.
            suffix: Short output kind.

        Returns:
            Path to a unique temporary GeoTIFF.
        """
        safe_name = "".join(
            c if c.isalnum() or c in ("-", "_") else "_" for c in layer_name
        )
        safe_name = safe_name.strip("_") or "hypercoast"
        fd, path = tempfile.mkstemp(prefix=f"{safe_name}_{suffix}_", suffix=".tif")
        os.close(fd)
        try:
            os.remove(path)
        except OSError:
            pass
        return path

    def _set_layer_custom_properties(self, layer, data_info, wavelengths):
        """Persist HyperCoast metadata on a replacement layer.

        Args:
            layer: New QGIS raster layer.
            data_info: Registered HyperCoast layer metadata.
            wavelengths: Selected wavelengths.
        """
        layer.setCustomProperty("hypercoast/source_path", data_info.get("filepath", ""))
        layer.setCustomProperty("hypercoast/data_type", data_info.get("data_type", ""))
        layer.setCustomProperty(
            "hypercoast/rgb_wavelengths", ",".join(str(v) for v in wavelengths)
        )
        if data_info.get("crs"):
            layer.setCustomProperty("hypercoast/crs", str(data_info["crs"]))

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
                pass

    def _create_single_band_renderer(
        self, provider, colormap_name, vmin, vmax, auto_stretch
    ):
        """Create a single-band renderer honoring the selected colormap.

        Args:
            provider: Raster data provider.
            colormap_name: UI colormap name.
            vmin: Minimum display value.
            vmax: Maximum display value.
            auto_stretch: Whether to let QGIS stretch from statistics.

        Returns:
            QGIS raster renderer.
        """
        if colormap_name == "Grayscale":
            renderer = QgsSingleBandGrayRenderer(provider, 1)
            ce = QgsContrastEnhancement(provider.dataType(1))
            ce.setContrastEnhancementAlgorithm(
                QgsContrastEnhancement.StretchToMinimumMaximum
            )
            if not auto_stretch:
                ce.setMinimumValue(vmin)
                ce.setMaximumValue(vmax)
            renderer.setContrastEnhancement(ce)
            return renderer

        ramp = QgsColorRampShader()
        try:
            ramp.setColorRampType(QgsColorRampShader.Type.Interpolated)
        except AttributeError:
            ramp.setColorRampType(QgsColorRampShader.Interpolated)
        ramp.setColorRampItemList(self._color_ramp_items(colormap_name, vmin, vmax))
        shader = QgsRasterShader()
        shader.setRasterShaderFunction(ramp)
        return QgsSingleBandPseudoColorRenderer(provider, 1, shader)

    def _color_ramp_items(self, colormap_name, vmin, vmax):
        """Return QGIS color ramp items for a named colormap.

        Args:
            colormap_name: UI colormap name.
            vmin: Minimum display value.
            vmax: Maximum display value.

        Returns:
            List of QgsColorRampShader.ColorRampItem objects.
        """
        palettes = {
            "Jet": ["#00007f", "#0000ff", "#00ffff", "#ffff00", "#ff0000", "#7f0000"],
            "Viridis": ["#440154", "#31688e", "#35b779", "#fde725"],
            "Plasma": ["#0d0887", "#9c179e", "#ed7953", "#f0f921"],
            "Inferno": ["#000004", "#781c6d", "#ed6925", "#fcffa4"],
            "Magma": ["#000004", "#721f81", "#f1605d", "#fcfdbf"],
            "Hot": ["#000000", "#ff0000", "#ffff00", "#ffffff"],
            "Cool": ["#00ffff", "#ff00ff"],
            "Rainbow": ["#6e40aa", "#1ac7c2", "#aff05b", "#ff8c38", "#d41159"],
            "Spectral": ["#9e0142", "#fdae61", "#ffffbf", "#abdda4", "#5e4fa2"],
        }
        colors = palettes.get(colormap_name, palettes["Viridis"])
        if vmax <= vmin:
            vmax = vmin + 1.0
        step = (vmax - vmin) / max(len(colors) - 1, 1)
        return [
            QgsColorRampShader.ColorRampItem(vmin + i * step, QColor(color), color)
            for i, color in enumerate(colors)
        ]

    def closeEvent(self, event):
        """Stop the background export worker before the dock closes.

        Args:
            event: Qt close event.
        """
        worker = self._export_worker
        if worker is not None:
            try:
                if worker.isRunning():
                    worker.requestInterruption()
                    worker.quit()
                    worker.wait(5000)
            except RuntimeError:
                pass  # nosec B110
        super().closeEvent(event)
