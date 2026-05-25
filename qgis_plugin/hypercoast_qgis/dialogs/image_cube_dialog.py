# -*- coding: utf-8 -*-
"""
HyperCoast QGIS Plugin - 3D Image Cube Dialog

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

import json
import os
import subprocess
import sys
import uuid

from qgis.PyQt.QtCore import QObject, Qt, QThread, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from qgis.core import QgsMessageLog, QgsProject, Qgis

from ..cache_manager import generated_raster_cache_dir
from .tanager_search_dialog import TanagerBboxMapTool

DEFAULT_CLIM = (0.0, 0.5)
MAX_CUBE_POINTS = 5_000_000
SPECTRAL_DIMS = {"wavelength", "wavelengths", "band", "bands"}
WIDGET_OPTIONS = (
    ("None", None),
    ("Slice", "slice"),
    ("Orthogonal Slices", "orthogonal"),
    ("Threshold", "threshold"),
    ("Clip Box", "box"),
    ("Clip Plane", "plane"),
)


def _write_cube_dataset_file(dataset, path):
    """Write a prepared cube dataset to NetCDF.

    Args:
        dataset: Xarray dataset.
        path: Output NetCDF path.
    """
    loaded = dataset.load()
    try:
        loaded.to_netcdf(path, engine="scipy")
    except Exception:
        loaded.to_netcdf(path)


def _json_safe_options(options):
    """Return image cube options that can be serialized to JSON.

    Args:
        options: Raw keyword arguments for ``hypercoast.image_cube``.

    Returns:
        JSON-serializable dictionary.
    """
    safe_options = {}
    for key, value in options.items():
        if isinstance(value, tuple):
            safe_options[key] = list(value)
        else:
            safe_options[key] = value
    return safe_options


def _viewer_script_source():
    """Return the standalone image cube viewer script.

    Returns:
        Python source code for the viewer process.
    """
    return """\
import json
import sys
import traceback

import hypercoast
import xarray as xr


def main():
    dataset_path = sys.argv[1]
    options_path = sys.argv[2]
    with open(options_path, "r", encoding="utf-8") as options_file:
        options = json.load(options_file)
    dataset = xr.open_dataset(dataset_path).load()
    plotter = hypercoast.image_cube(dataset, **options)
    show = getattr(plotter, "show", None)
    if callable(show):
        show()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
"""


def _run_image_cube_launch(dataset, options, context):
    """Write cube viewer inputs and start the external viewer process.

    Args:
        dataset: Prepared xarray dataset.
        options: Keyword arguments for ``hypercoast.image_cube``.
        context: Launch context containing paths, environment, and command
            options.

    Returns:
        Tuple of subprocess handle and stderr log path.
    """
    _write_cube_dataset_file(dataset, context["dataset_path"])
    with open(context["options_path"], "w", encoding="utf-8") as options_file:
        json.dump(_json_safe_options(options), options_file)
    with open(context["script_path"], "w", encoding="utf-8") as script_file:
        script_file.write(_viewer_script_source())

    with open(context["stdout_path"], "w", encoding="utf-8") as stdout_file:
        with open(context["stderr_path"], "w", encoding="utf-8") as stderr_file:
            process = subprocess.Popen(
                context["cmd"],
                cwd=context["cache_dir"],
                env=context["env"],
                stdout=stdout_file,
                stderr=stderr_file,
                **context["popen_kwargs"],
            )
    return process, context["stderr_path"]


class ImageCubeLaunchWorker(QObject):
    """Worker for writing cube files and launching the standalone viewer."""

    finished = pyqtSignal(object, str)
    failed = pyqtSignal(str)

    def __init__(self, dataset, options, context):
        """Initialize the worker.

        Args:
            dataset: Prepared xarray dataset.
            options: Keyword arguments for ``hypercoast.image_cube``.
            context: External viewer launch context.
        """
        super().__init__()
        self.dataset = dataset
        self.options = options
        self.context = context

    def run(self):
        """Run the external viewer preparation and launch."""
        try:
            process, stderr_path = _run_image_cube_launch(
                self.dataset, self.options, self.context
            )
            self.finished.emit(process, stderr_path)
        except Exception as exc:
            self.failed.emit(str(exc))


class ImageCubeDialog(QDockWidget):
    """Dockable panel for creating PyVista 3D hyperspectral image cubes."""

    def __init__(self, iface, plugin, parent=None):
        """Initialize the dialog.

        Args:
            iface: QGIS interface.
            plugin: Reference to the main plugin.
            parent: Optional parent widget.
        """
        super().__init__(parent or iface.mainWindow())
        self.iface = iface
        self.plugin = plugin
        self._viewer_processes = []
        self._bbox_map_tool = None
        self._previous_map_tool = None
        self._subset_bbox = None
        self._cube_thread = None
        self._cube_worker = None

        self.setWindowTitle("3D Image Cube")
        self.setObjectName("HyperCoastImageCubeDock")
        self.setMinimumWidth(520)
        self.setMinimumHeight(420)

        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        content = QWidget(self)
        self.setWidget(content)
        layout = QVBoxLayout(content)

        layer_group = QGroupBox("Layer")
        layer_layout = QHBoxLayout(layer_group)
        self.layer_combo = QComboBox()
        self.layer_combo.currentIndexChanged.connect(self._on_layer_changed)
        layer_layout.addWidget(self.layer_combo, 1)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_layers)
        layer_layout.addWidget(refresh_btn)
        layout.addWidget(layer_group)

        cube_group = QGroupBox("Cube Options")
        cube_form = QFormLayout(cube_group)

        self.variable_combo = QComboBox()
        cube_form.addRow("Variable:", self.variable_combo)

        self.widget_combo = QComboBox()
        for label, value in WIDGET_OPTIONS:
            self.widget_combo.addItem(label, value)
        cube_form.addRow("Widget:", self.widget_combo)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["jet", "viridis", "plasma", "inferno", "gray"])
        cube_form.addRow("Colormap:", self.cmap_combo)

        clim_row = QHBoxLayout()
        self.vmin_spin = QDoubleSpinBox()
        self.vmin_spin.setRange(-1_000_000, 1_000_000)
        self.vmin_spin.setDecimals(4)
        self.vmin_spin.setValue(DEFAULT_CLIM[0])
        clim_row.addWidget(QLabel("Min:"))
        clim_row.addWidget(self.vmin_spin)
        self.vmax_spin = QDoubleSpinBox()
        self.vmax_spin.setRange(-1_000_000, 1_000_000)
        self.vmax_spin.setDecimals(4)
        self.vmax_spin.setValue(DEFAULT_CLIM[1])
        clim_row.addWidget(QLabel("Max:"))
        clim_row.addWidget(self.vmax_spin)
        cube_form.addRow("Value Range:", clim_row)

        rgb_row = QHBoxLayout()
        self.red_spin = self._wavelength_spin(650)
        self.green_spin = self._wavelength_spin(550)
        self.blue_spin = self._wavelength_spin(450)
        rgb_row.addWidget(QLabel("R:"))
        rgb_row.addWidget(self.red_spin)
        rgb_row.addWidget(QLabel("G:"))
        rgb_row.addWidget(self.green_spin)
        rgb_row.addWidget(QLabel("B:"))
        rgb_row.addWidget(self.blue_spin)
        cube_form.addRow("RGB Wavelengths:", rgb_row)

        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.01, 10.0)
        self.gamma_spin.setDecimals(2)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(1.0)
        cube_form.addRow("RGB Gamma:", self.gamma_spin)

        layout.addWidget(cube_group)

        subset_group = QGroupBox("Subset")
        subset_layout = QHBoxLayout(subset_group)
        self.subset_bbox_edit = QLineEdit()
        self.subset_bbox_edit.setReadOnly(True)
        self.subset_bbox_edit.setPlaceholderText("Draw a small rectangle on the map")
        subset_layout.addWidget(self.subset_bbox_edit, 1)
        draw_subset_btn = QPushButton("Draw Subset")
        draw_subset_btn.clicked.connect(self.start_subset_drawing)
        subset_layout.addWidget(draw_subset_btn)
        clear_subset_btn = QPushButton("Clear")
        clear_subset_btn.clicked.connect(self.clear_subset)
        subset_layout.addWidget(clear_subset_btn)
        layout.addWidget(subset_group)

        performance_group = QGroupBox("Performance")
        performance_form = QFormLayout(performance_group)

        self.spatial_stride_spin = QSpinBox()
        self.spatial_stride_spin.setRange(1, 100)
        self.spatial_stride_spin.setValue(4)
        performance_form.addRow("Spatial Stride:", self.spatial_stride_spin)

        self.spectral_stride_spin = QSpinBox()
        self.spectral_stride_spin.setRange(1, 100)
        self.spectral_stride_spin.setValue(1)
        performance_form.addRow("Spectral Stride:", self.spectral_stride_spin)

        self.crop_spin = QSpinBox()
        self.crop_spin.setRange(0, 100000)
        self.crop_spin.setValue(0)
        performance_form.addRow("Crop Pixels:", self.crop_spin)

        nodata_row = QHBoxLayout()
        self.nodata_check = QCheckBox("Mask")
        self.nodata_check.stateChanged.connect(self._sync_nodata_state)
        nodata_row.addWidget(self.nodata_check)
        self.nodata_spin = QDoubleSpinBox()
        self.nodata_spin.setRange(-1_000_000, 1_000_000)
        self.nodata_spin.setDecimals(4)
        self.nodata_spin.setValue(0.0)
        self.nodata_spin.setEnabled(False)
        nodata_row.addWidget(self.nodata_spin)
        performance_form.addRow("NoData:", nodata_row)

        layout.addWidget(performance_group)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        button_layout = QHBoxLayout()
        self.create_btn = QPushButton("Create 3D Cube")
        self.create_btn.clicked.connect(self.create_cube)
        button_layout.addWidget(self.create_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)

        layout.addStretch()
        self.refresh_layers()

    def _wavelength_spin(self, value):
        """Create a wavelength spin box.

        Args:
            value: Initial wavelength value.

        Returns:
            Configured spin box.
        """
        spin = QDoubleSpinBox()
        spin.setRange(0, 5000)
        spin.setDecimals(1)
        spin.setSuffix(" nm")
        spin.setValue(value)
        return spin

    def start_subset_drawing(self):
        """Activate a map tool for drawing a small image-cube subset."""
        try:
            canvas = self.iface.mapCanvas()
        except Exception as exc:
            QMessageBox.warning(self, "3D Image Cube", f"Could not access map: {exc}")
            return

        if self._bbox_map_tool is None:
            self._bbox_map_tool = TanagerBboxMapTool(canvas, self)
            self._bbox_map_tool.bbox_drawn.connect(self._on_subset_bbox_drawn)
            self._bbox_map_tool.canceled.connect(self._on_subset_drawing_canceled)

        map_tool = getattr(canvas, "mapTool", None)
        if callable(map_tool):
            try:
                current_tool = map_tool()
                if current_tool is not self._bbox_map_tool:
                    self._previous_map_tool = current_tool
            except Exception:
                self._previous_map_tool = None

        try:
            canvas.setMapTool(self._bbox_map_tool)
            self.status_label.setText("Draw a small rectangle for the 3D cube subset")
        except Exception as exc:
            QMessageBox.warning(
                self, "3D Image Cube", f"Could not activate map tool: {exc}"
            )

    def clear_subset(self):
        """Clear the drawn image-cube subset."""
        self._restore_previous_map_tool()
        self._subset_bbox = None
        self.subset_bbox_edit.clear()
        self.status_label.setText("")

    def _on_subset_bbox_drawn(self, bbox):
        """Store the drawn image-cube subset bbox.

        Args:
            bbox: Bounding box list in EPSG:4326.
        """
        self._subset_bbox = [float(value) for value in bbox]
        self.subset_bbox_edit.setText(
            ", ".join(f"{value:.8f}" for value in self._subset_bbox)
        )
        self.status_label.setText("Subset ready for 3D image cube")
        self._restore_previous_map_tool()

    def _on_subset_drawing_canceled(self):
        """Handle subset drawing cancellation."""
        self.status_label.setText("Subset drawing canceled")
        self._restore_previous_map_tool()

    def _restore_previous_map_tool(self):
        """Restore the previous QGIS map tool after subset drawing."""
        if self._bbox_map_tool is None:
            return
        try:
            canvas = self.iface.mapCanvas()
            map_tool = getattr(canvas, "mapTool", None)
            current_tool = map_tool() if callable(map_tool) else None
            if current_tool is not self._bbox_map_tool:
                return
            if self._previous_map_tool is not None:
                canvas.setMapTool(self._previous_map_tool)
            else:
                unset_map_tool = getattr(canvas, "unsetMapTool", None)
                if callable(unset_map_tool):
                    unset_map_tool(self._bbox_map_tool)
        except Exception:
            pass  # nosec B110
        finally:
            self._previous_map_tool = None

    def refresh_layers(self):
        """Refresh the available HyperCoast layer list."""
        current_layer_id = self.selected_layer_id()
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()

        layers = self.plugin.get_all_hyperspectral_layers()
        for layer_id in layers:
            layer = self._project_layer(layer_id)
            name = layer.name() if layer is not None else layer_id
            if not isinstance(name, str):
                name = layer_id
            self.layer_combo.addItem(name, layer_id)

        if self.layer_combo.count() == 0:
            self.layer_combo.addItem("No HyperCoast layers", None)
        elif current_layer_id:
            index = self.layer_combo.findData(current_layer_id)
            if index >= 0:
                self.layer_combo.setCurrentIndex(index)

        self.layer_combo.blockSignals(False)
        self._on_layer_changed()

    def selected_layer_id(self):
        """Return the selected layer ID.

        Returns:
            QGIS layer ID, or None.
        """
        return self.layer_combo.currentData() if hasattr(self, "layer_combo") else None

    def _on_layer_changed(self, *args):
        """Update controls for the selected layer."""
        data_info = self._selected_data_info()
        if not data_info:
            self.variable_combo.clear()
            return

        rgb_wavelengths = data_info.get("rgb_wavelengths") or []
        if len(rgb_wavelengths) >= 3:
            self.red_spin.setValue(float(rgb_wavelengths[0]))
            self.green_spin.setValue(float(rgb_wavelengths[1]))
            self.blue_spin.setValue(float(rgb_wavelengths[2]))

        dataset = data_info.get("dataset")
        self._populate_variables(getattr(dataset, "dataset", None), data_info)

    def _selected_data_info(self):
        """Return metadata for the selected HyperCoast layer.

        Returns:
            Layer metadata dictionary, or None.
        """
        layer_id = self.selected_layer_id()
        if not layer_id:
            return None
        return self.plugin.get_hyperspectral_data(layer_id) or {}

    def _populate_variables(self, dataset, data_info=None):
        """Populate the data-variable selector.

        Args:
            dataset: Xarray dataset, or None.
            data_info: Optional layer metadata.
        """
        current = self.variable_combo.currentData()
        selected = (data_info or {}).get("selected_variable")
        self.variable_combo.blockSignals(True)
        self.variable_combo.clear()

        names = (
            list(getattr(dataset, "data_vars", {}).keys())
            if dataset is not None
            else []
        )
        if names:
            for name in names:
                self.variable_combo.addItem(name, name)
            desired = current or selected
            if desired:
                index = self.variable_combo.findData(desired)
                if index >= 0:
                    self.variable_combo.setCurrentIndex(index)
        else:
            self.variable_combo.addItem("Default variable", None)

        self.variable_combo.blockSignals(False)

    def create_cube(self):
        """Create and display a 3D image cube for the selected layer."""
        layer_id = self.selected_layer_id()
        if not layer_id:
            QMessageBox.warning(self, "3D Image Cube", "No HyperCoast layer selected.")
            return

        dataset_obj = self.plugin.ensure_hyperspectral_dataset(layer_id)
        if dataset_obj is None or getattr(dataset_obj, "dataset", None) is None:
            QMessageBox.warning(
                self, "3D Image Cube", "Could not load the selected dataset."
            )
            return

        dataset = dataset_obj.dataset
        self._populate_variables(dataset, self.plugin.get_hyperspectral_data(layer_id))
        variable = self._selected_variable_name(dataset_obj)
        if not variable or variable not in dataset.data_vars:
            QMessageBox.warning(
                self, "3D Image Cube", "No raster-like data variable found."
            )
            return
        if dataset[variable].ndim != 3:
            QMessageBox.warning(
                self,
                "3D Image Cube",
                f"Expected a 3D data variable, got shape {dataset[variable].shape}.",
            )
            return

        try:
            cube_dataset = self._subset_dataset(dataset, variable, self._subset_bbox)
            cube_dataset = cube_dataset[[variable]]
            cube_dataset = self._downsample_dataset(
                cube_dataset,
                self.spatial_stride_spin.value(),
                self.spectral_stride_spin.value(),
            )
            cube_dataset = self._prepare_image_cube_dataset(cube_dataset, variable)
            if not self._validate_cube_size(cube_dataset, variable):
                return
            kwargs = self._image_cube_kwargs(cube_dataset, variable)
            self._start_image_cube_worker(cube_dataset, kwargs)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "3D Image Cube",
                f"Could not create 3D image cube.\n\nDetails:\n{exc}",
            )
            QgsMessageLog.logMessage(
                f"Could not create 3D image cube: {exc}",
                "HyperCoast",
                Qgis.MessageLevel.Warning,
            )
            self.status_label.setText("3D image cube failed")
            self._set_cube_busy(False)

    def _start_image_cube_worker(self, dataset, kwargs):
        """Start a worker thread that prepares and launches the cube viewer.

        Args:
            dataset: Prepared xarray dataset to visualize.
            kwargs: Keyword arguments for ``hypercoast.image_cube``.
        """
        if self._cube_thread is not None and self._cube_thread.isRunning():
            QMessageBox.warning(
                self,
                "3D Image Cube",
                "A 3D image cube is already being prepared.",
            )
            return

        context = self._image_cube_launch_context()
        thread = QThread(self)
        worker = ImageCubeLaunchWorker(dataset, kwargs, context)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_image_cube_worker_finished)
        worker.failed.connect(self._on_image_cube_worker_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_image_cube_worker)

        self._cube_thread = thread
        self._cube_worker = worker
        self._set_cube_busy(True, "Creating 3D image cube...")
        thread.start()

    def _on_image_cube_worker_finished(self, process, stderr_path):
        """Handle successful external viewer startup.

        Args:
            process: Viewer subprocess handle.
            stderr_path: Path to the viewer stderr log.
        """
        self._viewer_processes.append(process)
        QgsMessageLog.logMessage(
            f"3D image cube viewer launched. Error log: {stderr_path}",
            "HyperCoast",
            Qgis.MessageLevel.Info,
        )
        self._set_cube_busy(False, "3D image cube viewer launched")

    def _on_image_cube_worker_failed(self, message):
        """Handle image cube worker failure.

        Args:
            message: User-visible failure details.
        """
        QMessageBox.warning(
            self,
            "3D Image Cube",
            f"Could not create 3D image cube.\n\nDetails:\n{message}",
        )
        QgsMessageLog.logMessage(
            f"Could not create 3D image cube: {message}",
            "HyperCoast",
            Qgis.MessageLevel.Warning,
        )
        self._set_cube_busy(False, "3D image cube failed")

    def _clear_image_cube_worker(self):
        """Clear references to the completed image cube worker."""
        self._cube_thread = None
        self._cube_worker = None

    def _set_cube_busy(self, busy, message=None):
        """Show or hide the busy progress indicator.

        Args:
            busy: Whether image cube creation is running.
            message: Optional status text.
        """
        self.progress_bar.setVisible(busy)
        self.create_btn.setEnabled(not busy)
        if message is not None:
            self.status_label.setText(message)

    def _launch_image_cube_viewer(self, dataset, kwargs):
        """Launch ``hypercoast.image_cube`` in a separate Python process.

        Rendering PyVista/VTK inside QGIS can terminate the whole QGIS process
        when native libraries conflict. The dialog therefore prepares a small
        temporary NetCDF cube and starts a standalone viewer process.

        Args:
            dataset: Prepared xarray dataset to visualize.
            kwargs: Keyword arguments for ``hypercoast.image_cube``.

        Raises:
            OSError: If temporary files cannot be written or the viewer cannot
                be launched.
        """
        context = self._image_cube_launch_context()
        process, stderr_path = _run_image_cube_launch(dataset, kwargs, context)
        self._viewer_processes.append(process)
        QgsMessageLog.logMessage(
            f"3D image cube viewer launched. Error log: {stderr_path}",
            "HyperCoast",
            Qgis.MessageLevel.Info,
        )

    def _image_cube_launch_context(self):
        """Return paths and options for launching the external cube viewer.

        Returns:
            Launch context dictionary.
        """
        cache_dir = self._image_cube_cache_dir()
        token = uuid.uuid4().hex[:10]
        dataset_path = os.path.join(cache_dir, f"image_cube_{token}.nc")
        options_path = os.path.join(cache_dir, f"image_cube_{token}.json")
        script_path = os.path.join(cache_dir, f"image_cube_viewer_{token}.py")
        stdout_path = os.path.join(cache_dir, f"image_cube_{token}.out.log")
        stderr_path = os.path.join(cache_dir, f"image_cube_{token}.err.log")

        python_path = self._viewer_python_path()
        cmd = [python_path, script_path, dataset_path, options_path]
        env = self._viewer_environment(python_path)
        popen_kwargs = self._viewer_popen_kwargs()
        return {
            "cache_dir": cache_dir,
            "dataset_path": dataset_path,
            "options_path": options_path,
            "script_path": script_path,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "cmd": cmd,
            "env": env,
            "popen_kwargs": popen_kwargs,
        }

    def _image_cube_cache_dir(self):
        """Return the cache directory for temporary image cube viewer files.

        Returns:
            Absolute cache directory path.
        """
        cache_dir = os.path.join(
            generated_raster_cache_dir(QgsProject.instance()), "image_cube"
        )
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _write_cube_dataset(self, dataset, path):
        """Write a prepared cube dataset to NetCDF.

        Args:
            dataset: Xarray dataset.
            path: Output NetCDF path.
        """
        _write_cube_dataset_file(dataset, path)

    def _json_safe_options(self, options):
        """Return image cube options that can be serialized to JSON.

        Args:
            options: Raw keyword arguments for ``hypercoast.image_cube``.

        Returns:
            JSON-serializable dictionary.
        """
        return _json_safe_options(options)

    def _viewer_python_path(self):
        """Return the Python interpreter used to launch the cube viewer.

        Returns:
            Absolute Python executable path, or ``python`` as a fallback.
        """
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            executable = "python.exe" if os.name == "nt" else "python"
            conda_python = os.path.join(conda_prefix, executable)
            if os.path.isfile(conda_python):
                return conda_python

        try:
            from ..core.venv_manager import get_venv_python_path

            venv_python = get_venv_python_path()
            if os.path.isfile(venv_python):
                return venv_python
        except Exception:
            pass  # nosec B110

        if sys.executable and os.path.isfile(sys.executable):
            return sys.executable
        return "python"

    def _viewer_environment(self, python_path):
        """Return a subprocess environment for the standalone cube viewer.

        Args:
            python_path: Python executable used by the viewer.

        Returns:
            Environment dictionary.
        """
        env = os.environ.copy()
        for name in (
            "PYTHONHOME",
            "PYTHONPATH",
            "QGIS_PREFIX_PATH",
            "QGIS_PLUGINPATH",
            "PYQGIS_STARTUP",
        ):
            env.pop(name, None)
        env["PYTHONIOENCODING"] = "utf-8"
        source_root = self._source_package_root()
        if source_root:
            env["PYTHONPATH"] = source_root
        python_dir = os.path.dirname(python_path)
        if python_dir:
            env["PATH"] = python_dir + os.pathsep + env.get("PATH", "")
        return env

    def _source_package_root(self):
        """Return the repository root when running from a source checkout.

        Returns:
            Repository root path, or None when the HyperCoast package is not
            available beside the QGIS plugin checkout.
        """
        root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        return root if os.path.isdir(os.path.join(root, "hypercoast")) else None

    def _viewer_popen_kwargs(self):
        """Return platform-specific arguments for launching the viewer.

        Returns:
            Dictionary of ``subprocess.Popen`` keyword arguments.
        """
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            return {"creationflags": creationflags}
        return {"start_new_session": True}

    def _viewer_script(self):
        """Return the standalone image cube viewer script.

        Returns:
            Python source code for the viewer process.
        """
        return _viewer_script_source()

    def _subset_dataset(self, dataset, variable, bbox):
        """Subset a dataset to a WGS84 bounding box.

        Args:
            dataset: Xarray dataset.
            variable: Data variable name.
            bbox: Optional ``[xmin, ymin, xmax, ymax]`` bbox in EPSG:4326.

        Returns:
            Spatially subset dataset, or the original dataset when no bbox is
            supplied.

        Raises:
            ValueError: If a bbox is supplied but cannot be applied.
        """
        if not bbox:
            return dataset
        subset = self._subset_by_2d_latlon(dataset, bbox)
        if subset is not None:
            return subset
        subset = self._subset_by_1d_coords(dataset, bbox)
        if subset is not None:
            return subset
        if variable in dataset.data_vars:
            subset = self._subset_by_1d_coords(dataset[variable].to_dataset(), bbox)
            if subset is not None:
                return subset
        raise ValueError(
            "The selected dataset does not expose latitude/longitude coordinates "
            "that can be subset from the drawn rectangle."
        )

    def _subset_by_2d_latlon(self, dataset, bbox):
        """Subset a dataset with 2D latitude and longitude coordinates.

        Args:
            dataset: Xarray dataset.
            bbox: ``[xmin, ymin, xmax, ymax]`` in EPSG:4326.

        Returns:
            Subset dataset, or None when 2D lat/lon coordinates are unavailable.
        """
        if "latitude" not in dataset or "longitude" not in dataset:
            return None
        lat = dataset["latitude"]
        lon = dataset["longitude"]
        if lat.ndim != 2 or lon.ndim != 2:
            return None

        import numpy as np

        xmin, ymin, xmax, ymax = bbox
        mask = (
            (lon.values >= xmin)
            & (lon.values <= xmax)
            & (lat.values >= ymin)
            & (lat.values <= ymax)
        )
        if not np.any(mask):
            raise ValueError("The drawn rectangle does not intersect the dataset.")
        rows, cols = np.where(mask)
        y_dim, x_dim = lat.dims
        return dataset.isel(
            {
                y_dim: slice(int(rows.min()), int(rows.max()) + 1),
                x_dim: slice(int(cols.min()), int(cols.max()) + 1),
            }
        )

    def _subset_by_1d_coords(self, dataset, bbox):
        """Subset a dataset with 1D latitude/longitude or x/y coordinates.

        Args:
            dataset: Xarray dataset.
            bbox: ``[xmin, ymin, xmax, ymax]`` in EPSG:4326.

        Returns:
            Subset dataset, or None when usable 1D coordinates are unavailable.
        """
        x_name, y_name = self._xy_coord_names(dataset)
        if x_name is None or y_name is None:
            return None
        x_coord = dataset.coords.get(x_name)
        y_coord = dataset.coords.get(y_name)
        if x_coord is None or y_coord is None or x_coord.ndim != 1 or y_coord.ndim != 1:
            return None

        xmin, ymin, xmax, ymax = bbox
        indexers = {
            x_name: self._coord_slice(x_coord.values, xmin, xmax),
            y_name: self._coord_slice(y_coord.values, ymin, ymax),
        }
        subset = dataset.sel(indexers)
        if subset.sizes.get(x_name, 0) == 0 or subset.sizes.get(y_name, 0) == 0:
            raise ValueError("The drawn rectangle does not intersect the dataset.")
        return subset

    def _xy_coord_names(self, dataset):
        """Return likely horizontal coordinate names.

        Args:
            dataset: Xarray dataset.

        Returns:
            Tuple of x and y coordinate names, or ``(None, None)``.
        """
        for x_name, y_name in (
            ("longitude", "latitude"),
            ("lon", "lat"),
            ("x", "y"),
        ):
            if x_name in dataset.coords and y_name in dataset.coords:
                return x_name, y_name
        return None, None

    def _coord_slice(self, values, minimum, maximum):
        """Return a coordinate slice that respects axis order.

        Args:
            values: 1D coordinate values.
            minimum: Minimum requested coordinate.
            maximum: Maximum requested coordinate.

        Returns:
            Slice for xarray selection.
        """
        if len(values) > 1 and values[0] > values[-1]:
            return slice(maximum, minimum)
        return slice(minimum, maximum)

    def _validate_cube_size(self, dataset, variable):
        """Return whether a cube is small enough to render safely.

        Args:
            dataset: Xarray dataset.
            variable: Data variable name.

        Returns:
            True when rendering may proceed.
        """
        if variable not in dataset.data_vars:
            return True
        point_count = 1
        for size in dataset[variable].shape:
            point_count *= int(size)
        if point_count <= MAX_CUBE_POINTS:
            return True

        message = (
            f"The requested cube has {point_count:,} values after subsetting and "
            "stride. Draw a smaller subset or increase the spatial/spectral stride "
            "before creating a 3D cube."
        )
        QMessageBox.warning(self, "3D Image Cube", message)
        self.status_label.setText("3D image cube too large")
        return False

    def _selected_variable_name(self, dataset_obj):
        """Return the selected data variable name.

        Args:
            dataset_obj: HyperCoast dataset wrapper.

        Returns:
            Data variable name, or None.
        """
        variable = self.variable_combo.currentData()
        if variable:
            return variable
        data_var = dataset_obj.get_data_variable()
        return getattr(data_var, "name", None)

    def _image_cube_kwargs(self, dataset, variable):
        """Return keyword arguments for ``hypercoast.image_cube``.

        Args:
            dataset: Xarray dataset to visualize.
            variable: Data variable name.

        Returns:
            Dictionary of image cube options.
        """
        kwargs = {
            "variable": variable,
            "cmap": self.cmap_combo.currentText(),
            "clim": (self.vmin_spin.value(), self.vmax_spin.value()),
            "title": self._cube_title(variable),
            "rgb_gamma": self.gamma_spin.value(),
            "widget": self.widget_combo.currentData(),
            "crop": self.crop_spin.value() or None,
            "nodata": (
                self.nodata_spin.value() if self.nodata_check.isChecked() else None
            ),
        }
        if self._has_wavelength_axis(dataset):
            kwargs["rgb_wavelengths"] = [
                self.red_spin.value(),
                self.green_spin.value(),
                self.blue_spin.value(),
            ]
        return kwargs

    def _cube_title(self, variable):
        """Return a readable scalar bar title.

        Args:
            variable: Data variable name.

        Returns:
            Display title.
        """
        if "reflectance" in variable.lower():
            return "Reflectance"
        if "radiance" in variable.lower():
            return "Radiance"
        return variable

    def _downsample_dataset(self, dataset, spatial_stride, spectral_stride):
        """Downsample a dataset before 3D rendering.

        Args:
            dataset: Xarray dataset.
            spatial_stride: Stride for the first two spatial dimensions.
            spectral_stride: Stride for spectral dimensions.

        Returns:
            Downsampled dataset.
        """
        indexers = {}
        spatial_dims = [
            dim
            for dim in dataset.sizes
            if dim not in SPECTRAL_DIMS and dataset.sizes[dim] > 1
        ]
        if spatial_stride > 1:
            for dim in spatial_dims[:2]:
                indexers[dim] = slice(None, None, spatial_stride)
        if spectral_stride > 1:
            for dim in dataset.sizes:
                if dim in SPECTRAL_DIMS:
                    indexers[dim] = slice(None, None, spectral_stride)
        return dataset.isel(indexers) if indexers else dataset

    def _prepare_image_cube_dataset(self, dataset, variable):
        """Return a dataset with the selected cube's spectral axis last.

        ``hypercoast.image_cube`` expects RGB wavelength selections to produce
        an array shaped as rows, columns, and RGB channels. Some products, such
        as Tanager HDF5 assets, can store the spectral axis first, which would
        otherwise make the RGB overlay invalid.

        Args:
            dataset: Xarray dataset.
            variable: Data variable name.

        Returns:
            Dataset with the selected variable transposed when needed.
        """
        if variable not in dataset.data_vars:
            return dataset
        data_var = dataset[variable]
        spectral_dims = [dim for dim in data_var.dims if dim in SPECTRAL_DIMS]
        if not spectral_dims:
            return dataset
        spectral_dim = spectral_dims[0]
        if data_var.dims[-1] == spectral_dim:
            return dataset

        ordered_dims = [dim for dim in data_var.dims if dim != spectral_dim]
        ordered_dims.append(spectral_dim)
        prepared = dataset.copy(deep=False)
        prepared[variable] = data_var.transpose(*ordered_dims)
        return prepared

    def _has_wavelength_axis(self, dataset):
        """Return whether ``image_cube`` can add an RGB wavelength overlay.

        Args:
            dataset: Xarray dataset.

        Returns:
            True when a ``wavelength`` dimension or coordinate is present.
        """
        return "wavelength" in dataset.dims or "wavelength" in dataset.coords

    def _show_plotter(self, plotter):
        """Show a PyVista plotter with a non-closing preference.

        Args:
            plotter: PyVista plotter-like object.
        """
        show = getattr(plotter, "show", None)
        if not callable(show):
            return
        for kwargs in (
            {"auto_close": False, "interactive_update": True},
            {"auto_close": False},
            {},
        ):
            try:
                show(**kwargs)
                return
            except TypeError:
                continue

    def _sync_nodata_state(self, *args):
        """Enable or disable the NoData value spin box."""
        self.nodata_spin.setEnabled(self.nodata_check.isChecked())

    def _project_layer(self, layer_id):
        """Return a QGIS project layer.

        Args:
            layer_id: QGIS layer ID.

        Returns:
            QGIS layer object, or None.
        """
        try:
            return QgsProject.instance().mapLayer(layer_id)
        except Exception:
            return None
