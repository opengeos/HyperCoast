# -*- coding: utf-8 -*-
"""
HyperCoast QGIS Plugin - Settings Dock Widget

Provides a dockable settings panel for managing plugin dependencies.

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

import traceback

from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.PyQt.QtGui import QColor, QFont
from qgis.PyQt.QtWidgets import (
    QDockWidget,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class DepsInstallWorker(QThread):
    """Worker thread for creating venv and installing dependencies."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, plugin_dir, parent=None):
        """Initialize the worker.

        Args:
            plugin_dir: Path to the plugin directory containing requirements.txt.
            parent: Optional parent QObject.
        """
        super().__init__(parent)
        self.plugin_dir = plugin_dir
        self._cancelled = False

    def cancel(self):
        """Request cancellation of the installation."""
        self._cancelled = True

    def run(self):
        """Run the venv creation and dependency installation."""
        try:
            from ..core.venv_manager import create_venv_and_install

            success, message = create_venv_and_install(
                plugin_dir=self.plugin_dir,
                progress_callback=lambda percent, msg: self.progress.emit(percent, msg),
                cancel_check=lambda: self._cancelled,
            )
            self.finished.emit(success, message)
        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.finished.emit(False, error_msg)


class SettingsDockWidget(QDockWidget):
    """A dockable settings panel for managing HyperCoast plugin dependencies."""

    # Emitted when dependencies are successfully installed
    deps_installed = pyqtSignal()

    def __init__(self, plugin_dir, iface, parent=None):
        """Initialize the settings dock widget.

        Args:
            plugin_dir: Path to the plugin directory containing requirements.txt.
            iface: QGIS interface instance.
            parent: Parent widget.
        """
        super().__init__("HyperCoast Settings", parent)
        self.plugin_dir = plugin_dir
        self.iface = iface
        self._deps_worker = None

        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self._setup_ui()
        self._check_packages()

    def _setup_ui(self):
        """Set up the settings UI."""
        main_widget = QWidget()
        self.setWidget(main_widget)

        layout = QVBoxLayout(main_widget)
        layout.setSpacing(10)

        # Header
        header_label = QLabel("HyperCoast Settings")
        header_font = QFont()
        header_font.setPointSize(11)
        header_font.setBold(True)
        header_label.setFont(header_font)
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)

        # Info label
        info_label = QLabel(
            "This plugin requires additional Python packages.\n"
            "Click 'Install Dependencies' to install them in an\n"
            "isolated virtual environment (~/.qgis_hypercoast/).\n"
            "A standalone Python interpreter will be downloaded\n"
            "on first install."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(info_label)

        # Package status group
        status_group = QGroupBox("Package Status")
        status_layout = QVBoxLayout(status_group)

        self.package_table = QTableWidget()
        self.package_table.setColumnCount(4)
        self.package_table.setHorizontalHeaderLabels(
            ["Package", "Required", "Installed", "Status"]
        )
        self.package_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.package_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self.package_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        self.package_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeToContents
        )
        self.package_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.package_table.setSelectionMode(QTableWidget.NoSelection)
        self.package_table.verticalHeader().setVisible(False)
        status_layout.addWidget(self.package_table)

        layout.addWidget(status_group)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Install button
        self.install_btn = QPushButton("Install Dependencies")
        self.install_btn.setStyleSheet("background-color: #2e7d32; color: white;")
        self.install_btn.clicked.connect(self._on_install_clicked)
        layout.addWidget(self.install_btn)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Progress label
        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        self.progress_label.setVisible(False)
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(self.progress_label)

        # Cancel button (hidden by default)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet("color: red;")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        layout.addWidget(self.cancel_btn)

        # Button row
        btn_layout = QHBoxLayout()

        self.refresh_btn = QPushButton("Refresh Status")
        self.refresh_btn.clicked.connect(self._check_packages)
        btn_layout.addWidget(self.refresh_btn)

        layout.addLayout(btn_layout)

        # Stretch at the end
        layout.addStretch()

    def _check_packages(self):
        """Check which packages are installed and update the table."""
        from ..core.venv_manager import check_packages

        packages = check_packages(self.plugin_dir)
        self._populate_table(packages)

        missing_count = sum(
            1 for p in packages if p["status"] in ("missing", "outdated")
        )
        total_count = len(packages)

        if missing_count == 0:
            self.status_label.setText(
                f"All {total_count} packages are installed and up to date."
            )
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.install_btn.setText("Re-install Dependencies")
        else:
            self.status_label.setText(
                f"{missing_count} of {total_count} packages need to be installed."
            )
            self.status_label.setStyleSheet("color: #d32f2f; font-weight: bold;")
            self.install_btn.setText(
                f"Install Dependencies ({missing_count} missing)"
            )

        self.install_btn.setEnabled(True)

    def _populate_table(self, packages):
        """Populate the package table with check results.

        Args:
            packages: List of package status dicts from check_packages().
        """
        self.package_table.setRowCount(len(packages))

        for row, pkg in enumerate(packages):
            # Package name
            name_item = QTableWidgetItem(pkg["package"])
            self.package_table.setItem(row, 0, name_item)

            # Required version
            req_item = QTableWidgetItem(pkg["required"])
            req_item.setTextAlignment(Qt.AlignCenter)
            self.package_table.setItem(row, 1, req_item)

            # Installed version
            installed = pkg["installed"] or "Not installed"
            inst_item = QTableWidgetItem(installed)
            inst_item.setTextAlignment(Qt.AlignCenter)
            self.package_table.setItem(row, 2, inst_item)

            # Status
            status = pkg["status"]
            if status == "ok":
                status_text = "OK"
                color = QColor(46, 125, 50)  # Green
            elif status == "outdated":
                status_text = "Outdated"
                color = QColor(245, 124, 0)  # Orange
            else:
                status_text = "Missing"
                color = QColor(211, 47, 47)  # Red

            status_item = QTableWidgetItem(status_text)
            status_item.setTextAlignment(Qt.AlignCenter)
            status_item.setForeground(color)
            font = status_item.font()
            font.setBold(True)
            status_item.setFont(font)
            self.package_table.setItem(row, 3, status_item)

    def _on_install_clicked(self):
        """Handle install button click."""
        # Guard against concurrent installs
        if self._deps_worker is not None and self._deps_worker.isRunning():
            return

        reply = QMessageBox.question(
            self,
            "Install Dependencies",
            "This will download a standalone Python interpreter and install all "
            "required packages. This may take a few minutes.\n\n"
            "Do you want to continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if reply != QMessageBox.Yes:
            return

        # Update UI for installation mode
        self.install_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setVisible(True)
        self.progress_label.setText("Starting installation...")
        self.progress_label.setStyleSheet("font-size: 10px;")
        self.cancel_btn.setVisible(True)
        self.cancel_btn.setEnabled(True)

        self.status_label.setText("Installing dependencies...")
        self.status_label.setStyleSheet("font-weight: bold;")

        # Start worker
        self._deps_worker = DepsInstallWorker(self.plugin_dir)
        self._deps_worker.progress.connect(self._on_progress)
        self._deps_worker.finished.connect(self._on_finished)
        self._deps_worker.start()

    def _on_cancel_clicked(self):
        """Handle cancel button click during installation."""
        if self._deps_worker is not None and self._deps_worker.isRunning():
            self._deps_worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.progress_label.setText("Cancelling...")

    def _on_progress(self, percent, message):
        """Handle progress updates from the dependency install worker.

        Args:
            percent: Installation progress percentage (0-100).
            message: Status message describing current operation.
        """
        self.progress_bar.setValue(percent)
        self.progress_label.setText(message)

    def _on_finished(self, success, message):
        """Handle completion of the dependency installation.

        Args:
            success: Whether installation succeeded.
            message: Result message.
        """
        # Reset UI
        self.progress_bar.setVisible(False)
        self.progress_label.setText(message)
        self.progress_label.setVisible(True)
        self.cancel_btn.setVisible(False)
        self.refresh_btn.setEnabled(True)

        if success:
            self.progress_label.setStyleSheet("color: green; font-size: 10px;")
            self.iface.messageBar().pushSuccess(
                "HyperCoast", "Dependencies installed successfully!"
            )
            self.deps_installed.emit()
        else:
            self.progress_label.setStyleSheet("color: red; font-size: 10px;")
            self.install_btn.setEnabled(True)

        # Refresh status display
        self._check_packages()

    def show_dependencies_tab(self):
        """Refresh the dependency status display.

        Called externally when this dock is opened for the first time
        or when the user needs to be prompted to install dependencies.
        """
        self._check_packages()
