# -*- coding: utf-8 -*-
"""
HyperCoast QGIS Plugin - Dependency Installer Dialog

Provides a dialog for checking and installing plugin dependencies
into an isolated virtual environment.

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

import traceback

from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.PyQt.QtGui import QColor, QFont
from qgis.PyQt.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
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
            from ..venv_manager import create_venv_and_install

            success, message = create_venv_and_install(
                plugin_dir=self.plugin_dir,
                progress_callback=lambda percent, msg: self.progress.emit(percent, msg),
                cancel_check=lambda: self._cancelled,
            )
            self.finished.emit(success, message)
        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.finished.emit(False, error_msg)


class DependencyInstallerDialog(QDialog):
    """Dialog for checking and installing plugin dependencies."""

    deps_installed = pyqtSignal()

    def __init__(self, plugin_dir, parent=None):
        """Initialize the dialog.

        Args:
            plugin_dir: Path to the plugin directory containing requirements.txt.
            parent: Optional parent QWidget.
        """
        super().__init__(parent)
        self.plugin_dir = plugin_dir
        self.install_worker = None

        self.setWindowTitle("HyperCoast - Install Dependencies")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        self._setup_ui()
        self._check_packages()

    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header
        header_label = QLabel("HyperCoast Dependency Manager")
        header_font = QFont()
        header_font.setPointSize(14)
        header_font.setBold(True)
        header_label.setFont(header_font)
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)

        # Description
        desc_label = QLabel(
            "The HyperCoast plugin requires several Python packages. "
            "These will be installed in an isolated virtual environment "
            "so your QGIS Python environment stays clean."
        )
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc_label)

        # Package table
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
        layout.addWidget(self.package_table)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Progress label
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("font-size: 11px; color: palette(text);")
        layout.addWidget(self.progress_label)

        # Buttons
        button_layout = QHBoxLayout()

        self.install_btn = QPushButton("Install Dependencies")
        self.install_btn.clicked.connect(self._on_install_clicked)
        button_layout.addWidget(self.install_btn)

        self.cancel_btn = QPushButton("Cancel Installation")
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        self.cancel_btn.setVisible(False)
        button_layout.addWidget(self.cancel_btn)

        self.recheck_btn = QPushButton("Re-check")
        self.recheck_btn.clicked.connect(self._check_packages)
        button_layout.addWidget(self.recheck_btn)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)

        # Info label
        info_label = QLabel(
            "<small>Packages are installed in an isolated virtual environment "
            "(~/.qgis_hypercoast/). QGIS may need to be restarted after "
            "installation for some changes to take effect.</small>"
        )
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)

    def _check_packages(self):
        """Check which packages are installed and update the table."""
        from ..venv_manager import check_packages

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
            self.install_btn.setText("Install Dependencies")

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
        reply = QMessageBox.question(
            self,
            "Install Dependencies",
            "This will create a virtual environment and install all required "
            "packages. This may take a few minutes.\n\n"
            "Do you want to continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if reply != QMessageBox.Yes:
            return

        self.install_btn.setEnabled(False)
        self.recheck_btn.setEnabled(False)
        self.close_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setVisible(True)
        self.progress_label.setText("Starting installation...")

        self.status_label.setText("Installing dependencies...")
        self.status_label.setStyleSheet("color: palette(text); font-weight: bold;")

        self.install_worker = DepsInstallWorker(self.plugin_dir)
        self.install_worker.progress.connect(self._on_install_progress)
        self.install_worker.finished.connect(self._on_install_finished)
        self.install_worker.start()

    def _on_cancel_clicked(self):
        """Handle cancel button click during installation."""
        if self.install_worker and self.install_worker.isRunning():
            self.install_worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.cancel_btn.setText("Cancelling...")

    def _on_install_progress(self, percent, message):
        """Update progress display.

        Args:
            percent: Progress percentage (0-100).
            message: Status message.
        """
        self.progress_bar.setValue(percent)
        self.progress_label.setText(message)

    def _on_install_finished(self, success, message):
        """Handle installation completion.

        Args:
            success: Whether installation succeeded.
            message: Result message.
        """
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.cancel_btn.setVisible(False)
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.setText("Cancel Installation")
        self.recheck_btn.setEnabled(True)
        self.close_btn.setEnabled(True)

        # Re-check packages to update the table
        self._check_packages()

        if success:
            self.status_label.setText("Dependencies installed successfully!")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            QMessageBox.information(
                self,
                "Installation Complete",
                "All dependencies have been installed successfully.\n\n"
                "You may need to restart QGIS for some changes to take effect.",
            )
            self.deps_installed.emit()
        else:
            self.status_label.setText(f"Installation issue: {message[:100]}")
            self.status_label.setStyleSheet("color: #d32f2f; font-weight: bold;")
            QMessageBox.warning(
                self,
                "Installation Issue",
                f"There was an issue during installation:\n\n{message}\n\n"
                "You can try clicking 'Install Dependencies' again.",
            )
            self.install_btn.setEnabled(True)

    def closeEvent(self, event):
        """Handle dialog close event.

        Args:
            event: QCloseEvent.
        """
        if self.install_worker and self.install_worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Installation in Progress",
                "An installation is in progress. "
                "Are you sure you want to cancel and close?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
            self.install_worker.cancel()
            self.install_worker.wait(5000)

        event.accept()
