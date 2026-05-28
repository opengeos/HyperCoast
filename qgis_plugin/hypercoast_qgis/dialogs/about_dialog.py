# -*- coding: utf-8 -*-
"""
HyperCoast QGIS Plugin - About Dialog

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

import os

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QPixmap
from qgis.PyQt.QtWidgets import (
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class AboutDialog(QDockWidget):
    """Dockable About panel for HyperCoast plugin."""

    def __init__(self, parent=None):
        """Initialize the About panel.

        :param parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("About HyperCoast")
        self.setObjectName("HyperCoastAboutDock")
        self.setMinimumWidth(360)
        self.setup_ui()

    def setup_ui(self):
        """Set up the panel UI."""
        content = QWidget(self)
        self.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setSpacing(12)
        layout.setContentsMargins(14, 14, 14, 14)

        # Icon and title
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)
        header_layout.setContentsMargins(0, 0, 0, 0)

        # Load icon
        icon_label = QLabel()
        icon_label.setFixedSize(64, 64)
        icon_label.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        )
        plugin_dir = os.path.dirname(os.path.dirname(__file__))
        icon_path = os.path.join(plugin_dir, "icons", "hypercoast.png")
        if os.path.exists(icon_path):
            pixmap = QPixmap(icon_path).scaled(
                64,
                64,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            icon_label.setPixmap(pixmap)
        header_layout.addWidget(icon_label)

        # Title
        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_label = QLabel("<h2 style='margin: 0;'>HyperCoast</h2>")
        version_label = QLabel(f"<b>Version:</b> {self._get_plugin_version()}")
        title_layout.addWidget(title_label)
        title_layout.addWidget(version_label)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()

        layout.addLayout(header_layout)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        # Description
        description = QLabel(
            "<b>HyperCoast QGIS Plugin</b> provides powerful tools for "
            "hyperspectral data visualization and analysis.<br><br>"
            "<b>Features:</b><br>"
            "• Load and display hyperspectral formats (EMIT, PACE, DESIS, etc.)<br>"
            "• Interactive band combination visualization<br>"
            "• Spectral signature inspection with click-to-plot<br>"
            "• Export spectral profiles to CSV"
        )
        description.setWordWrap(True)
        description.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(description)

        # Author info
        author_label = QLabel(
            "<b>Website:</b> <a href='https://hypercoast.org'>https://hypercoast.org</a><br>"
            "<b>License:</b> MIT"
        )
        author_label.setWordWrap(True)
        author_label.setTextFormat(Qt.TextFormat.RichText)
        author_label.setOpenExternalLinks(True)
        layout.addWidget(author_label)

        layout.addStretch(1)

        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_button = QPushButton("Close")
        close_button.setFixedWidth(100)
        close_button.clicked.connect(self.close)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

    def _get_plugin_version(self):
        """Return the plugin version from metadata.txt."""
        plugin_dir = os.path.dirname(os.path.dirname(__file__))
        metadata_path = os.path.join(plugin_dir, "metadata.txt")
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("version="):
                        return line.split("=", 1)[1].strip()
        except OSError:
            pass
        return "Unknown"
