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
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
)


class AboutDialog(QDialog):
    """About dialog for HyperCoast plugin."""

    def __init__(self, parent=None):
        """Initialize the About dialog.

        :param parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("About HyperCoast")
        self.setMinimumWidth(450)
        self.setup_ui()
        self.adjustSize()

    def setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 15, 20, 15)

        # Icon and title
        header_layout = QHBoxLayout()

        # Load icon
        icon_label = QLabel()
        plugin_dir = os.path.dirname(os.path.dirname(__file__))
        icon_path = os.path.join(plugin_dir, "icons", "hypercoast.png")
        if os.path.exists(icon_path):
            pixmap = QPixmap(icon_path).scaled(
                64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            icon_label.setPixmap(pixmap)
        header_layout.addWidget(icon_label)

        # Title
        title_layout = QVBoxLayout()
        title_label = QLabel("<h2>HyperCoast</h2>")
        version_label = QLabel("<b>Version:</b> 0.1.0")
        title_layout.addWidget(title_label)
        title_layout.addWidget(version_label)
        title_layout.addStretch()
        header_layout.addLayout(title_layout)
        header_layout.addStretch()

        layout.addLayout(header_layout)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
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
        description.setTextFormat(Qt.RichText)
        layout.addWidget(description)

        # Author info
        author_label = QLabel(
            "<b>Website:</b> <a href='https://hypercoast.org'>https://hypercoast.org</a><br>"
            "<b>License:</b> MIT"
        )
        author_label.setWordWrap(True)
        author_label.setTextFormat(Qt.RichText)
        author_label.setOpenExternalLinks(True)
        layout.addWidget(author_label)

        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_button = QPushButton("Close")
        close_button.setFixedWidth(100)
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)
