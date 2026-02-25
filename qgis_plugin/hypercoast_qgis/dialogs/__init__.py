# -*- coding: utf-8 -*-
"""
HyperCoast QGIS Plugin - Dialogs Package

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

# Dependency installer has no heavy deps, always available
from .dependency_installer import DependencyInstallerDialog

# These dialogs have no heavy deps, always available
from .update_checker import UpdateCheckerDialog

# Data-related dialogs depend on numpy (via hyperspectral_provider),
# so they may fail to import if dependencies are not installed yet.
try:
    from .load_data_dialog import LoadDataDialog
    from .band_combination_dialog import BandCombinationDialog
    from .spectral_inspector_tool import SpectralInspectorTool
    from .spectral_plot_dialog import SpectralPlotDialog

    _DATA_DIALOGS_AVAILABLE = True
except ImportError:
    _DATA_DIALOGS_AVAILABLE = False

__all__ = [
    "DependencyInstallerDialog",
    "LoadDataDialog",
    "BandCombinationDialog",
    "SpectralInspectorTool",
    "SpectralPlotDialog",
    "UpdateCheckerDialog",
    "_DATA_DIALOGS_AVAILABLE",
]
