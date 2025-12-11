# -*- coding: utf-8 -*-
"""
HyperCoast QGIS Plugin
A plugin for visualizing and analyzing hyperspectral data

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""


def classFactory(iface):
    """Load HyperCoastPlugin class from file hypercoast_plugin.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    from .hypercoast_plugin import HyperCoastPlugin

    return HyperCoastPlugin(iface)
