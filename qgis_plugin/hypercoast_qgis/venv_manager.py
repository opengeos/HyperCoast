# -*- coding: utf-8 -*-
"""
Backward-compatibility shim.

All logic has been moved to core.venv_manager. This module re-exports
everything so that existing imports continue to work.

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

from .core.venv_manager import *  # noqa: F401,F403
