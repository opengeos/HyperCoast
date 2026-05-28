# SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for optional ML helpers."""

import pytest

from hypercoast import ml


def test_list_models_includes_moe_vae():
    """ML metadata should advertise the optional MoE/VAE family."""
    models = ml.list_models()

    assert models["moe_vae"]["required_extra"] == "ml"


def test_require_ml_dependencies_reports_missing_torch(monkeypatch):
    """Missing optional ML dependencies should raise an actionable error."""
    monkeypatch.setattr(ml, "find_spec", lambda name: None)

    with pytest.raises(ImportError, match="ml"):
        ml.require_ml_dependencies()
