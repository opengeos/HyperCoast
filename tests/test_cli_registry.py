# SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for the CLI and sensor registry."""

import pytest

from hypercoast import cli
from hypercoast.registry import get_sensor, list_sensors


def test_registry_lists_core_sensors():
    """The registry should expose the core sensor handlers."""
    sensors = list_sensors()

    assert "aviris" in sensors
    assert "emit" in sensors
    assert "pace" in sensors
    assert "tanager" in sensors


def test_registry_resolves_aliases():
    """Sensor aliases should resolve to canonical handlers."""
    assert get_sensor("planet-tanager").name == "tanager"
    assert get_sensor("neon_aop").name == "neon"


def test_registry_rejects_unknown_sensor():
    """Unknown sensors should raise a useful lookup error."""
    with pytest.raises(KeyError, match="Unsupported sensor"):
        get_sensor("unknown")


def test_cli_info(capsys):
    """The info command should print version and sensor details."""
    assert cli.main(["info"]) == 0

    output = capsys.readouterr().out
    assert "HyperCoast" in output
    assert "Sensors:" in output


def test_cli_sensors(capsys):
    """The sensors command should list registered sensors."""
    assert cli.main(["sensors"]) == 0

    output = capsys.readouterr().out
    assert "pace" in output
    assert "tanager" in output
