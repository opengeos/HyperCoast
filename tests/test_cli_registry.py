# SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for the CLI and sensor registry."""

import pytest

from hypercoast import cli
from hypercoast.registry import get_sensor, list_sensors, qgis_data_types


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


def test_qgis_data_types_use_registry_metadata():
    """QGIS metadata should be generated from the sensor registry."""
    data_types = qgis_data_types()

    assert data_types["PACE"]["variable"] == "Rrs"
    assert data_types["Wyvern"]["default_rgb"] == [799, 679, 570]
    assert "Generic" in data_types


def test_cli_registry_json(capsys):
    """The registry command should expose serializable metadata."""
    assert cli.main(["registry", "--json"]) == 0

    output = capsys.readouterr().out
    assert '"pace"' in output
    assert '"default_rgb"' in output


def test_cli_validate_and_inspect(tmp_path, capsys):
    """Validate and inspect should use registry extensions."""
    path = tmp_path / "scene.nc"
    path.write_bytes(b"placeholder")

    assert cli.main(["validate", "pace", str(path)]) == 0
    assert cli.main(["inspect", str(path), "--json"]) == 0

    output = capsys.readouterr().out
    assert "valid for pace" in output
    assert "matched_sensors" in output


def test_cli_workflow_runs_on_local_netcdf(tmp_path, capsys):
    """The workflow command should run on a small xarray fixture."""
    np = pytest.importorskip("numpy")
    xr = pytest.importorskip("xarray")
    input_path = tmp_path / "cube.nc"
    output_path = tmp_path / "ndwi.nc"
    dataset = xr.Dataset(
        {
            "reflectance": (
                ("wavelength", "y", "x"),
                np.array(
                    [
                        [[0.6, 0.8], [0.4, 0.2]],
                        [[0.2, 0.2], [0.1, 0.1]],
                    ],
                    dtype="float32",
                ),
            )
        },
        coords={"wavelength": [560.0, 860.0], "y": [0, 1], "x": [0, 1]},
    )
    dataset.to_netcdf(input_path)

    assert cli.main(["workflow", "ndwi", str(input_path), str(output_path)]) == 0

    output = capsys.readouterr().out
    assert str(output_path) in output
    result = xr.open_dataarray(output_path)
    assert result.name == "nd_560_860"
