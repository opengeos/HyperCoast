"""Tests for QGIS plugin packaging and install folder conventions."""

import sys
import zipfile
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

import install_plugin  # noqa: E402


def test_install_plugin_zip_uses_hypercoast_root(tmp_path):
    """Package files under the official folder name."""
    zip_path = install_plugin.package_plugin(tmp_path)

    assert zip_path.name == f"hypercoast_{install_plugin.VERSION}.zip"

    with zipfile.ZipFile(zip_path) as zip_file:
        names = zip_file.namelist()

    top_level = {name.split("/", 1)[0] for name in names if name}
    assert top_level == {"hypercoast"}
    assert "hypercoast/metadata.txt" in names
    assert not any(name.startswith("hypercoast_qgis/") for name in names)


def test_install_plugin_installs_to_hypercoast_folder(tmp_path, monkeypatch):
    """Local installation should create a hypercoast plugin folder."""
    zip_path = install_plugin.package_plugin(tmp_path / "dist")
    plugins_dir = tmp_path / "plugins"
    monkeypatch.setattr(
        install_plugin,
        "get_qgis_plugins_dir",
        lambda: plugins_dir,
    )

    install_path = install_plugin.install_plugin(zip_path=zip_path, force=True)

    assert install_path == plugins_dir / "hypercoast"
    assert (install_path / "metadata.txt").is_file()
    assert not (plugins_dir / "hypercoast_qgis").exists()


def test_readme_advertises_hypercoast_install_folder():
    """README should document hypercoast as the QGIS plugin folder name."""
    readme = (PLUGIN_ROOT / "README.md").read_text(encoding="utf-8")

    assert "name the copied folder `hypercoast`" in readme
    assert "Copy the `hypercoast_qgis` folder" not in readme
