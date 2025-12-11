#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-platform script to package and install the HyperCoast QGIS Plugin.

Usage:
    python install_plugin.py              # Package the plugin
    python install_plugin.py --install    # Package and install to QGIS
    python install_plugin.py --help       # Show help
"""

import os
import sys
import shutil
import zipfile
import argparse
import tempfile
import platform
from pathlib import Path


# Plugin configuration
PLUGIN_NAME = "hypercoast_qgis"


def get_version_from_metadata():
    """Read the version from metadata.txt."""
    script_dir = get_script_dir()
    metadata_path = script_dir / "metadata.txt"
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            if line.strip().lower().startswith("version"):
                # Handles lines like: version=0.1.0
                return line.split("=", 1)[1].strip()
    raise RuntimeError("Version not found in metadata.txt")


# Files/directories to exclude from package
EXCLUDE_PATTERNS = [
    "*.pyc",
    "__pycache__",
    "*.py.bak",
    ".DS_Store",
    "*.pyo",
    ".git",
    ".gitignore",
    "*.egg-info",
    ".pytest_cache",
    "*.log",
]


def get_script_dir():
    """Get the directory where this script is located."""
    return Path(__file__).parent.resolve()


def get_qgis_plugins_dir():
    """Get the QGIS plugins directory based on the operating system."""
    system = platform.system()
    home = Path.home()

    if system == "Windows":
        # Windows: %APPDATA%\QGIS\QGIS3\profiles\default\python\plugins
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            return (
                Path(appdata)
                / "QGIS"
                / "QGIS3"
                / "profiles"
                / "default"
                / "python"
                / "plugins"
            )
        else:
            return (
                home
                / "AppData"
                / "Roaming"
                / "QGIS"
                / "QGIS3"
                / "profiles"
                / "default"
                / "python"
                / "plugins"
            )

    elif system == "Darwin":
        # macOS: ~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins
        return (
            home
            / "Library"
            / "Application Support"
            / "QGIS"
            / "QGIS3"
            / "profiles"
            / "default"
            / "python"
            / "plugins"
        )

    else:
        # Linux: ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins
        return (
            home
            / ".local"
            / "share"
            / "QGIS"
            / "QGIS3"
            / "profiles"
            / "default"
            / "python"
            / "plugins"
        )


def should_exclude(path, patterns):
    """Check if a path should be excluded based on patterns."""
    name = path.name

    for pattern in patterns:
        if pattern.startswith("*"):
            # Extension match
            if name.endswith(pattern[1:]):
                return True
        elif name == pattern:
            return True

    return False


def copy_plugin_files(src_dir, dst_dir, exclude_patterns):
    """Copy plugin files to destination, excluding certain patterns."""
    if dst_dir.exists():
        shutil.rmtree(dst_dir)

    dst_dir.mkdir(parents=True, exist_ok=True)

    for item in src_dir.iterdir():
        if should_exclude(item, exclude_patterns):
            continue

        dst_path = dst_dir / item.name

        if item.is_dir():
            copy_plugin_files(item, dst_path, exclude_patterns)
        else:
            shutil.copy2(item, dst_path)


def create_zip_package(source_dir, output_path, plugin_name):
    """Create a ZIP package of the plugin."""
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not should_exclude(Path(d), EXCLUDE_PATTERNS)]

            for file in files:
                if should_exclude(Path(file), EXCLUDE_PATTERNS):
                    continue

                file_path = Path(root) / file
                # Create archive path with plugin name as root
                rel_path = file_path.relative_to(source_dir.parent)
                zipf.write(file_path, rel_path)

    return output_path


def package_plugin(output_dir=None):
    """Package the plugin into a ZIP file.

    Args:
        output_dir: Optional output directory. Defaults to 'dist' in script directory.

    Returns:
        Path to the created ZIP file.
    """
    script_dir = get_script_dir()
    plugin_dir = script_dir / PLUGIN_NAME

    if not plugin_dir.exists():
        raise FileNotFoundError(f"Plugin directory not found: {plugin_dir}")

    # Create output directory
    if output_dir is None:
        output_dir = script_dir / "dist"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary directory for clean copy
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_plugin_dir = Path(temp_dir) / PLUGIN_NAME

        # Copy files
        print(f"Copying plugin files...")
        copy_plugin_files(plugin_dir, temp_plugin_dir, EXCLUDE_PATTERNS)

        # Create ZIP archive
        zip_name = f"{PLUGIN_NAME}_{VERSION}.zip"
        zip_path = output_dir / zip_name

        print(f"Creating ZIP archive...")
        create_zip_package(temp_plugin_dir, zip_path, PLUGIN_NAME)

    print(f"\nPlugin packaged successfully: {zip_path}")
    return zip_path


def install_plugin(zip_path=None, force=False):
    """Install the plugin to QGIS plugins directory.

    Args:
        zip_path: Path to the plugin ZIP file. If None, packages first.
        force: If True, overwrite existing installation.

    Returns:
        Path to the installed plugin directory.
    """
    # Get or create the ZIP package
    if zip_path is None:
        zip_path = package_plugin()
    else:
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    # Get QGIS plugins directory
    plugins_dir = get_qgis_plugins_dir()

    # Check if plugins directory exists
    if not plugins_dir.parent.exists():
        print(f"\nWarning: QGIS profile directory not found: {plugins_dir.parent}")
        print("Please ensure QGIS is installed and has been run at least once.")
        while True:
            try:
                create = input("Create directory anyway? [y/N]: ").strip().lower()
            except EOFError:
                create = ""
            if create in ("y", "n", ""):
                break
            print("Please enter 'y' to create the directory or 'n' to cancel.")
        if create != "y":
            return None

    plugins_dir.mkdir(parents=True, exist_ok=True)

    # Install path
    install_path = plugins_dir / PLUGIN_NAME

    # Check for existing installation
    if install_path.exists():
        if not force:
            print(f"\nPlugin already installed at: {install_path}")
            overwrite = (
                input("Overwrite existing installation? [y/N]: ").strip().lower()
            )
            if overwrite != "y":
                print("Installation cancelled.")
                return None

        print(f"Removing existing installation...")
        shutil.rmtree(install_path)

    # Extract ZIP to plugins directory
    print(f"Installing plugin to: {install_path}")

    with zipfile.ZipFile(zip_path, "r") as zipf:
        zipf.extractall(plugins_dir)

    print(f"\nPlugin installed successfully!")
    print(f"\nTo enable the plugin:")
    print(f"  1. Open QGIS")
    print(f"  2. Go to Plugins -> Manage and Install Plugins")
    print(f"  3. Find 'HyperCoast' in the list")
    print(f"  4. Check the box to enable it")

    return install_path


def uninstall_plugin():
    """Uninstall the plugin from QGIS plugins directory."""
    plugins_dir = get_qgis_plugins_dir()
    install_path = plugins_dir / PLUGIN_NAME

    if not install_path.exists():
        print(f"Plugin not found at: {install_path}")
        return False

    confirm = input(f"Remove plugin from {install_path}? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Uninstallation cancelled.")
        return False

    shutil.rmtree(install_path)
    print(f"Plugin uninstalled successfully from: {install_path}")
    return True


def show_info():
    """Show plugin and system information."""
    script_dir = get_script_dir()
    plugins_dir = get_qgis_plugins_dir()
    install_path = plugins_dir / PLUGIN_NAME

    print(f"\n{'=' * 60}")
    print(f"HyperCoast QGIS Plugin Information")
    print(f"{'=' * 60}")
    print(f"Plugin Name:     {PLUGIN_NAME}")
    print(f"Version:         {VERSION}")
    print(f"Script Location: {script_dir}")
    print(f"{'=' * 60}")
    print(f"System:          {platform.system()} {platform.release()}")
    print(f"Python:          {sys.version.split()[0]}")
    print(f"{'=' * 60}")
    print(f"QGIS Plugins Directory:")
    print(f"  {plugins_dir}")
    print(f"  Exists: {plugins_dir.exists()}")
    print(f"{'=' * 60}")
    print(f"Installation Status:")
    if install_path.exists():
        print(f"  Installed at: {install_path}")
    else:
        print(f"  Not installed")
    print(f"{'=' * 60}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Package and install the HyperCoast QGIS Plugin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install_plugin.py                    # Package the plugin
  python install_plugin.py --install          # Package and install
  python install_plugin.py --install --force  # Force reinstall
  python install_plugin.py --uninstall        # Remove plugin from QGIS
  python install_plugin.py --info             # Show plugin information
        """,
    )

    parser.add_argument(
        "--install",
        "-i",
        action="store_true",
        help="Install the plugin to QGIS plugins directory",
    )

    parser.add_argument(
        "--uninstall", "-u", action="store_true", help="Uninstall the plugin from QGIS"
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force overwrite existing installation",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for the packaged ZIP file",
    )

    parser.add_argument(
        "--info", action="store_true", help="Show plugin and system information"
    )

    args = parser.parse_args()

    try:
        if args.info:
            show_info()
            return 0

        if args.uninstall:
            success = uninstall_plugin()
            return 0 if success else 1

        if args.install:
            install_plugin(force=args.force)
        else:
            zip_path = package_plugin(args.output)
            print(f"\nTo install the plugin:")
            print(f"  python {Path(__file__).name} --install")
            print(f"\nOr manually:")
            print(f"  1. Open QGIS")
            print(f"  2. Go to Plugins -> Manage and Install Plugins")
            print(f"  3. Click 'Install from ZIP'")
            print(f"  4. Select: {zip_path}")
            print(f"  5. Click 'Install Plugin'")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
