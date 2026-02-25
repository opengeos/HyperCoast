# -*- coding: utf-8 -*-
"""
HyperCoast QGIS Plugin - Virtual Environment Manager

Manages an isolated virtual environment for plugin dependencies,
keeping QGIS's built-in Python environment clean.

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

import importlib
import importlib.metadata
import os
import re
import shutil
import subprocess
import sys

CACHE_DIR = os.path.expanduser("~/.qgis_hypercoast")
PYTHON_VERSION = f"py{sys.version_info.major}.{sys.version_info.minor}"
VENV_DIR = os.path.join(CACHE_DIR, f"venv_{PYTHON_VERSION}")

# Environment variables to strip when running subprocess commands
# to prevent QGIS from polluting the virtual environment
_STRIP_ENV_VARS = [
    "PYTHONPATH",
    "PYTHONHOME",
    "VIRTUAL_ENV",
    "QGIS_PREFIX_PATH",
    "QGIS_PLUGINPATH",
    "PROJ_DATA",
    "PROJ_LIB",
    "GDAL_DATA",
    "GDAL_DRIVER_PATH",
]


def _log(message, level=None):
    """Log a message to QGIS message log if available."""
    try:
        from qgis.core import QgsMessageLog, Qgis

        if level is None:
            level = Qgis.Info
        QgsMessageLog.logMessage(message, "HyperCoast", level)
    except Exception:
        pass


def _get_clean_env():
    """Get a clean environment for subprocess calls.

    Strips QGIS-specific environment variables that would pollute
    the virtual environment.

    Returns:
        dict: Clean environment dictionary.
    """
    env = os.environ.copy()
    for var in _STRIP_ENV_VARS:
        env.pop(var, None)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def get_venv_python_path(venv_dir=None):
    """Get the path to the Python binary in the virtual environment.

    Args:
        venv_dir: Path to the virtual environment directory.
            Defaults to VENV_DIR.

    Returns:
        str: Path to the venv Python binary.
    """
    if venv_dir is None:
        venv_dir = VENV_DIR
    if sys.platform == "win32":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python3")


def get_venv_site_packages(venv_dir=None):
    """Get the path to site-packages in the virtual environment.

    Args:
        venv_dir: Path to the virtual environment directory.
            Defaults to VENV_DIR.

    Returns:
        str: Path to the venv site-packages directory, or None if not found.
    """
    if venv_dir is None:
        venv_dir = VENV_DIR

    if sys.platform == "win32":
        sp = os.path.join(venv_dir, "Lib", "site-packages")
        if os.path.isdir(sp):
            return sp
        return sp  # Return expected path even if it doesn't exist yet

    # Unix: auto-detect python version in venv
    lib_dir = os.path.join(venv_dir, "lib")
    if os.path.isdir(lib_dir):
        for entry in sorted(os.listdir(lib_dir)):
            if entry.startswith("python") and os.path.isdir(
                os.path.join(lib_dir, entry)
            ):
                sp = os.path.join(lib_dir, entry, "site-packages")
                if os.path.isdir(sp):
                    return sp

    # Fallback to expected path
    return os.path.join(
        venv_dir,
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages",
    )


def venv_exists():
    """Check if the virtual environment exists and has a Python binary.

    Returns:
        bool: True if the venv Python binary exists.
    """
    python_path = get_venv_python_path()
    return os.path.isfile(python_path)


def create_venv(progress_callback=None):
    """Create a virtual environment using QGIS's Python.

    Args:
        progress_callback: Optional callable(percent, message) for progress.

    Returns:
        tuple: (success: bool, message: str)
    """
    if progress_callback:
        progress_callback(0, "Creating virtual environment...")

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # If venv already exists, skip creation
    if venv_exists():
        if progress_callback:
            progress_callback(100, "Virtual environment already exists")
        return True, "Virtual environment already exists"

    env = _get_clean_env()

    # Build the venv creation command
    cmd = [sys.executable, "-m", "venv", VENV_DIR]

    try:
        kwargs = {
            "capture_output": True,
            "text": True,
            "env": env,
            "timeout": 120,
        }

        # Windows: hide console window
        if sys.platform == "win32":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 0  # SW_HIDE
            kwargs["startupinfo"] = si
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        if progress_callback:
            progress_callback(30, "Running venv creation...")

        result = subprocess.run(cmd, **kwargs)

        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            _log(f"Venv creation failed: {error_msg}")
            # Clean up partial venv
            if os.path.exists(VENV_DIR):
                shutil.rmtree(VENV_DIR, ignore_errors=True)
            return False, f"Failed to create virtual environment: {error_msg}"

        if progress_callback:
            progress_callback(60, "Verifying virtual environment...")

        # Verify the venv was created successfully
        if not venv_exists():
            return False, "Virtual environment was created but Python binary not found"

        # Ensure pip is available
        if progress_callback:
            progress_callback(80, "Ensuring pip is available...")

        pip_cmd = [get_venv_python_path(), "-m", "ensurepip", "--upgrade"]
        pip_result = subprocess.run(pip_cmd, **kwargs)

        if pip_result.returncode != 0:
            _log(f"ensurepip warning: {pip_result.stderr}")
            # Not fatal - pip might already be available

        if progress_callback:
            progress_callback(100, "Virtual environment created successfully")

        _log("Virtual environment created successfully")
        return True, "Virtual environment created successfully"

    except subprocess.TimeoutExpired:
        if os.path.exists(VENV_DIR):
            shutil.rmtree(VENV_DIR, ignore_errors=True)
        return False, "Virtual environment creation timed out"
    except Exception as e:
        if os.path.exists(VENV_DIR):
            shutil.rmtree(VENV_DIR, ignore_errors=True)
        return False, f"Error creating virtual environment: {str(e)}"


def _parse_requirements(plugin_dir):
    """Parse requirements.txt from the plugin directory.

    Args:
        plugin_dir: Path to the plugin directory containing requirements.txt.

    Returns:
        list: List of dicts with keys: package, operator, version, spec.
    """
    req_path = os.path.join(plugin_dir, "requirements.txt")
    requirements = []

    if not os.path.isfile(req_path):
        _log(f"requirements.txt not found at {req_path}")
        return requirements

    with open(req_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Parse "package>=version" or "package==version" or just "package"
            match = re.match(r"^([a-zA-Z0-9_-]+)\s*(>=|==|<=|!=|>|<)?\s*(.+)?$", line)
            if match:
                requirements.append(
                    {
                        "package": match.group(1),
                        "operator": match.group(2) or ">=",
                        "version": match.group(3) or "0",
                        "spec": line,
                    }
                )

    return requirements


def _parse_version(v):
    """Parse a version string into a tuple of integers.

    Args:
        v: Version string like "1.2.3".

    Returns:
        tuple: Tuple of integers, e.g. (1, 2, 3).
    """
    parts = re.findall(r"\d+", v)
    return tuple(int(p) for p in parts)


def _version_satisfies(installed, operator, required):
    """Check if an installed version satisfies a version requirement.

    Args:
        installed: Installed version string.
        operator: Comparison operator (>=, ==, >, etc.).
        required: Required version string.

    Returns:
        bool: True if the requirement is satisfied.
    """
    try:
        inst = _parse_version(installed)
        req = _parse_version(required)
        if operator == ">=":
            return inst >= req
        elif operator == "==":
            return inst == req
        elif operator == ">":
            return inst > req
        elif operator == "<=":
            return inst <= req
        elif operator == "<":
            return inst < req
        elif operator == "!=":
            return inst != req
    except Exception:
        pass
    return True


def check_packages(plugin_dir):
    """Check which packages from requirements.txt are installed in the venv.

    Args:
        plugin_dir: Path to the plugin directory containing requirements.txt.

    Returns:
        list: List of dicts with keys: package, required, installed, status, spec.
            status is one of: "ok", "missing", "outdated".
    """
    requirements = _parse_requirements(plugin_dir)
    site_packages = get_venv_site_packages()
    results = []

    for req in requirements:
        pkg_name = req["package"]
        installed_ver = None
        status = "missing"

        # Try to find the package in venv's site-packages
        if site_packages and os.path.isdir(site_packages):
            # Check via metadata in the venv
            try:
                installed_ver = _get_package_version_from_venv(pkg_name, site_packages)
            except Exception:
                installed_ver = None

        if installed_ver is not None:
            if _version_satisfies(installed_ver, req["operator"], req["version"]):
                status = "ok"
            else:
                status = "outdated"

        results.append(
            {
                "package": pkg_name,
                "required": f"{req['operator']}{req['version']}",
                "installed": installed_ver,
                "status": status,
                "spec": req["spec"],
            }
        )

    return results


def _get_package_version_from_venv(package_name, site_packages):
    """Get a package's version from the venv's site-packages.

    Checks both .dist-info and .egg-info directories, then falls back
    to running ``pip show`` in the venv as a last resort.

    Args:
        package_name: Name of the package.
        site_packages: Path to the site-packages directory.

    Returns:
        str: Version string, or None if not found.
    """
    # Normalize package name for metadata lookup
    normalized = re.sub(r"[-_.]+", "_", package_name).lower()

    if not os.path.isdir(site_packages):
        return None

    for entry in os.listdir(site_packages):
        # Handle both .dist-info and .egg-info directories
        if entry.endswith(".dist-info"):
            suffix = ".dist-info"
        elif entry.endswith(".egg-info"):
            suffix = ".egg-info"
        else:
            continue

        # Strip suffix, then split name from version
        # Format: {name}-{version}.dist-info (e.g. h5netcdf-1.3.0.dist-info)
        stem = entry[: -len(suffix)]
        # Find first dash followed by a digit to separate name from version
        match = re.match(r"^(.+?)-(\d.*)$", stem)
        if not match:
            continue
        entry_name = match.group(1)
        entry_version = match.group(2)
        entry_normalized = re.sub(r"[-_.]+", "_", entry_name).lower()
        if entry_normalized == normalized:
            # Read authoritative version from METADATA file (.dist-info)
            metadata_path = os.path.join(site_packages, entry, "METADATA")
            if os.path.isfile(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    for mline in f:
                        if mline.startswith("Version:"):
                            return mline.split(":", 1)[1].strip()
            # Read from PKG-INFO (.egg-info)
            pkg_info_path = os.path.join(site_packages, entry, "PKG-INFO")
            if os.path.isfile(pkg_info_path):
                with open(pkg_info_path, "r", encoding="utf-8") as f:
                    for mline in f:
                        if mline.startswith("Version:"):
                            return mline.split(":", 1)[1].strip()
            # Fall back to version from directory name
            return entry_version

    # Fallback: use pip show via subprocess
    return _pip_show_version(package_name)


def _pip_show_version(package_name):
    """Get a package version by running pip show in the venv.

    Args:
        package_name: Name of the package.

    Returns:
        str: Version string, or None if not found.
    """
    python_path = get_venv_python_path()
    if not os.path.isfile(python_path):
        return None

    try:
        env = _get_clean_env()
        kwargs = {
            "capture_output": True,
            "text": True,
            "env": env,
            "timeout": 30,
        }
        if sys.platform == "win32":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 0
            kwargs["startupinfo"] = si
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        result = subprocess.run(
            [python_path, "-m", "pip", "show", package_name], **kwargs
        )
        if result.returncode == 0 and result.stdout:
            for line in result.stdout.splitlines():
                if line.startswith("Version:"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass

    return None


def install_dependencies(plugin_dir, progress_callback=None, cancel_check=None):
    """Install dependencies from requirements.txt into the virtual environment.

    Args:
        plugin_dir: Path to the plugin directory containing requirements.txt.
        progress_callback: Optional callable(percent, message) for progress.
        cancel_check: Optional callable() that returns True to cancel.

    Returns:
        tuple: (success: bool, message: str)
    """
    requirements = _parse_requirements(plugin_dir)
    if not requirements:
        return False, "No requirements found in requirements.txt"

    python_path = get_venv_python_path()
    if not os.path.isfile(python_path):
        return False, "Virtual environment Python not found"

    env = _get_clean_env()
    total = len(requirements)
    installed = []
    failed = []

    for i, req in enumerate(requirements):
        if cancel_check and cancel_check():
            return False, "Installation cancelled by user"

        pkg_name = req["package"]
        pkg_spec = req["spec"]
        percent = int((i / total) * 100)

        if progress_callback:
            progress_callback(percent, f"Installing {pkg_name} ({i + 1}/{total})...")

        _log(f"Installing {pkg_spec}...")

        pip_args = [
            python_path,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--prefer-binary",
            "--no-warn-script-location",
            "--disable-pip-version-check",
            pkg_spec,
        ]

        try:
            kwargs = {
                "capture_output": True,
                "text": True,
                "env": env,
                "timeout": 600,
            }

            # Windows: hide console window
            if sys.platform == "win32":
                si = subprocess.STARTUPINFO()
                si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                si.wShowWindow = 0
                kwargs["startupinfo"] = si
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            result = subprocess.run(pip_args, **kwargs)

            if result.returncode == 0:
                installed.append(pkg_name)
                _log(f"Successfully installed {pkg_spec}")
            else:
                error = result.stderr.strip() if result.stderr else "Unknown error"
                failed.append((pkg_name, error))
                _log(f"Failed to install {pkg_spec}: {error}")

        except subprocess.TimeoutExpired:
            failed.append((pkg_name, "Installation timed out"))
            _log(f"Timeout installing {pkg_spec}")
        except Exception as e:
            failed.append((pkg_name, str(e)))
            _log(f"Error installing {pkg_spec}: {e}")

    if progress_callback:
        progress_callback(100, "Installation complete")

    if failed:
        failed_names = ", ".join(f"{name}: {err[:50]}" for name, err in failed)
        msg = f"Installed {len(installed)}/{total} packages. Failed: {failed_names}"
        _log(msg)
        if not installed:
            return False, msg
        return True, msg

    return True, f"All {total} packages installed successfully"


def get_venv_status(plugin_dir):
    """Check the overall status of the virtual environment.

    Args:
        plugin_dir: Path to the plugin directory containing requirements.txt.

    Returns:
        tuple: (is_ready: bool, message: str)
    """
    if not venv_exists():
        return False, "Dependencies not installed"

    site_packages = get_venv_site_packages()
    if not site_packages or not os.path.isdir(site_packages):
        return False, "Virtual environment incomplete"

    # Quick check: verify key packages exist
    requirements = _parse_requirements(plugin_dir)
    for req in requirements:
        pkg_name = req["package"]
        version = _get_package_version_from_venv(pkg_name, site_packages)
        if version is None:
            return False, f"Missing package: {pkg_name}"

    return True, "Ready"


def ensure_venv_packages_available():
    """Add the venv's site-packages to sys.path so packages can be imported.

    This prepends the venv's site-packages to sys.path, giving it priority
    over QGIS's bundled packages. Also clears stale entries from sys.modules
    for packages that QGIS may bundle (like numpy).
    """
    site_packages = get_venv_site_packages()
    if not site_packages or not os.path.isdir(site_packages):
        _log("Venv site-packages not found, cannot add to sys.path")
        return False

    if site_packages not in sys.path:
        sys.path.insert(0, site_packages)
        _log(f"Added venv site-packages to sys.path: {site_packages}")

    # Clear stale module cache entries for key packages that QGIS may bundle
    # This ensures we import from the venv, not from QGIS's bundled versions
    _refresh_module("numpy")
    _refresh_module("pandas")

    # Invalidate import caches so Python re-scans directories
    importlib.invalidate_caches()

    return True


def _refresh_module(module_name):
    """Remove a module from sys.modules if it was loaded from outside the venv.

    This forces Python to re-import the module from the venv's site-packages
    on the next import.

    Args:
        module_name: Name of the module to refresh.
    """
    site_packages = get_venv_site_packages()
    if module_name in sys.modules:
        mod = sys.modules[module_name]
        mod_file = getattr(mod, "__file__", "") or ""
        # If the module was loaded from outside our venv, remove it
        if site_packages and site_packages not in mod_file:
            # Remove the module and any submodules
            to_remove = [
                key
                for key in sys.modules
                if key == module_name or key.startswith(module_name + ".")
            ]
            for key in to_remove:
                del sys.modules[key]
            _log(f"Cleared stale module {module_name} from sys.modules")


def create_venv_and_install(plugin_dir, progress_callback=None, cancel_check=None):
    """Create the virtual environment and install all dependencies.

    Orchestrates the full installation flow:
    - Create venv (0-15%)
    - Install dependencies (15-95%)
    - Verify (95-100%)

    Args:
        plugin_dir: Path to the plugin directory containing requirements.txt.
        progress_callback: Optional callable(percent, message) for progress.
        cancel_check: Optional callable() that returns True to cancel.

    Returns:
        tuple: (success: bool, message: str)
    """

    def _venv_progress(percent, message):
        """Map venv creation progress (0-100) to overall (0-15)."""
        if progress_callback:
            progress_callback(int(percent * 0.15), message)

    def _install_progress(percent, message):
        """Map install progress (0-100) to overall (15-95)."""
        if progress_callback:
            progress_callback(15 + int(percent * 0.80), message)

    # Step 1: Create virtual environment (0-15%)
    if cancel_check and cancel_check():
        return False, "Cancelled"

    success, message = create_venv(_venv_progress)
    if not success:
        return False, f"Venv creation failed: {message}"

    # Step 2: Install dependencies (15-95%)
    if cancel_check and cancel_check():
        return False, "Cancelled"

    success, message = install_dependencies(plugin_dir, _install_progress, cancel_check)
    if not success:
        return False, f"Installation failed: {message}"

    # Step 3: Verify (95-100%)
    if progress_callback:
        progress_callback(95, "Verifying installation...")

    is_ready, status_msg = get_venv_status(plugin_dir)
    if not is_ready:
        if progress_callback:
            progress_callback(100, f"Verification failed: {status_msg}")
        return False, f"Verification failed: {status_msg}"

    # Make packages available
    ensure_venv_packages_available()

    if progress_callback:
        progress_callback(100, "Installation complete!")

    _log("All dependencies installed and verified successfully")
    return True, message


def remove_venv():
    """Remove the virtual environment directory.

    Returns:
        tuple: (success: bool, message: str)
    """
    if not os.path.exists(VENV_DIR):
        return True, "No virtual environment to remove"

    try:
        shutil.rmtree(VENV_DIR)
        _log("Virtual environment removed")
        return True, "Virtual environment removed successfully"
    except Exception as e:
        return False, f"Failed to remove virtual environment: {str(e)}"
