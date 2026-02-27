# -*- coding: utf-8 -*-
"""
HyperCoast QGIS Plugin - Virtual Environment Manager

Creates and manages an isolated virtual environment for installing
the plugin's Python dependencies without modifying QGIS's built-in
Python environment.

Uses a standalone Python interpreter and uv for reliable, fast
installation across all platforms (including Windows where
sys.executable may point to qgis-bin.exe).

SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
SPDX-License-Identifier: MIT
"""

import importlib
import importlib.metadata
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

from qgis.core import QgsMessageLog, Qgis

CACHE_DIR = os.path.expanduser("~/.qgis_hypercoast")
VENV_DIR = os.path.join(CACHE_DIR, "venv")


def _log(message, level=Qgis.Info):
    """Log a message to the QGIS message log.

    Args:
        message: The message to log.
        level: The log level (Qgis.Info, Qgis.Warning, Qgis.Critical).
    """
    QgsMessageLog.logMessage(str(message), "HyperCoast", level=level)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def _get_clean_env_for_venv():
    """Create a clean environment dict for subprocess calls.

    Strips QGIS-specific variables that would interfere with the
    standalone Python or venv operations.

    Returns:
        A dict of environment variables.
    """
    env = os.environ.copy()

    vars_to_remove = [
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
    for var in vars_to_remove:
        env.pop(var, None)

    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _get_subprocess_kwargs():
    """Get platform-specific subprocess kwargs.

    On Windows, suppresses the console window that would otherwise pop up
    for each subprocess invocation.

    Returns:
        A dict of keyword arguments for subprocess.run / subprocess.Popen.
    """
    if platform.system() == "Windows":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def get_venv_python_path(venv_dir=None):
    """Get the path to the Python executable inside the venv.

    Args:
        venv_dir: Optional venv directory path. Defaults to VENV_DIR.

    Returns:
        The absolute path to the venv Python executable.
    """
    if venv_dir is None:
        venv_dir = VENV_DIR
    if platform.system() == "Windows":
        primary = os.path.join(venv_dir, "Scripts", "python.exe")
        if os.path.isfile(primary):
            return primary
        fallback = os.path.join(venv_dir, "Scripts", "python3.exe")
        if os.path.isfile(fallback):
            return fallback
        return primary  # Return expected path even if missing
    path = os.path.join(venv_dir, "bin", "python3")
    if os.path.isfile(path):
        return path
    return os.path.join(venv_dir, "bin", "python")


def get_venv_pip_path(venv_dir=None):
    """Get the path to pip inside the venv.

    Args:
        venv_dir: Optional venv directory path. Defaults to VENV_DIR.

    Returns:
        The absolute path to the venv pip executable.
    """
    if venv_dir is None:
        venv_dir = VENV_DIR
    if platform.system() == "Windows":
        return os.path.join(venv_dir, "Scripts", "pip.exe")
    return os.path.join(venv_dir, "bin", "pip")


def get_venv_site_packages(venv_dir=None):
    """Get the path to the site-packages directory inside the venv.

    Args:
        venv_dir: Optional venv directory path. Defaults to VENV_DIR.

    Returns:
        The path to the venv site-packages directory, or None if not found.
    """
    if venv_dir is None:
        venv_dir = VENV_DIR

    if platform.system() == "Windows":
        sp = os.path.join(venv_dir, "Lib", "site-packages")
        return sp if os.path.isdir(sp) else None

    # On Unix, detect the actual Python version directory in the venv
    lib_dir = os.path.join(venv_dir, "lib")
    if not os.path.isdir(lib_dir):
        return None
    for entry in sorted(os.listdir(lib_dir), reverse=True):
        if entry.startswith("python"):
            sp = os.path.join(lib_dir, entry, "site-packages")
            if os.path.isdir(sp):
                return sp
    return None


def venv_exists(venv_dir=None):
    """Check if the virtual environment exists.

    Args:
        venv_dir: Optional venv directory path. Defaults to VENV_DIR.

    Returns:
        True if the venv Python executable exists.
    """
    return os.path.exists(get_venv_python_path(venv_dir))


# ---------------------------------------------------------------------------
# System Python resolution
# ---------------------------------------------------------------------------


def _find_python_executable():
    """Find a working Python executable for venv creation.

    On QGIS Windows, sys.executable may point to qgis-bin.exe rather than
    a Python interpreter.  This function searches for the actual Python
    executable using multiple strategies.

    Returns:
        Path to a Python executable, or sys.executable as fallback.
    """
    if platform.system() != "Windows":
        return sys.executable

    # Strategy 1: Check if sys.executable is already Python
    exe_name = os.path.basename(sys.executable).lower()
    if exe_name in ("python.exe", "python3.exe"):
        return sys.executable

    # Strategy 2: Use sys._base_prefix to find the Python installation.
    base_prefix = getattr(sys, "_base_prefix", None) or sys.prefix
    python_in_prefix = os.path.join(base_prefix, "python.exe")
    if os.path.isfile(python_in_prefix):
        return python_in_prefix

    # Strategy 3: Look for python.exe next to sys.executable
    exe_dir = os.path.dirname(sys.executable)
    for name in ("python.exe", "python3.exe"):
        candidate = os.path.join(exe_dir, name)
        if os.path.isfile(candidate):
            return candidate

    # Strategy 4: Walk up from sys.executable to find apps/Python3x/python.exe
    parent = os.path.dirname(exe_dir)
    apps_dir = os.path.join(parent, "apps")
    if os.path.isdir(apps_dir):
        best_candidate = None
        best_version_num = -1
        for entry in os.listdir(apps_dir):
            lower_entry = entry.lower()
            if not lower_entry.startswith("python"):
                continue
            suffix = lower_entry.removeprefix("python")
            digits = "".join(ch for ch in suffix if ch.isdigit())
            if not digits:
                continue
            try:
                version_num = int(digits)
            except ValueError:
                continue
            candidate = os.path.join(apps_dir, entry, "python.exe")
            if os.path.isfile(candidate) and version_num > best_version_num:
                best_version_num = version_num
                best_candidate = candidate
        if best_candidate:
            return best_candidate

    # Strategy 5: Use shutil.which as last resort
    which_python = shutil.which("python")
    if which_python:
        return which_python

    # Fallback: return sys.executable (may fail, but preserves current behavior)
    return sys.executable


def _get_system_python():
    """Get the path to the Python executable for creating venvs.

    Uses the standalone Python downloaded by python_manager if available.
    On Windows, falls back to QGIS's bundled Python using multi-strategy
    detection (handles qgis-bin.exe, apps/Python3x/, etc.).

    Returns:
        The path to a usable Python executable.

    Raises:
        RuntimeError: If no usable Python is found.
    """
    from .python_manager import standalone_python_exists, get_standalone_python_path

    if standalone_python_exists():
        python_path = get_standalone_python_path()
        _log(f"Using standalone Python: {python_path}")
        return python_path

    # Fallback: find QGIS's bundled Python (critical on Windows where
    # sys.executable may be qgis-bin.exe)
    python_path = _find_python_executable()
    if python_path and os.path.isfile(python_path):
        _log(
            f"Standalone Python unavailable, using system Python: {python_path}",
            Qgis.Warning,
        )
        return python_path

    raise RuntimeError(
        "Python standalone not installed. "
        "Please click 'Install Dependencies' to download Python automatically."
    )


# ---------------------------------------------------------------------------
# Venv creation
# ---------------------------------------------------------------------------


def _cleanup_partial_venv(venv_dir):
    """Remove a partially-created venv directory.

    Args:
        venv_dir: The venv directory to remove.
    """
    if os.path.exists(venv_dir):
        try:
            shutil.rmtree(venv_dir, ignore_errors=True)
            _log(f"Cleaned up partial venv: {venv_dir}")
        except Exception:
            _log(f"Could not clean up partial venv: {venv_dir}", Qgis.Warning)


def create_venv(venv_dir=None, progress_callback=None):
    """Create a virtual environment using uv (preferred) or stdlib venv.

    When uv is available, uses ``uv venv`` which is faster and does not
    require pip to be bootstrapped inside the venv.  Falls back to
    ``python -m venv`` + ``ensurepip`` when uv is not available.

    Args:
        venv_dir: Optional venv directory path. Defaults to VENV_DIR.
        progress_callback: Function called with (percent, message).

    Returns:
        A tuple of (success: bool, message: str).
    """
    if venv_dir is None:
        venv_dir = VENV_DIR

    _log(f"Creating virtual environment at: {venv_dir}")

    if progress_callback:
        progress_callback(10, "Creating virtual environment...")

    system_python = _get_system_python()
    _log(f"Using Python: {system_python}")

    from .uv_manager import uv_exists, get_uv_path

    use_uv = uv_exists()

    if use_uv:
        uv_path = get_uv_path()
        cmd = [uv_path, "venv", "--python", system_python, venv_dir]
        _log("Creating venv with uv")
    else:
        cmd = [system_python, "-m", "venv", venv_dir]
        _log("Creating venv with stdlib venv")

    try:
        env = _get_clean_env_for_venv()
        kwargs = _get_subprocess_kwargs()

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            **kwargs,
        )

        if result.returncode == 0:
            _log("Virtual environment created successfully", Qgis.Success)

            # When using stdlib venv, ensure pip is available
            if not use_uv:
                pip_path = get_venv_pip_path(venv_dir)
                if not os.path.exists(pip_path):
                    _log("pip not found in venv, bootstrapping with ensurepip...")
                    python_in_venv = get_venv_python_path(venv_dir)
                    ensurepip_cmd = [
                        python_in_venv,
                        "-m",
                        "ensurepip",
                        "--upgrade",
                    ]
                    try:
                        ensurepip_result = subprocess.run(
                            ensurepip_cmd,
                            capture_output=True,
                            text=True,
                            timeout=120,
                            env=env,
                            **kwargs,
                        )
                        if ensurepip_result.returncode == 0:
                            _log("pip bootstrapped via ensurepip", Qgis.Success)
                        else:
                            err = ensurepip_result.stderr or ensurepip_result.stdout
                            _log(f"ensurepip failed: {err[:200]}", Qgis.Warning)
                            _cleanup_partial_venv(venv_dir)
                            return False, f"Failed to bootstrap pip: {err[:200]}"
                    except Exception as e:
                        _log(f"ensurepip exception: {e}", Qgis.Warning)
                        _cleanup_partial_venv(venv_dir)
                        return False, f"Failed to bootstrap pip: {str(e)[:200]}"

            if progress_callback:
                progress_callback(20, "Virtual environment created")
            return True, "Virtual environment created"
        else:
            error_msg = (
                result.stderr or result.stdout or f"Return code {result.returncode}"
            )
            _log(f"Failed to create venv: {error_msg}", Qgis.Critical)
            _cleanup_partial_venv(venv_dir)
            return False, f"Failed to create venv: {error_msg[:200]}"

    except subprocess.TimeoutExpired:
        _log("Virtual environment creation timed out", Qgis.Critical)
        _cleanup_partial_venv(venv_dir)
        return False, "Virtual environment creation timed out"
    except FileNotFoundError:
        _log(f"Python executable not found: {system_python}", Qgis.Critical)
        return False, f"Python not found: {system_python}"
    except Exception as e:
        _log(f"Exception during venv creation: {str(e)}", Qgis.Critical)
        _cleanup_partial_venv(venv_dir)
        return False, f"Error: {str(e)[:200]}"


# ---------------------------------------------------------------------------
# Requirements parsing (from requirements.txt)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Package version checking
# ---------------------------------------------------------------------------


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
        stem = entry[: -len(suffix)]
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
        env = _get_clean_env_for_venv()
        kwargs = _get_subprocess_kwargs()

        result = subprocess.run(
            [python_path, "-m", "pip", "show", package_name],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            **kwargs,
        )
        if result.returncode == 0 and result.stdout:
            for line in result.stdout.splitlines():
                if line.startswith("Version:"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Package installation
# ---------------------------------------------------------------------------


def _is_ssl_error(stderr):
    """Check if a pip error is SSL-related.

    Args:
        stderr: The stderr output from pip.

    Returns:
        True if the error is SSL-related.
    """
    ssl_markers = ["ssl", "certificate", "CERTIFICATE_VERIFY_FAILED"]
    lower = stderr.lower()
    return any(m.lower() in lower for m in ssl_markers)


def _is_network_error(stderr):
    """Check if a pip error is network-related.

    Args:
        stderr: The stderr output from pip.

    Returns:
        True if the error is network-related.
    """
    network_markers = [
        "ConnectionError",
        "connection refused",
        "connection reset",
        "timed out",
        "RemoteDisconnected",
        "NewConnectionError",
    ]
    return any(m.lower() in stderr.lower() for m in network_markers)


def _run_install_subprocess(
    cmd, env, kwargs, timeout, progress_callback=None, cancel_check=None
):
    """Run an install command with progress polling and cancellation support.

    Uses Popen with temporary files for stdout/stderr to avoid pipe buffer
    deadlocks on Windows (where pipe buffers are small and can fill up when
    installing many packages).

    Args:
        cmd: The command list to execute.
        env: Environment dict for the subprocess.
        kwargs: Additional subprocess kwargs.
        timeout: Timeout in seconds.
        progress_callback: Optional callback for progress updates (percent, msg).
        cancel_check: Optional function that returns True to cancel.

    Returns:
        A tuple of (returncode: int, stdout: str, stderr: str).
            returncode is -1 if cancelled, -2 if timed out.
    """
    # Use temporary files instead of pipes to avoid deadlock on Windows.
    # Pipe buffers are small (~4-64KB) and fill up when uv/pip produces
    # verbose output for many packages, causing the subprocess to block.
    stdout_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
    stderr_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            env=env,
            **kwargs,
        )
        start = time.time()
        poll_interval = 2  # seconds
        while True:
            try:
                proc.wait(timeout=poll_interval)
                break
            except subprocess.TimeoutExpired:
                pass

            # Check cancellation
            if cancel_check and cancel_check():
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return -1, "", "Installation cancelled by user."

            # Check overall timeout
            elapsed = time.time() - start
            if elapsed >= timeout:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return -2, "", f"Timed out after {timeout // 60} minutes."

            # Emit intermediate progress (25-85% range based on elapsed time)
            if progress_callback:
                fraction = min(elapsed / timeout, 1.0)
                percent = int(25 + fraction * 60)
                progress_callback(percent, "Installing packages...")

        # Read output from temporary files
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read()
        stderr = stderr_file.read()
        return proc.returncode, stdout, stderr

    finally:
        stdout_file.close()
        stderr_file.close()


def _run_install(
    cmd,
    env,
    kwargs,
    timeout=600,
    progress_callback=None,
    cancel_check=None,
    installer="pip",
):
    """Run a pip/uv install command with retry logic.

    Args:
        cmd: The command list to execute.
        env: Environment dict for the subprocess.
        kwargs: Additional subprocess kwargs.
        timeout: Timeout in seconds.
        progress_callback: Optional callback for progress updates (percent, msg).
        cancel_check: Optional function that returns True to cancel.
        installer: "pip" or "uv", used for retry flags and logging.

    Returns:
        A tuple of (success: bool, error_message: str).
    """
    try:
        returncode, stdout, stderr = _run_install_subprocess(
            cmd,
            env,
            kwargs,
            timeout,
            progress_callback,
            cancel_check,
        )

        if returncode == -1:
            return False, "Installation cancelled."
        if returncode == -2:
            return False, f"Installation timed out after {timeout // 60} minutes."
        if returncode == 0:
            return True, ""

        stderr = stderr or stdout or ""

        # Retry on SSL errors
        if _is_ssl_error(stderr):
            if installer == "uv":
                ssl_flags = [
                    "--allow-insecure-host",
                    "pypi.org",
                    "--allow-insecure-host",
                    "files.pythonhosted.org",
                ]
            else:
                ssl_flags = [
                    "--trusted-host",
                    "pypi.org",
                    "--trusted-host",
                    "files.pythonhosted.org",
                ]
            _log(
                f"SSL error installing dependencies via {installer}, "
                f"retrying with trusted hosts",
                Qgis.Warning,
            )
            retry_cmd = cmd + ssl_flags
            returncode, stdout, retry_stderr = _run_install_subprocess(
                retry_cmd,
                env,
                kwargs,
                timeout,
                progress_callback,
                cancel_check,
            )
            if returncode == -1:
                return False, "Installation cancelled."
            if returncode == 0:
                return True, ""
            stderr = retry_stderr or stderr

        # Retry on network errors with a delay
        if _is_network_error(stderr):
            _log(
                f"Network error installing dependencies via {installer}, "
                f"retrying in 5s...",
                Qgis.Warning,
            )
            time.sleep(5)
            returncode, stdout, retry_stderr = _run_install_subprocess(
                cmd,
                env,
                kwargs,
                timeout,
                progress_callback,
                cancel_check,
            )
            if returncode == -1:
                return False, "Installation cancelled."
            if returncode == 0:
                return True, ""
            stderr = retry_stderr or stderr

        # Classify the error for a user-friendly message
        return False, _classify_pip_error(stderr)

    except FileNotFoundError:
        if installer == "uv":
            return False, "uv executable not found."
        return False, "Python executable not found in virtual environment."
    except Exception as e:
        return False, f"Unexpected error installing dependencies: {str(e)}"


def _classify_pip_error(stderr):
    """Classify a pip/uv error into a user-friendly message.

    Args:
        stderr: The stderr output from pip/uv.

    Returns:
        A user-friendly error message string.
    """
    stderr_lower = stderr.lower()

    if "no matching distribution" in stderr_lower:
        return (
            "A required package was not found. "
            "Check your internet connection and try again."
        )
    if "permission" in stderr_lower or "denied" in stderr_lower:
        return (
            "Permission denied installing dependencies. "
            "Try running QGIS as administrator."
        )
    if "no space left" in stderr_lower:
        return "Not enough disk space to install dependencies."

    return f"Failed to install dependencies: {stderr[:300]}"


def install_dependencies(plugin_dir, progress_callback=None, cancel_check=None):
    """Install dependencies from requirements.txt into the virtual environment.

    Uses uv when available for significantly faster installation,
    falling back to pip otherwise.  Installs all packages in a single
    batch for efficiency.

    Args:
        plugin_dir: Path to the plugin directory containing requirements.txt.
        progress_callback: Function called with (percent, message).
        cancel_check: Function that returns True if operation should be cancelled.

    Returns:
        A tuple of (success: bool, message: str).
    """
    requirements = _parse_requirements(plugin_dir)
    if not requirements:
        return False, "No requirements found in requirements.txt"

    python_path = get_venv_python_path()
    if not os.path.exists(python_path):
        return False, "Virtual environment Python not found"

    env = _get_clean_env_for_venv()
    kwargs = _get_subprocess_kwargs()

    from .uv_manager import uv_exists, get_uv_path

    use_uv = uv_exists()
    if use_uv:
        uv_path = get_uv_path()
        _log("Installing dependencies with uv")
    else:
        _log("Installing dependencies with pip")

    # Build the full list of package specs for batch installation
    pkg_specs = [req["spec"] for req in requirements]
    pkg_names = [req["package"] for req in requirements]
    total = len(requirements)

    if cancel_check and cancel_check():
        return False, "Installation cancelled."

    # Scale timeout with number of packages (600s per package)
    timeout = 600 * total

    if progress_callback:
        progress_callback(20, f"Installing {', '.join(pkg_names[:3])}...")

    if use_uv:
        cmd = [
            uv_path,
            "pip",
            "install",
            "--python",
            python_path,
            "--upgrade",
        ] + pkg_specs
        success, error_msg = _run_install(
            cmd,
            env,
            kwargs,
            timeout=timeout,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            installer="uv",
        )
    else:
        cmd = [
            python_path,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--prefer-binary",
            "--disable-pip-version-check",
            "--no-warn-script-location",
        ] + pkg_specs
        success, error_msg = _run_install(
            cmd,
            env,
            kwargs,
            timeout=timeout,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            installer="pip",
        )

    if not success:
        return False, error_msg

    _log(f"Installed {total} package(s)", Qgis.Success)

    if progress_callback:
        progress_callback(90, "All packages installed")

    return True, f"Successfully installed {total} package(s)"


# ---------------------------------------------------------------------------
# Runtime integration
# ---------------------------------------------------------------------------


def ensure_venv_packages_available():
    """Make venv packages importable by adding site-packages to sys.path.

    This prepends the venv's site-packages to sys.path, giving it priority
    over QGIS's bundled packages. Also clears stale entries from sys.modules
    for packages that QGIS may bundle (like numpy).

    Returns:
        True if venv packages are available, False otherwise.
    """
    if not venv_exists():
        python_path = get_venv_python_path()
        _log(
            f"Venv does not exist: expected Python at {python_path}",
            Qgis.Warning,
        )
        return False

    site_packages = get_venv_site_packages()
    if site_packages is None:
        _log(f"Venv site-packages not found in: {VENV_DIR}", Qgis.Warning)
        return False

    if site_packages not in sys.path:
        # Insert at 0 so venv packages shadow QGIS-bundled numpy/pandas
        sys.path.insert(0, site_packages)
        _log(f"Added venv site-packages to sys.path: {site_packages}")

    # Clear stale module cache entries for key packages that QGIS may bundle.
    # This is especially important on Windows, where a preloaded QGIS package
    # can shadow the plugin venv package and break dataset backends.
    for module_name in [
        "numpy",
        "pandas",
        "xarray",
        "h5py",
        "h5netcdf",
        "netCDF4",
        "scipy",
        "rasterio",
        "rioxarray",
        "pyproj",
        "leafmap",
    ]:
        _refresh_module(module_name)

    # Invalidate import caches so Python re-scans directories
    importlib.invalidate_caches()

    # Runtime import diagnostics (helps debug "installed but not importable"
    # cases on Windows, e.g., h5py backend errors).
    _log(f"Runtime Python executable: {sys.executable}")
    if platform.system() == "Windows":
        _log(f"Windows runtime sys.path[0:3]: {sys.path[:3]}")

    for module_name in ["numpy", "xarray", "h5py", "h5netcdf", "hypercoast"]:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "unknown")
            mod_file = getattr(mod, "__file__", "<built-in>")
            _log(
                f"Runtime import OK: {module_name} {version} ({mod_file})",
                Qgis.Info,
            )
        except Exception as e:
            _log(
                f"Runtime import FAILED: {module_name} -> {e}",
                Qgis.Warning,
            )

    return True


def _import_check_in_venv(module_name):
    """Check whether a module can be imported by the venv's Python."""
    python_path = get_venv_python_path()
    if not os.path.isfile(python_path):
        return False, f"venv python not found: {python_path}"

    env = _get_clean_env_for_venv()
    kwargs = _get_subprocess_kwargs()

    code = (
        "import importlib,sys\n"
        "name=sys.argv[1]\n"
        "m=importlib.import_module(name)\n"
        "v=getattr(m,'__version__','unknown')\n"
        "f=getattr(m,'__file__','<built-in>')\n"
        "print(f'{name} {v} {f}')\n"
    )

    try:
        result = subprocess.run(
            [python_path, "-c", code, module_name],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            **kwargs,
        )
    except Exception as e:
        return False, str(e)

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        return False, detail

    return True, (result.stdout or "").strip()


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
        # If the module was loaded from outside our venv, remove it.
        # Use normcase() so the check is case-insensitive on Windows.
        if site_packages and os.path.normcase(site_packages) not in os.path.normcase(
            mod_file
        ):
            to_remove = [
                key
                for key in sys.modules
                if key == module_name or key.startswith(module_name + ".")
            ]
            for key in to_remove:
                del sys.modules[key]
            _log(f"Cleared stale module {module_name} from sys.modules")


# ---------------------------------------------------------------------------
# Status checking
# ---------------------------------------------------------------------------


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

    # Import smoke test in the venv runtime catches cases where metadata exists
    # but imports fail (e.g., h5py backend issues on Windows).
    for module_name in ["numpy", "xarray", "h5py", "h5netcdf", "hypercoast"]:
        ok, detail = _import_check_in_venv(module_name)
        if not ok:
            return False, f"Import check failed for {module_name}: {detail[:240]}"
        _log(f"Venv import OK: {detail}", Qgis.Info)

    return True, "Ready"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def create_venv_and_install(plugin_dir, progress_callback=None, cancel_check=None):
    """Complete installation: download Python + download uv + create venv + install.

    Progress breakdown:
        0-35%: Download Python standalone
        35-40%: Download uv package installer
        40-50%: Create virtual environment
        50-90%: Install packages
        90-100%: Verify

    Args:
        plugin_dir: Path to the plugin directory containing requirements.txt.
        progress_callback: Function called with (percent, message).
        cancel_check: Function that returns True if operation should be cancelled.

    Returns:
        A tuple of (success: bool, message: str).
    """
    from .python_manager import standalone_python_exists, download_python_standalone
    from .uv_manager import uv_exists as _uv_exists, download_uv

    start_time = time.time()

    # Step 1: Download Python standalone if needed (0-35%)
    if not standalone_python_exists():
        _log("Downloading Python standalone...")

        def python_progress(percent, msg):
            if progress_callback:
                progress_callback(int(percent * 0.35), msg)

        success, msg = download_python_standalone(
            progress_callback=python_progress,
            cancel_check=cancel_check,
        )

        if not success:
            # Fallback: use QGIS's bundled Python
            fallback = _find_python_executable()
            if fallback and os.path.isfile(fallback):
                _log(
                    f"Standalone download failed, using system Python: {fallback}",
                    Qgis.Warning,
                )
            else:
                return False, f"Failed to download Python: {msg}"

        if cancel_check and cancel_check():
            return False, "Installation cancelled"
    else:
        _log("Python standalone already installed")
        if progress_callback:
            progress_callback(35, "Python standalone ready")

    # Step 1b: Download uv package installer if needed (35-40%)
    if not _uv_exists():
        _log("Downloading uv package installer...")

        def uv_progress(percent, msg):
            if progress_callback:
                progress_callback(35 + int(percent * 0.05), msg)

        success, msg = download_uv(
            progress_callback=uv_progress,
            cancel_check=cancel_check,
        )

        if not success:
            # Non-fatal: fall back to pip
            _log(
                f"uv download failed ({msg}), will use pip instead",
                Qgis.Warning,
            )
        else:
            _log("uv package installer ready")

        if cancel_check and cancel_check():
            return False, "Installation cancelled"
    else:
        _log("uv already installed")
        if progress_callback:
            progress_callback(40, "uv ready")

    # Step 2: Create venv if needed (40-50%)
    if venv_exists():
        _log("Virtual environment already exists")
        if progress_callback:
            progress_callback(50, "Virtual environment ready")
    else:

        def venv_progress(percent, msg):
            if progress_callback:
                progress_callback(40 + int(percent * 0.10), msg)

        success, msg = create_venv(progress_callback=venv_progress)
        if not success:
            return False, msg

        if cancel_check and cancel_check():
            return False, "Installation cancelled"

    # Step 3: Install dependencies (50-90%)
    def deps_progress(percent, msg):
        if progress_callback:
            # Map 20-90 range from install_dependencies to 50-90
            mapped = 50 + int((percent - 20) * (40.0 / 70.0))
            progress_callback(min(mapped, 90), msg)

    success, msg = install_dependencies(
        plugin_dir,
        progress_callback=deps_progress,
        cancel_check=cancel_check,
    )

    if not success:
        return False, msg

    # Step 4: Verify (90-100%)
    if progress_callback:
        progress_callback(95, "Verifying installation...")

    is_ready, status_msg = get_venv_status(plugin_dir)
    if not is_ready:
        if progress_callback:
            progress_callback(100, f"Verification failed: {status_msg}")
        return False, f"Verification failed: {status_msg}"

    # Make packages available
    ensure_venv_packages_available()

    elapsed = time.time() - start_time
    if elapsed >= 60:
        minutes, seconds = divmod(int(elapsed), 60)
        elapsed_str = f"{minutes}:{seconds:02d}"
    else:
        elapsed_str = f"{elapsed:.1f}s"

    if progress_callback:
        progress_callback(100, f"All dependencies installed in {elapsed_str}")

    _log(f"All dependencies installed and verified in {elapsed_str}", Qgis.Success)
    return True, f"All dependencies installed successfully in {elapsed_str}"


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


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


def cleanup_old_venv_directories():
    """Remove old versioned venv directories (venv_py3.x) from previous layout.

    The plugin now uses a single ``venv/`` directory.  This helper removes
    leftover ``venv_py*`` directories created by earlier versions.

    Returns:
        A list of removed directory paths.
    """
    removed = []

    if not os.path.exists(CACHE_DIR):
        return removed

    try:
        for entry in os.listdir(CACHE_DIR):
            if entry.lower().startswith("venv_py"):
                old_path = os.path.join(CACHE_DIR, entry)
                if os.path.isdir(old_path):
                    try:
                        shutil.rmtree(old_path)
                        _log(f"Cleaned up old venv: {old_path}")
                        removed.append(old_path)
                    except Exception as e:
                        _log(
                            f"Failed to remove old venv {old_path}: {e}",
                            Qgis.Warning,
                        )
    except Exception as e:
        _log(f"Error scanning for old venvs: {e}", Qgis.Warning)

    return removed
