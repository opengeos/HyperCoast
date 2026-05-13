"""Shared pytest fixtures.

Stubs the ``qgis`` package so the plugin's modules can be imported without a
running QGIS instance. The stub reproduces the real ``qgis.PyQt`` shim
behavior on Qt6: it re-exports ``QAction``, ``QActionGroup`` and ``QShortcut``
from ``PyQt6.QtGui`` under ``qgis.PyQt.QtWidgets`` (they moved out of
``QtWidgets`` in Qt6).
"""

import sys
import types
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt6")

import PyQt6.QtCore  # noqa: E402
import PyQt6.QtGui  # noqa: E402
import PyQt6.QtNetwork  # noqa: E402
import PyQt6.QtWidgets  # noqa: E402


def _install_qgis_stub() -> None:
    qgis = types.ModuleType("qgis")
    qgis.__path__ = []
    sys.modules["qgis"] = qgis

    qgis_pyqt = types.ModuleType("qgis.PyQt")
    qgis_pyqt.__path__ = []
    sys.modules["qgis.PyQt"] = qgis_pyqt
    qgis.PyQt = qgis_pyqt

    pyqt_submodules = {
        "QtCore": PyQt6.QtCore,
        "QtGui": PyQt6.QtGui,
        "QtNetwork": PyQt6.QtNetwork,
        "QtWidgets": PyQt6.QtWidgets,
    }
    for name, real in pyqt_submodules.items():
        alias = types.ModuleType(f"qgis.PyQt.{name}")
        for attr in dir(real):
            if not attr.startswith("_"):
                setattr(alias, attr, getattr(real, attr))
        sys.modules[f"qgis.PyQt.{name}"] = alias
        setattr(qgis_pyqt, name, alias)

    # Qt6: QAction, QActionGroup, and QShortcut live in QtGui. The real
    # qgis.PyQt.QtWidgets shim re-exports them, so mirror that here.
    qtwidgets_alias = sys.modules["qgis.PyQt.QtWidgets"]
    for attr in ("QAction", "QActionGroup", "QShortcut"):
        setattr(qtwidgets_alias, attr, getattr(PyQt6.QtGui, attr))

    for submodule in ("QtSvg", "QtWebEngineWidgets"):
        alias = MagicMock()
        sys.modules[f"qgis.PyQt.{submodule}"] = alias
        setattr(qgis_pyqt, submodule, alias)

    for name in ("core", "gui", "utils"):
        stub = MagicMock()
        stub.__spec__ = None
        sys.modules[f"qgis.{name}"] = stub
        setattr(qgis, name, stub)

    core = sys.modules["qgis.core"]

    class QgsProcessingException(Exception):
        """Minimal Processing exception stub."""

    class QgsProcessingAlgorithm:
        """Minimal Processing algorithm stub."""

        def __init__(self) -> None:
            """Initialize stored parameters."""
            self.parameters = []

        def addParameter(self, parameter) -> None:
            """Store an algorithm parameter.

            Args:
                parameter: Parameter object.
            """
            self.parameters.append(parameter)

        def parameterAsFile(self, parameters, name, context):
            """Return a file parameter value."""
            return parameters.get(name, "")

        def parameterAsEnum(self, parameters, name, context):
            """Return an enum parameter value."""
            return int(parameters.get(name, 0))

        def parameterAsString(self, parameters, name, context):
            """Return a string parameter value."""
            return parameters.get(name, "")

        def parameterAsDouble(self, parameters, name, context):
            """Return a double parameter value."""
            return float(parameters.get(name, 0.0))

        def parameterAsInt(self, parameters, name, context):
            """Return an integer parameter value."""
            return int(parameters.get(name, 0))

        def parameterAsOutputLayer(self, parameters, name, context):
            """Return an output layer parameter value."""
            return parameters.get(name, "")

    class QgsProcessingProvider:
        """Minimal Processing provider stub."""

        def __init__(self) -> None:
            """Initialize stored algorithms."""
            self.algorithms = []

        def addAlgorithm(self, algorithm) -> None:
            """Store a provider algorithm.

            Args:
                algorithm: Algorithm instance.
            """
            self.algorithms.append(algorithm)

    class _Parameter:
        """Minimal Processing parameter stub."""

        def __init__(self, *args, **kwargs) -> None:
            """Store parameter construction arguments."""
            self.args = args
            self.kwargs = kwargs

    class QgsProcessingParameterFile(_Parameter):
        """Minimal file parameter stub."""

        File = 0

        class Behavior:
            """Minimal file behavior enum."""

            File = 0

    class QgsProcessingParameterNumber(_Parameter):
        """Minimal number parameter stub."""

        Double = 0
        Integer = 1

        class Type:
            """Minimal number type enum."""

            Double = 0
            Integer = 1

    core.QgsProcessingException = QgsProcessingException
    core.QgsProcessingAlgorithm = QgsProcessingAlgorithm
    core.QgsProcessingProvider = QgsProcessingProvider
    core.QgsProcessingParameterEnum = _Parameter
    core.QgsProcessingParameterFile = QgsProcessingParameterFile
    core.QgsProcessingParameterNumber = QgsProcessingParameterNumber
    core.QgsProcessingParameterRasterDestination = _Parameter
    core.QgsProcessingParameterString = _Parameter


_install_qgis_stub()
