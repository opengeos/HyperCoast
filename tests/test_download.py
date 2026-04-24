# SPDX-FileCopyrightText: 2026 Qiusheng Wu <giswqs@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for the NASA download helpers (regression tests for issue #244)."""

import os
import sys
import types
import unittest
from unittest import mock

import hypercoast
from hypercoast import common


class TestDownloadNasaDataReturnValue(unittest.TestCase):
    """Verify the NASA download helpers return the list of downloaded paths."""

    def setUp(self):
        """Ensure USE_MKDOCS is not set for the propagation tests."""
        self._saved_use_mkdocs = os.environ.pop("USE_MKDOCS", None)

    def tearDown(self):
        """Restore USE_MKDOCS to whatever it was before the test."""
        if self._saved_use_mkdocs is not None:
            os.environ["USE_MKDOCS"] = self._saved_use_mkdocs
        else:
            os.environ.pop("USE_MKDOCS", None)

    def _stub_earthaccess(self, return_value):
        """Install a stub ``earthaccess`` module with a recording ``download``.

        Args:
            return_value: The value the stubbed ``earthaccess.download`` should
                return.

        Returns:
            mock.MagicMock: The mock standing in for ``earthaccess.download``.
        """
        stub = types.ModuleType("earthaccess")
        stub.download = mock.MagicMock(return_value=return_value)
        patcher = mock.patch.dict(sys.modules, {"earthaccess": stub})
        patcher.start()
        self.addCleanup(patcher.stop)
        return stub.download

    def test_download_nasa_data_returns_paths(self):
        expected = ["/tmp/a.nc", "/tmp/b.nc"]
        download = self._stub_earthaccess(expected)

        result = common.download_nasa_data(
            granules=[{"id": 1}], out_dir="/tmp", provider="POCLOUD", threads=2
        )

        self.assertEqual(result, expected)
        download.assert_called_once_with(
            [{"id": 1}], local_path="/tmp", provider="POCLOUD", threads=2
        )

    def test_download_pace_propagates_paths(self):
        expected = ["/tmp/pace.nc"]
        self._stub_earthaccess(expected)

        result = hypercoast.download_pace(granules=[{"id": 1}], out_dir="/tmp")

        self.assertEqual(result, expected)

    def test_download_emit_propagates_paths(self):
        expected = ["/tmp/emit.nc"]
        self._stub_earthaccess(expected)

        result = hypercoast.download_emit(granules=[{"id": 1}], out_dir="/tmp")

        self.assertEqual(result, expected)

    def test_download_ecostress_propagates_paths(self):
        expected = ["/tmp/ecostress.nc"]
        self._stub_earthaccess(expected)

        result = hypercoast.download_ecostress(granules=[{"id": 1}], out_dir="/tmp")

        self.assertEqual(result, expected)

    def test_use_mkdocs_short_circuits_without_earthaccess(self):
        """USE_MKDOCS must return [] before importing earthaccess."""
        os.environ["USE_MKDOCS"] = "1"
        self.addCleanup(os.environ.pop, "USE_MKDOCS", None)

        patcher = mock.patch.dict(sys.modules, {"earthaccess": None})
        patcher.start()
        self.addCleanup(patcher.stop)

        result = common.download_nasa_data(granules=[{"id": 1}])

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
