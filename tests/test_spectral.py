# SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for spectral library utilities."""

import numpy as np

from hypercoast.spectral import (
    SpectralLibrary,
    match_spectra,
    read_spectral_library,
    spectral_angle,
    write_spectral_library,
)


def test_spectral_angle_identical_spectra_is_zero():
    """SAM should be zero for identical finite spectra."""
    score = spectral_angle([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    assert np.isclose(score, 0.0)


def test_match_spectra_returns_nearest_label():
    """Spectral matching should rank the closest library spectrum first."""
    library = SpectralLibrary(
        wavelengths=np.array([500.0, 600.0, 700.0]),
        spectra={
            "water": np.array([0.02, 0.03, 0.01]),
            "vegetation": np.array([0.04, 0.08, 0.5]),
        },
    )

    matches = match_spectra([0.021, 0.031, 0.011], library)

    assert matches.iloc[0]["label"] == "water"


def test_read_write_spectral_library(tmp_path):
    """CSV spectral libraries should round-trip."""
    path = tmp_path / "library.csv"
    library = SpectralLibrary(
        wavelengths=np.array([500.0, 600.0]),
        spectra={"water": np.array([0.01, 0.02])},
    )

    write_spectral_library(library, path)
    loaded = read_spectral_library(path)

    np.testing.assert_array_equal(loaded.wavelengths, library.wavelengths)
    np.testing.assert_array_equal(loaded.spectra["water"], library.spectra["water"])
