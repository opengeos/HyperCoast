# SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Spectral library and matching utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class SpectralLibrary:
    """Represent a named spectral library.

    Args:
        wavelengths: Wavelength values.
        spectra: Mapping of class names to spectral vectors.
    """

    wavelengths: np.ndarray
    spectra: Dict[str, np.ndarray]

    def to_dataframe(self) -> pd.DataFrame:
        """Return the library as a DataFrame.

        Returns:
            pd.DataFrame: Spectral library table.
        """
        data = {"wavelength": self.wavelengths}
        data.update(self.spectra)
        return pd.DataFrame(data)


def read_spectral_library(path: str | Path) -> SpectralLibrary:
    """Read a CSV spectral library.

    Args:
        path: CSV path with a ``wavelength`` column and one column per class.

    Returns:
        SpectralLibrary: Loaded spectral library.
    """
    df = pd.read_csv(path)
    if "wavelength" not in df.columns:
        raise ValueError("Spectral library CSV must contain a wavelength column.")
    spectra = {
        column: df[column].to_numpy(dtype=float)
        for column in df.columns
        if column != "wavelength"
    }
    return SpectralLibrary(df["wavelength"].to_numpy(dtype=float), spectra)


def write_spectral_library(library: SpectralLibrary, path: str | Path) -> str:
    """Write a spectral library to CSV.

    Args:
        library: Spectral library.
        path: Output CSV path.

    Returns:
        str: Output path.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    library.to_dataframe().to_csv(output, index=False)
    return str(output)


def spectral_angle(spectrum: Sequence[float], reference: Sequence[float]) -> float:
    """Calculate spectral angle mapper distance in radians.

    Args:
        spectrum: Input spectral vector.
        reference: Reference spectral vector.

    Returns:
        float: Spectral angle in radians.
    """
    a, b = _paired_arrays(spectrum, reference)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return float("nan")
    cosine = np.clip(float(np.dot(a, b) / denominator), -1.0, 1.0)
    return float(np.arccos(cosine))


def spectral_information_divergence(
    spectrum: Sequence[float],
    reference: Sequence[float],
    eps: float = 1e-12,
) -> float:
    """Calculate spectral information divergence.

    Args:
        spectrum: Input spectral vector.
        reference: Reference spectral vector.
        eps: Small positive value for numerical stability.

    Returns:
        float: Spectral information divergence.
    """
    a, b = _paired_arrays(spectrum, reference)
    a = np.maximum(a, 0) + eps
    b = np.maximum(b, 0) + eps
    p = a / np.sum(a)
    q = b / np.sum(b)
    return float(np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p)))


def continuum_remove(
    spectrum: Sequence[float],
    wavelengths: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Apply simple continuum removal with a linear hull approximation.

    Args:
        spectrum: Input spectral vector.
        wavelengths: Optional wavelength values.

    Returns:
        np.ndarray: Continuum-removed spectrum.
    """
    y = np.asarray(spectrum, dtype=float)
    if wavelengths is None:
        x = np.arange(y.size, dtype=float)
    else:
        x = np.asarray(wavelengths)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return np.full_like(y, np.nan, dtype=float)
    continuum = np.interp(x, x[finite][[0, -1]], y[finite][[0, -1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        return y / continuum


def derivative_spectrum(
    spectrum: Sequence[float],
    wavelengths: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Calculate a first derivative spectrum.

    Args:
        spectrum: Input spectral vector.
        wavelengths: Optional wavelength values.

    Returns:
        np.ndarray: First derivative values.
    """
    y = np.asarray(spectrum, dtype=float)
    if wavelengths is None:
        return np.gradient(y)
    return np.gradient(y, np.asarray(wavelengths, dtype=float))


def match_spectra(
    spectra: Sequence[float] | Iterable[Sequence[float]],
    library: SpectralLibrary,
    method: str = "sam",
    top_k: int = 1,
) -> pd.DataFrame:
    """Match one or more spectra against a spectral library.

    Args:
        spectra: One spectrum or an iterable of spectra.
        library: Spectral library.
        method: Matching method, ``"sam"`` or ``"sid"``.
        top_k: Number of matches to return per input spectrum.

    Returns:
        pd.DataFrame: Match table with ``spectrum``, ``label``, and ``score``.
    """
    matrix = np.asarray(spectra, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[np.newaxis, :]
    if method.lower() == "sam":
        scorer = spectral_angle
    else:
        scorer = spectral_information_divergence
    rows: List[Dict[str, float | int | str]] = []
    for index, spectrum in enumerate(matrix):
        scores = [
            (label, scorer(spectrum, reference))
            for label, reference in library.spectra.items()
        ]
        scores.sort(key=lambda item: item[1])
        for label, score in scores[:top_k]:
            rows.append({"spectrum": index, "label": label, "score": score})
    return pd.DataFrame(rows)


def _paired_arrays(
    spectrum: Sequence[float],
    reference: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite paired arrays."""
    a = np.asarray(spectrum, dtype=float)
    b = np.asarray(reference, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Spectrum and reference must have the same shape.")
    finite = np.isfinite(a) & np.isfinite(b)
    return a[finite], b[finite]


__all__ = [
    "SpectralLibrary",
    "continuum_remove",
    "derivative_spectrum",
    "match_spectra",
    "read_spectral_library",
    "spectral_angle",
    "spectral_information_divergence",
    "write_spectral_library",
]
