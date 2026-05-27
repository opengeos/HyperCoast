# SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Optional machine-learning helpers for HyperCoast."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec
from typing import Any


@dataclass(frozen=True)
class ModelInfo:
    """Describe an optional HyperCoast model family.

    Args:
        name: Model family name.
        task: Supported inference task.
        required_extra: Package extra needed to use the model.
        module: Import path that provides the implementation.
    """

    name: str
    task: str
    required_extra: str
    module: str

    def as_dict(self) -> dict[str, Any]:
        """Return serializable model metadata.

        Returns:
            dict: Model metadata.
        """
        return asdict(self)


MODEL_REGISTRY = {
    "moe_vae": ModelInfo(
        name="moe_vae",
        task="water-quality inference",
        required_extra="ml",
        module="hypercoast.moe_vae",
    ),
}


def list_models() -> dict[str, dict[str, Any]]:
    """Return optional model metadata.

    Returns:
        dict: Mapping of model names to metadata.
    """
    return {name: info.as_dict() for name, info in MODEL_REGISTRY.items()}


def require_ml_dependencies() -> None:
    """Raise a clear error when optional ML dependencies are unavailable.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    missing = [name for name in ("torch",) if find_spec(name) is None]
    if missing:
        packages = ", ".join(missing)
        raise ImportError(
            f"HyperCoast ML features require optional dependencies: {packages}. "
            "Install HyperCoast with the 'ml' extra."
        )


def load_moe_vae_modules():
    """Import and return the optional MoE/VAE package namespace.

    Returns:
        module: ``hypercoast.moe_vae`` package.
    """
    require_ml_dependencies()
    from . import moe_vae

    return moe_vae


__all__ = ["MODEL_REGISTRY", "ModelInfo", "list_models", "load_moe_vae_modules"]
