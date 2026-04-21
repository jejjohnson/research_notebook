"""Steady-state Gaussian plume model (JAX + NumPyro).

Submodules:
  - dispersion    — Briggs σ_y(x), σ_z(x) coefficients (A-F stability classes)
  - plume         — coordinate rotation + JIT forward model + xarray wrapper
  - inference     — NumPyro Bayesian inference for Q (and optional stability)

The ``inference`` submodule is imported lazily on first attribute access so
that ``from plume_simulation import gauss_plume`` does not pull in NumPyro
for consumers of only the forward model.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from plume_simulation.gauss_plume import dispersion, plume
from plume_simulation.gauss_plume.dispersion import (
    BRIGGS_DISPERSION_PARAMS,
    STABILITY_CLASSES,
    calculate_briggs_dispersion,
    get_dispersion_params,
)
from plume_simulation.gauss_plume.plume import (
    MIN_WIND_SPEED,
    plume_concentration,
    plume_concentration_vmap,
    rotate_to_wind_frame,
    simulate_plume,
)


_LAZY_INFERENCE_SYMBOLS = {
    "gaussian_plume_model",
    "infer_emission_rate",
}


def __getattr__(name: str):  # PEP 562 lazy module-level attribute
    if name == "inference" or name in _LAZY_INFERENCE_SYMBOLS:
        module = importlib.import_module("plume_simulation.gauss_plume.inference")
        if name == "inference":
            return module
        return getattr(module, name)
    raise AttributeError(
        f"module 'plume_simulation.gauss_plume' has no attribute {name!r}"
    )


if TYPE_CHECKING:  # make the lazy names visible to type checkers / IDEs
    from plume_simulation.gauss_plume import inference
    from plume_simulation.gauss_plume.inference import (
        gaussian_plume_model,
        infer_emission_rate,
    )


__all__ = [
    "dispersion",
    "plume",
    "inference",
    "BRIGGS_DISPERSION_PARAMS",
    "STABILITY_CLASSES",
    "MIN_WIND_SPEED",
    "calculate_briggs_dispersion",
    "get_dispersion_params",
    "rotate_to_wind_frame",
    "plume_concentration",
    "plume_concentration_vmap",
    "simulate_plume",
    "gaussian_plume_model",
    "infer_emission_rate",
]
