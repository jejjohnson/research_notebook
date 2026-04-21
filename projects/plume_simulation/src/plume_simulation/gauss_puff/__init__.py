"""Gaussian puff model (JAX + diffrax + NumPyro).

Submodules:
  - dispersion  — Pasquill-Gifford σ(s) coefficients + shared Briggs dispatch
  - wind        — piecewise-linear WindSchedule + diffrax cumulative integrals
  - puff        — single-puff kernel, evolve_puffs, simulate_puff (xarray)
  - inference   — NumPyro Bayesian inference for Q (constant) and Q_i (random walk)

The ``inference`` submodule is imported lazily on first attribute access so
that ``from plume_simulation import gauss_puff`` does not pull in NumPyro
for consumers of only the forward model.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from plume_simulation.gauss_puff import dispersion, puff, wind
from plume_simulation.gauss_puff.dispersion import (
    DISPERSION_SCHEMES,
    PG_DISPERSION_PARAMS,
    STABILITY_CLASSES,
    calculate_briggs_dispersion_xyz,
    calculate_pg_dispersion,
    get_dispersion_scheme,
    get_pg_params,
)
from plume_simulation.gauss_puff.puff import (
    PuffState,
    evolve_puffs,
    frequency_to_release_interval,
    make_release_times,
    puff_concentration,
    puff_concentration_vmap,
    release_interval_to_frequency,
    simulate_puff,
    simulate_puff_field,
)
from plume_simulation.gauss_puff.wind import (
    WindSchedule,
    cumulative_wind_integrals,
)


_LAZY_INFERENCE_SYMBOLS = {
    "gaussian_puff_model",
    "gaussian_puff_rw_model",
    "infer_emission_rate",
    "infer_emission_timeseries",
}


def __getattr__(name: str):  # PEP 562 lazy module-level attribute
    if name == "inference" or name in _LAZY_INFERENCE_SYMBOLS:
        module = importlib.import_module("plume_simulation.gauss_puff.inference")
        if name == "inference":
            return module
        return getattr(module, name)
    raise AttributeError(
        f"module 'plume_simulation.gauss_puff' has no attribute {name!r}"
    )


if TYPE_CHECKING:  # make the lazy names visible to type checkers / IDEs
    from plume_simulation.gauss_puff import inference
    from plume_simulation.gauss_puff.inference import (
        gaussian_puff_model,
        gaussian_puff_rw_model,
        infer_emission_rate,
        infer_emission_timeseries,
    )


__all__ = [
    "dispersion",
    "wind",
    "puff",
    "inference",
    "DISPERSION_SCHEMES",
    "PG_DISPERSION_PARAMS",
    "STABILITY_CLASSES",
    "PuffState",
    "WindSchedule",
    "calculate_briggs_dispersion_xyz",
    "calculate_pg_dispersion",
    "cumulative_wind_integrals",
    "evolve_puffs",
    "frequency_to_release_interval",
    "get_dispersion_scheme",
    "get_pg_params",
    "make_release_times",
    "puff_concentration",
    "puff_concentration_vmap",
    "release_interval_to_frequency",
    "simulate_puff",
    "simulate_puff_field",
    "gaussian_puff_model",
    "gaussian_puff_rw_model",
    "infer_emission_rate",
    "infer_emission_timeseries",
]
