"""Methane retrieval: thinned marked temporal point processes + POD models.

Subpackages:
  - intensity       — λ(t) equinox modules (Poisson, LGCP kernels, operational)
  - pod_functions   — P_d(·) equinox modules (logistic → full varying-coefficient)
  - paradox         — Monte Carlo simulation of the missing-mass paradox
  - fitting         — NumPyro NUTS fitting of a PoD-modified power-law (lazy)

The `fitting` submodule is imported lazily on first attribute access so that
`import methane_pod` doesn't pull in its heavier transitive deps (``pandas``,
``scipy.special.erfc``, the NumPyro MCMC machinery) for users who only need
the intensity / POD / paradox layers.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from methane_pod import intensity, paradox, pod_functions
from methane_pod.intensity import INTENSITY_REGISTRY
from methane_pod.paradox import (
    FacilityConfig,
    ParadoxResult,
    build_canonical_scenarios,
    compute_E_Pd,
    logistic_pod,
    lognormal_pdf,
    simulate_paradox,
)
from methane_pod.pod_functions import POD_REGISTRY


_LAZY_FITTING_SYMBOLS = {
    "X_MIN_DEFAULT",
    "X_MAX_DEFAULT",
    "lognorm_cdf",
    "power_law",
    "pod_powerlaw_model",
    "run_mcmc",
}


def __getattr__(name: str):  # PEP 562 lazy module-level attribute access
    if name == "fitting" or name in _LAZY_FITTING_SYMBOLS:
        module = importlib.import_module("methane_pod.fitting")
        if name == "fitting":
            return module
        return getattr(module, name)
    raise AttributeError(f"module 'methane_pod' has no attribute {name!r}")


if TYPE_CHECKING:  # make the lazy names visible to type checkers / IDEs
    from methane_pod import fitting
    from methane_pod.fitting import (
        X_MAX_DEFAULT,
        X_MIN_DEFAULT,
        lognorm_cdf,
        pod_powerlaw_model,
        power_law,
        run_mcmc,
    )


__all__ = [
    "intensity",
    "pod_functions",
    "paradox",
    "fitting",
    "INTENSITY_REGISTRY",
    "POD_REGISTRY",
    "FacilityConfig",
    "ParadoxResult",
    "simulate_paradox",
    "compute_E_Pd",
    "lognormal_pdf",
    "logistic_pod",
    "build_canonical_scenarios",
    "X_MIN_DEFAULT",
    "X_MAX_DEFAULT",
    "lognorm_cdf",
    "power_law",
    "pod_powerlaw_model",
    "run_mcmc",
]
