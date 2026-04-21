"""Methane retrieval: thinned marked temporal point processes + POD models.

Subpackages:
  - intensity       — λ(t) equinox modules (Poisson, LGCP kernels, operational)
  - pod_functions   — P_d(·) equinox modules (logistic → full varying-coefficient)
  - paradox         — Monte Carlo simulation of the missing-mass paradox
  - fitting         — NumPyro NUTS fitting of a PoD-modified power-law
"""

from methane_pod import intensity, pod_functions, paradox, fitting
from methane_pod.intensity import INTENSITY_REGISTRY
from methane_pod.pod_functions import POD_REGISTRY
from methane_pod.paradox import (
    FacilityConfig,
    ParadoxResult,
    simulate_paradox,
    compute_E_Pd,
    lognormal_pdf,
    logistic_pod,
    build_canonical_scenarios,
)
from methane_pod.fitting import (
    X_MIN_DEFAULT,
    X_MAX_DEFAULT,
    lognorm_cdf,
    power_law,
    pod_powerlaw_model,
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
