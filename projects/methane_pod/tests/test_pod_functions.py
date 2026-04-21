"""Shape, range [0,1], and prior-sampling smoke tests for POD modules."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpyro
import pytest

from methane_pod import pod_functions as pf
from methane_pod.pod_functions import POD_REGISTRY


N = 32
Q = jnp.linspace(1.0, 5_000.0, N)
# For monotone-in-Q tests, hold all other covariates at a single
# representative value so the only varying input is flux.
U_SWEEP = jnp.linspace(1.0, 10.0, N)
P_SWEEP = jnp.linspace(25.0, 500.0, N)
ALB_SWEEP = jnp.linspace(0.1, 0.9, N)
COS_THETA_SWEEP = jnp.linspace(0.3, 1.0, N)
SIGMA_NOISE_SWEEP = jnp.linspace(0.01, 0.5, N)
N_BANDS_SWEEP = jnp.linspace(10.0, 250.0, N)
RES_NM_SWEEP = jnp.linspace(0.5, 20.0, N)
U_FIXED = jnp.full(N, 5.0)
P_FIXED = jnp.full(N, 100.0)
ALB_FIXED = jnp.full(N, 0.5)
COS_THETA_FIXED = jnp.full(N, 0.8)
SIGMA_NOISE_FIXED = jnp.full(N, 0.1)
N_BANDS_FIXED = jnp.full(N, 100.0)
RES_NM_FIXED = jnp.full(N, 5.0)


# Inputs that sweep every covariate — for shape / [0,1] / prior tests.
CALL_ARGS_SWEEP = {
    "LogisticPOD": (Q,),
    "LogLogisticPOD": (Q,),
    "ConcentrationProxyPOD": (Q, U_SWEEP, P_SWEEP),
    "AdditiveMultiCovariatePOD": (Q, U_SWEEP, P_SWEEP, ALB_SWEEP, COS_THETA_SWEEP),
    "VaryingCoefficientPOD": (Q, U_SWEEP, P_SWEEP, ALB_SWEEP, COS_THETA_SWEEP),
    "SNRBasedPOD": (Q, U_SWEEP, P_SWEEP, ALB_SWEEP, SIGMA_NOISE_SWEEP),
    "SpectralAwarePOD": (Q, U_SWEEP, P_SWEEP, ALB_SWEEP, N_BANDS_SWEEP, RES_NM_SWEEP),
    "FullVaryingCoefficientPOD": (
        Q, U_SWEEP, P_SWEEP, ALB_SWEEP, COS_THETA_SWEEP, N_BANDS_SWEEP, SIGMA_NOISE_SWEEP
    ),
    "ProbitPOD": (Q, U_SWEEP, P_SWEEP),
    "CloglogPOD": (Q, U_SWEEP, P_SWEEP),
}

# Inputs that hold every non-Q covariate fixed — for monotone-in-Q test.
CALL_ARGS_Q_ONLY = {
    "LogisticPOD": (Q,),
    "LogLogisticPOD": (Q,),
    "ConcentrationProxyPOD": (Q, U_FIXED, P_FIXED),
    "AdditiveMultiCovariatePOD": (Q, U_FIXED, P_FIXED, ALB_FIXED, COS_THETA_FIXED),
    "VaryingCoefficientPOD": (Q, U_FIXED, P_FIXED, ALB_FIXED, COS_THETA_FIXED),
    "SNRBasedPOD": (Q, U_FIXED, P_FIXED, ALB_FIXED, SIGMA_NOISE_FIXED),
    "SpectralAwarePOD": (Q, U_FIXED, P_FIXED, ALB_FIXED, N_BANDS_FIXED, RES_NM_FIXED),
    "FullVaryingCoefficientPOD": (
        Q, U_FIXED, P_FIXED, ALB_FIXED, COS_THETA_FIXED, N_BANDS_FIXED, SIGMA_NOISE_FIXED
    ),
    "ProbitPOD": (Q, U_FIXED, P_FIXED),
    "CloglogPOD": (Q, U_FIXED, P_FIXED),
}


CONSTRUCTORS = {
    "LogisticPOD": dict(Q_50=100.0, k=0.02),
    "LogLogisticPOD": dict(beta_0=-8.0, beta_1=2.0),
    "ConcentrationProxyPOD": dict(beta_0=-3.0, beta_1=2.0),
    "AdditiveMultiCovariatePOD": dict(
        beta_0=-5.0, beta_Q=2.0, beta_U=-1.0, beta_p=-0.5, beta_A=1.5, beta_theta=1.0
    ),
    "VaryingCoefficientPOD": dict(
        beta_0=-3.5, gamma_base=1.5, gamma_albedo=1.0, gamma_cos_theta=0.5
    ),
    "SNRBasedPOD": dict(beta_0=-2.0, beta_1=2.0),
    "SpectralAwarePOD": dict(
        beta_0=-5.0,
        beta_proxy=2.0,
        beta_n_bands=1.0,
        beta_spectral_res=-0.5,
        beta_albedo=1.0,
    ),
    "FullVaryingCoefficientPOD": dict(
        beta_0_base=-3.5,
        beta_0_noise=-0.5,
        gamma_base=1.5,
        gamma_albedo=1.0,
        gamma_cos_theta=0.5,
        gamma_n_bands=0.5,
    ),
    "ProbitPOD": dict(beta_0=-2.0, beta_1=1.5),
    "CloglogPOD": dict(beta_0=-3.0, beta_1=1.5),
}


@pytest.mark.parametrize("cls_name", list(CONSTRUCTORS), ids=list(CONSTRUCTORS))
def test_pod_output_in_unit_interval(cls_name):
    cls = getattr(pf, cls_name)
    model = cls(**CONSTRUCTORS[cls_name])
    y = model(*CALL_ARGS_SWEEP[cls_name])
    assert y.shape == Q.shape
    assert jnp.all(jnp.isfinite(y))
    assert jnp.all(y >= 0.0) and jnp.all(y <= 1.0 + 1e-6)


@pytest.mark.parametrize("cls_name", list(CONSTRUCTORS), ids=list(CONSTRUCTORS))
def test_pod_monotone_in_Q(cls_name):
    """P_d should be non-decreasing in Q (all models have β_Q > 0 by design)."""
    cls = getattr(pf, cls_name)
    model = cls(**CONSTRUCTORS[cls_name])
    y = model(*CALL_ARGS_Q_ONLY[cls_name])
    diffs = jnp.diff(y)
    assert jnp.all(diffs >= -1e-4), f"{cls_name} is non-monotone in Q: min diff={float(diffs.min())}"


@pytest.mark.parametrize("cls_name", list(CONSTRUCTORS), ids=list(CONSTRUCTORS))
def test_priors_sample_and_evaluate(cls_name):
    cls = getattr(pf, cls_name)

    def model():
        m = cls.sample_priors()
        y = m(*CALL_ARGS_SWEEP[cls_name])
        numpyro.deterministic("y_finite", jnp.all(jnp.isfinite(y)))
        numpyro.deterministic("y_in_unit", jnp.all((y >= 0.0) & (y <= 1.0 + 1e-6)))

    with numpyro.handlers.seed(rng_seed=jr.PRNGKey(0)):
        trace = numpyro.handlers.trace(model).get_trace()
    assert bool(trace["y_finite"]["value"])
    assert bool(trace["y_in_unit"]["value"])


def test_registry_keys_map_to_classes():
    for key, cls in POD_REGISTRY.items():
        assert hasattr(cls, "__call__")
        assert hasattr(cls, "sample_priors"), f"{key} missing sample_priors"
