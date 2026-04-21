"""Shape, finiteness, and prior-sampling smoke tests for intensity modules."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpyro
import pytest

from methane_pod import intensity
from methane_pod.intensity import INTENSITY_REGISTRY


T_GRID = jnp.linspace(0.0, 365.0, 257)


CONSTRUCTOR_ARGS = {
    "ConstantIntensity": dict(lambda_0=2.0),
    "DiurnalSinusoidalIntensity": dict(lambda_0=5.0, amplitude=2.0),
    "SeasonalSinusoidalIntensity": dict(lambda_0=2.0, amplitude=1.0),
    "DiurnalSeasonalIntensity": dict(lambda_0=3.0, amp_diurnal=1.5, amp_seasonal=1.0),
    "SeasonallyModulatedDiurnalIntensity": dict(
        lambda_0=4.0, amp_summer=3.0, amp_winter=0.5
    ),
    "OperationalScheduleIntensity": dict(lambda_active=8.0, lambda_idle=1.0),
    "WeibullRenewalIntensity": dict(scale=0.5, shape=3.0),
    "PeriodicBatchIntensity": dict(
        lambda_background=0.5, lambda_peak=20.0, period_days=0.5
    ),
    "CoalMineVentilationIntensity": dict(
        lambda_extraction=10.0, lambda_maintenance=1.0, lambda_idle=3.0
    ),
    "LandfillIntensity": dict(
        lambda_0=4.0, amp_seasonal=2.0, amp_diurnal=1.5, amp_barometric=1.0
    ),
    "OffshorePlatformIntensity": dict(
        lambda_0=5.0, amp_tidal=1.5, amp_operational=2.0
    ),
    "WetlandPermafrostIntensity": dict(lambda_peak=5.0, lambda_frozen=0.1),
    "LivestockFeedlotIntensity": dict(
        lambda_0=3.0, amp_feeding=2.0, amp_seasonal=1.0
    ),
}


@pytest.mark.parametrize(
    "cls_name,kwargs", list(CONSTRUCTOR_ARGS.items()), ids=list(CONSTRUCTOR_ARGS)
)
def test_intensity_call_shape_and_finite(cls_name, kwargs):
    cls = getattr(intensity, cls_name)
    model = cls(**kwargs)
    if cls_name == "WeibullRenewalIntensity":
        t = jnp.linspace(0.05, 5.0, 64)
    else:
        t = T_GRID
    lam = model(t)
    assert lam.shape == t.shape
    assert jnp.all(jnp.isfinite(lam)), f"{cls_name} produced non-finite λ"


@pytest.mark.parametrize(
    "cls_name,kwargs", list(CONSTRUCTOR_ARGS.items()), ids=list(CONSTRUCTOR_ARGS)
)
def test_intensity_nonneg(cls_name, kwargs):
    cls = getattr(intensity, cls_name)
    model = cls(**kwargs)
    if cls_name == "WeibullRenewalIntensity":
        t = jnp.linspace(0.05, 5.0, 64)
    else:
        t = T_GRID
    lam = model(t)
    assert jnp.all(lam >= -1e-6), f"{cls_name} produced negative intensity"


@pytest.mark.parametrize(
    "cls_name",
    [n for n in CONSTRUCTOR_ARGS if n != "WeibullRenewalIntensity"],
)
def test_priors_sample_and_evaluate(cls_name):
    """Prior-factory returns a runnable module; λ(t) is finite on the grid."""
    cls = getattr(intensity, cls_name)

    def model():
        m = cls.sample_priors()
        lam = m(T_GRID)
        numpyro.deterministic("lam_finite", jnp.all(jnp.isfinite(lam)))

    with numpyro.handlers.seed(rng_seed=jr.PRNGKey(0)):
        trace = numpyro.handlers.trace(model).get_trace()
    assert bool(trace["lam_finite"]["value"])


def test_registry_keys_map_to_classes():
    for key, cls in INTENSITY_REGISTRY.items():
        assert hasattr(cls, "__call__")
        assert hasattr(cls, "sample_priors"), f"{key} missing sample_priors"
