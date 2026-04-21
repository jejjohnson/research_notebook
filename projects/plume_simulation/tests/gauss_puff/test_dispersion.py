"""Tests for the Gaussian-puff dispersion module (Pasquill-Gifford)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from plume_simulation.gauss_puff.dispersion import (
    DISPERSION_SCHEMES,
    PG_DISPERSION_PARAMS,
    STABILITY_CLASSES,
    calculate_briggs_dispersion_xyz,
    calculate_pg_dispersion,
    get_dispersion_scheme,
    get_pg_params,
)


def test_stability_classes_registered():
    assert set(PG_DISPERSION_PARAMS.keys()) == set(STABILITY_CLASSES)


def test_pg_param_shape():
    for cls in STABILITY_CLASSES:
        assert PG_DISPERSION_PARAMS[cls].shape == (6,)


def test_get_pg_params_roundtrip():
    params = get_pg_params("C")
    np.testing.assert_array_equal(params, PG_DISPERSION_PARAMS["C"])


def test_get_pg_params_rejects_bad_class():
    with pytest.raises(ValueError, match=r"stability_class must be one of"):
        get_pg_params("Z")


def test_pg_dispersion_positive_and_x_equals_y():
    distance = jnp.array([10.0, 100.0, 1000.0, 10000.0])
    params = PG_DISPERSION_PARAMS["C"]
    sx, sy, sz = calculate_pg_dispersion(distance, params)
    assert jnp.all(sx > 0)
    assert jnp.all(sy > 0)
    assert jnp.all(sz > 0)
    np.testing.assert_array_equal(sx, sy)


def test_pg_dispersion_monotone_increasing():
    distance = jnp.array([10.0, 100.0, 1000.0, 10000.0])
    for cls in STABILITY_CLASSES:
        sx, sy, sz = calculate_pg_dispersion(distance, PG_DISPERSION_PARAMS[cls])
        # σ_y monotone increasing in travel distance over this range
        assert jnp.all(jnp.diff(sy) > 0), f"σ_y not monotone for class {cls}"
        # σ_z also monotone in this range for classes A-F
        assert jnp.all(jnp.diff(sz) > 0), f"σ_z not monotone for class {cls}"


def test_pg_dispersion_clamps_nonpositive_distance():
    # Distance clamp ensures ln(s) stays finite at s ≤ 0.
    params = PG_DISPERSION_PARAMS["D"]
    sx_zero, _, sz_zero = calculate_pg_dispersion(jnp.array(0.0), params)
    sx_one, _, sz_one = calculate_pg_dispersion(jnp.array(1.0), params)
    np.testing.assert_allclose(sx_zero, sx_one, rtol=1e-6)
    np.testing.assert_allclose(sz_zero, sz_one, rtol=1e-6)
    sx_neg, _, _ = calculate_pg_dispersion(jnp.array(-10.0), params)
    np.testing.assert_allclose(sx_neg, sx_one, rtol=1e-6)


def test_pg_unstable_spreads_more_than_stable():
    # At any given travel distance, class A (unstable) should give larger σ_y
    # than class F (very stable).
    distance = jnp.array(500.0)
    sxA, syA, szA = calculate_pg_dispersion(distance, PG_DISPERSION_PARAMS["A"])
    sxF, syF, szF = calculate_pg_dispersion(distance, PG_DISPERSION_PARAMS["F"])
    assert float(syA) > float(syF)
    assert float(szA) > float(szF)


def test_dispersion_schemes_registered():
    assert set(DISPERSION_SCHEMES) == {"pg", "briggs"}


def test_get_dispersion_scheme_returns_consistent_pair():
    params_dict, fn = get_dispersion_scheme("pg")
    assert fn is calculate_pg_dispersion
    assert params_dict is PG_DISPERSION_PARAMS
    params_dict, fn = get_dispersion_scheme("briggs")
    assert fn is calculate_briggs_dispersion_xyz


def test_get_dispersion_scheme_rejects_bad_scheme():
    with pytest.raises(ValueError, match=r"scheme` must be one of"):
        get_dispersion_scheme("unknown")


def test_briggs_xyz_wrapper_matches_plume_output():
    from plume_simulation.gauss_plume.dispersion import (
        BRIGGS_DISPERSION_PARAMS,
        calculate_briggs_dispersion,
    )

    distance = jnp.array([50.0, 500.0, 5000.0])
    params = BRIGGS_DISPERSION_PARAMS["D"]
    sy_plume, sz_plume = calculate_briggs_dispersion(distance, params)
    sx, sy, sz = calculate_briggs_dispersion_xyz(distance, params)
    np.testing.assert_array_equal(sx, sy_plume)
    np.testing.assert_array_equal(sy, sy_plume)
    np.testing.assert_array_equal(sz, sz_plume)
