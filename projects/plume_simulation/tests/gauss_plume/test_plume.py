"""Tests for Gaussian plume forward model + simulate wrapper."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from plume_simulation.gauss_plume.dispersion import BRIGGS_DISPERSION_PARAMS
from plume_simulation.gauss_plume.plume import (
    MIN_WIND_SPEED,
    plume_concentration,
    plume_concentration_vmap,
    rotate_to_wind_frame,
    simulate_plume,
)


# ── Coordinate rotation ──────────────────────────────────────────────────────


def test_rotation_identity_for_easterly_wind():
    # Wind from west (u > 0, v = 0) → downwind axis aligns with +x.
    x = jnp.array([100.0, 200.0, 300.0])
    y = jnp.array([0.0, 50.0, -50.0])
    x_wind, y_wind = rotate_to_wind_frame(x, y, 0.0, 0.0, 5.0, 0.0)
    np.testing.assert_allclose(np.asarray(x_wind), [100.0, 200.0, 300.0], atol=1e-4)
    np.testing.assert_allclose(np.asarray(y_wind), [0.0, 50.0, -50.0], atol=1e-4)


def test_rotation_northerly_wind():
    # Wind toward +y (u=0, v=5): downwind should be dy, crosswind should be -dx.
    x = jnp.array([50.0, -50.0])
    y = jnp.array([100.0, 200.0])
    x_wind, y_wind = rotate_to_wind_frame(x, y, 0.0, 0.0, 0.0, 5.0)
    np.testing.assert_allclose(np.asarray(x_wind), [100.0, 200.0], atol=1e-4)
    np.testing.assert_allclose(np.asarray(y_wind), [-50.0, 50.0], atol=1e-4)


def test_rotation_translates_by_source():
    x_wind, y_wind = rotate_to_wind_frame(
        jnp.array([100.0]), jnp.array([50.0]),
        source_x=100.0, source_y=50.0, wind_u=5.0, wind_v=0.0,
    )
    np.testing.assert_allclose(np.asarray(x_wind), [0.0], atol=1e-4)
    np.testing.assert_allclose(np.asarray(y_wind), [0.0], atol=1e-4)


# ── Forward model ────────────────────────────────────────────────────────────


def _sample_points() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x = jnp.array([500.0, 500.0, 500.0, -100.0, 1000.0])
    y = jnp.array([0.0, 50.0, -50.0, 0.0, 0.0])
    z = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    return x, y, z


def test_concentration_is_non_negative_everywhere():
    x, y, z = _sample_points()
    c = plume_concentration(
        x, y, z, 0.0, 0.0, 2.0, 5.0, 0.0, 0.1, BRIGGS_DISPERSION_PARAMS["C"]
    )
    assert np.all(np.asarray(c) >= 0.0)


def test_concentration_zero_upwind():
    # Upwind point (x=-100 when wind is westerly) must be exactly zero.
    x = jnp.array([-100.0, -500.0])
    y = jnp.array([0.0, 0.0])
    z = jnp.array([1.0, 1.0])
    c = plume_concentration(
        x, y, z, 0.0, 0.0, 2.0, 5.0, 0.0, 0.1, BRIGGS_DISPERSION_PARAMS["C"]
    )
    np.testing.assert_array_equal(np.asarray(c), [0.0, 0.0])


def test_concentration_maximum_on_centerline():
    # At fixed downwind x and height, the crosswind y=0 point must dominate.
    x = jnp.array([500.0, 500.0, 500.0])
    y = jnp.array([0.0, 50.0, -50.0])
    z = jnp.array([1.0, 1.0, 1.0])
    c = plume_concentration(
        x, y, z, 0.0, 0.0, 2.0, 5.0, 0.0, 0.1, BRIGGS_DISPERSION_PARAMS["C"]
    )
    c = np.asarray(c)
    assert c[0] > c[1]
    assert c[0] > c[2]


def test_concentration_decays_downwind():
    # Along the centerline the concentration at ground level is non-increasing
    # with downwind distance for neutral stability (class D).
    x = jnp.array([200.0, 500.0, 1000.0, 2000.0, 5000.0])
    y = jnp.zeros_like(x)
    z = jnp.zeros_like(x)  # ground level
    c = plume_concentration(
        x, y, z, 0.0, 0.0, 2.0, 5.0, 0.0, 0.1, BRIGGS_DISPERSION_PARAMS["D"]
    )
    c = np.asarray(c)
    # Compare well downwind, past any near-source artefacts from the clamp.
    assert c[1] > c[2] > c[3] > c[4]


def test_concentration_linear_in_emission_rate():
    x, y, z = _sample_points()
    params = BRIGGS_DISPERSION_PARAMS["C"]
    c1 = np.asarray(
        plume_concentration(x, y, z, 0.0, 0.0, 2.0, 5.0, 0.0, 0.1, params)
    )
    c2 = np.asarray(
        plume_concentration(x, y, z, 0.0, 0.0, 2.0, 5.0, 0.0, 0.3, params)
    )
    np.testing.assert_allclose(c2, 3.0 * c1, rtol=1e-6)


def test_concentration_rotation_invariance():
    # Rotating the (receptor + source + wind) system by 90° should give the
    # same scalar field values at matched points.
    params = BRIGGS_DISPERSION_PARAMS["D"]

    # Easterly wind, downwind receptor.
    c_east = plume_concentration(
        jnp.array([500.0]), jnp.array([50.0]), jnp.array([2.0]),
        0.0, 0.0, 2.0, 5.0, 0.0, 0.1, params,
    )

    # Same point, rotated 90° CCW: (500, 50) -> (-50, 500), wind (5, 0) -> (0, 5).
    c_north = plume_concentration(
        jnp.array([-50.0]), jnp.array([500.0]), jnp.array([2.0]),
        0.0, 0.0, 2.0, 0.0, 5.0, 0.1, params,
    )
    np.testing.assert_allclose(
        np.asarray(c_east), np.asarray(c_north), rtol=1e-5
    )


def test_wind_speed_clamp_at_calm_regime():
    # A tiny wind speed (< MIN_WIND_SPEED) should behave as if the wind were
    # exactly at MIN_WIND_SPEED (direction preserved).
    params = BRIGGS_DISPERSION_PARAMS["C"]
    x = jnp.array([500.0])
    y = jnp.array([0.0])
    z = jnp.array([2.0])

    c_tiny = plume_concentration(x, y, z, 0.0, 0.0, 2.0, 1e-6, 0.0, 0.1, params)
    c_clamp = plume_concentration(
        x, y, z, 0.0, 0.0, 2.0, MIN_WIND_SPEED, 0.0, 0.1, params
    )
    np.testing.assert_allclose(np.asarray(c_tiny), np.asarray(c_clamp), rtol=1e-5)


def test_plume_concentration_vmap_matches_scalar():
    params = BRIGGS_DISPERSION_PARAMS["C"]
    x, y, z = _sample_points()
    c_batch = plume_concentration_vmap(
        x, y, z, 0.0, 0.0, 2.0, 5.0, 0.0, 0.1, params
    )
    c_ref = plume_concentration(
        x, y, z, 0.0, 0.0, 2.0, 5.0, 0.0, 0.1, params
    )
    np.testing.assert_allclose(
        np.asarray(c_batch), np.asarray(c_ref), rtol=1e-6
    )


def test_concentration_gradient_wrt_emission_rate():
    import jax
    params = BRIGGS_DISPERSION_PARAMS["D"]
    x = jnp.array([500.0])
    y = jnp.array([0.0])
    z = jnp.array([2.0])

    def scalar(q):
        return plume_concentration(x, y, z, 0.0, 0.0, 2.0, 5.0, 0.0, q, params)[0]

    grad_fn = jax.grad(scalar)
    dC_dQ = float(grad_fn(0.1))
    # For a linear-in-Q model, dC/dQ = C(Q) / Q.
    c_val = float(scalar(0.1))
    np.testing.assert_allclose(dC_dQ, c_val / 0.1, rtol=1e-5)


# ── simulate_plume ───────────────────────────────────────────────────────────


def test_simulate_plume_builds_dataset():
    ds = simulate_plume(
        emission_rate=0.1,
        source_location=(0.0, 0.0, 2.0),
        wind_speed=5.0,
        wind_direction=270.0,  # from west
        stability_class="D",
        domain_x=(-200.0, 2000.0, 45),
        domain_y=(-500.0, 500.0, 31),
        domain_z=(0.0, 200.0, 11),
        background_conc=1e-8,
    )
    assert set(ds.data_vars) == {"concentration", "column_concentration"}
    assert ds["concentration"].dims == ("x", "y", "z")
    assert ds["column_concentration"].dims == ("x", "y")
    assert ds["concentration"].shape == (45, 31, 11)
    # Background present everywhere (including upwind where plume mass is 0).
    upwind = ds["concentration"].sel(x=-100.0, method="nearest").values
    assert np.allclose(upwind, 1e-8)
    assert ds["concentration"].max().values > 1e-8
    # Metadata sanity.
    assert ds.attrs["stability_class"] == "D"
    assert ds.attrs["wind_speed"] == 5.0


def test_simulate_plume_rejects_bad_inputs():
    common = dict(
        source_location=(0.0, 0.0, 2.0),
        wind_speed=5.0,
        wind_direction=270.0,
        stability_class="D",
        domain_x=(-200.0, 2000.0, 10),
        domain_y=(-500.0, 500.0, 10),
        domain_z=(0.0, 200.0, 5),
    )
    with pytest.raises(ValueError, match=r"`emission_rate` must be > 0"):
        simulate_plume(emission_rate=-0.1, **common)
    with pytest.raises(ValueError, match=r"`wind_speed` must be > 0"):
        simulate_plume(
            emission_rate=0.1,
            source_location=(0.0, 0.0, 2.0),
            wind_speed=0.0,
            wind_direction=270.0,
            stability_class="D",
            domain_x=(-200.0, 2000.0, 10),
            domain_y=(-500.0, 500.0, 10),
            domain_z=(0.0, 200.0, 5),
        )
    with pytest.raises(
        ValueError, match=r"simulate_plume: `stability_class` must be one of"
    ):
        simulate_plume(
            emission_rate=0.1,
            source_location=(0.0, 0.0, 2.0),
            wind_speed=5.0,
            wind_direction=270.0,
            stability_class="Z",
            domain_x=(-200.0, 2000.0, 10),
            domain_y=(-500.0, 500.0, 10),
            domain_z=(0.0, 200.0, 5),
        )
    with pytest.raises(ValueError, match=r"source height must be ≥ 0"):
        simulate_plume(
            emission_rate=0.1,
            source_location=(0.0, 0.0, -1.0),
            wind_speed=5.0,
            wind_direction=270.0,
            stability_class="D",
            domain_x=(-200.0, 2000.0, 10),
            domain_y=(-500.0, 500.0, 10),
            domain_z=(0.0, 200.0, 5),
        )
    with pytest.raises(ValueError, match=r"`domain_x` requires start < stop"):
        simulate_plume(
            emission_rate=0.1,
            source_location=(0.0, 0.0, 2.0),
            wind_speed=5.0,
            wind_direction=270.0,
            stability_class="D",
            domain_x=(1000.0, 100.0, 10),
            domain_y=(-500.0, 500.0, 10),
            domain_z=(0.0, 200.0, 5),
        )
    with pytest.raises(ValueError, match=r"`domain_z` requires n_points ≥ 2"):
        simulate_plume(
            emission_rate=0.1,
            source_location=(0.0, 0.0, 2.0),
            wind_speed=5.0,
            wind_direction=270.0,
            stability_class="D",
            domain_x=(-200.0, 2000.0, 10),
            domain_y=(-500.0, 500.0, 10),
            domain_z=(0.0, 200.0, 1),
        )


def test_simulate_plume_wind_direction_convention():
    # wind_direction = 270° → wind from west, u > 0, v ≈ 0 → plume tongue on +x.
    ds = simulate_plume(
        emission_rate=0.1,
        source_location=(0.0, 0.0, 2.0),
        wind_speed=5.0,
        wind_direction=270.0,
        stability_class="D",
        domain_x=(-200.0, 2000.0, 45),
        domain_y=(-500.0, 500.0, 31),
        domain_z=(0.0, 200.0, 5),
    )
    # Column concentration should peak at x > 0 (downwind).
    col = ds["column_concentration"].values
    peak_x_idx, _ = np.unravel_index(np.argmax(col), col.shape)
    assert ds["x"].values[peak_x_idx] > 0.0
    # And should be ~zero for x < -10 (upwind of source).
    upwind_mask = ds["x"].values < -10.0
    assert col[upwind_mask, :].max() < 1e-20
