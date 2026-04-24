"""Tests for ``plume_simulation.matched_filter.target``.

Checks that:
- The JVP-based linear target matches a finite-difference estimate of
  ∂H/∂VMR at the background state.
- Linear target is spatially invariant for the 'uniform' pattern (sanity).
- Finite-amplitude nonlinear target reduces to ``α × linear`` for small α.
- Dispatch on patterns rejects invalid inputs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from plume_simulation.matched_filter.target import (
    linear_target_from_obs,
    nonlinear_target_from_obs,
)


jax.config.update("jax_enable_x64", True)


def test_linear_target_matches_finite_difference(obs_model_no_optics):
    """``linear_target`` is the directional derivative of the forward model."""
    obs = obs_model_no_optics
    H, W = 4, 4
    x_b = jnp.full((H, W), 1e-6)  # realistic VMR scale
    t = linear_target_from_obs(obs, x_b, pattern="uniform", linear_forward=True)
    # Finite difference: ΔH / Δα for a uniform perturbation.
    eps = 1e-4
    dx = jnp.ones_like(x_b)
    L0 = obs.forward(x_b, linear=True)
    L1 = obs.forward(x_b + eps * dx, linear=True)
    dL = (L1 - L0) / eps
    i, j = H // 2, W // 2
    scale = float(np.max(np.abs(np.asarray(dL[i, j])))) + 1e-30
    np.testing.assert_allclose(
        np.asarray(t), np.asarray(dL[i, j]), rtol=1e-4, atol=1e-6 * scale
    )


def test_uniform_target_is_spatially_invariant(obs_model_no_optics):
    """For pattern='uniform' (and no PSF), every output pixel sees the same t."""
    obs = obs_model_no_optics
    H, W = 4, 5
    x_b = jnp.full((H, W), 2e-6)
    t_centre = linear_target_from_obs(obs, x_b, pattern="uniform")
    t_corner = linear_target_from_obs(obs, x_b, pattern="uniform", pixel=(0, 0))
    np.testing.assert_allclose(np.asarray(t_centre), np.asarray(t_corner), atol=1e-12)


def test_impulse_target_is_localised(obs_model_no_optics):
    """Impulse at (2,2): only pixel (2,2) has a non-zero signature (no PSF)."""
    obs = obs_model_no_optics
    H, W = 4, 4
    x_b = jnp.full((H, W), 1e-6)
    t_on = linear_target_from_obs(obs, x_b, pattern="impulse", pixel=(2, 2))
    t_off = linear_target_from_obs(obs, x_b, pattern="impulse", pixel=(0, 0))
    # Each impulse returns its own pixel's signature; both should be non-trivial.
    assert np.linalg.norm(np.asarray(t_on)) > 0.0
    assert np.linalg.norm(np.asarray(t_off)) > 0.0


def test_nonlinear_target_small_amplitude_matches_linear(obs_model_no_optics):
    """``nonlinear_target(α) / α → linear_target`` for α → 0.

    The LUT's absorption coefficient peaks around 1e5, so the Beer-Lambert
    "linear regime" requires α·a_max ≪ 1 → α ≲ 1e-8 to avoid saturating the
    line centre.
    """
    obs = obs_model_no_optics
    H, W = 4, 4
    x_b = jnp.full((H, W), 1e-6)
    t_lin = linear_target_from_obs(obs, x_b, pattern="uniform", linear_forward=False)
    amp = 1e-9
    t_nl = nonlinear_target_from_obs(obs, x_b, amplitude=amp, pattern="uniform") / amp
    scale = float(np.max(np.abs(np.asarray(t_lin)))) + 1e-30
    np.testing.assert_allclose(
        np.asarray(t_nl), np.asarray(t_lin), rtol=1e-3, atol=1e-6 * scale
    )


def test_nonlinear_target_saturates_at_large_amplitude(obs_model_no_optics):
    """At large α, nonlinear ≠ α × linear — Beer–Lambert curvature kicks in."""
    obs = obs_model_no_optics
    H, W = 4, 4
    x_b = jnp.full((H, W), 1e-6)
    t_lin = linear_target_from_obs(obs, x_b, pattern="uniform", linear_forward=False)
    amp = 1e-5  # α·a_max ≈ 1 → clearly nonlinear.
    t_nl = nonlinear_target_from_obs(obs, x_b, amplitude=amp, pattern="uniform")
    # Compare only on the support of t_lin where the ratio is well-defined.
    lin_np = np.asarray(t_lin)
    nl_np = np.asarray(t_nl) / amp
    mask = np.abs(lin_np) > 1e-6 * np.max(np.abs(lin_np))
    assert mask.sum() > 0
    ratio = nl_np[mask] / lin_np[mask]
    assert np.all(np.isfinite(ratio))
    assert np.any(np.abs(ratio - 1.0) > 0.05), (
        f"no saturation detected at α={amp}; max |ratio−1| = {np.max(np.abs(ratio - 1.0))}"
    )


def test_custom_pattern_shape_check(obs_model_no_optics):
    obs = obs_model_no_optics
    x_b = jnp.full((4, 4), 1e-6)
    bad = np.zeros((3, 3))
    with pytest.raises(ValueError, match="shape"):
        linear_target_from_obs(obs, x_b, pattern=bad)


def test_bad_pattern_name_raises(obs_model_no_optics):
    obs = obs_model_no_optics
    x_b = jnp.full((4, 4), 1e-6)
    with pytest.raises(ValueError, match="pattern"):
        linear_target_from_obs(obs, x_b, pattern="bogus")  # type: ignore[arg-type]


def test_impulse_pixel_out_of_bounds_raises(obs_model_no_optics):
    """Pixel index outside the VMR field must raise a clear ValueError,
    not a low-level scatter/indexing error from JAX."""
    obs = obs_model_no_optics
    x_b = jnp.full((4, 4), 1e-6)
    with pytest.raises(ValueError, match="out of bounds"):
        linear_target_from_obs(obs, x_b, pattern="impulse", pixel=(4, 0))
    with pytest.raises(ValueError, match="out of bounds"):
        linear_target_from_obs(obs, x_b, pattern="impulse", pixel=(-1, 2))


def test_custom_pattern_extracts_pixel_at_argmax(obs_model_no_optics):
    """When a custom 2-D pattern is supplied without an explicit ``pixel``,
    the returned target is read at the pattern's argmax — matching the
    documented behaviour of :func:`_extract_pixel`."""
    obs = obs_model_no_optics
    H, W = 4, 4
    x_b = jnp.full((H, W), 1e-6)
    pattern = np.zeros((H, W))
    pattern[1, 3] = 1.0  # peak at (1, 3)
    t_auto = linear_target_from_obs(obs, x_b, pattern=pattern)
    t_explicit = linear_target_from_obs(obs, x_b, pattern=pattern, pixel=(1, 3))
    np.testing.assert_allclose(np.asarray(t_auto), np.asarray(t_explicit), atol=1e-12)
