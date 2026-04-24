"""End-to-end integration: target from obs → background → MF → amplitude.

Ties the four modules together on a realistic synthetic scene built with the
same LUT + SRF plumbing used by the 3D-Var test suite. If any layer drifts
(target generation, covariance wrapping, score formula) this test breaks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from plume_simulation.matched_filter.background import (
    estimate_cov_shrunk,
    estimate_mean,
)
from plume_simulation.matched_filter.core import apply_image
from plume_simulation.matched_filter.target import linear_target_from_obs


jax.config.update("jax_enable_x64", True)


def test_mf_detects_known_plume(obs_model_no_optics, rng):
    """Inject a rectangular plume via the real forward model; the MF score
    map must clearly separate plume from background and recover amplitude
    to within Beer–Lambert curvature."""
    obs = obs_model_no_optics
    H, W = 12, 16
    x_ref = 0.0

    # 1) Small enough plume that Beer–Lambert is nearly linear (a·Δx ≪ 1).
    amp_map = np.zeros((H, W), dtype=float)
    amp_true = 1e-7  # ~100 ppb enhancement — within linear regime.
    amp_map[4:8, 6:11] = amp_true
    vmr_field = jnp.asarray(amp_map + x_ref)
    radiance_clean = obs.forward(vmr_field, linear=False)
    noise_std = 5e-5 * float(jnp.max(radiance_clean))
    noise = noise_std * rng.standard_normal(radiance_clean.shape)
    observed = np.asarray(radiance_clean) + noise

    # 2) Target from the forward model at the background state.
    x_b = jnp.full((H, W), x_ref)
    target = linear_target_from_obs(obs, x_b, pattern="uniform", linear_forward=False)

    # 3) Background μ, Σ — use median so the plume pixels do not bias μ.
    mu = estimate_mean(observed, method="median")
    cov_op = estimate_cov_shrunk(observed, mean=mu, method="ledoit_wolf")

    # 4) Score map.
    scores = apply_image(
        jnp.asarray(observed),
        mean=jnp.asarray(mu),
        cov_op=cov_op,
        target=target,
    )
    scores_np = np.asarray(scores)
    plume_mean = scores_np[5:7, 7:10].mean()
    offplume_mean = np.concatenate(
        [scores_np[:3, :].ravel(), scores_np[-3:, :].ravel()]
    ).mean()
    # Detection: plume region is clearly above off-plume pixels.
    assert plume_mean > 5 * abs(offplume_mean), (
        f"MF failed to separate plume ({plume_mean:.2e}) from background "
        f"({offplume_mean:.2e})."
    )
    # Amplitude: in the linear regime we recover α to within 30%.
    assert 0.7 * amp_true < plume_mean < 1.3 * amp_true, (
        f"retrieved amp {plume_mean:.2e} out of ±30% of injected {amp_true:.2e}"
    )
