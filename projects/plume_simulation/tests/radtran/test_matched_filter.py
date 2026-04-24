"""Matched-filter retrieval end-to-end."""

from __future__ import annotations

import numpy as np
import pytest
from plume_simulation.radtran.matched_filter import (
    matched_filter_image,
    matched_filter_pixel,
    matched_filter_snr,
)


def test_pixel_zero_when_innovation_zero():
    t = np.array([-0.01, -0.03])
    cov_inv = np.eye(2)
    mu = np.array([0.3, 0.5])
    abundance = matched_filter_pixel(mu, mu, cov_inv, t)
    assert abundance == pytest.approx(0.0, abs=1e-14)


def test_pixel_unity_on_exact_target_addition():
    # (x - μ) = ε · t → ε̂ = ε exactly.
    t = np.array([-0.01, -0.03])
    cov_inv = np.eye(2)
    mu = np.array([0.3, 0.5])
    x = mu + 0.5 * t
    assert matched_filter_pixel(x, mu, cov_inv, t) == pytest.approx(0.5, rel=1e-12)


def test_image_shape_and_values():
    t = np.array([-0.01, -0.03])
    cov_inv = np.eye(2)
    mu = np.array([0.3, 0.5])
    # Clean scene plus half-strength target at one pixel.
    scene = np.broadcast_to(mu[:, None, None], (2, 5, 5)).astype(float).copy()
    scene[:, 2, 2] += 0.5 * t
    eps_map = matched_filter_image(scene, mu, cov_inv, t)
    assert eps_map.shape == (5, 5)
    assert eps_map[2, 2] == pytest.approx(0.5, rel=1e-12)
    # All other pixels are zero.
    eps_map[2, 2] = 0.0
    np.testing.assert_allclose(eps_map, 0.0, atol=1e-14)


def test_snr_scales_with_target_energy():
    t_weak = np.array([-0.001, -0.003])
    t_strong = np.array([-0.01, -0.03])
    cov_inv = np.eye(2)
    abundance = np.array([[1.0]])
    snr_weak = matched_filter_snr(abundance, cov_inv, t_weak)
    snr_strong = matched_filter_snr(abundance, cov_inv, t_strong)
    assert snr_strong > snr_weak


def test_rejects_non_pd_cov():
    t = np.array([-0.01, -0.03])
    # Zero covariance → target_norm is 0 → rejected.
    with pytest.raises(ValueError, match="tᵀ Σ⁻¹ t"):
        matched_filter_pixel(np.zeros(2), np.zeros(2), np.zeros((2, 2)), t)


def test_image_rejects_band_mismatch():
    t = np.array([-0.01, -0.03, -0.02])  # 3-band target
    cov_inv = np.eye(3)
    scene = np.ones((2, 4, 4))  # 2-band scene
    with pytest.raises(ValueError, match="bands but target"):
        matched_filter_image(scene, np.zeros(3), cov_inv, t)


def test_matched_filter_recovers_column_map(synthetic_lut):
    """End-to-end sanity: target built from the LUT + retrieval on injected plume."""
    from plume_simulation.radtran.nb_lut import build_nb_lut, inject_plume
    from plume_simulation.radtran.srf import SpectralResponseFunction
    from plume_simulation.radtran.target import (
        target_bands,
        target_spectrum_normalized_linear,
    )

    # Hyperspectral SRF: one band per wavenumber sample → retrieval is direct.
    nu = synthetic_lut["wavenumber"].values
    wl = 1e7 / nu
    sort = np.argsort(wl)
    srf = SpectralResponseFunction(
        wavelengths_hr_nm=wl[sort],
        band_centers_nm=wl[sort],
        band_widths_nm=np.full(wl.size, 5.0),  # narrow
        band_names=tuple(f"c{i}" for i in range(wl.size)),
        srf_type="gaussian",
    )

    geom = dict(T_K=280.0, p_atm=1.0, path_length_cm=8.4e5, amf=2.0)
    # build_nb_lut doesn't take path_length_cm (ΔX is already a column).
    lut = build_nb_lut(
        synthetic_lut, srf,
        T_K=geom["T_K"], p_atm=geom["p_atm"], amf=geom["amf"],
        max_delta_column=20.0, n_grid=1001,
    )

    # Clean flat scene (normalised radiance = 1 everywhere).
    ny, nx = 16, 16
    clean = np.ones((srf.n_bands, ny, nx))
    # Inject a Gaussian plume centred on (8, 8).
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny))
    plume = 2.0 * np.exp(-((xs - 8) ** 2 + (ys - 8) ** 2) / (2 * 3**2))  # mol/m^2
    dirty = inject_plume(clean, plume, lut)

    # Build a reference target at ΔVMR = 1e-6.
    delta_ref = 1e-6
    t_hr = target_spectrum_normalized_linear(synthetic_lut, nu, delta_vmr=delta_ref, **geom)
    t_b = target_bands(t_hr, srf, 1e7 / nu)

    # Trivial "background" since the scene is flat: μ = 1, Σ = I.
    mu = np.ones(srf.n_bands)
    cov_inv = np.eye(srf.n_bands)
    eps_hat = matched_filter_image(dirty, mu, cov_inv, t_b)

    # The retrieval should peak at the plume centre.
    peak_idx = np.unravel_index(np.argmax(eps_hat), eps_hat.shape)
    assert peak_idx == (8, 8)
    # And produce a monotone decrease outward along one slice.
    assert eps_hat[8, 8] > eps_hat[8, 10] > eps_hat[8, 12]
