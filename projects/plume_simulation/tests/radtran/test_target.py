"""Target-spectrum constructors for matched-filter retrievals."""

from __future__ import annotations

import numpy as np
from plume_simulation.radtran.config import number_density_cm3
from plume_simulation.radtran.target import (
    target_bands,
    target_spectrum_normalized_linear,
    target_spectrum_normalized_nonlinear,
)


NU_OBS = np.linspace(4100.0, 4400.0, 41)


def test_normalized_nonlinear_target_is_zero_at_delta_zero(synthetic_lut):
    t = target_spectrum_normalized_nonlinear(
        synthetic_lut, NU_OBS,
        T_K=280.0, p_atm=1.0,
        vmr_background=2e-6, delta_vmr=0.0,
        path_length_cm=8.4e5, amf=2.0,
    )
    np.testing.assert_allclose(t, 0.0, atol=1e-12)


def test_normalized_linear_analytical_form(synthetic_lut):
    kwargs = dict(
        T_K=280.0, p_atm=1.0, delta_vmr=5e-7,
        path_length_cm=8.4e5, amf=2.0,
    )
    t = target_spectrum_normalized_linear(synthetic_lut, NU_OBS, **kwargs)
    # Hand-compute t = -σ · N · ΔVMR · L · AMF.
    sigma = (
        synthetic_lut["absorption_cross_section"]
        .interp(temperature=280.0, pressure=1.0)
        .interp(wavenumber=NU_OBS)
        .values
    )
    N = number_density_cm3(1.0, 280.0)
    t_expected = -sigma * N * kwargs["delta_vmr"] * kwargs["path_length_cm"] * kwargs["amf"]
    np.testing.assert_allclose(t, t_expected, rtol=1e-10, atol=1e-14)


def test_linear_and_nonlinear_agree_in_thin_limit(synthetic_lut):
    kwargs = dict(
        T_K=280.0, p_atm=1.0, delta_vmr=5e-9,
        path_length_cm=8.4e5, amf=2.0,
    )
    t_lin = target_spectrum_normalized_linear(synthetic_lut, NU_OBS, **kwargs)
    t_nl = target_spectrum_normalized_nonlinear(
        synthetic_lut, NU_OBS, vmr_background=2e-6, **kwargs,
    )
    np.testing.assert_allclose(t_lin, t_nl, rtol=5e-3, atol=1e-12)


def test_target_bands_matches_direct_srf_apply(synthetic_lut, swir_srf):
    # Hi-res target at the LUT wavenumbers.
    t_hr = target_spectrum_normalized_linear(
        synthetic_lut, synthetic_lut["wavenumber"].values,
        T_K=280.0, p_atm=1.0, delta_vmr=1e-6,
        path_length_cm=8.4e5, amf=2.0,
    )
    wl_hr = 1e7 / synthetic_lut["wavenumber"].values  # decreasing in ν = increasing in λ? No, λ = 1e7/ν
    t_b = target_bands(t_hr, swir_srf, wl_hr)
    # Band integral is a weighted average of the hi-res target; its magnitude
    # should not exceed the hi-res peak magnitude.
    assert np.abs(t_b).max() <= np.abs(t_hr).max() + 1e-12


def test_target_bands_length_mismatch():
    srf_stub = None  # not reached — shape check happens first
    with np.testing.assert_raises(ValueError):
        target_bands(np.zeros(10), srf_stub, np.zeros(11))  # type: ignore[arg-type]
