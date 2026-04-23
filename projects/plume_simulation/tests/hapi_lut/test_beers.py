"""Tests for ``plume_simulation.hapi_lut.beers`` (pure numpy / xarray)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from plume_simulation.hapi_lut import beers


def _toy_lut_dataset() -> xr.Dataset:
    """Minimal LUT with σ = 1e-23 · (T/200) · p (bilinear; realistic magnitude)."""
    nu = np.linspace(4000.0, 4010.0, 5)
    T = np.array([200.0, 300.0])
    P = np.array([0.2, 1.0])
    sigma = np.zeros((nu.size, T.size, P.size))
    for i_T, t in enumerate(T):
        for i_P, p in enumerate(P):
            sigma[:, i_T, i_P] = 1e-23 * (t / 200.0) * p
    return xr.Dataset(
        data_vars={"absorption_cross_section": (["wavenumber", "temperature", "pressure"], sigma)},
        coords={
            "wavenumber": nu,
            "wavelength": ("wavenumber", 1e7 / nu),
            "temperature": T,
            "pressure": P,
        },
    )


def test_number_density_matches_loschmidt_at_stp():
    n = beers.number_density(p_atm=1.0, T_K=273.15)
    assert n == pytest.approx(2.6867811e19, rel=5e-4)


def test_amf_nadir_is_two():
    assert beers.air_mass_factor(sza_deg=0.0, vza_deg=0.0) == pytest.approx(2.0)


def test_amf_grows_with_zenith_angle():
    a = beers.air_mass_factor(sza_deg=60.0, vza_deg=0.0)
    b = beers.air_mass_factor(sza_deg=60.0, vza_deg=60.0)
    assert a > 2.0 and b > a


def test_interpolate_cross_section_hits_bilinear_midpoint():
    ds = _toy_lut_dataset()
    sigma = beers.interpolate_cross_section(ds, T_K=250.0, p_atm=0.6)
    # Bilinear at (250, 0.6): σ = 1e-23 · (250/200) · 0.6 = 7.5e-24 (exact — σ is bilinear).
    assert float(sigma.isel(wavenumber=0)) == pytest.approx(7.5e-24, rel=1e-12)


def test_transmittance_of_zero_absorption_is_one():
    tau = beers.transmittance(np.zeros(5), l_vert_cm=1e5, sza_deg=0.0, vza_deg=0.0)
    assert np.all(tau == 1.0)


def test_beers_law_from_lut_is_monotone_in_vmr():
    ds = _toy_lut_dataset()
    kwargs = dict(T_K=250.0, p_atm=0.5, l_vert_cm=1e5, sza_deg=30.0, vza_deg=20.0)
    tau_lo = beers.beers_law_from_lut(ds, vmr=1e-6, **kwargs)
    tau_hi = beers.beers_law_from_lut(ds, vmr=2e-6, **kwargs)
    # Larger VMR → more absorption → smaller transmittance at every ν.
    assert bool(np.all(tau_hi.values <= tau_lo.values + 1e-15))
    assert float(tau_lo.min()) < 1.0


def test_plume_ratio_equals_total_over_background():
    ds = _toy_lut_dataset()
    kwargs = dict(T_K=250.0, p_atm=0.5, l_vert_cm=1e5, sza_deg=30.0, vza_deg=20.0)
    tau_bg = beers.beers_law_from_lut(ds, vmr=1e-6, **kwargs)
    tau_tot = beers.beers_law_from_lut(ds, vmr=1.5e-6, **kwargs)
    ratio = beers.plume_ratio_spectrum(
        ds, vmr_background=1e-6, vmr_total=1.5e-6, **kwargs
    )
    np.testing.assert_allclose(ratio.values, (tau_tot / tau_bg).values, rtol=1e-12)


def test_plume_ratio_is_at_most_one_for_positive_enhancement():
    ds = _toy_lut_dataset()
    ratio = beers.plume_ratio_spectrum(
        ds,
        vmr_background=1e-6,
        vmr_total=1.2e-6,
        T_K=250.0,
        p_atm=0.5,
        l_vert_cm=1e5,
        sza_deg=30.0,
        vza_deg=20.0,
    )
    assert float(ratio.max()) <= 1.0 + 1e-15
