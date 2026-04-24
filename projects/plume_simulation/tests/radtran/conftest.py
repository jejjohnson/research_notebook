"""Shared fixtures for the radtran tests.

Keeps every test file HAPI-free by synthesising a small absorption-cross-section
dataset with the same schema as :func:`plume_simulation.hapi_lut.build_lut_dataset`
— a Gaussian absorption line at 4300 cm⁻¹ plus a weaker line at 4200 cm⁻¹,
shaped crudely like the CH₄ SWIR window.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def synthetic_lut() -> xr.Dataset:
    """Tiny σ(ν, T, P) LUT: 2 Gaussian lines on a coarse ν/T/P grid."""
    nu = np.linspace(4000.0, 4500.0, 201)  # cm^-1 (matches hapi_lut schema)
    wl = 1e7 / nu  # nm
    T_grid = np.array([220.0, 260.0, 300.0])
    P_grid = np.array([0.5, 1.0])

    centres = np.array([4200.0, 4300.0])
    strengths_T = np.array([1e-21, 3e-21])  # at 300 K
    widths = np.array([3.0, 5.0])  # cm^-1 HWHM-ish

    sigma = np.zeros((nu.size, T_grid.size, P_grid.size), dtype=float)
    for i_T, T in enumerate(T_grid):
        # Mild T dependence: strength ∝ exp(-ΔE/kT) with a toy ΔE.
        temp_fac = np.exp(-(300.0 - T) / 150.0)
        for i_P, p in enumerate(P_grid):
            # Pressure-broadened (linear FWHM ∝ p).
            pres_fac = p
            for c, s, w in zip(centres, strengths_T, widths, strict=True):
                sigma[:, i_T, i_P] += (
                    s
                    * temp_fac
                    * np.exp(-0.5 * ((nu - c) / (w * pres_fac)) ** 2)
                )

    ds = xr.Dataset(
        data_vars={
            "absorption_cross_section": (
                ["wavenumber", "temperature", "pressure"], sigma,
                {"units": "cm^2 / molecule"},
            ),
        },
        coords={
            "wavenumber": (["wavenumber"], nu, {"units": "cm^-1"}),
            "wavelength": (["wavenumber"], wl, {"units": "nm"}),
            "temperature": (["temperature"], T_grid, {"units": "K"}),
            "pressure": (["pressure"], P_grid, {"units": "atm"}),
        },
        attrs={"molecule": "TOY_CH4"},
    )
    return ds


@pytest.fixture
def swir_srf():
    """Two-band Gaussian SRF (B11-like @ 1610 nm, B12-like @ 2190 nm)."""
    from plume_simulation.radtran.srf import SpectralResponseFunction

    # HR grid covering both bands with a comfortable margin.
    wl = np.linspace(1400.0, 2500.0, 4001)
    return SpectralResponseFunction(
        wavelengths_hr_nm=wl,
        band_centers_nm=np.array([1610.0, 2190.0]),
        band_widths_nm=np.array([90.0, 180.0]),
        band_names=("B11", "B12"),
        srf_type="gaussian",
    )
