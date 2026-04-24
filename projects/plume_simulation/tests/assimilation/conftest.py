"""Shared fixtures for the assimilation tests.

Defines a tiny synthetic ``σ(ν, T, P)`` LUT (same schema as
``hapi_lut.build_lut_dataset``) and an obs-operator factory that wires the
LUT through the SRF + (optional) PSF/GSD chain. Every assimilation test
starts from the same physical setup, so cost / solver / diagnostic suites
can't drift apart.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from plume_simulation.assimilation.obs_operator import RadianceObservationModel
from plume_simulation.radtran.instrument import (
    GroundSamplingDistance,
    PointSpreadFunction,
)
from plume_simulation.radtran.srf import SpectralResponseFunction


@pytest.fixture
def synthetic_lut() -> xr.Dataset:
    """Tiny σ(ν, T, P) LUT: 2 Gaussian lines on a coarse ν/T/P grid.

    Mirrors the radtran tests' fixture so assimilation tests don't need to
    cross-import siblings.
    """
    nu = np.linspace(4000.0, 4500.0, 201)
    T_grid = np.array([220.0, 260.0, 300.0])
    P_grid = np.array([0.5, 1.0])
    centres = np.array([4200.0, 4300.0])
    strengths_T = np.array([1e-21, 3e-21])
    widths = np.array([3.0, 5.0])

    sigma = np.zeros((nu.size, T_grid.size, P_grid.size), dtype=float)
    for i_T, T in enumerate(T_grid):
        temp_fac = np.exp(-(300.0 - T) / 150.0)
        for i_P, p in enumerate(P_grid):
            for c, s, w in zip(centres, strengths_T, widths, strict=True):
                sigma[:, i_T, i_P] += s * temp_fac * np.exp(
                    -0.5 * ((nu - c) / (w * p)) ** 2
                )

    return xr.Dataset(
        data_vars={
            "absorption_cross_section": (
                ["wavenumber", "temperature", "pressure"], sigma,
            ),
        },
        coords={
            "wavenumber": (["wavenumber"], nu),
            "temperature": (["temperature"], T_grid),
            "pressure": (["pressure"], P_grid),
        },
    )


@pytest.fixture
def hyperspectral_srf(synthetic_lut):
    """One band per wavenumber sample — SRF is essentially the identity per ν."""
    nu = synthetic_lut["wavenumber"].values
    wl = 1e7 / nu  # nm
    sort = np.argsort(wl)
    return SpectralResponseFunction(
        wavelengths_hr_nm=wl[sort],
        band_centers_nm=wl[sort],
        band_widths_nm=np.full(wl.size, 5.0),
        band_names=tuple(f"c{i}" for i in range(wl.size)),
        srf_type="gaussian",
    )


@pytest.fixture
def obs_model_no_optics(synthetic_lut, hyperspectral_srf):
    """Obs operator with no PSF / GSD — fastest path for cost+grad tests."""
    return RadianceObservationModel.from_lut(
        synthetic_lut,
        srf=hyperspectral_srf,
        T_K=280.0,
        p_atm=1.0,
        path_length_cm=8.4e5,
        amf=2.0,
        vmr_reference=0.0,
    )


@pytest.fixture
def obs_model_with_psf(synthetic_lut, hyperspectral_srf):
    """Obs operator with a small Gaussian PSF — used by the dual==primal test."""
    return RadianceObservationModel.from_lut(
        synthetic_lut,
        srf=hyperspectral_srf,
        T_K=280.0,
        p_atm=1.0,
        path_length_cm=8.4e5,
        amf=2.0,
        vmr_reference=0.0,
        psf=PointSpreadFunction.gaussian(fwhm_pixels=1.5, kernel_size=5),
    )


@pytest.fixture
def obs_model_with_gsd(synthetic_lut, hyperspectral_srf):
    """Obs operator with a 2× GSD downsample for the shape-changing test."""
    return RadianceObservationModel.from_lut(
        synthetic_lut,
        srf=hyperspectral_srf,
        T_K=280.0,
        p_atm=1.0,
        path_length_cm=8.4e5,
        amf=2.0,
        vmr_reference=0.0,
        gsd=GroundSamplingDistance(downsample_factor=2),
    )
