"""LUT interpolation + Beer-Lambert transmittance for one-shot forward models.

This module closes the loop between an offline-generated σ(ν, T, P) LUT and
the naïve single-layer radiative-transfer model used in the tutorial
notebooks. The physics is:

    N_total = p / (k_B T)              [molecules/cm³]  (ideal gas)
    N_gas   = N_total · VMR_gas        [molecules/cm³]
    α(ν)    = σ(ν; T, p) · N_gas       [1/cm]
    AMF     = 1/cos(SZA) + 1/cos(VZA)  [plane-parallel, two-way]
    τ(ν)    = exp[ -α(ν) · L_vert · AMF ]   [Beer-Lambert]

For plume-enhancement retrievals the differential form is

    τ_ratio(ν) = τ_total(ν) / τ_background(ν)
               = exp[ -σ(ν; T, p) · N_total · L_vert · AMF · (VMR_total - VMR_bg) ]

— any factor common to the "plume" and "background" pixel (solar spectrum,
broadband albedo, aerosol) cancels, leaving the narrow-band gas signature.

Units: the LUT is in HAPI's HITRAN convention (σ in cm², p in atm). We work
in cgs throughout — pressures are converted to Pa only to evaluate number
density via the ideal-gas law with ``k_B = 1.380649e-23 J/K``.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

# Physical constants — cgs-compatible forms.
BOLTZMANN_J_PER_K: float = 1.380649e-23  # [J/K]
ATM_TO_PA: float = 101325.0  # [Pa/atm]
CM3_PER_M3: float = 1e6


def interpolate_cross_section(
    ds: xr.Dataset,
    T_K: float,
    p_atm: float,
    *,
    var: str = "absorption_cross_section",
) -> xr.DataArray:
    """Return σ(ν) at a given (T, p) via bilinear LUT interpolation.

    The LUT dataset must carry ``temperature`` [K] and ``pressure`` [atm]
    coords as in ``build_lut_dataset``. No extrapolation — callers should
    clip to the LUT range before interpolating.
    """
    if var not in ds:
        raise KeyError(f"variable {var!r} not in dataset (have {list(ds.data_vars)})")
    return ds[var].interp(temperature=T_K, pressure=p_atm)


def number_density(p_atm: float, T_K: float) -> float:
    """Return total number density [molecules / cm³] from the ideal-gas law."""
    p_pa = p_atm * ATM_TO_PA
    n_per_m3 = p_pa / (BOLTZMANN_J_PER_K * T_K)
    return n_per_m3 / CM3_PER_M3


def air_mass_factor(sza_deg: float, vza_deg: float) -> float:
    """Plane-parallel two-way AMF for sun → surface → sensor.

    Breaks down past ~75° zenith; fine for the tutorial regime.
    """
    sza = np.deg2rad(sza_deg)
    vza = np.deg2rad(vza_deg)
    return 1.0 / np.cos(sza) + 1.0 / np.cos(vza)


def absorption_coefficient(
    sigma: xr.DataArray | np.ndarray,
    *,
    vmr: float,
    p_atm: float,
    T_K: float,
) -> xr.DataArray | np.ndarray:
    """α(ν) = σ(ν) · N_gas [1/cm]."""
    n_total = number_density(p_atm, T_K)
    return sigma * (n_total * vmr)


def transmittance(
    alpha: xr.DataArray | np.ndarray,
    *,
    l_vert_cm: float,
    sza_deg: float,
    vza_deg: float,
) -> xr.DataArray | np.ndarray:
    """τ(ν) = exp(-α · L_vert · AMF)."""
    amf = air_mass_factor(sza_deg, vza_deg)
    return np.exp(-alpha * l_vert_cm * amf)


def beers_law_from_lut(
    ds: xr.Dataset,
    *,
    vmr: float,
    T_K: float,
    p_atm: float,
    l_vert_cm: float,
    sza_deg: float,
    vza_deg: float,
    var: str = "absorption_cross_section",
) -> xr.DataArray:
    """End-to-end: σ-from-LUT → α → τ(ν) for a single homogeneous layer.

    Returns an ``xarray.DataArray`` carrying the ``wavenumber`` + ``wavelength``
    coords so callers can plot τ(λ) without re-introducing units.
    """
    sigma = interpolate_cross_section(ds, T_K=T_K, p_atm=p_atm, var=var)
    alpha = absorption_coefficient(sigma, vmr=vmr, p_atm=p_atm, T_K=T_K)
    tau = transmittance(alpha, l_vert_cm=l_vert_cm, sza_deg=sza_deg, vza_deg=vza_deg)
    return tau.assign_attrs(
        description="Single-layer Beer-Lambert transmittance from LUT",
        vmr=vmr,
        T_K=T_K,
        p_atm=p_atm,
        l_vert_cm=l_vert_cm,
        sza_deg=sza_deg,
        vza_deg=vza_deg,
    )


def plume_ratio_spectrum(
    ds: xr.Dataset,
    *,
    vmr_background: float,
    vmr_total: float,
    T_K: float,
    p_atm: float,
    l_vert_cm: float,
    sza_deg: float,
    vza_deg: float,
    var: str = "absorption_cross_section",
) -> xr.DataArray:
    """Differential-Beer-Lambert ratio τ_total(ν) / τ_bg(ν).

    Callers typically pass ``vmr_total = vmr_background * (1 + enhancement)``
    where ``enhancement`` is the fitting parameter in the plume retrieval.
    """
    tau_bg = beers_law_from_lut(
        ds, vmr=vmr_background, T_K=T_K, p_atm=p_atm, l_vert_cm=l_vert_cm,
        sza_deg=sza_deg, vza_deg=vza_deg, var=var,
    )
    tau_total = beers_law_from_lut(
        ds, vmr=vmr_total, T_K=T_K, p_atm=p_atm, l_vert_cm=l_vert_cm,
        sza_deg=sza_deg, vza_deg=vza_deg, var=var,
    )
    ratio = tau_total / tau_bg
    return ratio.assign_attrs(
        description="Plume/background transmittance ratio (differential Beer-Lambert)",
        vmr_background=vmr_background,
        vmr_total=vmr_total,
    )
