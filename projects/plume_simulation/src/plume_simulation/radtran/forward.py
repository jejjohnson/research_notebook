"""Beer-Lambert forward models for atmospheric retrieval.

Three flavours, each returning radiance + Jacobian + transmittance:

- :func:`forward_nonlinear`  — exact Beer-Lambert ``L = L₀ · exp(-τ)``.
- :func:`forward_maclaurin`  — polynomial in ``VMR`` via Maclaurin of ``exp(-τ)``
  around ``VMR = 0``. Order 1 is the classical "linear" retrieval.
- :func:`forward_taylor`     — Taylor-linearised around a background state
  ``VMR_bg`` (the state used in 3D-/4D-Var inner loops).

Plus normalised variants that divide by the background radiance:

- :func:`forward_nonlinear_normalized` — ``L_norm = exp(-Δτ)``; cancels
  surface reflectance, solar irradiance, and common aerosol slope.
- :func:`forward_maclaurin_normalized` / :func:`forward_taylor_normalized`.

All functions take an ``xarray.Dataset`` LUT carrying an
``absorption_cross_section`` variable with dims ``(wavenumber, temperature,
pressure)`` — the output of
:func:`plume_simulation.hapi_lut.build_lut_dataset`.

Ported and adapted from
``jej_vc_snippets/methane_retrieval/lut_model_beers.py`` — the Jacobian is
now always returned alongside the radiance so the same functions can drive
variational retrievals (``H = dL/dVMR``) and matched-filter target-spectrum
construction (``t = H · ΔVMR``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from plume_simulation.radtran.config import number_density_cm3


@dataclass(frozen=True)
class ForwardResult:
    """Bundle of (radiance, Jacobian, transmittance) returned by the forward models.

    Using a frozen dataclass rather than a plain tuple gives the caller
    named access — ``result.jacobian`` — which matters because several
    functions in :mod:`plume_simulation.radtran.target` and
    :mod:`plume_simulation.radtran.matched_filter` consume only the
    Jacobian or only the transmittance.

    Attributes
    ----------
    radiance : np.ndarray
        Simulated radiance at each wavenumber, shape ``(n_nu,)``.
    jacobian : np.ndarray
        ``dL/dVMR`` evaluated at the supplied state, shape ``(n_nu,)``.
    transmittance : np.ndarray
        Atmospheric transmittance ``exp(-τ)`` (non-normalised variants) or
        ``exp(-Δτ)`` (normalised variants), shape ``(n_nu,)``.
    """

    radiance: np.ndarray
    jacobian: np.ndarray
    transmittance: np.ndarray


def _interp_sigma(
    ds: xr.Dataset,
    nu_obs: np.ndarray,
    T_K: float,
    p_atm: float,
    var: str,
) -> np.ndarray:
    """Interpolate σ(ν, T, P) from the LUT onto ``nu_obs``.

    Raises a ``KeyError`` if ``var`` is missing — the caller is expected to
    have passed the correct dataset.
    """
    if var not in ds:
        raise KeyError(
            f"forward model: variable {var!r} not in dataset "
            f"(have {list(ds.data_vars)})"
        )
    nu_da = xr.DataArray(np.asarray(nu_obs, dtype=float), dims=["obs_nu"])
    sigma = ds[var].interp(
        wavenumber=nu_da, temperature=T_K, pressure=p_atm, method="linear"
    )
    return np.asarray(sigma.values, dtype=float)


# ── Nonlinear (exact) Beer-Lambert ───────────────────────────────────────────


def forward_nonlinear(
    ds: xr.Dataset,
    nu_obs: np.ndarray,
    *,
    T_K: float,
    p_atm: float,
    vmr: float,
    path_length_cm: float,
    amf: float,
    surface_reflectance: float = 1.0,
    solar_irradiance: float = 1.0,
    var: str = "absorption_cross_section",
) -> ForwardResult:
    """Exact Beer-Lambert forward model.

    Radiance:    ``L(ν) = (F₀ R / π) · exp(-τ(ν, VMR))``
    Optical τ:   ``τ = σ · N_total · VMR · L · AMF``
    Jacobian:    ``dL/dVMR = -L · (dτ/dVMR)``, ``dτ/dVMR = σ · N_total · L · AMF``

    Defaults ``surface_reflectance = solar_irradiance = 1`` collapse the
    prefactor to ``1/π``, which is convenient for toy tests where the
    absolute radiance scale does not matter.
    """
    sigma = _interp_sigma(ds, nu_obs, T_K, p_atm, var)
    N_total = number_density_cm3(p_atm, T_K)
    tau = sigma * N_total * vmr * path_length_cm * amf
    transmittance = np.exp(-tau)
    L0 = solar_irradiance * surface_reflectance / np.pi
    radiance = L0 * transmittance
    dtau_dvmr = sigma * N_total * path_length_cm * amf
    jacobian = -radiance * dtau_dvmr
    return ForwardResult(radiance=radiance, jacobian=jacobian, transmittance=transmittance)


def forward_nonlinear_normalized(
    ds: xr.Dataset,
    nu_obs: np.ndarray,
    *,
    T_K: float,
    p_atm: float,
    vmr_background: float,
    delta_vmr: float,
    path_length_cm: float,
    amf: float,
    var: str = "absorption_cross_section",
) -> ForwardResult:
    """Exact Beer-Lambert *normalised* by the background radiance.

    ``L_norm = L(VMR_bg + ΔVMR) / L(VMR_bg) = exp(-Δτ)``, which cancels any
    multiplicative surface/solar/aerosol factors that appear equally in the
    plume and background pixels.

    Returns
    -------
    ForwardResult
        ``radiance`` is the normalised transmittance ``exp(-Δτ)``;
        ``transmittance`` is the same quantity (kept for API symmetry with
        :func:`forward_nonlinear`); ``jacobian`` is
        ``d(L_norm)/d(ΔVMR) = -exp(-Δτ) · (σ · N · L · AMF)``.
    """
    sigma = _interp_sigma(ds, nu_obs, T_K, p_atm, var)
    N_total = number_density_cm3(p_atm, T_K)
    dtau_d_dvmr = sigma * N_total * path_length_cm * amf
    delta_tau = dtau_d_dvmr * delta_vmr
    L_norm = np.exp(-delta_tau)
    jacobian = -L_norm * dtau_d_dvmr
    return ForwardResult(radiance=L_norm, jacobian=jacobian, transmittance=L_norm)


# ── Maclaurin (expansion around VMR = 0) ─────────────────────────────────────


def forward_maclaurin(
    ds: xr.Dataset,
    nu_obs: np.ndarray,
    *,
    T_K: float,
    p_atm: float,
    vmr: float,
    path_length_cm: float,
    amf: float,
    surface_reflectance: float = 1.0,
    solar_irradiance: float = 1.0,
    order: int = 1,
    var: str = "absorption_cross_section",
) -> ForwardResult:
    """Maclaurin-series forward model: expand ``exp(-τ(VMR))`` around VMR = 0.

    With ``a = σ · N · L · AMF`` so ``τ = a · VMR``:

    - order 1: ``T ≈ 1 − a·VMR`` (linear in VMR).
    - order 2: ``T ≈ 1 − a·VMR + ½ (a·VMR)²``.
    - order 3: ``T ≈ 1 − a·VMR + ½ (a·VMR)² − ⅙ (a·VMR)³``.

    Accurate when the total optical depth ``a·VMR`` is ≪ 1 — the regime of
    classical linear retrievals for thin absorbers.
    """
    if order not in (1, 2, 3):
        raise ValueError(f"forward_maclaurin: `order` must be 1, 2 or 3 (got {order!r})")
    sigma = _interp_sigma(ds, nu_obs, T_K, p_atm, var)
    N_total = number_density_cm3(p_atm, T_K)
    a = sigma * N_total * path_length_cm * amf
    a_vmr = a * vmr

    if order == 1:
        transmittance = 1.0 - a_vmr
        dtrans_dvmr = -a
    elif order == 2:
        transmittance = 1.0 - a_vmr + 0.5 * a_vmr**2
        dtrans_dvmr = -a + (a**2) * vmr
    else:  # order == 3
        transmittance = 1.0 - a_vmr + 0.5 * a_vmr**2 - (1.0 / 6.0) * a_vmr**3
        dtrans_dvmr = -a + (a**2) * vmr - 0.5 * (a**3) * vmr**2

    L0 = solar_irradiance * surface_reflectance / np.pi
    radiance = L0 * transmittance
    jacobian = L0 * dtrans_dvmr
    return ForwardResult(radiance=radiance, jacobian=jacobian, transmittance=transmittance)


def forward_maclaurin_normalized(
    ds: xr.Dataset,
    nu_obs: np.ndarray,
    *,
    T_K: float,
    p_atm: float,
    delta_vmr: float,
    path_length_cm: float,
    amf: float,
    order: int = 1,
    var: str = "absorption_cross_section",
) -> ForwardResult:
    """Maclaurin expansion of ``exp(-Δτ)`` around ``ΔVMR = 0``.

    At order 1 this reduces to ``L_norm ≈ 1 − σ · N · ΔVMR · L · AMF`` — the
    canonical linearised retrieval signal.
    """
    if order not in (1, 2, 3):
        raise ValueError("forward_maclaurin_normalized: `order` must be 1, 2 or 3.")
    sigma = _interp_sigma(ds, nu_obs, T_K, p_atm, var)
    N_total = number_density_cm3(p_atm, T_K)
    a = sigma * N_total * path_length_cm * amf
    a_dvmr = a * delta_vmr
    if order == 1:
        L_norm = 1.0 - a_dvmr
        jac = -a
    elif order == 2:
        L_norm = 1.0 - a_dvmr + 0.5 * a_dvmr**2
        jac = -a + (a**2) * delta_vmr
    else:
        L_norm = 1.0 - a_dvmr + 0.5 * a_dvmr**2 - (1.0 / 6.0) * a_dvmr**3
        jac = -a + (a**2) * delta_vmr - 0.5 * (a**3) * delta_vmr**2
    return ForwardResult(radiance=L_norm, jacobian=jac, transmittance=L_norm)


# ── Taylor (expansion around VMR_background) ────────────────────────────────


def forward_taylor(
    ds: xr.Dataset,
    nu_obs: np.ndarray,
    *,
    T_K: float,
    p_atm: float,
    vmr: float,
    vmr_background: float,
    path_length_cm: float,
    amf: float,
    surface_reflectance: float = 1.0,
    solar_irradiance: float = 1.0,
    var: str = "absorption_cross_section",
) -> ForwardResult:
    """First-order Taylor expansion around ``vmr_background``.

    ``L(VMR) ≈ L_b + H · (VMR − VMR_b)`` with ``H = dL/dVMR |_{VMR_b}``. This
    is the linearisation used in the inner loops of 3D-/4D-Var.
    """
    sigma = _interp_sigma(ds, nu_obs, T_K, p_atm, var)
    N_total = number_density_cm3(p_atm, T_K)
    dtau_dvmr = sigma * N_total * path_length_cm * amf
    tau_b = dtau_dvmr * vmr_background
    L0 = solar_irradiance * surface_reflectance / np.pi
    L_b = L0 * np.exp(-tau_b)
    H = -L_b * dtau_dvmr
    radiance = L_b + H * (vmr - vmr_background)
    transmittance = np.exp(-tau_b) + (-dtau_dvmr * np.exp(-tau_b)) * (vmr - vmr_background)
    return ForwardResult(radiance=radiance, jacobian=H, transmittance=transmittance)


def forward_taylor_normalized(
    ds: xr.Dataset,
    nu_obs: np.ndarray,
    *,
    T_K: float,
    p_atm: float,
    delta_vmr: float,
    path_length_cm: float,
    amf: float,
    var: str = "absorption_cross_section",
) -> ForwardResult:
    """Taylor expansion of ``L_norm`` around ``ΔVMR = 0``.

    Since ``L_norm(0) = 1`` and ``dL_norm/dΔVMR(0) = -σ · N · L · AMF``,
    the first-order Taylor expansion coincides with the order-1 Maclaurin
    expansion. Kept as a named function for API symmetry with
    :func:`forward_taylor`.
    """
    return forward_maclaurin_normalized(
        ds,
        nu_obs,
        T_K=T_K,
        p_atm=p_atm,
        delta_vmr=delta_vmr,
        path_length_cm=path_length_cm,
        amf=amf,
        order=1,
        var=var,
    )
