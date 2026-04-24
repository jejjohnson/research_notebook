"""Target-spectrum construction for matched-filter retrievals.

The matched filter projects an observed pixel spectrum onto the expected
*plume signature*, normalised by the background covariance. That signature
— the "target spectrum" — is the radiance (or normalised radiance)
perturbation a reference plume enhancement would produce given the current
atmospheric state.

This module offers three flavours, all returning NumPy arrays shaped
``(n_channels,)`` where ``n_channels`` is either the LUT wavenumber count
(hyperspectral) or the number of SRF bands (multispectral) — see
:func:`target_bands` for the band-integrated variant.

- :func:`target_spectrum_normalized_nonlinear` — exact ``exp(-Δτ) - 1``.
  Use when ``Δτ`` can approach unity (strong-plume regime).
- :func:`target_spectrum_normalized_linear`    — first-order Maclaurin,
  ``t ≈ -σ·N·ΔVMR·L·AMF``. Classical linearised matched filter.
- :func:`target_bands`                          — integrate any hi-res
  target through an SRF to produce a band-space target.

All three are ported with simplifications from
``jej_vc_snippets/methane_retrieval/matched_filter_beerslaw.py`` — dropping
the duplicate "reverse order" Taylor/Maclaurin variants since they agree
with the standard-order linearisation at first order.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from plume_simulation.radtran.forward import (
    forward_maclaurin_normalized,
    forward_nonlinear_normalized,
)
from plume_simulation.radtran.srf import SpectralResponseFunction


def target_spectrum_normalized_nonlinear(
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
) -> np.ndarray:
    """Exact normalised-radiance target ``t(ν) = exp(-Δτ(ν)) − 1``.

    Negative for absorption. Magnitude tracks the cross-section profile and
    scales with ``ΔVMR``.

    ``vmr_background`` is accepted for API symmetry with the linearised
    form but is not used here: the *normalised* nonlinear model depends
    only on ``ΔVMR``. Keeping the kwarg means callers can swap forward
    models without editing the call site.
    """
    _ = vmr_background  # see docstring
    result = forward_nonlinear_normalized(
        ds,
        nu_obs,
        T_K=T_K,
        p_atm=p_atm,
        vmr_background=vmr_background,
        delta_vmr=delta_vmr,
        path_length_cm=path_length_cm,
        amf=amf,
        var=var,
    )
    return result.radiance - 1.0


def target_spectrum_normalized_linear(
    ds: xr.Dataset,
    nu_obs: np.ndarray,
    *,
    T_K: float,
    p_atm: float,
    delta_vmr: float,
    path_length_cm: float,
    amf: float,
    var: str = "absorption_cross_section",
) -> np.ndarray:
    """First-order linearised target ``t(ν) = -σ · N · ΔVMR · L · AMF``.

    Equivalent to ``target_spectrum_normalized_nonlinear`` in the thin-plume
    limit ``Δτ ≪ 1`` and considerably faster — no exponential per pixel.
    This is the canonical matched-filter target for weak retrievals.
    """
    result = forward_maclaurin_normalized(
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
    return result.radiance - 1.0


def target_bands(
    target_hr: np.ndarray,
    srf: SpectralResponseFunction,
    wavelengths_nm: np.ndarray,
) -> np.ndarray:
    """Band-integrate a hi-res target spectrum through an SRF.

    Parameters
    ----------
    target_hr : np.ndarray
        Target spectrum on the LUT wavenumber grid, shape ``(n_nu,)``.
    srf : SpectralResponseFunction
        Instrument SRF defined on ``srf.wavelengths_hr_nm``.
    wavelengths_nm : np.ndarray
        Wavelength grid ``1e7 / nu_obs`` [nm] at which ``target_hr`` is
        evaluated. Must be the same length as ``target_hr`` but may be in
        decreasing order (we sort internally).

    Returns
    -------
    target_b : np.ndarray
        Shape ``(n_bands,)``.
    """
    if target_hr.shape != wavelengths_nm.shape:
        raise ValueError(
            f"target_bands: `target_hr` {target_hr.shape} and "
            f"`wavelengths_nm` {wavelengths_nm.shape} must match."
        )
    sort_idx = np.argsort(wavelengths_nm)
    lam = np.asarray(wavelengths_nm)[sort_idx]
    t = np.asarray(target_hr)[sort_idx]
    t_on_srf_grid = np.interp(srf.wavelengths_hr_nm, lam, t, left=0.0, right=0.0)
    return srf.apply(t_on_srf_grid)
