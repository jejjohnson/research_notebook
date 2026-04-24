"""JAX observation operator for the methane VMR retrieval.

The full forward map factors as

    x  ──┐
         │ ΔVMR = x − x_ref
         ▼
        Δτ(ν, i, j) = a(ν) · ΔVMR(i, j)         a(ν) = σ(ν, T, P) · N · ΔX · AMF
         │
         ▼
        L_hr(i, j, ν) = exp(−Δτ)                 (or 1 − Δτ for linear)
         │
         ▼ PSF (per channel)
         ▼ GSD (per channel)
         ▼ SRF (band integration)
        y_obs(I, J, b)

Everything below the LUT lookup is JAX, so

- ``forward`` runs end-to-end under :func:`jax.jit`,
- ``jax.grad(loss∘forward)`` produces the **discrete adjoint** for free,
- ``jax.jvp(forward, …)`` produces the tangent-linear,

without us writing or maintaining any explicit adjoint. The reason
``radtran/instrument.py`` *does* expose hand-rolled ``adjoint`` methods is to
let unit tests verify the inner-product identity independently — JAX's autodiff
inherits the correctness, not the other way around.

The factory :meth:`RadianceObservationModel.from_lut` precomputes ``a(ν)`` once
from the HAPI cross-section LUT so the per-iteration cost in 3D-Var is dominated
by the cube-shaped operations (PSF, GSD, SRF).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import numpy as np

from plume_simulation.radtran.config import number_density_cm3
from plume_simulation.radtran.instrument import (
    GroundSamplingDistance,
    PointSpreadFunction,
)


if TYPE_CHECKING:
    import xarray as xr

    from plume_simulation.radtran.srf import SpectralResponseFunction


@dataclass(frozen=True)
class RadianceObservationModel:
    """Bundle of precomputed pieces that defines the forward operator H.

    Attributes
    ----------
    absorption_coeff_hr : jnp.ndarray
        Per-wavenumber absorption coefficient ``a(ν) = σ(ν,T,P) · N · ΔX · AMF``,
        shape ``(n_lambda,)``. Captures everything *except* the VMR field, so the
        forward map is just ``L_hr(i,j,ν) = exp(-a(ν) · ΔVMR(i,j))``.
    srf_matrix : jnp.ndarray
        L1-normalised SRF matrix, shape ``(n_bands, n_lambda)``.
    vmr_reference : float
        Background VMR ``x_ref`` around which radiances are normalised. Setting
        ``vmr_reference = 0`` recovers absolute Beer-Lambert; setting it to the
        scene's true background produces the differential signal that matched
        filter retrievals consume.
    psf, gsd : optional
        JAX-side instrument operators from :mod:`plume_simulation.radtran.instrument`.
        ``None`` skips that stage — convenient for hyperspectral tests where
        the spectral cube is already at sensor resolution.
    """

    absorption_coeff_hr: jax.Array
    srf_matrix: jax.Array
    vmr_reference: float = 0.0
    psf: PointSpreadFunction | None = None
    gsd: GroundSamplingDistance | None = None
    n_lambda: int = field(init=False)
    n_bands: int = field(init=False)

    def __post_init__(self) -> None:
        a = jnp.asarray(self.absorption_coeff_hr)
        s = jnp.asarray(self.srf_matrix)
        if a.ndim != 1:
            raise ValueError(
                f"RadianceObservationModel: absorption_coeff_hr must be 1-D, "
                f"got shape {a.shape}."
            )
        if s.ndim != 2 or s.shape[1] != a.size:
            raise ValueError(
                f"RadianceObservationModel: srf_matrix shape {s.shape} does not "
                f"align with absorption_coeff_hr (n_lambda={a.size})."
            )
        # Frozen dataclass: bypass __setattr__ for derived fields.
        object.__setattr__(self, "absorption_coeff_hr", a)
        object.__setattr__(self, "srf_matrix", s)
        object.__setattr__(self, "n_lambda", int(a.size))
        object.__setattr__(self, "n_bands", int(s.shape[0]))

    # ── factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_lut(
        cls,
        lut: "xr.Dataset",
        *,
        srf: "SpectralResponseFunction",
        T_K: float,
        p_atm: float,
        path_length_cm: float,
        amf: float,
        vmr_reference: float = 0.0,
        psf: PointSpreadFunction | None = None,
        gsd: GroundSamplingDistance | None = None,
        var: str = "absorption_cross_section",
    ) -> "RadianceObservationModel":
        """Build the operator from a HAPI σ(ν,T,P) LUT and an SRF.

        Pre-collapses ``σ · N · ΔX · AMF`` into a single ``a(ν)`` vector so the
        forward call only multiplies by ``ΔVMR`` per pixel.

        The HR spectral grid is **taken from the SRF** (``srf.wavelengths_hr_nm``)
        and converted to wavenumbers for the σ lookup. This is the grid the SRF
        matrix's columns are indexed by, so the subsequent
        ``einsum("bl, ijl -> ijb")`` band integration contracts aligned spectral
        bins by construction. Earlier revisions accepted a separate ``nu_obs``
        argument; when callers passed a wavenumber-ordered grid while the SRF
        was wavelength-ordered (common whenever ``wl = 1e7/ν`` was sorted),
        the einsum silently mixed non-corresponding bins — self-consistent for
        twin retrievals but physically wrong for anything cross-validated.
        """
        if var not in lut:
            raise KeyError(
                f"RadianceObservationModel.from_lut: variable {var!r} not in "
                f"dataset (have {list(lut.data_vars)})."
            )
        import xarray as xr_mod

        # Source-of-truth HR grid: the SRF's own wavelength axis (ascending
        # by wavelength = descending by wavenumber). Convert to cm⁻¹ for σ
        # interpolation; keep the wavelength ordering for the output array so
        # axis l aligns with srf.matrix[:, l].
        wl_hr_nm = np.asarray(srf.wavelengths_hr_nm, dtype=float)
        nu_hr = 1e7 / wl_hr_nm
        nu_da = xr_mod.DataArray(nu_hr, dims=["obs_nu"])
        sigma = lut[var].interp(
            wavenumber=nu_da, temperature=T_K, pressure=p_atm, method="linear"
        )
        sigma_np = np.asarray(sigma.values, dtype=float)
        N_total = number_density_cm3(p_atm, T_K)
        a = sigma_np * N_total * path_length_cm * amf
        return cls(
            absorption_coeff_hr=jnp.asarray(a),
            srf_matrix=jnp.asarray(srf.matrix),
            vmr_reference=float(vmr_reference),
            psf=psf,
            gsd=gsd,
        )

    # ── forward operator ────────────────────────────────────────────────────

    def forward(self, vmr_field: jax.Array, *, linear: bool = False) -> jax.Array:
        """Map a 2-D VMR field to a band-integrated obs cube.

        Parameters
        ----------
        vmr_field : Array
            Per-pixel VMR, shape ``(ny_hr, nx_hr)``.
        linear : bool
            If True, replace ``exp(-Δτ)`` by its Maclaurin-1 approximation
            ``1 - Δτ``. The tangent-linear of the nonlinear forward at
            ``vmr_field == vmr_reference`` coincides with this; the matched
            filter is exactly this case with ``δx`` propagated through.

        Returns
        -------
        Array
            Sensor-space radiance cube, shape ``(ny_lr, nx_lr, n_bands)``
            (``ny_lr = ny_hr // f``, ``nx_lr = nx_hr // f`` if a GSD is set).
        """
        if vmr_field.ndim != 2:
            raise ValueError(
                f"RadianceObservationModel.forward: vmr_field must be 2-D, got shape {vmr_field.shape}."
            )
        delta_vmr = vmr_field - self.vmr_reference
        # Outer product (ny, nx) ⊗ (n_lambda,) → (ny, nx, n_lambda).
        delta_tau = delta_vmr[..., None] * self.absorption_coeff_hr
        # `linear` is a Python bool (static); branch on it directly so JAX
        # doesn't trace the unused branch — `exp(huge δτ)` would otherwise
        # overflow on stress-tests like the adjoint inner-product check.
        L_hr = (1.0 - delta_tau) if linear else jnp.exp(-delta_tau)
        if self.psf is not None:
            L_hr = self.psf.apply(L_hr)
        if self.gsd is not None:
            L_hr = self.gsd.apply(L_hr)
        # Band integration: contract last (HR-λ) axis with SRF.
        return jnp.einsum("bl, ijl -> ijb", self.srf_matrix, L_hr)

    def make_forward(self, *, linear: bool = False) -> Callable[[jax.Array], jax.Array]:
        """Return a closure that captures the model so it can be ``jit``-compiled.

        ``jax.jit(model.forward)`` works in JAX ≥ 0.4 because frozen dataclasses
        without ``register_pytree_node_class`` are hashable and become static
        arguments — but explicit closures are easier to compose with
        ``jax.value_and_grad`` and friends, and they work uniformly across
        JAX versions.
        """
        linear_flag = bool(linear)

        def _fwd(vmr_field: jax.Array) -> jax.Array:
            return self.forward(vmr_field, linear=linear_flag)

        return _fwd
