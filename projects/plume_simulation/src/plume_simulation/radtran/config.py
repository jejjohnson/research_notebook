"""Observation-geometry and instrument configuration for ``radtran``.

The forward models in :mod:`plume_simulation.radtran.forward` need three
groups of inputs: atmospheric state (``T``, ``P``, background VMR), a
gas-species LUT (σ(ν, T, P), from :mod:`plume_simulation.hapi_lut`), and
the viewing geometry and instrument response.

This module factors the latter two into frozen dataclasses:

- :class:`ObservationGeometry` — solar/viewing zenith angles, vertical path
  length, optional AMF override.
- :class:`InstrumentSpec` — band centres and widths for a synthetic
  multispectral instrument. Real instruments should build a
  :class:`~plume_simulation.radtran.srf.SpectralResponseFunction` from
  measured SRFs; ``InstrumentSpec`` is a convenient constructor for the
  tutorial notebooks.

``BOLTZMANN_J_PER_K`` and ``ATM_TO_PA`` mirror the constants in
:mod:`plume_simulation.hapi_lut.beers` so the two forward-model stacks agree
on the ideal-gas number density computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid circular import at module load
    from plume_simulation.radtran.srf import SpectralResponseFunction


# cgs-consistent constants, mirroring hapi_lut.beers for cross-module agreement.
BOLTZMANN_J_PER_K: float = 1.380649e-23  # [J/K]
ATM_TO_PA: float = 101325.0  # [Pa/atm]
CM3_PER_M3: float = 1e6


@dataclass(frozen=True)
class ObservationGeometry:
    """Viewing geometry for a single pixel.

    Attributes
    ----------
    sza_deg : float
        Solar zenith angle [degrees], 0 (overhead sun) to ~90 (grazing).
    vza_deg : float
        Viewing zenith angle [degrees], 0 (nadir) to ~90 (limb).
    path_length_cm : float
        Vertical atmospheric path [cm] traversed by the line of sight before
        AMF correction. For a single-slab column over the full atmosphere,
        a common choice is ``L = p_surf / (ρ · g) ≈ 8.4 km = 8.4e5 cm``
        (scale height of dry air at 1 atm, 288 K).
    amf : float, optional
        Air-mass factor. If ``None`` the plane-parallel two-way form
        ``1/cos(SZA) + 1/cos(VZA)`` is used.

    Notes
    -----
    The plane-parallel AMF breaks down past ~75° zenith; beyond that a
    spherical correction (Chapman function) is needed. For the
    tutorial regime (SZA, VZA < 60°) the plane-parallel form is within a
    few percent of the exact value.
    """

    sza_deg: float = 30.0
    vza_deg: float = 0.0
    path_length_cm: float = 8.4e5
    amf: float | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.sza_deg < 90.0):
            raise ValueError(
                f"ObservationGeometry: `sza_deg` must be in [0, 90) "
                f"(got {self.sza_deg!r})"
            )
        if not (0.0 <= self.vza_deg < 90.0):
            raise ValueError(
                f"ObservationGeometry: `vza_deg` must be in [0, 90) "
                f"(got {self.vza_deg!r})"
            )
        if not (self.path_length_cm > 0.0):
            raise ValueError(
                f"ObservationGeometry: `path_length_cm` must be > 0 "
                f"(got {self.path_length_cm!r})"
            )
        if self.amf is not None and not (self.amf > 0.0):
            raise ValueError(
                f"ObservationGeometry: `amf` must be > 0 (got {self.amf!r})"
            )

    @property
    def air_mass_factor(self) -> float:
        """Plane-parallel two-way AMF, or the explicit override if provided."""
        if self.amf is not None:
            return float(self.amf)
        sza = np.deg2rad(self.sza_deg)
        vza = np.deg2rad(self.vza_deg)
        return float(1.0 / np.cos(sza) + 1.0 / np.cos(vza))


@dataclass(frozen=True)
class InstrumentSpec:
    """Multispectral instrument description.

    Attributes
    ----------
    name : str
        Free-text instrument identifier for dataset metadata.
    band_centers_nm : np.ndarray
        Band centre wavelengths [nm], shape ``(n_bands,)``.
    band_widths_nm : np.ndarray
        Full-width at half-maximum of each band [nm], shape ``(n_bands,)``.
    band_names : tuple of str
        Short names for each band (e.g. ``('B11', 'B12')``). Length matches
        ``band_centers_nm``.
    srf_type : str
        Shape of the synthetic SRF built by
        :meth:`InstrumentSpec.make_srf` — one of ``'gaussian'``,
        ``'rectangular'``, ``'triangular'``.

    Examples
    --------
    Sentinel-2 B11/B12-like synthetic instrument::

        InstrumentSpec(
            name='S2A-like',
            band_centers_nm=np.array([1610.0, 2190.0]),
            band_widths_nm=np.array([90.0, 180.0]),
            band_names=('B11', 'B12'),
        )
    """

    name: str = "synthetic"
    band_centers_nm: np.ndarray = field(
        default_factory=lambda: np.array([1610.0, 2190.0], dtype=float)
    )
    band_widths_nm: np.ndarray = field(
        default_factory=lambda: np.array([90.0, 180.0], dtype=float)
    )
    band_names: tuple[str, ...] = ("B11", "B12")
    srf_type: str = "gaussian"

    def __post_init__(self) -> None:
        centers = np.asarray(self.band_centers_nm, dtype=float)
        widths = np.asarray(self.band_widths_nm, dtype=float)
        if centers.ndim != 1 or widths.ndim != 1:
            raise ValueError("InstrumentSpec: band arrays must be 1-D.")
        if centers.shape != widths.shape:
            raise ValueError(
                f"InstrumentSpec: `band_centers_nm` shape {centers.shape} "
                f"must match `band_widths_nm` shape {widths.shape}"
            )
        if len(self.band_names) != centers.size:
            raise ValueError(
                f"InstrumentSpec: `band_names` length {len(self.band_names)} "
                f"must match number of bands {centers.size}"
            )
        if np.any(widths <= 0.0):
            raise ValueError(
                "InstrumentSpec: `band_widths_nm` entries must be > 0"
            )
        if self.srf_type not in {"gaussian", "rectangular", "triangular"}:
            raise ValueError(
                f"InstrumentSpec: `srf_type` must be one of "
                f"'gaussian', 'rectangular', 'triangular' "
                f"(got {self.srf_type!r})"
            )

    @property
    def n_bands(self) -> int:
        """Number of bands."""
        return int(np.asarray(self.band_centers_nm).size)

    def make_srf(
        self, wavelengths_nm: np.ndarray
    ) -> SpectralResponseFunction:
        """Construct a :class:`SpectralResponseFunction` over ``wavelengths_nm``.

        Imported locally to avoid a circular import at module load.
        """
        from plume_simulation.radtran.srf import SpectralResponseFunction

        return SpectralResponseFunction(
            wavelengths_hr_nm=np.asarray(wavelengths_nm, dtype=float),
            band_centers_nm=np.asarray(self.band_centers_nm, dtype=float),
            band_widths_nm=np.asarray(self.band_widths_nm, dtype=float),
            band_names=tuple(self.band_names),
            srf_type=self.srf_type,
        )


def number_density_cm3(p_atm: float, T_K: float) -> float:
    """Ideal-gas total number density [molecules / cm³]."""
    if not (T_K > 0.0 and p_atm > 0.0):
        raise ValueError(
            "number_density_cm3: `T_K` and `p_atm` must be > 0 "
            f"(got T_K={T_K!r}, p_atm={p_atm!r})"
        )
    p_pa = p_atm * ATM_TO_PA
    n_per_m3 = p_pa / (BOLTZMANN_J_PER_K * T_K)
    return float(n_per_m3 / CM3_PER_M3)
