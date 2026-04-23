"""Gas + grid configuration for HAPI absorption-cross-section look-up tables.

A LUT is an N-dimensional array of absorption cross-sections

    σ(ν, T, P)           [cm² / molecule]

computed line-by-line via HAPI's Voigt routine. ``GasConfig`` identifies the
HITRAN line set (molecule + isotopologue + spectral range) and ``LUTGridConfig``
describes the common (T, P, ν-step) grid used across gases.

The ``ATMOSPHERIC_GASES`` registry covers the six species most commonly needed
for SWIR/NIR remote sensing (CH₄, CO₂, H₂O, O₂, N₂O, CO). Add entries for
other HITRAN species by constructing a ``GasConfig`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Typical dry-air volume mixing ratios, used as the "self" fraction when
# building HAPI's Diluent dict. Small errors here only affect self-vs-air
# collisional broadening and are second-order for most retrievals.
DEFAULT_VMR_NOMINAL: dict[str, float] = {
    "CH4": 2e-6,
    "CO2": 400e-6,
    "H2O": 1e-2,
    "O2": 0.21,
    "N2O": 0.3e-6,
    "CO": 0.1e-6,
}


@dataclass(frozen=True)
class GasConfig:
    """Configuration for a single HITRAN gas species.

    Attributes:
        name:            Short name used as HAPI SourceTable id (e.g. 'CH4').
        molecule_id:     HITRAN molecule id (see `hitran.org`).
        isotopologue_id: HITRAN isotopologue id (1 = most abundant).
        nu_min:          Lower wavenumber bound [cm⁻¹].
        nu_max:          Upper wavenumber bound [cm⁻¹].
        description:     Free-text description for dataset metadata.
    """

    name: str
    molecule_id: int
    isotopologue_id: int
    nu_min: float
    nu_max: float
    description: str = ""

    def __repr__(self) -> str:
        return f"GasConfig({self.name}, M{self.molecule_id}, I{self.isotopologue_id})"


@dataclass
class LUTGridConfig:
    """Grid configuration shared across gases for a single LUT build.

    Attributes:
        T_grid:  Temperature knots [K]. Should bracket the atmospheric range
                 of interest (upper-troposphere ~200 K to surface ~320 K).
        P_grid:  Pressure knots [atm]. Should bracket the altitude range of
                 interest (upper-tropo ~0.1 atm to surface ~1.0 atm).
        nu_step: Spectral step [cm⁻¹] on the output wavenumber grid. HITRAN
                 lines have HWHMs down to ~0.005 cm⁻¹ at low pressure — step
                 ≤ 0.01 cm⁻¹ for hyperspectral work, 0.05 cm⁻¹ for tutorials.
        diluent_composition: Optional override for the air diluent fractions.
                 If None, each gas gets {'air': 1-VMR, 'self': VMR}.
    """

    T_grid: np.ndarray = field(
        default_factory=lambda: np.array([200, 220, 240, 260, 280, 300, 320], dtype=float)
    )
    P_grid: np.ndarray = field(
        default_factory=lambda: np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=float)
    )
    nu_step: float = 0.01
    diluent_composition: dict[str, float] | None = None

    def get_diluent_for_gas(
        self,
        gas_name: str,
        vmr_nominal: float | None = None,
    ) -> dict[str, float]:
        """Return the HAPI ``Diluent`` dict for a given gas.

        If an explicit ``diluent_composition`` override was supplied at
        construction time it is returned verbatim; otherwise a standard
        ``{'air': 1 - VMR, 'self': VMR}`` split is built from
        ``DEFAULT_VMR_NOMINAL[gas_name]`` (falling back to 1e-6).
        """
        if self.diluent_composition is not None:
            return dict(self.diluent_composition)
        if vmr_nominal is None:
            vmr_nominal = DEFAULT_VMR_NOMINAL.get(gas_name, 1e-6)
        return {"air": 1.0 - vmr_nominal, "self": vmr_nominal}


# Canonical SWIR/NIR gases for methane/CO₂ remote sensing.
ATMOSPHERIC_GASES: dict[str, GasConfig] = {
    "CH4": GasConfig(
        name="CH4",
        molecule_id=6,
        isotopologue_id=1,  # 12CH4
        nu_min=4000.0,  # SWIR (2500 nm)
        nu_max=8000.0,  # NIR (1250 nm)
        description="Methane — primary greenhouse-gas retrieval target.",
    ),
    "CO2": GasConfig(
        name="CO2",
        molecule_id=2,
        isotopologue_id=1,  # 12C16O2
        nu_min=4000.0,
        nu_max=8000.0,
        description="Carbon dioxide — reference gas for column retrievals.",
    ),
    "H2O": GasConfig(
        name="H2O",
        molecule_id=1,
        isotopologue_id=1,  # H2-16O
        nu_min=4000.0,
        nu_max=25000.0,
        description="Water vapour — dominant atmospheric absorber.",
    ),
    "O2": GasConfig(
        name="O2",
        molecule_id=7,
        isotopologue_id=1,  # 16O2
        nu_min=12500.0,  # O2 A-band (~760 nm)
        nu_max=14000.0,
        description="Oxygen — surface-pressure retrievals via A-band.",
    ),
    "N2O": GasConfig(
        name="N2O",
        molecule_id=4,
        isotopologue_id=1,  # 14N2-16O
        nu_min=4000.0,
        nu_max=5000.0,
        description="Nitrous oxide — greenhouse-gas side channel.",
    ),
    "CO": GasConfig(
        name="CO",
        molecule_id=5,
        isotopologue_id=1,  # 12C-16O
        nu_min=4000.0,
        nu_max=5000.0,
        description="Carbon monoxide — pollution / combustion tracer.",
    ),
}
