"""Multi-gas LUT orchestration — separate-per-gas and combined builds."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from plume_simulation.hapi_lut.config import (
    ATMOSPHERIC_GASES,
    GasConfig,
    LUTGridConfig,
)
from plume_simulation.hapi_lut.generator import (
    build_lut_dataset,
    compute_absorption_lut,
    default_cache_dir,
    fetch_hitran_data,
    save_lut,
)

logger = logging.getLogger(__name__)


def create_multi_gas_luts(
    gases: list[str],
    grid_config: LUTGridConfig | None = None,
    *,
    output_dir: Path | str,
    cache_dir: Path | str | None = None,
    force_recompute: bool = False,
) -> dict[str, Path]:
    """Build one separate ``<gas>_absorption_lut.nc`` per gas.

    Gases are looked up in ``ATMOSPHERIC_GASES``. Unknown names are skipped
    with a warning rather than raising — lets users pass a superset list.

    Args:
        gases:           List of gas names (e.g. ``['CH4', 'CO2', 'H2O']``).
        grid_config:     Common grid; defaults to ``LUTGridConfig()``.
        output_dir:      Required. Where to write NetCDF LUTs. No default
                         is provided to avoid dropping unignored artefacts
                         into the current working directory; callers should
                         point this at the project-local git-ignored
                         ``projects/plume_simulation/data/hapi_lut/`` path
                         (or an out-of-tree scratch directory).
        cache_dir:       HITRAN cache dir; defaults to ``default_cache_dir()``.
        force_recompute: If False, skip gases whose NetCDF already exists.

    Returns:
        Mapping ``{gas_name: Path}`` for successfully-built LUTs.
    """
    grid_config = grid_config if grid_config is not None else LUTGridConfig()
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lut_files: dict[str, Path] = {}
    for gas_name in gases:
        if gas_name not in ATMOSPHERIC_GASES:
            logger.warning("Unknown gas %r — skipping.", gas_name)
            continue
        gas_config = ATMOSPHERIC_GASES[gas_name]

        path = out / f"{gas_name.lower()}_absorption_lut.nc"
        if path.exists() and not force_recompute:
            logger.info("LUT already exists for %s (%s); skipping.", gas_name, path)
            lut_files[gas_name] = path
            continue

        fetch_hitran_data(gas_config, cache_dir=cache)
        nu_grid, sigma_lut, wavelength_nm = compute_absorption_lut(
            gas_config, grid_config, cache_dir=cache
        )
        ds = build_lut_dataset(gas_config, grid_config, nu_grid, sigma_lut, wavelength_nm)
        lut_files[gas_name] = save_lut(ds, out, gas_name)

    logger.info("Built %d of %d requested LUTs.", len(lut_files), len(gases))
    return lut_files


def create_combined_lut(
    gases: list[str],
    grid_config: LUTGridConfig | None = None,
    *,
    output_dir: Path | str,
    cache_dir: Path | str | None = None,
    filename: str = "combined_atmospheric_lut.nc",
) -> Path:
    """Build a single NetCDF with one cross-section variable per gas.

    The shared wavenumber axis is the intersection of the per-gas ranges:
    ``[max(nu_min), min(nu_max)]``. All variables therefore live on one
    ``(wavenumber, temperature, pressure)`` coordinate system — convenient
    for matched-filter retrievals where CH₄ and CO₂ share the same SWIR window.

    ``output_dir`` has no default for the same reason as
    :func:`create_multi_gas_luts` — forcing an explicit path keeps
    generated artefacts out of the current working directory.
    """
    grid_config = grid_config if grid_config is not None else LUTGridConfig()
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    missing = [g for g in gases if g not in ATMOSPHERIC_GASES]
    if missing:
        raise KeyError(f"Unknown gases: {missing}")

    nu_min = max(ATMOSPHERIC_GASES[g].nu_min for g in gases)
    nu_max = min(ATMOSPHERIC_GASES[g].nu_max for g in gases)
    if nu_min >= nu_max:
        raise ValueError(
            f"Gases {gases} have no common spectral range ({nu_min:.1f} >= {nu_max:.1f})."
        )
    nu_grid = np.arange(nu_min, nu_max, grid_config.nu_step)
    wavelength_nm = 1e7 / nu_grid

    logger.info(
        "Building combined LUT for %s on %.1f-%.1f cm^-1 (%d pts).",
        gases, nu_min, nu_max, nu_grid.size,
    )

    data_vars: dict[str, tuple] = {}
    for gas_name in gases:
        base = ATMOSPHERIC_GASES[gas_name]
        sub = GasConfig(
            name=base.name,
            molecule_id=base.molecule_id,
            isotopologue_id=base.isotopologue_id,
            nu_min=nu_min,
            nu_max=nu_max,
            description=base.description,
        )
        fetch_hitran_data(sub, cache_dir=cache)
        _, sigma_lut, _ = compute_absorption_lut(sub, grid_config, cache_dir=cache)
        data_vars[f"{gas_name}_cross_section"] = (
            ["wavenumber", "temperature", "pressure"],
            sigma_lut,
            {
                "units": "cm^2 / molecule",
                "long_name": f"{gas_name} absorption cross-section",
                "molecule_id": base.molecule_id,
                "isotopologue_id": base.isotopologue_id,
            },
        )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "wavenumber": (["wavenumber"], nu_grid, {"units": "cm^-1"}),
            "wavelength": (["wavenumber"], wavelength_nm, {"units": "nm"}),
            "temperature": (["temperature"], np.asarray(grid_config.T_grid, dtype=float), {"units": "K"}),
            "pressure": (["pressure"], np.asarray(grid_config.P_grid, dtype=float), {"units": "atm"}),
        },
        attrs={
            "title": "Combined Atmospheric Absorption LUT",
            "gases": ", ".join(gases),
            "line_shape": "Voigt profile",
            "conventions": "CF-1.8",
        },
    )

    path = out / filename
    ds.to_netcdf(
        path,
        encoding={var: {"zlib": True, "complevel": 4, "dtype": "float32"} for var in data_vars},
    )
    logger.info("Saved combined LUT to %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path
