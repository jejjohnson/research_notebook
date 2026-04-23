"""Single-gas absorption-cross-section LUT builder.

Given a ``GasConfig`` and a ``LUTGridConfig`` this module fetches HITRAN line
data (cached under a user-provided directory), evaluates a Voigt-profile
line-by-line sum on every (T, P) grid cell, and wraps the result as a
CF-conformant ``xarray.Dataset`` ready for NetCDF persistence.

HAPI is imported lazily inside the functions that need it so ``import
plume_simulation.hapi_lut`` does not require ``hitran-api`` to be installed.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from plume_simulation.hapi_lut.config import (
    DEFAULT_VMR_NOMINAL,
    GasConfig,
    LUTGridConfig,
)

logger = logging.getLogger(__name__)

# Environment variable consulted by ``default_cache_dir`` when the caller
# does not explicitly pass ``cache_dir``. Kept as a string name only — this
# module does not read global state at import time.
HAPI_CACHE_ENV = "HAPI_CACHE_DIR"


def default_cache_dir() -> Path:
    """Return the HITRAN line-data cache directory.

    Resolution order:
      1. ``$HAPI_CACHE_DIR`` if set.
      2. ``~/.cache/plume_simulation/hitran`` — a per-user cache, so that
         invoking the generator from anywhere on disk does not leave
         unignored artefacts in the repository tree. The
         ``projects/plume_simulation/data/hitran_cache`` path used by the
         notebooks is always passed explicitly.
    """
    env = os.environ.get(HAPI_CACHE_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / ".cache" / "plume_simulation" / "hitran"


def _init_hapi_cache(cache_dir: Path) -> None:
    """Create the cache directory and point HAPI's ``db_begin`` at it."""
    from hapi import db_begin

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_begin(str(cache_dir))


def fetch_hitran_data(
    gas_config: GasConfig,
    cache_dir: Path | str | None = None,
) -> Path:
    """Fetch HITRAN line parameters for ``gas_config`` into ``cache_dir``.

    HAPI writes a ``<GasConfig.name>.data`` + ``.header`` pair under
    ``cache_dir``. If a cached pair already exists this call no-ops without
    contacting HITRAN; otherwise the line parameters are downloaded and
    parsed by HAPI.

    Raises:
        RuntimeError: if ``fetch()`` raises (e.g. no network, invalid range)
            *and* no cached pair is already present. The underlying HAPI
            exception is chained so the network / API error is visible.

    Returns:
        The resolved cache directory.
    """
    from hapi import fetch

    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    _init_hapi_cache(cache)

    data_path = cache / f"{gas_config.name}.data"
    header_path = cache / f"{gas_config.name}.header"
    already_cached = data_path.exists() and header_path.exists()

    logger.info(
        "Fetching HITRAN %s (M%d I%d, %.1f-%.1f cm^-1)%s",
        gas_config.name,
        gas_config.molecule_id,
        gas_config.isotopologue_id,
        gas_config.nu_min,
        gas_config.nu_max,
        " [cache hit]" if already_cached else "",
    )

    try:
        fetch(
            gas_config.name,
            gas_config.molecule_id,
            gas_config.isotopologue_id,
            gas_config.nu_min,
            gas_config.nu_max,
        )
    except Exception as exc:
        # HAPI raises plain Exceptions on network failures and API errors.
        # If we have a cached pair we can proceed — otherwise the user sees
        # a clear failure instead of silently-corrupt LUTs downstream.
        if not already_cached:
            raise RuntimeError(
                f"HITRAN fetch failed for {gas_config.name} "
                f"(M{gas_config.molecule_id} I{gas_config.isotopologue_id}, "
                f"{gas_config.nu_min:.1f}-{gas_config.nu_max:.1f} cm^-1) "
                f"and no cached data was found at {cache}."
            ) from exc
        logger.warning(
            "fetch() raised %s — proceeding with cached %s.",
            type(exc).__name__, data_path,
        )

    if not (data_path.exists() and header_path.exists()):
        raise RuntimeError(
            f"HITRAN fetch for {gas_config.name} returned without error but "
            f"no {gas_config.name}.data/.header pair was produced under {cache}."
        )
    return cache


def compute_absorption_lut(
    gas_config: GasConfig,
    grid_config: LUTGridConfig,
    *,
    cache_dir: Path | str | None = None,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate σ(ν, T, P) on the full LUT grid.

    Assumes ``fetch_hitran_data`` has already populated the cache. Fails
    fast — a HAPI failure at any ``(T, P)`` knot, or any non-finite
    cross-section in the result, raises rather than leaving silent NaN
    holes in the LUT.

    Args:
        gas_config:   Target gas (molecule + isotopologue + ν range).
        grid_config:  Common T/P/ν-step grid.
        cache_dir:    Where the fetched HITRAN ``.data/.header`` pair lives.
        progress:     Emit per-(T, P) log lines at INFO.

    Raises:
        RuntimeError: if ``absorptionCoefficient_Voigt`` raises at any
            grid knot, or if the returned cross-section contains any
            non-finite values.

    Returns:
        ``(nu_grid, sigma_lut, wavelength_nm)``:
        - ``nu_grid``: wavenumbers [cm⁻¹], shape ``(n_nu,)``
        - ``sigma_lut``: cross-sections [cm²/molecule], shape ``(n_nu, n_T, n_P)``.
        - ``wavelength_nm``: wavelengths [nm], shape ``(n_nu,)``.
    """
    from hapi import absorptionCoefficient_Voigt

    if cache_dir is not None:
        _init_hapi_cache(Path(cache_dir))

    nu_grid = np.arange(gas_config.nu_min, gas_config.nu_max, grid_config.nu_step)
    wavelength_nm = 1e7 / nu_grid

    n_T = len(grid_config.T_grid)
    n_P = len(grid_config.P_grid)
    sigma_lut = np.zeros((nu_grid.size, n_T, n_P), dtype=np.float64)

    vmr_nominal = DEFAULT_VMR_NOMINAL.get(gas_config.name, 1e-6)
    diluent = grid_config.get_diluent_for_gas(gas_config.name, vmr_nominal)

    total = n_T * n_P
    for i_T, T in enumerate(grid_config.T_grid):
        for i_P, P in enumerate(grid_config.P_grid):
            if progress:
                logger.info(
                    "  [%d/%d] %s T=%.0fK P=%.2fatm",
                    i_T * n_P + i_P + 1, total, gas_config.name, T, P,
                )
            try:
                _, coef = absorptionCoefficient_Voigt(
                    SourceTables=gas_config.name,
                    WavenumberGrid=nu_grid,
                    Environment={"T": float(T), "p": float(P)},
                    Diluent=diluent,
                    HITRAN_units=True,  # cm²/molecule
                )
            except Exception as exc:
                raise RuntimeError(
                    f"absorptionCoefficient_Voigt failed for {gas_config.name} "
                    f"at T={T:.1f} K, p={P:.3f} atm."
                ) from exc
            sigma_lut[:, i_T, i_P] = coef

    if not np.all(np.isfinite(sigma_lut)):
        n_bad = int(np.sum(~np.isfinite(sigma_lut)))
        raise RuntimeError(
            f"compute_absorption_lut produced {n_bad} non-finite entries for "
            f"{gas_config.name}; refusing to return a partial LUT."
        )

    return nu_grid, sigma_lut, wavelength_nm


def build_lut_dataset(
    gas_config: GasConfig,
    grid_config: LUTGridConfig,
    nu_grid: np.ndarray,
    sigma_lut: np.ndarray,
    wavelength_nm: np.ndarray,
) -> xr.Dataset:
    """Wrap the raw LUT arrays in a CF-conformant ``xarray.Dataset``."""
    now = datetime.now(timezone.utc).isoformat()
    ds = xr.Dataset(
        data_vars={
            "absorption_cross_section": (
                ["wavenumber", "temperature", "pressure"],
                sigma_lut,
                {
                    "units": "cm^2 / molecule",
                    "long_name": f"{gas_config.name} absorption cross-section",
                    "description": (
                        "Voigt-profile line-by-line absorption cross-section "
                        "from HITRAN, computed with HAPI."
                    ),
                    "standard_name": (
                        f"absorption_cross_section_of_{gas_config.name.lower()}_in_air"
                    ),
                },
            ),
        },
        coords={
            "wavenumber": (
                ["wavenumber"],
                nu_grid,
                {"units": "cm^-1", "long_name": "Wavenumber", "standard_name": "wavenumber"},
            ),
            "wavelength": (
                ["wavenumber"],
                wavelength_nm,
                {"units": "nm", "long_name": "Wavelength", "standard_name": "radiation_wavelength"},
            ),
            "temperature": (
                ["temperature"],
                np.asarray(grid_config.T_grid, dtype=float),
                {"units": "K", "long_name": "Temperature", "standard_name": "air_temperature"},
            ),
            "pressure": (
                ["pressure"],
                np.asarray(grid_config.P_grid, dtype=float),
                {"units": "atm", "long_name": "Pressure", "standard_name": "air_pressure"},
            ),
        },
        attrs={
            "title": f"{gas_config.name} Absorption Cross-Section Look-Up Table",
            "institution": "Generated using HITRAN API (HAPI)",
            "source": "HITRAN molecular spectroscopic database",
            "molecule": gas_config.name,
            "molecule_id": gas_config.molecule_id,
            "isotopologue_id": gas_config.isotopologue_id,
            "spectral_range_cm_1": f"{nu_grid.min():.1f}-{nu_grid.max():.1f}",
            "spectral_range_nm": f"{wavelength_nm[-1]:.1f}-{wavelength_nm[0]:.1f}",
            "temperature_range_K": f"{float(grid_config.T_grid.min()):.0f}-{float(grid_config.T_grid.max()):.0f}",
            "pressure_range_atm": f"{float(grid_config.P_grid.min()):.2f}-{float(grid_config.P_grid.max()):.2f}",
            "line_shape": "Voigt profile",
            "description": gas_config.description,
            "conventions": "CF-1.8",
            "creation_date": now,
        },
    )
    return ds


def save_lut(ds: xr.Dataset, output_dir: Path | str, gas_name: str) -> Path:
    """Persist ``ds`` to ``<output_dir>/<gas_name>_absorption_lut.nc`` (zlib)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{gas_name.lower()}_absorption_lut.nc"
    ds.to_netcdf(
        path,
        encoding={
            "absorption_cross_section": {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
            }
        },
    )
    logger.info("Saved %s LUT to %s (%.1f MB)", gas_name, path, path.stat().st_size / 1e6)
    return path


def generate_single_gas_lut(
    gas_config: GasConfig,
    grid_config: LUTGridConfig | None = None,
    *,
    cache_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    save: bool = False,
) -> xr.Dataset:
    """End-to-end single-gas pipeline: fetch → compute → wrap → (optionally) save."""
    grid_config = grid_config if grid_config is not None else LUTGridConfig()
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()

    fetch_hitran_data(gas_config, cache_dir=cache)
    nu_grid, sigma_lut, wavelength_nm = compute_absorption_lut(
        gas_config, grid_config, cache_dir=cache
    )
    ds = build_lut_dataset(gas_config, grid_config, nu_grid, sigma_lut, wavelength_nm)
    if save:
        if output_dir is None:
            raise ValueError("save=True requires output_dir.")
        save_lut(ds, output_dir, gas_config.name)
    return ds
