"""Tests for ``plume_simulation.hapi_lut.generator`` with HAPI stubbed.

The real ``hitran-api`` package requires network access to populate its
line-parameter cache on first use, so these tests inject a stub module into
``sys.modules`` under the name ``hapi``. The stub exposes just the three
callables that ``generator.py`` imports lazily: ``db_begin``, ``fetch``, and
``absorptionCoefficient_Voigt``. The last of these returns a synthetic
``σ(ν) = base + 0.01*T + 0.1*p`` so the tests can check shape, ordering,
and metadata without asserting on specific HITRAN numbers.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from plume_simulation.hapi_lut import (
    GasConfig,
    LUTGridConfig,
    build_lut_dataset,
    compute_absorption_lut,
    save_lut,
)


@pytest.fixture
def stub_hapi(monkeypatch):
    """Inject a fake ``hapi`` module into sys.modules for the test."""
    fake = types.ModuleType("hapi")

    def db_begin(path):  # noqa: ARG001 — path intentionally unused in stub
        return None

    def fetch(name, mol_id, iso_id, nu_min, nu_max):  # noqa: ARG001
        return None

    def absorptionCoefficient_Voigt(  # noqa: N802 — mirror HAPI API
        *,
        SourceTables,
        WavenumberGrid,
        Environment,
        Diluent,  # noqa: ARG001 — accepted, not used in stub
        HITRAN_units,  # noqa: ARG001
    ):
        nu = np.asarray(WavenumberGrid, dtype=float)
        T = Environment["T"]
        p = Environment["p"]
        # Deterministic synthetic cross-section: constant in ν, linear in (T, p).
        coef = np.full_like(nu, 1e-23) + 1e-26 * T + 1e-25 * p
        return nu, coef

    fake.db_begin = db_begin
    fake.fetch = fetch
    fake.absorptionCoefficient_Voigt = absorptionCoefficient_Voigt
    monkeypatch.setitem(sys.modules, "hapi", fake)
    return fake


def test_compute_absorption_lut_has_expected_shape_and_ordering(stub_hapi, tmp_path):
    gas = GasConfig("CH4", molecule_id=6, isotopologue_id=1, nu_min=4000.0, nu_max=4001.0)
    grid = LUTGridConfig(
        T_grid=np.array([200.0, 300.0]),
        P_grid=np.array([0.2, 1.0]),
        nu_step=0.1,
    )
    nu, sigma, wl = compute_absorption_lut(gas, grid, cache_dir=tmp_path, progress=False)
    assert nu.shape == (10,)  # (4001 - 4000) / 0.1
    assert sigma.shape == (10, 2, 2)
    # σ monotone in T, p (stub: + 1e-26·T + 1e-25·p):
    assert bool(np.all(sigma[:, 1, :] > sigma[:, 0, :]))
    assert bool(np.all(sigma[:, :, 1] > sigma[:, :, 0]))
    # Wavelength coord is 1e7/ν (HAPI convention).
    np.testing.assert_allclose(wl, 1e7 / nu)


def test_build_lut_dataset_round_trip_through_netcdf(stub_hapi, tmp_path):
    gas = GasConfig("CH4", molecule_id=6, isotopologue_id=1, nu_min=4000.0, nu_max=4001.0)
    grid = LUTGridConfig(
        T_grid=np.array([220.0, 280.0]),
        P_grid=np.array([0.5, 1.0]),
        nu_step=0.2,
    )
    nu, sigma, wl = compute_absorption_lut(gas, grid, cache_dir=tmp_path, progress=False)
    ds = build_lut_dataset(gas, grid, nu, sigma, wl)
    path = save_lut(ds, tmp_path, "CH4")
    assert path.exists() and path.suffix == ".nc"

    import xarray as xr

    with xr.open_dataset(path) as loaded:
        assert "absorption_cross_section" in loaded.data_vars
        assert loaded.attrs["molecule"] == "CH4"
        assert loaded.attrs["molecule_id"] == 6
        assert loaded.attrs["line_shape"] == "Voigt profile"
        assert loaded.sizes == {"wavenumber": 5, "temperature": 2, "pressure": 2}
