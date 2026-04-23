"""Tests for ``plume_simulation.hapi_lut.multi`` (HAPI stubbed)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import xarray as xr

from plume_simulation.hapi_lut import (
    LUTGridConfig,
    create_combined_lut,
    create_multi_gas_luts,
)


@pytest.fixture
def stub_hapi(monkeypatch):
    from pathlib import Path

    fake = types.ModuleType("hapi")
    state: dict[str, Path] = {}

    def db_begin(path):
        state["cache"] = Path(path)

    def fetch(name, *args, **kwargs):  # noqa: ARG001
        cache = state["cache"]
        (cache / f"{name}.data").write_text("stub\n")
        (cache / f"{name}.header").write_text("stub\n")

    def absorptionCoefficient_Voigt(  # noqa: N802
        *, SourceTables, WavenumberGrid, Environment, Diluent, HITRAN_units  # noqa: ARG001
    ):
        nu = np.asarray(WavenumberGrid, dtype=float)
        T = Environment["T"]
        p = Environment["p"]
        # Make the "gas" identity affect σ so the test can distinguish variables.
        base = {"CH4": 1.0, "CO2": 2.0, "H2O": 0.5}.get(SourceTables, 1.0)
        coef = base * (1e-23 + 1e-26 * T + 1e-25 * p) * np.ones_like(nu)
        return nu, coef

    fake.db_begin = db_begin
    fake.fetch = fetch
    fake.absorptionCoefficient_Voigt = absorptionCoefficient_Voigt
    monkeypatch.setitem(sys.modules, "hapi", fake)
    return fake


@pytest.fixture
def tiny_grid():
    return LUTGridConfig(
        T_grid=np.array([220.0, 300.0]),
        P_grid=np.array([0.5, 1.0]),
        nu_step=0.2,
    )


def test_create_multi_gas_luts_writes_one_file_per_gas(stub_hapi, tiny_grid, tmp_path):
    files = create_multi_gas_luts(
        ["CH4", "CO2"],
        grid_config=tiny_grid,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
    )
    assert set(files) == {"CH4", "CO2"}
    for path in files.values():
        assert path.exists() and path.suffix == ".nc"


def test_create_multi_gas_luts_skips_existing_without_force(stub_hapi, tiny_grid, tmp_path):
    out = tmp_path / "out"
    first = create_multi_gas_luts(["CH4"], grid_config=tiny_grid, cache_dir=tmp_path / "c", output_dir=out)
    mtime0 = first["CH4"].stat().st_mtime_ns
    second = create_multi_gas_luts(["CH4"], grid_config=tiny_grid, cache_dir=tmp_path / "c", output_dir=out)
    assert second["CH4"].stat().st_mtime_ns == mtime0


def test_create_multi_gas_luts_skips_unknown_gas(stub_hapi, tiny_grid, tmp_path):
    files = create_multi_gas_luts(
        ["CH4", "UNOBTAINIUM"],
        grid_config=tiny_grid,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
    )
    assert set(files) == {"CH4"}


def test_create_combined_lut_has_one_var_per_gas(stub_hapi, tiny_grid, tmp_path):
    path = create_combined_lut(
        ["CH4", "CO2"],
        grid_config=tiny_grid,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
    )
    with xr.open_dataset(path) as ds:
        assert "CH4_cross_section" in ds.data_vars
        assert "CO2_cross_section" in ds.data_vars
        # CO2 stub has 2x the CH4 coefficient — check this shows up in the data.
        ratio = ds["CO2_cross_section"].values / ds["CH4_cross_section"].values
        np.testing.assert_allclose(ratio, 2.0, rtol=1e-5)


def test_create_combined_lut_raises_when_ranges_disjoint(stub_hapi, tiny_grid, tmp_path):
    # O2 is 12500-14000 cm^-1 and N2O is 4000-5000 cm^-1 — no overlap.
    with pytest.raises(ValueError, match="no common spectral range"):
        create_combined_lut(
            ["O2", "N2O"],
            grid_config=tiny_grid,
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "out",
        )
