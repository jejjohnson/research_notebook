"""Tests for ``plume_simulation.hapi_lut.generator`` with HAPI stubbed.

The real ``hitran-api`` package requires network access to populate its
line-parameter cache on first use, so these tests inject a stub module into
``sys.modules`` under the name ``hapi``. The stub exposes just the three
callables that ``generator.py`` imports lazily: ``db_begin``, ``fetch``, and
``absorptionCoefficient_Voigt``. The last of these returns a synthetic
``σ(ν) = base + 0.01*T + 0.1*p`` so the tests can check shape, ordering,
and metadata without asserting on specific HITRAN numbers.

``db_begin`` records the cache path so ``fetch`` can drop empty marker
``.data``/``.header`` files — ``fetch_hitran_data`` refuses to return
unless that pair actually exists on disk.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from plume_simulation.hapi_lut import (
    GasConfig,
    LUTGridConfig,
    build_lut_dataset,
    compute_absorption_lut,
    fetch_hitran_data,
    save_lut,
)


def _make_hapi_stub(
    *,
    fetch_raises: type[BaseException] | None = None,
    voigt_raises_on: tuple[float, float] | None = None,
    voigt_returns_nan_on: tuple[float, float] | None = None,
) -> types.ModuleType:
    """Return a fake ``hapi`` module, tunable for the failure-path tests."""
    fake = types.ModuleType("hapi")
    state: dict[str, Path] = {}

    def db_begin(path):
        state["cache"] = Path(path)

    def fetch(name, mol_id, iso_id, nu_min, nu_max):  # noqa: ARG001
        if fetch_raises is not None:
            raise fetch_raises("simulated HAPI fetch failure")
        cache = state["cache"]
        (cache / f"{name}.data").write_text("stub\n")
        (cache / f"{name}.header").write_text("stub\n")

    def absorptionCoefficient_Voigt(  # noqa: N802
        *, SourceTables, WavenumberGrid, Environment, Diluent, HITRAN_units  # noqa: ARG001
    ):
        nu = np.asarray(WavenumberGrid, dtype=float)
        T = Environment["T"]
        p = Environment["p"]
        if voigt_raises_on is not None and (T, p) == voigt_raises_on:
            raise RuntimeError(f"simulated Voigt failure at T={T}, p={p}")
        coef = np.full_like(nu, 1e-23) + 1e-26 * T + 1e-25 * p
        if voigt_returns_nan_on is not None and (T, p) == voigt_returns_nan_on:
            coef[0] = np.nan
        return nu, coef

    fake.db_begin = db_begin
    fake.fetch = fetch
    fake.absorptionCoefficient_Voigt = absorptionCoefficient_Voigt
    return fake


@pytest.fixture
def stub_hapi(monkeypatch):
    fake = _make_hapi_stub()
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


def test_fetch_hitran_data_writes_cache_files(stub_hapi, tmp_path):
    gas = GasConfig("CH4", molecule_id=6, isotopologue_id=1, nu_min=4000.0, nu_max=4001.0)
    fetch_hitran_data(gas, cache_dir=tmp_path)
    assert (tmp_path / "CH4.data").exists()
    assert (tmp_path / "CH4.header").exists()


def test_fetch_hitran_data_raises_when_fetch_fails_and_no_cache(monkeypatch, tmp_path):
    fake = _make_hapi_stub(fetch_raises=RuntimeError)
    monkeypatch.setitem(sys.modules, "hapi", fake)
    gas = GasConfig("CH4", molecule_id=6, isotopologue_id=1, nu_min=4000.0, nu_max=4001.0)
    with pytest.raises(RuntimeError, match="HITRAN fetch failed for CH4"):
        fetch_hitran_data(gas, cache_dir=tmp_path)


def test_fetch_hitran_data_tolerates_fetch_failure_if_cache_present(monkeypatch, tmp_path):
    # Pre-populate the cache and then make fetch() raise — should not abort.
    (tmp_path / "CH4.data").write_text("stub\n")
    (tmp_path / "CH4.header").write_text("stub\n")
    fake = _make_hapi_stub(fetch_raises=RuntimeError)
    monkeypatch.setitem(sys.modules, "hapi", fake)
    gas = GasConfig("CH4", molecule_id=6, isotopologue_id=1, nu_min=4000.0, nu_max=4001.0)
    out = fetch_hitran_data(gas, cache_dir=tmp_path)
    assert out == tmp_path


def test_compute_absorption_lut_raises_when_voigt_call_fails(monkeypatch, tmp_path):
    fake = _make_hapi_stub(voigt_raises_on=(300.0, 1.0))
    monkeypatch.setitem(sys.modules, "hapi", fake)
    gas = GasConfig("CH4", molecule_id=6, isotopologue_id=1, nu_min=4000.0, nu_max=4001.0)
    grid = LUTGridConfig(
        T_grid=np.array([200.0, 300.0]),
        P_grid=np.array([1.0]),
        nu_step=0.2,
    )
    with pytest.raises(RuntimeError, match="absorptionCoefficient_Voigt failed"):
        compute_absorption_lut(gas, grid, cache_dir=tmp_path, progress=False)


def test_compute_absorption_lut_raises_on_non_finite_output(monkeypatch, tmp_path):
    fake = _make_hapi_stub(voigt_returns_nan_on=(300.0, 1.0))
    monkeypatch.setitem(sys.modules, "hapi", fake)
    gas = GasConfig("CH4", molecule_id=6, isotopologue_id=1, nu_min=4000.0, nu_max=4001.0)
    grid = LUTGridConfig(
        T_grid=np.array([200.0, 300.0]),
        P_grid=np.array([1.0]),
        nu_step=0.2,
    )
    with pytest.raises(RuntimeError, match="non-finite entries"):
        compute_absorption_lut(gas, grid, cache_dir=tmp_path, progress=False)


def test_default_cache_dir_respects_env_var(monkeypatch, tmp_path):
    from plume_simulation.hapi_lut import default_cache_dir

    monkeypatch.setenv("HAPI_CACHE_DIR", str(tmp_path))
    assert default_cache_dir() == tmp_path.resolve()


def test_default_cache_dir_fallback_is_under_home_not_cwd(monkeypatch, tmp_path):
    from plume_simulation.hapi_lut import default_cache_dir

    monkeypatch.delenv("HAPI_CACHE_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    out = default_cache_dir()
    # Must not be under the current working dir — that was the old behaviour.
    assert tmp_path not in out.parents
    assert out == Path.home() / ".cache" / "plume_simulation" / "hitran"
