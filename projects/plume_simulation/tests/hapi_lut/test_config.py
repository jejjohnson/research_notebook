"""Tests for ``plume_simulation.hapi_lut.config``."""

from __future__ import annotations

import numpy as np
import pytest

from plume_simulation.hapi_lut import ATMOSPHERIC_GASES, GasConfig, LUTGridConfig


def test_atmospheric_gases_registry_has_expected_species():
    for name in ["CH4", "CO2", "H2O", "O2", "N2O", "CO"]:
        assert name in ATMOSPHERIC_GASES
        cfg = ATMOSPHERIC_GASES[name]
        assert cfg.name == name
        assert cfg.nu_min < cfg.nu_max
        assert cfg.molecule_id >= 1
        assert cfg.isotopologue_id >= 1


def test_lut_grid_config_defaults_are_independent_instances():
    a = LUTGridConfig()
    b = LUTGridConfig()
    # default_factory must produce separate arrays (not a shared class var).
    assert a.T_grid is not b.T_grid
    assert a.P_grid is not b.P_grid
    assert np.all(a.T_grid == b.T_grid)
    assert np.all(a.P_grid == b.P_grid)


def test_get_diluent_for_known_gas_uses_nominal_vmr():
    grid = LUTGridConfig()
    diluent = grid.get_diluent_for_gas("CH4")
    assert diluent == {"air": pytest.approx(1.0 - 2e-6), "self": pytest.approx(2e-6)}


def test_get_diluent_for_unknown_gas_falls_back_to_1e_minus_6():
    grid = LUTGridConfig()
    diluent = grid.get_diluent_for_gas("UNOBTAINIUM")
    assert diluent == {"air": pytest.approx(1.0 - 1e-6), "self": pytest.approx(1e-6)}


def test_explicit_diluent_override_is_returned_verbatim():
    grid = LUTGridConfig(diluent_composition={"air": 0.5, "N2": 0.5})
    diluent = grid.get_diluent_for_gas("CH4")
    assert diluent == {"air": 0.5, "N2": 0.5}


def test_gas_config_repr_is_informative():
    cfg = GasConfig("CH4", molecule_id=6, isotopologue_id=1, nu_min=4000.0, nu_max=4500.0)
    assert "CH4" in repr(cfg)
    assert "M6" in repr(cfg)
    assert "I1" in repr(cfg)
