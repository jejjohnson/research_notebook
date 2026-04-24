"""Config dataclasses: ObservationGeometry, InstrumentSpec."""

from __future__ import annotations

import numpy as np
import pytest
from plume_simulation.radtran.config import (
    InstrumentSpec,
    ObservationGeometry,
    number_density_cm3,
)


def test_observation_geometry_amf_plane_parallel():
    geom = ObservationGeometry(sza_deg=30.0, vza_deg=0.0)
    # 1/cos(30°) + 1/cos(0°) = 1.1547 + 1 = 2.1547
    assert geom.air_mass_factor == pytest.approx(1.0 / np.cos(np.deg2rad(30.0)) + 1.0, rel=1e-6)


def test_observation_geometry_amf_override_used_when_set():
    geom = ObservationGeometry(sza_deg=45.0, vza_deg=15.0, amf=3.5)
    assert geom.air_mass_factor == 3.5


@pytest.mark.parametrize("bad", [-1.0, 90.0, 91.0])
def test_observation_geometry_rejects_bad_sza(bad):
    with pytest.raises(ValueError, match="sza_deg"):
        ObservationGeometry(sza_deg=bad, vza_deg=0.0)


def test_observation_geometry_rejects_nonpositive_path_length():
    with pytest.raises(ValueError, match="path_length_cm"):
        ObservationGeometry(path_length_cm=0.0)


def test_observation_geometry_rejects_nonpositive_amf_override():
    with pytest.raises(ValueError, match="amf"):
        ObservationGeometry(amf=-1.0)


def test_instrument_spec_defaults_are_two_bands():
    spec = InstrumentSpec()
    assert spec.n_bands == 2
    assert spec.band_names == ("B11", "B12")


def test_instrument_spec_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="must match `band_widths_nm`"):
        InstrumentSpec(
            band_centers_nm=np.array([1610.0, 2190.0]),
            band_widths_nm=np.array([90.0]),
            band_names=("B11", "B12"),
        )


def test_instrument_spec_rejects_name_length_mismatch():
    with pytest.raises(ValueError, match="band_names"):
        InstrumentSpec(band_names=("B11",))


def test_instrument_spec_rejects_nonpositive_widths():
    with pytest.raises(ValueError, match="band_widths_nm"):
        InstrumentSpec(band_widths_nm=np.array([90.0, 0.0]))


def test_instrument_spec_make_srf_round_trip():
    spec = InstrumentSpec()
    wl = np.linspace(1400.0, 2500.0, 2001)
    srf = spec.make_srf(wl)
    # Each band's SRF row sums to 1 (L1-normalised by construction).
    np.testing.assert_allclose(srf.matrix.sum(axis=1), 1.0, atol=1e-12)


def test_number_density_ideal_gas_at_1atm_300K():
    # n = p / (k_B T); at 1 atm, 300 K this is ~2.446e19 molec/cm^3
    n = number_density_cm3(1.0, 300.0)
    assert n == pytest.approx(2.446e19, rel=1e-2)


def test_number_density_rejects_nonpositive_state():
    with pytest.raises(ValueError):
        number_density_cm3(0.0, 300.0)
    with pytest.raises(ValueError):
        number_density_cm3(1.0, -300.0)
