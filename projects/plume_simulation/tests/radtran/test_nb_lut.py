"""Normalised-brightness LUT + plume injection."""

from __future__ import annotations

import numpy as np
import pytest
from plume_simulation.radtran.nb_lut import (
    NBLookup,
    build_nb_lut,
    inject_plume,
    lookup_nb,
)


def test_build_nb_lut_starts_at_one(synthetic_lut, swir_srf):
    lut = build_nb_lut(
        synthetic_lut, swir_srf,
        T_K=280.0, p_atm=1.0, path_length_cm=8.4e5, amf=2.0,
        max_delta_column=100.0, n_grid=501,
    )
    np.testing.assert_allclose(lut.nB[:, 0], 1.0, atol=1e-12)


def test_build_nb_lut_monotonically_decreasing(synthetic_lut, swir_srf):
    lut = build_nb_lut(
        synthetic_lut, swir_srf,
        T_K=280.0, p_atm=1.0, path_length_cm=8.4e5, amf=2.0,
        max_delta_column=50.0, n_grid=201,
    )
    for b in range(lut.n_bands):
        diffs = np.diff(lut.nB[b])
        assert (diffs <= 1e-14).all(), f"nB for band {b} not monotone"


def test_build_nb_lut_B12_absorbs_more_than_B11(synthetic_lut, swir_srf):
    # Our synthetic line is at 4300 cm^-1 ≈ 2326 nm, in the B12 (2190 nm)
    # window. B12 should see more absorption than B11 (1610 nm).
    lut = build_nb_lut(
        synthetic_lut, swir_srf,
        T_K=280.0, p_atm=1.0, path_length_cm=8.4e5, amf=2.0,
        max_delta_column=100.0, n_grid=501,
    )
    # At the maximum ΔX, the B12 nB is lower (more absorbed) than B11's.
    assert lut.nB[1, -1] < lut.nB[0, -1]


def test_lookup_nb_clips_out_of_range(synthetic_lut, swir_srf):
    lut = build_nb_lut(
        synthetic_lut, swir_srf,
        T_K=280.0, p_atm=1.0, path_length_cm=8.4e5, amf=2.0,
        max_delta_column=10.0, n_grid=101,
    )
    plume = np.array([[-5.0, 0.0, 5.0, 20.0]])  # -5 and 20 are outside
    out = lookup_nb(plume, lut)
    assert out.shape == (lut.n_bands, 1, 4)
    # Negative clipped to 0 → nB == 1.
    np.testing.assert_allclose(out[:, 0, 0], 1.0, atol=1e-12)
    # Above-range clipped to max → nB at the tail of the grid.
    np.testing.assert_allclose(out[:, 0, 3], lut.nB[:, -1], atol=1e-12)


def test_lookup_nb_preserves_nans(synthetic_lut, swir_srf):
    lut = build_nb_lut(
        synthetic_lut, swir_srf,
        T_K=280.0, p_atm=1.0, path_length_cm=8.4e5, amf=2.0,
    )
    plume = np.array([[1.0, np.nan]])
    out = lookup_nb(plume, lut)
    assert np.isnan(out[0, 0, 1])
    assert np.isnan(out[1, 0, 1])
    # The non-NaN entries are fine.
    assert not np.isnan(out[0, 0, 0])


def test_inject_plume_multiplicative(synthetic_lut, swir_srf):
    lut = build_nb_lut(
        synthetic_lut, swir_srf,
        T_K=280.0, p_atm=1.0, path_length_cm=8.4e5, amf=2.0,
    )
    scene = np.ones((2, 4, 4))  # flat clean scene, two bands
    plume = np.zeros((4, 4))
    plume[2, 2] = 5.0
    dirty = inject_plume(scene, plume, lut)
    # Untouched pixels remain at 1.
    np.testing.assert_allclose(dirty[:, 0, 0], 1.0, atol=1e-12)
    # Plume pixel is dimmer in both bands.
    assert dirty[0, 2, 2] < 1.0
    assert dirty[1, 2, 2] < 1.0


def test_inject_plume_rejects_band_mismatch(synthetic_lut, swir_srf):
    lut = build_nb_lut(synthetic_lut, swir_srf, T_K=280.0, p_atm=1.0,
                      path_length_cm=8.4e5, amf=2.0)
    scene = np.ones((3, 4, 4))  # 3-band scene, LUT has 2
    with pytest.raises(ValueError, match="LUT has"):
        inject_plume(scene, np.zeros((4, 4)), lut)


def test_nb_lookup_to_dataset_roundtrip():
    grid = np.linspace(0.0, 10.0, 11)
    nB = np.exp(-grid)[None, :].repeat(2, 0)  # mock
    lut = NBLookup(delta_column=grid, nB=nB, band_names=("A", "B"))
    ds = lut.to_dataset()
    assert "nB" in ds.data_vars
    np.testing.assert_allclose(ds["nB"].values, nB)
    assert list(ds["band"].values) == ["A", "B"]


def test_build_nb_lut_rejects_invalid_grid(synthetic_lut, swir_srf):
    with pytest.raises(ValueError, match="n_grid"):
        build_nb_lut(
            synthetic_lut, swir_srf,
            T_K=280.0, p_atm=1.0, path_length_cm=8.4e5, amf=2.0,
            n_grid=1,
        )
    with pytest.raises(ValueError, match="max_delta_column"):
        build_nb_lut(
            synthetic_lut, swir_srf,
            T_K=280.0, p_atm=1.0, path_length_cm=8.4e5, amf=2.0,
            max_delta_column=0.0,
        )
