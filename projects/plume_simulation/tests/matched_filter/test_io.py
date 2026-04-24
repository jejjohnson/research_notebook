"""Tests for ``plume_simulation.matched_filter.io`` (xarray glue)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from plume_simulation.matched_filter.background import estimate_cov_empirical
from plume_simulation.matched_filter.core import apply_image
from plume_simulation.matched_filter.io import apply_image_xarray


def test_apply_image_xarray_matches_plain_apply_image(rng):
    H, W, B = 8, 9, 5
    cube = 1.0 + rng.standard_normal((H, W, B)) * 0.05
    target = rng.standard_normal(B) * 0.1
    mean = cube.reshape(-1, B).mean(axis=0)
    cov_op = estimate_cov_empirical(cube, mean=mean, ridge=1e-6)
    import jax.numpy as jnp

    scores_plain = apply_image(
        jnp.asarray(cube),
        mean=jnp.asarray(mean),
        cov_op=cov_op,
        target=jnp.asarray(target),
    )
    da = xr.DataArray(
        cube,
        dims=("y", "x", "band"),
        coords={"y": np.arange(H), "x": np.arange(W), "band": np.arange(B)},
    )
    scores_xr = apply_image_xarray(
        da, mean=mean, cov_op=cov_op, target=target, band_dim="band", dask="allowed"
    )
    assert scores_xr.dims == ("y", "x")
    assert scores_xr.shape == (H, W)
    np.testing.assert_allclose(scores_xr.values, np.asarray(scores_plain), atol=1e-10)
    # Coordinates are preserved on the spatial dims.
    np.testing.assert_array_equal(scores_xr["y"].values, np.arange(H))
    np.testing.assert_array_equal(scores_xr["x"].values, np.arange(W))


def test_apply_image_xarray_rejects_missing_band_dim(rng):
    cube = xr.DataArray(rng.standard_normal((3, 4, 5)), dims=("y", "x", "channel"))
    target = rng.standard_normal(5)
    from plume_simulation.matched_filter.background import estimate_cov_empirical

    cov_op = estimate_cov_empirical(cube.values, ridge=1e-6)
    with pytest.raises(ValueError, match="band"):
        apply_image_xarray(
            cube, mean=cube.mean(("y", "x")).values, cov_op=cov_op, target=target
        )
