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


def test_open_multi_scene_respects_band_dim_anywhere(tmp_path, rng):
    """Regression test: files stored as ``(band, y, x)`` must still yield
    ``(n_pixels, n_bands)`` batches with spectra intact. Previously the
    flatten assumed band was the last axis and silently scrambled pixels."""
    from plume_simulation.matched_filter.io import open_multi_scene

    n_scenes, n_bands, H, W = 2, 4, 3, 5
    scenes = rng.standard_normal((n_scenes, n_bands, H, W))
    paths = []
    for i in range(n_scenes):
        da = xr.DataArray(
            scenes[i],
            dims=("band", "y", "x"),
            name="radiance",
        )
        p = tmp_path / f"scene_{i}.nc"
        da.to_dataset().to_netcdf(p)
        paths.append(str(p))
    batches = list(open_multi_scene(paths, band_dim="band"))
    assert len(batches) == n_scenes
    for i, batch in enumerate(batches):
        assert batch.shape == (H * W, n_bands)
        # The per-pixel spectrum at (y, x) must equal the original
        # scenes[i, :, y, x] vector — i.e., the axes must have been
        # transposed before the reshape.
        expected = np.moveaxis(scenes[i], 0, -1).reshape(-1, n_bands)
        np.testing.assert_allclose(batch, expected, atol=1e-12)


def test_open_multi_scene_closes_dataset_on_early_stop(tmp_path, rng):
    """Closing the generator mid-iteration must close the backing Dataset
    (file handles released). Regression test for a past code path that
    kept the dataset open for the lifetime of the generator with no
    ``try/finally``."""
    scenes = rng.standard_normal((2, 3, 4, 5))  # (n_scenes, y, x, band)
    paths = []
    for i in range(2):
        da = xr.DataArray(scenes[i], dims=("y", "x", "band"), name="radiance")
        p = tmp_path / f"scene_{i}.nc"
        da.to_dataset().to_netcdf(p)
        paths.append(str(p))

    from plume_simulation.matched_filter.io import open_multi_scene

    opened: list[xr.Dataset] = []
    real_open = xr.open_dataset

    def _tracking_open(*args, **kwargs):
        ds = real_open(*args, **kwargs)
        opened.append(ds)
        return ds

    orig = xr.open_dataset
    xr.open_dataset = _tracking_open
    try:
        gen = open_multi_scene(paths, band_dim="band")
        next(gen)  # one batch, then stop early — triggers the finally path
        gen.close()
    finally:
        xr.open_dataset = orig

    assert len(opened) == 1, f"expected exactly one dataset opened, got {len(opened)}"
    # After Dataset.close(), xarray clears the internal backend handle. The
    # attribute name differs across xarray versions ('_close' on newer,
    # '_file_obj' on older); either one being falsy confirms cleanup.
    ds = opened[0]
    closed = getattr(ds, "_close", getattr(ds, "_file_obj", None)) is None
    assert closed, f"dataset not closed after generator.close() (ds={ds!r})"
