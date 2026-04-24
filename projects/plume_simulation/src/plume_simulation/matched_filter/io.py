"""xarray glue: apply the matched filter to DataArrays and multi-file stores.

The MF kernel in :mod:`.core` is pure JAX and expects plain arrays. For real
data workflows the scene is usually an :class:`xarray.DataArray` with named
dims (``y``, ``x``, ``band``), coordinates, and possibly lazy Dask backing.
These helpers thread the xarray metadata through while the compute stays in
JAX:

- :func:`apply_image_xarray` — wraps :func:`apply_image` in
  :func:`xarray.apply_ufunc` so band-last DataArrays produce a score DataArray
  with ``(y, x)`` dims and preserved spatial coordinates. Dask-backed inputs
  are evaluated lazily with chunk-wise parallelism.
- :func:`open_multi_scene` — thin wrapper around :func:`xarray.open_mfdataset`
  that concatenates along a user-named scene dim and returns a streaming
  iterable of ``(n_samples, n_bands)`` batches for feeding into
  :class:`~plume_simulation.matched_filter.streaming.WelfordAccumulator`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    import lineax as lx
    import xarray as xr
    from jaxtyping import Array, Float

    LinearOperator = lx.AbstractLinearOperator


def apply_image_xarray(
    cube: xr.DataArray,
    mean: np.ndarray | Float[Array, "B"],
    cov_op: LinearOperator,
    target: np.ndarray | Float[Array, "B"],
    *,
    band_dim: str = "band",
    dask: str = "parallelized",
) -> xr.DataArray:
    """Apply the matched filter to an xarray DataArray.

    Parameters
    ----------
    cube
        Radiance cube with a band-last convention (one dim named ``band_dim``).
        Other dims are preserved on the output — typically ``y, x`` but any
        additional dims (e.g. ``time``) pass through unchanged.
    mean, cov_op, target
        Same semantics as :func:`plume_simulation.matched_filter.core.apply_image`.
    band_dim
        Name of the spectral dimension. Default ``'band'``.
    dask
        Forwarded to :func:`xarray.apply_ufunc`:
        ``'parallelized'`` (default — chunk-wise) or ``'allowed'``.

    Returns
    -------
    xr.DataArray
        Score DataArray with ``band_dim`` collapsed.
    """
    import xarray as xr

    from plume_simulation.matched_filter.core import apply_image

    if band_dim not in cube.dims:
        raise ValueError(
            f"apply_image_xarray: DataArray has no dim named {band_dim!r}; "
            f"dims are {cube.dims}."
        )

    def _kernel(arr: np.ndarray) -> np.ndarray:
        # arr has shape (..., n_bands) where ... is the non-band dims. We
        # reshape once to 3-D (H, W, B) with H = product of non-band dims, W=1
        # so apply_image's einsum works uniformly.
        flat = arr.reshape(-1, 1, arr.shape[-1])
        scores = apply_image(flat, mean=mean, cov_op=cov_op, target=target)
        return np.asarray(scores).reshape(arr.shape[:-1])

    return xr.apply_ufunc(
        _kernel,
        cube,
        input_core_dims=[[band_dim]],
        output_core_dims=[[]],
        exclude_dims={band_dim},
        dask=dask,
        output_dtypes=[np.float64],
    )


def open_multi_scene(
    paths: str | Iterable[str],
    *,
    band_dim: str = "band",
    variable: str | None = None,
    batch_dim: str = "scene",
    **open_kwargs,
) -> Iterator[np.ndarray]:
    """Yield ``(n_pixels, n_bands)`` batches from a glob of NetCDF / Zarr scenes.

    Opens with :func:`xarray.open_mfdataset` (Dask-backed, lazy), iterates one
    scene at a time, and yields a flattened 2-D batch suitable for
    :class:`~plume_simulation.matched_filter.streaming.WelfordAccumulator.update`.

    Parameters
    ----------
    paths
        A glob string (e.g. ``"scenes/*.nc"``) or an iterable of paths.
    band_dim
        Name of the spectral dimension on the stored DataArray.
    variable
        If the file contains multiple variables, select this one. If ``None``,
        uses the first data variable.
    batch_dim
        Name of the concatenation dimension (``open_mfdataset`` parameter).
    **open_kwargs
        Forwarded to :func:`xarray.open_mfdataset`.
    """
    import xarray as xr

    ds = xr.open_mfdataset(paths, combine="nested", concat_dim=batch_dim, **open_kwargs)
    if variable is None:
        var_name = next(iter(ds.data_vars))
        da = ds[var_name]
    else:
        da = ds[variable]
    if band_dim not in da.dims:
        raise ValueError(
            f"open_multi_scene: variable has no dim {band_dim!r}; dims are {da.dims}."
        )
    for idx in range(da.sizes[batch_dim]):
        scene = da.isel({batch_dim: idx}).values
        if scene.ndim == 3:
            yield scene.reshape(-1, scene.shape[-1])
        elif scene.ndim == 2:
            yield scene
        else:
            raise ValueError(
                f"open_multi_scene: scene at index {idx} has unexpected shape {scene.shape}."
            )
