# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# title: "Recipe — JAX vmap over a SpatialPatcher"
# ---
#
# # Recipe: JAX `vmap` over a `SpatialPatcher`
#
# Builds a `(N, *patch_shape)` batched array suitable for
# `eqx.filter_vmap` / `jax.vmap`. `geopatcher` does not depend on
# JAX — bring your own:
#
# ```
# pip install jax equinox
# ```
#
# The primitives this recipe uses:
#
# | Primitive | Role |
# |---|---|
# | `patcher.anchors(field)` | Indexable anchor list |
# | `patcher.patch_at(field, anchor)` | Per-anchor read |
# | `gp.stack_patches(patches)` | Stack to `(N, H, W)` for `vmap` |

# %%
import equinox as eqx
import geopatcher as gp
import jax
import jax.numpy as jnp
import numpy as np
import rasterio
from georeader.geotensor import GeoTensor


arr = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
field = gp.RasterField(
    GeoTensor(values=arr, transform=rasterio.Affine.identity(), crs="EPSG:32630")
)

patcher = gp.SpatialPatcher(
    geometry=gp.SpatialRectangular(size=(32, 32)),
    sampler=gp.SpatialRegularStride(step=32),  # non-overlapping for vmap
    window=gp.SpatialBoxcar(),
    aggregation=gp.SpatialSum(),
)

# %% [markdown]
# ## Materialise then stack
#
# Patcher iteration is Python — not traced. Materialise to a list,
# then `stack_patches` to a `(N, H, W)` ndarray, then `jnp.asarray`
# to push to device.

# %%
anchors = patcher.anchors(field)
patches = [patcher.patch_at(field, a) for a in anchors]

batch_np = gp.stack_patches(patches)  # (N, 32, 32) float32
weights_np = gp.stack_patches(patches, attr="weights")
batch = jnp.asarray(batch_np)
print(f"batch shape: {batch.shape}, dtype: {batch.dtype}")


# %% [markdown]
# ## `vmap` a tiny model
#
# `eqx.filter_vmap` over the batch axis runs the same model on
# every patch in parallel. The output preserves the `(N, *)` shape.


# %%
class TinyModel(eqx.Module):
    scale: jax.Array

    def __init__(self, key):
        self.scale = jax.random.normal(key, ())

    def __call__(self, x):
        return x * self.scale


model = TinyModel(jax.random.PRNGKey(0))
outputs = eqx.filter_vmap(model)(batch)
print(f"outputs shape: {outputs.shape}, mean: {float(outputs.mean()):.3f}")

# %% [markdown]
# ## Stitching back
#
# Pull the JAX outputs back to numpy, wrap each slice as a `Patch`,
# and feed `patcher.merge`. The patcher's `SpatialSum` aggregation
# handles the assembly.

# %%
outputs_np = np.asarray(outputs)
out_patches = [
    gp.Patch(
        data=outputs_np[i],
        anchor=p.anchor,
        indices=p.indices,
        weights=p.weights,
    )
    for i, p in enumerate(patches)
]
stitched = patcher.merge(out_patches, field.domain)
print(f"stitched shape: {stitched.shape}")

# %% [markdown]
# ## Ragged geometries
#
# `stack_patches` raises a clear `ValueError` when patches have
# different shapes (e.g. `SpatialRadiusGraph`,
# `SpatialPolygonIntersection`). For those, drop `vmap` and run the
# model one patch at a time — JAX's `pad → jit → vmap` pattern is
# the right escape hatch.
