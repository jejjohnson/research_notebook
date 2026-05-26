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
# title: "Recipe — JAX Grain MapDataset over a SpatialPatcher"
# ---
#
# # Recipe: Grain `MapDataset` over a `SpatialPatcher`
#
# [Grain](https://github.com/google/grain) is a deterministic,
# distributed-friendly data loader for JAX. `geopatcher` ships
# primitives that satisfy Grain's random-access source protocol
# directly. Install:
#
# ```
# pip install grain
# ```
#
# Why Grain over a hand-rolled torch `Dataset`:
#
# - Built-in deterministic sharding for multi-host training.
# - Worker checkpoint/resume — the loader can be restarted mid-epoch
#   and pick up at the exact same anchor.
# - No torch dependency in a JAX pipeline.
#
# Grain's contract for a random-access source: `__len__` +
# `__getitem__`. Same shape as torch `Dataset` — same primitives.

# %%
import geopatcher as gp
import grain.python as grain
import numpy as np
import rasterio
from georeader.geotensor import GeoTensor


arr = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
field = gp.RasterField(
    GeoTensor(values=arr, transform=rasterio.Affine.identity(), crs="EPSG:32630")
)

patcher = gp.SpatialPatcher(
    geometry=gp.SpatialRectangular(size=(32, 32)),
    sampler=gp.SpatialRegularStride(step=24),
    window=gp.SpatialHann(),
    aggregation=gp.SpatialOverlapAdd(),
)


# %% [markdown]
# ## Implementing Grain's `RandomAccessDataSource`
#
# The same two primitives (`anchors`, `patch_at`) that fed the torch
# `Dataset` satisfy Grain's protocol verbatim. The class is
# deliberately tiny — Grain handles batching, shuffling, sharding,
# and checkpointing on top.


# %%
class PatcherSource:
    """Grain `RandomAccessDataSource` over a `SpatialPatcher`."""

    def __init__(self, patcher: gp.SpatialPatcher, field: gp.RasterField):
        self.patcher = patcher
        self.field = field
        self.anchor_list = patcher.anchors(field)

    def __len__(self) -> int:
        return len(self.anchor_list)

    def __getitem__(self, idx: int):
        patch = self.patcher.patch_at(self.field, self.anchor_list[idx])
        return {
            "data": np.asarray(patch.data),
            "weights": np.asarray(patch.weights),
            "anchor": np.asarray(patch.anchor, dtype=np.int32),
        }


source = PatcherSource(patcher, field)
print(f"source length: {len(source)}")
print(f"first item data shape: {source[0]['data'].shape}")

# %% [markdown]
# ## Driving with `grain.MapDataset`
#
# `grain.MapDataset.source(...)` wraps the random-access source.
# From there it's plain Grain — `.shuffle(seed=...)`, `.batch(n)`,
# `.repeat()`, `.prefetch()`.

# %%
dataset = grain.MapDataset.source(source).shuffle(seed=42).batch(batch_size=8)

for batch in dataset:
    # batch["data"] has shape (8, 32, 32) after default batching
    print(
        {
            k: v.shape if hasattr(v, "shape") else type(v).__name__
            for k, v in batch.items()
        }
    )
    break

# %% [markdown]
# ## Distributed sharding
#
# Grain's sharding is the killer feature over a hand-rolled loader.
# Two hosts, two workers each: each worker sees a deterministic,
# non-overlapping slice of `range(len(source))`.
#
# ```python
# shard_options = grain.ShardOptions(
#     shard_index=jax.process_index(),
#     shard_count=jax.process_count(),
#     drop_remainder=False,
# )
# dataset = grain.MapDataset.source(source).shard(shard_options).shuffle(seed=42)
# ```
#
# Because `patcher.anchors(field)` is deterministic under a fixed
# sampler seed (see `docs/patching.md` → "Determinism"), every
# worker computes the same anchor list and Grain partitions it
# cleanly. The whole job is reproducible end-to-end.

# %% [markdown]
# ## Why this stays primitive-only
#
# The same two-method protocol satisfies torch `Dataset` and Grain
# `RandomAccessDataSource`. A `PatchDataset` class in `geopatcher`
# would either pick one framework (breaking the other) or carry both
# as optional deps. Shipping `patch_at` + `anchors` and letting the
# user wire either keeps the patcher's dep graph at numpy + scipy.
