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
# title: "Recipe — torch Dataset over a SpatialPatcher"
# ---
#
# # Recipe: torch `Dataset` over a `SpatialPatcher`
#
# `geopatcher` ships **primitives, not adapters** — there is no
# `PatchDataset` or `PatchLoader` class. This page shows how to wire
# the primitives into a torch `Dataset` in ~30 lines so the framework
# choice stays in user code and the patcher stays framework-free.
#
# Install requirements (not pulled in by `geopatcher`):
#
# ```
# pip install torch
# ```
#
# The three primitives this recipe uses:
#
# | Primitive | Role |
# |---|---|
# | `patcher.anchors(field)` | `Dataset.__len__` + indexable lookup |
# | `patcher.patch_at(field, anchor)` | `Dataset.__getitem__(i)` |
# | `patcher.merge(patches, domain)` | Stitch model outputs back into a field |

# %%
import geopatcher as gp
import numpy as np
import rasterio
import torch
from georeader.geotensor import GeoTensor
from torch.utils.data import DataLoader, Dataset


arr = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
field = gp.RasterField(
    GeoTensor(
        values=arr,
        transform=rasterio.Affine.identity(),
        crs="EPSG:32630",
    )
)

patcher = gp.SpatialPatcher(
    geometry=gp.SpatialRectangular(size=(32, 32)),
    sampler=gp.SpatialRegularStride(
        step=16
    ),  # overlap; (256-32)/16+1 = 15 covers the full domain
    window=gp.SpatialHann(),
    aggregation=gp.SpatialOverlapAdd(),
)


# %% [markdown]
# ## Map-style `Dataset` (lazy `patch_at`)
#
# `patch_at` reads one patch per `__getitem__` — right for huge
# fields that don't fit in RAM. Anchors are deterministic given the
# sampler's seed (or trivially so for `SpatialRegularStride`), so
# the dataset is replayable across runs.


# %%
class PatcherDataset(Dataset):
    """Random-access dataset over a `SpatialPatcher` + `Field`.

    Lazy reads via `patch_at`: each `__getitem__` triggers one
    `Field.select` call. Right shape for cloud-hosted COGs where
    materialising the whole iterator would download everything up
    front.
    """

    def __init__(self, patcher: gp.SpatialPatcher, field: gp.RasterField):
        self.patcher = patcher
        self.field = field
        self.anchor_list = patcher.anchors(field)

    def __len__(self) -> int:
        return len(self.anchor_list)

    def __getitem__(self, idx: int):
        patch = self.patcher.patch_at(self.field, self.anchor_list[idx])
        return {
            "data": torch.from_numpy(np.asarray(patch.data).copy()),
            "weights": torch.from_numpy(np.asarray(patch.weights).copy()),
            "anchor": patch.anchor,
        }


ds = PatcherDataset(patcher, field)
print(f"dataset length: {len(ds)}")
print(f"first item shape: {ds[0]['data'].shape}, anchor={ds[0]['anchor']}")


# %% [markdown]
# ## `DataLoader` with a custom collate
#
# The default `torch.utils.data.default_collate` doesn't handle
# tuple anchors gracefully, so we provide one. Workers are safe
# because `patcher.patch_at` only reads the field — no shared state
# is mutated.


# %%
def collate(items):
    return {
        "data": torch.stack([i["data"] for i in items]),
        "weights": torch.stack([i["weights"] for i in items]),
        "anchors": [i["anchor"] for i in items],
    }


loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate)
for batch in loader:
    print(batch["data"].shape, len(batch["anchors"]))
    break


# %% [markdown]
# ## Stitching model outputs back
#
# Run the model per batch, wrap each output as a `Patch`, and feed
# the whole sequence to `patcher.merge`. The patcher's
# `OverlapAdd` aggregation does the windowed stitch.


# %%
# Identity "model" for the demo
def model(x):
    return x


out_patches = []
for batch in DataLoader(ds, batch_size=8, collate_fn=collate):
    outputs = model(batch["data"])
    for i, anchor in enumerate(batch["anchors"]):
        ref = patcher.patch_at(field, anchor)
        out_patches.append(
            gp.Patch(
                data=outputs[i].numpy(),
                anchor=anchor,
                indices=ref.indices,
                weights=ref.weights,
            )
        )

stitched = patcher.merge(out_patches, field.domain)
print(f"stitched shape: {stitched.shape}")

# OverlapAdd with a Hann window reconstructs the *interior* exactly
# under identity ops — the outermost rows/cols are attenuated
# because Hann is zero at its boundary cells. Real inference
# pipelines accept this; if you need exact-to-the-edge
# reconstruction, use SpatialBoxcar instead.
interior = stitched[16:-16, 16:-16]
np.testing.assert_allclose(interior, arr[16:-16, 16:-16], rtol=1e-5)
print("identity model → interior reconstructs exactly ✔")

# %% [markdown]
# ## When to materialise instead
#
# For small fields (≲ a few thousand patches), the lazy
# `patch_at`-per-`__getitem__` form pays the per-item read overhead
# repeatedly. The materialise-once form is faster:
#
# ```python
# class MaterializedPatcherDataset(Dataset):
#     def __init__(self, patcher, field):
#         self.patches = list(patcher.split(field))
#
#     def __len__(self):
#         return len(self.patches)
#
#     def __getitem__(self, idx):
#         p = self.patches[idx]
#         return {"data": torch.from_numpy(np.asarray(p.data).copy()),
#                 "anchor": p.anchor}
# ```
#
# Rule of thumb: pick the lazy form for cloud-hosted readers
# (`AsyncRasterField`, `RioXarrayField` on remote zarr) and the
# materialised form for in-RAM `GeoTensor` data.
