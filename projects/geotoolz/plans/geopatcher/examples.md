# Patcher Examples: Geometry × Data

The natural way to navigate the Patcher's design space is by pairing a `PatchGeometry` with a `Domain`. Each pairing has its own natural Sampler/Window/Aggregation choices and its own type signature for the resulting `Patch`. This report walks through the most useful combinations.

## Types reference

Each `Patch` is generic in three things — anchor type, indices type, data type — which together fully characterize it:

```python
from dataclasses import dataclass
from typing import Generic, TypeVar

AnchorT  = TypeVar("AnchorT")     # where the patch sits        (from Sampler)
IndicesT = TypeVar("IndicesT")    # how to subset the field     (from Geometry × Domain)
DataT    = TypeVar("DataT")       # the patch contents          (from Field backend)

@dataclass
class Patch(Generic[AnchorT, IndicesT, DataT]):
    anchor:  AnchorT
    indices: IndicesT
    data:    DataT
    weights: Array                # same spatial shape as data's locality dims
```

Throughout, `Array[D1, D2, ...]` is informal shape annotation (think `Float[Array, "D1 D2"]` in jaxtyping). Concrete instantiations are shown per example.

A small helper type used below:

```python
@dataclass
class MaskedWindow:
    window: Window                # bounding rectangle (rasterio)
    mask:   Array                 # bool[h, w], True inside the polygon
```

---

## Section 1 — Rectangular geometry

The most common case: axis-aligned boxes on gridded data.

### 1a. Rectangular × RasterDomain — tiling a satellite scene

Extract training tiles from a multi-band raster on disk.

```python
field: Field[RasterDomain] = RasterioField(open_cog("scene.tif"))
# underlying raster: (C, H, W) = (12, 12000, 12000), lazy COG

patcher = Patcher(
    geometry    = Rectangular(size=(256, 256)),
    sampler     = RegularStride(step=(256, 256)),
    window      = Boxcar(),
    aggregation = OverlapAdd(),
)

patches: list[Patch[
    tuple[int, int],          # AnchorT  = (row, col) top-left pixel
    Window,                   # IndicesT = rasterio Window
    Array,                    # DataT    = Array[12, 256, 256] = (C, H, W)
]] = list(patcher.split(field))
```

A CNN consumes `patch.data` directly. `OverlapAdd` is only used if you reconstruct a full output field at inference.

### 1b. Rectangular × GridDomain — patching a reanalysis cube

Cut a (time × level × lat × lon) ERA5 field into spatial patches for downscaling.

```python
field: Field[GridDomain] = XarrayField(era5)
# underlying DataArray: dims = (time, level, lat, lon), shape (T, L, 720, 1440)

patcher = Patcher(
    geometry    = Rectangular(size=(64, 64)),               # over (lat, lon)
    sampler     = RegularStride(step=(48, 48)),             # 16-cell overlap
    window      = Hann(),
    aggregation = OverlapAdd(),
)

patches: list[Patch[
    dict[str, int],           # AnchorT  = {"lat": 100, "lon": 50}
    dict[str, slice],         # IndicesT = {"lat": slice(100, 164), "lon": slice(50, 114)}
    XarrayField,              # DataT    = sub-DataArray of shape (T, L, 64, 64)
]] = patcher.split(field)
```

Same `Rectangular(size=(64, 64))` as the raster case — but indices are now label-aware dict-of-slices and the patch data preserves the time/level dims. Coordinate-aware operators use `patch.data` as an xarray field; numeric operators call `.values`.

---

## Section 2 — Spherical cap geometry

For lat/lon fields where rectangular patches near the poles are wrong.

### 2a. SphericalCap × GridDomain — global SST around stations

Patch a global SST field by geodesic neighborhoods around in-situ station locations.

```python
field: Field[GridDomain] = XarrayField(sst)
# global SST: dims = (time, lat, lon), shape (365, 720, 1440)

patcher = Patcher(
    geometry    = SphericalCap(radius_km=500),
    sampler     = Explicit(anchors_=station_locations),   # list[tuple[float, float]]
    window      = Gaussian(sigma=200),                    # km
    aggregation = WeightedSum(weight_fn=lambda p: p.weights),
)

patches: list[Patch[
    tuple[float, float],      # AnchorT  = (lat, lon) in degrees
    Array,                    # IndicesT = Array[N_i, 2] of (lat_idx, lon_idx)
    Array,                    # DataT    = Array[T, N_i], ragged N_i per patch
]] = patcher.split(field)
# weights: Array[N_i], Gaussian-tapered by geodesic distance from the anchor
```

The indices are no longer a `Window` or `dict[str, slice]` — they're a ragged list of `(lat_idx, lon_idx)` pairs within the cap. `N_i` varies per patch (caps near the equator hold more grid cells than caps near the poles for the same radius). This raggedness is the price of doing the sphere correctly; the spatial 2-D structure is flattened to a per-cap list of cells, which is the natural input shape for a GP or graph operator.

---

## Section 3 — Radius graph geometry

For point and polygon data where neighborhoods are defined by distance.

### 3a. RadiusGraph × PointDomain — Argo float neighborhoods

Build per-region patches from scattered float profiles for local oceanographic analysis.

```python
field: Field[PointDomain] = XvecField(argo_profiles)
# profiles: dataset with dim "profile" of length N=50000,
#   geometry coord (lon, lat); data vars: temp(depth), salinity(depth)

patcher = Patcher(
    geometry    = RadiusGraph(radius=50_000),             # 50 km, projected coords
    sampler     = PoissonDisk(min_dist=40_000),
    window      = Gaussian(sigma=20_000),
    aggregation = WeightedSum(weight_fn=lambda p: p.weights),
)

patches: list[Patch[
    Array,                    # AnchorT  = Array[2], (x, y) anchor coords
    list[int],                # IndicesT = indices into the profile dim, length k_i
    XvecField,                # DataT    = sub-dataset with profile dim = k_i
]] = patcher.split(field)
# weights: Array[k_i], Gaussian on distance from the anchor
```

`k_i` is variable — that's the nature of scattered data. Operators that handle this naturally: GPs (kernel between point coords), neural processes, GNN message passing. Operators that don't (CNN, FNO) need rasterization first.

### 3b. RadiusGraph × VectorDomain — administrative regions near a facility

For each emission facility, find every administrative polygon within 25 km.

```python
field: Field[VectorDomain] = GeoPandasField(admin_polygons)
# polygons: GeoDataFrame, N=3000 rows, MultiPolygon geometries

patcher = Patcher(
    geometry    = RadiusGraph(radius=25_000),
    sampler     = Explicit(anchors_=facility_points),     # list[shapely.Point]
    window      = Boxcar(),                               # no weighting for vectors
    aggregation = ByIndex(),                              # results keyed by anchor
)

patches: list[Patch[
    shapely.Point,            # AnchorT
    list[int],                # IndicesT = polygon row indices, length k_i
    GeoPandasField,           # DataT    = GeoDataFrame slice, k_i rows
]] = patcher.split(field)
```

Same `RadiusGraph` geometry, different `Domain` — the dispatch in `RadiusGraph.neighborhood` picks the polygon spatial-index path instead of the kdtree path. The data is now a GeoDataFrame slice; downstream operators compute area overlaps, population sums, or attribute joins per facility.

---

## Section 4 — KNN graph geometry

For point data where a fixed neighborhood size is more useful than a fixed radius.

### 4a. KNNGraph × PointDomain — fixed-k neighborhoods for GNN training

For each in-situ station, gather its k nearest neighbors as a training sample for a GNN.

```python
field: Field[PointDomain] = GeoPandasField(stations)
# stations: GeoDataFrame, N=20000 Point geometries with feature columns

patcher = Patcher(
    geometry    = KNNGraph(k=16),
    sampler     = Explicit(anchors_=stations.index.tolist()),
    window      = Boxcar(),
    aggregation = ByIndex(),
)

patches: list[Patch[
    int,                      # AnchorT  = station row index
    Array,                    # IndicesT = Array[16] of neighbor indices (fixed k)
    GeoPandasField,           # DataT    = 16-row GeoDataFrame
]] = patcher.split(field)
```

Unlike `RadiusGraph`, the patch size is uniform (k=16 always), making it directly batchable for a GNN: stack the patches into `Array[batch, 16, features]`. `ByIndex` keys outputs by station id, so each station gets one prediction.

---

## Section 5 — Polygon-intersection geometry

For vector–raster joins: the patch is the pixels falling inside a polygon.

### 5a. PolygonIntersection × RasterDomain — facility-level methane aggregation

For each facility polygon, extract the methane pixels inside it and aggregate to a per-facility number.

```python
field: Field[RasterDomain] = GeoTensorField(methane_field)
# methane: GeoTensor, shape (1, 6000, 6000), regularly gridded, georeferenced

patcher = Patcher(
    geometry    = PolygonIntersection(polygons=facilities.geometry),
    sampler     = Explicit(anchors_=facilities.index.tolist()),
    window      = Boxcar(),
    aggregation = ByIndex(),
)

patches: list[Patch[
    int,                      # AnchorT  = facility row index
    MaskedWindow,             # IndicesT = bounding Window + bool mask
    Array,                    # DataT    = Array[1, h_i, w_i] rectangular crop
]] = patcher.split(field)
# weights: Array[h_i, w_i], bool mask of pixels strictly inside the polygon
```

`MaskedWindow` does two jobs: the `window` enables lazy I/O via rasterio (read only the bounding rectangle), and the `mask` zeroes out pixels outside the polygon. Operators reduce the masked patch to summary statistics:

```python
def operator(patch) -> dict:
    pixels_in = patch.data * patch.weights              # mask-zeroed
    n         = patch.weights.sum()
    return {
        "n_pixels": int(n),
        "mean":     (pixels_in.sum() / n).item(),
        "p90":      float(np.percentile(pixels_in[patch.weights > 0], 90)),
        "area_m2":  float(n * pixel_area),
    }

results: dict[int, dict] = {p.anchor: operator(p) for p in patcher.split(field)}
```

`ByIndex` returns a `dict[facility_id, feature_dict]` rather than a reconstructed field — the natural output shape for downstream regression or tabular storage.

---

## Summary: types across all combinations

| Geometry | Domain | `AnchorT` | `IndicesT` | `DataT` | Shape regime |
|----------|--------|-----------|------------|---------|--------------|
| Rectangular | RasterDomain | `tuple[int, int]` | `Window` | `Array[C, H, W]` | **uniform** |
| Rectangular | GridDomain | `dict[str, int]` | `dict[str, slice]` | `XarrayField` | **uniform** |
| SphericalCap | GridDomain | `tuple[float, float]` | `Array[N_i, 2]` | `Array[..., N_i]` | **ragged** |
| RadiusGraph | PointDomain | `Array[2]` | `list[int]` length `k_i` | `XvecField` | **ragged** |
| RadiusGraph | VectorDomain | `shapely.Point` | `list[int]` length `k_i` | `GeoPandasField` | **ragged** |
| KNNGraph | PointDomain | `int` | `Array[k]` fixed | `GeoPandasField` | **uniform** |
| PolygonIntersection | RasterDomain | `int` | `MaskedWindow` | `Array[C, h_i, w_i]` | **ragged** |

Three patterns worth holding in your head:

1. **AnchorT follows the Sampler.** Pixel coordinates from `RegularStride`, dict-of-ints from grid samplers, `shapely.Point` from explicit polygon samplers, integer ids from entity-based samplers.
2. **IndicesT follows the Geometry × Domain dispatch.** The same `Rectangular` returns a `Window` on rasters and a `dict[str, slice]` on grids. The same `RadiusGraph` returns kdtree results on points and spatial-index results on polygons. This is exactly the multi-method dispatch from the framework's backend layer.
3. **DataT splits into uniform vs ragged shapes.** Rectangular and KNN patches are uniform → batchable for dense neural-net operators (CNN, FNO, GNN with fixed k). Spherical caps, radius-graph patches, and polygon intersections are ragged → either pad them, use jagged structures (jraph, torch_geometric), or feed them to operators that are natively ragged-friendly (GP, point-process models, message-passing GNN).

The uniform/ragged distinction is the single most consequential choice for downstream operators. The geometry tells you which path you're on.
