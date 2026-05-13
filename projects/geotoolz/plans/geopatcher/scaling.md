# Scaling: Lazy Reads, Streaming Writes, and Hierarchical Composition

## Overview

Global-scale reconstruction blows up memory in five predictable places along the `split → operate → merge` pipeline. Most have one-line fixes once the framework is wired correctly; the rest need a small set of well-targeted extensions. This section maps the bottlenecks, describes the fixes, identifies which combinations of inputs and outputs are actually streamable, and shows how the framework composes recursively when data doesn't fit in RAM at any single level.

### The five memory pressure points

```
                Memory pressure in the split → operate → merge pipeline
  ┌──────────────────────────────────────────────────────────────────────┐
  │                                                                      │
  │  GlobalField  ──▶  anchors  ──▶  patches  ──▶  outputs  ──▶  merge   │
  │      [1]            [2]           [3]           [4]           [5]    │
  │      ⚠️             ⚠️            ⚠️            ⚠️          ⚠️⚠️⚠️  │
  │                                                                      │
  │   full source      anchor       patches      operator       rec +    │
  │   array in RAM     list in      list in      output list    count +  │
  │                    RAM          RAM          in RAM         div (3!) │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘

   [1] Lazy backend           [2] Sampler is             [3,4] Iterator split
       (rasterio, dask)           Iterable already             + streaming flag
       ✓ free if reader            ✓ free                      📦 extension
       picked correctly      
                              [5] Disk-backed Aggregation 
                                  (framework-owned Zarr)
                                  📦 extension
```

### What goes where

| Pressure point | Solution | Where it lives | Status |
|----------------|----------|----------------|--------|
| [1] Full field in RAM | Lazy backend readers | `Field` adapter | Already supported |
| [2] Anchor list in RAM | `Sampler.anchors()` is `Iterable` | `Sampler` protocol | Already supported |
| [3] Patch list in RAM | `Patcher.split()` returns `Iterator` | `Patcher` API | Small extension |
| [4] Output list in RAM | Consume + accumulate one at a time | Pattern, not code | Documented |
| [5] Three full-size arrays | Disk-backed `Aggregation` (flag) | `Aggregation` impl | Extension |
| Single field too big | Hierarchical Patcher-of-Patchers | Composition pattern | Documented |

Three workflows result: a trivial small-data path (in-RAM, eager), a streaming inference path (lazy reads + iterator split + disk-backed aggregation), and a hierarchical path for problems that don't fit at any single level.

---

## What the framework gives you for free

Two pressure points are already addressed by the design — pick the right backend and respect the iterator nature of `Sampler`.

### Lazy reads from the backend

```python
field = RasterioField(open_cog("global_methane.tif"))   # nothing read yet
patch = field.select(window)                            # still nothing read
arr   = patch.data.values                               # ONLY now we read from disk
```

| Backend | Lazy by default? | How to make it lazy |
|---------|------------------|---------------------|
| `RasterioField` (COG) | ✓ | windowed I/O is the API |
| `RioXarrayField` + dask | ✓ | `open_rasterio(..., chunks={...})` |
| `GeoTensorField` | ✗ (materialized) | use `RasterioField` instead |
| `XarrayField` + Zarr | ✓ | `open_zarr(...)` |
| `XarrayField` + netCDF | ✗ | `open_dataset(..., chunks={...})` to dask-back |
| `GeoPandasField` | ✗ (loads full GeoDataFrame) | GeoParquet + row-group filtering |
| `XvecField` | ✓ if underlying xarray is dask-backed | inherits xarray laziness |

Rule of thumb: rasters and gridded data are lazy for free if you pick the right reader. Vectors and points require explicit work.

### Sampler anchors are already an iterator

```python
class Sampler(Protocol):
    def anchors(self, domain, geometry) -> Iterable[Anchor]: ...   # not list!
```

`RegularStride`, `Random`, and `PoissonDisk` are all generator-friendly. The protocol already commits to laziness; no extra work needed.

---

## Four extensions

### 1. `Patcher.split` returns an Iterator

A single-line change with system-wide consequences:

```python
@dataclass
class Patcher:
    def split(self, field: Field) -> Iterator[Patch]:        # was list[Patch]
        for a in self.sampler.anchors(field.domain, self.geometry):
            idx = self.geometry.neighborhood(field.domain, a)
            w   = self.window.weights(self.geometry)
            yield Patch(data=field.select(idx), anchor=a, indices=idx, weights=w)
```

Materializing remains a one-liner when you want it: `list(patcher.split(field))`. The default API forces streaming consumption.

### 2. Disk-backed `Aggregation` (flag on the existing class)

`OverlapAdd` accumulates in RAM by default. With `streaming=True` and a `target_path`, it accumulates into a framework-managed Zarr store on disk. The math is identical; only the storage moves.

```python
@dataclass
class OverlapAdd(Aggregation):
    streaming:   bool = False
    target_path: str | None = None        # where the Zarr store goes
    chunks:      tuple[int, ...] | None = None

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Patch], domain: Domain) -> Field:
        if self.streaming:
            return self._merge_streaming(patches, domain)
        return self._merge_in_memory(patches, domain)

    def _merge_streaming(self, patches, domain):
        rec   = zarr.open(f"{self.target_path}/rec.zarr",   mode="w",
                          shape=domain.shape, chunks=self.chunks, dtype="float32")
        count = zarr.open(f"{self.target_path}/count.zarr", mode="w",
                          shape=domain.shape, chunks=self.chunks, dtype="float32")
        for p in patches:                                # one patch at a time
            target = _resolve_indexer(p.indices)         # Window → slice, etc.
            rec[target]   += p.data * p.weights
            count[target] += p.weights
        for ck in iter_chunks(rec):                      # chunked, in-place division
            rec[ck] = np.divide(rec[ck], count[ck],
                                out=rec[ck], where=count[ck] > 0)
        return ZarrField(rec)
```

The flag + framework-managed store address bottlenecks [3], [4], [5] in one mechanism. The user passes a target path and gets back a `Field` they can read; they don't open, close, or clean up anything. Memmap is a drop-in zero-dependency alternative (`zarr.open` → `np.memmap`); the interface is identical.

### 3. Hierarchical reconstruction (Patcher-of-Patchers)

When even one super-tile is larger than convenient, the framework composes recursively. No new class — just two `Patcher` instances at different scales:

```
   GlobalField (petabyte-scale, lazy on disk)
        │
        ▼
   ┌─────────────────────────┐
   │   Outer Patcher         │   coarse super-tiles (e.g. 4×4 of globe)
   │   geometry = (4096²)    │
   │   sampler  = RegStride  │
   │   aggregation = None    │   (write per super-tile; no global merge)
   └────────────┬────────────┘
                │ for each super-tile (streaming):
                ▼
          ┌──────────┐ ◄── one super-tile loaded lazily, fits in RAM
          │ Super-   │
          │  Tile    │
          └────┬─────┘
               │
               ▼
        ┌─────────────────────────┐
        │   Inner Patcher         │   fine patches within super-tile
        │   geometry = (256²)     │
        │   sampler  = RegStride  │
        │   aggregation = Overlap │   in-RAM accumulator, super-tile-sized
        │                Add()    │
        └────────────┬────────────┘
                     │ for each patch (streaming):
                     ▼
               ┌──────────┐
               │  Patch   │ ──▶ operator ──▶ result
               └──────────┘
                                                 │
              ┌──────────────────────────────────┘
              ▼
        ┌──────────────────────┐
        │ Inner OverlapAdd     │   in-RAM, super-tile shaped
        └─────────┬────────────┘
                  │
                  ▼
          write to global Zarr at super-tile offset, free RAM, next tile
                  │
                  ▼
        ┌──────────────────────┐
        │   Zarr store         │   full global field, on disk
        └──────────────────────┘
```

User code is ~20 lines:

```python
outer = Patcher(geometry=Rectangular((4096, 4096)),
                sampler=RegularStride((4096, 4096)),
                window=Boxcar(), aggregation=None)

inner = Patcher(geometry=Rectangular((256, 256)),
                sampler=RegularStride((192, 192)),
                window=Hann(), aggregation=OverlapAdd())   # in-RAM

global_store = zarr.open("global.zarr", mode="w", shape=field.shape, ...)

for super_tile in outer.split(field):
    super_acc = inner.merge(
        (operator(p) for p in inner.split(super_tile.data)),
        super_tile.indices,
    )                                                       # super-tile-sized
    global_store[_resolve_indexer(super_tile.indices)] = super_acc.values
    del super_acc                                            # free for next tile
```

The recursion uses *only existing pieces* — `Patcher`, `split`, `merge`, `Field`. The pattern is documented; no `HierarchicalPatcher` class is introduced. If real workflows show a consistent shape across 2–3 use cases, abstract later.

Coarse-to-fine reconstruction (e.g. *Multi-Stage Progressive Image Restoration*, CVPR multi-scale reconstruction) is a related but distinct idea: a single field reconstructed at multiple resolutions, with each stage's output feeding the next. That's an *operator* pattern, not a *Patcher* pattern. The framework supports it (different `Rectangular` sizes per stage), but the staging is orchestrated by the operator, not by Patcher composition.

### 4. `streaming_safe` property on `Aggregation`

Not every aggregation can be done one patch at a time. The next subsection explains why — but the property itself is simple:

```python
class Aggregation:
    streaming_safe: ClassVar[bool] = False     # conservative default

class OverlapAdd(Aggregation):
    streaming_safe = True
```

When a user sets `streaming=True` on an aggregation with `streaming_safe = False`, the framework **emits a warning** rather than raising. The warning explains the issue and points to two-pass or approximation alternatives. Hard-erroring is too aggressive for a property that can depend on use case; silent failure is exactly what this property is designed to prevent.

---

## The streaming asymmetry: it's about the output, not the input

The previous extensions raise a natural question — *what makes a workflow streamable in the first place?* The answer is a single key insight that's worth pulling out explicitly:

> **Streaming is easy when the output address space is preallocatable, and hard when it isn't.**

The *input* domain doesn't matter — raster, grid, points, polygons all stream fine, because each patch's input is independently readable through `Field.select`. The asymmetry is on the *output* side: the `Aggregation` has to write each patch's contribution somewhere, and "somewhere" needs to exist before the patch arrives.

```
   ANY input          Patcher           OUTPUT structure
   ────────           ───────           ────────────────
   raster                                ┌────────────────┐
   grid           ──▶  split    ──▶  ──▶ │ PREALLOCATABLE │ ─▶ STREAMS ✓
   points              + op              │  • regular     │   (write to
   polygons                              │    grid        │    known
                                         │  • fixed key   │    address)
   (lazy reads                           │    set         │
    are free)                            │  • known       │
                                         │    query pts   │
                                         └────────────────┘
                                         
                                         ┌────────────────┐
                                         │ NON-PREALLOC.  │ ─▶ DOES NOT ✗
                                         │  • dynamic     │   (no address
                                         │    points      │    to write
                                         │  • growing     │    to)
                                         │    graph       │
                                         │  • locations   │
                                         │    self-       │
                                         │    determined  │
                                         └────────────────┘

   The bottleneck lives on the OUTPUT side.
```

### The matrix of common cases

| Input domain | Output domain | Streaming? | Aggregation |
|--------------|---------------|------------|-------------|
| Raster | Raster | ✓ | `OverlapAdd` |
| Raster | Polygons (fixed) | ✓ | `ByIndex` |
| Raster | Points (fixed query set) | ✓ | `ByIndex` |
| **Points (radius/knn)** | **Raster** | **✓** | **`OverlapAdd`** |
| Points | Points (fixed set) | ✓ | `ByIndex` |
| Points | Points (dynamic / growing) | ✗ | — |
| Anything | Anything (dynamic) | ✗ | — |

The bold row is worth flagging — predicting from scattered observations (Argo, in-situ stations, swath samples) onto a regular grid is one of the most common geoscience workflows, and it streams cleanly. The radius/knn part is about how the input is gathered per patch; it doesn't constrain the output.

This is exactly what the `streaming_safe` flag captures. It's really asking "is this aggregation's output address space preallocatable?" `OverlapAdd`-into-Zarr is safe; `ByIndex` is safe; a hypothetical `Scatter` writing to dynamically-determined locations is not.

### One subtlety: points → grid at scale

Even when the *output* streams cleanly, the *input* may not fit in RAM. With 100M Argo profiles you can't hold a single KDTree over the full point cloud. The pattern is the same hierarchical Patcher-of-Patchers from above, but applied for a different reason: the outer Patcher partitions the **output grid** into super-tiles, and for each super-tile you load only the points whose neighborhoods could touch that region (via a disk-backed spatial index, or by spatially partitioning the points to begin with).

```
1. Outer Patcher  : super-tile the output grid
2. Per super-tile : load only the points whose neighborhoods could touch it
3. Inner Patcher  : build radius/knn patches around grid cells in this super-tile
4. Operator       : predict per cell (GP, GNN, neural process, …)
5. OverlapAdd     : stream into Zarr for the super-tile's grid region
```

Output streams via `OverlapAdd`; input streams via the outer Patcher selecting which points to load. Two streaming mechanisms, working at different levels of the recursion.

### The genuinely hard case

Streaming breaks when the output structure is determined by the predictions themselves: adaptive-mesh refinement, active learning, on-the-fly point insertion. You can't preallocate, and each prediction may need to spatially-index against all previous predictions. That's not really streamable in our sense; it needs different abstractions (dynamic spatial structures, incremental indexing).

The pragmatic workaround is **two-pass**: a first pass with a static (possibly coarse) output set using a streaming-safe aggregation, then a second pass that refines using the first-pass result. The framework supports this naturally — two Patchers, two passes — but it's not automatic. The user opts in.

The same two-pass pattern covers **global-context aggregations** (global normalization, attention across all patches, learned aggregations needing the full batch): pass 1 computes the global statistic with a streaming-safe aggregation, pass 2 uses it. Approximation alternatives (t-digest for quantiles, sketch-based attention) are useful when two passes are too expensive.

---

## Two workflows, not one "streaming mode"

The strategies above support two distinct use cases that should be documented separately.

### Training-time streaming (DataLoader pattern)

One patch at a time, no reconstruction. Patches are consumed by a model and thrown away. The adapter is small:

```python
class PatchDataset:
    def __init__(self, patcher: Patcher, field: Field):
        self.patcher = patcher
        self.field   = field
        # Anchor list materialized once (small); patch data never is
        self.anchors = list(patcher.sampler.anchors(field.domain, patcher.geometry))

    def __len__(self): return len(self.anchors)

    def __getitem__(self, i: int) -> Patch:
        a   = self.anchors[i]
        idx = self.patcher.geometry.neighborhood(self.field.domain, a)
        w   = self.patcher.window.weights(self.patcher.geometry)
        return Patch(data=self.field.select(idx), anchor=a, indices=idx, weights=w)
```

Drop-in for `torch.utils.data.Dataset` or any indexable dataset. The DataLoader handles batching, shuffling, and parallel I/O; the Patcher provides the patch definition. Memory footprint: one anchor list (small) + one patch per worker (small).

### Inference-time streaming (reconstruct the global field)

Lazy reads + iterator split + streaming-flagged disk-backed aggregation, optionally wrapped in hierarchical composition.

```python
inference = Patcher(
    geometry    = Rectangular((256, 256)),
    sampler     = RegularStride((192, 192)),
    window      = Hann(),
    aggregation = OverlapAdd(streaming=True, target_path="global.zarr",
                             chunks=(256, 256)),
)

result_field = inference.merge(
    (operator(p) for p in inference.split(field)),
    field.domain,
)
# result_field is a ZarrField — read it back with xarray if you want
```

If a single super-tile is too big, wrap this block in the outer Patcher of section 3.

---

## Ragged geometries: a streaming non-issue

One more thing worth flagging up front: **ragged geometries** (`SphericalCap`, `RadiusGraph`, `PolygonIntersection`) stream fine at the Patcher level — each patch is independently computed and accumulated. The complication is downstream batching: if your operator is a GNN expecting padded batches, you need a batching layer between split and operator (jraph, torch_geometric handle this natively).

So ragged input ≠ non-streamable. Ragged input + GNN-style batching is a separate engineering problem, not a Patcher problem.

---

## Quick wins (small fixes that just become defaults)

- **In-place division** in `OverlapAdd._merge_in_memory`. There is no reason to allocate the third full-size array — `np.divide(rec, count, out=rec, where=count > 0)` is the right primitive.
- **Lazy coordinate generation** is implicit once `Sampler.anchors()` is treated as `Iterable`; nothing to do beyond removing any old code that materialized them eagerly.

---

## Open questions

A few choices left deliberately open, to revisit once real workflows force the shape:

- **Two-pass operator support.** Common enough (any normalization, any cross-patch attention, refinement loops) that an explicit `TwoPassPatcher` pattern might earn its keep. Defer until a real use case lands.
- **Coarse-to-fine reconstruction as a workflow.** Operator-led, but interacts with the Patcher (different geometries per stage). Worth its own short section once a concrete use case (multi-scale methane retrieval, for example) is in flight.
- **Streaming-safety: warning vs hard error vs configurable.** Currently warning. A global `strict` flag would let CI-like contexts fail fast without per-call API noise.
- **Distributed reconstruction.** Zarr writes parallelize across workers via dask; the framework doesn't need to know. The hierarchical pattern is the natural unit of distribution (one worker per super-tile). Worth documenting as a recipe rather than building in.

---

## Summary

The framework handles scaling through three orthogonal mechanisms: **backend laziness** (free — just pick the right reader), **iterator-based traversal** (small protocol change to `Patcher.split`), and **disk-backed aggregation** (flag on `Aggregation`, framework-managed Zarr). When the global field doesn't fit at any single level, the same Patcher abstraction **composes recursively** as a Patcher-of-Patchers, with no new class introduced.

The key conceptual frame is the **streaming asymmetry**: input domains stream freely, but the output structure has to be preallocatable for streaming to work. Regular grids, fixed key sets, and known query points are preallocatable; dynamic point clouds and growing graphs are not. The `streaming_safe` property on `Aggregation` encodes this distinction, and the two-pass pattern is the escape hatch for the hard cases.

Two distinct workflows — **training-time streaming** via a `PatchDataset` adapter, and **inference-time streaming** via flagged disk-backed aggregation — cover the common cases.


# Streaming-Compatible Aggregations

`OverlapAdd` is the canonical example, but there's a much richer family of aggregations that stream. The structural property they share: each can be expressed as a fold over one or more parallel accumulators, where every per-patch update is independent of every other.

Two paths to streaming follow from this — **exact** (associative reductions) and **approximate** (bounded-error sketches). The split matters: it tells you when streaming preserves the answer and when it preserves it only up to a guaranteed error bound.

```
                     STREAMING-COMPATIBLE AGGREGATIONS
                     ═════════════════════════════════

           EXACT                              APPROXIMATE
     ────────────────                ─────────────────────────
     Associative reductions          Bounded-error sketches
     (one or more parallel           (substitute for stats that
      accumulators, no error)         can't reduce associatively)

     ┌─────────────────────┐         ┌──────────────────────────┐
     │  Sum, WeightedSum   │         │  ApproxQuantile          │
     │  Max, Min, Range    │         │   (t-digest, KLL, GK)    │
     │  Mean, Variance     │         │  ApproxCardinality       │
     │  OverlapAdd         │         │   (HyperLogLog)          │
     │  HardVote, SoftVote │         │  ApproxMode              │
     │  InvVarWeightedMean │         │   (Misra-Gries)          │
     └─────────────────────┘         │  StreamingHistogram      │
                                     │  Reservoir(k)            │
     no information loss             └──────────────────────────┘
                                       ε-bounded error
                                              │
                                              ▼
                                     substitute for stats
                                     that DON'T stream exactly:
                                     median, percentiles,
                                     exact mode, k-th order stat
```

---

## Exact streaming aggregations

Associative + commutative reductions. The per-patch update commutes with every other patch, so streaming gives bit-identical results to a batch computation (modulo floating-point order).

### Monoidal (one accumulator)

| Aggregation | Accumulator | Per-patch update | Final |
|---|---|---|---|
| `Sum` | `acc` | `acc[idx] += data` | `acc` |
| `WeightedSum` | `acc` | `acc[idx] += data * w` | `acc` |
| `Max` | `acc` (init −∞) | `acc[idx] = maximum(acc[idx], data)` | `acc` |
| `Min` | `acc` (init +∞) | `acc[idx] = minimum(acc[idx], data)` | `acc` |
| `Any` / `All` (bool) | `acc` (init 0/1) | `acc[idx] \|= data` / `&=` | `acc` |
| `Count` | `acc` (int) | `acc[idx] += 1` | `acc` |

### Compound monoidal (parallel accumulators, combine at the end)

Two or more monoidal accumulators run in parallel; the final step is a chunked map.

| Aggregation | Accumulators | Final op |
|---|---|---|
| `Mean` | `sum`, `count` | `sum / count` |
| `OverlapAdd` | `weighted_sum`, `weight` | `weighted_sum / weight` |
| `Range` | `min`, `max` | `max - min` |
| `Variance` (Welford) | `mean`, `M2`, `count` | `M2 / (count - 1)` |
| `InvVarWeightedMean` | `Σ(μ/σ²)`, `Σ(1/σ²)` | `acc1 / acc2`, `1 / acc2` |

### Categorical / voting

| Aggregation | Accumulator | Per-patch update | Final |
|---|---|---|---|
| `HardVote` | `acc[K, ...]` | `acc[argmax(data), idx] += 1` | `argmax(acc, axis=0)` |
| `SoftVote` | `acc[K, ...]` | `acc[:, idx] += data_probs` | `argmax(acc, axis=0)` |

Both stream cleanly; just `K` channels instead of one. Useful for ensembling classification models patch-by-patch.

---

## Approximate streaming aggregations (sketches)

The aggregations above all reduce associatively. Quantiles, mode, cardinality, and the full empirical distribution do *not* — exact computation needs all values per cell. Sketches sidestep this by maintaining a compact summary per cell with a bounded error guarantee.

| Aggregation | Sketch backend | Error | Per-cell memory |
|---|---|---|---|
| `ApproxQuantile` | t-digest, KLL, GK | ε-relative on rank | O(log(1/ε)/ε) |
| `ApproxCardinality` | HyperLogLog | ~1.6%/√m | O(2^p) |
| `ApproxMode` / heavy hitters | Misra-Gries, Space-Saving | bounded freq error | O(k) |
| `StreamingHistogram` | equi-width or t-digest | bin error | O(bins) |
| `Reservoir(k)` | reservoir sampling | sampling variance | O(k) |

Sketches **substitute for**, but do not equal, their exact counterparts:

| Exact (not streamable) | Approximate substitute (streamable) |
|------------------------|--------------------------------------|
| Median, percentiles | `ApproxQuantile(q=0.5)`, t-digest |
| Mode | `ApproxMode` via Misra-Gries |
| Unique-value count | `ApproxCardinality` via HyperLogLog |
| Full empirical distribution | `StreamingHistogram` |
| Arbitrary downstream statistic | `Reservoir(k)` + post-hoc computation |

This is the central tradeoff: **you cannot have exact median in a streaming pipeline, but you can have ε-bounded median that does stream.** Naming the operators `ApproxX` makes the tradeoff explicit at the call site rather than hiding it.

One practical gotcha: per-cell sketches can dominate memory. A t-digest is ~1 KB per cell; on a 720×1440 grid that's ~1 GB just for sketches. **Sketch per Zarr chunk** rather than per cell is usually the right level — one t-digest summarizing a (256, 256) chunk gives you per-region percentiles without the per-cell overhead.

---

## Bayesian merging: `InvVarWeightedMean`

Worth flagging on its own because it's the natural answer to a common geoscience workflow: merging overlapping local-Bayesian predictions (local GPs, BNNs, neural processes) into a global field while preserving uncertainty.

When each patch produces a mean and variance, the optimal Gaussian combination is Kalman-style inverse-variance weighting:

$$\mu_{\text{global}} = \frac{\sum_i w_i \, \mu_i / \sigma_i^2}{\sum_i w_i / \sigma_i^2}, \qquad \sigma_{\text{global}}^2 = \frac{1}{\sum_i w_i / \sigma_i^2}$$

with patch window weights $w_i$ as an extra modeling knob. This streams with two accumulators:

```python
mu_i, var_i = operator(patch)            # per-patch Bayesian output

mu_acc[idx]   += patch.weights * mu_i / var_i
prec_acc[idx] += patch.weights / var_i

# at the end
mu_global  = mu_acc / prec_acc
var_global = 1     / prec_acc
```

For Case 2 local-GP workflows (no pooling, per-patch $\theta_i$), this is the right way to stitch overlapping posteriors into a global field with calibrated uncertainty. Both the mean and the variance fall out of the streaming reduction at no extra cost.

---

## What stays non-streamable

Some aggregations genuinely cannot be done one patch at a time — they need either the full per-cell history or cross-patch context.

| Aggregation | Why it fails | Streaming substitute |
|---|---|---|
| Exact median, exact percentiles | Needs the full per-cell distribution | `ApproxQuantile` |
| Exact mode | Needs all values | `ApproxMode` |
| Order statistics (k-th smallest) | Needs sorted per-cell list | `ApproxQuantile` |
| `Learned(NN)` aggregation | Typically needs full batch of outputs | Two-pass or per-region |
| Bayesian Model Averaging (exact) | Needs all posteriors jointly | `InvVarWeightedMean` (Gaussian approx) |
| Global normalization | Needs global statistic before applying | Two-pass: compute stat, then normalize |

These are the honest `streaming_safe = False` cases. The two-pass pattern is the general escape hatch (compute the global stat in pass 1 with a streaming-safe aggregation, then apply in pass 2). Sketches are the specific escape hatch for distribution statistics.

---

## Suggested framework exposure

```python
# ── Exact: single-accumulator monoidal ──
class Sum                (Aggregation): streaming_safe = True
class WeightedSum        (Aggregation): streaming_safe = True
class Max                (Aggregation): streaming_safe = True
class Min                (Aggregation): streaming_safe = True

# ── Exact: compound monoidal ──
class Mean               (Aggregation): streaming_safe = True
class OverlapAdd         (Aggregation): streaming_safe = True
class Variance           (Aggregation): streaming_safe = True   # Welford
class InvVarWeightedMean (Aggregation): streaming_safe = True   # Bayesian merge

# ── Exact: categorical ──
class HardVote           (Aggregation): streaming_safe = True
class SoftVote           (Aggregation): streaming_safe = True

# ── Approximate: sketches ──
class ApproxQuantile     (Aggregation): streaming_safe = True   # t-digest backend
class ApproxCardinality  (Aggregation): streaming_safe = True   # HyperLogLog
class ApproxMode         (Aggregation): streaming_safe = True   # Misra-Gries
class StreamingHistogram (Aggregation): streaming_safe = True
class Reservoir          (Aggregation): streaming_safe = True

# ── Honest non-streamable ──
class Median             (Aggregation): streaming_safe = False  # use ApproxQuantile
class Mode               (Aggregation): streaming_safe = False  # use ApproxMode
class Learned            (Aggregation): streaming_safe = False  # use two-pass
```

Users who want streaming median write `ApproxQuantile(q=0.5)`. The naming makes the exact/approximate choice explicit at the call site rather than burying it in a flag.

---

## A bonus property of monoidal aggregations: distributability

Every exact streaming aggregation above is **trivially distributable**. Each worker maintains a local accumulator over its slice of patches; the final step is a tree-reduce of the accumulators. For compound monoidals (`Mean` = Sum / Count), each worker keeps both Sum and Count, the Sums are reduced separately, the Counts are reduced separately, and the division happens once at the end.

This means the hierarchical Patcher-of-Patchers pattern parallelizes with no extra machinery — one worker per super-tile, tree-reduce the accumulators when they come back, finalize once. The Aggregation classes designed for streaming get distribution for free.
