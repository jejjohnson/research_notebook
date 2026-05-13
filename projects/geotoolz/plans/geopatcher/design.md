# Spatiotemporal Operators in Geoscience: A Unified Framework

This report builds up a unified picture of spatiotemporal operator learning in four layers: **patching** (locality of data presentation), **models** (locality of parameters via pooling), **algorithm families** (chosen by data domain), and **backends** (Field/Domain adapter layer). At the end, we add **time** as a peer axis with its own structure that composes with the spatial framework.

---

## 1. Patching: The Locality Layer

> Patching acts as a locality operator that controls the receptive field of the operator.

The patcher is configured by four axes. Each axis is a single decision with a clear contract; any patching scheme (sliding window, jittered crops, multi-scale pyramids, graph neighborhoods, geodesic caps) is a point in this 4-D space.

| Axis | Controls | Examples |
|------|----------|----------|
| **Patch Geometry** | Shape + scale of the neighborhood (and the domain topology) | Rectangular, Spherical cap, knn-graph, radius-graph |
| **Sampler** | Where anchors are placed (overlap/redundancy is emergent) | Regular stride, jittered stride, Poisson-disk, random, explicit |
| **Window** | Boundary treatment (spectral leakage, edge artifacts) | Boxcar, Hann, Tukey, Gaussian |
| **Aggregation** | Local → global reconstruction | Overlap-add, mean, weighted sum, learned |

Patch *size* is folded into geometry (a `Rectangular` has a size, a `SphericalCap` has a radius, a `KNNGraph` has a k); there is no separate size axis. Stride is the grid-specific case of `Sampler`; treating it as such makes Poisson-disk, random, and explicit anchors first-class.

### Pseudocode

```python
# ─────────────────────────────────────────────────────────────
# Axis 1: Patch Geometry — defines a neighborhood (shape + scale)
# ─────────────────────────────────────────────────────────────
class PatchGeometry:
    def neighborhood(self, domain, anchor) -> Indices: ...
    def extent(self, domain) -> Extent: ...   # used by Sampler to place anchors

class Rectangular(PatchGeometry):   size: tuple[int, ...]
class SphericalCap(PatchGeometry):  radius_km: float
class KNNGraph(PatchGeometry):      k: int; metric: Callable
class RadiusGraph(PatchGeometry):   radius: float; metric: Callable

# ─────────────────────────────────────────────────────────────
# Axis 2: Sampler — where to place anchors
# ─────────────────────────────────────────────────────────────
class Sampler:
    def anchors(self, domain, geometry: PatchGeometry) -> Iterable[Anchor]: ...

class RegularStride(Sampler):   step: tuple[int, ...]
class JitteredStride(Sampler):  step: tuple; jitter: float
class Random(Sampler):          n_samples: int
class PoissonDisk(Sampler):     min_dist: float
class Explicit(Sampler):        anchors_: list[Anchor]

# ─────────────────────────────────────────────────────────────
# Axis 3: Window — boundary treatment
# ─────────────────────────────────────────────────────────────
class Window:
    def weights(self, geometry: PatchGeometry) -> Array: ...

class Boxcar(Window): ...
class Hann(Window): ...
class Tukey(Window):    alpha: float
class Gaussian(Window): sigma: float
class Custom(Window):   fn: Callable

# ─────────────────────────────────────────────────────────────
# Axis 4: Aggregation — local predictions → global field
# ─────────────────────────────────────────────────────────────
class Aggregation:
    def merge(self, patches: list[Patch], domain) -> GlobalField: ...

class OverlapAdd(Aggregation):    normalize_by_window: bool = True
class Mean(Aggregation): ...
class Median(Aggregation): ...
class WeightedSum(Aggregation):   weight_fn: Callable
class Learned(Aggregation):       model: NN

# ─────────────────────────────────────────────────────────────
# The Patcher: composes the four axes
# ─────────────────────────────────────────────────────────────
@dataclass
class Patcher:
    geometry:    PatchGeometry
    sampler:     Sampler
    window:      Window
    aggregation: Aggregation

    def split(self, field) -> list[Patch]:
        patches = []
        for a in self.sampler.anchors(field.domain, self.geometry):
            idx = self.geometry.neighborhood(field.domain, a)
            w   = self.window.weights(self.geometry)
            patches.append(Patch(data=field.select(idx) * w,
                                 anchor=a, indices=idx, weights=w))
        return patches

    def merge(self, patches: list[Patch], domain) -> GlobalField:
        return self.aggregation.merge(patches, domain)
```

### Structure and data flow

```
                          ┌─────────────────────────┐
                          │        Patcher          │
                          │  (composes 4 axes)      │
                          │   .split() / .merge()   │
                          └────────────┬────────────┘
                                       │ has-a
            ┌──────────────────┬───────┴───────┬──────────────────┐
            ▼                  ▼               ▼                  ▼
   ┌─────────────────┐  ┌─────────────┐  ┌───────────┐  ┌──────────────────┐
   │ PatchGeometry   │  │   Sampler   │  │  Window   │  │   Aggregation    │
   │ shape + scale   │  │ anchor      │  │ boundary  │  │ local → global   │
   └────────┬────────┘  │ placement   │  │ treatment │  │ reconstruction   │
            │           └──────┬──────┘  └─────┬─────┘  └────────┬─────────┘
            ▼                  ▼               ▼                 ▼
     Rectangular,       RegularStride,      Boxcar,        OverlapAdd,
     SphericalCap,      Random,             Hann,          Mean,
     KNNGraph,          PoissonDisk,        Tukey,         WeightedSum,
     RadiusGraph        Explicit            Gaussian       Learned

   Data flow:
   ═══════════
      GlobalField                                              GlobalField
           │                                                        ▲
           ▼  split()                                       merge() │
   ┌───────────────┐    ┌──────────┐    ┌──────────┐    ┌───────────────┐
   │ 1. Sampler    │───▶│ anchors  │───▶│ Geometry │───▶│ 4. Aggregation│
   │    places     │    └──────────┘    │ defines  │    │    blends     │
   │    anchors    │                    │ neighbor │    │    patches    │
   └───────────────┘                    │ -hoods   │    └───────────────┘
                                        └────┬─────┘            ▲
                                             ▼                  │
                                    ┌─────────────────┐         │
                                    │ 3. Window       │         │
                                    │    weights      │         │
                                    └────────┬────────┘         │
                                             ▼                  │
                                        [ patches ]             │
                                             ▼                  │
                                      ┌─────────────┐           │
                                      │  Operator   │───────────┘
                                      │  (PCA, FFT, │
                                      │   GP, NN…)  │
                                      └─────────────┘
```

The operator sits *outside* the Patcher — that's the entire point. The Patcher handles locality; the operator handles modeling. Swap either independently.

---
## 5. Time as a Distinct Axis

Time is, in a narrow technical sense, just another dimension you could fold into `Rectangular(size=(T, H, W))`. But that hides four properties of time that have real consequences for both patching and modeling, and that the framework should make first-class:

1. **Causality.** Past → future is asymmetric. Spatial neighborhoods are typically symmetric; temporal windows are usually causal (only the past).
2. **Periodicity.** Diurnal, seasonal, annual cycles are first-class temporal structure. Spatial periodicity exists (the sphere) but is the exception; temporal periodicity is the rule.
3. **Multi-scale.** Hourly, daily, monthly, annual aren't refinements of one scale — they're categorically different scales we often want simultaneously (e.g. hourly weather + seasonal climatology).
4. **Forecasting vs reconstruction.** Predicting forward in time is a fundamentally different task from filling in space, with its own loss structure and uncertainty propagation.

A fifth, subtler point: **time-aware operators embed time structure architecturally** (RNN, causal CNN, state-space, AR, neural ODE), in a way that spatial operators rarely embed spatial structure. Hiding time as a generic axis would lose this distinction.

The proposal: **time gets its own peer structure** — a `TimePatcher` with four time-aware axes — and composes with the spatial `Patcher` via a `SpatioTemporalPatcher`. This keeps the spatial framework reusable as-is, gives time its own knobs, and makes the composition explicit.

### The time axes

| Axis | Controls | Examples |
|------|----------|----------|
| **TimeGeometry** | Window shape (lookback, horizon, multi-scale, phase) | `FixedLookback`, `LookbackHorizon`, `MultiScale`, `PhaseWindow` |
| **TimeSampler** | Anchor placement in time | `RegularTimeStride`, `CausalRolling`, `EventTriggered`, `RandomTime`, `ExplicitTimes` |
| **TimeWindow** | Temporal boundary treatment | `CausalBoxcar`, `ExponentialDecay`, `TaperedTukey`, `Periodic` |
| **TimeAggregation** | Time → time reconstruction | `Sequential` (RNN-like fold), `TemporalMean`, `HierarchicalCombine`, `Forecast` |

Each axis is the temporal analog of its spatial counterpart, but with time-specific subclasses that encode the four properties above (causality in `CausalBoxcar`/`CausalRolling`, periodicity in `Periodic`/`PhaseWindow`, multi-scale in `MultiScale`/`HierarchicalCombine`, forecasting in `LookbackHorizon`/`Forecast`).

### Pseudocode

```python
# ─────────────────────────────────────────────────────────────
# Time Geometry
# ─────────────────────────────────────────────────────────────
class TimeGeometry:
    def window(self, time_axis, anchor: Time) -> TimeRange: ...

class FixedLookback(TimeGeometry):     length: Duration
class LookbackHorizon(TimeGeometry):   lookback: Duration; horizon: Duration  # forecasting
class MultiScale(TimeGeometry):        scales: list[Duration]                  # hourly+daily+annual
class PhaseWindow(TimeGeometry):       period: Duration; phase_width: Duration # diurnal/seasonal

# ─────────────────────────────────────────────────────────────
# Time Sampler
# ─────────────────────────────────────────────────────────────
class TimeSampler:
    def anchors(self, time_axis) -> Iterable[Time]: ...

class RegularTimeStride(TimeSampler):  step: Duration
class CausalRolling(TimeSampler):      step: Duration   # past-only relative to a reference
class EventTriggered(TimeSampler):     event_times: Iterable[Time]
class RandomTime(TimeSampler):         n: int           # training-time augmentation
class ExplicitTimes(TimeSampler):      times: list[Time]

# ─────────────────────────────────────────────────────────────
# Time Window (boundary treatment)
# ─────────────────────────────────────────────────────────────
class TimeWindow:
    def weights(self, geometry: TimeGeometry) -> Array: ...

class CausalBoxcar(TimeWindow):     ...   # no future info, hard past cutoff
class ExponentialDecay(TimeWindow): tau: Duration       # recency weighting
class TaperedTukey(TimeWindow):     alpha: float        # spectral leakage in time
class Periodic(TimeWindow):         period: Duration    # cyclic boundary (diurnal/annual)

# ─────────────────────────────────────────────────────────────
# Time Aggregation
# ─────────────────────────────────────────────────────────────
class TimeAggregation:
    def merge(self, patches) -> TimeSeries: ...

class Sequential(TimeAggregation): ...          # RNN-like fold: state-passing
class TemporalMean(TimeAggregation): ...
class HierarchicalCombine(TimeAggregation):    scales: list[Duration]
class Forecast(TimeAggregation): ...            # predict horizon from lookback

# ─────────────────────────────────────────────────────────────
# The TimePatcher
# ─────────────────────────────────────────────────────────────
@dataclass
class TimePatcher:
    geometry:    TimeGeometry
    sampler:     TimeSampler
    window:      TimeWindow
    aggregation: TimeAggregation
```

### Time pooling: stationarity in the time dimension

The three pooling cases from Section 2 apply independently to the time axis, with the assumption being **temporal stationarity** (regime stability) instead of spatial stationarity:

- **Complete time pooling.** Time-invariant parameters $\theta$ shared across all temporal windows. Assumes regime stability — the dynamics don't change over time. AR/ARMA with fixed coefficients, FNO with time as a dim, ConvLSTM with shared weights.
- **No time pooling.** Separate $\theta_t$ per temporal window. Each window fit independently. Regime-specific models, sliding-window AR with re-estimation.
- **Partial time pooling.** Time-varying $\theta_t$ with smooth prior, $\theta_t \mid \phi \sim p(\theta_t \mid \phi)$. State-space models, Kalman filters with GP priors on transition matrices, time-varying coefficient models.

Combined with the three spatial cases, you get a 3×3 grid. Not all cells are common, but the natural diagonals are:

| | **Space: complete** | **Space: none** | **Space: partial** |
|---|---|---|---|
| **Time: complete** | FNO + time dim, ConvLSTM (shared) | Local GP × time-invariant | Hierarchical GP × time-invariant |
| **Time: none** | CNN re-fit per window | Local GP per (patch, window) | Hierarchical GP per window |
| **Time: partial** | State-space CNN | Local state-space GP | Deep state-space / hierarchical Bayesian dynamic GP |

The "fully hierarchical in space and time" cell is where dynamic spatiotemporal Bayesian models live — and it's the case most relevant to non-stationary geophysical fields (climate drift, regime shifts).

### Time-aware operator families by domain

| Time-aware domain | Natural operator family | Examples |
|-------------------|-------------------------|----------|
| **Regular spatio-temporal grid** (4D rasters, climate output) | Spatio-temporal convolution / spectral | ConvLSTM, Conv3D, FNO+time, video transformer |
| **Trajectories** (time on points) | Sequence models, neural ODEs | RNN/LSTM over points, neural ODE on trajectories, GP trajectory models |
| **Time on a sphere** (global forecast models) | Spherical spectral + recurrence/attention | GraphCast, FourCastNet, Pangu-Weather, SFNO+time |
| **Event sequences** (irregular times) | Point processes | Hawkes, neural point processes, marked temporal point processes |

The same observation as Section 3: shared-parameter and per-window versions of these are the same family differing only in pooling.

### Composition: the SpatioTemporalPatcher

```python
@dataclass
class SpatioTemporalPatcher:
    spatial:  Patcher
    temporal: TimePatcher
    coupling: str = "product"   # "product" (Cartesian) or "coupled" (paired anchors)

    def split(self, field: Field) -> list[Patch]:
        if self.coupling == "product":
            # Every spatial anchor × every time anchor
            spatial_patches = self.spatial.split(field)
            patches = []
            for sp in spatial_patches:
                for t in self.temporal.sampler.anchors(field.time_axis):
                    tr = self.temporal.geometry.window(field.time_axis, t)
                    tw = self.temporal.window.weights(self.temporal.geometry)
                    sub = sp.data.select_time(tr) * tw
                    patches.append(SpatioTemporalPatch(data=sub, space=sp.anchor, time=t))
            return patches
        elif self.coupling == "coupled":
            # Anchors are explicit (space, time) tuples — e.g. event-triggered patches
            ...

    def merge(self, patches, field):
        # First aggregate temporally within each spatial patch,
        # then aggregate spatially across patches — or the reverse,
        # depending on the operator's prediction structure.
        ...
```

Two coupling modes matter:

- **Product coupling** (default): every spatial patch × every time window. The right default for dense gridded data where space and time are independent grids (climate model output, regular satellite revisits).
- **Coupled anchors**: explicit `(space, time)` tuples. The right choice for event-triggered patches — e.g. patching around methane plume detections (where the event has both a location and a time), Argo profile locations, or storm tracks. This is also where `EventTriggered` samplers shine.

### Why the composition is the right abstraction

Three reasons composition beats adding time as a fifth axis to `Patcher`:

1. **Independent reuse.** Spatial-only and time-only Patchers are useful in isolation (image patching has no time; pure time series has no space). The composition pattern keeps them that way.
2. **Different protocol surface.** Time needs `TimeRange`, `Duration`, `Time` types that don't fit naturally into a spatial `Anchor`. Pretending they do leads to unification at the cost of type clarity.
3. **Operator alignment.** Time-aware operators (RNN, neural ODE, state-space) have a structurally different signature than spatial operators — they have hidden state, they roll forward, they make causal predictions. Composing patchers mirrors composing operators, which is what actually happens in practice (a ConvLSTM is a Conv composed with an LSTM).

---

## 6. Putting It All Together

The full stack, top to bottom:

```
   SpatioTemporalPatcher (composition)
        ├── Patcher (spatial)              ← Section 1
        │     ├── PatchGeometry            ← dispatches on Domain (Section 4)
        │     ├── Sampler
        │     ├── Window
        │     └── Aggregation              ← dispatches on Domain
        │
        └── TimePatcher (temporal)         ← Section 5
              ├── TimeGeometry
              ├── TimeSampler
              ├── TimeWindow
              └── TimeAggregation

   ↑ produces Patches over ↑

   Field (Section 4)
        ├── RasterioField  ─┐
        ├── GeoTensorField  ├──→ RasterDomain
        ├── RioXarrayField ─┘
        ├── XarrayField      ──→ GridDomain
        ├── GeoPandasField   ──→ VectorDomain
        └── XvecField        ──→ PointDomain

   ↑ patches are consumed by ↑

   Operator (Sections 2 & 3)
        ├── Parameter Sharing:  shared θ / per-patch θᵢ / θᵢ tied by φ
        ├── Pooling:            complete / none / partial
        ├── Predict Scope:      local / global
        └── Family (by domain): spectral / graph / kernel / spectral-on-manifold
```

Three orthogonal sets of decisions. The Patcher (and TimePatcher) handles locality of data presentation; the Field/Domain layer handles backend ergonomics; the model handles locality of parameters and choice of operator family. Each set is independently configurable, and the natural pairings (Rectangular ↔ RasterDomain ↔ FNO; RadiusGraph ↔ PointDomain ↔ GP; etc.) come out as defaults rather than hard constraints.


---

## 2. Models: Parameter Sharing and Pooling

With the data presented as patches, the model still has independent choices about how its *parameters* relate to those patches. The design space here is organized by three axes:

- **Parameter Sharing** — shared $\theta$ / per-patch $\theta_i$, independent / per-patch $\theta_i$ with shared hyperparameters $\phi$
- **Pooling (Fit Scope)** — complete (one $\theta$ from all data) / none (each patch fit alone) / partial (joint hierarchical fit)
- **Predict Scope** — local (evaluate per patch, aggregate via the Patcher) / global (evaluate on the full field directly)

Parameter Sharing and Pooling are tightly coupled: shared parameters force complete pooling, fully independent parameters give no pooling, and partial pooling is the unique middle ground that requires a hierarchical prior. Predict Scope is independent of both — any of the three fitting regimes can produce local or global predictions depending on what the Patcher's `merge` step does.

| Case | Sharing | Pooling | Assumption | Canonical examples |
|------|---------|---------|------------|--------------------|
| **1** | shared $\theta$ | complete | stationarity, translation-equivariance | CNN, FNO, global PCA |
| **2** | per-patch $\theta_i$, independent | none | arbitrary non-stationarity | Local GP, patch-PCA, moving-window kriging |
| **3** | per-patch $\theta_i$ tied by $\phi$ | partial | structured non-stationarity | Hierarchical GP, multi-task GP, mixed-effects |

### Case 1 — Global fit, shared parameters (*complete pooling*)

```
   [Patch 1] [Patch 2] [Patch 3]  ...  [Patch N]
       \        |        |                 /
        ▼       ▼        ▼                ▼
       ┌──────────────────────────────────┐
       │    pooled likelihood             │
       │    ∏ᵢ p(yᵢ | xᵢ, θ)              │
       └────────────────┬─────────────────┘
                        ▼
                      ┌───┐
                      │ θ │   ◄── one shared parameter
                      └─┬─┘
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
    [Patch 1]       [Patch 2]   ...  [Patch N]
     predict         predict          predict
```

A single $\theta$ is shared across the field. All patches contribute to one joint likelihood
$$p(\mathcal{D} \mid \theta) = \prod_i p(y_i \mid x_i, \theta),$$
fit by maximizing this (or its Bayesian posterior). The implicit assumption is **stationarity** — the statistics of the field do not vary across space — and usually also **translation-equivariance** of the operator. Pooling every observation into one estimate maximizes statistical strength when stationarity holds; it smears local structure when it doesn't.

*Examples:* CNN and FNO (weight-sharing across space *is* complete pooling made architectural); global PCA / EOFs (one shared basis $\{\phi_k\}$).

### Case 2 — Local fit, per-patch parameters (*no pooling*)

```
   [Patch 1]      [Patch 2]      [Patch 3]   ...   [Patch N]
       │              │              │                  │
       ▼              ▼              ▼                  ▼
   p(θ₁|y₁)       p(θ₂|y₂)       p(θ₃|y₃)          p(θ_N|y_N)
       │              │              │                  │
       ▼              ▼              ▼                  ▼
     ┌────┐         ┌────┐         ┌────┐            ┌────┐
     │ θ₁ │         │ θ₂ │         │ θ₃ │    ...     │ θ_N│
     └─┬──┘         └─┬──┘         └─┬──┘            └──┬─┘
       ▼              ▼              ▼                  ▼
   predict 1      predict 2      predict 3          predict N
        \              \             /                  /
         └──────────► Patcher.merge() ◄────────────────┘

       (no arrows between patches — statistical independence)
```

Each patch has its own $\theta_i$ fit from only its own data,
$$\theta_i \mid y_i, x_i \sim p(\theta_i \mid y_i, x_i) \propto p(y_i \mid x_i, \theta_i)\,p(\theta_i).$$
Patches are statistically independent — there is **no pooling**. The assumption is **non-stationarity**: the field's statistics may vary arbitrarily from patch to patch, and the model commits to nothing being shared. The cost is high variance for small/sparse patches and discontinuities at patch boundaries (which aggregation can paper over but not eliminate).

*Examples:* local GP, patch-PCA / dictionary learning (KSVD), moving-window kriging.

### Case 3 — Hierarchical fit, per-patch parameters tied by $\phi$ (*partial pooling*)

```
                         ┌─────┐
                         │  φ  │   ◄── shared hyperparameters
                         └──┬──┘       with hyperprior p(φ)
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
        p(θ₁|φ)         p(θ₂|φ)    ...  p(θ_N|φ)
            │               │               │
            ▼               ▼               ▼
          ┌────┐          ┌────┐          ┌────┐
          │ θ₁ │          │ θ₂ │   ...    │ θ_N│
          └─┬──┘          └─┬──┘          └──┬─┘
            ▼               ▼                ▼
        p(y₁|x₁,θ₁)    p(y₂|x₂,θ₂)     p(y_N|x_N,θ_N)
            ▲               ▲                ▲
            │               │                │
        [Patch 1]       [Patch 2]   ...  [Patch N]

   information flows up to φ (pooling) and back down to θᵢ (shrinkage)
```

Each patch has $\theta_i$ coupled through shared hyperparameters $\phi$ with hyperprior $p(\phi)$:
$$p(\{\theta_i\}, \phi \mid \mathcal{D}) \propto p(\phi)\prod_i p(\theta_i \mid \phi)\,p(y_i \mid x_i, \theta_i).$$
This is **partial pooling**: $\theta_i$ adapts locally but is shrunk toward the population-level $\phi$ inferred jointly. The assumption is the middle ground — **structured non-stationarity**: patches differ, but they're drawn from a shared population. Patches with little data borrow strength from $\phi$; patches with lots of data are free to deviate; neighboring $\theta_i$'s are pulled toward each other, softening boundary discontinuities.

*Examples:* hierarchical GP, multi-task / coregionalized GP, mixed-effects / spatial random-effects models.

---

## 3. Algorithm Families: Choosing by Data Domain

Algorithms split cleanly along **what kind of domain the data lives on**, which is essentially the *geometry* axis from the Patcher poking through into the model layer.

| Domain | Defining property | Connectivity known? | Examples |
|--------|-------------------|---------------------|----------|
| **Regular** | Gridded, uniform spacing | Implicit (lattice) | Satellite raster, lat/lon grid output, image |
| **Structured irregular** | Fixed mesh / graph, known topology | Yes, explicit | Unstructured ocean mesh, finite-volume grid, molecular graph |
| **Unstructured** | Scattered points, no fixed topology | No (or only via metric) | Argo floats, ship tracks, in-situ stations, swath samples, lidar |
| **Spherical / manifold** | Non-Euclidean geometry | Implicit (manifold) | Global atmosphere/ocean, full-sphere remote sensing |

This is independent of the three pooling cases — you can do complete, no, or partial pooling on any of these domains. But the *natural operator family* changes with the domain, because each family makes different assumptions about what "neighborhood" means.

**Spectral neural operators (FNO and friends) → regular domains.** FNO parameterizes the operator in Fourier space, requiring a uniform grid and committing to periodicity and translation-equivariance. That is precisely the *complete-pooling, stationarity* corner of the previous table.

**Graph neural operators (GNO, MeshGraphNets) → structured irregular domains.** Locality is given by the adjacency matrix, not Euclidean balls. Weight-sharing is across *edges*, generalizing the CNN's translation-equivariance to permutation-equivariance on the graph.

**Gaussian processes → unstructured domains (and the natural Bayesian choice everywhere).** A GP only needs a kernel $k(x, x')$; it doesn't care whether points come from a grid, a mesh, or a scatter. Two clarifications: (a) GPs are not exclusively unstructured — they work on grids too, just usually more expensively than FFT-based methods; (b) a GP with a graph-Laplacian kernel is the GP analog of a graph neural operator (graph GP, Borovitskiy et al. 2021).

**Spherical / manifold operators → spherical domains.** The sphere is its own case because Euclidean methods are wrong near the poles. Spherical harmonics-based methods (SFNO, DeepSphere), equirectangular/cubed-sphere CNNs (pragmatic but distorting), and geodesic GPs (kernels via great-circle distance or Laplace–Beltrami spectra, e.g. HSGP-on-the-sphere).

### Unified mapping

| Domain | Locality encoded by | Operator family | Shared-parameter version | Per-patch / Bayesian version |
|--------|--------------------|-----------------|----------------------------|------------------------------|
| Regular | Lattice + Fourier basis | Spectral / convolutional | CNN, FNO | Local GP, patch-PCA |
| Structured irregular | Adjacency matrix | Graph-based | GNO, MeshGraphNets | Graph GP, GP with graph-Laplacian kernel |
| Unstructured | Kernel $k(x, x')$ | Kernel methods | Inducing-point / sparse GP, neural process | Local GP, hierarchical GP |
| Spherical / manifold | Spherical harmonics or geodesic kernel | Spectral-on-manifold | Spherical FNO, DeepSphere | HSGP on sphere, geodesic GP |

The rows are not exclusive (you can run a CNN on unstructured data after gridding), and the shared/per-patch columns are the *same family* differing only in pooling regime. GPs span all four rows depending on the kernel — that's what makes them the universal Bayesian counterpart.

---

## 4. Backends: The Field/Domain Adapter Layer

The previous sections are intentionally backend-agnostic. To make them executable, we add a thin protocol layer between the Patcher and the actual data structures.

### Notes on specific backends

**Points.** There is no single dominant choice in geoscience. The real options:
- **xvec** — vector data cubes (xarray + shapely). The modern answer for stations/floats/swath samples with multiple variables and times.
- **xarray with CF Discrete Sampling Geometries** — the conventions-based traditional answer (`station`/`profile`/`trajectory` dims with lat/lon as non-dim coords). No new library, no spatial index.
- **geopandas with Point geometries + scipy.spatial.cKDTree** — most pragmatic for pure point clouds; gives a real spatial index for neighbor queries.

Recommendation: xvec for in-situ multivariate data, geopandas+KDTree for pure point clouds. Both coexist under the same protocol.

**Rasters: rasterio vs georeader vs rioxarray.** These aren't competitors — they live at different levels of the stack. **rasterio** is foundational (wraps GDAL; provides windowed I/O, CRS, affine transforms, and the `Window` abstraction itself). **georeader (spaceml-org)** is a thin convenience layer on rasterio that gives ML-friendly `GeoTensor` access. **rioxarray** puts a rasterio I/O backend behind an xarray DataArray. All three speak `Window` and produce the same `RasterDomain`. Rasterio is the right choice for lazy windowed reads from disk-backed COGs (patching huge rasters that don't fit in memory); the others are ergonomic wrappers.

**Unstructured meshes.** `uxarray` for unstructured-grid model output (MPAS, FVCOM, ICON) — the "structured irregular" row, distinct from raw points.

### Field/Domain protocols

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Field(Protocol):
    @property
    def domain(self) -> "Domain": ...
    def select(self, indexer) -> "Field": ...
    def with_data(self, array) -> "Field": ...

@runtime_checkable
class Domain(Protocol):
    @property
    def crs(self): ...
    @property
    def bounds(self): ...

# Concrete Domain types encode what kind of indexing the backend supports
class RasterDomain(Domain):     # rasterio/georeader/rioxarray: affine + Window
    transform: Affine; shape: tuple[int, int]; crs: CRS

class GridDomain(Domain):       # xarray non-raster: labeled N-D coords, isel/sel
    coords: dict[str, np.ndarray]; crs: CRS | None

class VectorDomain(Domain):     # geopandas polygons: geometries + spatial index
    geometry: gpd.GeoSeries; sindex: SpatialIndex; crs: CRS

class PointDomain(Domain):      # xvec or geopandas+KDTree: points + neighbor index
    coords: np.ndarray; kdtree: cKDTree; crs: CRS | None
```

### Backend adapters

```python
# 1a. Rasters — rasterio (foundational, lazy, disk-backed)
class RasterioField:
    def __init__(self, ds): self._ds = ds
    @property
    def domain(self): return RasterDomain(self._ds.transform,
                                          (self._ds.height, self._ds.width),
                                          self._ds.crs)
    def select(self, window): return _InMemoryRaster(self._ds.read(window=window), ...)

# 1b. Rasters — georeader.GeoTensor (rasterio + ML wrapper)
class GeoTensorField:
    def __init__(self, gt): self._gt = gt
    @property
    def domain(self): return RasterDomain(self._gt.transform, self._gt.shape, self._gt.crs)
    def select(self, window): return GeoTensorField(self._gt.read_from_window(window))

# 1c. Rasters — rioxarray (rasterio behind an xarray DataArray)
class RioXarrayField:
    def __init__(self, da): self._da = da
    @property
    def domain(self): return RasterDomain(self._da.rio.transform(),
                                          (self._da.rio.height, self._da.rio.width),
                                          self._da.rio.crs)
    def select(self, window): return RioXarrayField(self._da.rio.isel_window(window))

# 2. Dense non-raster grids — xarray.DataArray
class XarrayField:
    def __init__(self, da): self._da = da
    @property
    def domain(self): return GridDomain({d: self._da[d].values for d in self._da.dims},
                                        crs=getattr(self._da.rio, "crs", None))
    def select(self, isel): return XarrayField(self._da.isel(**isel))

# 3. Polygons — geopandas.GeoDataFrame
class GeoPandasField:
    def __init__(self, gdf): self._gdf = gdf
    @property
    def domain(self): return VectorDomain(self._gdf.geometry, self._gdf.sindex, self._gdf.crs)
    def select(self, mask): return GeoPandasField(self._gdf.iloc[mask].copy())

# 4. Points — xvec
class XvecField:
    def __init__(self, ds):
        self._ds = ds
        coords = np.c_[ds.geometry.values.x, ds.geometry.values.y]
        self._kdtree = cKDTree(coords)
    @property
    def domain(self): return PointDomain(np.c_[self._ds.geometry.values.x,
                                                self._ds.geometry.values.y],
                                          self._kdtree, self._ds.xvec.geom_crs)
    def select(self, idx): return XvecField(self._ds.isel(geometry=idx))
```

### Dispatch in PatchGeometry

`PatchGeometry.neighborhood` becomes a multi-method that dispatches on the `Domain` type:

```python
class Rectangular(PatchGeometry):
    size: tuple[int, ...]

    @singledispatchmethod
    def neighborhood(self, domain, anchor):
        raise NotImplementedError(f"Rectangular doesn't support {type(domain).__name__}")

    @neighborhood.register
    def _(self, domain: RasterDomain, anchor):
        return Window(col_off=anchor[1], row_off=anchor[0],
                      width=self.size[1], height=self.size[0])

    @neighborhood.register
    def _(self, domain: GridDomain, anchor):
        return {dim: slice(anchor[dim], anchor[dim] + sz)
                for dim, sz in zip(domain.coords, self.size)}

class RadiusGraph(PatchGeometry):
    radius: float

    @neighborhood.register
    def _(self, domain: PointDomain, anchor):
        return domain.kdtree.query_ball_point(anchor, r=self.radius)

    @neighborhood.register
    def _(self, domain: VectorDomain, anchor):
        buf = shapely.Point(*anchor).buffer(self.radius)
        return list(domain.sindex.query(buf, predicate="intersects"))
```

Natural pairings drop out:

| `PatchGeometry` | Natural `Domain` | Backends |
|-----------------|------------------|----------|
| `Rectangular` | `RasterDomain` | **rasterio**, georeader, rioxarray |
| `Rectangular` | `GridDomain` | xarray |
| `SphericalCap` | `GridDomain`, `PointDomain` | xarray, xvec |
| `RadiusGraph` | `PointDomain`, `VectorDomain` | xvec, geopandas |
| `KNNGraph` | `PointDomain` | xvec, geopandas+KDTree |

A `Rectangular` patch on a `VectorDomain` raises — which is correct. Cross-cases that genuinely make sense are added as additional `@register` methods when needed.

---

