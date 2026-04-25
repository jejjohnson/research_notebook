---
title: "Scaling SSH reconstruction to the globe — patch decomposition with xrpatcher + dask"
---

# Scaling to the globe — patch decomposition

The cost numbers in [01_efficient_machinery.md](01_efficient_machinery.md) for the global ocean are uncomfortable. Even with the matrix-free / fully-RFF Route B+ machinery, a reanalysis day at SWOT-era data densities sits at $\sim 5\,\text{s}$ per day on an A100 — and that depends on a single $C \in \mathbb{R}^{m \times m}$ with $m \sim 6 \times 10^6$ fitting into memory, which it largely does not. Cholesky-route exact GP is infeasible past North Atlantic scales; even the implicit operators stop scaling once the per-CG-iter $\mathcal{O}(m^2)$ kernel scan blows past a few minutes.

The standard fix is older than altimetry itself: **localise the inversion**. The SSH covariance has a finite spatial decorrelation length $L_s \sim 100\,\text{km}$, so observations more than $\sim 3 L_s$ from a target grid cell carry essentially no information about it. Cut the globe into overlapping patches, run a local GP on each patch, and stitch the patches back together with a partition-of-unity weighting. Compute scales linearly with the number of patches, memory drops to whatever fits per-patch, and the work is embarrassingly parallel.

This is **exactly what DUACS does operationally**[^taburet2019] — it runs sub-domain optimal interpolation on overlapping $\sim 1000\,\text{km}$ tiles. MIOST relaxes the partition by working in a global wavelet basis instead, but at the cost of giving up some of the parallelism. The patch-wise approach trades a small accuracy degradation at patch edges for arbitrary horizontal scaling.

This note maps the patch-decomposition recipe onto **xrpatcher**[^xrpatcher] (your patch extraction layer for xarray), **dask** (per-patch parallelism), **sklearn `BallTree`** (radius-neighbour queries), and the **Gaspari–Cohn weighted overlap-add** already implemented in your local [`quasigeostrophic_model/_tiling.py`](file:///home/azureuser/localfiles/jej_vc_snippets/quasigeostrophic_model/_tiling.py). No new theory — just the assembly.

---

## 1. The localisation argument

Recall the posterior mean from 00:

$$
\mu_{f|y}(x^*) \;=\; \sum_{j=1}^m k_\theta(x^*, x_j)\, \alpha_j, \qquad \alpha = C^{-1}y.
$$

For any kernel with finite spatial range $L_s$, the kernel weight $k_\theta(x^*, x_j)$ is below machine epsilon when $\|x^* - x_j\| \gtrsim 5 L_s$ (Matérn-3/2: $k(5L_s)/k(0) \approx e^{-\sqrt{3}\cdot 5} \approx 1.7 \times 10^{-4}$). Two consequences:

- **The dual weight $\alpha_j$ is computed using the global $C^{-1}$**, but the *contribution* of $\alpha_j$ to a far-away $x^*$ is zero. So the long-range coupling in $C^{-1}$ does *not* propagate beyond $\sim 5 L_s$ in the *predicted field*.
- **Conversely, $\alpha_j$ itself only depends on observations within $\sim 5 L_s$ of $x_j$** (this follows from the fact that the same kernel that suppresses long-range prediction also suppresses long-range entries of $C$).

Together: solving the global $C^{-1} y$ and then evaluating at $x^*$ is, up to a small "boundary tail", equivalent to solving a *local* $C^{-1}_{\text{local}}\, y_{\text{local}}$ where "local" means "within a few $L_s$ of $x^*$". This is the **screening property** of local Gaussian inversions, formalised in the data-assimilation literature as covariance localisation[^houtekamer2005] [^bishop2017].

The patch-wise recipe makes this explicit: each patch solves a small local GP using only nearby observations, the patches overlap by enough to absorb the boundary tail, and a smooth partition-of-unity blend stitches them back together.

---

## 2. Patch geometry with xrpatcher

[`xrpatcher`](https://github.com/jejjohnson/xrpatcher) (your package, with Quentin Febvre) is the right primitive for the **prediction grid** decomposition. It is a single-class API:

```python
from xrpatcher import XRDAPatcher
import xarray as xr

# Empty grid DataArray defining the prediction lat/lon mesh.
grid = xr.DataArray(
    data=np.zeros((1801, 3601), dtype=np.float32),     # 0.1° global ocean
    dims=("lat", "lon"),
    coords={
        "lat": np.linspace(-90, 90, 1801),
        "lon": np.linspace(-180, 180, 3601),
    },
)

patcher = XRDAPatcher(
    da            = grid,
    patches       = {"lat": 256, "lon": 256},          # 25.6° × 25.6° at 0.1°
    strides       = {"lat": 192, "lon": 192},          # ⇒ 64-cell overlap on each axis
    domain_limits = {"lat": slice(-80, 80)},           # skip pole rows
    check_full_scan = False,
)
```

Key parameters from the actual [`xrpatcher/_src/base.py`](https://github.com/jejjohnson/xrpatcher/blob/main/xrpatcher/_src/base.py):

| Argument | Role |
|---|---|
| `patches` | Per-dim patch size, in cells |
| `strides` | Per-dim stride. `stride < patch_size` ⇒ overlap; the overlap is the partition-of-unity buffer |
| `domain_limits` | `da.sel(**domain_limits)` applied first — useful to mask land or skip polar rows |
| `cache=True` | Materialise patches on first access (recommended for dask-backed `da`) |

Each `patcher[i]` returns an `xr.DataArray` carrying the patch's coordinates, which you use to bound the per-patch observation query.

**Overlap sizing rule.** The overlap on each axis must be at least $\sim 5 L_s$ in cells for the screening argument above to hold. At $0.1°$ resolution and $L_s \approx 100\,\text{km} \approx 1°$ at mid-latitudes, that is $\sim 50$ cells. A 64-cell overlap (12.8°) gives a comfortable margin and pushes patch-edge artefacts below the altimeter noise floor.

---

## 3. Selecting observations per patch — radius-neighbour queries

Each patch needs the subset of observations within its halo. With $\sim 10^6$ obs per day globally and $\sim 200$ patches, naive distance computation is $200 \times 10^6 = 2 \times 10^8$ great-circle distance evaluations — fine, but a single `BallTree` does it faster and gives you the per-patch index lists in one shot.

```python
from sklearn.neighbors import BallTree

# Build once per time window (NOT per patch).
obs_lonlat_rad = np.deg2rad(obs_lonlat[:, [1, 0]])   # BallTree expects (lat, lon) in radians
tree = BallTree(obs_lonlat_rad, metric="haversine")

EARTH_R_KM = 6371.0
HALO_KM = 5 * Ls_km                                  # 5 × spatial decorrelation length

def obs_for_patch(patch_da):
    centre_lat = float(patch_da.lat.mean())
    centre_lon = float(patch_da.lon.mean())
    # Effective patch radius = half-diagonal + halo
    half_diag_km = EARTH_R_KM * np.deg2rad(
        np.hypot(patch_da.lat.size, patch_da.lon.size) * 0.05  # 0.1° / 2
    )
    radius_km = half_diag_km + HALO_KM
    ind = tree.query_radius(
        np.deg2rad([[centre_lat, centre_lon]]),
        r = radius_km / EARTH_R_KM,
    )[0]
    return ind                                        # indices into the global obs array
```

`BallTree.query_radius` is $\mathcal{O}(\log m_{\text{global}})$ per patch on the haversine metric. Per-patch obs counts at SWOT-era density and $25°\times 25°$ patches: $\sim 10^4{-}10^5$ — well inside the regime where the gaussx Route A (PCG against implicit kernel) from 01 is fastest.

Time observations follow the same pattern with a 1-D temporal halo of $\sim 5 L_t$ days.

---

## 4. The per-patch GP

This is where 00 / 01 plug in unchanged. Each patch is a small SSH reconstruction problem with $m_{\text{patch}} \sim 10^4{-}10^5$ obs and $n_{\text{patch}} \sim 256^2 = 6.5 \times 10^4$ grid cells:

```python
import gaussx as gx
import lineax as lx

def reconstruct_patch(patch_da, obs_idx, y_full, X_obs_full, key):
    X_obs   = X_obs_full[obs_idx]                     # (m_patch, 3)  lon/lat/time
    y_obs   = y_full[obs_idx]                         # (m_patch,)
    X_grid  = grid_coords(patch_da)                   # (n_patch, 3)

    C_op = gx.ImplicitKernelOperator(
        kernel_fn = k_st_curried,
        X         = X_obs,
        noise_var = sigma_obs ** 2,
        tags      = lx.positive_semidefinite_tag,
    )
    Kxs_op = gx.ImplicitCrossKernelOperator(
        kernel_fn  = k_st_curried,
        X_data     = X_grid,
        X_inducing = X_obs,
        batch_size = 4096,
    )
    solver = gx.PreconditionedCGSolver(
        preconditioner_rank = 50,
        shift               = sigma_obs ** 2,
        rtol                = 1e-6,
        max_steps           = 200,
    )

    alpha     = gx.solve(C_op, y_obs, solver=solver)
    mean_field = Kxs_op.mv(alpha).reshape(patch_da.shape)

    # Optional: posterior samples via Matheron — see 01 §3 for the loop.
    return mean_field                                  # np.ndarray with patch's shape
```

Per-patch wall-clock at $m_{\text{patch}} = 5 \times 10^4$, $n_{\text{patch}} = 6.5 \times 10^4$ on an A100 (using 01's Route A column scaled down): $\sim 200\,\text{ms}/\text{patch}$. With $\sim 200$ patches covering the global ocean and full per-patch pathwise sampling ($S=100$): $\sim 1\,\text{min}/\text{day}$ at full GPU utilisation — almost an order of magnitude faster than the global RFF+ run from 01, and *exact* GP rather than RFF-approximated.

---

## 5. Stitching — Gaspari–Cohn weighted overlap-add

Two reconstructions of the same grid cell from neighbouring patches are not identical (they used different obs subsets), so the stitching must use a smooth partition-of-unity weight to avoid visible patch boundaries. The right weight is **Gaspari–Cohn**[^gaspari1999]: a compactly-supported 5th-order polynomial that approximates a Gaussian on $[0, 1]$ with $w(0) = 1$, $w(1) = 0$, $w'(0) = w'(1) = 0$, and an exact zero outside $[0, 1]$. The advantage over a Gaussian is the compact support — no truncation, no overlap-tail bleed beyond the patch.

You have an exact implementation in [`quasigeostrophic_model/_tiling.py`](file:///home/azureuser/localfiles/jej_vc_snippets/quasigeostrophic_model/_tiling.py) — `_tile_gaspari_cohn_weights(tile_size, overlap)` builds a 2-D weight array peaked at the patch centre, decaying smoothly to zero at the patch edge. Reuse it directly:

```python
from quasigeostrophic_model._tiling import _tile_gaspari_cohn_weights

w_patch = _tile_gaspari_cohn_weights(
    tile_size = 256,
    overlap   = 64,
)                                                      # (256, 256), peaks at centre
```

xrpatcher's `XRDAPatcher.reconstruct(items, weight=...)` then does the weighted overlap-add for free:

```python
patch_results = [reconstruct_patch(patcher[i], ...) for i in range(len(patcher))]

global_field = patcher.reconstruct(
    items  = patch_results,
    weight = w_patch,
)                                                      # xr.DataArray on the full grid
```

Internally `reconstruct` accumulates `patch * w` into `rec_da` and `w` into `count_da` per patch position, then returns `rec_da / count_da` — i.e. a normalised partition of unity. With Gaspari–Cohn weights and 64-cell overlap, the resulting field is $C^2$-continuous across patch boundaries and free of visible seams.

**Variance estimates compose the same way** — accumulate `var_patch * w**2` (because variances of weighted sums of *independent* draws scale as $\sum w^2 \sigma^2$). For posterior *samples*, blend at the sample level so the per-pixel correlations from each patch's pathwise draw survive.

---

## 6. Parallelism — dask for patches

Each `reconstruct_patch` call is fully independent of the others; the only shared state is the global `BallTree` and the observation arrays (read-only). This is the textbook embarrassingly-parallel pattern. Two routes:

### 6a. dask.delayed for cluster-wide parallelism

```python
import dask
from dask.distributed import Client

client = Client(n_workers=8, threads_per_worker=1)     # one worker per GPU/CPU socket

# Scatter the read-only obs arrays once — workers reuse without re-shipping.
y_full_remote     = client.scatter(y_full,     broadcast=True)
X_obs_full_remote = client.scatter(X_obs_full, broadcast=True)
tree_remote       = client.scatter(tree,       broadcast=True)

futures = []
for i in range(len(patcher)):
    patch_da = patcher[i]                              # cheap — just slicing
    obs_idx  = obs_for_patch(patch_da)                 # in-process — small lookup
    fut = client.submit(
        reconstruct_patch,
        patch_da, obs_idx, y_full_remote, X_obs_full_remote,
        key = jr.fold_in(master_key, i),
    )
    futures.append(fut)

patch_results = client.gather(futures)
global_field  = patcher.reconstruct(patch_results, weight=w_patch)
```

Throughput scales linearly until you saturate either GPU memory bandwidth or the inter-worker scatter — both happen comfortably past 100 workers.

### 6b. jax.pmap over local devices

For single-node multi-GPU (A100×8 / H100×8), `jax.pmap` over a stack of patches is faster because it avoids the dask serialization overhead. Pad patches to a uniform shape (constant $m_{\text{patch}}$ via dummy obs) so the JIT compiles once:

```python
@jax.jit
def reconstruct_one_patch_jit(X_obs, y_obs, X_grid, key):
    ...                                                # body of reconstruct_patch above

batched = jax.pmap(reconstruct_one_patch_jit)
patch_results = batched(X_obs_stack, y_obs_stack, X_grid_stack, key_stack)
```

Padding wastes some compute on the dummy obs but the JIT cache hit and the lack of inter-process communication usually wins. Use this when the global problem fits on one node; switch to dask.delayed when it does not.

### 6c. dask-backed xarray observations

If your raw observation NetCDFs are larger than memory (a year of SWOT L3 data is $\sim 1\,\text{TB}$), open them lazily with `xr.open_mfdataset` and let dask materialise the per-day slice when the BallTree is built. xrpatcher slices stay lazy until `reconstruct_patch` calls `.values`, so memory is bounded by per-patch working set + the dask graph.

---

## 7. Cost analysis — what scales how

Let $P$ be the number of patches and $m_p, n_p$ the per-patch obs / grid sizes. Total cost decomposes as:

| Stage | Cost | Comment |
|---|---|---|
| Build global `BallTree` | $\mathcal{O}(m \log m)$ | Once per time window |
| `query_radius` for $P$ patches | $\mathcal{O}(P \log m + P\, m_p)$ | Cheap |
| Per-patch GP (Route A) | $\mathcal{O}(\text{n\_iters} \cdot m_p^2 + S\, n_p m_p)$ | The dominant compute |
| Total (sequential) | $P \cdot \mathcal{O}(\text{n\_iters} \cdot m_p^2 + S\, n_p m_p)$ | |
| Total (parallel, $W$ workers) | $\frac{P}{W} \cdot \mathcal{O}(\text{n\_iters} \cdot m_p^2 + S\, n_p m_p)$ | Linear speed-up |
| Stitching | $\mathcal{O}(P\, n_p) = \mathcal{O}(n \cdot \text{overlap-factor})$ | Free |

### Concrete numbers, global ocean, SWOT-era reanalysis day, $L_s = 100\,\text{km}$

- Patch size $25° \times 25° = 256 \times 256$ at $0.1°$ ⇒ $n_p = 6.5 \times 10^4$.
- Stride $19° \times 19° \Rightarrow$ overlap $6° = 60\,\text{cells} \approx 5 L_s$ at mid-latitudes ✓.
- Patch count $P \approx 200$ over the global ocean (after masking land — bookkeeping done by `domain_limits` + a land mask).
- Daily obs $\sim 2 \times 10^6$ globally ⇒ $m_p \sim 2 \times 10^6 / 200 \times \text{(overlap factor 1.5)} \approx 1.5 \times 10^4$ per patch.

Plug into 01's Route A per-day cost model with $S=100$:

- Per-patch flops: $100 \times 100 \times (1.5\text{e}4)^2 + 100 \times 6.5\text{e}4 \times 1.5\text{e}4 \approx 2.3\text{e}{12} + 9.7\text{e}{10} \approx 2.4 \times 10^{12}$.
- Total flops/day: $200 \times 2.4\text{e}{12} = 4.8 \times 10^{14}$ — vs $4 \times 10^{17}$ for monolithic Route A. **~800× faster.**
- A100 wall-clock: $\sim 100\,\text{s}/\text{day}$ on a single GPU sequential; **~12 s/day on 8× A100**. For 30 days: $\sim 6\,\text{min}$. For 6 months: $\sim 70\,\text{min}$.

This brings global SWOT-era reanalysis on a single 8-GPU node into the same time bracket as the *fully-RFF* numbers from 01, but with **exact per-patch GP** instead of the RFF approximation — better fidelity at finer scales, no $\mathcal{O}(1/\sqrt{r})$ kernel-approximation bias.

---

## 8. Failure modes and what to watch for

- **Patches with too few observations.** Polar oceans in winter (sea-ice mask), continental shelves with bad QC. Guard each `reconstruct_patch` with an `if len(obs_idx) < min_obs: return prior_mean_only`, where `min_obs` ~ $\mathcal{O}(L_s^2 / \Delta x^2)$ — below this the GP is not constrained anywhere and you should just return the patch's prior mean (zero for SLA).
- **Coastal patches** have observation distributions that are highly non-uniform (no obs over land); the GP variance in the land-shadow within a patch is large. Either mask land *before* the per-patch solve so it never sees those grid cells, or accept large variances on coastal land cells (they will be discarded downstream anyway).
- **Patch-boundary discontinuities at long temporal wavelengths.** If you patch in space but treat time globally, no problem. If you also patch in time, the temporal halo must be at least $5 L_t \approx 25{-}50$ days — which forces large temporal patches and largely defeats the purpose. Recommendation: patch in space, run the time window monolithically per patch.
- **Hyperparameter consistency across patches.** Every patch solves with the same $(\sigma_\eta^2, L_s, L_t, \sigma_{obs}^2)$ — fit hyperparameters once globally on a representative window. Letting each patch fit its own hyperparameters introduces patch-to-patch jumps that show up as visible seams even with Gaspari–Cohn blending.
- **Land-bridge artefacts.** Two patches separated by a thin land barrier (e.g. Panama, Bering, Bosphorus) can both pull from the same observations on either side, propagating spurious correlations *through* the land. Either use a great-circle distance with a land penalty (route around), or post-process by setting all land-cell values to NaN and verifying patch boundaries do not cross narrow channels.
- **xrpatcher's `reconstruct` is eager.** It allocates `np.zeros(global_shape)` so for the global $0.1°$ field at $1801 \times 3601 \times 4\,\text{bytes} = 26\,\text{MB}$ this is fine; for a global $0.05°$ output (4× more cells × 8 bytes for float64) you are at $\sim 800\,\text{MB}$ which is still fine but pushes the boundary. For ultra-high-resolution outputs, write each patch directly to a Zarr store at its target offset and skip `reconstruct` entirely.

---

## 9. End-to-end pseudocode

```python
import dask
from dask.distributed import Client
from sklearn.neighbors import BallTree
from xrpatcher import XRDAPatcher
from quasigeostrophic_model._tiling import _tile_gaspari_cohn_weights

# --- One-time global setup ---
grid     = build_global_prediction_grid(resolution_deg=0.1)
patcher  = XRDAPatcher(grid, patches={"lat": 256, "lon": 256},
                       strides={"lat": 192, "lon": 192},
                       domain_limits={"lat": slice(-80, 80)})
w_patch  = _tile_gaspari_cohn_weights(tile_size=256, overlap=64)
client   = Client(n_workers=8, threads_per_worker=1)

# --- Per reanalysis day ---
def reconstruct_day(t_target):
    obs_lonlat, obs_time, y_obs = load_obs_window(t_target, tau=5)        # CMEMS data
    tree = BallTree(np.deg2rad(obs_lonlat[:, [1, 0]]), metric="haversine")

    obs_lonlat_r = client.scatter(obs_lonlat, broadcast=True)
    obs_time_r   = client.scatter(obs_time,   broadcast=True)
    y_obs_r      = client.scatter(y_obs,      broadcast=True)
    tree_r       = client.scatter(tree,       broadcast=True)

    futures = []
    for i, patch_da in enumerate(patcher):
        obs_idx = obs_for_patch(patch_da, tree, halo_km=500)
        fut = client.submit(
            reconstruct_patch,
            patch_da, obs_idx, y_obs_r, obs_lonlat_r, obs_time_r,
            key=jr.fold_in(master_key, i * 10000 + day_index(t_target)),
        )
        futures.append(fut)

    patch_results = client.gather(futures)
    return patcher.reconstruct(patch_results, weight=w_patch)

# --- Loop over the reanalysis window ---
for t in pd.date_range(t_start, t_end, freq="D"):
    eta_field = reconstruct_day(t)
    eta_field.to_zarr(output_store, region={"time": slice_for_t(t)})
```

That is the global SSH pathwise pipeline at scale: `XRDAPatcher` for patch geometry, `BallTree` for obs lookup, the gaussx pathwise machinery from 01 inside each patch, `_tile_gaspari_cohn_weights` for the seam-free blend, dask for parallelism, Zarr for the output. No piece is novel — every component already lives in the user's local repos or established libraries — but the assembly is the answer to "how do I run my GP on the whole ocean without OOM-ing."

---

[^taburet2019]: Taburet, G., Sanchez-Roman, A., Ballarotta, M., Pujol, M.-I., Legeais, J.-F., Fournier, F., Faugere, Y., Dibarboure, G. (2019). "DUACS DT2018: 25 years of reprocessed sea level altimetry products." *Ocean Science* 15, 1207–1224.

[^gaspari1999]: Gaspari, G., Cohn, S. E. (1999). "Construction of correlation functions in two and three dimensions." *Q. J. R. Meteorol. Soc.* 125, 723–757.

[^houtekamer2005]: Houtekamer, P. L., Mitchell, H. L. (2005). "Ensemble Kalman filtering." *Q. J. R. Meteorol. Soc.* 131, 3269–3289. (Covariance localisation as the screening property in EnKF.)

[^bishop2017]: Bishop, C. H., Whitaker, J. S., Lei, L. (2017). "Gain form of the ensemble transform Kalman filter and its relevance to satellite data assimilation with model space ensemble covariance localization." *Mon. Weather Rev.* 145, 4575–4592.

[^xrpatcher]: Johnson, J. E., Febvre, Q. (2024). "xrpatcher: an xarray patch extractor for ML on geophysical fields." [github.com/jejjohnson/xrpatcher](https://github.com/jejjohnson/xrpatcher). Used as the data layer for OceanBench and the SSH-mapping ocean data challenges.
