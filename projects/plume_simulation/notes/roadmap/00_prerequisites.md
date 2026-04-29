# Prerequisites — Cross-tier infrastructure

Before any dispersion model, you need three things that **every tier shares**: meteorological forcing (winds, turbulence, PBL), static surface fields (orography, roughness, land-use), and the observation-side glue (L1/L2 ingest, averaging kernels). These are not models — they are the data interfaces all tiers read from. Getting them right early pays off because all four tiers will read from the same APIs; getting them wrong forces parallel ports of every tier later.

The page is grouped into:

1. **Forcing — meteorology** (dynamic 3D fields)
2. **Static surface fields** (orography, land-use, roughness)
3. **Geometry** (coordinate frames and time)
4. **Inversion priors** (background emission inventory)
5. **Observation side** (L1/L2 ingest + averaging kernel)

The fixed [forward interface](#fixed-forward-interface) is the contract that ties them together.

---

## 1 · Forcing — meteorology

### Reanalysis-agnostic met reader

**Met forcing has a many-to-one interface.** WRF is the highest-fidelity option (you control resolution, microphysics, nesting) and is the industry baseline, but ERA5 / MERRA-2 / HRRR / GEOS-FP are all valid forcing sources. The `MetField` PyTree is the abstraction; the WRF reader is one concrete loader.

- **Sources (priority order):**
  - WRF-ARW NetCDF (`wrfout_d0X_*.nc`) — primary.
  - ERA5 (Copernicus CDS) — fallback for global / climatological work.
  - HRRR / GEOS-FP — operational near-real-time.
- **Output:** a `MetField` PyTree with consistent units, time stamping, and grid metadata (schema in [Fixed forward interface](#fixed-forward-interface)).
- **Interpolation:** bilinear in horizontal, log-pressure or geometric height vertical.
- **Caching:** pre-resampled grids cached to disk (zarr) so notebooks don't re-interpolate WRF on every run. Cache key is `(source, native_grid_hash, target_grid_hash, time_window)`.

### Planetary boundary layer (PBL) height

PBL height is **not optional** — Tier I plume rise, Tier II trajectory reflection, and Tier IV column partitioning all depend on it. WRF emits it as a diagnostic (`PBLH`); ERA5 has `blh`. Carry it as a first-class field on `MetField`, not buried inside a derived helper.

### Pasquill–Gifford stability classifier

Maps surface observations (wind speed, cloud cover, time-of-day, solar elevation) to PG stability class A–F. Used by Gaussian-tier σ_y, σ_z parameterizations.

- Reference: Turner (1970), Briggs (1973) updates.
- Output: per-grid-cell stability class as int (0–5) or one-hot.
- Status: partially implemented in [`gauss_plume/dispersion.py`](src/plume_simulation/gauss_plume/dispersion.py).
- **Sunset note:** PG is Tier-I scaffolding. Once Tier II/III are operational, MO similarity supersedes it. Don't add features to PG beyond what Tier I needs.

### Monin-Obukhov similarity

Surface-layer wind and turbulence profiles. Provides `σ_y(x)` and `σ_z(x)` as functions of downwind distance, friction velocity `u*`, Obukhov length `L`, and surface roughness `z₀`.

- Reference: Stull (1988), ch. 9.
- Used by: Tier I Gaussian σ functions; Tier II particle-trajectory turbulence; Tier III sub-grid eddy diffusivity `K`.

---

## 2 · Static surface fields

These are time-invariant geophysical fields. Distinct enough from dynamic met to deserve their own loader, but they live alongside `MetField` (typically broadcast against the time axis).

### Surface roughness `z₀` and land-use

- **Source:** MODIS IGBP land-use (500 m global) or WRF static (`LU_INDEX` + table lookup).
- **Used by:** MO similarity (`z₀` directly), Tier II reflection (different reflection coefficients over water vs. land), Tier IV viewing-geometry corrections.
- **Status:** ☐ not started.

### Orography / terrain

- **Source:** SRTM 90 m or GMTED2010; resampled to analysis grid.
- **Used by:** Tier I AERMOD-style plume rise corrections; Tier II trajectory reflection off topography; Tier III boundary conditions and pressure-coordinate transforms.
- **Status:** ☐ not started — currently only mentioned in the Tier I AERMOD aside.

---

## 3 · Geometry

### Coordinate transforms

- `lat/lon ↔ local Cartesian` (UTM or local tangent plane). Use `pyproj` for the heavy lifting; wrap in a JAX-compatible `LocalFrame` PyTree so frame metadata travels with the data.
- `pressure ↔ geometric height` via hydrostatic balance + WRF temperature profile.
- `time` normalized to UTC throughout; never mix in local time downstream.

**Critical constraint:** `pyproj` is **not JAX-traceable**. The convention is *compute once, carry as static metadata*: build the `LocalFrame` outside `jax.jit`, pass it as a static argument or as part of an `equinox.Module` with non-array fields. Document this on every transform helper.

### Time and calendar

UTC throughout — but enforce it. A small `Timestamp` PyTree (`epoch_seconds: int64, tz: Literal["UTC"]`) makes the invariant load-bearing instead of aspirational. Document leap-year / leap-second handling at the boundary (most projects botch this once and never again).

---

## 4 · Inversion priors

### Background emission inventory `q_a`

Every Bayesian inversion (Tiers I–IV, Step 2) uses a prior `q_a` over the source field. Naming a single source matters because results are sensitive to it.

- **Sources:** EDGAR v8 (global anthropogenic, ~0.1°), GFEI (oil & gas, ~0.1°), EPA GHGI (US gridded), Scarpelli et al. (sectoral CH₄).
- **Output:** spatial source field on the analysis grid + sectoral metadata + uncertainty estimate (typically lognormal `σ_log Q ≈ 0.5`).
- **Used by:** all `B` (background covariance) constructions in Tier II/III inversion; the prior on `Q` for Tier I MAP/MCMC.
- **Status:** ☐ not started.

---

## 5 · Observation side

### L1 / L2 ingest

Symmetric to the met reader. Parses raw satellite products into the shared `Observations` PyTree.

- **Inputs:** TROPOMI (NetCDF), GHGSat (HDF5), EMIT (NetCDF), Tanager (TIFF + sidecar JSON), Sentinel-2 / Landsat (TIFF). One sub-loader per instrument, dispatched by file extension + product header.
- **Output:** `Observations` PyTree (radiance or column XCH₄ + lat/lon footprint + time + per-pixel uncertainty + quality mask + AK).
- **Status:** ☐ not started — the satellite catalog at [`satellites.ipynb`](../satellites.ipynb) describes the *targets*; the ingest layer is the missing implementation.

### Averaging-kernel operator

Applies the satellite averaging kernel to a model column:

```
ŷ = AK · (h^T x + (1 − h^T) x_a)
```

where `x` is the model state (CH₄ mixing-ratio profile), `x_a` is the prior used in the L2 retrieval, `h` is the column-averaging weighting, and `AK` is the satellite-product averaging kernel matrix. Needed by Tiers II–IV when comparing to L2 XCH₄ products instead of L1 radiances.

- Status: scaffold in [`assimilation/obs_operator.py`](src/plume_simulation/assimilation/obs_operator.py).
- **Provider design:** one `Instrument` registry that returns `(AK, x_a, h)` keyed on instrument name. Single hook avoids per-tier branching.

---

## Fixed forward interface

All four tiers implement the same shape:

```python
def forward(params: Params, met: MetField) -> Observations:
    """Map source/state parameters + met forcing → simulated observations.

    Each tier provides its own concrete `Params` PyTree, but the call
    signature, return type, and JAX traceability are identical, so any
    inference loop (NumPyro, vardaX, filterax) takes any tier as a drop-in.
    """
```

### `MetField` schema

The single most-shared object in `plumax`. Concrete fields, units, and conventions:

| Field | Shape | dtype | Units | Source | Notes |
|-------|-------|-------|-------|--------|-------|
| `u`, `v`, `w` | `(T, Z, Y, X)` | f32 | m/s | dynamic | wind components, cell-centred |
| `T` | `(T, Z, Y, X)` | f32 | K | dynamic | temperature |
| `p` | `(T, Z, Y, X)` | f32 | Pa | dynamic | pressure |
| `q` | `(T, Z, Y, X)` | f32 | kg/kg | dynamic | water vapour mixing ratio |
| `tke` | `(T, Z, Y, X)` | f32 | m²/s² | dynamic | turbulent KE (optional; some loaders provide) |
| `pblh` | `(T, Y, X)` | f32 | m | dynamic | PBL height |
| `z0` | `(Y, X)` | f32 | m | static | surface roughness |
| `lu` | `(Y, X)` | i8 | — | static | land-use class (IGBP) |
| `hgt` | `(Y, X)` | f32 | m | static | terrain elevation |
| `frame` | — | static metadata | — | static | `LocalFrame` (pyproj-built, non-traced) |
| `time` | `(T,)` | i64 | UTC seconds | static metadata | `Timestamp` PyTree |
| `ensemble_dim` | scalar | int | — | static metadata | `0` for deterministic; `>0` carries an outer ensemble axis on dynamic fields |

**Conventions:**

- Coordinate axes ordered `(time, vertical, y, x)` always — never reordered downstream. `coordax` is the natural representation; the loader returns a `coordax.Dataset` with these names.
- Units enforced at load time; downstream code may assume them.
- Time alignment: dynamic fields are time-stamped at their native cadence (typically hourly). The forward interface accepts a *temporal interpolation policy* (snapshot / piecewise-linear / nearest) — see open questions.

### `Params` and `Observations`

- `Params`: tier-specific PyTree (e.g. `(Q, x0, H)` for Tier I, `S(x,t)` field for Tier III).
- `Observations`: per-instrument PyTree from the L1/L2 ingest layer — radiances or column XCH₄ + footprint + mask + uncertainty + AK. Same shape as the L1/L2 product the satellite returned.

This contract is what makes Step 6 ("upgrade any component") tractable: replace `forward` with an emulator, the inference loop doesn't notice.

---

## Module layout

| Concern | Module | Status | Blocks |
|---------|--------|--------|--------|
| Met loader (WRF) | `plume_simulation.met.wrf` | ☐ | Tier II, III |
| Met loader (ERA5) | `plume_simulation.met.era5` | ☐ | global Tier II/III |
| PBL diagnostics | `plume_simulation.met.pbl` | ☐ | Tier II reflection, Tier IV partitioning |
| Static fields (z₀, LU, terrain) | `plume_simulation.met.static` | ☐ | MO similarity, Tier III BCs |
| PG stability | `plume_simulation.gauss_plume.dispersion` | 🚧 partial | Tier I (only) |
| MO similarity | `plume_simulation.met.surface_layer` | ☐ | Tier II turbulence, Tier III diffusivity |
| Coord transforms | `plume_simulation.met.frames` | ☐ | all tiers |
| Time / Timestamp | `plume_simulation.met.time` | ☐ | all tiers |
| Emission inventory loader | `plume_simulation.priors.inventory` | ☐ | Tier I–IV inversion priors |
| L1/L2 ingest | `plume_simulation.obs.ingest` | ☐ | Tier IV (and any real-data work) |
| AK operator | `plume_simulation.assimilation.obs_operator` | 🚧 scaffold | Tier IV column-space comparison |

A `plume_simulation.met` subpackage doesn't exist yet — proposed home for the prerequisites that aren't tied to any particular tier. Same for `plume_simulation.priors` and `plume_simulation.obs`.

---

## Validation strategy

- **Met reader:** round-trip — read a WRF file, re-grid to the analysis grid, integrate column mass, compare to direct WRF column integration. Should agree to floating-point precision. **CI fixture:** pin a small synthetic `wrfout` (~1 MB, 5×5×10×3) under `tests/fixtures/met/` so the test runs without external downloads.
- **Reanalysis parity:** load the same time window from WRF and ERA5, regrid both to a coarse common grid, compare column-mean wind speed. Should agree to within climatological variability — confirms the loaders share conventions.
- **PG classifier:** reproduce the textbook table from Turner (1970) for canonical (wind, cloud, hour) inputs.
- **MO similarity:** cross-check against `metpy.calc.surface_layer_*` for a handful of sounding inputs.
- **Coordinate transforms:** round-trip lat/lon → UTM → lat/lon, max error < 1 mm.
- **Inventory loader:** integrate EDGAR over a known basin (Permian) and compare to the published basin total in the EDGAR documentation.
- **AK operator:** apply identity AK and confirm `ŷ == column-average(x)`; apply published TROPOMI AK to a known profile and compare to the official L2 product.

---

## Open questions

- **Met grid resolution.** Do we keep WRF native or always re-grid to a fixed analysis grid? Trade-off: native preserves physics fidelity, fixed simplifies cross-tier comparison. **Leaning:** fixed analysis grid for inversion; native for forward-only diagnostics.
- **Temporal interpolation policy.** Met is hourly; satellite overpass is instantaneous; satellite footprints span minutes. Commit to one of: (a) snapshot at nearest hour, (b) piecewise-linear between hours, (c) advect tracers with sub-hourly interpolated wind. Tier I tolerates (a); Tier II–III need at least (b); high-fidelity work needs (c). Default: (b), with (c) as an opt-in.
- **Ensemble met / UQ propagation.** Honest UQ requires that met itself carries an ensemble axis (WRF ensemble, ERA5 EDA). Either commit (carry `ensemble_dim` on `MetField`, `vmap` the forward over it) or flag as out-of-scope for v1. **Leaning:** scaffold the axis now (cheap), populate later.
- **`pyproj` traceability.** `pyproj` is not JAX-traceable. The convention is "build frame outside jit, carry as static metadata" — but this needs a documented pattern with one canonical example. Open: do we wrap `pyproj` calls in a `jax.pure_callback` for the rare case where the transform must run inside `jit`?
- **Multi-instrument AK.** Single `Instrument` registry returning `(AK, x_a, h)` keyed by instrument name. Decision affects how Tier IV multi-pass fusion is structured.
- **Off-grid sources.** Should the source location be snapped to the analysis grid, or do we carry it as continuous lat/lon with bilinear injection? Affects gradient sharpness in 4D-Var.
- **`coordax` adoption.** `MetField` is a near-perfect fit for a `coordax.Dataset`. Commit to it, or keep raw PyTrees for Step-1 simplicity? **Leaning:** `coordax` everywhere — the dimension naming pays for itself by Tier II.
- **Inventory provenance.** EDGAR / GFEI / EPA disagree by ~factor 2 in well-studied basins. Which is the default `q_a`, and how do we expose the choice as a configurable rather than a hard-coded prior?
