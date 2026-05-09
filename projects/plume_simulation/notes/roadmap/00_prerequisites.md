---
title: "Prerequisites — Cross-tier infrastructure"
short_title: "Prerequisites"
subject: "plumax — shared data interfaces"
authors:
  - name: J. Emmanuel Johnson
    affiliations: [UNEP, IMEO, MARS]
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: [methane, plume, dispersion, meteorology, MetField, WRF, ERA5, PBL, MO similarity, averaging kernel, emission inventory, EDGAR, pyproj]
---

# Prerequisites — Cross-tier infrastructure

Before any dispersion model, you need three things that **every tier shares**: meteorological forcing (winds, turbulence, PBL), static surface fields (orography, roughness, land-use), and the observation-side glue (L1/L2 ingest, averaging kernels). These are not models — they are the data interfaces all tiers read from. Getting them right early pays off because all four tiers will read from the same APIs; getting them wrong forces parallel ports of every tier later.

The page is grouped into:

1. **Forcing — meteorology** (dynamic 3D fields)
2. **Static surface fields** (orography, land-use, roughness)
3. **Geometry** (coordinate frames and time)
4. **Inversion priors** (background emission inventory)
5. **Observation side** (L1/L2 ingest + averaging kernel)

The fixed [forward interface](#prereqs-forward-interface) is the contract that ties them together.

---

(prereqs-met)=
## 1 · Forcing — meteorology

### Reanalysis-agnostic met reader

**Met forcing has a many-to-one interface.** WRF is the highest-fidelity option (you control resolution, microphysics, nesting) and is the industry baseline, but ERA5 / MERRA-2 / HRRR / GEOS-FP are all valid forcing sources. The `MetField` PyTree is the abstraction; the WRF reader is one concrete loader.

- **Sources (priority order):**
  - WRF-ARW NetCDF (`wrfout_d0X_*.nc`) — primary.
  - ERA5 (Copernicus CDS) — fallback for global / climatological work.
  - HRRR / GEOS-FP — operational near-real-time.
- **Output:** a `MetField` PyTree with consistent units, time stamping, and grid metadata (schema in [Fixed forward interface](#prereqs-forward-interface)).
- **Interpolation:** bilinear in horizontal, log-pressure or geometric height vertical.
- **Caching:** pre-resampled grids cached to disk (zarr) so notebooks don't re-interpolate WRF on every run. Cache key is `(source, native_grid_hash, target_grid_hash, time_window)`.

### Planetary boundary layer (PBL) height

PBL height is **not optional** — Tier I plume rise, Tier II trajectory reflection, and Tier IV column partitioning all depend on it. WRF emits it as a diagnostic (`PBLH`); ERA5 has `blh`. Carry it as a first-class field on `MetField`, not buried inside a derived helper.

(prereqs-pasquill-gifford)=
### Pasquill–Gifford stability classifier

Maps surface observations (wind speed, cloud cover, time-of-day, solar elevation) to PG stability class A–F. Used by Gaussian-tier $\sigma_y$, $\sigma_z$ parameterizations.

- Reference: {cite:p}`turner1970`, with {cite:p}`briggs1973` updates.
- Output: per-grid-cell stability class as int (0–5) or one-hot.
- Status: partially implemented in [`gauss_plume/dispersion.py`](../../src/plume_simulation/gauss_plume/dispersion.py).

:::{warning} Sunset note
PG is Tier-I scaffolding. Once Tier II/III are operational, MO similarity supersedes it.

**Why:** PG is a categorical proxy for surface-layer turbulence; once we resolve the surface layer with MO theory, the categorical buckets add nothing.
**How to apply:** Don't add features to PG beyond what Tier I needs.
:::

(prereqs-mo-similarity)=
### Monin–Obukhov similarity

Surface-layer wind and turbulence profiles. Provides $\sigma_y(x)$ and $\sigma_z(x)$ as functions of downwind distance, friction velocity $u_*$, Obukhov length $L$, and surface roughness $z_0$.

- Reference: {cite:p}`monin1954`, {cite:p}`stull1988` (ch. 9).
- Used by: Tier I Gaussian $\sigma$ functions; Tier II particle-trajectory turbulence; Tier III sub-grid eddy diffusivity $K$.

---

(prereqs-static)=
## 2 · Static surface fields

These are time-invariant geophysical fields. Distinct enough from dynamic met to deserve their own loader, but they live alongside `MetField` (typically broadcast against the time axis).

### Surface roughness $z_0$ and land-use

- **Source:** MODIS IGBP land-use (500 m global) or WRF static (`LU_INDEX` + table lookup).
- **Used by:** MO similarity ($z_0$ directly), Tier II reflection (different reflection coefficients over water vs. land), Tier IV viewing-geometry corrections.
- **Status:** ☐ not started.

### Orography / terrain

- **Source:** SRTM 90 m or GMTED2010; resampled to analysis grid.
- **Used by:** Tier I AERMOD-style ({cite:p}`cimorelli2005aermod`) plume rise corrections; Tier II trajectory reflection off topography; Tier III boundary conditions and pressure-coordinate transforms.
- **Status:** ☐ not started — currently only mentioned in the Tier I AERMOD aside.

---

(prereqs-geometry)=
## 3 · Geometry

(prereqs-coordinate-transforms)=
### Coordinate transforms

- `lat/lon ↔ local Cartesian` (UTM or local tangent plane). Use `pyproj` for the heavy lifting; wrap in a JAX-compatible `LocalFrame` PyTree so frame metadata travels with the data.
- `pressure ↔ geometric height` via hydrostatic balance + WRF temperature profile.
- `time` normalized to UTC throughout; never mix in local time downstream.

:::{important} Critical constraint — `pyproj` is not JAX-traceable
The convention is *compute once, carry as static metadata*: build the `LocalFrame` outside `jax.jit`, pass it as a static argument or as part of an `equinox.Module` with non-array fields.

**Why:** `pyproj` calls into PROJ via Python; it can't be traced by JAX, so any attempt to compute the frame inside `jit` will fail or silently fall back to host execution.
**How to apply:** document this constraint on every transform helper; if the transform must run inside `jit`, wrap in `jax.pure_callback` (see [open questions](#prereqs-open-questions)).
:::

### Time and calendar

UTC throughout — but enforce it. A small `Timestamp` PyTree (`epoch_seconds: int64, tz: Literal["UTC"]`) makes the invariant load-bearing instead of aspirational. Document leap-year / leap-second handling at the boundary (most projects botch this once and never again).

---

(prereqs-priors)=
## 4 · Inversion priors

(prereqs-emission-inventory)=
### Background emission inventory $q_a$

Every Bayesian inversion (Tiers I–IV, Step 2) uses a prior $q_a$ over the source field. Naming a single source matters because results are sensitive to it.

- **Sources:** EDGAR v8 ({cite:p}`crippa2023edgar`, global anthropogenic, ~0.1°), GFEI ({cite:p}`scarpelli2022gfei`, oil & gas, ~0.1°), gridded EPA GHGI ({cite:p}`maasakkers2023ghgi,epa_ghgi`, US gridded), Scarpelli et al. ({cite:p}`scarpelli2020sectoral`, sectoral CH₄).
- **Output:** spatial source field on the analysis grid + sectoral metadata + uncertainty estimate (typically lognormal $\sigma_{\log Q} \approx 0.5$).
- **Used by:** all $\mathbf{B}$ (background covariance) constructions in Tier II/III inversion; the prior on $Q$ for Tier I MAP/MCMC.
- **Status:** ☐ not started.

---

(prereqs-observation)=
## 5 · Observation side

(prereqs-l1-l2-ingest)=
### L1 / L2 ingest

Symmetric to the met reader. Parses raw satellite products into the shared `Observations` PyTree.

- **Inputs:** TROPOMI ({cite:p}`s5p_tropomi`, NetCDF), GHGSat ({cite:p}`ghgsat,jervis2021ghgsat`, HDF5), EMIT ({cite:p}`emit`, NetCDF), Tanager ({cite:p}`carbon_mapper`, TIFF + sidecar JSON), Sentinel-2 ({cite:p}`sentinel2`) / Landsat ({cite:p}`landsat89`, TIFF). One sub-loader per instrument, dispatched by file extension + product header.
- **Output:** `Observations` PyTree (radiance or column XCH₄ + lat/lon footprint + time + per-pixel uncertainty + quality mask + AK).
- **Status:** ☐ not started — the satellite catalog at [`satellites.md`](../satellites.md) describes the *targets*; the ingest layer is the missing implementation.

(prereqs-ak-operator)=
### Averaging-kernel operator

Applies the satellite averaging kernel to a model column:

```{math}
:label: eq-ak-operator
\hat{y} \;=\; \mathbf{A}\!\left(\mathbf{h}^{\top} \mathbf{x} \,+\, (1 - \mathbf{h}^{\top})\, \mathbf{x}_a\right)
```

where $\mathbf{x}$ is the model state (CH₄ mixing-ratio profile), $\mathbf{x}_a$ is the prior used in the L2 retrieval, $\mathbf{h}$ is the column-averaging weighting, and $\mathbf{A}$ is the satellite-product averaging kernel matrix. The construction follows the optimal-estimation framework of {cite:p}`rodgers2000`. Needed by Tiers II–IV when comparing to L2 XCH₄ products instead of L1 radiances.

- Status: scaffold in [`assimilation/obs_operator.py`](../../src/plume_simulation/assimilation/obs_operator.py).
- **Provider design:** one `Instrument` registry that returns $(\mathbf{A}, \mathbf{x}_a, \mathbf{h})$ keyed on instrument name. Single hook avoids per-tier branching.

---

(prereqs-forward-interface)=
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

(prereqs-metfield-schema)=
### `MetField` schema

The single most-shared object in `plumax`. Concrete fields, units, and conventions:

```{list-table} `MetField` PyTree fields, shapes, units, and provenance.
:label: tbl-metfield-schema
:header-rows: 1

* - Field
  - Shape
  - dtype
  - Units
  - Source
  - Notes
* - `u`, `v`, `w`
  - `(T, Z, Y, X)`
  - f32
  - m/s
  - dynamic
  - wind components, cell-centred
* - `T`
  - `(T, Z, Y, X)`
  - f32
  - K
  - dynamic
  - temperature
* - `p`
  - `(T, Z, Y, X)`
  - f32
  - Pa
  - dynamic
  - pressure
* - `q`
  - `(T, Z, Y, X)`
  - f32
  - kg/kg
  - dynamic
  - water vapour mixing ratio
* - `tke`
  - `(T, Z, Y, X)`
  - f32
  - m²/s²
  - dynamic
  - turbulent KE (optional; some loaders provide)
* - `pblh`
  - `(T, Y, X)`
  - f32
  - m
  - dynamic
  - PBL height
* - `z0`
  - `(Y, X)`
  - f32
  - m
  - static
  - surface roughness
* - `lu`
  - `(Y, X)`
  - i8
  - —
  - static
  - land-use class (IGBP)
* - `hgt`
  - `(Y, X)`
  - f32
  - m
  - static
  - terrain elevation
* - `frame`
  - —
  - static metadata
  - —
  - static
  - `LocalFrame` (pyproj-built, non-traced)
* - `time`
  - `(T,)`
  - i64
  - UTC seconds
  - static metadata
  - `Timestamp` PyTree
* - `ensemble_dim`
  - scalar
  - int
  - —
  - static metadata
  - `0` for deterministic; `>0` carries an outer ensemble axis on dynamic fields
```

**Conventions:**

- Coordinate axes ordered `(time, vertical, y, x)` always — never reordered downstream. `coordax` is the natural representation; the loader returns a `coordax.Dataset` with these names.
- Units enforced at load time; downstream code may assume them.
- Time alignment: dynamic fields are time-stamped at their native cadence (typically hourly). The forward interface accepts a *temporal interpolation policy* (snapshot / piecewise-linear / nearest) — see open questions.

### `Params` and `Observations`

- `Params`: tier-specific PyTree (e.g. $(Q, x_0, H)$ for Tier I, $S(x,t)$ field for Tier III).
- `Observations`: per-instrument PyTree from the L1/L2 ingest layer — radiances or column XCH₄ + footprint + mask + uncertainty + AK. Same shape as the L1/L2 product the satellite returned.

This contract is what makes Step 6 ("upgrade any component") tractable: replace `forward` with an emulator, the inference loop doesn't notice.

---

(prereqs-modules)=
## Module layout

```{list-table} Module-level breakdown — concern, target module, status, downstream blockers.
:label: tbl-prereqs-modules
:header-rows: 1

* - Concern
  - Module
  - Status
  - Blocks
* - Met loader (WRF)
  - `plume_simulation.met.wrf`
  - ☐
  - Tier II, III
* - Met loader (ERA5)
  - `plume_simulation.met.era5`
  - ☐
  - global Tier II/III
* - PBL diagnostics
  - `plume_simulation.met.pbl`
  - ☐
  - Tier II reflection, Tier IV partitioning
* - Static fields ($z_0$, LU, terrain)
  - `plume_simulation.met.static`
  - ☐
  - MO similarity, Tier III BCs
* - PG stability
  - `plume_simulation.gauss_plume.dispersion`
  - 🚧 partial
  - Tier I (only)
* - MO similarity
  - `plume_simulation.met.surface_layer`
  - ☐
  - Tier II turbulence, Tier III diffusivity
* - Coord transforms
  - `plume_simulation.met.frames`
  - ☐
  - all tiers
* - Time / Timestamp
  - `plume_simulation.met.time`
  - ☐
  - all tiers
* - Emission inventory loader
  - `plume_simulation.priors.inventory`
  - ☐
  - Tier I–IV inversion priors
* - L1/L2 ingest
  - `plume_simulation.obs.ingest`
  - ☐
  - Tier IV (and any real-data work)
* - AK operator
  - `plume_simulation.assimilation.obs_operator`
  - 🚧 scaffold
  - Tier IV column-space comparison
```

A `plume_simulation.met` subpackage doesn't exist yet — proposed home for the prerequisites that aren't tied to any particular tier. Same for `plume_simulation.priors` and `plume_simulation.obs`.

---

(prereqs-validation)=
## Validation strategy

- **Met reader:** round-trip — read a WRF file, re-grid to the analysis grid, integrate column mass, compare to direct WRF column integration. Should agree to floating-point precision. **CI fixture:** pin a small synthetic `wrfout` (~1 MB, 5×5×10×3) under `tests/fixtures/met/` so the test runs without external downloads.
- **Reanalysis parity:** load the same time window from WRF and ERA5, regrid both to a coarse common grid, compare column-mean wind speed. Should agree to within climatological variability — confirms the loaders share conventions.
- **PG classifier:** reproduce the textbook table from {cite:t}`turner1970` for canonical (wind, cloud, hour) inputs.
- **MO similarity:** cross-check against `metpy.calc.surface_layer_*` for a handful of sounding inputs.
- **Coordinate transforms:** round-trip lat/lon → UTM → lat/lon, max error < 1 mm.
- **Inventory loader:** integrate EDGAR ({cite:p}`crippa2023edgar`) over a known basin (Permian) and compare to the published basin total in the EDGAR documentation.
- **AK operator:** apply identity AK and confirm $\hat{y} \approx \overline{\mathbf{x}}$ (column average); apply published TROPOMI AK ({cite:p}`s5p_tropomi`) to a known profile and compare to the official L2 product.

---

(prereqs-open-questions)=
## Open questions

:::{attention} Met grid resolution
Do we keep WRF native or always re-grid to a fixed analysis grid? Trade-off: native preserves physics fidelity, fixed simplifies cross-tier comparison. **Leaning:** fixed analysis grid for inversion; native for forward-only diagnostics.
:::

:::{attention} Temporal interpolation policy
Met is hourly; satellite overpass is instantaneous; satellite footprints span minutes. Commit to one of: (a) snapshot at nearest hour, (b) piecewise-linear between hours, (c) advect tracers with sub-hourly interpolated wind. Tier I tolerates (a); Tier II–III need at least (b); high-fidelity work needs (c). **Default:** (b), with (c) as an opt-in.
:::

:::{attention} Ensemble met / UQ propagation
Honest UQ requires that met itself carries an ensemble axis (WRF ensemble, ERA5 EDA). Either commit (carry `ensemble_dim` on `MetField`, `vmap` the forward over it) or flag as out-of-scope for v1. **Leaning:** scaffold the axis now (cheap), populate later.
:::

:::{attention} `pyproj` traceability
`pyproj` is not JAX-traceable. The convention is "build frame outside jit, carry as static metadata" — but this needs a documented pattern with one canonical example. Open: do we wrap `pyproj` calls in a `jax.pure_callback` for the rare case where the transform must run inside `jit`?
:::

:::{attention} Multi-instrument AK
Single `Instrument` registry returning $(\mathbf{A}, \mathbf{x}_a, \mathbf{h})$ keyed by instrument name. Decision affects how Tier IV multi-pass fusion is structured.
:::

:::{attention} Off-grid sources
Should the source location be snapped to the analysis grid, or do we carry it as continuous lat/lon with bilinear injection? Affects gradient sharpness in 4D-Var.
:::

:::{attention} `coordax` adoption
`MetField` is a near-perfect fit for a `coordax.Dataset`. Commit to it, or keep raw PyTrees for Step-1 simplicity? **Leaning:** `coordax` everywhere — the dimension naming pays for itself by Tier II.
:::

:::{attention} Inventory provenance
EDGAR / GFEI / EPA disagree by ~factor 2 in well-studied basins. Which is the default $q_a$, and how do we expose the choice as a configurable rather than a hard-coded prior?
:::
