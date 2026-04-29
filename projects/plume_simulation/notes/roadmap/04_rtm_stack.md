# Radiative transfer (RTM) stack — parallel track

The RTM is the **observation operator** `H_obs: c(x,t) → y_radiance`. It connects Tiers II–IV to actual satellite measurements. Independent of transport tier — can be developed in parallel by a different person without coordination.

If you're working with **Level-2 XCH₄ products** (e.g. TROPOMI official retrieval), the entire RTM stack collapses to just the [averaging-kernel operator](00_prerequisites.md#averaging-kernel-operator); this whole page becomes "use the published L2." This page assumes Level-1 (radiance) work, where you build the retrieval yourself.

---

## (1) Simple model — line-by-line via HAPI

[HAPI (HITRAN Application Programming Interface)](https://hitran.org/hapi/) provides absorption cross-sections `σ(ν, T, p)` from the HITRAN database. **All operational satellites for methane retrieval are SWIR** — solar reflection, not thermal emission. The two-way path matters and so does scattering; a pure clear-sky Beer-Lambert model is biased by 10–30% on aerosol-loaded scenes.

### Clear-sky two-way Beer-Lambert (SWIR scope)

```
L(ν) = (F_solar(ν) / π) · A_surf(ν) · cos(SZA) · exp(−τ_total(ν))

τ_total(ν) = τ(ν) / cos(SZA) + τ(ν) / cos(VZA)
τ(ν)       = ∫ σ(ν, T(z), p(z)) · ρ_CH4(z) dz
```

- `F_solar(ν)` — top-of-atmosphere solar irradiance.
- `A_surf(ν)` — surface albedo (Lambertian for v1, BRDF for v2).
- `(SZA, VZA, RAA)` — solar zenith, viewing zenith, relative azimuth from L1 metadata.
- `τ_total` — **two-way** optical depth: light goes down through the atmosphere, reflects, comes back up. The earlier `exp(−τ)` form was one-way and silently biased at oblique geometries.

### Thermal-IR addendum

For TIR work (legacy / portability), the surface term is emission, not solar reflection:

```
L_TIR(ν) = ε_surf(ν) · B(T_surf, ν) · exp(−τ(ν))  +  ∫ B(ν, T(z)) · d exp(−τ(ν, z))
```

Different physics, different priors (`ε_surf` instead of `A_surf`, `T_surf` instead of `F_solar`). Don't conflate the two surface models in code — split per spectral regime.

### Scattering — out-of-scope for v1, planned for v2

SWIR aerosol scattering is the leading systematic for methane retrievals over bright/aerosol-loaded scenes. v1 commits to **clear-sky direct-beam** with explicit AOD-based screening (reject pixels with `AOD > 0.2`); v2 couples to **LIDORT / DISORT / 6S** for multiple-scattering Jacobians. Validation against operational L2 in Step 4 must stratify by AOD to expose the v1 limit.

### Line shape and continuum

- **Voigt profile** with explicit wing cutoffs (default: 25 cm⁻¹). HAPI is the source of truth; expose the cutoff as a config knob — too tight underestimates absorption in the wings, too wide adds noise.
- **CO₂-specific line mixing** in the methane window. Required because CO₂ overlaps the 1.65 μm CH₄ band; ignoring line mixing biases retrievals by ~5 ppb.
- **MT_CKD H₂O continuum.** Non-trivial in the methane band; load alongside HAPI cross-sections.

### HAPI traceability

HAPI is **not JAX-traceable** (NumPy + cached database lookups). The architectural choice is explicit:

- **Path A — pre-tabulate, JAX-trace.** HAPI generates `σ(ν, T, p)` lookup tables offline; runtime `forward.py` interpolates inside `jax.jit`. This is what [`hapi_lut/`](../../src/plume_simulation/hapi_lut/) already implements. Default for v1.
- **Path B — `jax.pure_callback` with custom VJP.** Wrap HAPI calls in a callback when needed inside differentiable code. Use only for cross-section sensitivities not pre-tabulated.

Document the chosen path on every forward-RT helper. `jax.jacobian` "exactly" works only on Path A; on Path B, the VJP is whatever you wrote.

---

## (2) Model-based inference

### Joint state vector

Operational SWIR retrievals do **not** retrieve XCH₄ alone. The state vector is jointly:

```
x = (profile_CH4, profile_H2O, A_surf, AOD, surface_pressure_offset)
```

- `profile_CH4` — CH₄ vertical profile (typically 12–30 layers).
- `profile_H2O` — H₂O profile is coupled (overlapping bands, continuum).
- `A_surf` — spectrally-resolved albedo.
- `AOD` — aerosol optical depth (coarse-mode and fine-mode separately for high-fidelity work).
- `surface_pressure_offset` — DEM error proxy; small but matters for column accounting.

Single-parameter retrievals (XCH₄ only) are biased — the coupling is real and load-bearing.

### Prior `S_a`

| Element | Form | Notes |
|---------|------|-------|
| `profile_CH4` | climatological covariance + AR(1) in vertical | smoothness prior; covariance from CAMS reanalysis |
| `profile_H2O` | same structure | from ECMWF or in-situ profile climatology |
| `A_surf` | per-band Gaussian around L1 prior or MODIS climatology | |
| `AOD` | LogNormal(μ_AOD, σ²) | non-negative, heavy-tail |
| `p_surf_offset` | tight Gaussian around DEM | sub-pixel terrain uncertainty |

Diagonal `S_a` is wrong — produces wildly noisy retrieved profiles. Always carry vertical correlation in the gas profiles.

### Iterative Gauss-Newton

The closed-form formula in the prior version is the *first* update. The converged retrieval iterates:

```
x^{k+1} = x_a + G_k · (y − F(x^k) + K_k (x^k − x_a))
G_k     = (K_k^T S_ε^{-1} K_k + S_a^{-1})^{-1} K_k^T S_ε^{-1}
K_k     = ∂F/∂x at x^k    (JAX, on Path A)
```

**Convergence criterion** (Rodgers 2000 §5.7):

```
δχ²_k = (x^{k+1} − x^k)^T (S*)^{-1} (x^{k+1} − x^k)  <  0.5 · DOFs
```

Cap at `k_max = 10`. Pixels that fail to converge get a quality flag.

### Posterior covariance and information content

Standard outputs of optimal estimation — should appear on every retrieved pixel:

```
S*   = (K^T S_ε^{-1} K + S_a^{-1})^{-1}            posterior covariance
A    = G K                                          averaging kernel
DOFs = tr(A)                                        degrees of freedom for signal
H    = ½ log det(S_a) − ½ log det(S*)              Shannon information content
ΔS   = tr(S_a) − tr(S*)                             posterior contraction
```

These are load-bearing for instrument-design questions (EMIT vs Tanager vs TROPOMI comparisons) and for the cross-tier UQ pipeline that Tier IV assembles. Currently absent from the doc — must appear in the retrieved-product schema.

### Quality flags

Each retrieval emits a flag bitmask:

- `χ²` excessive (poor fit)
- non-convergence (`k = k_max` reached)
- cloud / cirrus
- sun-glint over water
- AOD > screening threshold
- snow / ice
- DEM error excessive

Without these flags, downstream Tiers II–IV silently consume bad retrievals → corrupted source posteriors.

---

## (3) Model emulator — two levels

### Level A — factorised LUT RTM

A dense LUT over `(T, p, q_CH4, A_surf, SZA, VZA, RAA, AOD)` has ~`4 × 10¹³` cells — untrainable. **Operational practice: factorise.**

Decompose the radiance into multiplicative / additive components on lower-dimensional sub-LUTs:

```
L(ν) = T_gas(ν | T(z), p(z), q_CH4(z))      sub-LUT 1: gas transmittance per layer
     × R_surf(ν | A_surf, SZA, VZA, RAA)    sub-LUT 2: surface BRDF kernel
     + S_scatt(ν | AOD, SZA, VZA, RAA)      sub-LUT 3 (v2): scattering source term
```

Sub-LUT sizes are tractable (~10⁶ cells each). Combine analytically at runtime. **This is the only viable path for high-dimensional SWIR retrieval** — without factorisation, LUTs don't fit.

- Pros: bit-exact reproducibility, conservative.
- Cons: factorisation introduces approximation error at sub-LUT interaction boundaries; needs Step-4-style validation.

### Level B — Neural RTM

MLP / Fourier-feature network mapping `(profile, geometry, A_surf, AOD) → L(ν)`. Train on factorised LUT or directly on HAPI outputs.

- **Architecture choice.** SIREN is mentioned in the original draft but is justified for *spatial* signals; spectra are smooth in ν with sharp absorption features, which favours **Fourier-feature MLP** or **per-band wavelet basis**. SIREN works but isn't the obvious pick — benchmark before committing.
- **Per-scene-class heads.** Land vs. water vs. sun-glint vs. ice have different spectral signatures and different aerosol regimes. Either one network per scene class (dispatched by L1 scene flag) or scene class as a categorical input — *don't* train a single net across all scenes.
- **Spectral resolution.** TROPOMI ~1000 channels in the methane window; EMIT ~285. Per-instrument heads at native resolution; don't pre-convolve to a common grid.
- Pros: smooth, differentiable everywhere, compact (~MB vs ~GB for an LUT).
- Cons: training is non-trivial, needs validation against HAPI on out-of-distribution states.
- Reference implementations: JPL FastMDA, ESA's neural SCIAMACHY RTM.

### Neural-Jacobian calibration is mandatory

Backprop through a trained neural RTM gives *some* gradient — whether it matches HAPI's `K` is an empirical question, and the entire Step-4 retrieval depends on it. **Hard validation test:** neural-RTM Jacobian vs. HAPI Jacobian on a held-out state set, `<5%` relative error in operator norm. If the neural RTM has accurate forward predictions but a biased Jacobian, the retrieval converges to the wrong state.

---

## (4) Emulator-based inference

Replace HAPI with the neural RTM in the optimal-estimation loop. The entire retrieval becomes differentiable end-to-end:

```
jax.grad(‖NeuralRTM(x) − y‖²) → ∇_x J
```

Same Gauss-Newton iteration, ~1000× faster per step, gradients trivially available.

- **Forward validation:** posterior from neural-RTM retrieval ≈ posterior from HAPI retrieval on the same observations.
- **Adjoint validation:** neural-RTM Jacobian ≈ HAPI Jacobian (Step-3 calibration test). If this fails, the inversion is biased even when forward predictions look fine.

---

## (5) Amortized inference (predictor)

```
f_θ: (y_radiance, geometry, prior_atm, instrument_id) → p(profile_CH4, A_surf, AOD | y)
```

### Output the joint posterior

Collapsing to `p(XCH₄ | y)` throws away information Tier IV wants — the joint `(profile, albedo, AOD)` is the right output. Collapse to XCH₄ at the consumer side, not in the predictor.

### Per-instrument heads

Different spectral resolutions (TROPOMI ~1000 ch, EMIT ~285 ch, GHGSat hyperspectral, Tanager hyperspectral) → per-instrument predictor heads dispatched by `instrument_id`. Same pattern as Tiers II/III.

### Context conditioning

`prior_atm = (T(z), p(z), q_H2O(z))` from the [met field](00_prerequisites.md#metfield-schema) and `geometry = (SZA, VZA, RAA)` from L1 metadata. Wire in via FiLM / hypernet primitives in [`pyrox.nn`](https://github.com/jejjohnson/pyrox) — same pattern as Tiers I/II/III.

### Posterior over the spatial profile

Conditional flow over the vertical profile (1D — `gauss_flows` handles this natively, no 2D extension needed) for `profile_CH4` and joint Gaussian for `(A_surf, AOD)` is the simplest split. Validate via SBC against HAPI-based optimal-estimation posteriors on synthetic data, **stratified by SNR / SZA / AOD**.

### Uncertainty calibration

SBC is necessary but not sufficient — the *specific* requirement is that the predictor's standard deviation matches the empirical RMSE on a held-out set, stratified by SNR / SZA / AOD. Without stratification, calibration on the easy regime hides miscalibration on the hard regime.

---

## (6) Improve

- **Multi-window retrieval.** Joint CH₄ + CO + H₂O across multiple SWIR bands tightens the posterior and resolves degeneracies.
- **Multiple scattering.** Couple to LIDORT / DISORT / 6S for AOD > 0.2 scenes (the v1 screening threshold).
- **Surface BRDF.** Cox-Munk for sun-glint over water; RPV / Ross-Li for vegetated surfaces; per-band Lambertian for snow. Replace the single-Lambertian assumption with a BRDF registry.
- **Heteroscedastic retrieval uncertainty.** Output retrieval-error covariance per observation, not a fixed `S_ε`.
- **Polarisation.** For polarisation-capable instruments (Sentinel-3 SLSTR, future missions), add Stokes-vector RTM. Out of scope for `plumax` v1, deferred-decision note.
- **Hierarchical Matern length-scale on profile prior.** Promote the AR(1) decorrelation length to a hyperparameter — same pattern as Tiers II/III.

---

## Module layout

| Step | Concern | Module | Status |
|------|---------|--------|--------|
| 1 | HAPI Beer-Lambert (clear-sky, two-way) | [`hapi_lut/beers.py`](../../src/plume_simulation/hapi_lut/beers.py) | ✓ — **add** two-way path if not already |
| 1 | LUT generator | [`hapi_lut/generator.py`](../../src/plume_simulation/hapi_lut/generator.py) | ✓ |
| 1 | LUT config | [`hapi_lut/config.py`](../../src/plume_simulation/hapi_lut/config.py) | ✓ |
| 1 | Multi-gas LUT | [`hapi_lut/multi.py`](../../src/plume_simulation/hapi_lut/multi.py) | ✓ |
| 1 | Factorised LUT (gas × surf × scatt) | `plume_simulation.hapi_lut.factorised` | ☐ |
| 1 | Forward RTM | [`radtran/forward.py`](../../src/plume_simulation/radtran/forward.py) | 🚧 |
| 1 | Spectral response (SRF) | [`radtran/srf.py`](../../src/plume_simulation/radtran/srf.py) | ✓ |
| 1 | Instrument model | [`radtran/instrument.py`](../../src/plume_simulation/radtran/instrument.py) | ✓ |
| 1 | Background atmosphere | [`radtran/background.py`](../../src/plume_simulation/radtran/background.py) | ✓ |
| 1 | Target gas spec | [`radtran/target.py`](../../src/plume_simulation/radtran/target.py) | ✓ |
| 1 | Surface model — SWIR (albedo / BRDF) | `plume_simulation.radtran.surface_swir` | ☐ |
| 1 | Surface model — TIR (emissivity) | `plume_simulation.radtran.surface_tir` | ☐ |
| 1 | Aerosol / scattering coupling (v2) | `plume_simulation.radtran.scattering` | ☐ |
| — | Matched filter (detection) | [`radtran/matched_filter.py`](../../src/plume_simulation/radtran/matched_filter.py), [`matched_filter/`](../../src/plume_simulation/matched_filter/) | ✓ |
| — | gaussx-based linear solve | [`radtran/gaussx_solve.py`](../../src/plume_simulation/radtran/gaussx_solve.py) | ✓ |
| 2 | Optimal-estimation iterative loop | `plume_simulation.radtran.retrieval` | ☐ — clarify scope vs. `gaussx_solve.py` (linear solve only there) |
| 2 | Quality flags + screening | `plume_simulation.radtran.quality` | ☐ |
| 2 | Information-content diagnostics | `plume_simulation.radtran.diagnostics` | ☐ |
| 2 | Posterior export → Tier IV | `plume_simulation.radtran.posterior_export` | ☐ |
| 3 | Neural RTM (per scene class) | `plume_simulation.radtran.neural_rtm` | ☐ |
| 3 | Neural-Jacobian calibration harness | `plume_simulation.radtran.neural_jacobian_test` | ☐ |
| 5 | Direct retrieval predictor (per instrument) | `plume_simulation.radtran.predictor` | ☐ |

---

## Validation strategy

- **HAPI Beer-Lambert — unit optical depth.** For a known column and a known cross-section, `τ` must match the analytical product `σ × column`.
- **Two-way path consistency.** At nadir (SZA = VZA = 0), `τ_total = 2τ`; at oblique geometries, the airmass factor must match `1/cos(SZA) + 1/cos(VZA)`. Catches one-way / two-way bugs.
- **LUT vs. HAPI — in-distribution.** Interpolation error <1% on the radiance for any state inside the LUT bounding box.
- **LUT vs. HAPI — OOD.** Random states *outside* the box should fail loudly via boundary checks — not silently extrapolate.
- **Neural RTM forward.** Held-out set spans all training-distribution corners; report worst-case relative error per channel, stratified by scene class.
- **Neural-RTM Jacobian.** Calibration vs. HAPI Jacobian, `<5%` operator-norm error. **Hard test** — failure means Step 4 retrieval is biased.
- **Optimal estimation against synthetic truth.** Generate radiances for a known profile via HAPI, retrieve, compare retrieved profile to truth. Should sit inside the reported posterior covariance with ~68% frequency over many trials.
- **Information-content recovery.** Generate synthetic radiances at varying SNR / SZA, report retrieved DOFs vs. theoretical maximum from instrument SRF. Catches retrieval-side regressions invisible to mean-error tests.
- **Real-data benchmark.** Compare HAPI-based retrieval to TROPOMI / EMIT / GHGSat *official* L2 product on overlap pixels. Median bias < 5 ppb, RMSE within instrument noise floor. Stratify by AOD to expose the v1 clear-sky-only limit. **Without this the retrieval is a synthetic exercise.**
- **Predictor calibration.** SBC stratified by SNR / SZA / AOD; standard-deviation calibration vs. empirical RMSE on held-out set.

---

## Open questions

- **Spectral resolution storage.** Native HAPI is sub-cm⁻¹. Operational instruments are ~0.1 nm. Store at native HAPI then convolve with SRF at runtime (storage-efficient but slow), or pre-convolve per instrument (fast but bloats storage)? Probably runtime convolution with a sparse SRF representation — open: pick the SRF sparsity scheme.
- **Vertical profile representation.** Layer-mean concentrations vs. discretised continuous profiles. Affects how the AK is constructed and how the prior covariance is parameterised.
- **Surface BRDF beyond Lambertian.** Cox-Munk for water sun-glint, RPV / Ross-Li for vegetation, per-band Lambertian for snow. Build a BRDF registry keyed on land-use class? In scope for v1.5 / v2.
- **Cloud / cirrus screening.** Internal detection step or rely on L1 cloud mask? Likely the latter for v1, but document the trust assumption — TROPOMI cloud mask is conservative, EMIT lacks one, Tanager cloud handling is in flux.
- **Scattering scope.** Strict clear-sky (AOD < 0.2) for v1 vs. multiple-scattering hybrid for v2 — when do we promote? Probably driven by Tier-IV bias diagnostics.
- **HAPI traceability.** Path A (pre-tabulate) vs. Path B (`pure_callback`). v1 default is A. When does B become necessary — when retrieving cross-section sensitivities not pre-computed?
- **Polarisation deferred-decision.** Sentinel-3 SLSTR and future missions are polarised. Plan a v3 Stokes-vector RTM, or stay scalar and discount polarised instruments?
- **Aerosol/cloud joint retrieval vs. screening.** Joint retrieval extends the operational AOD ceiling (currently 0.2) but adds two state-vector elements and slows convergence. Open: do we promote AOD from a *screening flag* to an *inverted parameter*?
