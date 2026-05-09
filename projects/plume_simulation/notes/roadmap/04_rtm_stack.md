---
title: "Radiative transfer (RTM) stack — parallel track"
short_title: "RTM stack"
subject: "plumax — observation operator"
authors:
  - name: J. Emmanuel Johnson
    affiliations: [UNEP, IMEO, MARS]
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: [radiative transfer, HITRAN, HAPI, Beer-Lambert, optimal estimation, SWIR retrieval, methane retrieval, neural RTM, averaging kernel, optimal estimation, Rodgers]
---

# Radiative transfer (RTM) stack — parallel track

The RTM is the **observation operator** $\mathbf{H}_\text{obs} : c(\mathbf{x},t) \to \mathbf{y}_\text{radiance}$. It connects Tiers II–IV to actual satellite measurements. Independent of transport tier — can be developed in parallel by a different person without coordination.

If you're working with **Level-2 XCH₄ products** (e.g. TROPOMI official retrieval; {cite:p}`s5p_tropomi`), the entire RTM stack collapses to just the [averaging-kernel operator](00_prerequisites.md#prereqs-ak-operator); this whole page becomes "use the published L2." This page assumes Level-1 (radiance) work, where you build the retrieval yourself.

---

(rtm-simple-model)=
## (1) Simple model — line-by-line via HAPI

[HAPI (HITRAN Application Programming Interface)](https://hitran.org/hapi/) ({cite:p}`gordon2022hitran,kochanov2016hapi`) provides absorption cross-sections $\sigma(\nu, T, p)$ from the HITRAN database. **All operational satellites for methane retrieval are SWIR** — solar reflection, not thermal emission. The two-way path matters and so does scattering; a pure clear-sky Beer–Lambert model is biased by 10–30% on aerosol-loaded scenes.

(rtm-beer-lambert)=
### Clear-sky two-way Beer–Lambert (SWIR scope)

```{math}
:label: eq-rtm-radiance
L(\nu) \;=\; \frac{F_\text{solar}(\nu)}{\pi}\, A_\text{surf}(\nu)\, \cos(\text{SZA})\, \exp\!\bigl(-\tau_\text{total}(\nu)\bigr)
```

```{math}
:label: eq-rtm-tau-total
\tau_\text{total}(\nu) \;=\; \frac{\tau(\nu)}{\cos(\text{SZA})} \;+\; \frac{\tau(\nu)}{\cos(\text{VZA})}
```

```{math}
:label: eq-rtm-tau
\tau(\nu) \;=\; \int \sigma\!\bigl(\nu, T(z), p(z)\bigr)\, \rho_{\text{CH}_4}(z)\, \mathrm{d}z
```

- $F_\text{solar}(\nu)$ — top-of-atmosphere solar irradiance.
- $A_\text{surf}(\nu)$ — surface albedo (Lambertian for v1, BRDF for v2).
- $(\text{SZA}, \text{VZA}, \text{RAA})$ — solar zenith, viewing zenith, relative azimuth from L1 metadata.
- $\tau_\text{total}$ — **two-way** optical depth: light goes down through the atmosphere, reflects, comes back up. The earlier $\exp(-\tau)$ form was one-way and silently biased at oblique geometries.

(rtm-thermal-ir)=
### Thermal-IR addendum

For TIR work (legacy / portability), the surface term is emission, not solar reflection:

```{math}
:label: eq-rtm-thermal-ir
L_\text{TIR}(\nu) \;=\; \varepsilon_\text{surf}(\nu)\, B(T_\text{surf}, \nu)\, \exp\!\bigl(-\tau(\nu)\bigr) \;+\; \int B(\nu, T(z))\, \mathrm{d}\!\exp\!\bigl(-\tau(\nu, z)\bigr)
```

Different physics, different priors ($\varepsilon_\text{surf}$ instead of $A_\text{surf}$, $T_\text{surf}$ instead of $F_\text{solar}$). Don't conflate the two surface models in code — split per spectral regime.

### Scattering — out-of-scope for v1, planned for v2

SWIR aerosol scattering is the leading systematic for methane retrievals over bright/aerosol-loaded scenes. v1 commits to **clear-sky direct-beam** with explicit AOD-based screening (reject pixels with $\text{AOD} > 0.2$); v2 couples to **LIDORT / DISORT / 6S** for multiple-scattering Jacobians. Validation against operational L2 in Step 4 must stratify by AOD to expose the v1 limit.

### Line shape and continuum

- **Voigt profile** with explicit wing cutoffs (default: 25 cm⁻¹). HAPI is the source of truth; expose the cutoff as a config knob — too tight underestimates absorption in the wings, too wide adds noise.
- **CO₂-specific line mixing** in the methane window. Required because CO₂ overlaps the 1.65 μm CH₄ band; ignoring line mixing biases retrievals by ~5 ppb.
- **MT_CKD H₂O continuum.** Non-trivial in the methane band; load alongside HAPI cross-sections.

(rtm-hapi-trace)=
### HAPI traceability

:::{important} HAPI is not JAX-traceable
The architectural choice is explicit: pre-tabulate, then trace.

- **Path A — pre-tabulate, JAX-trace.** HAPI generates $\sigma(\nu, T, p)$ lookup tables offline; runtime `forward.py` interpolates inside `jax.jit`. This is what [`hapi_lut/`](../../src/plume_simulation/hapi_lut/) already implements. Default for v1.
- **Path B — `jax.pure_callback` with custom VJP.** Wrap HAPI calls in a callback when needed inside differentiable code. Use only for cross-section sensitivities not pre-tabulated.

**Why:** `jax.jacobian` "exactly" works only on Path A; on Path B, the VJP is whatever you wrote. Document the chosen path on every forward-RT helper.
:::

---

(rtm-inference)=
## (2) Model-based inference

(rtm-state-vector)=
### Joint state vector

Operational SWIR retrievals do **not** retrieve XCH₄ alone. The state vector is jointly:

```{math}
:label: eq-rtm-state-vector
\mathbf{x} \;=\; (\text{profile}_{\text{CH}_4},\; \text{profile}_{\text{H}_2\text{O}},\; A_\text{surf},\; \text{AOD},\; p_\text{surf,offset})
```

- $\text{profile}_{\text{CH}_4}$ — CH₄ vertical profile (typically 12–30 layers).
- $\text{profile}_{\text{H}_2\text{O}}$ — H₂O profile is coupled (overlapping bands, continuum).
- $A_\text{surf}$ — spectrally-resolved albedo.
- $\text{AOD}$ — aerosol optical depth (coarse-mode and fine-mode separately for high-fidelity work).
- $p_\text{surf,offset}$ — DEM error proxy; small but matters for column accounting.

:::{caution} Don't retrieve XCH₄ alone
Single-parameter retrievals (XCH₄ only) are biased — the coupling is real and load-bearing.
:::

(rtm-prior)=
### Prior $\mathbf{S}_a$

```{list-table} Prior structure for the joint RTM state vector.
:label: tbl-rtm-prior
:header-rows: 1

* - Element
  - Form
  - Notes
* - $\text{profile}_{\text{CH}_4}$
  - climatological covariance + AR(1) in vertical
  - smoothness prior; covariance from CAMS reanalysis
* - $\text{profile}_{\text{H}_2\text{O}}$
  - same structure
  - from ECMWF or in-situ profile climatology
* - $A_\text{surf}$
  - per-band Gaussian around L1 prior or MODIS climatology
  -
* - $\text{AOD}$
  - $\operatorname{LogNormal}(\mu_\text{AOD}, \sigma^2)$
  - non-negative, heavy-tail
* - $p_\text{surf,offset}$
  - tight Gaussian around DEM
  - sub-pixel terrain uncertainty
```

:::{important} Diagonal $\mathbf{S}_a$ is wrong
Always carry vertical correlation in the gas profiles — diagonal $\mathbf{S}_a$ produces wildly noisy retrieved profiles.
:::

(rtm-gauss-newton)=
### Iterative Gauss–Newton

The closed-form formula in the prior version is the *first* update. The converged retrieval iterates ({cite:p}`rodgers2000`):

```{math}
:label: eq-rtm-gauss-newton
\mathbf{x}^{k+1} \;=\; \mathbf{x}_a + \mathbf{G}_k\, \bigl(\mathbf{y} - F(\mathbf{x}^k) + \mathbf{K}_k (\mathbf{x}^k - \mathbf{x}_a)\bigr)
```

```{math}
:label: eq-rtm-gain
\mathbf{G}_k \;=\; (\mathbf{K}_k^{\top}\, \mathbf{S}_\varepsilon^{-1}\, \mathbf{K}_k + \mathbf{S}_a^{-1})^{-1}\, \mathbf{K}_k^{\top}\, \mathbf{S}_\varepsilon^{-1}, \qquad \mathbf{K}_k \;=\; \frac{\partial F}{\partial \mathbf{x}}\bigg|_{\mathbf{x}^k}
```

**Convergence criterion** ({cite:t}`rodgers2000` §5.7):

```{math}
:label: eq-rtm-convergence
\delta \chi^{2}_{k} \;=\; (\mathbf{x}^{k+1} - \mathbf{x}^{k})^{\top}\, (\mathbf{S}^{*})^{-1}\, (\mathbf{x}^{k+1} - \mathbf{x}^{k}) \;<\; 0.5 \cdot \text{DOFs}
```

Cap at $k_\text{max} = 10$. Pixels that fail to converge get a quality flag.

(rtm-info-content)=
### Posterior covariance and information content

Standard outputs of optimal estimation — should appear on every retrieved pixel:

```{math}
:label: eq-rtm-posterior-cov
\mathbf{S}^{*} \;=\; (\mathbf{K}^{\top}\, \mathbf{S}_\varepsilon^{-1}\, \mathbf{K} + \mathbf{S}_a^{-1})^{-1}
```

```{math}
:label: eq-rtm-averaging-kernel
\mathbf{A} \;=\; \mathbf{G}\, \mathbf{K}, \qquad \text{DOFs} \;=\; \operatorname{tr}(\mathbf{A})
```

```{math}
:label: eq-rtm-info-content
H \;=\; \tfrac{1}{2} \log \det(\mathbf{S}_a) - \tfrac{1}{2} \log \det(\mathbf{S}^{*}), \qquad \Delta \mathbf{S} \;=\; \operatorname{tr}(\mathbf{S}_a) - \operatorname{tr}(\mathbf{S}^{*})
```

These are load-bearing for instrument-design questions (EMIT vs Tanager vs TROPOMI comparisons) and for the cross-tier UQ pipeline that Tier IV assembles. Currently absent from the doc — must appear in the retrieved-product schema.

(rtm-quality-flags)=
### Quality flags

Each retrieval emits a flag bitmask:

- $\chi^{2}$ excessive (poor fit)
- non-convergence ($k = k_\text{max}$ reached)
- cloud / cirrus
- sun-glint over water
- AOD > screening threshold
- snow / ice
- DEM error excessive

:::{caution} Quality flags are mandatory upstream
Without these flags, downstream Tiers II–IV silently consume bad retrievals → corrupted source posteriors.
:::

---

(rtm-emulator)=
## (3) Model emulator — two levels

(rtm-emu-lut)=
### Level A — factorised LUT RTM

A dense LUT over $(T, p, q_{\text{CH}_4}, A_\text{surf}, \text{SZA}, \text{VZA}, \text{RAA}, \text{AOD})$ has $\sim 4 \times 10^{13}$ cells — untrainable. **Operational practice: factorise.**

Decompose the radiance into multiplicative / additive components on lower-dimensional sub-LUTs:

```{math}
:label: eq-rtm-factorised-lut
L(\nu) \;=\; \underbrace{T_\text{gas}\!\bigl(\nu \,\big|\, T(z), p(z), q_{\text{CH}_4}(z)\bigr)}_{\text{sub-LUT 1: gas transmittance}}
\;\times\; \underbrace{R_\text{surf}\!\bigl(\nu \,\big|\, A_\text{surf}, \text{SZA}, \text{VZA}, \text{RAA}\bigr)}_{\text{sub-LUT 2: surface BRDF kernel}}
\;+\; \underbrace{S_\text{scatt}\!\bigl(\nu \,\big|\, \text{AOD}, \text{SZA}, \text{VZA}, \text{RAA}\bigr)}_{\text{sub-LUT 3 (v2): scattering source term}}
```

Sub-LUT sizes are tractable (~$10^{6}$ cells each). Combine analytically at runtime. **This is the only viable path for high-dimensional SWIR retrieval** — without factorisation, LUTs don't fit.

- Pros: bit-exact reproducibility, conservative.
- Cons: factorisation introduces approximation error at sub-LUT interaction boundaries; needs Step-4-style validation.

(rtm-emu-neural)=
### Level B — Neural RTM

MLP / Fourier-feature network mapping $(\text{profile}, \text{geometry}, A_\text{surf}, \text{AOD}) \to L(\nu)$. Train on factorised LUT or directly on HAPI outputs.

- **Architecture choice.** SIREN is mentioned in earlier drafts but is justified for *spatial* signals; spectra are smooth in $\nu$ with sharp absorption features, which favours **Fourier-feature MLP** or **per-band wavelet basis**. SIREN works but isn't the obvious pick — benchmark before committing.
- **Per-scene-class heads.** Land vs. water vs. sun-glint vs. ice have different spectral signatures and different aerosol regimes. Either one network per scene class (dispatched by L1 scene flag) or scene class as a categorical input — *don't* train a single net across all scenes.
- **Spectral resolution.** TROPOMI ~1000 channels in the methane window; EMIT ~285. Per-instrument heads at native resolution; don't pre-convolve to a common grid.
- Pros: smooth, differentiable everywhere, compact (~MB vs ~GB for an LUT).
- Cons: training is non-trivial, needs validation against HAPI on out-of-distribution states.
- Reference implementations: JPL FastMDA, ESA's neural SCIAMACHY RTM ({cite:p}`sciamachy`).

(rtm-neural-jacobian)=
### Neural-Jacobian calibration is mandatory

Backprop through a trained neural RTM gives *some* gradient — whether it matches HAPI's $\mathbf{K}$ is an empirical question, and the entire Step-4 retrieval depends on it.

:::{important} Hard validation test
Neural-RTM Jacobian vs. HAPI Jacobian on a held-out state set, $<5\%$ relative error in operator norm. If the neural RTM has accurate forward predictions but a biased Jacobian, the retrieval converges to the wrong state.
:::

---

(rtm-emu-inference)=
## (4) Emulator-based inference

Replace HAPI with the neural RTM in the optimal-estimation loop. The entire retrieval becomes differentiable end-to-end:

```{math}
:label: eq-rtm-jax-grad
\nabla_{\mathbf{x}} J \;=\; \texttt{jax.grad}\!\Bigl(\bigl\lVert \texttt{NeuralRTM}(\mathbf{x}) - \mathbf{y}\bigr\rVert^{2}\Bigr)
```

Same Gauss–Newton iteration, ~1000× faster per step, gradients trivially available.

- **Forward validation:** posterior from neural-RTM retrieval ≈ posterior from HAPI retrieval on the same observations.
- **Adjoint validation:** neural-RTM Jacobian ≈ HAPI Jacobian (Step-3 calibration test). If this fails, the inversion is biased even when forward predictions look fine.

---

(rtm-amortized)=
## (5) Amortized inference (predictor)

```{math}
:label: eq-rtm-amortized
f_\theta : (\mathbf{y}_\text{radiance},\, \text{geometry},\, \text{prior}_\text{atm},\, \text{instrument\_id}) \;\longmapsto\; p(\text{profile}_{\text{CH}_4},\, A_\text{surf},\, \text{AOD} \mid \mathbf{y})
```

### Output the joint posterior

Collapsing to $p(\text{XCH}_4 \mid \mathbf{y})$ throws away information Tier IV wants — the joint $(\text{profile}, \text{albedo}, \text{AOD})$ is the right output. Collapse to XCH₄ at the consumer side, not in the predictor.

### Per-instrument heads

Different spectral resolutions (TROPOMI ~1000 ch {cite:p}`s5p_tropomi`, EMIT ~285 ch {cite:p}`emit`, GHGSat hyperspectral {cite:p}`ghgsat`, Tanager hyperspectral {cite:p}`carbon_mapper`) → per-instrument predictor heads dispatched by `instrument_id`. Same pattern as Tiers II/III.

### Context conditioning

$\text{prior}_\text{atm} = (T(z), p(z), q_{\text{H}_2\text{O}}(z))$ from the [met field](00_prerequisites.md#prereqs-metfield-schema) and $\text{geometry} = (\text{SZA}, \text{VZA}, \text{RAA})$ from L1 metadata. Wire in via FiLM / hypernet primitives in [`pyrox.nn`](https://github.com/jejjohnson/pyrox) — same pattern as Tiers I/II/III.

### Posterior over the spatial profile

Conditional flow over the vertical profile (1D — `gauss_flows` handles this natively, no 2D extension needed) for $\text{profile}_{\text{CH}_4}$ and joint Gaussian for $(A_\text{surf}, \text{AOD})$ is the simplest split. Validate via SBC against HAPI-based optimal-estimation posteriors on synthetic data, **stratified by SNR / SZA / AOD**.

### Uncertainty calibration

SBC is necessary but not sufficient — the *specific* requirement is that the predictor's standard deviation matches the empirical RMSE on a held-out set, stratified by SNR / SZA / AOD. Without stratification, calibration on the easy regime hides miscalibration on the hard regime.

---

(rtm-improve)=
## (6) Improve

- **Multi-window retrieval.** Joint CH₄ + CO + H₂O across multiple SWIR bands tightens the posterior and resolves degeneracies.
- **Multiple scattering.** Couple to LIDORT / DISORT / 6S for $\text{AOD} > 0.2$ scenes (the v1 screening threshold).
- **Surface BRDF.** Cox–Munk for sun-glint over water; RPV / Ross–Li for vegetated surfaces; per-band Lambertian for snow. Replace the single-Lambertian assumption with a BRDF registry.
- **Heteroscedastic retrieval uncertainty.** Output retrieval-error covariance per observation, not a fixed $\mathbf{S}_\varepsilon$.
- **Polarisation.** For polarisation-capable instruments (Sentinel-3 SLSTR {cite:p}`sentinel3`, future missions), add Stokes-vector RTM. Out of scope for `plumax` v1, deferred-decision note.
- **Hierarchical Matérn length-scale on profile prior.** Promote the AR(1) decorrelation length to a hyperparameter — same pattern as Tiers II/III.

---

(rtm-modules)=
## Module layout

```{list-table} RTM stack module layout — step, concern, target module, status.
:label: tbl-rtm-modules
:header-rows: 1

* - Step
  - Concern
  - Module
  - Status
* - 1
  - HAPI Beer-Lambert (clear-sky, two-way)
  - [`hapi_lut/beers.py`](../../src/plume_simulation/hapi_lut/beers.py)
  - ✓ — **add** two-way path if not already
* - 1
  - LUT generator
  - [`hapi_lut/generator.py`](../../src/plume_simulation/hapi_lut/generator.py)
  - ✓
* - 1
  - LUT config
  - [`hapi_lut/config.py`](../../src/plume_simulation/hapi_lut/config.py)
  - ✓
* - 1
  - Multi-gas LUT
  - [`hapi_lut/multi.py`](../../src/plume_simulation/hapi_lut/multi.py)
  - ✓
* - 1
  - Factorised LUT (gas × surf × scatt)
  - `plume_simulation.hapi_lut.factorised`
  - ☐
* - 1
  - Forward RTM
  - [`radtran/forward.py`](../../src/plume_simulation/radtran/forward.py)
  - 🚧
* - 1
  - Spectral response (SRF)
  - [`radtran/srf.py`](../../src/plume_simulation/radtran/srf.py)
  - ✓
* - 1
  - Instrument model
  - [`radtran/instrument.py`](../../src/plume_simulation/radtran/instrument.py)
  - ✓
* - 1
  - Background atmosphere
  - [`radtran/background.py`](../../src/plume_simulation/radtran/background.py)
  - ✓
* - 1
  - Target gas spec
  - [`radtran/target.py`](../../src/plume_simulation/radtran/target.py)
  - ✓
* - 1
  - Surface model — SWIR (albedo / BRDF)
  - `plume_simulation.radtran.surface_swir`
  - ☐
* - 1
  - Surface model — TIR (emissivity)
  - `plume_simulation.radtran.surface_tir`
  - ☐
* - 1
  - Aerosol / scattering coupling (v2)
  - `plume_simulation.radtran.scattering`
  - ☐
* - —
  - Matched filter (detection)
  - [`radtran/matched_filter.py`](../../src/plume_simulation/radtran/matched_filter.py), [`matched_filter/`](../../src/plume_simulation/matched_filter/)
  - ✓
* - —
  - gaussx-based linear solve
  - [`radtran/gaussx_solve.py`](../../src/plume_simulation/radtran/gaussx_solve.py)
  - ✓
* - 2
  - Optimal-estimation iterative loop
  - `plume_simulation.radtran.retrieval`
  - ☐ — clarify scope vs. `gaussx_solve.py` (linear solve only there)
* - 2
  - Quality flags + screening
  - `plume_simulation.radtran.quality`
  - ☐
* - 2
  - Information-content diagnostics
  - `plume_simulation.radtran.diagnostics`
  - ☐
* - 2
  - Posterior export → Tier IV
  - `plume_simulation.radtran.posterior_export`
  - ☐
* - 3
  - Neural RTM (per scene class)
  - `plume_simulation.radtran.neural_rtm`
  - ☐
* - 3
  - Neural-Jacobian calibration harness
  - `plume_simulation.radtran.neural_jacobian_test`
  - ☐
* - 5
  - Direct retrieval predictor (per instrument)
  - `plume_simulation.radtran.predictor`
  - ☐
```

---

(rtm-validation)=
## Validation strategy

- **HAPI Beer–Lambert — unit optical depth.** For a known column and a known cross-section, $\tau$ must match the analytical product $\sigma \times \text{column}$.
- **Two-way path consistency.** At nadir ($\text{SZA} = \text{VZA} = 0$), $\tau_\text{total} = 2\tau$; at oblique geometries, the airmass factor must match $1/\cos(\text{SZA}) + 1/\cos(\text{VZA})$. Catches one-way / two-way bugs.
- **LUT vs. HAPI — in-distribution.** Interpolation error $<1\%$ on the radiance for any state inside the LUT bounding box.
- **LUT vs. HAPI — OOD.** Random states *outside* the box should fail loudly via boundary checks — not silently extrapolate.
- **Neural RTM forward.** Held-out set spans all training-distribution corners; report worst-case relative error per channel, stratified by scene class.
- **Neural-RTM Jacobian.** Calibration vs. HAPI Jacobian, $<5\%$ operator-norm error. **Hard test** — failure means Step 4 retrieval is biased.
- **Optimal estimation against synthetic truth.** Generate radiances for a known profile via HAPI, retrieve, compare retrieved profile to truth. Should sit inside the reported posterior covariance with ~68% frequency over many trials.
- **Information-content recovery.** Generate synthetic radiances at varying SNR / SZA, report retrieved DOFs vs. theoretical maximum from instrument SRF. Catches retrieval-side regressions invisible to mean-error tests.
- **Real-data benchmark.** Compare HAPI-based retrieval to TROPOMI / EMIT / GHGSat *official* L2 product on overlap pixels ({cite:p}`s5p_tropomi,emit,ghgsat`). Median bias < 5 ppb, RMSE within instrument noise floor. Stratify by AOD to expose the v1 clear-sky-only limit. **Without this the retrieval is a synthetic exercise.**
- **Predictor calibration.** SBC stratified by SNR / SZA / AOD; standard-deviation calibration vs. empirical RMSE on held-out set.

---

(rtm-open-questions)=
## Open questions

:::{attention} Spectral resolution storage
Native HAPI is sub-cm⁻¹. Operational instruments are ~0.1 nm. Store at native HAPI then convolve with SRF at runtime (storage-efficient but slow), or pre-convolve per instrument (fast but bloats storage)? Probably runtime convolution with a sparse SRF representation — open: pick the SRF sparsity scheme.
:::

:::{attention} Vertical profile representation
Layer-mean concentrations vs. discretised continuous profiles. Affects how the AK is constructed and how the prior covariance is parameterised.
:::

:::{attention} Surface BRDF beyond Lambertian
Cox–Munk for water sun-glint, RPV / Ross–Li for vegetation, per-band Lambertian for snow. Build a BRDF registry keyed on land-use class? In scope for v1.5 / v2.
:::

:::{attention} Cloud / cirrus screening
Internal detection step or rely on L1 cloud mask? Likely the latter for v1, but document the trust assumption — TROPOMI cloud mask is conservative, EMIT lacks one, Tanager cloud handling is in flux.
:::

:::{attention} Scattering scope
Strict clear-sky ($\text{AOD} < 0.2$) for v1 vs. multiple-scattering hybrid for v2 — when do we promote? Probably driven by Tier-IV bias diagnostics.
:::

:::{attention} HAPI traceability
Path A (pre-tabulate) vs. Path B (`pure_callback`). v1 default is A. When does B become necessary — when retrieving cross-section sensitivities not pre-computed?
:::

:::{attention} Polarisation deferred-decision
Sentinel-3 SLSTR and future missions are polarised. Plan a v3 Stokes-vector RTM, or stay scalar and discount polarised instruments?
:::

:::{attention} Aerosol/cloud joint retrieval vs. screening
Joint retrieval extends the operational AOD ceiling (currently 0.2) but adds two state-vector elements and slows convergence. Open: do we promote AOD from a *screening flag* to an *inverted parameter*?
:::
