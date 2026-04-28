---
title: "SIREN ↔ RFF ↔ VSSGP — Bayesian Fourier features for geoscience interpolation"
---

# SIREN, RFF, and Variational Sparse Spectrum GPs for geoscience

A reading of the SIREN / Random Fourier Feature (RFF) / Variational Sparse Spectrum GP (VSSGP) family from a geoscience-interpolation viewpoint. The thesis: **SIREN and RFF are special cases of VSSGP with degenerate spectral priors** — and for sparse, banded, irregularly-sampled geophysical fields (SSH, SST, SSS, ocean colour) the VSSGP framing is *strictly more useful* because the spectral structure is something we already know.

This note is split into seven layers, in increasing order of structural sophistication:

1. **§0** — applied-math foundations (Bochner, RFF as Monte Carlo, weight-space ↔ function-space duality, sample complexity, marginal likelihood as Occam's razor).
2. **§1** — SIREN / RFF as special cases of VSSGP, with the four-rung hierarchy.
3. **§2** — why geoscience is the wrong domain for SIREN, with a spectral-budget argument.
4. **§3** — physical priors for SSH / SST / SSS / OC, including the operational-prior recipe (MIOST / DYMOST / DUACS → VSSGP).
5. **§4** — additive-component construction in VSSGP, with a worked SSH example.
6. **§5** — approximation-error and marginal-likelihood-compass results.
7. **§6** — the experiment plan (datasets, metrics, ablations, compute budget).

Companion: [01_physics_constraints.md](01_physics_constraints.md) — how to enforce physical constraints (basis / data / loss) on top of any of these methods, with explicit out-of-domain prediction in mind.

---

## 0. Why Any of This Works — Applied Foundations

Before the hierarchy of methods, it's worth setting down four results that everything else rests on. Each is one line of theory plus one line of "what this buys you in practice."

### 0.1 Bochner's theorem — the bridge

For any continuous, shift-invariant, positive-definite kernel `k(x, x') = κ(x − x')` normalized so `κ(0) = 1`:

```
κ(τ) = ∫ exp(i ωᵀ τ) p(ω) dω
```

i.e. **`κ` and a probability density `p(ω)` are a Fourier pair**. Positive-definiteness of `κ` is equivalent to non-negativity of `p`.

*Engineering consequence.* Picking a kernel and picking a spectral density are the same act. Once you have either, the other is determined by an FFT. This is why you are allowed to design priors directly in the Fourier domain — which is exactly where physical knowledge of geoscience signals lives (annual cycles, mesoscale band, tidal frequencies).

| Kernel | `p(ω)` | Geoscience use |
|---|---|---|
| Squared exponential | `N(0, ℓ⁻²)` | smooth fields, no preferred scale |
| Matérn-ν | Multivariate Student-t | rough fields with controlled differentiability |
| Cosine | `δ(ω − ω₀)` | exact harmonic (annual, diurnal) |
| Spectral mixture (Wilson & Adams) | `Σ wᵢ N(μᵢ, Σᵢ)` | banded structure (mesoscale + gyre + tide) |

### 0.2 RFF as Monte Carlo of the Bochner integral

Take the real form of the Bochner identity (`κ` is real, so we keep only the cosine):

```
κ(x − x') = E_{ω ~ p, b ~ U[0, 2π]} [ 2 cos(ωᵀx + b) cos(ωᵀx' + b) ]
```

Pull `M` samples `(ωᵢ, bᵢ)`. The MC estimator is the inner product of the feature map

```
φ(x) = √(2/M) [ cos(ω₁ᵀx + b₁), …, cos(ω_Mᵀx + b_M) ]
k̂(x, x') = φ(x)ᵀ φ(x')
```

*Sample complexity.* Rahimi & Recht (2008, "Uniform Approximation of Functions with Random Bases") show that on a compact set `X ⊂ ℝ^d` of diameter `R`,

```
P[ sup_{x, x' ∈ X} |k̂(x, x') − k(x, x')| ≥ ε ] ≤ 2⁸ (σ_p R / ε)² exp(−M ε² / (4(d + 2)))
```

so to drive uniform error below `ε` with high probability you need

```
M = O( (d / ε²) log(σ_p R / ε) )
```

*Engineering consequence.* `M` scales linearly with input dimension and inverse-squared in the kernel error you tolerate. Doubling the precision quadruples `M`. For SSH on `(lon, lat, t)` (`d = 3`) and 1% kernel error you typically need `M` in the low thousands — well within reach of a JAX dense matmul.

### 0.3 Weight-space ↔ function-space duality

If you put a Gaussian prior on the weights of the random feature model

```
f(x) = φ(x)ᵀ w,    w ~ N(0, Σ_w)
```

then `f` is itself a Gaussian process with mean `0` and covariance

```
Cov(f(x), f(x')) = φ(x)ᵀ Σ_w φ(x')
```

Setting `Σ_w = σ_f² I` recovers exactly the RFF kernel scaled by `σ_f²`. So the two views are not analogies — they are the **same model, written in different bases**.

| View | Object you optimize | Cost dominated by |
|---|---|---|
| Function-space | Posterior mean `μ(x*) = k_*ᵀ (K + σ_n²I)⁻¹ y` | `O(N³)` for `(K + σ_n²I)⁻¹` |
| Weight-space (RFF) | Posterior mean `μ(x*) = φ(x*)ᵀ E[w | y]` | `O(N M² + M³)` |

*Engineering consequence.* When `M ≪ N` (the geoscience regime — `N ~ 10⁶`, `M ~ 10³`), weight-space is asymptotically `(N/M)²` cheaper. The whole reason SSGP/VSSGP exists as a *practical* method, not just a theoretical bridge, is that the duality flips the cost direction.

### 0.4 Posterior in closed form (the bit that SIREN throws away)

For RFF/SSGP with Gaussian likelihood `y = Φ w + ε`, `ε ~ N(0, σ_n²I)`:

```
Σ_post = (Φᵀ Φ / σ_n² + Σ_w⁻¹)⁻¹              # (M, M)
μ_post = Σ_post Φᵀ y / σ_n²                     # (M,)

mean(f*)  = φ(x*)ᵀ μ_post
var(f*)   = φ(x*)ᵀ Σ_post φ(x*) + σ_n²
```

*Engineering consequence.* This is one Cholesky on an `M × M` matrix. You get not just a point prediction, but a full predictive distribution at every test point — gap-filling uncertainty for free. SIREN replaces this entire block with `argmin_w ‖y − Φw‖² + λ‖w‖²` (i.e. ridge regression with a single tuned scalar `λ`), losing the per-point variance and the principled regularization-from-the-prior.

### 0.5 Marginal likelihood as automatic Occam's razor

The model evidence (with `Σ_w = σ_f² I`)

```
log p(y | Ω, σ_f, σ_n)
   = − ½ yᵀ (σ_f² Φ Φᵀ + σ_n² I)⁻¹ y
     − ½ log |σ_f² Φ Φᵀ + σ_n² I|
     − (N/2) log 2π
```

decomposes into **data-fit** (first term) plus **complexity penalty** (log-determinant). Optimizing this w.r.t. `Ω` and amplitudes pulls spectral mass onto frequencies that explain `y` — but pays a `log|·|` cost for adding mass at frequencies that don't help. There is no equivalent self-regularizing objective in MSE training of SIREN; ω₀ has to be tuned by hand.

*Engineering consequence.* The marginal likelihood gradient w.r.t. spectral points is exactly the signal that says "your spectral prior is wrong, move mass *here*". This is what SSGP exploits; VSSGP softens point-optimized `Ω` to a variational posterior `q(Ω)` to prevent the well-known SSGP overfitting pathology (Lázaro-Gredilla et al. 2010, §5).

---

## 1. SIREN / RFF as Special Cases of VSSGP

**Core claim:** The SIREN and Random Fourier Features (RFF) methods from the neural networks community are special cases of Variational Sparse Spectrum GPs (VSSGP) and Random Feature Expansions (RFE) from the GP community — with uninformative/degenerate priors and no posterior.

### Correspondence Table

| Neural Field | GP Counterpart | What's Missing in the NN Version |
|---|---|---|
| RFF (Rahimi & Recht) | Sparse Spectrum GP (fixed ω, no Bayes) | No posterior, just a kernel approximation |
| SIREN (1-layer) | VSSGP with specific spectral prior p(ω) | Bayesian treatment of weights + frequencies |
| Deep SIREN | Deep GP with sinusoidal RFEs | Uncertainty propagation across layers |
| ω₀ + init heuristic | Kernel lengthscale / spectral density | Principled derivation via Bochner's theorem |

### RFF → Sparse Spectrum GP

Rahimi & Recht's RFF is an MC approximation to a shift-invariant kernel via Bochner's theorem:

```
k(x, y) ≈ φ(x)ᵀφ(y),    φ(x) = √(2/D) [cos(ωᵢᵀx + bᵢ)]
with ωᵢ ~ p_spectral(ω) = FT[k](ω)
```

The Sparse Spectrum GP (Lázaro-Gredilla et al. 2010) does the same but **optimizes** the ωᵢ via marginal likelihood. VSSGP puts a full variational distribution q(ω) over the spectral points.

### SIREN → VSSGP with Box Spectral Prior

A 1-layer SIREN:
```
f(x) = W₂ sin(ω₀ W₁ x + b₁)
```
with `W₁ ~ Uniform(-√(6/n), √(6/n))` is implicitly assuming a spectral prior:

```
p(ω) ∝ Uniform[-ω₀√(6/n), ω₀√(6/n)]ᵈ
```

This is a **box spectral prior** — corresponding to a product sinc kernel. VSSGP would let you:
1. **Infer** the spectral support instead of heuristically choosing ω₀
2. Place a proper Gaussian prior on W₂ → closed-form posterior
3. Optimize the ELBO rather than MSE → built-in regularization

### The "Better Constrained" Argument

| Issue | SIREN | VSSGP |
|---|---|---|
| ω₀ tuning | Heuristic, fragile | Marginal likelihood gradient |
| Weight prior | Implicit (optimizer trajectory) | Explicit N(0, σ²I) |
| Function space | NTK (implicit) | Named RKHS |
| Uncertainty | None | Posterior q(w) |

**SIREN is VSSGP with a degenerate (improper flat) weight prior and frequencies fixed by initialization rather than optimized/marginalized.**

### The Four-Rung Hierarchy

All four methods share the same forward model `f(x) = φ(x)ᵀ w` with `φ` built from `Ω`. They differ only in **what is treated as fixed, point-estimated, or distributional**:

| Method | `Ω` (frequencies) | `w` (weights) | Objective | Cost per train step |
|---|---|---|---|---|
| RFF (Rahimi-Recht) | sampled once from `p(ω)`, **fixed** | ridge regression closed-form | `‖y − Φw‖² + λ‖w‖²` | `O(N M² + M³)` once |
| SSGP (Lázaro-Gredilla 2010) | **point-optimized** via marginal likelihood | closed-form posterior `N(μ, Σ)` | `log p(y \| Ω, θ)` | `O(N M² + M³)` per gradient step |
| VSSGP (Gal & Turner 2015) | variational `q(Ω) = N(μ_Ω, σ_Ω²)` with prior `p(ω)` | variational `q(w)` (or marginalized) | ELBO | `O(N M² + M³)` per ELBO step |
| SIREN (Sitzmann 2020) | **fixed by init**, scaled by ω₀ | point-optimized via SGD | `‖y − Φw‖²` (no prior) | `O(N M)` per SGD step (depth-multiplied) |

The progression is **decreasing assumptions, increasing cost** — but for sparse geoscience data, the cost is dominated by `M`, not by which rung you're on, so going to the top rung is essentially free relative to the compute you've already committed.

### What SIREN's Init Heuristic Actually Sets

In a 1-layer SIREN, `W₁ ~ U(−√(6/n), √(6/n))` and the activation is `sin(ω₀ W₁ x)`. The *effective* spectral measure is

```
ω_effective = ω₀ · W₁  ⇒  p(ω) = U[ −ω₀√(6/n), +ω₀√(6/n) ]ᵈ
```

— i.e. a `d`-dimensional uniform box. The Fourier dual of a box is a sinc, so SIREN's implicit prior kernel is a **product of sinc functions**. This kernel has the worst-of-both-worlds spectral signature: it puts non-trivial mass on *every* frequency up to ω₀ (no preference for known scales) and **zero** mass beyond ω₀ (cannot represent anything finer-scale than the chosen cutoff). For a geoscience signal where the energy lies in a known narrow band, this is exactly inverted from what you want.

---

## 2. Why Geoscience Data is the Wrong Domain for SIREN

### Spectral Budget — How Many of Your `M` Features Land Where the Signal Is?

Suppose the signal's true spectral support is a set `B ⊂ ℝᵈ` (e.g. for SSH: a band around `|ω| ∈ [2π/300km, 2π/50km]`, plus a delta at the annual frequency). Define the **useful fraction**

```
η(p) = ∫_B p(ω) dω
```

i.e. the probability that a single feature lands in the band where signal lives. Then by linearity of expectation, the expected number of useful features out of `M` is `M_useful = M · η(p)`.

For SIREN's box prior in `d = 3` (lon, lat, t), with ω₀ chosen large enough to cover annual + mesoscale (`ω_max ≈ 2π/50km`), the useful fraction is roughly

```
η_SIREN ≈ vol(B) / vol([−ω_max, ω_max]³) ≈ 10⁻³–10⁻²
```

For a spectral-mixture prior centered on the band, `η_VSSGP → 1` by construction. The implication: **with `M = 1024` features, SIREN gets `~10` features in the right place; VSSGP gets `~1024`**. To match VSSGP's effective resolution, SIREN needs `M_SIREN ≈ M_VSSGP / η`, i.e. **two orders of magnitude wider** at this `d` and band geometry. Capacity isn't free — wider networks cost more flops and overfit harder on sparse data.

This is the cleanest engineering argument: the cost of an uninformative prior is paid in spent feature budget, and that budget isn't recoverable by training longer.

### The Spectral Structure is Known A Priori

| Field | Known Spectral Features |
|---|---|
| Ocean SSH | Mesoscale eddy band ~100–300 km, internal tides, inertial oscillations |
| Atmosphere | Diurnal/semidiurnal harmonics, synoptic scales, planetary waves |
| Land surface temp | Annual + semiannual cycles, diurnal, known spatial correlation lengths |
| Soil moisture | Seasonal + storm-scale, exponential spatial decay |
| Methane | Seasonal biogenic signal, specific plume spatial scales |

SIREN's ω₀ is a wild guess at this. VSSGP lets you encode it directly:

```
# Informative spectral prior for ocean SSH
ω ~ MixtureOfGaussians(
    μ₁ = 2π / 200km,   # mesoscale eddy peak
    μ₂ = 2π / 500km,   # large-scale gyre
)

# vs SIREN's implicit prior:
ω ~ Uniform[-ω₀√(6/n), ω₀√(6/n)]   # knows nothing
```

### Geoscience Data is Sparse and Irregular

SIREN was designed for dense, regular signals (images, SDFs). Geoscience data is almost never this:
- **Ocean**: Argo floats are random, altimetry has track geometry
- **Atmosphere**: Radiosonde networks are land-biased, satellite swaths have gaps
- **Land surface**: Stations clustered around populated areas

SIREN has no principled way to handle this. VSSGP gives:
```
posterior predictive:  p(f* | X*, X, y)  — proper uncertainty in gaps
```

### Physical Constraints Map to the GP Framework Naturally

- **Smoothness** → Matérn kernel order ν
- **Divergence-free flows** → Helmholtz GP decomposition
- **Periodic boundaries** → Periodic kernels with known period
- **Non-stationarity** → Spatially varying lengthscale

### SIREN Convergence is Structurally Fragile

The initialization solves two coupled problems with a single heuristic (ω₀):
1. Keep activation distributions stable through layers
2. Cover the target frequency range

In VSSGP this doesn't exist as a problem — spectral points are initialized from your physical prior and optimized via marginal likelihood.

---

## 3. Physical Priors for Ocean Variables

### 3.0 Operational SSH mapping schemes ARE fitted spectral priors

Three operational schemes — DUACS, MIOST/MASSH, and DYMOST — have spent the last 30 years fitting prior covariances to global altimetry. Their tuned hyperparameters drop into VSSGP with almost no translation work. Treating them as "the spectral prior we'd otherwise have to learn from scratch" is the single biggest free win available for SSH.

#### 3.0.1 MIOST → spectral-mixture prior with Gabor components

MIOST (Multi-scale Inversion of Ocean Surface Topography, Ardhuin et al. 2021; Le Guillou et al. 2021) decomposes SSH into a sum of **Gabor wavelets** — plane waves modulated by Gaspari-Cohn windows — at three to four scales:

| MIOST component | Spatial scale `L_s` | Temporal scale `τ` | Variance share |
|---|---|---|---|
| Large-scale (gyre) | 800–1500 km | 60–120 days | ~30–40% |
| Mesoscale | 150–300 km | 20–40 days | ~40–50% |
| Sub-mesoscale (SWOT-era) | 30–80 km | 3–10 days | ~10–20% |
| Equatorial / waveguide | 200–500 km × 50–100 km (anisotropic) | 30–60 days | regional only |

A Gabor wavelet `g(x) = exp(i ω₀ᵀx) · w(x − x₀)` with window `w` of width `L_s` has Fourier transform `ĝ(ω) = ŵ(ω − ω₀)` — i.e. a **Gaussian-like bump centred on `ω₀ = 2π/L_s` with bandwidth `~1/L_s`**. So MIOST's wavelet dictionary is, in spectral terms, a sum of Gaussians:

```
p_MIOST(ω) = Σ_n (variance_n / total_var) · N(ω; 2π/L_s,n, σ_n²)

with σ_n ≈ (2π/L_s,n) · 0.3   # bandwidth ~30% of centre frequency
```

This is *literally* a spectral mixture prior à la Wilson & Adams (2013), with the centres and weights pre-fitted by the altimetry community.

*Engineering consequence.* In VSSGP you don't need to discover that the energy lives in a gyre + mesoscale + sub-mesoscale band — you initialize `μ_Ω` at the MIOST centres, allocate `M_n` features per component proportionally to the variance share, and let the marginal-likelihood gradient *refine* the centres locally instead of *finding* them globally. This converts a non-convex spectral search into local refinement.

#### 3.0.2 DYMOST → anisotropic, flow-aligned spectral measure

DYMOST (Ubelmann et al. 2015, 2020) advects the SSH covariance along a Lagrangian first-guess velocity field `u_g`, derived from a 1.5-layer QG model run on the prior mean. The effective covariance has two properties that a stationary `p(ω)` cannot capture:

1. **Anisotropy:** along-stream lengthscale `L_∥ ≈ 2–4 × L_⊥` in strong currents (Gulf Stream, Kuroshio, ACC).
2. **Spatial dependence:** `L_∥(x), L_⊥(x)` track local eddy kinetic energy.

The spectral dual of along-stream stretching is **wavenumber compression in the perpendicular direction**:

```
Σ_ω(x) = R(θ_flow(x))ᵀ · diag( (2π/L_⊥)², (2π/L_∥)² ) · R(θ_flow(x))

# wavenumber covariance is squashed perpendicular to the flow direction
```

In a VSSGP this becomes a **location-conditioned spectral covariance**: instead of a single global `Σ_ω`, sample `ωᵢ` from `N(0, Σ_ω(x_patch))` per patch, where `θ_flow` and `L_⊥/L_∥` come from a low-pass-filtered first-guess (or a climatology like AVISO MDT).

*Engineering consequence.* This pairs naturally with the patch-decomposition recipe in [03_global_scaling_patches](../notebooks/03_global_scaling_patches.md). Each patch gets its own `Σ_ω(x_patch)` from the DYMOST first-guess; the global VSSGP becomes a mixture of patch-local anisotropic VSSGPs, stitched with Gaspari-Cohn weights.

#### 3.0.3 DUACS → empirical variance and lengthscale fields

DUACS (the operational AVISO/CMEMS product, Pujol et al. 2016; Taburet et al. 2019) fits **per-pixel** variance `σ²(x)` and isotropic lengthscale `L(x)` from a 30-year SLA reanalysis. These are public data products — `cmems_obs-sl_glo_phy-mdt_my_allsat-l4-duacs_P1Y` and the auxiliary lengthscale grids.

Two direct uses in VSSGP:

| DUACS field | VSSGP role | Where it goes |
|---|---|---|
| `σ²(x)` (SLA variance) | Spatially varying weight-prior amplitude | `σ_w(x) = √σ²(x)` per patch |
| `L(x)` (correlation lengthscale) | Spatial scaling of `Σ_ω` | `Σ_ω(x) ∝ (2π/L(x))² I` |
| `θ_flow(x), L_∥/L_⊥(x)` (DYMOST aux) | Anisotropy of `Σ_ω` | rotation + axis-ratio in §3.0.2 |

*Engineering consequence.* Three operationally-tuned fields — variance, isotropic lengthscale, anisotropy — give you a fully location-conditioned VSSGP prior with **no free hyperparameters left to tune**. The only thing the marginal-likelihood gradient still needs to do is local refinement around the operational defaults; everything global has been done by 30 years of community fitting.

#### 3.0.4 Practical integration recipe

The handoff is short:

```python
# 1. Load operational priors (one-time, per region)
σ_field   = load_duacs_variance(region)         # (H, W)
L_field   = load_duacs_lengthscale(region)      # (H, W)
θ_field   = load_dymost_flow_angle(region)      # (H, W)
ratio     = load_dymost_axis_ratio(region)      # (H, W) ; L_par / L_perp

# 2. Per patch (see 03_global_scaling_patches), build patch-local prior
def patch_prior(x_patch):
    σ_patch  = σ_field.interp(x_patch)
    L_patch  = L_field.interp(x_patch)
    θ_patch  = θ_field.interp(x_patch)
    r_patch  = ratio.interp(x_patch)

    # MIOST-style mixture, scaled to local lengthscale
    centres = jnp.array([
        2*jnp.pi / (L_patch * 5.0),    # gyre   (5x lengthscale)
        2*jnp.pi / (L_patch * 1.0),    # meso   (1x lengthscale)
        2*jnp.pi / (L_patch * 0.25),   # sub    (0.25x lengthscale)
    ])
    weights = jnp.array([0.35, 0.45, 0.20])  # variance shares

    # DYMOST-style anisotropy
    R = rotation(θ_patch)
    Σ_ω = R.T @ jnp.diag([1.0, r_patch**2]) @ R

    return centres, weights, Σ_ω, σ_patch

# 3. Initialize VSSGP variational params μ_Ω, σ_Ω from this prior
#    Train (refines locally; doesn't search globally)
```

The bookkeeping is the same as a generic VSSGP — just the *initialization and prior* change. No new code paths.

#### 3.0.5 What the operational schemes don't give you (and VSSGP does)

- **Posterior uncertainty.** DUACS reports a formal mapping error from the OI cost, but it's a single scalar per pixel, not a draw-able posterior over `f`. VSSGP gives `q(w) → q(f*)`, full samples included.
- **Online refinement.** MIOST hyperparameters are fitted once per release cycle. VSSGP's marginal-likelihood gradient updates them per-patch, per-batch — useful when SWOT keeps shifting the sub-mesoscale variance budget downward.
- **Joint inference with non-Gaussian likelihoods.** DUACS assumes Gaussian along-track noise. VSSGP slots into the same machinery as the Student-T/log-normal likelihoods you'd want for SST/Chl-a, so you can do joint multivariate SSH+SST+Chl-a inference with a coupled `p(ω)` and per-channel likelihood.

The summary: **MIOST and DYMOST tell you where to put the spectral mass; DUACS tells you how it varies in space; VSSGP tells you the posterior**. The three together are strictly more than any one alone.

### SSH (Sea Surface Height)

```
# Spatial: power-law spectral density
p(ω) ∝ |ω|^{-α},   α ≈ 4–5   (mesoscale range)

# Mixture:
p(ω) = w₁ · N(2π/L_gyre, σ₁)      # L_gyre ~ 1000 km
      + w₂ · N(2π/L_meso, σ₂)      # L_meso ~ 100–300 km
      + w₃ · N(2π/L_tide, σ₃)      # L_tide ~ 100 km

# Temporal:
p(ω_t) = w₁ · δ(2π/365.25)         # annual
        + w₂ · δ(2π/182.6)          # semiannual
        + w₃ · N(0, σ_meso)         # mesoscale (weeks-months)

# Weight prior (anomaly around mean dynamic topography)
w ~ N(0, σ_w²),    σ_w ~ 0.1–0.3 m

# Likelihood
y | f ~ N(f(x), σ_n²),    σ_n ~ 0.02–0.05 m
# Note: along-track altimetry has correlated noise → correlated noise kernel
```

### SST (Sea Surface Temperature)

```
# Temporal — seasonal dominates
p(ω_t) = w₁ · N(2π/365.25, σ₁)    # annual (dominant)
        + w₂ · N(2π/182.6, σ₂)     # semiannual
        + w₃ · N(2π/1, σ₃)         # diurnal (~0.5–1°C)
        + w₄ · N(0, σ_slow)         # interannual (ENSO)

# Weight prior (as anomaly from climatology)
w ~ N(0, σ_w²),    σ_w ~ 2–5°C mid-latitude, ~1–2°C tropics

# Likelihood — Student-T for cloud-contaminated IR retrievals
y | f ~ StudentT(f(x), σ², ν),    ν ~ 4–7
```

### SSS (Sea Surface Salinity)

```
# Temporal
p(ω_t) = w₁ · N(2π/365.25, σ₁)    # seasonal (precipitation cycle)
        + w₂ · N(0, σ_slow)         # interannual

# Weight prior — non-stationary near rivers
w ~ N(0, σ_w²),    σ_w ~ 0.5–1.5 PSU open ocean
                        ~ 3–5 PSU near river mouths

# Likelihood — SMOS/SMAP have RFI contamination
y | f ~ StudentT(f(x), σ², ν),    ν ~ 3–5
σ_n ~ 0.2–0.5 PSU per retrieval
```

### OC / Chlorophyll-a

**Key prior: work in log space.** Chl-a is log-normally distributed across the full range (0.01 to >100 mg/m³).

```
# Model log(Chl-a), not Chl-a directly
g(x) = log(Chl-a(x))
g ~ VSSGP(0, k)

# Temporal
p(ω_t) = w₁ · N(2π/365.25, σ₁)    # spring bloom — very strong
        + w₂ · N(2π/182.6, σ₂)     # secondary bloom
        + w₃ · N(0, σ_slow)         # interannual

# Weight prior (in log space)
w ~ N(0, σ_w²),    σ_w ~ 1.0–1.5 log₁₀ units

# Likelihood — log-normal is canonical
log(y) | f ~ N(f(x), σ_n²),    σ_n ~ 0.1–0.3 log₁₀ units
```

### Summary Table

| Variable | Transform | Spectral Prior | Weight σ | Likelihood |
|---|---|---|---|---|
| SSH | none (anomaly) | Power-law + mixture at eddy/tide scales | 0.1–0.3 m | Gaussian (or correlated along-track) |
| SST | none (anomaly) | Strong annual + diurnal harmonics | 2–5°C | Student-T (ν~5) |
| SSS | none (anomaly) | Seasonal + non-stationary σ near rivers | 0.5–5 PSU | Student-T (ν~3–4) |
| OC / Chl-a | log | Annual bloom + patchy spatial | 1.0–1.5 log units | Log-normal |

---

## 4. How Additive Components Are Injected into VSSGP

### The Core SSGP Equation

```
k(x, x') ≈ φ(x)ᵀ φ(x')

φ(x) = √(2/M) cos(Ω x + b)

where:
  x  : (D,)       input coordinates
  Ω  : (M, D)     spectral points, each row ωᵢ ~ p(ω)
  b  : (M,)       phases, bᵢ ~ Uniform(0, 2π)
  φ  : (M,)       feature vector for one input

Batched over N inputs:
  X  : (N, D)
  φ(X) = √(2/M) cos(X @ Ωᵀ + b)   : (N, M)
```

Model is Bayesian linear regression in feature space:
```
w   : (M,)        w ~ N(0, σ_w² I)
f   : (N,)        f = φ(X) @ w   =  (N, M) @ (M,) → (N,)
y   : (N,)        y ~ N(f, σ_n² I)
```

### Additive Kernel = Concatenated Feature Maps

```
φ(X) = [ φ₁(X) | φ₂(X) | φ₃(X) ]     # (N, M1+M2+M3)
w    = [ w₁    | w₂    | w₃    ]       # (M1+M2+M3,)

f = φ(X) @ w
  = φ₁(X) @ w₁  +  φ₂(X) @ w₂  +  φ₃(X) @ w₃   # (N,) each
```

### Full SSH Example with D=3 inputs (lon, lat, t)

```python
import jax.numpy as jnp
import jax.random as jr

# hyperparameters (physical prior knowledge)
L_gyre   = 1000.0   # km
L_meso   = 200.0    # km
T_annual = 365.25   # days
T_meso   = 60.0     # days
σ_slow   = 1/500.0  # cycles/day

M1, M2, M3 = 32, 64, 128   # slow | annual×spatial | mesoscale
D = 3                        # (lon, lat, t)


# ================================================================
# Step 1: Sample spectral points Ωⱼ for each component
# ================================================================

def sample_spectral_points(key):

    k1, k2, k3, k4, k5 = jr.split(key, 5)

    # Component 1: slow temporal trend
    # prior: ω_t ~ N(0, σ_slow²),  ω_lon = ω_lat = 0
    ω_t_slow = jr.normal(k1, (M1, 1)) * σ_slow    # (M1, 1)
    ω_zeros  = jnp.zeros((M1, 2))                  # (M1, 2)
    Ω1       = jnp.concatenate(                     # (M1, 3)
                   [ω_zeros, ω_t_slow], axis=-1)
    # Ω1[i] = (0, 0, ω_t_i)  → pure temporal features

    # Component 2: annual cycle × large-scale spatial
    # prior: ω_t ~ N(2π/T_annual, σ_annual²)   ← centred on annual freq
    #        ω_lon, ω_lat ~ N(0, (2π/L_gyre)²)
    σ_annual = 2*jnp.pi / T_annual * 0.1           # 10% bandwidth
    σ_gyre   = 2*jnp.pi / L_gyre

    ω_t_ann  = jr.normal(k2, (M2, 1)) * σ_annual \
               + 2*jnp.pi/T_annual                  # (M2, 1)
    ω_s_ann  = jr.normal(k3, (M2, 2)) * σ_gyre     # (M2, 2)
    Ω2       = jnp.concatenate(                     # (M2, 3)
                   [ω_s_ann, ω_t_ann], axis=-1)
    # Ω2[i] = (ω_lon_i, ω_lat_i, ω_t_i) with ω_t near annual freq

    # Component 3: mesoscale spatiotemporal
    # prior: ω ~ N(0, diag(σ_meso_s², σ_meso_s², σ_meso_t²))
    σ_meso_s = 2*jnp.pi / L_meso                   # mesoscale wavenumber
    σ_meso_t = 2*jnp.pi / T_meso                   # mesoscale frequency
    scales   = jnp.array([σ_meso_s, σ_meso_s, σ_meso_t])  # (3,)
    Ω3       = jr.normal(k4, (M3, D)) * scales      # (M3, 3)
    # Ω3[i] = (ω_lon_i, ω_lat_i, ω_t_i) fully coupled

    return Ω1, Ω2, Ω3
    # shapes: (M1,3), (M2,3), (M3,3)


# ================================================================
# Step 2: Build feature maps for each component
# ================================================================

def make_features(X, Ω1, Ω2, Ω3):
    """
    X    : (N, 3)   input coords (lon, lat, t)
    Ωj   : (Mj, 3)  spectral points for component j
    """
    key_b = jr.PRNGKey(42)
    kb1, kb2, kb3 = jr.split(key_b, 3)

    b1 = jr.uniform(kb1, (M1,), minval=0, maxval=2*jnp.pi)  # (M1,)
    b2 = jr.uniform(kb2, (M2,), minval=0, maxval=2*jnp.pi)  # (M2,)
    b3 = jr.uniform(kb3, (M3,), minval=0, maxval=2*jnp.pi)  # (M3,)

    # X @ Ωⱼᵀ : (N,3) @ (3,Mj) → (N, Mj)
    φ1 = jnp.sqrt(2/M1) * jnp.cos(X @ Ω1.T + b1)   # (N, M1)
    φ2 = jnp.sqrt(2/M2) * jnp.cos(X @ Ω2.T + b2)   # (N, M2)
    φ3 = jnp.sqrt(2/M3) * jnp.cos(X @ Ω3.T + b3)   # (N, M3)

    φ  = jnp.concatenate([φ1, φ2, φ3], axis=-1)     # (N, M1+M2+M3)
    return φ, (φ1, φ2, φ3)


# ================================================================
# Step 3: Bayesian linear model (the SSGP model)
# ================================================================

def ssgp_model(X, y, Ω1, Ω2, Ω3):
    """
    X  : (N, 3)
    y  : (N,)
    """
    M = M1 + M2 + M3   # 224

    φ, _ = make_features(X, Ω1, Ω2, Ω3)   # (N, M)

    # weight prior — amplitude scale per component
    σ_w = jnp.concatenate([
        jnp.full(M1, 0.20),    # (M1,)  slow trend
        jnp.full(M2, 0.15),    # (M2,)  annual cycle
        jnp.full(M3, 0.10),    # (M3,)  mesoscale
    ])                          # (M,)

    w   = numpyro.sample("w", dist.Normal(jnp.zeros(M), σ_w))   # (M,)
    f   = φ @ w                                                   # (N,)

    σ_n = numpyro.sample("σ_n", dist.HalfNormal(0.03))
    numpyro.sample("y", dist.Normal(f, σ_n), obs=y)


# ================================================================
# VSSGP extension: spectral points become variational parameters
# ================================================================

# Replace fixed Ω3 with variational distribution q(ω) = N(μ_ω, diag(σ_ω²))
# μ_Ω3 = numpyro.param("μ_Ω3", Ω3_init)           # (M3, 3)
# σ_Ω3 = numpyro.param("σ_Ω3", 0.1*jnp.ones(...), constraint=positive)
# Ω3   = numpyro.sample("Ω3", dist.Normal(μ_Ω3, σ_Ω3))  # (M3, 3)
# → spectral points move to explain data, regularized toward physical prior
```

### Shape Flow Summary

```
X       : (N, 3)          ← lon, lat, t for N observations

Ω1      : (M1, 3)         ← spectral pts, slow component
Ω2      : (M2, 3)         ← spectral pts, annual component
Ω3      : (M3, 3)         ← spectral pts, mesoscale component

φ1      : (N, M1)         ← features, slow
φ2      : (N, M2)         ← features, annual
φ3      : (N, M3)         ← features, mesoscale

φ       : (N, M1+M2+M3)   ← concatenated features

w       : (M1+M2+M3,)     ← weights,  w ~ N(0, diag(σ_w²))

f       : (N,)            ← f = φ @ w
y       : (N,)            ← y ~ N(f, σ_n²)
```

**Physical knowledge lives entirely in how Ωⱼ are constructed — their initialization, prior distribution, and (in VSSGP) constraints on how far they move. Everything downstream is linear algebra.**

---

## 5. Approximation Error and the Marginal-Likelihood Compass

Two more results worth having in your back pocket — both engineering-flavored, both useful for sizing experiments.

### 5.1 Mercer truncation: how much of the kernel are you keeping?

The kernel admits a Mercer expansion `k(x, x') = Σᵢ λᵢ ψᵢ(x) ψᵢ(x')` with `λ₁ ≥ λ₂ ≥ …`. Truncating to the top `M` eigenfunctions gives an approximation error

```
‖k − k_M‖² = Σ_{i > M} λᵢ²
```

For Bochner-type kernels the spectrum's tail is set by the *smoothness* of the field (Matérn-ν tails like `(1 + |ω|²ℓ²)^{−(ν + d/2)}`). A field with high `ν` (e.g. the mean SSH after removing eddies) needs **far fewer** features for a target tolerance than a rough field (e.g. instantaneous SSH including mesoscale).

*Engineering consequence.* Pick `M` from the kernel tail you've targeted, not from a rule of thumb. For SSH-mesoscale (`ν ≈ 3/2`, `ℓ ≈ 100 km`, `d = 3`) on a `1000 × 1000 × 30` grid, `M ≈ 1024` captures ≥ 99% of `tr(K)`. Doubling `ν` (smoother field) drops the same coverage to `M ≈ 256`.

### 5.2 Marginal likelihood gradient as a spectral compass

For SSGP, differentiating the log-marginal-likelihood w.r.t. a single spectral point `ωᵢ`:

```
∂/∂ωᵢ log p(y | Ω) = (yᵀ A⁻¹ ∂Φ/∂ωᵢ wᵢ_eff) − tr(A⁻¹ ∂Φ/∂ωᵢ Φᵀ)
```

where `A = σ_f² Φ Φᵀ + σ_n² I` and `wᵢ_eff` is the contribution of feature `i` to the posterior. The first term **rewards moving `ωᵢ` toward frequencies where the residual `y − Φ μ_post` has spectral mass**; the second penalizes redundancy with already-placed features.

*Engineering consequence.* You don't need to know the right spectral prior in advance — running a few epochs of SSGP/VSSGP starting from a generic prior reveals it. Inspect the histogram of optimized `Ω` after training and you have an empirical estimate of the true spectral density. This is impossible to extract from a trained SIREN.

### 5.3 When to *not* trust this argument

Three failure modes of the "VSSGP > SIREN for geoscience" thesis worth flagging honestly:

| Failure mode | What goes wrong | Fix |
|---|---|---|
| **Strongly non-stationary fields** (e.g. coastal SSH, frontal SST) | A single global `p(ω)` is the wrong object — local lengthscale varies | Deep GP / mixture-of-experts SSGP / patchwise approach (see [03_global_scaling_patches](../notebooks/03_global_scaling_patches.md)) |
| **Highly nonlinear forward models** (e.g. radiative transfer, log-Chl-a beyond linear regime) | RFE/SSGP is linear-in-features; nonlinearity needs depth | Deep RFE (Cutajar 2017) — compose RFF layers as a deep GP |
| **Massive `M` regimes** (`M > 10⁴`) | `O(M³)` posterior covariance dominates | Switch to inducing points / SVGP, or matrix-free CG (see [01_efficient_machinery](../notebooks/01_efficient_machinery.md)) |

In the first two cases SIREN's *flexibility* — fully learned features, deep nonlinearity — is genuinely useful. The argument here is not "VSSGP wins everywhere" but "for the dominant geoscience regime (sparse, irregular, banded spectrum, mostly stationary within a region) the structural advantages compound".

---

## 6. Experiment Planning — SSH, SST, SSS, OC

This section turns the theoretical argument into a concrete benchmark. Four ocean variables, four regional zoom levels, four classes of metrics, four-rung method ladder. Every combination should be a directly runnable experiment in `gaussx`/`pyrox`.

### 6.1 What we're trying to demonstrate

| Claim | Test |
|---|---|
| **C1.** Informative `p(ω)` outperforms uninformative `p(ω)` at fixed `M` | RFF (uniform) vs SSGP-with-MIOST-init vs SIREN, all at `M = 1024` |
| **C2.** Posterior > point estimate for sparse data | SSGP point predictions vs VSSGP credible intervals — measure CRPS gap |
| **C3.** Operational priors transfer | Init VSSGP from MIOST/DUACS in Region A, evaluate on Region B without re-fitting hyperparameters |
| **C4.** The hierarchy ranks by sample efficiency | Train all four methods on `N/k` for `k ∈ {1, 2, 4, 8}`; plot RMSE vs `N` — gap should widen as `N` shrinks |
| **C5.** Spectral structure is preserved, not just RMSE | Compute power-spectral coherence — VSSGP should match the truth across the full mesoscale band; SIREN drops off |
| **C6.** Derived physical quantities are improved | Geostrophic currents from `∇η̂` should match drifters better when prior is informative |

### 6.2 Datasets and regions

Four nested zoom levels — same hierarchy of regions across all four variables, so comparisons stay on-axis:

| Zoom | Region | Lon × Lat | Why |
|---|---|---|---|
| **Z1 — small** | Mediterranean | `[-6, 36] × [30, 46]` | Eddy-rich, well-instrumented, manageable `N` |
| **Z2 — medium** | Gulf Stream extension | `[-80, -40] × [25, 50]` | Strong anisotropy → tests DYMOST-style flow-aligned prior |
| **Z3 — large** | North Atlantic | `[-80, 0] × [10, 60]` | Cross-regime: subtropical gyre + Gulf Stream + sub-polar |
| **Z4 — global** | Global ocean | `[-180, 180] × [-80, 80]` | Tests patch-decomposition stitching from [03_global_scaling_patches](../notebooks/03_global_scaling_patches.md) |

Per-variable data products (CMEMS / NASA / ESA, all freely accessible):

| Variable | Sparse obs (training) | Gridded reference (eval) | In-situ (independent eval) |
|---|---|---|---|
| **SSH** | `SEALEVEL_GLO_PHY_L3_MY_008_062` (nadir altimetry, ~3×10⁵ obs/day) + `_069` (SWOT KaRIn, ~1.7×10⁶ obs/day) | DUACS L4 `SEALEVEL_GLO_PHY_L4_MY_008_047` | tide gauges `INSITU_GLO_PHY_SSH_DISCRETE_NRT_013_059`, GDP drifters (for `u_g`) |
| **SST** | GHRSST L2P MODIS-Aqua/Terra, AVHRR-19/MetOp | OSTIA L4 `SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001` | Argo floats, drifting buoys (NDBC) |
| **SSS** | SMOS L2 v700, SMAP RSS V5 (~6×10⁴ obs/day each) | ESA SMOS L4 BEC, Multi-Mission Optimal Interpolated SSS | Argo (calibrated), TSG underway |
| **OC / Chl-a** | OC-CCI L3 v6.0 daily (sun-synch + cloud gaps) | OC-CCI L4 monthly composite | BGC-Argo Chl fluorometers, in-situ HPLC (NASA SeaBASS) |

### 6.3 Experimental protocols

Three protocols, each isolating a different failure mode:

#### 6.3.1 OSSE (controlled — ground truth available)

Use a high-resolution model run as truth; sample synthetic observations at real altimeter / SMAP / drifter geometries.

- **Truth:** GLORYS12 reanalysis (1/12°, daily) for SSH+SST+SSS; CMEMS biogeochemical reanalysis for Chl-a
- **Obs operator:** along-track sampling at real CMEMS L3 geometries; add Gaussian (SSH) or correlated (SST cloud-gap) noise
- **Eval:** full grid available — RMSE, spectral coherence, PIT all computed pixelwise

This is the *gold standard* for the spectral-coherence and effective-resolution metrics — they require dense ground truth.

#### 6.3.2 Leave-one-track-out (real data — independent altimeter)

For SSH only, the SSH community standard (Ballarotta et al. 2019). Train on `{SARAL, Sentinel-3A/B, Cryosat-2, Jason-3}` minus one; predict the held-out altimeter's tracks; evaluate at observation points.

- **Held-out altimeter rotates** across runs to control for orbit-geometry confounds
- Evaluates *real-data* skill without needing model truth

#### 6.3.3 In-situ withholding (real data — independent platform)

Train on satellite obs only; evaluate on tide gauges (SSH) / Argo (SST, SSS) / BGC-Argo (Chl-a). Independent measurement system, so satellite biases don't leak.

| Protocol | Pros | Cons | Use for |
|---|---|---|---|
| OSSE | Dense truth, controlled noise | Model-truth mismatch | Spectral metrics, calibration |
| Leave-one-track | Real data, real noise | Same instrument family | Effective resolution |
| In-situ | Truly independent | Sparse, biased to coasts | Bias detection, regional skill |

### 6.4 Metrics — four classes

The interpolation community routinely under-reports metrics; a single RMSE hides the differences between methods that matter most. Four families, each catching a different failure mode:

#### 6.4.1 Point accuracy

Standard but necessary:

```
RMSE  = sqrt( mean( (ŷ − y_true)² ) )
MAE   = mean( |ŷ − y_true| )
bias  = mean( ŷ − y_true )
nRMSE = RMSE / std(y_true)               # cross-region comparable
R²    = 1 − var(ŷ − y_true) / var(y_true)
```

Per-variable target (OSSE on Z2 Gulf Stream extension, 1 month, full SWOT-era):

| Variable | DUACS-class baseline RMSE | VSSGP target | Threshold for "informative prior helps" |
|---|---|---|---|
| SSH | 4.5 cm | < 3.5 cm | nRMSE drop ≥ 15% |
| SST | 0.45 K | < 0.30 K | nRMSE drop ≥ 25% |
| SSS | 0.25 PSU | < 0.20 PSU | nRMSE drop ≥ 15% |
| log Chl-a | 0.30 log₁₀ | < 0.22 log₁₀ | nRMSE drop ≥ 20% |

#### 6.4.2 Probabilistic / calibration

Point methods (RFF-ridge, SIREN) cannot compute these — that's the point.

```
NLPD  = − mean( log p(y_true | x*) )
      # for Gaussian: ½ log(2π σ²) + ½ (ŷ − y)² / σ²

CRPS  = mean( ∫ ( F(z; x*) − 𝟙{z ≥ y_true} )² dz )
      # for Gaussian-posterior: closed-form CRPS

PIT   = F(y_true ; x*) ∈ [0, 1]
      # histogram should be Uniform[0, 1] under correct calibration

Reliability:
   bin predictions by quantile q; check empirical coverage = q
   ECE = mean( |q − coverage(q)| )
```

CRPS is the headline probabilistic metric — proper, scale-aware, low-dimensional. NLPD is sensitive to outlier underprediction (heavy tails) so it disambiguates Gaussian vs. Student-T likelihoods.

#### 6.4.3 Spectral metrics

This is where SIREN is *expected* to do badly even at competitive RMSE — it gets the spatial autocorrelation wrong.

```
Power spectral density (PSD):
   PSD_pred(k)  = |FFT(ŷ − ⟨ŷ⟩)|²
   PSD_true(k)  = |FFT(y_true − ⟨y_true⟩)|²
   PSD_error(k) = |FFT(ŷ − y_true)|²

Spectral coherence (Ballarotta 2019, the SSH community standard):
   T(k) = 1 − PSD_error(k) / PSD_true(k)

Effective resolution L_eff:
   smallest scale where T(k) ≥ 0.5
   i.e. signal dominates noise

Spectral RMSE:
   ‖ log PSD_pred − log PSD_true ‖₂   # in log space, by decade
```

Per-variable target on Z2 (Gulf Stream, OSSE):

| Variable | Reference `L_eff` (DUACS-class) | VSSGP target |
|---|---|---|
| SSH | 100–150 km | < 80 km (resolves mesoscale) |
| SST | 60 km | < 30 km (resolves frontal scale) |
| SSS | 200 km | < 150 km |
| Chl-a (log) | 150 km | < 80 km |

#### 6.4.4 Physical / derived-quantity metrics

For SSH especially, the *derived* quantities matter more than `η` itself — operational users want currents, divergence, and vorticity:

```
Geostrophic velocity:
   u_g = − (g/f) ∂η/∂y     v_g = (g/f) ∂η/∂x
   RMSE_u = sqrt( mean( (û − u_drifter)² ) )      # vs GDP drifters

Vorticity:
   ζ = ∂v/∂x − ∂u/∂y
   RMSE_ζ                                          # vs OSSE truth

SST gradient (front detection):
   ∇T magnitude — Sobel
   F1 score for fronts above threshold

Chl-a bloom timing:
   day-of-year of annual maximum (per pixel)
   RMSE in days (cf. Henson et al. 2018)
```

These metrics are the ones DUACS/MIOST publish — they're how the community accepts a new SSH product. The VSSGP claim has to land here, not just on RMSE.

### 6.5 Per-variable experimental protocols

#### 6.5.1 SSH — anchor variable, fullest treatment

```
Method ladder × 4:    {RFF-ridge, SIREN, SSGP, VSSGP-MIOST-init}
Region ladder × 4:    {Med, Gulf Stream, N. Atl, Global}
Time horizon × 3:     {1 month, 3 months, 6 months}
SWOT ablation × 2:    {nadir-only, nadir + SWOT}

⇒ 4 × 4 × 3 × 2 = 96 runs (Z4 × 6mo only for VSSGP, halve to ~80)
```

- Likelihood: Gaussian for nadir, **correlated-noise kernel** for along-track residuals (Le Traon-style)
- Prior: §3.0 MIOST + DYMOST + DUACS recipe
- Metric stack: full §6.4 (point + CRPS + spectral coherence + `u_g` RMSE)
- Baselines: DUACS L4, MIOST, BFN-QG, 4DVarNet (latter two: cite published Ballarotta numbers, don't re-run)

#### 6.5.2 SST — diurnal cycle and fronts

```
Method ladder: same 4
Region: Med + Gulf Stream + Global (skip N. Atl alone — covered)
Likelihood: Student-T with ν ∈ {3, 5, 7} ablation (cloud-contamination tail)
Special diagnostic: gradient magnitude F1 score for fronts
```

- Diurnal-cycle preservation: phase + amplitude error of the 24-h harmonic (one number per pixel)
- Cloud-gap stress test: artificially mask a 200 × 200 km block; reconstruct; measure RMSE in the gap
- Baseline: OSTIA L4

#### 6.5.3 SSS — extreme non-stationarity

```
Region: Amazon plume (3-week experiment), tropical Atlantic, global
Special: σ_w(x) field MUST be spatially varying — open-ocean σ_w ≈ 0.5 PSU,
         plume σ_w ≈ 3-5 PSU; flat σ_w fails by construction
```

- Extreme value capture: skill at 95th and 99th percentile (RMSE conditional on `y > q_95`)
- River plume tracking: cross-correlation of low-salinity tongue position vs MODIS Chl proxy
- Baseline: ESA SMOS L4 BEC

#### 6.5.4 OC / Chl-a — log-space and bloom dynamics

```
Region: N. Atlantic spring bloom (March-May), Arabian Sea (SW monsoon),
        Southern Ocean (austral spring), global
Transform: log₁₀ — the entire model lives in log space
Special: bloom timing metric — most operational mappings smear the bloom edge
```

- Log-space RMSE + log-space CRPS (proper for log-normal posterior)
- Bloom phenology: day-of-year of pixel maximum, RMSE in days
- Cloud-gap stress same as SST
- Baseline: OC-CCI L4

### 6.6 Ablation matrices

Five ablations isolate which design choice is doing the work. Each is one row of the result table; designed to fail differently.

| # | Ablation | What it isolates | Expected outcome |
|---|---|---|---|
| **A1** | VSSGP-uniform-prior vs VSSGP-MIOST-prior | Value of operational prior | A1 narrows nRMSE gap by ~50% (rest of gap is from posterior, not prior) |
| **A2** | `M ∈ {64, 128, 256, 512, 1024, 2048}` | Sample-complexity scaling | RMSE vs `M` should be a clean power-law for VSSGP, ragged for SIREN |
| **A3** | `σ_w` flat vs DUACS-derived `σ_w(x)` | Value of spatially varying amplitude | Big gap on SSS, small on SSH |
| **A4** | Isotropic vs DYMOST-anisotropic `Σ_ω` | Value of flow-aligned prior | Gulf Stream RMSE drop ≥10% on `v_g` |
| **A5** | Train Med, eval N. Atlantic | Prior transfer | VSSGP-MIOST should hold; SSGP-no-prior should fail |

### 6.7 Compute budget

Order-of-magnitude per run on 1× A100 80GB (per the wall-clock numbers in [01_efficient_machinery](../notebooks/01_efficient_machinery.md)):

| Region | `N` (full SWOT-era, 1mo) | `M` | Train/run | Eval/run | Per-method total |
|---|---|---|---|---|---|
| Z1 Med | ~1.5 × 10⁵ | 1024 | 5 min | 1 min | 6 min |
| Z2 Gulf Stream | ~3 × 10⁵ | 1024 | 12 min | 2 min | 14 min |
| Z3 N. Atlantic | ~1.5 × 10⁶ | 2048 | 1.5 hr | 5 min | 1.6 hr |
| Z4 Global (patched) | ~6 × 10⁷ | per-patch 1024 | 2 hr (8× A100, dask) | 10 min | 2.2 hr |

For the full method × region × variable × ablation grid: roughly **300–400 GPU-hours**, dominated by Z3+Z4 + the `M`-sweep ablation. Tractable on a small cluster over a week.

### 6.8 Reporting template

Each variable + region combo produces one results row. Recommended format:

```
| Method        | RMSE  | CRPS  | NLPD  | L_eff | RMSE u_g | ECE |
|---------------|-------|-------|-------|-------|----------|-----|
| RFF-ridge     | 4.2cm |   —   |   —   | 140km |  9.3cm/s |  —  |
| SIREN         | 3.9cm |   —   |   —   | 165km |  9.8cm/s |  —  |
| SSGP          | 3.5cm | 1.8cm | -0.42 | 105km |  7.1cm/s | .12 |
| VSSGP-MIOST   | 3.1cm | 1.5cm | -0.61 |  82km |  6.2cm/s | .04 |
| ----          |       |       |       |       |          |     |
| DUACS L4      | 4.5cm |   —   |   —   | 150km |  8.5cm/s |  —  |
| MIOST         | 3.6cm |   —   |   —   | 110km |  7.4cm/s |  —  |
```

Bolding rule: per metric column, bold the best of {RFF, SIREN, SSGP, VSSGP}; italicize if it also beats *all* operational baselines in that column. The story of the paper is whichever cells consistently land in italic-bold.

---

## Key References

**Foundations**
- Bochner, S. (1933) — Monotone Funktionen, Stieltjessche Integrale und harmonische Analyse
- Rahimi & Recht (2007) — Random Features for Large-Scale Kernel Machines
- Rahimi & Recht (2008) — Uniform Approximation of Functions with Random Bases (sample-complexity bound)

**The hierarchy**
- Lázaro-Gredilla et al. (2010) — Sparse Spectrum Gaussian Process Regression (SSGP)
- Gal & Turner (2015) — Improving the Gaussian Process Sparse Spectrum Approximation (VSSGP)
- Wilson & Adams (2013) — Gaussian Process Kernels for Pattern Discovery and Extrapolation (Spectral Mixture)
- Cutajar et al. (2017) — Random Feature Expansions for Deep Gaussian Processes

**Neural-field side**
- Sitzmann et al. (2020) — Implicit Neural Representations with Periodic Activation Functions (SIREN)
- Tancik et al. (2020) — Fourier Features Let Networks Learn High-Frequency Functions in Low-Dimensional Domains
- Yang & Salman (2019) — A Fine-Grained Spectral Perspective on Neural Networks (NTK ↔ kernel)

**Geoscience priors**
- Chelton et al. (2011) — The Influence of Nonlinear Mesoscale Eddies on Near-Surface Oceanic Chlorophyll
- Le Traon et al. (1990) — A spectral analysis of the Geosat altimeter signal (SSH spectral slopes)
- Stogryn (1985) — Distribution of sea surface temperatures (Student-T tails for IR retrievals)