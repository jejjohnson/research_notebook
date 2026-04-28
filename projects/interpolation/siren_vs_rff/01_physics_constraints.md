---
title: "Enforcing physics in NF / VSSGP training — basis, data, loss"
---

# Enforcing physics in the VSSGP / neural-field training pipeline

Companion to [00_siren_rff_vssgp.md](00_siren_rff_vssgp.md). This note answers the question: *given that we are training a neural field (SIREN) or a VSSGP and then evaluating it — including outside the training footprint — where does physical knowledge enter, and how do those entry points differ between the two methods?*

```{important}
**Scope.** This is a training-paradigm document. It is **not** about Matheron's-rule pathwise sampling on dense grids ([01_efficient_machinery](../notebooks/01_efficient_machinery.md)) or about patch decomposition for global scale ([03_global_scaling_patches](../notebooks/03_global_scaling_patches.md)). Both methods here learn `θ` from data and then evaluate at any prediction point `x*`. The most interesting prediction regime — and the one that tests the value of an informative prior most strongly — is **out-of-domain `x*`** (spatial or temporal extrapolation), which is exactly where the prior, not the data, dominates the answer.
```

The setup, written so it covers both methods:

| | Neural field (SIREN) | VSSGP |
|---|---|---|
| Function form | $f_\theta(x) = \mathrm{NN}_\theta(x)$ | $f(x) = \phi(x)^{\!\top} w$ |
| What is learned | Network weights $\theta$ via SGD on a loss | Weight posterior $p(w \mid y)$ in closed form (linear case) or via SVI |
| Prediction at $x^*$ | $f_\theta(x^*)$ — point estimate | $\mathbb{E}[f(x^*)] = \phi(x^*)^{\!\top} \mu_{\text{post}}$ + closed-form variance |
| Out-of-domain behaviour | Whatever the network extrapolates to (often poorly behaved) | Reverts smoothly to prior — variance grows, mean → 0; spectral structure of $p(\omega)$ remains |

Both share three axes where physics can enter the pipeline:

1. **Basis** — the function class (RFF spectral prior; SIREN architecture).
2. **Data** — input preprocessing, input embeddings, output preprocessing, output parameterisation.
3. **Loss** — data fit, regularisers, PDE-residual / soft-physics terms.

These are nearly independent. You can intervene on any subset; effects on training and on out-of-domain behaviour are roughly additive. The rest of the note is one section per axis.

---

## 0. Method neutrality — what carries across, what does not

Before the axis-by-axis treatment: the two methods agree on *what physics looks like at each axis*, but disagree on *what's free vs. expensive at each axis*.

| Move | NF (SIREN) | VSSGP |
|---|---|---|
| Spectral prior on energy distribution | ❌ no native — heuristic ω₀ at first layer is a degenerate proxy | ✅ explicit `p(ω)` (the headline of 00_siren_rff_vssgp.md) |
| Anisotropic / flow-aligned structure | ⚠️ approximate via input coordinate transform | ✅ Matérn-SPDE prior — exact |
| Periodic features for known harmonics | ✅ stack into input layer (positional encoding) | ✅ stack into basis $\phi$ |
| Predict scalar potential, derive constrained field via autodiff | ✅ `jax.grad(NN_θ)` | ✅ analytic differentiation of $\phi$ |
| Output transform (log, link function) | ✅ trivial | ✅ trivial — but breaks closed-form for non-Gaussian |
| Data-fit likelihood | Gaussian by default; Student-T / log-normal via custom loss | Gaussian closed-form; non-Gaussian via SVI / Laplace |
| PDE-residual / virtual-obs loss | ✅ standard PINN — joint SGD | ✅ closed-form for *linear* PDEs; SVI/Laplace otherwise |
| Learned operator inside the loss | ✅ native — NN-in-loss is the norm | ⚠️ awkward — breaks closed-form Bayes |
| Posterior over predictions (incl. OOD) | ❌ point estimate — needs ensembles or MCD for any uncertainty | ✅ closed-form Gaussian posterior at every point |

The pattern is consistent: **VSSGP is naturally Bayesian and natively handles linear constraints in closed form; NF is naturally flexible and natively handles arbitrary nonlinear losses with a learned operator inside**. For the kinds of physics the ocean reconstruction problem asks for — mostly linear (geostrophy, periodicity, advection-with-frozen-velocity, harmonic tides) plus a small amount of genuinely uncertain nonlinear physics (Chl-a sources) — VSSGP fills more of the space cleanly. NF earns its niche specifically where the operator itself needs to be learned.

---

## 1. AXIS — basis (function class)

The basis controls *what shapes of function `f` is allowed to take*, before any data is seen. For VSSGP this is `p(ω)` plus the input dimensionality of the RFF features; for NF this is the architecture (depth, width, activation, ω₀ for SIREN, positional encoding).

### 1.1 Informative spectral prior — VSSGP-native, NF approximate

For VSSGP, this is the headline mechanism from [00_siren_rff_vssgp.md §3](00_siren_rff_vssgp.md#3-physical-priors-for-ocean-variables): pick `p(ω)` to match the known spectral structure of the field (mesoscale band for SSH, annual harmonic for SST, etc.). Spectral mass on the right frequencies → most of the M features are useful.

For SIREN, there is no `p(ω)`, but there is the first-layer scaling `ω₀`. SIREN init is `W₁ ~ U(−√(6/n), √(6/n))`, then activations are `sin(ω₀ W₁ x)`. The effective spectral measure is uniform on a box `|ω| ≤ ω₀ √(6/n)`, dimensionwise. This is a *bad* approximation to a banded geophysical signal — see [00_siren_rff_vssgp.md §2](00_siren_rff_vssgp.md#2-why-geoscience-data-is-the-wrong-domain-for-siren) for the spectral-budget argument.

A cleaner SIREN cousin is **Fourier feature networks** (Tancik et al. 2020): replace the first layer's learned `W₁` with a *fixed* random matrix sampled from your chosen `p(ω)`, then put the MLP on top of these features. This brings SIREN into the same spectral-prior framework as VSSGP at the architecture level — strictly recommended for any SIREN run targeting geoscience data.

**Recipe.**

```python
# VSSGP: draw RFF frequencies from informative p(ω)
Omega = sample_spectral_mixture(M, components=miost_components, key=key)

# SIREN: fix the first-layer matrix to RFF frequencies, put MLP on top
class FourierFeatureSIREN(eqx.Module):
    Omega_fixed: Array     # (M, d), drawn from p(ω) once, frozen
    mlp: ...

    def __call__(self, x):
        gamma = jnp.concatenate([jnp.cos(self.Omega_fixed @ x),
                                 jnp.sin(self.Omega_fixed @ x)])
        return self.mlp(gamma)
```

Either way, the basis-axis physics is the **same act**: tell the model where energy lives in frequency space.

### 1.2 Anisotropic / flow-aligned basis

Genuine VSSGP/GP win — Matérn-SPDE prior with a position-dependent diffusivity tensor `D(x) = R(θ_flow) diag(L_∥², L_⊥²) R(θ_flow)ᵀ` encodes DYMOST-style anisotropy directly into the kernel ([00_siren_rff_vssgp.md §3.0.2](00_siren_rff_vssgp.md#302-dymost--anisotropic-flow-aligned-spectral-measure)).

For NF, the only available approximation is an input coordinate transform that biases the network toward the right axes:

```python
def aligned_input(x, theta_flow_field, ratio_field):
    R = flow_rotation(x, theta_flow_field)
    L = jnp.diag([1.0, ratio_field(x)])
    return R.T @ L @ R @ x
```

This is a soft architectural prior, not a hard basis constraint — it biases the implicit kernel of the NF without enforcing it. If anisotropy is critical (Gulf Stream, Kuroshio), VSSGP wins on this axis.

### 1.3 Multi-scale / additive components

For both methods. VSSGP: stack `Φ = [Φ_gyre | Φ_meso | Φ_sub]` with per-component variance scaling — see [00_siren_rff_vssgp.md §4](00_siren_rff_vssgp.md#4-how-additive-components-are-injected-into-vssgp). NF: parallel branches (one per scale octave) with their own first-layer `Ω_n`, summed at the output. Both express the same physical content: "the field is a sum of components, each with a known scale band and amplitude."

---

## 2. AXIS — data (input and output)

The data axis covers four sub-mechanisms: **(2A)** preprocessing the inputs `x`, **(2B)** augmenting the inputs with engineered features, **(2C)** preprocessing the observed `y`, and **(2D)** parameterising what `f` predicts (output structure). The first three are about *what enters* the model; the fourth is about *what comes out*. All four work the same way for NF and VSSGP.

### 2A. Input preprocessing

Coordinate normalisation, lat-lon → 3D Cartesian on the sphere (avoids the longitudinal seam), time → days since epoch with a per-domain offset. Standard ML hygiene; no physics yet, but matters because both SIREN's `ω₀` and the VSSGP RFF frequencies are scale-sensitive.

For SSH specifically, working in **f-plane regional patches** (small enough that Coriolis `f` is approximately constant) lets you treat `f/g` as a constant — needed for the scalar-potential trick in §2D below.

### 2B. Input embeddings — periodic features for known harmonics

Annual (T = 365.25 d), semiannual (182.6 d), diurnal (1.0 d), M2 tide (0.5175 d) are exact periodicities. Stack them into the input as

$$
\gamma(t) = \bigl[\cos(2\pi t/T_k),\; \sin(2\pi t/T_k)\bigr]_{k=1}^{K}
$$

before any other processing. Both methods then have a hard subspace where the harmonic content lives, learnt via the standard data fit.

```python
def periodic_embedding(x, t):
    periods = jnp.array([365.25, 182.6, 1.0, 0.5175])     # days
    h = jnp.concatenate([jnp.cos(2*jnp.pi*t/periods),
                         jnp.sin(2*jnp.pi*t/periods)])
    return jnp.concatenate([x, h])
```

Recipe is identical for VSSGP (`φ(periodic_embedding(x, t))`) and SIREN (input layer takes the embedding). High-leverage move because it removes a chunk of the training residual that the basis would otherwise have to spend M features approximating.

### 2C. Output preprocessing — subtract known physics from `y`

Before the model sees the observations, subtract everything that is already known. For SSH (covered in [02 §2a](../notebooks/02_physics_aware_ssh.md#2a-subtract--pre-processing-the-obs)):

| Subtraction | Source | Removes |
|---|---|---|
| Wet/dry tropospheric, ionospheric, sea-state bias | CMEMS L3 standard corrections | Path delay biases (~few cm each) |
| Solid Earth + pole tide, ocean tides FES2014/2022 | tide model | Earth-body + ocean tides (~30 cm + ~1 m peak) |
| Dynamic Atmospheric Correction (DAC) | MOG2D + IB | Storm-driven barotropic response (~10 cm at mid-lat) |
| Mean Dynamic Topography | CNES-CLS-22 | Time-mean SSH so you model the anomaly $\eta_a$ |

For SST: subtract climatology (Reynolds OISST mean) so the model sees an anomaly, not the absolute temperature. For SSS: subtract the multi-year mean. For Chl-a: log-transform first (the field is log-normal), then optionally subtract a seasonal climatology in log-space.

This is upstream of the model entirely and works identically for NF and VSSGP. **Highest-leverage single move on the data axis** — without it, the field's mean dynamic structure dominates the spectral content and the prior's mesoscale band gets drowned out.

### 2D. Output parameterisation — predict a scalar latent, derive the constrained field

The most under-used trick on the data axis. Instead of training `f` to predict the constrained vector field directly, train it to predict a *scalar latent* and apply a known operator to get the field. Constraints linear in `f` become **automatic** for every prediction — including out-of-domain ones.

The canonical example is divergence-free flow from a streamfunction:

$$
\psi_\theta(x) \;\;\xrightarrow{\;\;\text{predict}\;\;}\;\;
(u, v) = (-\partial_y \psi_\theta,\; +\partial_x \psi_\theta)
$$

For SSH: train on `η`, then derive geostrophic currents post-hoc as `(u_g, v_g) = (-(g/f)∂_y η, (g/f)∂_x η)`. Working with the scalar `η` *is* the scalar-latent trick — the resulting `(u_g, v_g)` is divergence-free for every `η` your model predicts.

| Output structure | NF recipe | VSSGP recipe |
|---|---|---|
| Direct field $f(x)$ | $f = \mathrm{NN}_\theta(x)$ | $f = \phi(x)^{\!\top} w$ |
| Scalar potential, derive vector field | $\psi = \mathrm{NN}_\theta(x);\; \mathbf{v} = \nabla^\perp \psi$ via `jax.grad` | $\psi = \phi(x)^{\!\top} w;\; \mathbf{v}$ via analytic derivative of $\phi$ |
| Log-transformed field | $f = \exp(\mathrm{NN}_\theta(x))$ for Chl-a | $f = \exp(\phi(x)^{\!\top} w)$; loses Gaussian likelihood |
| Multi-output (joint SSH+SST) | Vector head; shared backbone | Block-diagonal $w$, independent posteriors per channel; couple via shared spectral prior |

**Why this is so powerful for OOD prediction.** The constraint holds for *every* prediction, including in regions far from training data — divergence-freeness, harmonic content, log-positivity all transfer to the extrapolation regime. This is qualitatively different from a soft-loss approach where the constraint is only enforced where collocation points are placed. For OOD prediction, **output parameterisation is the only mechanism that reliably enforces a constraint everywhere**.

### 2E. Data augmentation — derivative observations

Drifter / HF-radar / Doppler-altimetry velocities give `(u_g, v_g)` directly, which by geostrophy is `-(g/f) ∂_y η, (g/f) ∂_x η` — a *linear* functional of `η`. Add these to the training set as derivative observations:

```python
y_augmented = jnp.concatenate([eta_obs, -(f/g) * v_drifter, (f/g) * u_drifter])
loss_data = mse(predict_eta_and_grad(x, theta), y_augmented)
```

For NF: `predict_eta_and_grad` computes `η` from the network and uses `jax.grad` to get gradient observations. For VSSGP: cross-covariance kernel rows `K_{η, ∂η} = ∂_{x'} k(x, x')` via `jax.grad(k)` — same construction, closed-form.

Both methods get the same physical content from this data augmentation.

---

## 3. AXIS — loss

The loss combines the data-fit term with regularisers and (for soft-physics) PDE-residual terms. Three sub-mechanisms.

### 3A. Data-fit likelihood

| Variable | Likelihood | NF form | VSSGP form |
|---|---|---|---|
| SSH | Gaussian (or correlated-noise for along-track) | MSE | Closed-form |
| SST | Student-T (cloud-contaminated IR) | Robust loss | SVI / Laplace |
| SSS | Student-T (RFI-contaminated SMOS) | Robust loss | SVI / Laplace |
| Chl-a (in log-space) | Gaussian on `log y` | MSE on `log f` | Closed-form on `log f` |

For VSSGP, anything beyond Gaussian breaks closed-form. The cost is real — Laplace / SVI introduces SGD-like training even on the GP side. For Gaussian-or-near-Gaussian variables (SSH, SST anomalies after climatology subtraction), VSSGP keeps its closed-form advantage; for genuinely heavy-tailed cases the methods are roughly on par cost-wise.

### 3B. Soft-physics — PDE residual at collocation points

The PINN move, with the same algebraic content for both methods:

$$
\mathcal{L}(\theta) = \underbrace{\sum_{i=1}^{N} \tfrac{(y_i - f_\theta(x_i))^2}{\sigma_{\text{obs}}^2}}_{\text{data fit}}
\;+\; \underbrace{\sum_{c=1}^{m_c} \tfrac{\|L f_\theta(x_c)\|^2}{\sigma_{\text{pde}}^2}}_{\text{soft physics at collocation pts}}
\;+\; \underbrace{\mathcal{R}(\theta)}_{\text{regulariser}}.
$$

In VSSGP language the second term is exactly the negative log-likelihood of *virtual observations* `0 = L f(x_c) + ε_c`, with `ε_c ~ N(0, σ_pde²)`. So it slots into the GP solve as additional rows in `y` and `K`. The cross-covariance machinery `K_{f, Lf}(x, x') = L_{x'} k(x, x')` is computed via `jax.grad(k)` — same primitive used for derivative observations in §2E.

| Aspect | NF | VSSGP |
|---|---|---|
| Linear PDE (`L f` linear in `f`) | Joint SGD | **Closed-form posterior** in one solve |
| Nonlinear PDE | Joint SGD | Laplace / SVI |
| Per-collocation-point cost | One forward + autodiff per epoch | One row in augmented Gram |
| Sensitivity to `σ_pde` | Loss-landscape stiffness; well-known PINN failure mode[^krishnapriyan2021] | Conditioning of augmented Gram; standard Tikhonov fixes |

```{tip}
**Key VSSGP win on this axis.** For a *linearised* PDE — linearised PV `∂_t (∇²η − η/L_R²) + βv = 0`, advection-diffusion with frozen velocity, harmonic balance — VSSGP gives the constrained posterior in a single solve. SIREN does the same constraint but needs full SGD training. For non-linear PDEs (full QG PV with the Jacobian), parity.
```

#### Per-variable PDE residuals

| Variable | Operator | Linear in `f`? |
|---|---|---|
| SSH | Geostrophic balance: $u_g = -(g/f)\partial_y\eta$ | Yes — but enforced **hard** via §2D, no loss term needed |
| SSH | Linearised PV: $\partial_t(\nabla^2\eta - \eta/L_R^2) + \beta v = 0$ | Yes — soft constraint, closed-form for VSSGP |
| SSH | Full nonlinear PV: $Dq/Dt = 0$ with quadratic Jacobian | No — Laplace / SGD |
| SST/SSS/Chl-a | Advection-diffusion (frozen `u`): $\partial_t T + \mathbf{u}\cdot\nabla T - \kappa\nabla^2 T = S$ | Yes (with frozen `u`) — soft constraint, two-stage |
| Chl-a | Add bio source-sink term `S` | No — needs learned $S_\theta$ (see §3D) |

The two-stage workflow for tracers: first reconstruct SSH (using §1 + §2 + §3A), freeze `u_g`, then for the tracer add the advection-diffusion residual as a soft constraint. Cost: one frozen velocity field, then a linear soft-physics solve.

### 3C. Regularisation — weight prior

For VSSGP this is built in (`w ~ N(0, Σ_w)`); for NF it has to be added as `‖θ‖²` weight decay or a structured regulariser. Worth distinguishing:

| Regulariser | NF | VSSGP |
|---|---|---|
| Plain `‖θ‖²` weight decay | Standard | Equivalent to isotropic `Σ_w = σ_w² I` |
| Per-component variance scale (different `σ_w` per scale octave) | Manual — pre-compute parameter groups, apply different decay to each | Closed-form via block-diagonal `Σ_w` (see [00_siren_rff_vssgp.md §4](00_siren_rff_vssgp.md#4-how-additive-components-are-injected-into-vssgp)) |
| Spatially varying amplitude `σ_w(x)` from DUACS | Awkward — needs gating into the network | Closed-form (one line in `Σ_w`) |
| Sparsity (Laplace prior on weights) | L1 weight decay | Spike-and-slab; expensive |

VSSGP is more flexible at the regulariser level because the weight prior is an explicit object, not a single scalar `λ`.

### 3D. Learned operators — when `L` itself is the unknown

For Chl-a, the source term `S(T, x, t)` is biology-dependent and uncertain. Replace `S` with a small neural network `S_θ(η, T, ∇T, x)` and train end-to-end with the data fit + advection-diffusion residual:

```python
def loss(theta_field, theta_S, x_data, y_data, x_collocation):
    # Data fit
    L_data = mse(f_theta_field(x_data) - y_data)
    # Soft physics with learned source
    residual = advect_diffuse(f_theta_field, x_collocation, u_g_frozen) \
               - S_theta_S(f_theta_field, x_collocation)
    L_phys = jnp.mean(residual ** 2) / sigma_pde**2
    # Regulariser on the learned operator
    L_reg  = mu * jnp.sum(theta_S ** 2)
    return L_data + L_phys + L_reg
```

This is **NF-native territory** — putting an NN inside the loss is the standard pattern. The VSSGP equivalent (NN coefficients as marginal-likelihood hyperparameters) loses the closed-form posterior on `θ_S` and is awkward in practice. For the four target variables, only Chl-a clearly needs this; the other operators are well-known enough that fixed coefficients suffice.

---

## 4. Decision matrix — which axis × method for which constraint?

For each constraint candidate across SSH/SST/SSS/Chl-a:

| Constraint | Variable | Axis | Mechanism | NF | VSSGP |
|---|---|---|---|---|---|
| Mesoscale spectral content | SSH | Basis | Informative `p(ω)` (MIOST init) | ⚠️ Fourier-feature wrapper | ✅ native |
| Annual / diurnal harmonics | All | Data — input | Periodic embedding | ✅ | ✅ |
| Anisotropic flow-aligned structure | SSH | Basis | Matérn-SPDE / coord transform | ⚠️ approximate | ✅ exact |
| Subtract tides + DAC + MDT | SSH | Data — output preproc | Upstream subtraction | ✅ | ✅ |
| Subtract climatology | SST/SSS | Data — output preproc | Standardise to anomaly | ✅ | ✅ |
| Log-transform | Chl-a | Data — output param | Predict `log f` | ✅ | ⚠️ breaks Gaussian likelihood |
| Geostrophic currents from `η` | SSH | Data — output param | Scalar latent + `jax.grad` / analytic deriv | ✅ | ✅ |
| Drifter velocities as ∂η obs | SSH | Data — augmentation | Cross-cov kernel rows / autodiff | ✅ | ✅ closed-form |
| Linearised PV conservation | SSH | Loss | Soft / virtual obs | ✅ SGD | ✅ **closed-form** |
| Full nonlinear PV | SSH | Loss | Soft (nonlinear) | ✅ SGD | ⚠️ SVI/Laplace — same cost as SGD |
| Advection-diffusion (frozen `u`) | SST/SSS/Chl-a | Loss | Soft, two-stage | ✅ SGD | ✅ **closed-form** |
| Bio source/sink | Chl-a | Loss | Learned `S_θ` | ✅ native | ❌ awkward |
| Sub-grid closure | All | Loss | Learned operator | ✅ native | ❌ |

The pattern (consistent with §0):

- **Linear constraints favour VSSGP** — closed-form posterior, single solve, calibrated OOD uncertainty for free.
- **Nonlinear-but-known constraints are roughly equal** — both need iterative optimisation.
- **Learned operators favour NF** — the Chl-a source term is the only place this clearly matters across the four variables.

The most expensive axis (loss with learned operator) is also the one where the methods most clearly diverge.

---

## 5. Recommended sequence — what to implement first

For each variable, in order of leverage (highest first), with method and axis labelled:

### SSH

1. **[Data — output preproc]** Subtract tides, DAC, MDT → work with anomaly $\eta_a$. ½-day fix; without it nothing else is right. Both methods.
2. **[Data — input]** Periodic harmonics (annual, M2). One-line input embedding. Both methods.
3. **[Data — output param]** Predict scalar `η`, derive currents post-hoc. Free divergence-free constraint everywhere — including OOD. Both methods.
4. **[Basis]** Informative spectral prior (MIOST mixture) — VSSGP native. For NF, use Fourier-feature first layer with the same `p(ω)`.
5. **[Data — augmentation]** Drifter velocities as derivative obs. Cheap, high-physical-content. Both methods.
6. **[Loss]** Linearised PV soft constraint at collocation points. **VSSGP wins here** — closed-form, single solve. NF fallback: standard PINN.
7. **[Loss]** Full nonlinear PV. Optional, only after 1–6 plateau. Parity between methods.

### SST

1. **[Data — output preproc]** Subtract Reynolds OISST climatology.
2. **[Data — input]** Periodic harmonics (annual, semiannual, diurnal).
3. **[Loss]** Student-T likelihood for cloud-contaminated retrievals.
4. **[Basis]** Spectral prior centred on frontal scales (~50 km).
5. **[Loss]** Two-stage advection-diffusion with `u_g` from SSH reconstruction. VSSGP closed-form for the linear PDE.

### SSS

Same as SST minus the diurnal harmonic (negligible). Spatially-varying `σ_pde` near coasts where `S` is uncertain.

### Chl-a

1. **[Data — output param]** Log-transform. Predict `log f`. NF: trivial. VSSGP: breaks Gaussian likelihood — need SVI / Laplace.
2. **[Data — input]** Periodic harmonic (annual; bloom seasonality).
3. **[Loss]** Two-stage advection-diffusion **without** source term. Captures bloom-edge propagation.
4. **[Loss]** Learned source `S_θ` only if observed bloom edges are sharper than the reconstruction. **NF wins here** — putting an NN inside the loss is native.

### Out-of-domain experiments

The most important test of the basis-axis informative prior is **prediction outside the training footprint**. Two recommended OOD splits:

- **Spatial OOD** — train on a subdomain (e.g. Z2 Gulf Stream), evaluate on a held-out adjacent region (Z3 \ Z2 in 00_siren_rff_vssgp.md §6.2). The prior's spectral structure dominates beyond the training boundary.
- **Temporal OOD** — train on Jan–Sep, evaluate on Oct–Dec. Tests whether the seasonal embeddings + spectral prior generalise over time.

For VSSGP, report the posterior mean *and* per-point standard deviation — the latter should grow as you move out of domain, with the right spectral content (visible by drawing a few posterior samples and inspecting their power spectrum). For NF, report a deep-ensemble mean ± std (5–10 independently trained networks); SIREN extrapolation is famously unstable[^lipman2021] and the ensemble spread is the cheapest signal of that instability.

---

## 6. My take

The constraint-mechanism story for this project is **three axes (basis / data / loss), not one**, and they have very different leverage profiles for OOD prediction:

> **Axis ordering by OOD leverage:** Data > Basis > Loss.

That ranking is specific to *out-of-domain* prediction and matters because it inverts the "in-domain" intuition.

- **Data-axis interventions enforce constraints *everywhere*** — including OOD. Predict `η` not `(u, v)`, work in log-space for Chl-a, subtract MDT, embed periodic harmonics — these constraints hold in the extrapolation regime by construction. They are the highest-leverage moves and they are *method-agnostic*.
- **Basis-axis interventions shape what the prior asserts in the gap** — they don't enforce constraints, they bias which prior samples look plausible. This is where the informative `p(ω)` from 00_siren_rff_vssgp.md does its work, and it's *most measurable* in OOD regions where the prior dominates.
- **Loss-axis interventions enforce constraints only at the collocation points used during training** — they do not generalise to OOD by themselves. A PINN trained with PV-residual collocation in Region A does not produce PV-conserving extrapolations in Region B unless you place collocation points there. **For OOD specifically, soft-physics losses are the weakest of the three axes.**

This is the part of the picture I had wrong in earlier framings of the doc: I was treating soft physics as the workhorse and basis/data as additions. For an OOD-prediction project the order is the reverse.

**Concrete recommendation for siren_vs_rff:**

1. Lock in all the data-axis moves first (preprocessing, periodic embeddings, scalar-latent parameterisation, derivative obs). These are the OOD-leverage moves and they apply identically to NF and VSSGP, so you can run them as the controlled-comparison baseline.
2. Then ablate the basis-axis informative prior — VSSGP with MIOST init vs. VSSGP with uniform `p(ω)` vs. SIREN with default ω₀ vs. SIREN with Fourier-feature wrapper using the same `p(ω)`. This is the cleanest test of the 00_siren_rff_vssgp.md spectral-budget claim, and the OOD region is where the differences should be most visible.
3. Add loss-axis soft physics last, with collocation points on a grid that *covers the OOD region as well as the training region* — otherwise the constraint won't help the metric you actually care about.

For the methods themselves: **VSSGP wins on basis and on linear-loss; NF wins on Chl-a's learned source term**. Use both, additively, where each is strong. The siren_vs_rff comparison is most informative if it reports `(method × axis × OOD-vs-in-domain)` cells separately rather than a single headline RMSE — the headline number hides the place where the methods most differ.

```{seealso}
- [00_siren_rff_vssgp.md](00_siren_rff_vssgp.md) — VSSGP foundations, spectral priors, four-rung hierarchy, experiment plan
- [02_physics_aware_ssh.md](../notebooks/02_physics_aware_ssh.md) — same DATA + KERNEL framing for the pathwise GP
- [04_variational_dynamical_priors.md](../notebooks/04_variational_dynamical_priors.md) — the variational-DA framework, when you want a dynamical prior instead
```

---

## References

[^lindgren2011]: Lindgren, F., Rue, H., Lindström, J. (2011). "An explicit link between Gaussian fields and Gaussian Markov random fields: the stochastic partial differential equation approach." *Journal of the Royal Statistical Society B* 73, 423–498.

[^alvarez2009]: Álvarez, M., Luengo, D., Lawrence, N. (2009). "Latent force models." *AISTATS* 2009, 9–16.

[^raissi2019]: Raissi, M., Perdikaris, P., Karniadakis, G. E. (2019). "Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics* 378, 686–707.

[^krishnapriyan2021]: Krishnapriyan, A., Gholami, A., Zhe, S., Kirby, R., Mahoney, M. (2021). "Characterizing possible failure modes in physics-informed neural networks." *NeurIPS* 34.

[^tancik2020]: Tancik, M., Srinivasan, P., Mildenhall, B., et al. (2020). "Fourier features let networks learn high-frequency functions in low-dimensional domains." *NeurIPS* 33.

[^richter2022]: Richter-Powell, J., Lipman, Y., Chen, R. T. Q. (2022). "Neural Conservation Laws: A Divergence-Free Perspective." *NeurIPS* 35.

[^lipman2021]: Lipman, Y., Aharoni, M. (2021). "On the failure modes of implicit neural representations under extrapolation." Workshop on Implicit Neural Representations.

[^chen2018]: Chen, R. T. Q., Rubanova, Y., Bettencourt, J., Duvenaud, D. (2018). "Neural ordinary differential equations." *NeurIPS* 31.

[^li2020fno]: Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., Anandkumar, A. (2020). "Fourier neural operator for parametric partial differential equations." *ICLR* 2021.

[^held1995]: Held, I., Pierrehumbert, R., Garner, S., Swanson, K. (1995). "Surface quasi-geostrophic dynamics." *J. Fluid Mech.* 282, 1–20.
