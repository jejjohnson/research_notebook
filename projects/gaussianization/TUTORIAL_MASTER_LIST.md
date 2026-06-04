# Gaussianization Tutorial Master List

A reconciled, exhaustive curriculum spanning what currently exists in **rbig**, **gauss_flows**, and **research_notebook/projects/gaussianization**, plus gaps surfaced from the package APIs, open issues, and design docs. Goal: the most complete Gaussianization tutorial sequence we could ship.

> Companion to [`../gaussian_processes/TUTORIAL_MASTER_LIST.md`](../gaussian_processes/TUTORIAL_MASTER_LIST.md). Cross-listed items (spatial-extremes margins, spatiotemporal applications, Kalman / filtering) are flagged 🔁 and tagged with `xref:GP#X.Y`.

**Legend** — Source columns:
- `B` = exists in **rbig** (`docs/notebooks/<name>`)
- `F` = exists in **gauss_flows** (`docs/notebooks/<name>`)
- `K` = exists in **research_notebook gauss_keras** (`projects/gaussianization/notebooks/<name>`)
- `R` = exists in **research_notebook** elsewhere
- `—` = does not exist yet (gap)

**Scope tag**: 🧱 fundamental · 🔬 research · 🌉 bridge · 🔁 cross-listed

**Refs column**: `gh:<repo>#N` = open GitHub issue · `dd:path` = design-doc path · `api:foo` = exported symbol · `xref:GP#X.Y` = cross-ref to GP master list.

**Framing**: Gaussianization is a *transformation* to a target Gaussian, so this curriculum is organized along three axes: (1) **what we map with** — 1D marginals, rotations, couplings, ODEs, surjections; (2) **how we fit it** — iterative / greedy vs. end-to-end NLL; (3) **what we do with it** — sampling, density, IT measures, inverse problems, filtering, fair learning. Normalizing-flow cousins (MAF / IAF / NSF / Glow / FFJORD) are treated as **bridges**, not first-class entries, except where they're the canonical implementation of a Gaussianization idea.

---

## Curriculum at a glance

A bird's-eye view of the parts and their subparts. Skim this first to orient; the detailed per-tutorial tables live below.

- **Part 0 — Foundations**
  - 0.A — Change of variables & log-determinant
  - 0.B — Why standard Gaussian as target
  - 0.C — Density destructors
  - 0.D — Numerical mechanics
  - 0.E — Diagnostics
- **Part 1 — 1D Marginal Transforms**
  - 1.A — Empirical CDF & histograms
  - 1.B — KDE / Gaussian-mixture CDFs
  - 1.C — Monotone-spline CDFs
  - 1.D — Mixture-CDF as a learnable bijector
  - 1.E — Inversion strategies
- **Part 2 — Rotations & Orthogonal Mixers**
  - 2.A — Linear-rotation zoo (PCA / ICA / random / Picard)
  - 2.B — Householder products & trainable orthogonals
  - 2.C — Fixed orthogonal & PCA warm starts
  - 2.D — Invertible 1×1 conv (LU parameterization)
  - 2.E — ActNorm & per-channel affine
- **Part 3 — Iterative Gaussianization (RBIG)**
  - 3.A — Canonical RBIG loop
  - 3.B — Convergence & stopping criteria
  - 3.C — Rotation-choice studies
  - 3.D — RBIG as warm-start for parametric flows
  - 3.E — Boundary issues & support extension
- **Part 4 — Parametric Gaussianization Flows**
  - 4.A — NLL training of stacked blocks
  - 4.B — Diagonal vs. coupling marginal flow
  - 4.C — Factory walkthroughs
  - 4.D — Layer-wise inspection
- **Part 5 — Coupling-based Gaussianization**
  - 5.A — The coupling pattern
  - 5.B — Bijector menu for coupling
  - 5.C — Conditioner architectures *(headline)*
  - 5.D — Mask design
  - 5.E — Coupling ↔ diagonal equivalence
  - 5.F — Depth, residual coupling, stability
- **Part 6 — Continuous-time Gaussianization** *(bridge)*
  - 6.A — FFJORD
  - 6.B — Hutchinson trace estimator
  - 6.C — Matrix-exponential / linear neural flows
  - 6.D — Latent ODEs
- **Part 7 — Conditional Gaussianization**
  - 7.A — Conditioner zoo for marginals & couplings
  - 7.B — Conditional density estimation
  - 7.C — Three-pattern conditional flow
  - 7.D — Conditioning for inverse problems
- **Part 8 — SurVAE: Surjections & Stochastic Transforms**
  - 8.A — Bijection / surjection / stochastic taxonomy
  - 8.B — Slicing & augmentation surjections
  - 8.C — Stochastic transforms & ELBO
- **Part 9 — Relaxed-Bijectivity & Non-Invertible Flows**
  - 9.A — Injective / lossy Gaussianization
  - 9.B — Augmented / lifted flows
  - 9.C — Continuously-indexed flows (CIF)
  - 9.D — Stochastic normalizing flows
  - 9.E — Diffusion as continuous stochastic Gaussianization
  - 9.F — Residual / implicit flows
  - 9.G — One-shot / Trumpet-style Gaussianizers
  - 9.H — When non-invertibility helps
- **Part 10 — Non-Euclidean Gaussianization**
  - 10.A — Circle / torus
  - 10.B — Sphere
  - 10.C — Lie groups / Riemannian manifolds
- **Part 11 — Time-Series Gaussianization**
  - 11.A — Per-timestep marginal Gaussianization
  - 11.B — Autoregressive flows for sequences
  - 11.C — Conditioning on past context
  - 11.D — AR(p) in Gaussianized state
  - 11.E — Latent ODEs for irregular series
  - 11.F — Long-range / hierarchical temporal couplings
  - 11.G — Multiscale temporal flows
  - 11.H — Time-series anomaly & changepoint detection
- **Part 12 — Spatial / Image Gaussianization**
  - 12.A — Multiscale Squeeze / unsqueeze
  - 12.B — Invertible 1×1 conv & Haar wavelet
  - 12.C — Patch-based image flows
  - 12.D — Equivariant flows (rotation / translation)
  - 12.E — Spatial random-field Gaussianization (GRF prior)
  - 12.F — Image-rotation diagnostics
  - 12.G — Glow end-to-end
- **Part 13 — Spatiotemporal Fields & Videos**
  - 13.A — Separable space×time coupling
  - 13.B — Frame-conditioned video flows
  - 13.C — Spatiotemporal RBIG (lat × lon × time)
  - 13.D — Latent ODEs for field dynamics
  - 13.E — Equivariant spatiotemporal flows
  - 13.F — Climate-field scale-up
  - 13.G — Multiscale spatiotemporal
- **Part 14 — Information-Theoretic Estimation**
  - 14.A — Entropy & negentropy from Gaussianized residual
  - 14.B — Mutual information & total correlation
  - 14.C — KL between empirical distributions
  - 14.D — Dependence measures
  - 14.E — Real-data IT pipelines
  - 14.F — Bias-variance & sample complexity
- **Part 15 — Fair Learning with Frozen Gaussianization Flows** 🔬
  - 15.A — Frozen flow as differentiable independence loss
  - 15.B — G-XCOV vs G-HSIC vs CKA
  - 15.C — Pretrain + freeze workflow
  - 15.D — Synthetic fairness sweeps & Pareto curves
  - 15.E — Adult census case study
  - 15.F — Drop-in with `FairModelWrapper`
  - 15.G — Open research directions
- **Part 16 — Plug-and-Play Priors with Gaussianization**
  - 16.A — PnP recap (denoiser-as-prior)
  - 16.B — Closed-form prox in Gaussianized latent space
  - 16.C — HQS / ADMM with Gaussianization
  - 16.D — Patch-based PnP for images
  - 16.E — Linear inverse problems (deblur / SR / inpaint / CS)
  - 16.F — Comparison with score / diffusion PnP
- **Part 17 — Filtering & Data Assimilation**
  - 17.A — Normalizing Kalman filter (closed-form)
  - 17.B — State vs observation vs QoI Gaussianization
  - 17.C — Gaussianized ensemble Kalman filter
  - 17.D — Non-Gaussian likelihoods in DA
  - 17.E — Sequential Bayesian updates in latent space
  - 17.F — Comparison with EKF / UKF / particle filter
- **Part 18 — Geoscience Case Studies**
  - 18.A — Quantile mapping / bias correction
  - 18.B — Climate-field anomaly detection
  - 18.C — Spatial extremes (GEV / Gumbel margins) 🔁
  - 18.D — Ocean SST / sea-level extremes
  - 18.E — Precipitation Gaussianization
  - 18.F — Wind / atmospheric tracers
  - 18.G — Satellite-image emulation
  - 18.H — Climate-data assimilation
- **Part 19 — Probabilistic-Programming Integration**
  - 19.A — `FlowDist` in NumPyro
  - 19.B — Flows as priors in BHMs
  - 19.C — Flows as guides in SVI
  - 19.D — pyrox integration patterns
- **Part 20 — Metrics, Calibration, Diagnostics**
  - 20.A — NLL / bits-per-dim
  - 20.B — QQ / PIT / coverage for flows
  - 20.C — Sample quality (FID-style / MMD)
  - 20.D — Roundtrip invertibility & numerical tolerance
  - 20.E — Cross-validation for IT estimators

---

## Part 0 — Foundations

### 0.A — Change of variables & log-determinant

**Key equations / models:**
- Change of variables: $p_X(x) = p_Z(T(x))\,|\det J_T(x)|$, $T$ a $C^1$-diffeomorphism
- Log-density: $\log p_X(x) = \log p_Z(T(x)) + \log|\det J_T(x)|$
- Composition: $T = T_K \circ \cdots \circ T_1 \Rightarrow \log|\det J_T(x)| = \sum_{k=1}^{K} \log|\det J_{T_k}(x_{k-1})|$
- Forward (data → latent) vs. inverse (latent → data) parameterisation

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 0.1 | Change of variables from scratch — 1D, 2D, $d$-D, both directions | K `00_foundations/00_change_of_variables` | 🧱 | pedagogical anchor; verifies CoV against `gauss_flows` log-det |
| 0.2 | Composition of bijectors & additive log-determinant | K `00_foundations/01_composition_logdet` | 🧱 | rotations are free; `flowjax.Chain` |
| 0.3 | Forward vs. inverse parameterisation — "density estimation" vs. "generation" trade-offs | K `00_foundations/02_forward_vs_inverse` | 🧱 | `optimistix` root-find + implicit-adjoint gradients |

### 0.B — Why standard Gaussian as target

**Key equations / models:**
- Max-entropy result: $\arg\max_{p}\,H(p)$ s.t. fixed first/second moment $= \mathcal{N}(\mu,\Sigma)$
- Separable: $\mathcal{N}(0,I) = \prod_i \mathcal{N}(0,1)$ → IT measures decompose per-coordinate
- Trivial sampler, trivial score $\nabla\log\mathcal{N}(z;0,I) = -z$, trivial prox

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 0.4 | Why $\mathcal{N}(0,I)$? Max-entropy + separability + trivial primitives | K `00_foundations/03_why_standard_gaussian` | 🧱 | `rbig.negentropy`/`total_correlation`; sets up 16.B (prox), 14.A (IT), 17 (Kalman) |

### 0.C — Density destructors

**Key equations / models:**
- Density destructor: invertible map $T: \mathbb{R}^d \to \mathbb{R}^d$ with $T_\#p_X = \mathcal{N}(0,I)$
- Gaussianization = whitening (rotation + scaling) composed with element-wise nonlinearity (CDF map), iterated to convergence
- Inverse generative direction: sample $z\sim\mathcal{N}(0,I)$, return $T^{-1}(z)$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 0.5 | Density destructors — Inouye & Ravikumar 2018 framing | K `00_foundations/04_density_destructors` | 🧱 | unifies flow / Gaussianization / destructor |
| 0.6 | Gaussianization = iterated whitening + nonlinearity — intuition pictures | K `00_foundations/04_density_destructors` | 🧱 | `rbig.AnnealedRBIG`; two-moons → N(0,I) morph |

### 0.D — Numerical mechanics

**Key equations / models:**
- Jitter on CDFs: clip to $[\epsilon, 1-\epsilon]$ before $\Phi^{-1}$ to avoid $\pm\infty$
- Mixed-precision log-det: accumulate in float64 even when forward is float32
- Stable $\log\Phi^{-1}$ tails via series / asymptotic expansions
- Invertibility check: $\|T^{-1}(T(x)) - x\|_\infty < \tau$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 0.7 | Numerical stability for bijectors — jitter, mixed precision, tail expansions | K `00_foundations/05_numerical_mechanics` | 🧱 | pairs with `xref:GP#0.11` (jitter / safe Cholesky) |
| 0.8 | Log-determinant accumulation across deep stacks | K `00_foundations/05_numerical_mechanics` | 🧱 | float32 vs float64 drift |
| 0.9 | Roundtrip invertibility tests in CI | K `00_foundations/05_numerical_mechanics` | 🧱 | caught `gh:gauss_flows#108` (fixed in 0.1.7) |

### 0.E — Diagnostics

**Key equations / models:**
- QQ-plot against $\mathcal{N}(0,1)$ per coordinate
- Sample skewness / excess kurtosis: $\gamma_1, \gamma_2$ → both 0 under Gaussian
- Negentropy $J(p) = H(\mathcal{N}(0,\Sigma_p)) - H(p) \geq 0$
- Multivariate KS, Henze–Zirkler

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 0.10 | QQ-plot & moment-based Gaussianity diagnostics | K `00_foundations/06_gaussianity_diagnostics` | 🧱 | QQ + skew/kurtosis before vs after |
| 0.11 | Negentropy as a convergence signal for RBIG | K `00_foundations/06_gaussianity_diagnostics` | 🧱 | `rbig.negentropy`; feeds 3.B stopping criterion |
| 0.12 | Multivariate Gaussianity tests (Henze–Zirkler, energy) | K `00_foundations/06_gaussianity_diagnostics` | 🧱 | energy distance to N(0,I); HZ noted |

---

## Part 1 — 1D Marginal Transforms

The atomic operation of Gaussianization: turn each coordinate's distribution into a standard Gaussian via $z_i = \Phi^{-1}(F_i(x_i))$ for some monotone CDF estimator $F_i$.

### 1.A — Empirical CDF & histograms

**Key equations / models:**
- Empirical CDF: $\hat F_n(x) = \tfrac{1}{n}\sum_{i=1}^n \mathbf{1}\{x_i\le x\}$
- Histogram-CDF: piecewise linear interpolation of bin counts
- Glivenko–Cantelli: $\sup_x|\hat F_n(x) - F(x)| \to 0$ a.s.

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 1.1 | Marginal transforms — ECDF & histograms | B `01_marginal_transforms` | 🧱 | rbig canonical entry; rank → uniform → normal |
| 1.2 | Boundary issues & support extension | B `07_boundary_issues` | 🧱 | tails, $\pm\infty$ handling, rbig's three remedies |
| 1.3 | Glivenko–Cantelli & finite-sample bias | — | 🧱 | **GAP** — pedagogical |

### 1.B — KDE / Gaussian-mixture CDFs

**Key equations / models:**
- KDE: $\hat f_h(x) = \tfrac{1}{nh}\sum_i K\!\big(\tfrac{x-x_i}{h}\big)$
- Mixture-of-Gaussians CDF: $F(x) = \sum_k \pi_k\,\Phi(x;\mu_k,\sigma_k^2)$
- Bandwidth: Silverman, ISJ, cross-validated

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 1.4 | KDE-based 1D CDF Gaussianization | — | 🧱 | **GAP** |
| 1.5 | Gaussian-mixture CDF — analytic forward & inverse | — | 🧱 | api: `MixtureOfGaussians` (gauss_keras) |
| 1.6 | Bandwidth / component-count selection | — | 🧱 | **GAP** |

### 1.C — Monotone-spline CDFs

**Key equations / models:**
- Monotone cubic Hermite (Fritsch–Carlson): piecewise cubic with positive slopes
- Rational-quadratic spline (RQS, Durkan 2019): $F(x) = \frac{\alpha y^2 + \beta y + \gamma}{a y^2 + b y + c}$, exact inverse
- Log-det = sum of log derivatives at knot intervals

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 1.7 | Monotone cubic spline CDF Gaussianization | — | 🧱 | **GAP** |
| 1.8 | Rational-quadratic spline as a 1D bijector | — | 🧱 | **GAP** — feeds 5.B (coupling bijector menu) |

### 1.D — Mixture-CDF as a learnable bijector

**Key equations / models:**
- Forward: $u = F_\theta(x) = \sum_k \pi_k\,\Phi(x;\mu_k,\sigma_k)$, $z = \Phi^{-1}(u)$
- Log-det: $\log f_\theta(x) - \log\phi(z)$
- Inverse: $z\to u = \Phi(z)$, then $x = F_\theta^{-1}(u)$ via bisection

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 1.9 | Mixture-CDF Gaussianization layer end-to-end | K `gauss_keras.MixtureCDFGaussianization` | 🧱 | api: `MixtureCDFGaussianization` |
| 1.10 | Differentiating through the mixture-CDF — implicit-function gradient | — | 🧱 | **GAP** — pairs with `xref:GP#0.10` |

### 1.E — Inversion strategies

**Key equations / models:**
- Bisection: $O(\log(1/\epsilon))$ iterations, derivative-free, robust
- Newton: quadratic convergence near root, needs $F'$
- Hybrid: bisection bracket → Newton refine
- Vectorised inversion via `jax.lax.while_loop` / Keras ops

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 1.11 | Bisection vs. Newton for monotone CDF inversion | — | 🧱 | **GAP** |
| 1.12 | Vectorised batched root-find across leading axes | — | 🧱 | **GAP** |

---

## Part 2 — Rotations & Orthogonal Mixers

The "between-coordinate" half of Gaussianization: orthogonal mixers that redistribute information across dimensions so the next marginal pass has something to do.

### 2.A — Linear-rotation zoo

**Key equations / models:**
- PCA: $W = Q^\top$, $Q$ from eigendecomposition of $\mathrm{Cov}(X)$
- ICA: maximise non-Gaussianity of marginals (FastICA, Infomax)
- Random orthogonal: $Q\sim\mathrm{Haar}(O(d))$ via QR of Gaussian
- Picard: Riemannian L-BFGS on the orthogonal manifold

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 2.1 | Rotation choices — PCA / ICA / random / Picard | B `08_rotation_choices` | 🧱 | rbig canonical |
| 2.2 | Why rotation matters between marginal passes | — | 🧱 | **GAP** — pedagogical |

### 2.B — Householder products & trainable orthogonals

**Key equations / models:**
- Householder reflection: $H = I - 2vv^\top / \|v\|^2$
- Product: $Q = H_1 H_2 \cdots H_K$, exactly orthogonal by construction
- Cayley parameterisation: $Q = (I - A)(I + A)^{-1}$, $A$ skew-symmetric

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 2.3 | Householder products as trainable orthogonals | K `gauss_keras.Householder` | 🧱 | api: `Householder`, `_householder_decomp` |
| 2.4 | Cayley & exponential parameterisations of $O(d)$ | — | 🧱 | **GAP** |

### 2.C — Fixed orthogonal & PCA warm starts

**Key equations / models:**
- `FixedOrtho` from data PCA: freeze $Q$, only learn marginals downstream
- Warm-start: initialise Householder product to match a PCA $Q$ via QR decomposition

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 2.5 | FixedOrtho & `from_pca` factory | K `gauss_keras.FixedOrtho.from_pca` | 🧱 | api: `FixedOrtho.from_pca` |
| 2.6 | Initialising a Householder stack from a target $Q$ | — | 🧱 | **GAP** — api: `_householder_decomp` |

### 2.D — Invertible 1×1 conv (LU parameterization)

**Key equations / models:**
- 1×1 conv at each spatial location: $z_{ij} = W x_{ij}$, $W\in\mathbb{R}^{C\times C}$
- LU parameterisation: $W = PL(U + \mathrm{diag}(s))$, $\log|\det W| = \sum\log|s|$
- Connects 2.B (orthogonal mixers) to 12.B (image flows)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 2.7 | Invertible 1×1 conv as a per-pixel orthogonal mixer | — | 🧱 | **GAP** — pairs with 12.B |

### 2.E — ActNorm & per-channel affine

**Key equations / models:**
- ActNorm: $z = s \odot x + b$, $s, b$ data-dependent initialised to give zero mean / unit variance at init
- Log-det = $\sum \log|s|$
- Why it matters: makes deep stacks trainable, classical pre-flow whitening

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 2.8 | ActNorm — data-dependent affine pre-conditioning | — | 🧱 | **GAP** |

---

## Part 3 — Iterative Gaussianization (RBIG)

The classical, non-parametric Gaussianization algorithm: alternate marginal CDFs (Part 1) with a rotation (Part 2) until the joint converges to $\mathcal{N}(0,I)$.

### 3.A — Canonical RBIG loop

**Key equations / models:**
- Iteration $k$: $x^{(k+1)} = Q_k\,T_k(x^{(k)})$, $T_k$ per-coordinate Gaussianization, $Q_k$ rotation
- Convergence (Laparra et al. 2011): $\mathrm{KL}(p_k \Vert \mathcal{N}(0,I)) \to 0$ for generic $Q$
- Forward stack composes log-dets additively

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 3.1 | RBIG walkthrough — the iterated algorithm | B `03_rbig_walkthrough` | 🧱 | canonical |
| 3.2 | RBIG demo on 2-D toy distributions | B `04_rbig_demo` | 🧱 | |

### 3.B — Convergence & stopping criteria

**Key equations / models:**
- Per-layer information reduction: $J(p_{k+1}) < J(p_k)$ (Laparra 2011 monotone decrease)
- Stop when $|J(p_k) - J(p_{k+1})| < \tau$ or fixed depth $K$
- Bias-corrected mutual information change across iterations

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 3.3 | RBIG loss / negentropy as a stopping signal | B `05_rbig_loss` | 🧱 | |
| 3.4 | Depth selection — fixed-K vs. early-stop | — | 🧱 | **GAP** |

### 3.C — Rotation-choice studies

**Key equations / models:**
- Convergence rate as a function of $Q$ family
- PCA: variance-aligning, good when scales differ
- ICA: aligns to non-Gaussian directions, faster on heavy-tailed data
- Random: minimax-flavoured, no fitting cost

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 3.5 | Rotation choices revisited — convergence comparison | B `08_rotation_choices` | 🧱 | rbig canonical (also 2.A) |
| 3.6 | Picard rotation for fast RBIG | — | 🔬 | **GAP** |

### 3.D — RBIG as warm-start for parametric flows

**Key equations / models:**
- Greedy fit each block to data, then jointly fine-tune via NLL
- `initialize_flow_from_ig` — sklearn PCA + GMM per block

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 3.7 | Iterative Gaussianization warm-start | K `03_iterative_gaussianization_init` | 🧱 | api: `initialize_flow_from_ig` |
| 3.8 | RBIG warm-start for FlowJax-style flows | F `03_rbig_warmstart` | 🌉 | |

### 3.E — Boundary issues & support extension

**Key equations / models:**
- Quantile clipping: $u \in [\epsilon, 1-\epsilon]$ to keep $\Phi^{-1}(u)$ finite
- Tail extrapolation via GPD / Gumbel
- Support extension by mixture with uniform / Gaussian noise (dequantisation)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 3.9 | Boundary issues & tail handling in RBIG | B `07_boundary_issues` | 🧱 | (also 1.2) |
| 3.10 | Dequantisation for discrete inputs | — | 🧱 | **GAP** — pairs with 8.B |

---

## Part 4 — Parametric Gaussianization Flows

Stack the rotation + marginal blocks into a differentiable graph and train end-to-end with maximum likelihood.

### 4.A — NLL training of stacked blocks

**Key equations / models:**
- Loss: $\mathcal{L}(\theta) = -\tfrac{1}{n}\sum_i \big[\log p_Z(T_\theta(x_i)) + \log|\det J_{T_\theta}(x_i)|\big]$
- $p_Z = \mathcal{N}(0,I)$ so $-\log p_Z(z) = \tfrac{1}{2}\|z\|^2 + \tfrac{d}{2}\log 2\pi$
- Gradient through `bisection` inverse via implicit-function theorem

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 4.1 | Gaussianization flow on two-moons end-to-end | K `01_gaussianization_2d` | 🧱 | iterative vs. parametric on the same target |
| 4.2 | Gaussianization flow 2D — FlowJax variant | F `01_gaussianization_flow_2d` | 🌉 | |
| 4.3 | NLL loss anatomy — base + log-det decomposition | — | 🧱 | api: `base_nll_loss` |

### 4.B — Diagonal vs. coupling marginal flow

**Key equations / models:**
- Diagonal: independent 1D CDF per coordinate, no cross-coordinate conditioning
- Coupling: bijector on one half conditioned on the other → expressive non-separability

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 4.4 | Diagonal vs. coupling — depth-vs-expressiveness study | — | 🧱 | **GAP** — feeds 5.E |

### 4.C — Factory walkthroughs

**Key equations / models:**
- `make_gaussianization_flow(...)` — stacked rotation + diagonal marginal
- `make_coupling_flow(...)` — stacked rotation + mixture-CDF coupling

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 4.5 | `make_gaussianization_flow` walkthrough | — | 🧱 | api: `make_gaussianization_flow` |
| 4.6 | `make_coupling_flow` walkthrough | — | 🧱 | api: `make_coupling_flow` |

### 4.D — Layer-wise inspection

**Key equations / models:**
- `forward_with_intermediates(x)` returns the trajectory $(x, T_1(x), T_2 T_1(x), \dots)$
- Diagnose where Gaussianisation is "stuck" via per-layer QQ / skew / negentropy

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 4.7 | Layer-wise inspection of a Gaussianization flow | — | 🧱 | api: `GaussianizationFlow.forward_with_intermediates` |

---

## Part 5 — Coupling-based Gaussianization

Coupling is the expressive engine of modern Gaussianization: split coordinates with a mask, apply a per-coordinate bijector whose parameters are predicted by a **conditioner** from the unchanged half.

### 5.A — The coupling pattern

**Key equations / models:**
- Split: $x = (x_A, x_B)$ via mask $m$
- Forward: $z_A = x_A$, $z_B = T_{\theta(x_A)}(x_B)$
- Log-det: $\sum_i \log\partial_{x_{B,i}} T_{\theta(x_A)}(x_{B,i})$ — free because Jacobian is triangular
- Inverse: $x_A = z_A$, $x_B = T^{-1}_{\theta(z_A)}(z_B)$ — no neural-net inverse needed

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 5.1 | Coupling pattern from RealNVP to mixture-CDF coupling | K `02_coupling_flow_2d` | 🧱 | |
| 5.2 | Coupling flow 2D — FlowJax variant | F `02_coupling_flow_2d` | 🌉 | |
| 5.3 | Triangular Jacobian — why coupling log-det is free | — | 🧱 | **GAP** — pedagogical |

### 5.B — Bijector menu for coupling

**Key equations / models:**
- Affine: $T(x;s,b) = s\odot x + b$, $\log|\det| = \sum\log|s|$
- Mixture-CDF: $T(x_B;\theta) = \Phi^{-1}\!\big(\sum_k \pi_k(\theta)\,\Phi(x_B;\mu_k(\theta),\sigma_k(\theta))\big)$
- Deep sigmoid: cascaded $\sigma$-shifts, expressive monotone
- Rational-quadratic spline (NSF, Durkan 2019)
- Residual coupling: $T(x) = x + g_\theta(x)$ with Lipschitz $g$ (preview of 9.E)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 5.4 | Affine coupling — RealNVP foundation | — | 🧱 | **GAP** |
| 5.5 | Mixture-CDF coupling | K `gauss_keras.MixtureCDFCoupling` | 🧱 | api: `MixtureCDFCoupling` |
| 5.6 | Deep sigmoid coupling | — | 🧱 | **GAP** |
| 5.7 | Rational-quadratic spline (NSF) coupling | — | 🧱 | **GAP** — feeds 1.8 |

### 5.C — Conditioner architectures *(headline)*

> The conditioner *is* the expressive part — the bijector is just a triangular wrapper that makes log-det free. Every structured-data part (11 / 12 / 13) revisits this menu and picks the modality-appropriate architecture.

**Key equations / models:**
- Conditioner $\theta = c_\phi(x_A)$, $c_\phi : \mathbb{R}^{|A|} \to \mathbb{R}^{n_\text{params}(B)}$
- MLP: $h_\ell = \mathrm{act}(W_\ell h_{\ell-1} + b_\ell)$
- Shared MLP: one trunk, separate per-coordinate heads (parameter-efficient)
- CNN-conditioner (image): conv stack preserving spatial dims (12.C)
- RNN / Transformer / Mamba conditioner (sequence): causal attention or recurrence (11.B)
- GNN-conditioner: message-passing on graph-structured inputs
- Equivariant conditioner: enforce symmetry of the data domain
- Hypernetwork conditioner: $c_\phi(x_A) = \mathrm{hypernet}(z) \cdot \mathrm{net}(x_A)$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 5.8 | MLP & shared-MLP conditioners | K `make_mlp_conditioner`, `make_shared_mlp_conditioner` | 🧱 | api builders |
| 5.9 | Conditioner output parameterisation — log-scale clamping for stability | K `tanh_log_scale_clamp`, `sigmoid_log_scale_clamp` | 🧱 | api: clamps |
| 5.10 | ResNet & deep MLP conditioners | — | 🧱 | **GAP** |
| 5.11 | CNN conditioner for image coupling | — | 🧱 | **GAP** — referenced by 12.C |
| 5.12 | RNN / Mamba / Transformer conditioners for sequence coupling | — | 🌉 | **GAP** — referenced by 11.B |
| 5.13 | GNN conditioner for graph-structured coupling | — | 🔬 | **GAP** |
| 5.14 | Equivariant conditioners | — | 🔬 | **GAP** — referenced by 12.D / 13.E |
| 5.15 | Hypernetwork conditioners | — | 🔬 | **GAP** |
| 5.16 | Parameter budget vs. expressiveness — when does adding conditioner depth help? | — | 🧱 | **GAP** — pedagogical |
| 5.17 | Three-pattern conditional flow construction | F `08_conditional_flow_three_ways` | 🌉 | also 7.C |

### 5.D — Mask design

**Key equations / models:**
- Checkerboard / striped: spatial alternation
- Channel-wise (split halves): standard RealNVP
- Learned mask: differentiable via Gumbel-softmax
- Stacking: alternate masks so every coordinate is updated and conditions

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 5.18 | Mask design — checkerboard / channel-wise / learned | — | 🧱 | api: `default_half_mask` |
| 5.19 | Mask stacking & receptive-field analysis | — | 🧱 | **GAP** |

### 5.E — Coupling ↔ diagonal equivalence

**Key equations / models:**
- Zero-kernel init: if the conditioner outputs constants, coupling collapses to a diagonal-marginal flow
- Numerical-equivalence proof: matched outputs at init within `1e-6`
- Training "breaks the equivalence" — diagnostic for whether the conditioner is doing anything

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 5.20 | Coupling ↔ diagonal equivalence — proof & empirical check | K `04_coupling_equivalence` | 🧱 | also F `04_coupling_equivalence` |

### 5.F — Depth, residual coupling, stability

**Key equations / models:**
- Stacked coupling: $T = T_K \circ \cdots \circ T_1$ with alternating masks
- Residual: $T(x) = x + g_\theta(x)$ with Lipschitz constraint
- Gradient pathology at depth — ActNorm pre-conditioning helps (2.E)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 5.21 | Depth-vs-expressiveness study for coupling Gaussianization | — | 🧱 | **GAP** |
| 5.22 | Residual coupling & Lipschitz constraints | — | 🌉 | **GAP** — preview of 9.F |

---

## Part 6 — Continuous-time Gaussianization *(bridge)*

A continuous-time bijector is a flow ODE $\dot x = v_\theta(x, t)$ whose pushforward at $t=T$ matches $\mathcal{N}(0,I)$. This is the "infinite-depth coupling" limit and the bridge to diffusion models (9.E).

### 6.A — FFJORD

**Key equations / models:**
- Instantaneous CoV: $\partial_t \log p_t(x) = -\mathrm{tr}(\nabla_x v_\theta(x, t))$
- Log-det as a line integral: $\log|\det J_T(x)| = -\int_0^T \mathrm{tr}(\nabla_x v_\theta(x_t, t))\,dt$
- Free-form $v_\theta$ — no architectural invertibility constraint

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 6.1 | FFJORD 2D — continuous-time Gaussianization | F `05_ffjord_2d` | 🧱 | |

### 6.B — Hutchinson trace estimator

**Key equations / models:**
- Stochastic trace: $\mathrm{tr}(A) \approx \mathbb{E}_{z\sim\mathrm{Rademacher}}[z^\top A z]$
- Replaces $O(d)$ Jacobian-vector products with $O(1)$ per training step
- Variance: $\mathrm{Var}(z^\top A z) = 2\|A\|_F^2 - 2\sum_i A_{ii}^2$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 6.2 | Hutchinson trace for FFJORD log-det | — | 🧱 | **GAP** — pairs with `xref:GP#1.19` (SLQ) |

### 6.C — Matrix-exponential / linear neural flows

**Key equations / models:**
- Linear neural flow: $\dot x = Ax$ → $x(T) = \exp(AT)\,x(0)$
- Closed-form log-det: $\log|\det \exp(AT)| = T\,\mathrm{tr}(A)$
- Useful as a building block for non-linear FFJORD

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 6.3 | Matrix-exponential neural flow | F `06_matrix_exponential_neural_flow` | 🌉 | |

### 6.D — Latent ODEs

**Key equations / models:**
- Encode $\to$ latent ODE in $z$-space $\to$ decode
- Closed-form Gaussianization on the latent state $z(0)$ if encoder is invertible

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 6.4 | Latent ODE on spirals | F `09_latent_ode_spirals` | 🌉 | also 11.E |

---

## Part 7 — Conditional Gaussianization

Make every parameter of the flow depend on a context $y$ — gives a tractable conditional density $p(x \mid y)$.

### 7.A — Conditioner zoo for marginals & couplings

**Key equations / models:**
- Conditional marginal: $z_i = \Phi^{-1}\!\big(F_{\theta(y)}(x_i)\big)$
- Conditional coupling: bijector parameters depend on both $x_A$ and $y$
- Conditional rotation: $Q(y)$ from a conditioner producing skew-symmetric or Householder vectors

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 7.1 | Building a conditional Gaussianization flow — three patterns | F `08_conditional_flow_three_ways` | 🧱 | (also 5.17) |
| 7.2 | Conditional marginals — when to make the CDF $y$-dependent | — | 🧱 | **GAP** |

### 7.B — Conditional density estimation

**Key equations / models:**
- $\log p(x\mid y) = \log p_Z(T_\theta(x\mid y)) + \log|\det J_{T_\theta}(x\mid y)|$
- ELBO if conditioner produces variational parameters

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 7.3 | Conditional density estimation benchmarks | — | 🧱 | **GAP** |

### 7.C — Three-pattern conditional flow

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 7.4 | Three-pattern conditional flow tutorial | F `08_conditional_flow_three_ways` | 🧱 | |

### 7.D — Conditioning for inverse problems

**Key equations / models:**
- Conditional flow as posterior $p(x\mid y)$ for inverse problems $y = Ax + \eta$
- Amortised inference — train once, sample posterior for any $y$ in one pass

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 7.5 | Conditional flow as an amortised posterior — toy inverse problem | — | 🌉 | **GAP** — feeds 16.E |

---

## Part 8 — SurVAE: Surjections & Stochastic Transforms

Generalise bijections to *surjections* (one direction is many-to-one) and *stochastic transforms* (one direction adds randomness) while keeping a tractable density / ELBO. Companion proof in `survae_flows_proof.md`.

### 8.A — Bijection / surjection / stochastic taxonomy

**Key equations / models:**
- Bijection: $|\det J|$ in both directions
- Surjection: forward deterministic, inverse stochastic → ELBO term $\log q(x\mid z)$
- Stochastic: both directions stochastic

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 8.1 | SurVAE taxonomy with worked examples | R `survae_flows_proof.md` | 🧱 | |
| 8.2 | Surjective Gaussianization — formal density / ELBO | — | 🌉 | **GAP** |

### 8.B — Slicing & augmentation surjections

**Key equations / models:**
- Slicing: $z = x_A$, drop $x_B$ — useful for dimension reduction
- Augmentation: $z = (x, u)$ for auxiliary $u\sim q(u\mid x)$
- Dequantisation: integer $x \to x + u$, $u\sim\mathrm{Uniform}[0,1)$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 8.3 | Dequantisation surjection for discrete inputs | — | 🧱 | **GAP** (also 3.10) |
| 8.4 | Augmentation surjection — auxiliary $u$ | — | 🌉 | **GAP** — feeds 9.B |

### 8.C — Stochastic transforms & ELBO

**Key equations / models:**
- ELBO: $\log p(x) \geq \mathbb{E}_{q(z\mid x)}[\log p(z) + \log p(x\mid z)] + H[q(z\mid x)]$
- VAE as a single-step stochastic flow
- Connect to 9.D (stochastic NF) and 9.E (diffusion)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 8.5 | Stochastic transforms — VAE-as-flow framing | — | 🌉 | **GAP** |

---

## Part 9 — Relaxed-Bijectivity & Non-Invertible Flows

Drop strict invertibility for expressiveness or generality. Each sub-part keeps the **density / ELBO / score** machinery from going opaque.

### 9.A — Injective / lossy Gaussianization

**Key equations / models:**
- Data assumed to live on an $m$-dim submanifold of $\mathbb{R}^d$ with $m\leq d$
- *Injective* decoder $g:\mathbb{R}^m\to\mathbb{R}^d$ — image is the data manifold; one-to-one with no information loss
- *Lossy* / surjective encoder $E:\mathbb{R}^d\to\mathbb{R}^m$ — left-inverse of $g$ on-manifold, many-to-one off-manifold
- On-manifold density via $|\det J_g^\top J_g|^{1/2}$ (Brehmer & Cranmer 2020, M-Flow)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 9.1 | Injective decoder + lossy encoder — manifold flows on data of intrinsic dimension $m\leq d$ | — | 🔬 | **GAP** |

### 9.B — Augmented / lifted flows

**Key equations / models:**
- ANF (Huang 2020) / VFlow (Chen 2020): lift $x\to(x,u)$ with $u\sim q$, Gaussianize the joint
- Recover marginal via stochastic inverse

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 9.2 | Augmented Gaussianization via auxiliary lift | — | 🌉 | **GAP** |

### 9.C — Continuously-indexed flows (CIF)

**Key equations / models:**
- Index the bijector by a latent $u$: $T_u(x)$, marginalise / ELBO over $u$
- Cornish et al. 2020 — relaxes topological constraints of bijectors

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 9.3 | Continuously-indexed Gaussianization | — | 🔬 | **GAP** |

### 9.D — Stochastic normalizing flows

**Key equations / models:**
- Interleave deterministic bijectors with MCMC / Langevin kernels
- Wu, Köhler & Noé 2020 — tractable importance-weighted estimator

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 9.4 | Stochastic normalizing Gaussianization | — | 🔬 | **GAP** |

### 9.E — Diffusion as continuous stochastic Gaussianization

The forward diffusion process *is* a Gaussianization: it transports any data distribution to $\mathcal{N}(0,I)$ along a continuous noise schedule. The probability-flow ODE is its deterministic invertible counterpart and sits in the same family as Part 6.

**Key equations / models:**
- Forward VP SDE: $dx_t = -\tfrac{1}{2}\beta(t)x_t\,dt + \sqrt{\beta(t)}\,dW_t$, marginals $x_t\sim\mathcal{N}(\alpha(t)x_0,\sigma^2(t)I)$, $x_T\approx\mathcal{N}(0,I)$
- Reverse SDE: $dx_t = \big[-\tfrac{1}{2}\beta(t)x_t - \beta(t)\nabla\log p_t(x_t)\big]\,dt + \sqrt{\beta(t)}\,d\bar W_t$
- Probability-flow ODE (deterministic, invertible — a continuous Gaussianization): $\dot x_t = -\tfrac{1}{2}\beta(t)x_t - \tfrac{1}{2}\beta(t)\nabla\log p_t(x_t)$
- Score matching: $\mathbb{E}_{t,x_t}\big[\lambda(t)\,\|s_\theta(x_t,t) - \nabla\log p_t(x_t)\|^2\big]$
- Flow-matching / rectified-flow loss: $\mathbb{E}_{t,x_0,x_1}\big[\|v_\theta(x_t,t) - (x_1-x_0)\|^2\big]$, $x_0\sim p_\text{data}$, $x_1\sim\mathcal{N}(0,I)$
- $\sigma\to 0$ limit: recovers classical deterministic Gaussianization

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 9.5 | Forward & reverse SDE — diffusion as stochastic data-to-Gaussian transport | — | 🌉 | **GAP** — frame as the stochastic-NF cousin of Part 6 |
| 9.6 | Probability-flow ODE — the invertible Gaussianization hidden inside a diffusion model | — | 🌉 | **GAP** — bridge to 6.A |
| 9.7 | Flow Matching & Rectified Flow — learning the Gaussianization vector field without simulation | — | 🔬 | **GAP** — Lipman 2023, Liu 2022 |
| 9.8 | $\sigma\to 0$ limit & one-step distillation — recovering deterministic Gaussianization from diffusion | — | 🔬 | **GAP** — connects 9.E ↔ 9.G |

### 9.F — Residual / implicit flows

**Key equations / models:**
- Residual: $T(x) = x + g_\theta(x)$ with $\mathrm{Lip}(g)<1$ (Behrmann 2019, Chen 2019)
- Inverse by Banach fixed-point iteration
- Log-det via Hutchinson + power series of $\log\det(I+J_g)$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 9.9 | Residual flows — Lipschitz coupling | — | 🔬 | **GAP** |

### 9.G — One-shot / Trumpet-style Gaussianizers

**Key equations / models:**
- Single feed-forward encoder $E_\theta : x \to z$, trained with NLL or matching loss
- Trades NLL exactness for amortisation speed

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 9.10 | One-shot Gaussianization (Trumpet-style) | — | 🔬 | **GAP** |

### 9.H — When non-invertibility helps

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 9.11 | Decision recap — which downstream uses (sampling / density / IT / posterior) survive relaxed bijectivity | — | 🧱 | **GAP** — pedagogical |

---

## Part 10 — Non-Euclidean Gaussianization

Push the target back to $\mathcal{N}(0,I)$ when the data lives on a manifold (circle, torus, sphere, Lie group).

### 10.A — Circle / torus

**Key equations / models:**
- von Mises CDF as periodic Gaussianization in 1D
- Torus = $(S^1)^d$; per-axis circular flow + orthogonal mixer

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 10.1 | Circular Gaussianization on the torus | F `torus_circular_flow` | 🧱 | |

### 10.B — Sphere

**Key equations / models:**
- Lambert / stereographic chart for $S^2$ → Euclidean Gaussianization in chart
- Equivariant constructions for $SO(3)$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 10.2 | Spherical Gaussianization (global) | F `07_global_flow_sphere` | 🌉 | pairs with `xref:GP#7.18` (VISH) |

### 10.C — Lie groups & Riemannian manifolds

**Key equations / models:**
- Exponential map: $\exp_g : T_g M \to M$, Gaussianize on tangent space
- Riemannian flow models (Rezende 2020, Lou 2020)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 10.3 | Riemannian Gaussianization via tangent-space chart | — | 🔬 | **GAP** |

---

## Part 11 — Time-Series Gaussianization

Sequences have *temporal* structure: past conditions future. Choose conditioners accordingly (see 5.C and 11.B).

### 11.A — Per-timestep marginal Gaussianization

**Key equations / models:**
- Independent per-timestep CDF $F_t$ — works when distribution is stationary
- Sliding-window CDF for non-stationary series

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 11.1 | Per-timestep marginal Gaussianization | — | 🧱 | **GAP** |
| 11.2 | Sliding-window CDF for non-stationary series | — | 🧱 | **GAP** |

### 11.B — Autoregressive flows for sequences

**Key equations / models:**
- Autoregressive factorisation: $p(x_{1:T}) = \prod_t p(x_t \mid x_{<t})$
- Conditional Gaussianization: $z_t = \Phi^{-1}\!\big(F_{\theta(x_{<t})}(x_t)\big)$
- MAF / IAF as the canonical NF cousin (bridge to 11.A)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 11.3 | Autoregressive Gaussianization of sequences | — | 🌉 | **GAP** — MAF / IAF as the underlying flow |
| 11.4 | Choosing the temporal conditioner — RNN / Transformer / Mamba / TCN | — | 🌉 | **GAP** — leans on 5.12 |

### 11.C — Conditioning on past context

**Key equations / models:**
- Causal mask on Transformer-conditioner
- Stateful RNN conditioner with hidden $h_t$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 11.5 | Causal conditioner mechanics for time-series Gaussianization | — | 🧱 | **GAP** |

### 11.D — AR(p) in Gaussianized state

**Key equations / models:**
- Gaussianize each $x_t \to z_t$ via stationary marginal flow
- Fit linear AR(p): $z_t = \sum_{k=1}^p \phi_k z_{t-k} + \epsilon_t$ in latent space
- Closed-form likelihood + forecasting

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 11.6 | AR(p) on a Gaussianized series — closed-form forecasting | — | 🌉 | **GAP** — bridges to 17.A (normalising Kalman) |

### 11.E — Latent ODEs for irregular series

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 11.7 | Latent ODE Gaussianization for irregular time-series | F `09_latent_ode_spirals` | 🌉 | (also 6.D) |

### 11.F — Long-range / hierarchical temporal couplings

**Key equations / models:**
- Dilated coupling masks across time — exponentially-growing receptive field
- Hierarchical encoders (U-Net style) over the time axis

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 11.8 | Long-range coupling masks for time-series | — | 🌉 | **GAP** |

### 11.G — Multiscale temporal flows

**Key equations / models:**
- Dyadic Squeeze across time: $T \to T/2$ with channel doubling
- Wavelet temporal bijector (Haar in time)
- Multi-resolution AR flows

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 11.9 | Dyadic time-Squeeze & Haar-time wavelets | — | 🔬 | **GAP** |

### 11.H — Time-series anomaly & changepoint detection

**Key equations / models:**
- Anomaly score: $-\log p(x_t \mid x_{<t})$ from a conditional flow
- Changepoint: drift in running-mean of $\|z_t\|^2$ in Gaussianized space

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 11.10 | Anomaly detection via log-prob from a temporal Gaussianization flow | — | 🌉 | **GAP** — pairs with `xref:GP#8.31` |
| 11.11 | Changepoint detection in latent space | — | 🌉 | **GAP** |

---

## Part 12 — Spatial / Image Gaussianization

Images have spatial structure: locality + translation symmetry. Multiscale composition is the standard scaling pattern.

### 12.A — Multiscale Squeeze / unsqueeze

**Key equations / models:**
- Squeeze: $(H, W, C) \to (H/2, W/2, 4C)$ — locality-preserving rearrangement
- Multi-scale architecture: factor out half the channels at each scale, Gaussianize, continue
- Glow-style scale loop

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 12.1 | Squeeze / unsqueeze as a multiscale Gaussianization step | — | 🧱 | **GAP** |

### 12.B — Invertible 1×1 conv & Haar wavelet

**Key equations / models:**
- 1×1 conv (LU): per-pixel channel mixing (see 2.D)
- Haar wavelet: orthogonal multiresolution analysis (averages + differences)
- Both interpretable as Part 2 mixers applied across image structure

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 12.2 | Invertible 1×1 conv in image flows | — | 🧱 | **GAP** |
| 12.3 | Haar wavelet bijector | — | 🧱 | **GAP** — pairs with `xref:GP#1.4` (Toeplitz / FFT) |

### 12.C — Patch-based image flows

**Key equations / models:**
- Train Gaussianization flow on $p\times p$ patches; tile over the image
- Use as a learned image prior (feeds Part 16 PnP)
- Stationarity assumption + receptive-field analysis

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 12.4 | Patch-based Gaussianization flow | — | 🌉 | **GAP** — direct dependency of 16.D |
| 12.5 | Patch stationarity diagnostics & overlap-add inference | — | 🌉 | **GAP** |

### 12.D — Equivariant flows

**Key equations / models:**
- $G$-equivariant flow: $T(g\cdot x) = g\cdot T(x)$ for $g\in G$
- Translation-equivariance via CNN couplings
- Rotation-equivariance via steerable filters

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 12.6 | Translation-equivariant image Gaussianization | — | 🌉 | **GAP** |
| 12.7 | Rotation-equivariant flows for natural images | — | 🔬 | **GAP** — pairs with 5.14 |

### 12.E — Spatial random-field Gaussianization (GRF prior)

**Key equations / models:**
- Treat a spatial field $f(s)$ on a grid as one observation; Gaussianize → Gaussian-process-like latent
- Feeds spatial-extremes margins (18.C) and DA (17)
- Bridges to `xref:GP#4.B` (Toeplitz / Kronecker GRFs)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 12.8 | Spatial random-field Gaussianization | — | 🔬 | **GAP** — 🔁 |

### 12.F — Image-rotation diagnostics

**Key equations / models:**
- Rotation choices for image-level RBIG: PCA / random / patch-PCA / learned
- Visual inspection of recovered modes after each rotation

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 12.9 | Image rotations in RBIG | B `12_image_rotations` | 🌉 | |

### 12.G — Glow end-to-end

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 12.10 | Glow architecture end-to-end on natural images | — | 🌉 | **GAP** — composes 2.D, 2.E, 5.A, 12.A, 12.B |

---

## Part 13 — Spatiotemporal Fields & Videos

Lat × lon × time tensors and videos. Inherits machinery from Parts 11 and 12.

### 13.A — Separable space×time coupling

**Key equations / models:**
- Factored bijector: alternate spatial-only and temporal-only coupling blocks
- Receptive field grows in both axes via stacking

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 13.1 | Separable space×time coupling for fields | — | 🌉 | **GAP** |

### 13.B — Frame-conditioned video flows

**Key equations / models:**
- $p(x_{1:T}^\text{video}) = \prod_t p(x_t \mid x_{<t})$ with image-flow conditioned on past frames
- Conditioner is a temporal CNN / Transformer over frame features (5.C)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 13.2 | Frame-conditioned video Gaussianization | — | 🔬 | **GAP** |

### 13.C — Spatiotemporal RBIG (lat × lon × time)

**Key equations / models:**
- Apply RBIG to flattened lat × lon × time tensors with structured rotations
- Per-axis vs joint rotation choices

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 13.3 | RBIG on climate tensors (lat × lon × time) | — | 🔬 | **GAP** — 🔁 |

### 13.D — Latent ODEs for field dynamics

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 13.4 | Latent ODE Gaussianization for field dynamics | — | 🔬 | **GAP** — composes 6.D + 12.E |

### 13.E — Equivariant spatiotemporal flows

**Key equations / models:**
- Translation-in-space + translation-in-time equivariance
- Lifts to $SE(2)\times\mathbb{R}_t$ via steerable / equivariant conditioners (5.14)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 13.5 | Equivariant spatiotemporal Gaussianization | — | 🔬 | **GAP** |

### 13.F — Climate-field scale-up

**Key equations / models:**
- Sharded RBIG / coupling across patches × time-windows
- Streaming statistics for very large fields

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 13.6 | Scaling Gaussianization to global climate fields | — | 🔬 | **GAP** |

### 13.G — Multiscale spatiotemporal

**Key equations / models:**
- Joint Squeeze in $(H, W, T)$ — dyadic in all three axes
- 3D Haar wavelets / lifted-wavelet bijectors
- Mixed-resolution video flows

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 13.7 | 3D Haar wavelets for spatiotemporal Gaussianization | — | 🔬 | **GAP** — composes 11.G + 12.A |
| 13.8 | Video Glow — multiscale spatiotemporal architecture | — | 🔬 | **GAP** |

---

## Part 14 — Information-Theoretic Estimation

A killer downstream use of Gaussianization: once $T_\#p = \mathcal{N}(0,I)$, IT functionals decompose trivially.

### 14.A — Entropy & negentropy from Gaussianized residual

**Key equations / models:**
- Differential entropy: $H(p) = H(\mathcal{N}(0,\Sigma_p)) - J(p)$
- Negentropy: $J(p) \approx \sum_k \Delta J_k$ across RBIG iterations
- Closed-form Gaussian entropy: $\tfrac{1}{2}\log|2\pi e\,\Sigma|$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 14.1 | Entropy & negentropy from RBIG | B `06_information_theory` | 🧱 | |

### 14.B — Mutual information & total correlation

**Key equations / models:**
- $I(X;Y) = H(X) + H(Y) - H(X,Y)$
- Total correlation: $TC(X) = \sum_i H(X_i) - H(X)$, computable directly from the RBIG residual

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 14.2 | Mutual information & total correlation via Gaussianization | — | 🧱 | **GAP** — builds on 14.1 |

### 14.C — KL between empirical distributions

**Key equations / models:**
- Build $T_p$, $T_q$ from samples; estimate $\mathrm{KL}(p\Vert q) = \mathbb{E}_p[\log p - \log q]$ via change-of-variables

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 14.3 | Empirical KL via dual Gaussianization | — | 🧱 | **GAP** |

### 14.D — Dependence measures

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 14.4 | 1D dependence — RBIG-MI vs HSIC / MMD | B `09_dependence_1d` | 🧱 | |
| 14.5 | 2D dependence cases | B `10_dependence_2d` | 🧱 | |

### 14.E — Real-data IT pipelines

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 14.6 | Real-world IT estimation pipelines | B `11_real_world_it` | 🔬 | |
| 14.7 | Dimensionality reduction via Gaussianization | B `13_dimensionality_reduction` | 🔬 | |

### 14.F — Bias-variance & sample complexity

**Key equations / models:**
- Asymptotic bias of plug-in IT estimators
- Jackknife / bootstrap for IT confidence intervals
- Effective-sample-size for RBIG-MI in high $d$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 14.8 | Bias-variance of RBIG-based IT estimators | — | 🔬 | **GAP** |

---

## Part 15 — Fair Learning with Frozen Gaussianization Flows 🔬

Pretrain a flow on a dataset, **freeze its weights**, then use the Gaussianised representation as a *differentiable independence loss* inside any predictor. Active research — drops into [`fairkl.models.FairModelWrapper`](https://github.com/jejjohnson/keras-fairkl) as a replacement for `CKALoss`.

### 15.A — Frozen flow as differentiable independence loss

**Key equations / models:**
- Pretrain $T_\theta$ s.t. $T_\theta(x)\sim\mathcal{N}(0,I)$, freeze weights
- Use $T_\theta$ to extract Gaussianised features. Covariance-based independence measures on $(T(x), q)$ become a **tractable proxy** for independence; the equivalence "zero covariance ⇔ independence" requires the *joint* to be Gaussian, which marginal Gaussianisation of $x$ alone does not guarantee (and breaks especially for categorical $q$). In practice the proxy works well when marginal-shape mismatch dominates the dependence signal.

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 15.1 | Why Gaussianisation makes independence "easy" | — | 🧱 | **GAP** — pedagogical anchor for Part 15 |
| 15.2 | Pretrain + freeze workflow | K `05_fair_gauss_pretrain` | 🧱 | api: `pretrain.fit_and_freeze`, `freeze.freeze_flow` |

### 15.B — G-XCOV vs G-HSIC vs CKA

**Key equations / models:**
- **G-XCOV**: cross-covariance in Gaussianised space $\|\mathrm{Cov}(T(x), q)\|_F^2$
- **G-HSIC**: kernel HSIC computed on $T(x)$ instead of $x$
- CKA: cosine-normalised cross-covariance (baseline)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 15.3 | G-XCOV: cross-covariance after Gaussianization | — | 🧱 | api: `GaussianizedXCovLoss` |
| 15.4 | G-HSIC: HSIC in Gaussianised features | — | 🌉 | api: `GaussianizedHSICLoss` |
| 15.5 | CKA baseline comparison | — | 🌉 | **GAP** |

### 15.C — Pretrain + freeze workflow

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 15.6 | Diagnostics — NLL curve, QQ-plots, skew/kurt, freeze + invertibility | K `05_fair_gauss_pretrain` | 🧱 | |

### 15.D — Synthetic fairness sweeps & Pareto curves

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 15.7 | Synthetic regression: G-XCOV vs CKA Pareto sweep over fairness weight $\mu$ | K `06_fair_gauss_synthetic` | 🔬 | |

### 15.E — Adult census case study

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 15.8 | UCI Adult — AUC vs DP / EO differences | K `07_fair_gauss_adult` | 🔬 | |

### 15.F — Drop-in with `FairModelWrapper`

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 15.9 | Integrating G-XCOV / G-HSIC with `fairkl.FairModelWrapper` | R `docs/fair_gaussianization_experiment.md` | 🌉 | engineering doc |

### 15.G — Open research directions

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 15.10 | Follow-up experiment design (approaches A–G) | R `docs/fair_gaussianization_followups.md` | 🔬 | active research |

---

## Part 16 — Plug-and-Play Priors with Gaussianization

The central pedagogical hook: **the proximal operator of a Gaussianized prior is closed-form in latent space** — $z \mapsto z/(1 + 1/\tau)$ for the standard Gaussian — so PnP-ADMM / HQS schemes with a Gaussianization prior have no inner solver.

### 16.A — PnP recap (denoiser-as-prior)

**Key equations / models:**
- Inverse problem: $\hat x = \arg\min_x \tfrac{1}{2}\|Ax - y\|^2 + \lambda R(x)$
- PnP: replace $\mathrm{prox}_{\lambda R}$ with a generic denoiser
- Convergence (Sun 2019, Ryu 2019) under non-expansive denoisers

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 16.1 | PnP framework recap — denoiser-as-prior | — | 🧱 | **GAP** |

### 16.B — Closed-form prox in Gaussianized latent space

**Key equations / models:**
- Gaussianized prior log-density: $\log p_R(x) = \log\mathcal{N}(T(x); 0, I) + \log|\det J_T(x)|$
- Score: $\nabla\log p_R(x) = -J_T(x)^\top T(x) + \nabla\log|\det J_T(x)|$
- **Variable split**: rewrite the inverse problem in latent coordinates $z = T(x)$ via HQS / ADMM so the regulariser becomes the standard-Gaussian negative log-prior $\tfrac{1}{2}\|z\|^2$
- **Closed-form prox in *latent* coordinates**: $\mathrm{prox}_{\|z\|^2/(2\tau)}(z) = z/(1 + 1/\tau)$
- Caveat: this is the prox with respect to the *latent-space* Euclidean metric, **not** the Euclidean prox of the induced data-space prior $-\log p_R(x)$ — those coincide only when $T$ is linear / isometric. With general $T$, the data-space prox $\arg\min_x\,\tfrac{1}{2\tau}\|x-x_0\|^2 - \log p_R(x)$ has no closed form, which is precisely why the latent-split formulation is the practical recipe (cf. Asim 2020, Whang 2021).
- Pull-back update: $x_+ = T^{-1}(z_+/(1+1/\tau))$ where $z_+ = T(x)$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 16.2 | Closed-form prox in Gaussianized space — the central trick | — | 🌉 | **GAP** — pedagogical anchor for Part 16 |
| 16.3 | Score of a Gaussianization prior — algorithmic derivation | — | 🧱 | **GAP** |

### 16.C — HQS / ADMM with Gaussianization

**Key equations / models:**
- HQS: alternate $x$-update (data fit) and $z$-update (prox in $T$-space)
- ADMM: add dual variable $u$ for tighter coupling

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 16.4 | Gaussianization-HQS for linear inverse problems | — | 🔬 | **GAP** |
| 16.5 | Gaussianization-ADMM with dual update | — | 🔬 | **GAP** |

### 16.D — Patch-based PnP for images

**Key equations / models:**
- Train Gaussianization flow on patches (12.C)
- Apply per-patch prox, aggregate via overlap-add

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 16.6 | Patch-based Gaussianization PnP | — | 🔬 | **GAP** — direct dep on 12.4 |
| 16.7 | Overlap-add patch aggregation & boundary handling | — | 🔬 | **GAP** |

### 16.E — Linear inverse problems

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 16.8 | Deblurring with Gaussianization prior | — | 🔬 | **GAP** |
| 16.9 | Super-resolution | — | 🔬 | **GAP** |
| 16.10 | Inpainting | — | 🔬 | **GAP** |
| 16.11 | Compressed sensing | — | 🔬 | **GAP** |

### 16.F — Comparison with score / diffusion PnP

**Key equations / models:**
- Score-based PnP uses learned $\nabla\log p_t(x)$ at multiple noise levels
- Gaussianization-PnP uses one exact, deterministic, closed-form prox
- Trade-off table: cost, expressiveness, training pipeline

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 16.12 | Gaussianization-PnP vs score / diffusion PnP — head-to-head | — | 🔬 | **GAP** — pairs with 9.E |

---

## Part 17 — Filtering & Data Assimilation with Gaussianization

Gaussianize a non-Gaussian state / observation / QoI, run a *closed-form* Kalman recursion in latent space, then invert.

### 17.A — Normalizing Kalman filter (closed-form)

**Key equations / models:**
- Gaussianized state: $z_t = T(x_t)$, $T$ a learned bijector with $T(x_t) \sim \mathcal{N}(0, I)$ marginally
- Linear-Gaussian model in $z$-space: $z_{t+1} = A\,z_t + w_t$, $\tilde y_t = H\,z_t + v_t$
- Standard Kalman in $z$-space + invert: $\hat x_{t\mid t} = T^{-1}(\hat z_{t\mid t})$
- Posterior log-density: $\log p(x_t\mid y_{1:t}) = \log\mathcal{N}(T(x_t);\,\hat z_{t\mid t}, P_{t\mid t}) + \log|J_T(x_t)|$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 17.1 | Normalizing Kalman filter — closed form via state Gaussianization | — | 🔬 | **GAP** — pairs with `xref:GP#8.1` |
| 17.2 | RTS smoother in Gaussianized space | — | 🔬 | **GAP** — pairs with `xref:GP#8.1` |

### 17.B — State vs observation vs QoI Gaussianization

**Key equations / models:**
- *State*: $T_x: x \to z_x$, dynamics learned in $z_x$ space (most general, hardest)
- *Observation*: $T_y: y \to z_y$, observation operator linearised post-transform (cheapest)
- *QoI*: $T_q: Q(x) \to z_q$ applied to a derived quantity (post-hoc summary; trivially invertible)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 17.3 | Observation Gaussianization for non-Gaussian likelihoods | — | 🌉 | **GAP** |
| 17.4 | QoI Gaussianization for posterior summaries | — | 🧱 | **GAP** |
| 17.5 | State vs obs vs QoI — when each wins (decision tree) | — | 🧱 | **GAP** — pedagogical |

### 17.C — Gaussianized ensemble Kalman filter

**Key equations / models:**
- EnKF analysis in $z$-space: $z^a = z^f + P_e^z H^\top(HP_e^zH^\top + R)^{-1}(\tilde y - Hz^f)$
- Sample covariance $P_e^z$ from ensemble of Gaussianised states
- Push back via $T^{-1}$ to obtain $x^a$ ensemble

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 17.6 | Gaussianized EnKF | — | 🔬 | **GAP** — pairs with `xref:GP#8.23` |

### 17.D — Non-Gaussian likelihoods in DA

**Key equations / models:**
- Observation Gaussianization makes a Poisson / Bernoulli / heavy-tailed observation look Gaussian
- Closed-form update once $\tilde y_t = T_y(y_t)$ is Gaussian

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 17.7 | Gaussianization for non-Gaussian observation likelihoods | — | 🔬 | **GAP** — bridge to `xref:GP#8.29-31` |

### 17.E — Sequential Bayesian updates in latent space

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 17.8 | Sequential Bayesian update in latent space — natural-parameter addition | — | 🧱 | **GAP** — pairs with `xref:GP#0.6` |

### 17.F — Comparison with EKF / UKF / particle filter

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 17.9 | Gaussianized filter vs EKF / UKF / particle filter — accuracy + cost benchmarks | — | 🌉 | **GAP** |

---

## Part 18 — Geoscience Case Studies

Applied stories. Each entry composes pieces from earlier parts; flagged 🔁 where it cross-references the GP master list.

### 18.A — Quantile mapping / bias correction

**Key equations / models:**
- Bias correction: $x_\text{cal}(t) = F_\text{obs}^{-1}(F_\text{model}(x_\text{model}(t)))$ — exactly 1D Gaussianization round-trip
- Per-season / per-region conditional quantile mapping

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 18.1 | Quantile mapping for climate-model bias correction | — | 🔬 | **GAP** — direct application of 1.A |
| 18.2 | Statistical downscaling via conditional Gaussianization | — | 🔬 | **GAP** — leans on Part 7 |

### 18.B — Climate-field anomaly detection

**Key equations / models:**
- Flow log-prob as anomaly score on climate fields
- Threshold via empirical $\alpha$-quantile of training log-probs

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 18.3 | Climate-field anomaly detection via flow log-prob | — | 🔬 | **GAP** |

### 18.C — Spatial extremes (GEV / Gumbel margins) 🔁

**Key equations / models:**
- Marginal Gaussianization: GEV margin → Gaussian via $\Phi^{-1}(F_\text{GEV}(y))$
- Spatial dependence captured by rotation / coupling above the margins
- Combine with GP on Gaussianised space for `xref:GP#14.1-4`

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 18.4 | GEV-margin Gaussianization + spatial flow | — | 🔬 | **GAP** — 🔁 `xref:GP#14.A` |
| 18.5 | Gumbel-margin variant | — | 🔬 | **GAP** |

### 18.D — Ocean SST / sea-level extremes

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 18.6 | SST anomaly Gaussianization | — | 🔬 | **GAP** |
| 18.7 | Sea-level extreme tails | — | 🔬 | **GAP** — 🔁 `xref:GP#14.8` |

### 18.E — Precipitation Gaussianization

**Key equations / models:**
- Zero-inflation: censored Gaussian / mixture of point-mass at 0 + continuous tail
- Heavy-tail handling via GPD support extension (3.E)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 18.8 | Precipitation Gaussianization with zero-inflation & heavy tails | — | 🔬 | **GAP** |

### 18.F — Wind / atmospheric tracers

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 18.9 | Wind-vector Gaussianization (angular + magnitude) | — | 🔬 | **GAP** |
| 18.10 | Atmospheric tracer concentration Gaussianization | — | 🔬 | **GAP** |

### 18.G — Satellite-image emulation

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 18.11 | Satellite-image emulation via patch-Gaussianization | — | 🔬 | **GAP** — composes 12.C + 16 |

### 18.H — Climate-data assimilation

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 18.12 | Climate DA with Gaussianized observation operator | — | 🔬 | **GAP** — 🔁 `xref:GP#14.11` |
| 18.13 | Gaussianized EnKF for ocean reanalysis | — | 🔬 | **GAP** — composes 17.C + 13 |

---

## Part 19 — Probabilistic-Programming Integration

### 19.A — `FlowDist` in NumPyro

**Key equations / models:**
- `numpyro.sample("x", FlowDist(flow))` — a flow becomes a regular distribution
- Use as likelihood, prior, or guide

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 19.1 | `FlowDist` — wrapping a Gaussianization flow as a NumPyro distribution | — | 🧱 | **GAP** |

### 19.B — Flows as priors in BHMs

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 19.2 | Flow priors in Bayesian hierarchical models | — | 🌉 | **GAP** — pairs with `xref:GP#11.4` |

### 19.C — Flows as guides in SVI

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 19.3 | Flow guides for SVI — beyond mean-field | — | 🌉 | **GAP** — pairs with `xref:GP#6.14` |

### 19.D — pyrox integration patterns

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 19.4 | Three-pattern flow integration with pyrox | — | 🌉 | **GAP** — leans on `xref:GP#11.3` |

---

## Part 20 — Metrics, Calibration, Diagnostics

### 20.A — NLL / bits-per-dim

**Key equations / models:**
- NLL: $-\tfrac{1}{n}\sum_i \log p_X(x_i)$
- Bits-per-dim: $\mathrm{NLL}/(d\,\log 2)$, the standard image-flow benchmark

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 20.1 | NLL & bits-per-dim — what to log and how to compare | — | 🧱 | **GAP** |

### 20.B — QQ / PIT / coverage for flows

**Key equations / models:**
- PIT: $u_i = F(x_i)$ uniform under correctly-specified $F$
- Coverage at $1-\alpha$: empirical fraction of $x_i$ inside the predicted central interval
- QQ plots in Gaussianised space — per-axis diagnostic

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 20.2 | PIT, coverage, and QQ diagnostics for flows | — | 🧱 | **GAP** — pairs with `xref:GP#3.5` |

### 20.C — Sample quality

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 20.3 | Sample-quality metrics — FID-style, MMD, energy distance | — | 🧱 | **GAP** |

### 20.D — Roundtrip invertibility & numerical tolerance

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 20.4 | Roundtrip invertibility tests for deep stacks | — | 🧱 | **GAP** — pairs with 0.9 |

### 20.E — Cross-validation for IT estimators

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 20.5 | CV / bootstrap for IT estimator confidence intervals | — | 🧱 | **GAP** — pairs with 14.F |

---

## Summary of duplications to reconcile

| Topic | Locations | Suggestion |
|---|---|---|
| 2D Gaussianization flow | K `01_gaussianization_2d`, F `01_gaussianization_flow_2d` | Keep K as Keras canonical; F as FlowJax canonical; both surface from Part 4 |
| 2D coupling flow | K `02_coupling_flow_2d`, F `02_coupling_flow_2d` | Same — keep both, one per backend |
| Coupling ↔ diagonal equivalence | K `04_coupling_equivalence`, F `04_coupling_equivalence` | Keep both; K = pedagogical / Keras, F = FlowJax replicate |
| Rotation choices | B `08_rotation_choices` | Single canonical; referenced from 2.A and 3.C |

## Proposed final homes (high-level)

- **rbig/docs/notebooks/** → Parts 1, 2 (rotations), 3 (RBIG), 12.F (image rotations), 14 (IT estimation), case-study foundations
- **gauss_flows/docs/notebooks/** → Parts 4–7 (parametric / coupling / continuous / conditional), 10 (non-Euclidean), 11.E (latent ODE), bulk of 12–13 (image + spatiotemporal)
- **research_notebook gauss_keras** → Parts 4–5 (Keras canonical), Part 8 (SurVAE proof), Part 15 (fair learning)
- **research_notebook elsewhere** → Parts 16–18 (PnP, filtering, geoscience case studies), Part 19–20 (integration + metrics) once they land
