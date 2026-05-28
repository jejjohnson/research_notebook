# Bayesian Models & Neural Networks Tutorial Master List

A reconciled, exhaustive curriculum for **Bayesian linear models, parametric regression, parametric classification, and Bayesian neural networks**. The progression mirrors the textbook arc: Bayesian linear regression → basis functions → random / spectral features → shallow MLPs → deep BNNs. Each block lays out its own likelihood / constraint zoo and its own inference zoo so the curriculum can be entered at any depth.

Companion lists:
- Pure-GP machinery → [`../gaussian_processes/TUTORIAL_MASTER_LIST.md`](../gaussian_processes/TUTORIAL_MASTER_LIST.md)
- Neural fields / INRs / NeRF → [`../neural_fields/TUTORIAL_MASTER_LIST.md`](../neural_fields/TUTORIAL_MASTER_LIST.md)
- Filtering data assimilation → [`../assimilation/docs/TUTORIAL_MASTER_LIST_FILTER.md`](../assimilation/docs/TUTORIAL_MASTER_LIST_FILTER.md)
- Variational data assimilation → [`../assimilation/docs/TUTORIAL_MASTER_LIST_VARIATIONAL.md`](../assimilation/docs/TUTORIAL_MASTER_LIST_VARIATIONAL.md)
- Normalizing flows / gaussianization → [`../gaussianization/TUTORIAL_MASTER_LIST.md`](../gaussianization/TUTORIAL_MASTER_LIST.md)

Cross-listed items (RFF, deep kernels, last-layer-Bayes, BLR, Laplace, VI guides) are flagged 🔁.

**Legend** — Source columns:
- `G` = exists in gaussx (`docs/notebooks/<name>`)
- `P` = exists in pyrox (`docs/notebooks/<name>`)
- `R` = exists in research_notebook (`projects/<area>/notebooks/<path>`)
- `—` = does not exist yet (gap)

**Scope tag**:
- 🧱 **fundamental** — small, library-API demo (gaussx/pyrox docs)
- 🔬 **research** — applied / dataset-driven (research_notebook/projects/bayesian_nns)
- 🌉 **bridge** — useful in either; cross-link
- 🔁 **cross-listed** — also in GP or neural-fields master list

**Refs column**: `gh:<repo>#N` = open GitHub issue (e.g., `gh:pyrox#71`) · `dd:path` = pyrox `design_docs/pyrox/<path>` · `mc#` = numbered model from `examples/nn/regression_masterclass_eqx.md` · `xref:GP#X.Y` = pointer into GP master list.

---

## Curriculum at a glance

- **Part A — Bayesian Regression**
  - A.A — Bayesian linear regression
  - A.B — Fixed feature maps
  - A.C — Random Fourier features
  - A.D — Spectral basis layers (HSGP)
  - A.E — Bridges to GP
  - A.F — Likelihood zoo (regression heads)
  - A.G — Constrained & physics-informed losses
  - A.H — 9-model regression masterclass (pick-apart)
  - A.I — NN MAP baseline & library patterns
- **Part B — Bayesian Classification**
  - B.A — Bayesian logistic regression
  - B.B — Multinomial / softmax
  - B.C — Inference variants for classification
  - B.D — Feature-based & last-layer classifiers
  - B.E — Calibration for Bayesian classifiers
- **Part C — NN ↔ GP Bridges**
  - C.A — Theory (NNGP, NTK, ArcCosine)
  - C.B — Deep kernels
  - C.C — Functional priors
  - C.D — Pathwise BNN sampling
  - C.E — Shared infrastructure
- **Part D — Bayesian Inference for Neural Networks**
  - D.I — Point estimates: MLE / MAP / regularisation-as-prior
  - D.II — Gaussian (Laplace) approximations
  - D.III — Variational inference
  - D.IV — Sampling-based inference
  - D.V — Last-layer & functional posteriors
  - D.VI — Stochastic / implicit Bayes
  - D.VII — Tempering, prior choice, diagnostics
- **Part E — Ensembles**
- **Part F — Calibration, OOD, Active Learning**
- **Part G — Bayesian Neural Fields**
- **Part H — Applied Case Studies *(research_notebook/projects/bayesian_nns)***

---

## Part A — Bayesian Regression

### A.A — Bayesian linear regression

**Key equations / models:**
- BLR posterior: $\Sigma = (\Phi^\top R^{-1}\Phi + S_0^{-1})^{-1}$, $\mu = \Sigma\,\Phi^\top R^{-1}y$
- Precision (natural) form: $\Lambda = \Phi^\top R^{-1}\Phi + S_0^{-1}$, $\eta = \Phi^\top R^{-1} y + S_0^{-1}\mu_0$
- Sequential / online update via Sherman–Morrison: rank-1 covariance update per new observation
- Predictive: $p(y_*\mid x_*, \mathcal{D}) = \mathcal{N}(\phi(x_*)^\top\mu,\, \phi(x_*)^\top\Sigma\phi(x_*) + \sigma^2)$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| A.1 | Bayesian linear regression from scratch (mean-cov form) | — | 🧱 | **GAP** — dd:mc#1 polynomial features + Vandermonde + MCMC |
| A.2 | BLR in precision / natural form | G `numpyro_precision` | 🧱 🔁 | xref:GP#11.2 |
| A.3 | Sequential / online BLR updates | — | 🧱 🔁 | **GAP** — api: `blr_diag_update`, `blr_full_update`; *moved from GP#6.19* |
| A.4 | Polynomial basis regression with uncertainty | — | 🧱 | **GAP** — dd:mc#1 |
| A.5 | Empirical Bayes for BLR — type-II MLE for noise + prior scales | — | 🧱 | **GAP** |

### A.B — Fixed feature maps

**Key equations / models:**
- Generic linear model: $f(x) = \phi(x)^\top w$ with fixed $\phi:\mathbb{R}^d\to\mathbb{R}^D$
- Polynomial, Vandermonde, Chebyshev, Fourier, wavelet, Gaussian-bump bases

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| A.6 | Fixed feature maps: Fourier, polynomial, wavelet, Gaussian-bump | — | 🧱 | **GAP** |
| A.7 | Spectral kernel models — visual guide | P `spectral_kernel_models` | 🧱 🔁 | xref:GP#7.5 |

### A.C — Random Fourier features

**Key equations / models:**
- Rahimi–Recht: $\phi(x) = \sqrt{2/D}\cos(\omega^\top x + b)$, $\omega\sim S(\omega)$, $k(x,x')\approx \phi(x)^\top\phi(x')$
- SSGP: BLR in RFF space, $O(D^2 N)$
- VSSGP: variational posterior over frequencies $\omega$
- ORF: $\omega_i$ on the sphere → variance reduction

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| A.8 | Random Fourier Features → SSGP → VSSGP | P `random_fourier_features` | 🧱 🔁 | dd:mc#5, xref:GP#7.8 |
| A.9 | RFF as a (shallow) neural network — fixed / learned / ensemble | P `rff_as_neural_networks` | 🌉 🔁 | |
| A.10 | SSGP — Sparse Spectrum GP via RFF + BLR, $O(D^2 N)$ | — | 🧱 | **GAP** — dd:examples/nn/models.md |
| A.11 | Heteroscedastic RFF — dual-head (mean + log-noise) | — | 🧱 | **GAP** — dd:features/nn/random_features.md |
| A.12 | Approximate GP via RFF + hierarchical prior on signal variance | — | 🧱 | **GAP** — dd:mc#6 |
| A.13 | Variational Fourier Features (VSSGP) — learnable posterior over RFF freqs | — | 🧱 | **GAP** — pyrox `.plans/spectral-inducing-features.md` |
| A.14 | Orthogonal Random Features (ORF) | — | 🧱 | **GAP** — pyrox `.plans/spectral-inducing-features.md` |

### A.D — Spectral basis layers (HSGP)

**Key equations / models:**
- HSGP (Solin–Särkkä): $k(x,x')\approx \sum_{j=1}^M S(\sqrt{\lambda_j})\,\phi_j(x)\phi_j(x')$, $(\lambda_j,\phi_j)$ Laplacian eigenpairs
- Deterministic basis (vs random RFF) → diagonal $K_{uu}$, $O(NM + M^3)$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| A.15 | HSGP — Hilbert-Space GP layer, deterministic Laplacian basis + spectral-density prior | — | 🧱 🔁 | **GAP** — dd:features/nn/random_features.md; xref:GP#7.12 |

### A.E — Bridges to GP

**Key equations / models:**
- Whitened SVGP-as-BLR view: inducing variables $u = L_{mm}\tilde u$ → BLR on $\tilde u$
- Equivalence map: kernel ridge ↔ MAP-BLR ↔ exact GP posterior mean (with appropriate features)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| A.16 | Whitened SVGP as Bayesian linear regression | G `whitened_svgp` | 🌉 🔁 | xref:GP#5.7 |
| A.17 | Kernel ridge ↔ MAP-BLR ↔ exact GP mean — three views, one estimator | — | 🧱 | **GAP** |

### A.F — Likelihood zoo (regression heads)

**Key equations / models:**
- Gaussian: $y = f(x) + \epsilon$, $\epsilon\sim\mathcal{N}(0,\sigma^2)$
- Student-t: heavy-tailed, $\nu$ controls robustness
- Laplace: $|y-f|$ noise → L1 / Huber-like
- Heteroscedastic: $\sigma(x)$ predicted alongside $\mu(x)$
- Mixture density: $p(y\mid x) = \sum_k \pi_k(x)\mathcal{N}(y; \mu_k(x), \sigma_k(x))$
- Quantile / pinball: $\rho_\tau(u) = u(\tau - \mathbb{1}\{u<0\})$
- Censored / Tobit / survival: likelihood truncated / right-censored
- Log-Gaussian Cox Process: $\lambda(x) = \exp(f(x))$, Poisson observations
- Warped GP / NF-warped: $g(y) = f(x)$, $g$ monotone bijection (Box–Cox, NF)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| A.18 | Gaussian, Student-t, Laplace likelihoods — robust regression | — | 🧱 | **GAP** |
| A.19 | Heteroscedastic NLL — dual-head (mean + log-noise) | — | 🧱 | **GAP** |
| A.20 | Mixture density networks (MDN) | — | 🌉 | **GAP** |
| A.21 | Quantile / pinball / expectile regression | — | 🌉 | **GAP** |
| A.22 | Censored / Tobit / survival likelihoods | — | 🔬 | **GAP** |
| A.23 | Log-Gaussian Cox Process — spatial point-process intensity | — | 🔬 🔁 | **GAP** — *moved from GP#6.20*; dd:examples/gp/moments.md |
| A.24 | Warped regression (Box–Cox) — skewed targets | — | 🧱 🔁 | **GAP** — *moved from GP#6.21* |
| A.25 | Warped regression with normalizing-flow bijection | — | 🔬 🔁 | **GAP** — *moved from GP#6.24*; xref to gaussianization list |

### A.G — Constrained & physics-informed losses

**Key equations / models:**
- Positivity / monotone / convex output via reparameterization (softplus, cumulative)
- Equality constraint via augmented Lagrangian: $\mathcal{L}(\theta,\lambda) + \tfrac{\rho}{2}\|c(\theta)\|^2 + \lambda^\top c(\theta)$
- PDE residual (PINN): $\mathcal{L} = \mathcal{L}_\text{data} + \alpha\|\mathcal{N}[u_\theta](x_i)\|^2$ on collocation points
- Boundary / initial penalty: extra weighted term on $\partial\Omega$
- Conservation / symmetry: divergence-free reparameterization, equivariant layers
- Smoothness / TV regularisation: $\|\nabla u\|_2$, $\|\nabla u\|_1$
- KL annealing / β-VAE: trade reconstruction vs KL

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| A.26 | Positivity, monotone, convex output constraints via reparameterization | — | 🧱 | **GAP** |
| A.27 | Equality constraints via augmented Lagrangian | — | 🌉 | **GAP** |
| A.28 | Bayesian PINN — PDE residual + data likelihood under prior | — | 🔬 | **GAP** |
| A.29 | Boundary / initial-condition penalties | — | 🔬 | **GAP** |
| A.30 | Conservation / symmetry penalties (divergence-free, equivariant) | — | 🔬 | **GAP** |
| A.31 | Smoothness / TV regularisation as prior on outputs | — | 🌉 | **GAP** |
| A.32 | KL annealing & β-tempered ELBO ablation | — | 🧱 | **GAP** — see also D.VII |

### A.H — 9-model regression masterclass (pick-apart)

The pyrox `examples/nn/regression_masterclass_eqx.md` (~927 lines) is a single monolithic notebook. We break it into nine standalone tutorials, each running on the **same dataset** so a learner can ablate one block at a time.

**Key equations / models:** see Models 1–9 below — each row hyperlinks to the corresponding row in A.A–A.C / D.

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| A.33 | Model 1 — Bayesian linear regression with polynomial features (NUTS) | — | 🧱 | dd:mc#1; pairs with A.1, A.4 |
| A.34 | Model 2 — Neural network MAP — single hidden MLP via SVI + AutoDelta | — | 🧱 | dd:mc#2 (deterministic baseline); pairs with D.1 |
| A.35 | Model 3 — MC-Dropout NN — dropout as approximate Bayes | — | 🧱 | dd:mc#3; pairs with D.16 |
| A.36 | Model 4 — Bayesian NN via HMC/NUTS — small-scale weight-space inference | — | 🧱 | dd:mc#4; pairs with D.13 |
| A.37 | Model 5 — SVR via Random Fourier Features (BLR on $\phi(x)$) | — | 🧱 | dd:mc#5; pairs with A.8, A.10 |
| A.38 | Model 6 — Approximate GP via RFF + hierarchical prior on signal variance | — | 🧱 | dd:mc#6; pairs with A.12 |
| A.39 | Model 7 — Deep GP via stacked RFF layers (Cutajar 2017) | — | 🧱 🔁 | dd:mc#7; pairs with C.5 |
| A.40 | Model 8 — Last-layer Bayes (BLR on penultimate features) | — | 🧱 | **NEW** — dd:mc extension; pairs with D.7 |
| A.41 | Model 9 — Mean-field VI BNN — full weight-space VI on the same MLP | — | 🧱 | **NEW** — dd:mc extension; pairs with D.10 |
| A.42 | Capstone — calibration & predictive-distribution shootout across Models 1–9 | — | 🔬 | **GAP** — leads into Part F |

### A.I — NN MAP baseline & library patterns

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| A.43 | NN MAP baseline — single hidden MLP via SVI + AutoDelta (standalone) | — | 🧱 | **GAP** — dd:mc#2 |
| A.44 | Three-pattern regression masterclass — tree_at / pyrox_sample / Parameterized | P `regression_masterclass_treeat`, `_pyrox_sample`, `_parameterized` | 🧱 🔁 | xref:GP#11.3 |
| A.45 | sklearn-style `EstimatorBase` facade for parametric Bayes | — | 🧱 | **GAP** — gh:pyrox#71 |

## Part B — Bayesian Classification

### B.A — Bayesian logistic regression

**Key equations / models:**
- Binary likelihood: $y_i\sim\mathrm{Bernoulli}(\sigma(\phi(x_i)^\top w))$
- Probit alternative: $y_i\sim\mathrm{Bernoulli}(\Phi(\phi(x_i)^\top w))$
- Pólya–Gamma augmentation (Polson et al. 2013): $\sigma(\eta)^y(1-\sigma(\eta))^{1-y} = 2^{-1}\exp(\kappa\eta)\int_0^\infty\exp(-\omega\eta^2/2)p(\omega)\,d\omega$
- Augmented model → conjugate Gaussian update on $w \mid \omega$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| B.1 | Bayesian logistic regression from scratch (Bishop §4.5 walkthrough) | — | 🧱 | **GAP** |
| B.2 | Probit vs logit — link-function comparison | — | 🧱 | **GAP** |
| B.3 | Pólya–Gamma augmentation → conjugate Gibbs for logistic | — | 🧱 | **GAP** |
| B.4 | Jaakkola–Jordan variational bound for logistic | — | 🧱 | **GAP** |
| B.5 | Online / sequential logistic update — Laplace + Sherman–Morrison | — | 🧱 | **GAP** |

### B.B — Multinomial / softmax classification

**Key equations / models:**
- Multinomial: $y\sim\mathrm{Cat}(\mathrm{softmax}(W^\top\phi(x)))$
- One-vs-rest / multinomial-probit alternatives
- Augmentation schemes (Pólya–Gamma stick-breaking, Albert–Chib)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| B.6 | Bayesian softmax / multinomial logistic regression | — | 🧱 | **GAP** |
| B.7 | Stick-breaking + Pólya–Gamma for multinomial | — | 🧱 | **GAP** |
| B.8 | Ordinal regression — cumulative-link Bayesian model | — | 🌉 | **GAP** |
| B.9 | Multi-label classification — independent vs structured priors | — | 🔬 | **GAP** |

### B.C — Inference variants for classification

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| B.10 | Laplace approximation for logistic regression | — | 🧱 🔁 | **GAP** — canonical Bishop example; uses D.II machinery |
| B.11 | Variational logistic / softmax — mean-field & full-rank | — | 🧱 🔁 | **GAP** — uses D.III |
| B.12 | HMC / NUTS for small Bayesian classifiers | — | 🧱 🔁 | **GAP** — uses D.IV |
| B.13 | Expectation Propagation for GP classification — re-used here for BLR-classification | — | 🧱 🔁 | xref:GP#6.17 |

### B.D — Feature-based & last-layer classifiers

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| B.14 | RFF + Bayesian logistic regression (kernel classification) | — | 🌉 | **GAP** |
| B.15 | Last-layer Bayesian classifier (Laplace / BLR head on a deterministic MLP) | — | 🔬 | **GAP** |
| B.16 | SNGP for classification — distance-aware uncertainty | — | 🔬 | **GAP** — gh:pyrox#42; pairs with D.21 |
| B.17 | Random-feature GP classifier (LaplaceRandomFeatureCovariance) | — | 🧱 | **GAP** — dd:features/nn/edward2_layers.md |

### B.E — Calibration for Bayesian classifiers

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| B.18 | Reliability diagrams & ECE for Bayesian classifiers | — | 🔬 | **GAP** — see also F.1 |
| B.19 | Temperature scaling on top of Bayesian posteriors | — | 🔬 | **GAP** |
| B.20 | Predictive entropy / mutual information for classification uncertainty | — | 🔬 | **GAP** |

## Part C — NN ↔ GP Bridges

### C.A — Theory

**Key equations / models:**
- NNGP (Lee et al. 2018): infinite-width MLP with i.i.d. priors → GP with recursive kernel $K^{(\ell)} = T_\sigma(K^{(\ell-1)})$
- NTK (Jacot et al. 2018): $\Theta(x,x') = \mathbb{E}\langle\partial_\theta f(x),\partial_\theta f(x')\rangle$, frozen in the infinite-width limit
- ArcCosine (Cho & Saul 2009): $k_n(x,x') = \tfrac{1}{\pi}\|x\|^n\|x'\|^n J_n(\theta)$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| C.1 | Infinite-width NN as a GP (NNGP) | — | 🧱 | **GAP** — Lee et al. 2018 |
| C.2 | Neural Tangent Kernel intro | — | 🧱 | **GAP** |
| C.3 | ArcCosine kernel — NN-correspondence via infinite-width limits | — | 🧱 🔁 | **GAP** — dd:features/gp/gpflow.md; xref:GP#2.7 |

### C.B — Deep kernels

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| C.4 | Deep kernels — NN-warped GP inputs | R `pyroxgp/04_svgp_rff_nn` | 🌉 🔁 | xref:GP#2.6 |
| C.5 | Deep RFF / stacked spectral GPs (Cutajar 2017) | P `deep_random_fourier_features` | 🔬 🔁 | dd:mc#7 |

### C.C — Functional priors

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| C.6 | Functional priors — BNNs that match a target GP prior | — | 🔬 | **GAP** |
| C.7 | Prior predictive checks for BNNs — sample-and-visualise | — | 🌉 | **GAP** |

### C.D — Pathwise BNN sampling

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| C.8 | Pathwise sampling for BNNs (analogue of Wilson 2020) | — | 🧱 🔁 | **GAP** — needs gh:gaussx#77, #78; xref:GP#9.1 |

### C.E — Shared infrastructure

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| C.9 | Shared `pyrox._basis` — VFF (GP) + HSGP (NN) sharing Laplacian eigenfunctions | — | 🧱 | **GAP** — pyrox `.plans/spectral-inducing-features.md` |

## Part D — Bayesian Inference for Neural Networks

Reorganised around the **kind of approximation** rather than the layer flavour. Layer-flavour (Edward2, Conv/RNN/Attn) lives in D.VI/D.VIII.

### D.I — Point estimates: MLE / MAP / regularisation-as-prior

**Key equations / models:**
- MLE: $\hat\theta = \arg\max_\theta \prod_i p(y_i\mid x_i,\theta)$
- MAP: $\hat\theta = \arg\max_\theta p(\theta)\prod_i p(y_i\mid x_i,\theta)$
- L2 ridge ↔ Gaussian prior; L1 lasso ↔ Laplace prior; elastic-net = sum
- Weight decay / spectral / Jacobian regularisation as implicit priors

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| D.1 | MLE vs MAP — same architecture, prior-tuning sweep | — | 🧱 | **GAP** |
| D.2 | Regularisation-as-prior — L2/L1/elastic-net ↔ Gaussian / Laplace / mixture | — | 🧱 | **GAP** |
| D.3 | Spectral / Jacobian / weight-decay regularisation as implicit prior | — | 🌉 | **GAP** |

### D.II — Gaussian (Laplace) approximations

**Key equations / models:**
- Laplace: $q(\theta) = \mathcal{N}(\hat\theta, -H^{-1})$, $H = \nabla^2\log p(\theta\mid\mathcal{D})$
- GGN: $H \approx J^\top R J$ (drops 2nd-order); KFAC: block-Kronecker GGN
- Diagonal Laplace: $\mathrm{diag}(H)$ via Hutchinson
- Linearised Laplace: predict via $f_\theta(x)\approx f_{\hat\theta}(x) + J_{\hat\theta}(x)(\theta-\hat\theta)$
- SWAG: low-rank + diagonal Gaussian over SGD iterates
- Moment-matching / unscented predictive

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| D.4 | Laplace approximation — pure mechanics (canonical) | P `advanced_gp_laplace` | 🧱 🔁 | xref:GP#6.7 |
| D.5 | Gauss–Newton / GGN approximation | P `advanced_gp_gauss_newton` | 🧱 🔁 | xref:GP#6.8 |
| D.6 | Quasi-Newton / L-BFGS site update | P `advanced_gp_qn` | 🧱 🔁 | xref:GP#6.9 |
| D.7 | Posterior linearisation (Bayes-Newton) | P `advanced_gp_pl` | 🧱 🔁 | xref:GP#6.10 |
| D.8 | Hutchinson Hessian / GGN diagonal for BNN Laplace | — | 🧱 🔁 | **GAP** — api: `hutchinson_hessian_diag`; xref:GP#6.13 |
| D.9 | KFAC Laplace — block-Kronecker GGN over a full network | — | 🔬 | **GAP** |
| D.10 | Linearised Laplace predictive (Immer et al.) | — | 🔬 | **GAP** |
| D.11 | SWAG — stochastic weight averaging Gaussian | — | 🔬 | **GAP** |
| D.12 | Subspace inference — PCA of SGD trajectory | — | 🔬 | **GAP** |
| D.13 | Moment-matching predictive — unscented / sigma-point propagation through NN | — | 🌉 🔁 | **GAP** — xref:GP#10.1 |

### D.III — Variational inference

**Key equations / models:**
- ELBO: $\log p(y)\geq\mathbb{E}_q[\log p(y,\theta)] - \mathbb{E}_q[\log q(\theta)]$
- Variational families: delta · mean-field diagonal · low-rank ($S = VV^\top + \mathrm{diag}$) · full-rank Cholesky · normalising flow · whitened
- Natural gradient: $\tilde\nabla\mathcal{L} = F^{-1}\nabla\mathcal{L}$
- CVI sites (Khan & Lin 2017); reparameterisation, local-reparam, flipout

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| D.14 | Variational guides — delta / mean-field / low-rank / full-rank / whitened / flow | — | 🧱 🔁 | **GAP** — dd:features/gp/variational_families.md; xref:GP#6.14 |
| D.15 | Natural gradient VI | G `natural_gradient_vi` | 🌉 🔁 | xref:GP#6.15 |
| D.16 | Mean-field VI for BNNs (MFVI) — Bayes-by-Backprop | — | 🔬 | **GAP** — needs gh:gaussx#39 logdet |
| D.17 | Full-rank / low-rank structured VI for BNNs | — | 🔬 | **GAP** |
| D.18 | Normalising-flow posteriors over weights | — | 🔬 | **GAP** — bridge to gaussianization list |
| D.19 | Reparameterisation tricks — local reparam, flipout, weight-norm | — | 🧱 | **GAP** |
| D.20 | Functional VI — variational posterior on $f(\cdot)$ rather than $\theta$ | — | 🔬 | **GAP** |

### D.IV — Sampling-based inference

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| D.21 | HMC / NUTS for small BNNs | — | 🔬 | **GAP** — dd:mc#4 |
| D.22 | SGLD / SG-HMC — stochastic-gradient Langevin & Hamiltonian | — | 🔬 | **GAP** |
| D.23 | Stein Variational Gradient Descent (SVGD) | — | 🔬 | **GAP** |
| D.24 | Ensemble-of-MCMC — multi-chain pooling | — | 🔬 | **GAP** |
| D.25 | MCMC diagnostics for BNNs — $\hat R$, effective sample size, posterior-predictive checks | — | 🔬 | **GAP** |

### D.V — Last-layer & functional posteriors

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| D.26 | Last-layer Bayes via Laplace | — | 🔬 | **GAP** — api: `gauss_newton_precision`, `ggn_diagonal` |
| D.27 | Last-layer Bayes via RFF (BLR on penultimate features) | — | 🔬 | **GAP** |
| D.28 | RandomFeatureGaussianProcess + LaplaceRandomFeatureCovariance — SNGP output layer | — | 🧱 | **GAP** — dd:features/nn/edward2_layers.md |
| D.29 | Subnetwork inference — only-some-layers-Bayesian | — | 🔬 | **GAP** |

### D.VI — Stochastic / implicit Bayes

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| D.30 | MC-Dropout as approximate Bayes | — | 🔬 | **GAP** — dd:mc#3 |
| D.31 | DenseVariationalDropout — learned per-weight dropout rates | — | 🧱 | **GAP** — dd:features/nn/edward2_layers.md |
| D.32 | DenseDVI — analytic Gaussian moment propagation | — | 🧱 | **GAP** — dd:features/nn/edward2_layers.md |
| D.33 | DenseRank1 / BatchEnsemble — shared $W$ + per-member rank-1 perturbations | — | 🧱 | **GAP** — dd:features/nn/edward2_layers.md |
| D.34 | MCSoftmaxDenseFA / MCSigmoidDenseFA — heteroscedastic output (low-rank + diagonal) | — | 🧱 | **GAP** — dd:features/nn/edward2_layers.md |
| D.35 | DenseHierarchical — horseshoe prior (local + global shrinkage, ARD) | — | 🧱 | **GAP** — dd:features/nn/edward2_layers.md |
| D.36 | NCPNormalOutput — output-side noise contrastive prior | — | 🧱 | **GAP** — dd:features/nn/edward2_layers.md |
| D.37 | Conv2DReparameterization — Bayesian 2D conv | — | 🧱 | **GAP** — dd:features/nn/layers_conv_rnn.md |
| D.38 | Conv2DFlipout — lower-variance Bayesian conv | — | 🧱 | **GAP** — dd:features/nn/layers_conv_rnn.md |
| D.39 | LSTMCellVariational — Bayesian LSTM | — | 🧱 | **GAP** — dd:features/nn/layers_conv_rnn.md |
| D.40 | GRUCellVariational — Bayesian GRU (scan-compatible) | — | 🧱 | **GAP** — dd:features/nn/layers_conv_rnn.md |
| D.41 | MultiHeadAttentionVariational / MultiHeadAttentionBE — Bayesian attention | — | 🧱 | **GAP** — dd:features/nn/edward2_layers.md, layers_conv_rnn.md |

### D.VII — Tempering, prior choice, diagnostics

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| D.42 | Cold posteriors & temperature scaling | — | 🔬 | **GAP** |
| D.43 | KL annealing / β-tempered ELBO | — | 🧱 | **GAP** — see also A.32 |
| D.44 | Prior elicitation for BNNs — Gaussian / Laplace / horseshoe / mixture | — | 🌉 | **GAP** |
| D.45 | Posterior predictive checks & residual diagnostics | — | 🌉 | **GAP** |
| D.46 | Bayesian model averaging vs marginal likelihood / WAIC / LOO | — | 🔬 | **GAP** |
| D.47 | Continual / online BNN updates — Laplace propagation, BLR-style refresh | — | 🔬 | **GAP** — links to A.3 |

### D.VIII — Distance-aware uncertainty

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| D.48 | SNGP — Spectral-Normalized GP head | — | 🔬 | **GAP** — gh:pyrox#42, dd:features/nn/spectral_norm.md |
| D.49 | DUE — Deterministic Uncertainty Estimation (spectral norm + inducing-point GP head) | — | 🔬 | **GAP** — dd:features/nn/spectral_norm.md |

## Part E — Ensembles

### E.A — Vanilla ensembles
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| E.1 | Deep ensembles — vanilla | — | 🔬 | **GAP** |
| E.2 | Ensemble primitives — three ways | P `ensemble_primitives_tutorial` | 🧱 🔁 | xref:GP#12.1 |
| E.3 | EnsembleMAP & EnsembleVI runners | P `ensemble_runner_tutorial` | 🧱 🔁 | xref:GP#12.2 |
| E.4 | Ensemble-of-MAP / -of-VI runner via vmap over PRNG keys | — | 🧱 | **GAP** — gh:pyrox#70 |

### E.B — Diversity strategies
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| E.5 | Snapshot / cyclical-LR ensembles | — | 🔬 | **GAP** |
| E.6 | Hyper-deep ensembles (DenseRank1 substrate) | — | 🔬 | **GAP** |

### E.C — Comparison
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| E.7 | Deep ensembles vs MFVI vs Laplace — calibration shootout | — | 🔬 | **GAP** |

## Part F — Calibration, OOD, Active Learning

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| F.1 | Predictive calibration — ECE, reliability diagrams (regression + classification) | — | 🔬 | **GAP** |
| F.2 | Temperature scaling & post-hoc calibration | — | 🔬 | **GAP** |
| F.3 | NLPD / CRPS / coverage diagnostics for BNNs | — | 🔬 🔁 | **GAP** — xref:GP#15 |
| F.4 | Out-of-distribution detection with BNNs (predictive entropy, mutual info) | — | 🔬 | **GAP** |
| F.5 | Active learning / Bayesian acquisition functions (BALD, max-entropy) | — | 🔬 | **GAP** |
| F.6 | Selective prediction / abstention under uncertainty | — | 🔬 | **GAP** |

## Part G — Bayesian Neural Fields

Core (deterministic) neural-fields content lives in [`../neural_fields/TUTORIAL_MASTER_LIST.md`](../neural_fields/TUTORIAL_MASTER_LIST.md). This section is the **Bayesian** layer on top — point estimates and uncertainty for INRs.

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| G.1 | Bayesian INR — probabilistic SIREN with MFVI weights | — | 🔬 | **GAP** — pairs with `xref:NF#B.1` (SIREN) |
| G.2 | Bayesian INR via last-layer Laplace on a SIREN | — | 🔬 | **GAP** |
| G.3 | Bayesian NeRF — uncertainty in volumetric scenes | — | 🔬 | **GAP** — pairs with `xref:NF#C.1` (vanilla NeRF) |
| G.4 | Functional priors for INRs — match a target spatial GP | — | 🔬 | **GAP** |
| G.5 | BNF layer family + `BNFEstimator` / MLE / VI runners | — | 🔬 | **GAP** — gh:pyrox#72 |
| G.6 | Bayesian neural fields flagship demo (`bayesian_neural_fields.ipynb`) | — | 🔬 | **GAP** — gh:pyrox#73 |

## Part H — Applied Case Studies *(research_notebook/projects/bayesian_nns)*

### H.A — Bayesian benchmarks
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| H.1 | Last-layer Bayesian NN on UCI regression suite | — | 🔬 | **GAP** |
| H.2 | Deep RFF on geophysical / climate data | P `deep_random_fourier_features` (port + extend) | 🔬 🔁 | |
| H.3 | scalable_gp_spectral demo — 5k 1D regression, dense GP vs VFF, ≥10× speedup | — | 🔬 | **GAP** — pyrox `.plans/spectral-inducing-features.md` |

### H.B — Emulators & PDEs
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| H.4 | BNN emulator for a numerical simulator | — | 🔬 | **GAP** |
| H.5 | Bayesian PINN — Burgers / heat / shallow-water | — | 🔬 | **GAP** — pairs with A.28 |
| H.6 | Bayesian operator learning — DeepONet / FNO with weight uncertainty | — | 🔬 | **GAP** |

### H.C — Image / signal regression
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| H.7 | BNN for image regression / denoising | — | 🔬 | **GAP** |
| H.8 | Probabilistic super-resolution via RFF / INR | — | 🔬 | **GAP** |

### H.D — Capstone progressions
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| H.9 | Full 9-model regression masterclass — single dataset, methodical climb | — | 🔬 | **GAP** — dd:examples/nn/regression_masterclass_eqx.md (~927 lines); see A.H |
| H.10 | Classification capstone — same dataset, Models 1–9 ported to classification | — | 🔬 | **GAP** — mirrors A.H for Part B |

---

## Cross-list summary (items shared with GP list)

| Item | GP ID | BNN ID | Suggested canonical home |
|---|---|---|---|
| Spectral kernel models | GP 7.5 | A.7 | pyrox (GP), cross-listed |
| Random Fourier Features intro | GP 7.8 | A.8 | pyrox (canonical), link both |
| RFF as neural networks | — | A.9 | pyrox |
| Whitened SVGP / BLR view | GP 5.7 | A.16 | gaussx (mechanics) |
| BLR updates (`blr_*_update`) | *moved* | A.3 | gaussx primitive demo — **migrated out of GP list** |
| BLR in precision form | GP 11.2 | A.2 | gaussx |
| Three-pattern masterclass | GP 11.3 | A.44 | pyrox |
| Deep kernels | GP 2.6 | C.4 | research_notebook |
| Deep RFF (Cutajar) | — | C.5 / H.2 | research_notebook (BNN) |
| ArcCosine kernel | GP 2.7 | C.3 | pyrox |
| Pathwise sampling | GP 9.1 | C.8 | pyrox |
| Laplace mechanics | GP 6.7 | D.4 | pyrox |
| Gauss–Newton | GP 6.8 | D.5 | pyrox |
| Quasi-Newton sites | GP 6.9 | D.6 | pyrox |
| Posterior linearisation | GP 6.10 | D.7 | pyrox |
| Hutchinson Hessian diag | GP 6.13 | D.8 | gaussx primitive + BNN application |
| VI guides | GP 6.14 | D.14 | pyrox (canonical for both) |
| Natural-gradient VI | GP 6.15 | D.15 | gaussx |
| Moment matching predictive | GP 10.1 | D.13 | gaussx primitive |
| Log-Gaussian Cox Process | *moved* | A.23 | **migrated out of GP list** |
| Warped GP (Box–Cox) | *moved* | A.24 | **migrated out of GP list** |
| Warped GP w/ NF bijection | *moved* | A.25 | **migrated out of GP list** |
| EP for classification | GP 6.17 | B.13 | pyrox |
| Ensemble primitives | GP 12.1 | E.2 | pyrox |
| Ensemble runners | GP 12.2 | E.3 | pyrox |

## Proposed final homes

- **gaussx/docs/notebooks/** → A.A (BLR primitives), A.E (whitened SVGP / BLR view), D.II primitives (D.8 Hutchinson, D.13 moment matching), D.15 nat-grad VI
- **pyrox/docs/notebooks/** → A.B, A.C, A.D, A.F, A.G, A.H, A.I, B.A–B.E, C.* shared infra, D.IV–D.VI library demos, E.A–E.B
- **research_notebook/projects/bayesian_nns/notebooks/** → all of D.III–D.IV applied, D.V, D.VII, E.C, F, G, H

## In-scope vs aspirational

- **In scope today** (have library support in pyrox/gaussx): A.2, A.3, A.7, A.8, A.9, A.16, A.44, B.13, C.4, C.5, D.4–D.8, D.15, E.2, E.3, H.2
- **In scope with planned features** (open issues / `.plans/`): A.1, A.4–A.6, A.10–A.15, A.17–A.25, A.32, A.33–A.45, B.1–B.12, B.14–B.17, C.3, C.9, D.28, D.31–D.41, D.48, E.4, G.5, G.6, H.3, H.9, H.10
- **Aspirational** (need new infra or genuine research work): A.26–A.31, B.18–B.20, C.1, C.2, C.6, C.7, C.8, D.1–D.3, D.9–D.14 (applied), D.16–D.20, D.21–D.27 (applied), D.29, D.30, D.42–D.47, D.49, E.1, E.5–E.7, F.*, G.1–G.4, H.1, H.4–H.8
