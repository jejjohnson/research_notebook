# GP Tutorial Master List

A reconciled, exhaustive curriculum spanning what currently exists in **gaussx**, **pyrox**, and **research_notebook**, plus gaps surfaced from the gaussx + pyrox public APIs, open GitHub issues, and pyrox `design_docs/`. Goal: the most complete GP tutorial sequence we could ship.

> Bayesian NN / NeRF / basis-function-regression tutorials live in [`../bayesian_nns/TUTORIAL_MASTER_LIST.md`](../bayesian_nns/TUTORIAL_MASTER_LIST.md). Cross-listed items (RFF, deep kernels, BLR, last-layer-Bayes) are flagged 🔁.

**Legend** — Source columns:
- `G` = exists in gaussx (`docs/notebooks/<name>`)
- `P` = exists in pyrox (`docs/notebooks/<name>`)
- `R` = exists in research_notebook (`projects/gaussian_processes/notebooks/<path>`)
- `—` = does not exist yet (gap)

**Scope tag**: 🧱 fundamental · 🔬 research · 🌉 bridge · 🔁 cross-listed

**Refs column**: `gh#N` = open GitHub issue · `dd:path` = pyrox `design_docs/pyrox/<path>` · `api:foo` = gaussx exported symbol.

---

## Part 0 — Linear Algebra & Gaussian Foundations

### 0.A — The Multivariate Gaussian

**Key equations / models:**
- Density: $\mathcal{N}(x;\mu,\Sigma) = (2\pi)^{-d/2}|\Sigma|^{-1/2}\exp\!\big({-}\tfrac{1}{2}(x-\mu)^\top\Sigma^{-1}(x-\mu)\big)$
- Reparameterized sample: $x = \mu + L\epsilon$, $LL^\top = \Sigma$, $\epsilon\sim\mathcal{N}(0,I)$
- Entropy: $H = \tfrac{1}{2}\log|2\pi e\,\Sigma|$
- KL: $\mathrm{KL}(p\Vert q) = \tfrac{1}{2}\big[\mathrm{tr}(\Sigma_q^{-1}\Sigma_p) + \Delta\mu^\top\Sigma_q^{-1}\Delta\mu - d + \log\tfrac{|\Sigma_q|}{|\Sigma_p|}\big]$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 0.1 | [The Multivariate Gaussian: density, sampling, conditioning](notebooks/00_foundations/multivariate_gaussian.ipynb) | R `multivariate_gaussian` | 🧱 | pedagogical entry — three sampling routes, marginal & Schur conditioning, jitter |
| 0.2 | [`MultivariateNormal` & `MultivariateNormalPrecision` distribution API](notebooks/00_foundations/mvn_distribution_api.ipynb) | R `mvn_distribution_api` | 🧱 | covariance vs precision parameterisation, GMRF / banded Λ, round-trip equivalence |
| 0.3 | [Quadratic forms, entropy, KL between Gaussians](notebooks/00_foundations/gaussian_quantities.ipynb) | R `gaussian_quantities` | 🧱 | api: `gaussian_entropy`, `dist_kl_divergence`, `kl_standard_normal`, `quadratic_form`, `gaussian_expected_log_lik` — extended to cover score, cross-entropy, expected log-likelihood, mutual information, mini-ELBO |

### 0.B — Parameterizations

**Key equations / models:**
- Natural parameters: $\eta_1 = \Sigma^{-1}\mu$, $\eta_2 = -\tfrac{1}{2}\Sigma^{-1}$
- Expectation parameters: $m_1 = \mu$, $m_2 = \Sigma + \mu\mu^\top$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 0.4 | [Three parameterizations: mean-cov ↔ natural ↔ expectation](notebooks/00_foundations/natural_parameters.ipynb) | R `natural_parameters` | 🧱 | api: `mean_cov_to_natural`, `natural_to_mean_cov`, `natural_to_expectation`, `expectation_to_natural`, `damped_natural_update` — round-trip identities, conjugate update as natural-form addition, moment matching, damped VI/EP primitive, use-case map across the curriculum |

### 0.C — Bayesian Updates & Conditioning

**Key equations / models:**
- Sequential conjugate update: $p(\theta\mid y_{1:n}) \propto p(y_n\mid\theta)\,p(\theta\mid y_{1:n-1})$
- Schur conditional: $\mu_{a\mid b} = \mu_a + \Sigma_{ab}\Sigma_{bb}^{-1}(x_b-\mu_b)$, $\Sigma_{a\mid b} = \Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}$
- Structured-MVN sample: $x = \mu + \mathrm{root}(\Sigma)\,\epsilon$ with dispatched `root` per operator type

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 0.6 | [Bayesian updates from scratch (sequential conjugate)](notebooks/00_foundations/bayesian_updates.ipynb) | R `bayesian_updates` | 🧱 | natural-form addition recursion, batch = sequential = any order, GP regression as single application |
| 0.7 | [Conditional distributions & Schur complement](notebooks/00_foundations/conditional_distributions.ipynb) | R `conditional_distributions` | 🧱 | api: `gaussx.conditional`, `schur_complement`, `conditional_variance`, `cov_transform`; GP regression as joint conditioning |
| 0.8 | [Structured MVN sampling dispatch](notebooks/00_foundations/structured_sampling.ipynb) | R `structured_sampling` | 🧱 | api: `gaussx.cholesky`, `gaussx.sqrt`; dispatch on Diagonal / Kronecker / BlockDiag / BlockTriDiag; LowRank additive sampling; fast-sampling tracking issues [gaussx#168](https://github.com/jejjohnson/gaussx/issues/168) (Toeplitz), [#169](https://github.com/jejjohnson/gaussx/issues/169) (KroneckerSum), [#170](https://github.com/jejjohnson/gaussx/issues/170) (SumKronecker) |

### 0.D — Numerical Mechanics

**Key equations / models:**
- Joseph-form covariance update: $P^+ = (I-KH)\,P\,(I-KH)^\top + KRK^\top$ (PSD-preserving)
- Cholesky: $A = LL^\top$, $\log|A| = 2\sum_i \log L_{ii}$
- Implicit diff through `solve`: $\partial_\theta x = A^{-1}(\partial_\theta b - (\partial_\theta A)\,x)$
- Jacobi's formula: $\partial_\theta \log|A| = \mathrm{tr}(A^{-1}\partial_\theta A)$
- Jitter / safe Cholesky: $A + \epsilon I$, doubling $\epsilon$ until SPD
- Stable squared distances: mixed-precision $\|x-z\|^2$ to avoid catastrophic cancellation

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 0.5 | [Joseph-form covariance update](notebooks/00_foundations/joseph_form_update.ipynb) | R `joseph_form_update` | 🧱 | four equivalent covariance updates (standard / symmetric / information / Joseph), float32 stress test, connection to natural-parameter addition; **preservation** counterpart to 0.11's **recovery** tools |
| 0.9 | [Cholesky, log-det, trace primitives tour](notebooks/00_foundations/cholesky_logdet_trace.ipynb) | R `cholesky_logdet_trace` | 🧱 | api: `gaussx.cholesky`, `gaussx.logdet`, `gaussx.trace`, `gaussx.diag` — closed-form identities / compute / storage tables, theoretical-order plots, Hutchinson stochastic trace |
| 0.10 | [Differentiating through `solve`](notebooks/00_foundations/differentiating_solve.ipynb) | R `differentiating_solve` | 🧱 | implicit-function-theorem JVP/VJP via lineax, Jacobi's formula for `logdet` gradients, GP marginal-likelihood ascent in one `jax.grad` call |
| 0.11 | [Numerical stability: jitter, safe Cholesky, condition number](notebooks/00_foundations/numerical_stability.ipynb) | R `numerical_stability` | 🧱 | api: `gaussx.add_jitter`, `gaussx.safe_cholesky` — condition-number diagnostic, bias–stability U-curve trade-off, float32 stress; jitter as **recovery** vs Joseph as **preservation** |
| 0.12 | [Stable RBF & squared distances](notebooks/00_foundations/stable_rbf_distances.ipynb) | R `stable_rbf_distances` | 🧱 | api: `gaussx.stable_rbf_kernel`, `gaussx.stable_squared_distances` — mixed-precision recipe, catastrophic cancellation, three-stage robustness pipeline (stable distances → jitter / safe Cholesky → Joseph form) |

## Part 1 — Structured Linear Operators

### 1.A — Operator Zoo (catalog)

**Key equations / models:**
- Kronecker: $A\otimes B$, with Roth's lemma $(A\otimes B)\,\mathrm{vec}(X) = \mathrm{vec}(BXA^\top)$
- BlockDiag: $\mathrm{diag}(A_1,\dots,A_K)$
- LowRankUpdate: $L + UDV^\top$
- Toeplitz: $T_{ij} = t_{i-j}$, matvec via FFT in $O(n\log n)$
- BlockTriDiag: tridiagonal precision $\Lambda$ for Markov chains
- KroneckerSum: $A\oplus B = A\otimes I + I\otimes B$ (separable Laplacian)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 1.1 | Operator basics + structural tags & dispatch (Dense, Diagonal, Kronecker, BlockDiag, LowRankUpdate; tag inventory; isinstance dispatch; bring-your-own-operator Circulant demo) | G `operator_basics` | 🧱 | ✅ merged 1.1+1.7; replaces former `basics`/`operator_zoo` stubs |
| 1.2 | Lazy operator algebra (Sum, Scaled, Product) | G `lazy_algebra` | 🧱 | ✅ |
| 1.3 | KroneckerSum vs SumKronecker (additive vs superposed) | G `kronecker_sum_vs_sum_kronecker` | 🧱 | ✅ |
| 1.4 | Toeplitz operators for stationary 1-D / 2-D grids | G `toeplitz` | 🧱 | ✅ |
| 1.5 | BlockTriDiag (Markov / Kalman precision form) + Lower/Upper variants | G `block_tridiag` | 🧱 | ✅ |
| 1.6 | MaskedOperator for missing data on a structured grid (MVN / Toeplitz / Kron / BlockTriDiag bases) | G `masked_operator` | 🧱 | ✅ |

### 1.B — Matrix Identities & Decompositions

**Key equations / models:**
- Kron eigendecomp: $A\otimes B = (Q_A\otimes Q_B)(\Lambda_A\otimes\Lambda_B)(Q_A\otimes Q_B)^\top$
- Sherman–Morrison–Woodbury: $(A+UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1}+VA^{-1}U)^{-1}VA^{-1}$
- Det lemma: $|A+UCV| = |C^{-1}+VA^{-1}U|\,|C|\,|A|$
- Operator sandwich: $A\,P\,A^\top$ assembled lazily
- UDL of block-tridiagonal: $\Lambda = U^\top D^{-1} U$
- Discrete Lyapunov: $P_\infty = AP_\infty A^\top + Q$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 1.8 | Kronecker eigendecomposition | G `kronecker_eigen` | 🧱 | |
| 1.9 | Sherman–Morrison–Woodbury walkthrough | G `woodbury_solve` | 🧱 | |
| 1.10 | Operator sandwich `A P Aᵀ` without materialization | — | 🧱 | **GAP** — gh:gaussx#163 |
| 1.11 | UDL decomposition for block-tridiagonal precision | — | 🧱 | **GAP** — gh:gaussx#65 |
| 1.12 | Discrete Lyapunov solve (stationary covariance of LTI) | — | 🧱 | **GAP** — api: `discrete_lyapunov_solve` |

### 1.C — Matrix-Free / Implicit

**Key equations / models:**
- Matrix-free matvec: $y = K(X,X')\,v$ via nested `vmap` over rows, never forming $K$
- Cross-kernel matvec: $y = K(X,Z)\,v$ for prediction without $|X|\times|Z|$ allocation

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 1.13 | Matrix-free / implicit operators | G `implicit_kernel` | 🧱 | extend with `ImplicitCrossKernelOperator` (**GAP**) |

### 1.D — Solvers

**Key equations / models:**
- Cholesky solve: $LL^\top x = b$
- Conjugate Gradient: minimize $\tfrac{1}{2}x^\top A x - b^\top x$ in Krylov subspace
- BBMM: batched matvec drives `solve` + `logdet` + `grad` simultaneously (Gardner et al. 2018)
- MINRES: indefinite symmetric · LSMR: rectangular least squares
- Preconditioned CG: solve $M^{-1}Ax = M^{-1}b$ with $M\approx A$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 1.14 | Solver strategies overview (dense, CG, Lanczos) | G `solver_strategies`, G `solver_comparison` | 🧱 | merge candidates |
| 1.15 | Preconditioned CG | — | 🧱 | **GAP** — api: `PreconditionedCGSolver` |
| 1.16 | BBMM — Black-Box Matrix-Matrix Multiplication | — | 🧱 | **GAP** — api: `BBMMSolver`; GPyTorch-style |
| 1.17 | Indefinite/non-PSD: MINRES / LSMR | — | 🧱 | **GAP** — api: `MINRESSolver`, `LSMRSolver` |
| 1.18 | Auto-dispatch (`AutoSolver`, `ComposedSolver`) | — | 🧱 | **GAP** |

### 1.E — Trace, Log-Det, Roots

**Key equations / models:**
- Hutchinson trace: $\mathrm{tr}(A) \approx \tfrac{1}{m}\sum_{i=1}^m z_i^\top A z_i$, $z_i\sim\mathrm{Rademacher}$
- Stochastic Lanczos Quadrature: $\log|A| \approx \tfrac{n}{m}\sum z_i^\top \log(A) z_i$ via Lanczos tridiagonalization
- Contour-integral root: $A^{1/2} = \tfrac{1}{2\pi i}\oint \sqrt{z}(zI-A)^{-1}dz$
- Joint inv-quad-logdet: shared CG/Lanczos passes return $y^\top A^{-1}y$ and $\log|A|$ together

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 1.19 | Stochastic Lanczos Quadrature log-det | — | 🧱 | **GAP** — api: `SLQLogdet`, `IndefiniteSLQLogdet` |
| 1.20 | RNLA — randomized numerical linear algebra port | — | 🧱 | **GAP** — gh:gaussx#156 |
| 1.21 | Contour-integral `sqrt_inv_matmul` / `sqrt_matmul` | — | 🧱 | **GAP** — gh:gaussx#43 |
| 1.22 | Root & inverse-root decompositions | — | 🧱 | **GAP** — gh:gaussx#40 |
| 1.23 | Joint inverse-quadratic + log-det | — | 🧱 | **GAP** — gh:gaussx#39 |

## Part 2 — Kernels

### 2.A — Standard kernels

**Key equations / models:**
- RBF: $k(x,x') = \sigma^2\exp(-\tfrac{1}{2\ell^2}\|x-x'\|^2)$
- Matérn-ν: $k_\nu(r) = \sigma^2 \tfrac{2^{1-\nu}}{\Gamma(\nu)}(\sqrt{2\nu}\,r/\ell)^\nu K_\nu(\sqrt{2\nu}\,r/\ell)$, $\nu\in\{1/2,3/2,5/2\}$
- Periodic: $\sigma^2\exp(-2\sin^2(\pi|x-x'|/p)/\ell^2)$
- Linear: $\sigma^2 x^\top x'$ · Polynomial: $(\sigma^2 x^\top x' + c)^d$
- ARD lengthscale: $\|x-x'\|^2_{\Lambda} = \sum_d (x_d-x'_d)^2/\ell_d^2$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 2.1 | Kernel cookbook: RBF, Matérn, Periodic, Linear, Polynomial | — | 🧱 | **GAP** |
| 2.2 | Kernel composition: sum, product, warping | — | 🧱 | **GAP** |
| 2.3 | ARD & lengthscale interpretation | — | 🧱 | **GAP** |
| 2.4 | Stationary vs non-stationary kernels | — | 🧱 | **GAP** |
| 2.13 | Pytree kernel composition — sum / product / scaled as pytrees | — | 🧱 | **GAP** — canonical JAX pattern, important for gaussx/pyrox users |

### 2.B — Spectral & deep kernels

**Key equations / models:**
- Bochner: stationary $k(\tau) = \int e^{i\omega^\top\tau} S(\omega)\,d\omega$
- Spectral mixture (Wilson 2013): $S(\omega) = \sum_q w_q\,\mathcal{N}(\omega; \mu_q, \Sigma_q)$
- Deep kernel: $k_\theta(x,x') = k_{\mathrm{base}}(\phi_\theta(x), \phi_\theta(x'))$
- ArcCosine-$n$ (Cho & Saul 2009): $k_n(x,x') = \tfrac{1}{\pi}\|x\|^n\|x'\|^n J_n(\theta)$, NN-correspondence

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 2.5 | Spectral kernels — visual guide | P `spectral_kernel_models` | 🧱 🔁 | |
| 2.6 | Deep kernels (NN-warped inputs) | R `pyroxgp/04_svgp_rff_nn` | 🌉 🔁 | |
| 2.7 | ArcCosine kernel (NN-correspondence) | — | 🧱 🔁 | **GAP** — dd:features/gp/gpflow.md |
| 2.14 | Spectral Mixture (SM) kernel fitting — auto-discover periodicity from data (Wilson & Adams 2013) | — | 🔬 | **GAP** — visualise learned spectral components |

### 2.C — Multi-output kernels

**Key equations / models:**
- LMC (Linear Model of Coregionalization): $f_p(x) = \sum_q a_{pq}\,g_q(x)$, $k_{pq}(x,x') = \sum_q a_{pq}a_{qq'} k_q(x,x')$
- ICM (rank-1 LMC): $K = B \otimes K_X$ with output covariance $B$
- OILMM: orthogonal projection $H$ such that $H^\top H = I$ decouples outputs

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 2.8 | Multi-output: LMC, ICM, OILMM | P `multioutput_gp` | 🧱 | |
| 2.9 | OILMM mechanics: project / back-project | — | 🧱 | **GAP** — api: `oilmm_project`, `oilmm_back_project` |

### 2.D — Spherical / localized kernels

**Key equations / models:**
- Spherical harmonics: $Y_{\ell m}$ on $S^2$, zonal kernels $k(x,x') = \sum_\ell a_\ell P_\ell(x^\top x')$
- Spherical Slepian: solve $\int_R Y^*_{\ell m}Y_{\ell'm'}\,d\Omega \cdot c = \lambda c$ to maximize energy in region $R$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 2.10 | Slepian Positional Encodings (spherical, localized) | — | 🧱 🔁 | **GAP** — gh:pyrox#125 |

### 2.E — Kernel-based statistics & utilities

**Key equations / models:**
- Centered kernel: $\tilde K = HKH$, $H = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top$
- HSIC: $\tfrac{1}{n^2}\,\mathrm{tr}(K_x H K_y H)$
- MMD²: $\mathbb{E}_{x,x'}[k(x,x')] + \mathbb{E}_{y,y'}[k(y,y')] - 2\,\mathbb{E}_{x,y}[k(x,y)]$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 2.11 | Kernel centering & KPCA | — | 🧱 | **GAP** — api: `center_kernel`, `centering_operator` |
| 2.12 | Kernel-based statistics: HSIC & MMD | — | 🧱 | **GAP** — api: `hsic`, `mmd_squared` |

### 2.F — Non-Euclidean & operator-valued kernels

**Key equations / models:**
- Operator-valued kernel: $K(x,x') \in \mathcal{L}(\mathcal{Y},\mathcal{Y})$, predicts function-valued outputs (velocity fields, spectral curves)
- Graph heat kernel: $k(u,v) = \exp(-tL)_{uv}$ for graph Laplacian $L = D - A$
- Geodesic RBF on manifold: $k(x,x') = \sigma^2\exp(-d_g(x,x')^2/2\ell^2)$, $d_g$ = geodesic distance

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 2.15 | Operator-valued kernel regression — function-valued outputs (velocity fields, spectral curves) | — | 🔬 | **GAP** |
| 2.16 | GP regression on graphs — heat kernel $k(u,v)=\exp(-tL)_{uv}$ | — | 🔬 | **GAP** |
| 2.17 | GP regression on Riemannian manifolds — geodesic distance kernels | — | 🔬 | **GAP** |

## Part 3 — Exact GP Regression

### 3.A — Foundations

**Key equations / models:**
- GP prior: $f \sim \mathcal{GP}(0, k)$ with $y = f(x) + \epsilon$, $\epsilon \sim \mathcal{N}(0,\sigma^2)$
- Posterior mean: $m_*(x_*) = k_*^\top (K + \sigma^2 I)^{-1} y$
- Posterior variance: $v_*(x_*) = k(x_*,x_*) - k_*^\top (K + \sigma^2 I)^{-1} k_*$
- Log marginal likelihood: $\log p(y) = -\tfrac{1}{2}y^\top(K+\sigma^2 I)^{-1}y - \tfrac{1}{2}\log|K+\sigma^2 I| - \tfrac{n}{2}\log 2\pi$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 3.1 | Kernel ridge regression / GP "hello world" | G `kernel_regression` | 🧱 | |
| 3.2 | Exact GP regression — three patterns | P `exact_gp_regression` | 🧱 | |
| 3.3 | Hyperparameter learning: marginal likelihood | — | 🧱 | **GAP** |
| 3.8 | GP regression with mean function — constant / linear / NN mean; posterior shift under strong prior | — | 🧱 | **GAP** |
| 3.9 | Empirical Bayes / type-II MLE for hyperparameter priors — log-normal priors on $\ell, \sigma^2$, joint optimisation | — | 🧱 | **GAP** |
| 3.10 | Batch GP regression — `vmap` over $B$ independent GPs simultaneously | — | 🧱 | **GAP** — canonical JAX pattern |
| 3.11 | GPU-accelerated exact GP / tile-based Cholesky — block-Cholesky for exact GPs up to $N\approx 50k$ | — | 🔬 | **GAP** |

### 3.B — Diagnostics

**Key equations / models:**
- LOO-CV (LOVE): $\mu_{-i} = y_i - [(K+\sigma^2 I)^{-1}y]_i / [(K+\sigma^2 I)^{-1}]_{ii}$
- Probability integral transform (PIT) for calibration
- Coverage at $1-\alpha$: empirical fraction of $y_i \in [\mu_i \pm z_{\alpha/2}\sigma_i]$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 3.4 | LOVE: fast leave-one-out CV | G `love_crossval` | 🌉 | |
| 3.5 | Predictive variance & calibration diagnostics | — | 🧱 | **GAP** |
| 3.12 | Bayesian model selection — compare kernels via log marginal likelihood, Bayes factors, WAIC | — | 🧱 | **GAP** |
| 3.13 | Predictive distribution anatomy — decompose posterior mean vs variance; under/oversmoothing regimes | — | 🧱 | **GAP** — pedagogical |

### 3.C — Heteroscedastic noise

**Key equations / models:**
- Two-GP heteroscedastic: $y = f(x) + g(x)\,\epsilon$, with $f \sim \mathcal{GP}(0,k_f)$ and $\log g \sim \mathcal{GP}(0,k_g)$
- Joint ELBO over $(f, \log g)$ via posterior linearization or coupled cubature

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 3.6 | Heteroscedastic GP — two coupled latent GPs | — | 🧱 | **GAP** — dd:examples/gp/moments.md |

### 3.D — High-level API

**Key equations / models:**
- `GPEstimator(kernel, ...).fit(X, y).predict(X*, quantiles=...)` — sklearn-style facade

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 3.7 | sklearn-style `GPEstimator` facade | — | 🧱 | **GAP** — gh:pyrox#71 |

### 3.E — Constrained & Physics-informed GPs

**Key equations / models:**
- Monotone GP: derivative observations $f'(x)\ge 0$ via linear operator on prior; or projection to monotone function space
- Convex GP: Hessian positivity $\nabla^2 f(x) \succeq 0$ as inequality constraint
- Boundary-condition GP: zero mean at domain boundary $\partial\Omega$ via Dirichlet eigenfunction basis $\phi_j$ with $\phi_j\vert_{\partial\Omega}=0$
- PDE-constrained GP (Raissi et al.): encode $\mathcal{L}f = 0$ as derivative / linear-operator observations; cross-covariance $k_{\mathcal{L}}(x,x') = \mathcal{L}_x k(x,x')$
- Student-t process: $f \sim \mathcal{T}\nu(\mu, K)$; posterior analytically tractable with heavier tails than GP

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 3.14 | Monotone GP — derivative observations or monotone projection | — | 🔬 | **GAP** |
| 3.15 | Convex GP — Hessian positivity constraints | — | 🔬 | **GAP** |
| 3.16 | Boundary-condition GP — zero mean at domain boundary via eigenfunction basis | — | 🔬 | **GAP** |
| 3.17 | PDE-constrained GP — encode $\mathcal{L}f=0$ as linear operator observations (Raissi et al.) | — | 🔬 | **GAP** — generalises monotone GP to arbitrary linear operators |
| 3.18 | Student-t process — heavier-tailed alternative to GP with tractable posterior | — | 🧱 | **GAP** |

## Part 4 — Structured GPs

GPs whose covariance has direct algebraic structure (Kronecker, Toeplitz, grid, sparse-precision) — exploited by Part 1 operators.

### 4.A — Kronecker GPs

**Key equations / models:**
- 2D-grid GP: $K = K_x \otimes K_t$, solve in $O(n_x^3 + n_t^3)$ via Roth's lemma
- Kronecker + low-rank: $K = K_0 + UU^\top$, posterior via Woodbury
- Sum-of-Kronecker: $K = \sum_i K_i^x \otimes K_i^t$ (additive separable)
- Additive decomposition: $f(x,t) = f_\text{trend}(t) + f_\text{seasonal}(t) + f_\text{spatial}(x,t)$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 4.1 | GPs on 2D grids with Kronecker structure | G `gp_2d_grid` | 🌉 | |
| 4.2 | Combined Kronecker + low-rank | G `structured_gp` | 🌉 | |
| 4.3 | Sum-of-Kronecker (additive space + time) | R `kronecker/01_spain_extremes` (uses) | 🔬 | could break out a fundamental |
| 4.4 | Separable spatiotemporal & additive (trend + seasonal + residual) | — | 🧱 | **GAP** — dd:examples/gp/moments.md |
| 4.5 | Kronecker marginal log-likelihood & posterior predictive | — | 🧱 | **GAP** — api: `kronecker_mll`, `kronecker_posterior_predictive` |

### 4.B — Grid / Toeplitz GPs

**Key equations / models:**
- KISS-GP / SKI: $K \approx W^\top K_U W$ with cubic local interpolation $W$ to grid points $U$
- Toeplitz matvec via FFT: $T v = \mathrm{IFFT}(\hat t \odot \mathrm{FFT}(v))$, $O(n\log n)$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 4.6 | KISS-GP / SKI on grids | — | 🧱 | **GAP** — api: `InterpolatedOperator`, `cubic_interpolation_weights`, `grid_data`, `create_grid` |
| 4.7 | Lattice / Toeplitz GPs for stationary 1D | — | 🧱 | **GAP** — pairs with 1.4 |

### 4.C — Sparse-precision (mesh / GMRF)

**Key equations / models:**
- SPDE (Lindgren et al. 2011): $(\kappa^2 - \Delta)^{\alpha/2} f(x) = \mathcal{W}(x)$ → Matérn kernel
- FEM precision: $Q = \kappa^4 C + 2\kappa^2 G + GC^{-1}G$ on triangulated mesh, sparse banded

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 4.8 | SPDE / FEM Matérn — triangulated mesh GMRF, O(n^{3/2}) | — | 🌉 | **GAP** — gh:pyrox#50, dd:features/gp/spde_fem.md |

## Part 5 — Approximations & Scalability

GPs that scale to large N via inducing points, random features, or iterative solvers — distinct from Part 4 in that the structure is *imposed* rather than inherent to the data geometry.

### 5.A — Random features

**Key equations / models:**
- RFF (Rahimi & Recht 2007): $\phi(x) = \sqrt{2/D}\cos(\omega^\top x + b)$, $\omega\sim S(\omega)$, $k(x,x') \approx \phi(x)^\top\phi(x')$
- SSGP / VSSGP: BLR in RFF space, hierarchical / variational priors over $\omega$
- Nyström: $K \approx K_{nm}K_{mm}^{-1}K_{mn}$, rank-$m$ approximation
- FastFood: $W = SHG\Pi HB$ — structured frequency matrix, $O(D\log d)$ matvec

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 5.1 | Random Fourier Features → SSGP → VSSGP | P `random_fourier_features` | 🧱 🔁 | |
| 5.2 | Kernel approximations: Nyström vs RFF | G `kernel_approximations`, P `kernel_approximation` | 🧱 | **DUP** — pick one home |
| 5.3 | FastFood structured random features | — | 🧱 | **GAP** — gh:gaussx#62 |

### 5.B — Inducing-point fundamentals

**Key equations / models:**
- Inducing variables: $u = f(Z)$ with $Z \in \mathbb{R}^{m\times d}$; FITC: $K \approx Q_{nn} + \mathrm{diag}(K_{nn} - Q_{nn})$, $Q_{nn} = K_{nm}K_{mm}^{-1}K_{mn}$
- SVGP ELBO (Hensman 2013): $\mathcal{L} = \sum_i \mathbb{E}_{q(f_i)}[\log p(y_i\mid f_i)] - \mathrm{KL}(q(u)\Vert p(u))$
- Whitened: $u = L_{mm}\tilde u$, $\tilde u \sim \mathcal{N}(0,I)$ → isotropic optimization
- Collapsed ELBO (Titsias 2009): $\mathcal{L} = \log\mathcal{N}(y;0, Q_{nn}+\sigma^2 I) - \tfrac{1}{2\sigma^2}\mathrm{tr}(K_{nn}-Q_{nn})$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 5.4 | Inducing point methods (FITC, DTC, VFE) — theory | — | 🧱 | **GAP** |
| 5.5 | Sparse Variational GP (Titsias/Hensman) | G `sparse_variational_gp`, R `pyroxgp/01_svgp_standard` | 🌉 | **DUP** |
| 5.6 | Whitening mechanics: `whiten_covariance`, `unwhiten`, `unwhiten_covariance` | — | 🧱 | **GAP** |
| 5.7 | Whitened SVGP & Bayesian linear regression view | G `whitened_svgp` | 🌉 🔁 | |
| 5.8 | Collapsed ELBO | — | 🧱 | **GAP** — api: `collapsed_elbo` |
| 5.9 | Mini-batched SVGP / stochastic VI | R `pyroxgp/02_svgp_batched` | 🔬 | |
| 5.10 | Full SVGP tutorial — 6 guide families incl. orthogonal decoupled | — | 🧱 | **GAP** — dd:examples/gp/svgp_numpyro.py |
| 5.19 | Collapsed vs uncollapsed SVGP — explicit comparison of Titsias vs Hensman objectives, bias/variance tradeoff | — | 🧱 | **GAP** — pedagogical |
| 5.20 | Online sparse GP (Csató & Opper 2002) — sequential Bayesian update of inducing set without full retraining | — | 🔬 | **GAP** — complements streaming filter tutorial 7.33 |

### 5.C — Inter-domain features

**Key equations / models:**
- Inter-domain inducing variables: $u_j = \langle f, g_j \rangle_\mathcal{H}$ for chosen basis $\{g_j\}$
- VFF (Hensman 2018): Fourier basis on $[a,b]$, $K_{uu}$ diagonal, $O(NM)$
- VISH (Dutordoir 2020): spherical harmonics on $S^2$, Funk–Hecke gives diagonal $K_{uu}$
- Laplacian eigenfns: $-\Delta\phi_j = \lambda_j \phi_j$ on manifold/graph; $S(\lambda_j)$ from kernel spectral density
- Decoupled: separate basis for posterior mean (large) and covariance (small)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 5.11 | VISH — Variational Inducing Spherical Harmonics | R `pyroxgp/03_svgp_spherical_harmonics` | 🔬 | gh:pyrox#49 |
| 5.12 | VFF — Variational Fourier Features (bounded interval, diagonal K_uu) | — | 🧱 | **GAP** — gh:pyrox#49, dd:features/gp/inducing_features.md |
| 5.13 | Laplacian-eigenfunction inducing features (manifolds, graphs) | — | 🧱 | **GAP** — gh:pyrox#49 |
| 5.14 | Decoupled inter-domain features (mixed spatial + spectral) | — | 🧱 | **GAP** — gh:pyrox#49 |

### 5.D — Iterative-solver scaling

**Key equations / models:**
- CG-based GP: $\alpha = (K+\sigma^2 I)^{-1} y$ via CG; logdet via SLQ
- CGLB: $\log|A| \geq -\sum \log(1-\theta_k)$ from Lanczos eigenvalue bounds
- Preconditioned CG: $M^{-1}Ax = M^{-1}b$ with Nyström / pivoted-Cholesky preconditioner
- EigenPro: SGD with eigen-spectrum preconditioner $\Lambda^{-1}$
- Falkon (Rudi 2017): Newton iteration on Nyström-reduced system, $O(N\sqrt{N}\log(1/\epsilon))$
- LogFalkon: extends Falkon to GSC losses (logistic, exponential)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 5.15 | CG for exact GPs at scale + CGLB | — | 🧱 | **GAP** — pairs with 1.16; dd:features/gp/gpflow.md |
| 5.16 | EigenPro spectral preconditioning | — | 🧱 | **GAP** — gh:gaussx#63 |
| 5.17 | Falkon: Nyström preconditioner + solve recipe | — | 🌉 | **GAP** — gh:gaussx#49 |
| 5.18 | LogFalkon / GSC-Falkon — Newton outer + preconditioned CG | — | 🌉 | **GAP** — gh:pyrox#50, dd:features/gp/logfalkon.md |

### 5.E — Deep GPs

**Key equations / models:**
- Deep GP (Salimbeni & Deisenroth 2017): $f^{(L)} \circ \cdots \circ f^{(1)}$, each layer $f^{(l)}\sim\mathcal{GP}$
- Doubly stochastic ELBO: $\mathcal{L} = \sum_n\mathbb{E}_{q(f^{1:L}_n)}[\log p(y_n\mid f_n^{(L)})] - \sum_l\mathrm{KL}(q(u^{(l)})\Vert p(u^{(l)}))$
- Convolutional GP (van der Wilk 2017): patch-level inducing features, $f(x) = \sum_p f_p(x_p)$, $f_p \sim \mathcal{GP}$ over patches
- Inter-domain inducing variables per patch; $K_{uu}$ exploits translation-equivariance

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 5.21 | Deep GP — doubly stochastic VI ELBO, layer-by-layer sampling (Salimbeni & Deisenroth 2017) | — | 🔬 | **GAP** — clear gap in Part 5 hierarchy |
| 5.22 | Convolutional GP — patch-level inducing features for image data (van der Wilk 2017) | — | 🔬 | **GAP** — natural extension after VISH/VFF |

## Part 6 — Non-Conjugate Likelihoods & Inference

### 6.A — Likelihood & integrator zoos

**Key equations / models:**
- Generic non-conjugate factorization: $p(y\mid f) = \prod_i p(y_i\mid f_i)$
- Bernoulli ($\sigma(f)$), Poisson ($\exp(f)$), Student-t, Beta, Gamma, Exponential, Softmax
- ELL: $\mathbb{E}_{q(f)}[\log p(y\mid f)]$ via integrator
- Gauss–Hermite: $\int g(z)\mathcal{N}(z;\mu,\sigma^2)\,dz \approx \sum_i w_i\, g(\mu+\sqrt 2\sigma\,\xi_i)$
- Sigma points (Unscented): $2d+1$ deterministic points
- 5th-order cubature: symmetric $2d^2+1$ points

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 6.1 | Likelihood zoo: Bernoulli, Poisson, StudentT, Softmax, Heteroscedastic, Exponential, Beta, Gamma, Multi-latent | — | 🧱 | **GAP** — gh:pyrox#48 |
| 6.2 | Integrator zoo: Gauss–Hermite, MC, Unscented, Taylor, Assumed-Density Filter | — | 🧱 | **GAP** — api: gaussx `_quadrature` |
| 6.3 | Sigma points & cubature | — | 🧱 | **GAP** — api: `sigma_points`, `cubature_points` |
| 6.4 | Fifth-order symmetric cubature integrator | — | 🧱 | **GAP** — gh:gaussx#26 |
| 6.5 | Statistical Linear Regression via cubature (SLR) | — | 🧱 | **GAP** — gh:gaussx#25 |

### 6.B — Classification

**Key equations / models:**
- Latent GP classification: $f\sim\mathcal{GP}$, $y\sim\mathrm{Bernoulli}(\sigma(f))$
- Multi-class: $y\sim\mathrm{Cat}(\mathrm{softmax}(f_1,\dots,f_K))$, $K$ latent GPs

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 6.6 | Latent GP classification — three patterns (Bernoulli + softmax) | P `latent_gp_classification` | 🧱 | extend to multi-class per dd |

### 6.C — Newton / Gauss-Newton family

**Key equations / models:**
- Laplace: $q(f) = \mathcal{N}(\hat f, -H^{-1})$, $\hat f = \arg\max p(f\mid y)$, $H = \nabla^2\log p(f\mid y)$
- Newton update: $f \leftarrow f - H^{-1}\nabla$ with damping
- Gauss-Newton / GGN: $H \approx J^\top R J$ (drops 2nd-order terms)
- Posterior linearization (SLR): $p(y\mid f) \approx \mathcal{N}(y; Af + b, \Omega)$ matched on Gaussian
- Hutchinson Hessian diag: $\mathrm{diag}(H) \approx \mathbb{E}[z\odot Hz]$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 6.7 | Laplace approximation | P `advanced_gp_laplace` | 🧱 | |
| 6.8 | Gauss–Newton inference | P `advanced_gp_gauss_newton` | 🧱 | |
| 6.9 | Quasi-Newton inference (L-BFGS sites) | P `advanced_gp_qn` | 🧱 | |
| 6.10 | Posterior Linearization (Bayes-Newton) | P `advanced_gp_pl` | 🧱 | |
| 6.11 | Newton & damped natural updates | — | 🧱 | **GAP** — api: `newton_update`, `damped_natural_update` |
| 6.12 | Gauss–Newton & GGN diagonal | — | 🧱 | **GAP** — api: `gauss_newton_precision`, `ggn_diagonal` |
| 6.13 | Hutchinson Hessian diagonal & Riemannian PSD correction | — | 🧱 🔁 | **GAP** |

### 6.D — Variational inference

**Key equations / models:**
- Variational guides: delta · diagonal mean-field · low-rank ($S = VV^\top + \mathrm{diag}$) · full-rank Cholesky · normalizing flow · whitened
- ELBO: $\log p(y) \geq \mathbb{E}_q[\log p(y,f)] - \mathbb{E}_q[\log q(f)]$
- Natural gradient: $\tilde\nabla \mathcal{L} = F^{-1}\nabla \mathcal{L}$, $F$ = Fisher
- CVI sites (Khan & Lin 2017): $\eta^{(t+1)} = (1-\rho)\eta^{(t)} + \rho\,\hat\eta_{\mathrm{tilted}}$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 6.14 | Variational guides — full-rank, mean-field, low-rank, whitened, delta, flow | — | 🧱 | **GAP** — dd:examples/gp/vgp_numpyro.py + features/gp/variational_families.md |
| 6.15 | Natural gradient VI | G `natural_gradient_vi` | 🌉 | |
| 6.16 | Conjugate VI for GPs (CVI sites) | — | 🧱 | **GAP** — api: `cvi_update_sites`, `site_natural_from_tilted` |
| 6.23 | Full VGP (non-sparse) — $N$ variational parameters, $O(N^3)$, no inducing-point approximation | — | 🧱 | **GAP** — closes gap between sparse and exact tutorials |

### 6.E — Expectation Propagation

**Key equations / models:**
- Cavity: $q^{-i}(f_i) \propto q(f_i) / t_i(f_i)$
- Tilted: $\hat q(f_i) \propto q^{-i}(f_i)\, p(y_i\mid f_i)$
- Site update: choose $t_i$ such that $\mathrm{KL}(\hat q\,\Vert\, q^{-i} t_i)$ is minimized → moment match

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 6.17 | Expectation Propagation | P `advanced_gp_ep`, G `expectation_propagation` | 🌉 | **DUP** |
| 6.18 | EP cavity & tilted moments mechanics | — | 🧱 | **GAP** — api: `cavity_distribution`, `ep_tilted_moments`; gh:gaussx#24 |

### 6.F — Bayesian linear regression & non-standard outputs

**Key equations / models:**
- BLR posterior: $\Sigma = (\Phi^\top R^{-1}\Phi + S_0^{-1})^{-1}$, $\mu = \Sigma\,\Phi^\top R^{-1}y$
- Sequential update via Sherman–Morrison: rank-1 covariance update on each new observation
- Log-Gaussian Cox Process: $\lambda(x) = \exp(f(x))$, observations from Poisson process
- Warped GP (Snelson 2003): $g(y) = f(x)$ with monotone bijection $g$, transformed likelihood

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 6.19 | Bayesian linear regression updates | — | 🧱 🔁 | **GAP** — api: `blr_diag_update`, `blr_full_update` |
| 6.20 | Log-Gaussian Cox Process (spatial point-process intensity) | — | 🔬 | **GAP** — dd:examples/gp/moments.md |
| 6.21 | Warped GP (Box–Cox for skewed targets) | — | 🧱 | **GAP** — dd:examples/gp/moments.md |
| 6.24 | Warped GP with normalizing flows — learnable bijection $g$ extends Box–Cox to NF-parameterized warpings | — | 🔬 | **GAP** |

### 6.G — Aggregate Bayesian methods

**Key equations / models:**
- INLA (Rue et al. 2009): $p(\theta\mid y) \approx p(y\mid x^*,\theta)p(x^*\mid\theta)p(\theta)/q(x^*\mid y,\theta)$ at Laplace mode $x^*$
- Numerical integration over $\theta$ on grid, marginal posteriors of latent field

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 6.22 | R-INLA port — integrated nested Laplace approximation | — | 🌉 | **GAP** — gh:gaussx#155 |

## Part 7 — Markov / State-Space GPs

### 7.A — Foundations

**Key equations / models:**
- Discrete SSM: $x_{k+1} = A_k x_k + w_k$, $y_k = H_k x_k + v_k$, $w_k\sim\mathcal{N}(0,Q_k)$, $v_k\sim\mathcal{N}(0,R_k)$
- Kalman predict: $\bar x = A x$, $\bar P = A P A^\top + Q$
- Kalman update: $S = H\bar P H^\top + R$, $K = \bar P H^\top S^{-1}$, $x^+ = \bar x + K(y - H\bar x)$
- RTS smoother: backward $G_k = P_k A_k^\top \bar P_{k+1}^{-1}$
- Joseph form: $P^+ = (I-KH)\bar P(I-KH)^\top + KRK^\top$ (numerically stable)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 7.1 | Kalman filter + RTS smoother (pure SSM) | G `kalman_filter` | 🧱 | |
| 7.2 | SSM ↔ natural / expectation parameterizations | — | 🧱 | **GAP** — api: `ssm_to_naturals`, `naturals_to_ssm`, `expectations_to_ssm` |
| 7.3 | Pairwise marginals & sites | — | 🧱 | **GAP** — api: `pairwise_marginals`, `GaussianSites`, `sites_to_precision` |
| 7.4 | SDE autocovariance & process noise | — | 🧱 | **GAP** — api: `sde_autocovariance`, `process_noise_covariance` |
| 7.5 | Joseph-form Kalman update standalone | — | 🧱 | **GAP** |

### 7.B — SDE kernel zoo

**Key equations / models:**
- LTI SDE: $dx = Fx\,dt + L\,dW$, observation $f(t) = Hx(t)$
- Kernel ↔ SDE map (Hartikainen & Särkkä 2010): Matérn-3/2 → 2D LTI; Matérn-5/2 → 3D
- Discretization: $A_k = \exp(F\Delta t_k)$, $Q_k = P_\infty - A_k P_\infty A_k^\top$ via Lyapunov
- Periodic: truncated Fourier, block-diagonal $F$ · QuasiPeriodic: Matérn × Periodic via $\oplus$
- Drift-KL: $\mathrm{KL}(p\Vert q) = \int_0^T \tfrac{1}{2}\|f_p - f_q\|^2_{\Sigma^{-1}}\,dt$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 7.6 | Matérn kernels in state-space form | P `markov_gp_sde_kernels` | 🧱 | |
| 7.7 | Full SDE kernel zoo: Periodic, QuasiPeriodic, Cosine, Constant, Sum, Product, Subband Matérn | — | 🧱 | **GAP** — api: gaussx `_ssm` SDE kernels |
| 7.8 | SDE linearization & drift-KL helpers | — | 🧱 | **GAP** — gh:gaussx#70 |

### 7.C — Markov GP workflows

**Key equations / models:**
- Marginal log-likelihood: $\log p(y) = \sum_k \log\mathcal{N}(y_k; H\bar x_k, S_k)$ from filter pass
- Hyperparameter learning: gradient through Kalman filter
- Sparse variational Markov GP: ELBO via filter over inducing time points $u_{1:M}$
- KalmanGuide: pseudo-observations $\tilde y_k$, $\tilde R_k$ → standard Kalman + RTS for posterior

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 7.9 | Markov GP with Kalman filtering | P `markov_gp_kalman` | 🧱 | |
| 7.10 | Markov GP hyperparameter training | P `markov_gp_training` | 🧱 | |
| 7.11 | Non-Gaussian Markov GP | P `markov_gp_nongauss` | 🧱 | |
| 7.12 | Sparse variational Markov GP | P `sparse_markov_gp` | 🧱 | |
| 7.13 | KalmanGuide — Bayes-Newton via pseudo-observations + RTS | — | 🧱 | **GAP** — dd:features/gp/variational_families.md |

### 7.D — Parallel & scalable filtering

**Key equations / models:**
- Parallel scan (Särkkä & García-Fernández 2021): associative op $\otimes$ on $(A_k, b_k)$ pairs, $O(\log N)$ depth
- Square-root form: propagate $\sqrt P$ instead of $P$ for numerical stability
- SpInGP: parallel-in-time + sparse (banded) state representation
- Mean-field Kalman: block-diagonal $P$ across independent dimensions

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 7.14 | Parallel / batched Kalman filter | G `parallel_kalman` | 🌉 | |
| 7.15 | Square-root parallel Kalman filter / RTS | — | 🧱 | **GAP** — gh:gaussx#165 |
| 7.16 | SpInGP — sparse parallel-in-time GP | — | 🧱 | **GAP** — api: `spingp_log_likelihood`, `spingp_posterior` |
| 7.17 | Mean-field block-diagonal Kalman filter | — | 🧱 | **GAP** — gh:gaussx#29 |

### 7.E — Nonlinear filtering

**Key equations / models:**
- EKF: linearize $f, h$ around $\hat x$ via Jacobian
- UKF: propagate $2d+1$ sigma points through $f$, recompute moments
- CKF: $2d$ cubature points, no tuning parameter
- Innovation cov as `LowRankUpdate` when $R$ has structure: $S = H\bar P H^\top + R$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 7.18 | Nonlinear Gaussian Filter (UKF/EKF generalization) | — | 🧱 | **GAP** — gh:gaussx#161 |
| 7.19 | Extended Kalman Smoother (Taylor(1)) | — | 🧱 | **GAP** — dd:examples/gp/integration_detail.md |
| 7.20 | Unscented Kalman Smoother (PL + SigmaPoints + Kalman) | — | 🧱 | **GAP** — dd:examples/gp/integration_detail.md |
| 7.21 | Cubature Kalman Smoother | — | 🧱 | **GAP** — dd:examples/gp/integration_detail.md |
| 7.22 | Innovation cov as structured `LowRankUpdate` | — | 🧱 | **GAP** — gh:gaussx#164 |

### 7.F — Ensemble methods

**Key equations / models:**
- EnKF analysis: $X^a = X^f + P_e H^\top (HP_eH^\top + R)^{-1}(Y - HX^f)$, $P_e = \tfrac{1}{J-1}(X^f - \bar X^f)(X^f - \bar X^f)^\top$
- Bessel-corrected covariance: divide by $J-1$ not $J$
- Ensemble Kalman gain: low-rank $K_e = X^f X^{f\top}H^\top S^{-1}$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 7.23 | Ensemble Kalman Filter on Lorenz-63 | G `ensemble_kalman` | 🔬 | |
| 7.24 | Bessel-corrected EnKF + `ensemble_kalman_gain` | — | 🌉 | **GAP** — gh:gaussx#127 |

### 7.G — Steady-state & structured-Gaussian surfaces

**Key equations / models:**
- DARE: $P = APA^\top + Q - APH^\top(HPH^\top+R)^{-1}HPA^\top$ → unique SPD solution
- Infinite-horizon Kalman: solve DARE once, reuse $K_\infty$
- MarkovGaussian surface: structured Gaussian with $(A_k, Q_k)$ as PyTree, exposes filter/smoother/sample

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 7.25 | Infinite-horizon Kalman & DARE | — | 🧱 | **GAP** — api: `infinite_horizon_filter/smoother`, `dare` |
| 7.26 | DARE via Optimistix fixed-point + implicit diff | — | 🧱 | **GAP** — gh:gaussx#97 |
| 7.27 | MarkovGaussian structured surface | — | 🧱 | **GAP** — gh:gaussx#76 |
| 7.28 | Spatiotemporal SDE GPs | — | 🔬 | **GAP** |

### 7.H — Non-conjugate temporal case studies

**Key equations / models:**
- Laplace + Kalman: iterate site moments $(\mu_k, \tau_k)$ via Newton on each $p(y_k\mid f_k)$, run filter+smoother per iteration
- EP + Kalman: same loop with EP cavity / tilted moments
- Changepoints (additive): $f(t) = f_\text{trend}(t) + f_\text{fast}(t)$, Matérn-5/2 + Matérn-1/2
- Streaming filter-only: discard $x_{k<N-W}$ for fixed window $W$
- Time-varying lengthscale: $\ell(t)$ as random walk in `numpyro.scan`

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 7.29 | GP classification with Laplace + Kalman | — | 🧱 | **GAP** — dd:examples/gp/state_space.md |
| 7.30 | Poisson counts with EP + Kalman | — | 🧱 | **GAP** — dd:examples/gp/state_space.md |
| 7.31 | Changepoint detection via additive temporal GPs | — | 🌉 | **GAP** — dd:examples/gp/state_space.md |
| 7.32 | Latent temporal GP in a BHM | — | 🌉 | **GAP** — dd:examples/gp/state_space.md |
| 7.33 | Online / streaming GP (filter-only mode) | — | 🌉 | **GAP** — dd:examples/gp/state_space.md |
| 7.34 | Non-LTI temporal model via `numpyro.scan` | — | 🧱 | **GAP** — dd:examples/gp/state_space.md |

## Part 8 — Sampling, Pathwise, Conditioning

### 8.A — Pathwise sampling

**Key equations / models:**
- Pathwise (Wilson 2020): $f^*(x) = \underbrace{\sum_i w_i\phi_i(x)}_\text{prior basis} + \underbrace{k(x,X)\beta}_\text{update}$ with $\beta = (K+\sigma^2 I)^{-1}(y - \Phi w - \epsilon)$
- Decoupled SVGP sampling: parametric prior + non-parametric update from inducing variables only

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 8.1 | Pathwise GP posterior sampling (Wilson 2020) | P `gp_pathwise` | 🧱 | |
| 8.2 | Pathwise sampling with NumPyro | P `gp_pathwise_numpyro` | 🧱 | |
| 8.3 | Decoupled sampling for SVGP | — | 🧱 | **GAP** |

### 8.B — Matheron's-rule conditioning

**Key equations / models:**
- Matheron's rule: $f\mid y \stackrel{d}{=} f + K_*(K+\sigma^2 I)^{-1}(y - f(X) - \epsilon)$ where $f, \epsilon$ are independent prior samples
- Partitioned joint: factor $p(f_a, f_b\mid y)$ as $p(f_a\mid y)\,p(f_b\mid f_a, y)$ with shared Cholesky

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 8.4 | Matheron's-rule conditioning by sampling | — | 🧱 | **GAP** — gh:gaussx#77 |
| 8.5 | Partitioned joint conditional sampling | — | 🧱 | **GAP** — gh:gaussx#79 |

## Part 9 — Uncertainty Propagation & UQ

### 9.A — Foundations

**Key equations / models:**
- Moment matching: match $\mathbb{E}[g(x)]$ and $\mathrm{Var}[g(x)]$ for $x\sim\mathcal{N}(\mu,\Sigma)$
- Linearization (Taylor-1): $g(x) \approx g(\mu) + \nabla g(\mu)(x-\mu)$
- Unscented: sigma points $\mu \pm \sqrt{(d+\kappa)\Sigma}_i$
- Gauss–Hermite ELL: $\mathbb{E}_{\mathcal{N}(\mu,\sigma^2)}[g(z)] = \sum_i w_i g(\mu+\sqrt 2\sigma\xi_i)$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 9.1 | Moment matching, unscented transform, linearization | — | 🧱 | **GAP** |
| 9.2 | Gauss–Hermite quadrature for ELL | — | 🧱 | **GAP** — used in R kronecker series |

### 9.B — Uncertain inputs

**Key equations / models:**
- GP at uncertain $x^*\sim\mathcal{N}(\mu_*,\Sigma_*)$: $\mathbb{E}[f^*] = \mathbb{E}_{x^*}[m(x^*)]$, $\mathrm{Var}[f^*] = \mathbb{E}[v(x^*)] + \mathrm{Var}[m(x^*)]$
- PILCO chain: iterate moment-matched GP predictions $h$ steps ahead, track $(\mu_h, \Sigma_h)$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 9.3 | Uncertainty propagation through nonlinear functions | G `uncertainty_propagation` | 🌉 | |
| 9.4 | GPs with uncertain inputs (PILCO-style) | G `uncertain_gp_inputs` | 🔬 | |
| 9.5 | Multi-step-ahead PILCO autoregressive forecasting | — | 🔬 | **GAP** — dd:examples/gp/integration_detail.md |

### 9.C — Analytic moments

**Key equations / models:**
- Ψ-statistics for RBF (Titsias & Lawrence 2010):
  - $\Psi_0 = \mathbb{E}[k(x^*,x^*)]$
  - $\Psi_1 = \mathbb{E}[k(x^*, Z)]$
  - $\Psi_2 = \mathbb{E}[k(x^*,Z)^\top k(x^*,Z)]$
- Closed form for RBF: products of Gaussian integrals with kernel parameters

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 9.6 | Ψ-statistics & exact RBF closed form for uncertain inputs | — | 🧱 | **GAP** — api: `compute_psi_statistics`, `AnalyticalPsiStatistics` |
| 9.7 | Uncertain SVGP / VGP prediction (sigma-point + analytic) | — | 🧱 | **GAP** — api: `uncertain_svgp_predict`, `uncertain_vgp_predict` |
| 9.8 | Cost / mean / gradient expectations under Gaussian inputs | — | 🧱 | **GAP** — api: `cost_expectation`, `mean_expectation`, `gradient_expectation` |

### 9.D — BGPLVM

**Key equations / models:**
- Bayesian GPLVM: $X\sim q(X)$, $Y = f(X)+\epsilon$, marginalize via Ψ-statistics
- ELBO uses $\mathbb{E}_{q(X)}[\Psi_0], \mathbb{E}_{q(X)}[\Psi_1], \mathbb{E}_{q(X)}[\Psi_2]$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 9.9 | Bayesian GPLVM with uncertain inputs | — | 🔬 | **GAP** — api: `uncertain_bgplvm_predict` |
| 9.12 | GP-LVM (Lawrence 2004) — unsupervised manifold learning vs BGPLVM; Bayesian nonlinear PCA vs PCA/UMAP | — | 🔬 | **GAP** — distinct from 9.9 (Bayesian extension) |
| 9.13 | Supervised GPLVM — classification via latent GP representation | — | 🔬 | **GAP** |

### 9.E — Special integrators & quantiles

**Key equations / models:**
- Mixture quantile root-find: solve $\sum_k \pi_k F_k(q) = \alpha$ via Brent / Optimistix
- Importance-weighted MC: $\mathbb{E}_p[h] \approx \sum_i w_i h(x_i)$, $w_i = p(x_i)/q(x_i)$ for rare-event tails

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 9.10 | Mixture-quantile root-finder | — | 🧱 | **GAP** — gh:gaussx#121 |
| 9.11 | Custom integrator: importance-weighted MC for rare events | — | 🧱 | **GAP** — dd:examples/gp/integration_detail.md |
| 9.14 | GP quadrature / Bayesian cubature — $\int f(x)p(x)\,dx$ with GP prior on $f$; Bayesian numerical integration | — | 🔬 | **GAP** |

## Part 10 — Probabilistic Programming Integration

### 10.A — gaussx + NumPyro

**Key equations / models:**
- `numpyro.factor("gp", log_p)` with `log_p = log_marginal_likelihood`
- Precision-form Gaussian: $f \sim \mathcal{N}(\Lambda^{-1}\eta, \Lambda^{-1})$ for sparse $\Lambda$ (avoids Cholesky of $\Sigma$)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 10.1 | GP regression with NumPyro + gaussx | G `numpyro_gp` | 🧱 | |
| 10.2 | Bayesian linear regression in precision form | G `numpyro_precision` | 🧱 🔁 | |

### 10.B — pyrox patterns

**Key equations / models:**
- Pattern 1 — `eqx.tree_at(model, replace, sampled_values)`
- Pattern 2 — `PyroxModule.pyrox_sample(name, dist)`
- Pattern 3 — `Parameterized` with `register_param` + `set_prior`

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 10.3 | Three-pattern regression masterclass: tree_at / pyrox_sample / Parameterized | P `regression_masterclass_treeat`, `_pyrox_sample`, `_parameterized` | 🧱 🔁 | |

### 10.C — Hierarchical & sampling

**Key equations / models:**
- Hierarchical: $\theta_g \sim p(\theta_g)$, $\theta_l \sim p(\theta_l\mid\theta_g)$, $y\sim p(y\mid\theta_l)$
- NUTS: HMC with auto step size + tree-based termination

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 10.4 | Hierarchical / multi-task GPs in NumPyro | — | 🌉 | **GAP** |
| 10.5 | MCMC for GP hyperparameters (NUTS) | — | 🧱 | **GAP** |

## Part 11 — Ensembles

**Key equations / models:**
- Ensemble predictive: $\hat p(y\mid x) = \tfrac{1}{E}\sum_e p(y\mid x, \theta_e)$
- vmap over PRNG keys: $\theta_e = \mathrm{fit}(D, \mathrm{key}_e)$ for $e=1,\dots,E$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 11.1 | Ensemble primitives — three ways | P `ensemble_primitives_tutorial` | 🧱 | |
| 11.2 | EnsembleMAP & EnsembleVI runners | P `ensemble_runner_tutorial` | 🧱 | |

## Part 12 — Data Pipelines

**Key equations / models:**
- Spatial encoding: lat/lon ↔ Cartesian unit-vector on $S^2$
- Time encoding: cyclic $(\sin(2\pi t/T), \cos(2\pi t/T))$
- Standardization: $\tilde x = (x - \mu)/\sigma$ per dimension

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 12.1 | Spatiotemporal preprocessing (geo + time + pandas) | P `spatiotemporal_preprocessing` | 🧱 | |
| 12.2 | Loading climate data: xarray, zarr, ERA5 | — | 🔬 | **GAP** |

## Part 13 — Applied Case Studies *(research_notebook)*

### 13.A — Spatial extremes

**Key equations / models:**
- GEV CDF: $G(z;\mu,\sigma,\xi) = \exp\!\big({-}(1+\xi(z-\mu)/\sigma)^{-1/\xi}\big)$
- Multiplicative model: $\mu(s,t) = \mu_0(s) + \beta(s)(t - t_0)$ with $\beta(s)$ a spatial GP
- Non-stationary tails: $\sigma(s), \xi(s)$ each as spatial GPs
- Gaussian copula: $C(u,v) = \Phi_\rho(\Phi^{-1}(u), \Phi^{-1}(v))$ on residuals
- BHM: $\mathrm{GEV}(y\mid \mu(s), \sigma(s), \xi(s))$ with GP priors on each parameter
- Time-varying GEV: $(\mu(t), \sigma(t), \xi(t))$ as temporal GPs

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 13.1 | Kronecker GP + GEV likelihood (Spain extremes) | R `kronecker/01_spain_extremes` | 🔬 | |
| 13.2 | Kronecker-multiplicative GP (spatial warming rates) | R `kronecker/02_spain_multiplicative` | 🔬 | |
| 13.3 | Non-stationary GEV (location-dependent tails) | R `kronecker/03_spain_nonstationary` | 🔬 | |
| 13.4 | Gaussian copula spatial dependence | R `kronecker/04_spain_copula` | 🔬 | |
| 13.5 | BHM with GEV + spatial GPs (methane / precipitation extremes) | — | 🔬 | **GAP** — dd:examples/gp/moments.md |
| 13.6 | Temporal extremes: GEV with time-varying μ(t), σ(t), ξ(t) | — | 🔬 | **GAP** — dd:examples/gp/state_space.md |

### 13.B — SVGP applied

**Key equations / models:**
- Mini-batch ELBO: $\hat{\mathcal{L}} = \tfrac{N}{B}\sum_{i\in\mathcal{B}}\mathbb{E}_{q(f_i)}[\log p(y_i\mid f_i)] - \mathrm{KL}(q(u)\Vert p(u))$
- Inter-domain SVGP: $u = \langle f, g_j\rangle$ for spherical-harmonic basis on real climate data

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 13.7 | SVGP on real climate data (large-N) | R `pyroxgp/01–04` | 🔬 | could split: standard / batched / SH / deep-kernel |

### 13.C — Geophysics & emulation

**Key equations / models:**
- GP emulator: $f_\text{sim}(\theta) \approx \mathcal{GP}$ trained on simulator outputs
- somax composition: GP prior on diffusivity field $\kappa(s)$ feeds ocean PDE
- DA composition: learn dynamics $f: x_t \mapsto x_{t+1}$ as GP, plug into EnKF/4D-Var

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 13.8 | GP for ocean / SST / sea-level extremes | — | 🔬 | **GAP** |
| 13.9 | GP emulator for a numerical model | — | 🔬 | **GAP** |
| 13.10 | GP + somax — spatially smooth GP priors for ocean parameters | — | 🔬 | **GAP** — dd:examples/integration.md |
| 13.11 | GP + ekalmX/vardax — learned GP dynamics for DA | — | 🔬 | **GAP** — dd:examples/integration.md |
| 13.16 | Multi-fidelity GP (Kennedy & O'Hagan 2000) — fuse cheap (coarse) + expensive (fine) simulators via autoregressive GP | — | 🔬 | **GAP** |
| 13.17 | ABC-GP emulator — use GP surrogate to bypass expensive likelihood; sample $\theta$ via ABC with GP-matched summary statistics | — | 🔬 | **GAP** |

### 13.D — Optimization & decision

**Key equations / models:**
- Expected Improvement: $\alpha_\mathrm{EI}(x) = \mathbb{E}[\max(0, f^* - y(x))] = (f^*-\mu)\Phi(z) + \sigma\phi(z)$, $z = (f^*-\mu)/\sigma$
- Thompson sampling via pathwise posterior

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 13.12 | Bayesian optimization with GPs (Expected Improvement) | — | 🔬 | **GAP** — dd:examples/gp/integration_detail.md |
| 13.18 | GP-UCB acquisition — Upper Confidence Bound $\alpha_\mathrm{UCB}(x) = \mu(x) + \beta\sigma(x)$ | — | 🔬 | **GAP** |
| 13.19 | Probability of Improvement (PI) — simpler BO baseline alongside EI | — | 🔬 | **GAP** |
| 13.20 | Thompson sampling for BO — use pathwise posterior sampling (connects to 8.1) | — | 🔬 | **GAP** |
| 13.21 | Multi-objective BO — Pareto front approximation via GP surrogates | — | 🔬 | **GAP** — relevant for simulator calibration |
| 13.22 | GP for contextual bandits — GP reward model with online UCB/Thompson updates | — | 🔬 | **GAP** — connects BO and online learning |
| 13.23 | Optimal experimental design — sensor placement via mutual information maximisation $I(f; y_\mathcal{S})$ | — | 🔬 | **GAP** |

### 13.E — Causal & event data

**Key equations / models:**
- Counterfactual GP: condition $f$ on hypothetical intervention $\mathrm{do}(x=x')$
- Marked TPP: intensity $\lambda(t) = \exp(f(t))$, marks $m_i \sim p(m\mid f(t_i))$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 13.13 | Causal inference / counterfactual GPs | — | 🔬 | **GAP** |
| 13.14 | Marked temporal point process + GP intensity (seismology, methane plumes) | — | 🔬 | **GAP** — dd:examples/gp/moments.md |

### 13.F — Practical

**Key equations / models:**
- Masked likelihood: $\log p(y_\mathrm{obs}\mid f_\mathrm{obs})$, automatic imputation of $f_\mathrm{miss}$ via posterior

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 13.15 | Missing data / partial observations with masked likelihood | — | 🌉 | **GAP** — dd:examples/gp/moments.md |
| 13.24 | Covariate-shift GP — importance-weighted marginal likelihood for train/test distribution mismatch | — | 🔬 | **GAP** |

## Part 14 — Metrics & Calibration

**Key equations / models:**
- NLPD: $-\tfrac{1}{N}\sum_i \log p(y_i\mid x_i)$ → decomposes into calibration + sharpness
- ECE: $\sum_b |B_b|/N \cdot |\mathrm{acc}(B_b) - \mathrm{conf}(B_b)|$
- CRPS: $\int_{-\infty}^\infty (F(z) - \mathbf{1}\{y\le z\})^2\,dz$; closed form for Gaussian $F$
- Coverage at $1-\alpha$: $\tfrac{1}{N}\sum \mathbf{1}\{y_i \in [\mu_i \pm z_{\alpha/2}\sigma_i]\}$
- Interval width (sharpness): $\mathbb{E}[2 z_{\alpha/2}\sigma_i]$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| 14.1 | NLPD decomposition: calibration + sharpness | — | 🧱 | **GAP** — dd:features/gp/metrics.md |
| 14.2 | Expected Calibration Error (ECE) & coverage diagnostics | — | 🧱 | **GAP** |
| 14.3 | Continuous Ranked Probability Score (CRPS) | — | 🧱 | **GAP** |
| 14.4 | RMSE / MAE / R² / interval width | — | 🧱 | **GAP** |

---

## Summary of dups to reconcile

| Topic | Locations | Suggestion |
|---|---|---|
| Kernel approximations / RFF / Nyström | G `kernel_approximations`, P `kernel_approximation`, P `random_fourier_features` | Keep pyrox as canonical; gaussx version → low-level mechanics |
| Sparse VGP | G `sparse_variational_gp`, G `whitened_svgp`, R `pyroxgp/01_svgp_standard` | research_notebook = applied; gaussx ones = linear-algebra view |
| Expectation Propagation | G `expectation_propagation`, P `advanced_gp_ep` | gaussx = mechanics-from-scratch; pyrox = library API |
| Schur / conditioning | G `conditional_distributions`, G `sugar_ops` | merge |
| Operator basics | G `basics`, G `operator_zoo` | merge |
| Solver strategies | G `solver_strategies`, G `solver_comparison` | merge |

## Proposed final homes (high-level)

- **gaussx/docs/notebooks/** → Parts 0, 1, 9.A (mechanics), small subset of 5 (linear-algebra view), 10.A, 14
- **pyrox/docs/notebooks/** → Parts 2, 3, 5 (mostly), 6, 7, 8, 10.B, 11, 12.1
- **research_notebook/projects/gaussian_processes/** → Part 4 (applied), 9.B–9.D, 13, plus migrated gaussx fully-fledged items
