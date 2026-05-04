# Bayesian NN, NeRF & Basis-Function Regression Tutorial Master List

A reconciled, exhaustive curriculum for the **Bayesian deep learning** half of the suite. Covers basis-function / linear-model regression as the foundation, neural fields / implicit neural representations / NeRFs, and Bayesian neural networks proper.

> Pure-GP tutorials live in [`gp_tutorial_master_list.md`](gp_tutorial_master_list.md). Cross-listed items (RFF, deep kernels, last-layer-Bayes, BLR) are flagged 🔁.

**Legend** — Source columns:
- `G` = exists in gaussx (`docs/notebooks/<name>`)
- `P` = exists in pyrox (`docs/notebooks/<name>`)
- `R` = exists in research_notebook (`projects/<area>/notebooks/<path>`)
- `—` = does not exist yet (gap)

**Scope tag**:
- 🧱 **fundamental** — small, library API demo (gaussx/pyrox docs)
- 🔬 **research** — applied / dataset-driven (research_notebook/projects/bayesian_nns)
- 🌉 **bridge** — useful in either; cross-link
- 🔁 **cross-listed** — also in GP master list

**Refs column** lists provenance: `gh#N` = open GitHub issue, `dd:path` = pyrox `design_docs/pyrox/<path>`, `mc#` = 9-model masterclass entry from `examples/nn/regression_masterclass_eqx.md`.

---

## Part A — Basis-Function Regression Foundations

### A.A — Bayesian linear regression
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| A.1 | Bayesian linear regression from scratch (mean-cov form) | — | 🧱 | **GAP** — dd:mc#1 polynomial features + Vandermonde + MCMC |
| A.2 | BLR in precision / natural form | G `numpyro_precision` | 🧱 🔁 | |
| A.3 | Sequential / online BLR updates | — | 🧱 🔁 | **GAP** — api: `blr_diag_update`, `blr_full_update` |
| A.4 | Polynomial basis regression with uncertainty | — | 🧱 | **GAP** — dd:mc#1 |

### A.B — Fixed feature maps
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| A.5 | Fixed feature maps: Fourier, polynomial, wavelet | — | 🧱 | **GAP** |
| A.6 | Spectral kernel models — visual guide | P `spectral_kernel_models` | 🧱 🔁 | |

### A.C — Random Fourier features
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| A.7 | Random Fourier Features → SSGP → VSSGP | P `random_fourier_features` | 🧱 🔁 | dd:mc#5 (Rahimi & Recht 2007) |
| A.8 | RFF as a (shallow) neural network — fixed, learned, ensemble | P `rff_as_neural_networks` | 🌉 🔁 | |
| A.9 | SSGP — Sparse Spectrum GP via RFF + BLR, O(D²N) | — | 🧱 | **GAP** — dd:examples/nn/models.md |
| A.10 | Heteroscedastic RFF — dual-head (mean + log-noise) | — | 🧱 | **GAP** — dd:features/nn/random_features.md |
| A.11 | Approximate GP via RFF + hierarchical prior on signal variance | — | 🧱 | **GAP** — dd:mc#6 |
| A.12 | Variational Fourier Features (VSSGP — learnable posterior over RFF freqs) | — | 🧱 | **GAP** — pyrox `.plans/spectral-inducing-features.md` |
| A.13 | Orthogonal Random Features (ORF) | — | 🧱 | **GAP** — pyrox `.plans/spectral-inducing-features.md` |

### A.D — Spectral basis layers
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| A.14 | HSGP — Hilbert Space GP layer, deterministic Laplacian basis + spectral-density prior | — | 🧱 | **GAP** — dd:features/nn/random_features.md |

### A.E — Bridges to GP
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| A.15 | Whitened SVGP as Bayesian linear regression | G `whitened_svgp` | 🌉 🔁 | |

### A.F — Neural network baseline
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| A.16 | Neural network MAP — single hidden MLP via SVI + AutoDelta | — | 🧱 | **GAP** — dd:mc#2 (deterministic baseline) |

### A.G — Library patterns
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| A.17 | Three-pattern regression masterclass: tree_at / pyrox_sample / Parameterized | P `regression_masterclass_treeat`, `_pyrox_sample`, `_parameterized` | 🧱 🔁 | |
| A.18 | sklearn-style `EstimatorBase` facade | — | 🧱 | **GAP** — gh:pyrox#71 |

## Part B — Implicit Neural Representations & Neural Fields

### B.A — Foundations
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| B.1 | MLP as a continuous function approximator (1D / 2D) | — | 🧱 | **GAP** — pedagogical entry |
| B.2 | Positional encoding (Tancik 2020 / NeRF Fourier features) | — | 🧱 | **GAP** |
| B.3 | Gaussian-feature INRs (RFF as positional encoding) | — | 🧱 | **GAP** — connects A.7 ↔ B.4 |

### B.B — SIREN family
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| B.4 | SIREN — sinusoidal implicit neural representations | P `siren_inr` | 🌉 | move research-scale → projects/bayesian_nns |
| B.5 | MultiScaleSIREN — frequency-banded sinusoidal nets | — | 🌉 | **GAP** — gh:pyrox#91 |

### B.C — Non-MLP architectures
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| B.6 | Multiplicative Filter Networks (FourierNet, GaborNet) | — | 🌉 | **GAP** — gh:pyrox#87 |

### B.D — Conditioning
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| B.7 | Conditional neural fields: FiLM, Hyper-RFF | P `conditioning` | 🧱 | |
| B.8 | Hypernetworks for INRs | — | 🔬 | **GAP** — research_notebook only |

### B.E — Spherical / localized encodings
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| B.9 | Slepian Positional Encodings (spherical, localized) | — | 🌉 🔁 | **GAP** — gh:pyrox#125 |

### B.F — Multi-resolution encoding
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| B.10 | Hashgrid / multi-resolution encoding (Instant NGP) | — | 🔬 | **GAP** — research_notebook only |

### B.G — Volumetric / 3D
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| B.11 | NeRF — vanilla volumetric rendering | — | 🔬 | **GAP** — research_notebook only |
| B.12 | NeRF variants: mip-NeRF, Plenoxels, Gaussian Splatting | — | 🔬 | **GAP** — research_notebook only |

### B.H — Continuous-depth & Bayesian
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| B.13 | Neural ODEs / continuous-depth models | — | 🔬 | **GAP** — research_notebook only |
| B.14 | Bayesian INRs / Bayesian NeRF (uncertainty in scenes) | — | 🔬 | **GAP** — research_notebook only |

## Part C — Bridges: NN ↔ GP Correspondences

### C.A — Theory
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| C.1 | Infinite-width NN as a GP (NNGP) | — | 🧱 | **GAP** — Lee et al. 2018 |
| C.2 | Neural Tangent Kernel intro | — | 🧱 | **GAP** |
| C.3 | ArcCosine kernel — NN-correspondence via infinite-width limits | — | 🧱 🔁 | **GAP** — dd:features/gp/gpflow.md |

### C.B — Deep kernels
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| C.4 | Deep kernels: NN-warped GP inputs | R `pyroxgp/04_svgp_rff_nn` | 🌉 🔁 | |
| C.5 | Deep RFF / stacked spectral GPs (Cutajar 2017) | P `deep_random_fourier_features` | 🔬 🔁 | dd:mc#7 |

### C.C — Functional priors
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| C.6 | Functional priors: BNNs that match GP priors | — | 🔬 | **GAP** |

### C.D — Sampling
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| C.7 | Pathwise sampling for BNNs (analogue of Wilson 2020) | — | 🧱 | **GAP** — needs gh:gaussx#77, #78 |

### C.E — Shared infrastructure
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| C.8 | Shared `pyrox._basis` — VFF (gp) + HSGP (nn) sharing Laplacian eigenfunctions | — | 🧱 | **GAP** — pyrox `.plans/spectral-inducing-features.md` |

## Part D — Bayesian Inference for Neural Networks

### D.A — Last-layer Bayesian NNs
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| D.1 | Last-layer Bayes via Laplace | — | 🔬 | **GAP** — api: `gauss_newton_precision`, `ggn_diagonal` |
| D.2 | Last-layer Bayes via RFF (BLR on penultimate features) | — | 🔬 | **GAP** |
| D.3 | RandomFeatureGaussianProcess + LaplaceRandomFeatureCovariance — SNGP output layer | — | 🧱 | **GAP** — dd:features/nn/edward2_layers.md |

### D.B — Variational families
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| D.4 | Mean-field VI for BNNs (MFVI) | — | 🔬 | **GAP** — needs gh:gaussx#39 logdet |
| D.5 | Full-rank / structured VI for BNNs | — | 🔬 | **GAP** |
| D.6 | DenseVariationalDropout — learned per-weight dropout rates | — | 🧱 | **GAP** — dd:features/nn/edward2_layers.md |
| D.7 | DenseDVI — analytic Gaussian moment propagation | — | 🧱 | **GAP** — dd:features/nn/edward2_layers.md |

### D.C — Stochastic approximation
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| D.8 | MC-Dropout as approximate Bayes | — | 🔬 | **GAP** — dd:mc#3 |

### D.D — Weight-space Gaussian approximations
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| D.9 | SWAG — stochastic weight averaging Gaussian | — | 🔬 | **GAP** |
| D.10 | Subspace inference (PCA of SGD trajectory) | — | 🔬 | **GAP** |

### D.E — Hessian-based posteriors
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| D.11 | KFAC / GGN posteriors (Laplace approximation full network) | — | 🔬 | **GAP** |
| D.12 | Hutchinson Hessian diagonal for BNN Laplace | — | 🧱 🔁 | **GAP** — api: `hutchinson_hessian_diag` |

### D.F — Sampling-based inference
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| D.13 | HMC / NUTS for small BNNs | — | 🔬 | **GAP** — dd:mc#4 |
| D.14 | Stein Variational Gradient Descent (SVGD) | — | 🔬 | **GAP** |
| D.15 | Cold posteriors & temperature scaling | — | 🔬 | **GAP** |

### D.G — Distance-aware uncertainty
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| D.16 | SNGP — Spectral Normalized GP | — | 🔬 | **GAP** — gh:pyrox#42, dd:features/nn/spectral_norm.md |
| D.17 | DUE — Deterministic Uncertainty Estimation (spectral norm + inducing-point GP head) | — | 🔬 | **GAP** — dd:features/nn/spectral_norm.md |

### D.H — Bayesian Neural Fields
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| D.18 | BNF layer family + `BNFEstimator`/`MLE`/`VI` | — | 🔬 | **GAP** — gh:pyrox#72 |

### D.I — Edward2-style layers
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| D.19 | DenseRank1 — BatchEnsemble (shared W + per-member rank-1 perturbations) | — | 🧱 | **GAP** — dd:features/nn/edward2_layers.md |
| D.20 | MCSoftmaxDenseFA / MCSigmoidDenseFA — heteroscedastic output (low-rank + diagonal) | — | 🧱 | **GAP** — dd:features/nn/edward2_layers.md |
| D.21 | DenseHierarchical — horseshoe prior (local + global shrinkage, ARD) | — | 🧱 | **GAP** — dd:features/nn/edward2_layers.md |
| D.22 | NCPNormalOutput — output-side noise contrastive prior | — | 🧱 | **GAP** — dd:features/nn/edward2_layers.md |
| D.23 | MultiHeadAttentionBE — BatchEnsemble multi-head attention | — | 🧱 | **GAP** — dd:features/nn/edward2_layers.md |

### D.J — Conv / RNN / attention
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| D.24 | Conv2DReparameterization — Bayesian 2D conv | — | 🧱 | **GAP** — dd:features/nn/layers_conv_rnn.md |
| D.25 | Conv2DFlipout — lower-variance Bayesian conv | — | 🧱 | **GAP** — dd:features/nn/layers_conv_rnn.md |
| D.26 | LSTMCellVariational — Bayesian LSTM | — | 🧱 | **GAP** — dd:features/nn/layers_conv_rnn.md |
| D.27 | GRUCellVariational — Bayesian GRU (scan-compatible) | — | 🧱 | **GAP** — dd:features/nn/layers_conv_rnn.md |
| D.28 | MultiHeadAttentionVariational — Bayesian Q/K/V projections | — | 🧱 | **GAP** — dd:features/nn/layers_conv_rnn.md |

## Part E — Ensembles for Uncertainty

### E.A — Vanilla ensembles
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| E.1 | Deep ensembles — vanilla | — | 🔬 | **GAP** |
| E.2 | Ensemble primitives — three ways | P `ensemble_primitives_tutorial` | 🧱 | |
| E.3 | EnsembleMAP & EnsembleVI runners | P `ensemble_runner_tutorial` | 🧱 | |
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

## Part F — Calibration, OOD, and Diagnostics

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| F.1 | Predictive calibration: ECE & reliability diagrams | — | 🔬 | **GAP** |
| F.2 | Temperature scaling & post-hoc calibration | — | 🔬 | **GAP** |
| F.3 | Out-of-distribution detection with BNNs | — | 🔬 | **GAP** |
| F.4 | Active learning / Bayesian acquisition functions | — | 🔬 | **GAP** |

## Part G — Applied Case Studies *(research_notebook/projects/bayesian_nns)*

### G.A — INRs / NeRF
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| G.1 | SIREN on real images / signed distance fields | P `siren_inr` (port + extend) | 🔬 | |
| G.2 | NeRF on a small synthetic scene | — | 🔬 | **GAP** |
| G.3 | Bayesian INR / probabilistic SIREN | — | 🔬 | **GAP** |

### G.B — Spectral GPs / climate
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| G.4 | Deep RFF on geophysical / climate data | P `deep_random_fourier_features` (port + extend) | 🔬 | |
| G.5 | scalable_gp_spectral demo — 5k 1D regression, dense GP vs VFF, ≥10× speedup | — | 🔬 | **GAP** — pyrox `.plans/spectral-inducing-features.md` |

### G.C — Bayesian benchmarks
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| G.6 | Last-layer Bayesian NN on UCI regression suite | — | 🔬 | **GAP** |
| G.7 | Bayesian Neural Fields flagship demo (`bayesian_neural_fields.ipynb`) | — | 🔬 | **GAP** — gh:pyrox#73 |
| G.8 | 9-model regression masterclass — full progression on one dataset | — | 🔬 | **GAP** — dd:examples/nn/regression_masterclass_eqx.md (~927 lines) |

### G.D — Emulators & PDEs
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| G.9 | BNN emulator for a numerical simulator | — | 🔬 | **GAP** |
| G.10 | Bayesian PINN (physics-informed NN) | — | 🔬 | **GAP** |

### G.E — Image regression
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| G.11 | BNN for image regression / denoising | — | 🔬 | **GAP** |
| G.12 | Probabilistic super-resolution via RFF / INR | — | 🔬 | **GAP** |

---

## Cross-list summary (items shared with GP list)

| Item | GP ID | BNN ID | Suggested home |
|---|---|---|---|
| Spectral kernel models | GP 2.5 | A.6 | pyrox |
| Random Fourier Features intro | GP 5.1 | A.7 | pyrox (canonical), link both |
| RFF as neural networks | — | A.8 | pyrox |
| Whitened SVGP / BLR view | GP 5.7 | A.15 | gaussx (mechanics) |
| BLR updates (`blr_*_update`) | GP 6.19 | A.3 | gaussx primitive demo |
| BLR in precision form | GP 10.2 | A.2 | gaussx |
| Three-pattern masterclass | GP 10.3 | A.17 | pyrox |
| Deep kernels | GP 2.6 | C.4 | research_notebook |
| Deep RFF (Cutajar) | — | C.5 / G.4 | research_notebook (BNN) |
| ArcCosine kernel | GP 2.7 | C.3 | pyrox |
| Slepian positional encodings | GP 2.10 | B.9 | pyrox |
| Hutchinson Hessian diag | GP 6.13 | D.12 | gaussx primitive + BNN application |

## Proposed final homes

- **gaussx/docs/notebooks/** → A.A–A.B, A.E, primitive demo for D.12
- **pyrox/docs/notebooks/** → A.C, A.D, A.F, A.G, B.D (B.7), B.E, all of D.I–D.J, E.A (pyrox-side), C.E
- **research_notebook/projects/bayesian_nns/** → all of B.A–B.C, B.F–B.H, C.A–C.D, D.A–D.H, E.B–E.C, all of F, all of G

## In-scope vs aspirational

- **In scope today** (have library support in pyrox/gaussx): A.2, A.3, A.6, A.7, A.8, A.15, A.17, B.4, B.7, C.4, C.5, D.12, E.2, E.3
- **In scope with planned features** (open issues / `.plans/`): A.1, A.4, A.9–A.14, A.16, A.18, B.5, B.6, B.9, C.3, C.8, D.3, D.6, D.7, D.16–D.28, E.4, G.5, G.7, G.8
- **Aspirational** (need new infra or genuine research work): B.1–B.3, B.8, B.10–B.14, C.1, C.2, C.6, C.7, D.1, D.2, D.4, D.5, D.8–D.11, D.13–D.15, D.17, D.18, E.1, E.5–E.7, F.*, G.1–G.4, G.6, G.9–G.12
