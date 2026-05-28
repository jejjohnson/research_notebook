# Neural Fields / Implicit Neural Representations Tutorial Master List

A reconciled curriculum for **neural fields** — continuous-coordinate function approximators (1D signals, 2D images, 3D scenes, time-varying fields) parameterised by neural networks. Covers MLP fundamentals, positional / Fourier encodings, SIREN-family architectures, multiplicative filter networks, conditioning, multi-resolution / hashgrid encodings, NeRF and successors, continuous-depth models, and applied case studies.

Companion lists:
- Bayesian extensions (probabilistic SIREN, Bayesian NeRF, BNF) → [`../bayesian_nns/TUTORIAL_MASTER_LIST.md`](../bayesian_nns/TUTORIAL_MASTER_LIST.md) **Part G**
- Pure-GP machinery (RFF, spectral kernels, harmonic features) → [`../gaussian_processes/TUTORIAL_MASTER_LIST.md`](../gaussian_processes/TUTORIAL_MASTER_LIST.md)
- Normalizing flows (continuous-depth & invertible) → [`../gaussianization/TUTORIAL_MASTER_LIST.md`](../gaussianization/TUTORIAL_MASTER_LIST.md)

Cross-listed items (RFF-as-PE, Slepian, deep RFF, continuous-depth flows) are flagged 🔁.

**Legend** — Source columns:
- `G` = exists in gaussx (`docs/notebooks/<name>`)
- `P` = exists in pyrox (`docs/notebooks/<name>`)
- `R` = exists in research_notebook (`projects/neural_fields/notebooks/<path>`)
- `—` = does not exist yet (gap)

**Scope tag**: 🧱 fundamental · 🔬 research · 🌉 bridge · 🔁 cross-listed

**Refs**: `gh:<repo>#N` = open GitHub issue (e.g., `gh:pyrox#91`) · `dd:path` = pyrox `design_docs/pyrox/<path>` · `xref:BNN#X.Y` / `xref:GP#X.Y` = pointer into companion list.

---

## Curriculum at a glance

- **Part A — Foundations**
  - A.A — MLPs as continuous function approximators
  - A.B — Positional / Fourier encodings
  - A.C — Initialization & training dynamics
- **Part B — Architectures**
  - B.A — SIREN family
  - B.B — Multiplicative Filter Networks
  - B.C — Periodic / wavelet / Gabor INRs
- **Part C — Volumetric & 3D Scenes**
  - C.A — Vanilla NeRF
  - C.B — NeRF variants (mip-NeRF, Plenoxels, Gaussian Splatting)
  - C.C — Scene representations beyond MLP
- **Part D — Conditioning & Generalisation**
  - D.A — FiLM / Hyper-RFF
  - D.B — Hypernetworks
  - D.C — Meta-learning INRs (MAML, Reptile)
  - D.D — Latent-modulated INRs
- **Part E — Spatial Encoding Variants**
  - E.A — Spherical / Slepian
  - E.B — Multi-resolution hashgrid (Instant NGP)
  - E.C — Tri-plane & factorised grids
- **Part F — Continuous-Depth Models**
  - F.A — Neural ODEs
  - F.B — Neural CDEs / SDEs
- **Part G — Loss Constraints (deterministic)**
  - G.A — PDE residuals (PINN)
  - G.B — Symmetry / equivariance / conservation
  - G.C — Sparsity, smoothness, TV priors
- **Part H — Applied Case Studies**

---

## Part A — Foundations

### A.A — MLPs as continuous function approximators

**Key equations / models:**
- $f_\theta(x) = W_L\sigma(W_{L-1}\sigma(\cdots\sigma(W_1 x + b_1)\cdots) + b_{L-1}) + b_L$
- Spectral bias (Rahaman 2019): standard ReLU MLPs preferentially fit low-frequency components
- Universal approximation vs *learnability* — high-frequency targets need encoding tricks

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| A.1 | MLP as a continuous function approximator (1D / 2D) | — | 🧱 | **GAP** — pedagogical entry |
| A.2 | Spectral bias of ReLU MLPs — fitting high-frequency targets fails | — | 🧱 | **GAP** — Rahaman et al. 2019 |

### A.B — Positional / Fourier encodings

**Key equations / models:**
- NeRF positional encoding (Mildenhall 2020): $\gamma(x) = [\sin(2^k\pi x), \cos(2^k\pi x)]_{k=0}^{L-1}$
- Tancik (2020) Fourier feature mapping: $\gamma(x) = [\cos(2\pi Bx), \sin(2\pi Bx)]$, $B\sim\mathcal{N}(0, \sigma^2 I)$
- Equivalence: Fourier features $\leftrightarrow$ random features for a stationary kernel (Rahimi–Recht)

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| A.3 | Positional encoding (Tancik 2020 / NeRF Fourier features) | — | 🧱 | **GAP** |
| A.4 | Gaussian-feature INRs (RFF as positional encoding) | — | 🧱 🔁 | **GAP** — bridges `xref:BNN#A.8` (RFF) ↔ this list's B.1 (SIREN) |
| A.5 | Frequency bandwidth & lengthscale tuning for PE | — | 🧱 | **GAP** |

### A.C — Initialization & training dynamics

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| A.6 | SIREN-style $\sqrt{6/\mathrm{fan\_in}}$ init — preserve activation distribution | — | 🧱 | **GAP** |
| A.7 | Lazy training & feature learning in INRs | — | 🌉 | **GAP** |

## Part B — Architectures

### B.A — SIREN family

**Key equations / models:**
- SIREN (Sitzmann 2020): $f(x) = W_L\sin(\omega_0(W_{L-1}\sin(\cdots\omega_0(W_1 x + b_1)\cdots) + b_{L-1})) + b_L$
- $\omega_0 \approx 30$ controls frequency content
- Derivatives are themselves SIRENs → natural for PDE / SDF supervision

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| B.1 | SIREN — sinusoidal implicit neural representations | P `siren_inr` | 🌉 | move research-scale → projects/neural_fields |
| B.2 | MultiScaleSIREN — frequency-banded sinusoidal nets | — | 🌉 | **GAP** — gh:pyrox#91 |
| B.3 | SIREN derivative supervision — fit $f$ via $\nabla f$ / $\Delta f$ | — | 🧱 | **GAP** |

### B.B — Multiplicative Filter Networks

**Key equations / models:**
- MFN (Fathony 2021): $z_l = (W_l z_{l-1} + b_l)\odot g_l(x)$ with $g_l$ a Fourier/Gabor filter on $x$
- FourierNet: $g_l(x) = \sin(\omega_l x + \phi_l)$ · GaborNet: $g_l(x) = e^{-\gamma\|x-\mu_l\|^2}\sin(\omega_l(x-\mu_l))$

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| B.4 | Multiplicative Filter Networks (FourierNet, GaborNet) | — | 🌉 | **GAP** — gh:pyrox#87 |

### B.C — Periodic / wavelet / Gabor INRs

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| B.5 | WIRE — Gabor wavelet INR | — | 🌉 | **GAP** — Saragadam et al. 2023 |
| B.6 | Periodic INRs on the torus / sphere | — | 🧱 | **GAP** |

## Part C — Volumetric & 3D Scenes

### C.A — Vanilla NeRF

**Key equations / models:**
- Volume rendering: $C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\sigma(\mathbf{r}(t))c(\mathbf{r}(t), \mathbf{d})\,dt$, $T(t) = \exp(-\int_{t_n}^t \sigma(\mathbf{r}(s))\,ds)$
- Hierarchical coarse-to-fine sampling along rays

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| C.1 | NeRF — vanilla volumetric rendering | — | 🔬 | **GAP** |
| C.2 | Camera-pose / ray-marching mechanics | — | 🔬 | **GAP** |

### C.B — NeRF variants

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| C.3 | mip-NeRF — integrated positional encoding for anti-aliasing | — | 🔬 | **GAP** |
| C.4 | Plenoxels — explicit voxel-grid radiance fields | — | 🔬 | **GAP** |
| C.5 | Gaussian Splatting — anisotropic Gaussians instead of MLPs | — | 🔬 | **GAP** |
| C.6 | TensoRF / Tri-plane — factorised volumetric grids | — | 🔬 | **GAP** |

### C.C — Beyond MLPs

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| C.7 | SDF parameterisations — signed-distance fields with INRs | — | 🔬 | **GAP** |
| C.8 | Occupancy networks | — | 🔬 | **GAP** |

## Part D — Conditioning & Generalisation

### D.A — FiLM / Hyper-RFF

**Key equations / models:**
- FiLM (Perez 2018): $y = \gamma(c)\odot h + \beta(c)$ injects condition $c$ via per-channel affine
- Hyper-RFF: $\omega = h(c)$ — condition predicts the RFF frequencies

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| D.1 | Conditional neural fields — FiLM, Hyper-RFF | P `conditioning` | 🧱 | |

### D.B — Hypernetworks

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| D.2 | Hypernetworks for INRs | — | 🔬 | **GAP** — research_notebook only |
| D.3 | Set / function-space hypernetworks (DeepSets, Set Transformer) | — | 🔬 | **GAP** |

### D.C — Meta-learning

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| D.4 | MAML / Reptile for fast INR adaptation | — | 🔬 | **GAP** |
| D.5 | Learned initialisations for INRs (Sitzmann 2020 §5) | — | 🔬 | **GAP** |

### D.D — Latent-modulated INRs

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| D.6 | Latent-modulated INRs — $f_\theta(x; z)$ with $z$ shared per signal | — | 🔬 | **GAP** |
| D.7 | Auto-decoder training (DeepSDF-style) | — | 🔬 | **GAP** |

## Part E — Spatial Encoding Variants

### E.A — Spherical / Slepian

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| E.1 | Slepian positional encodings — spherical, localized | — | 🌉 🔁 | **GAP** — gh:pyrox#125; xref:GP#7.19 |
| E.2 | Spherical-harmonic INRs on $S^2$ | — | 🔬 🔁 | **GAP** — xref:GP#7.21 |

### E.B — Multi-resolution hashgrid

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| E.3 | Hashgrid / multi-resolution encoding (Instant NGP) | — | 🔬 | **GAP** — Müller et al. 2022 |
| E.4 | Hashgrid collisions & spatial-frequency trade-offs | — | 🔬 | **GAP** |

### E.C — Tri-plane & factorised grids

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| E.5 | Tri-plane encoding (EG3D) | — | 🔬 | **GAP** |
| E.6 | Tensorial decompositions for spatial encodings | — | 🔬 | **GAP** |

## Part F — Continuous-Depth Models

### F.A — Neural ODEs

**Key equations / models:**
- Neural ODE (Chen 2018): $\tfrac{dh}{dt} = f_\theta(h, t)$, adjoint method for $O(1)$-memory backprop
- ANODE — augmented Neural ODEs for non-homeomorphic transformations

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| F.1 | Neural ODE — basic mechanics + adjoint | — | 🔬 | **GAP** |
| F.2 | ANODE / augmented Neural ODE | — | 🔬 | **GAP** |

### F.B — Neural CDEs / SDEs

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| F.3 | Neural CDE for irregular time series | — | 🔬 | **GAP** |
| F.4 | Neural SDE — stochastic continuous-depth | — | 🔬 🔁 | **GAP** — bridge to gaussianization list |

## Part G — Loss Constraints (deterministic)

Bayesian / probabilistic versions of these losses live in BNN list **Part A.G**. This section covers the *deterministic* PINN / equivariance / regularisation lineage.

### G.A — PDE residuals

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| G.1 | PINN — Burgers / heat / shallow-water | — | 🔬 | **GAP** |
| G.2 | XPINN / domain-decomposed PINNs | — | 🔬 | **GAP** |
| G.3 | Adaptive loss weighting (Wang 2022 NTK-based) | — | 🔬 | **GAP** |

### G.B — Symmetry / equivariance / conservation

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| G.4 | Equivariant INRs (rotation, translation) | — | 🔬 | **GAP** |
| G.5 | Divergence-free / curl-free vector-field INRs | — | 🔬 | **GAP** |
| G.6 | Conservation laws as soft penalties (mass, momentum, energy) | — | 🔬 | **GAP** |

### G.C — Sparsity, smoothness, TV priors

| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| G.7 | Total-variation / smoothness penalties | — | 🌉 | **GAP** |
| G.8 | Sparsity-promoting regularisation (L1, group lasso) | — | 🌉 | **GAP** |
| G.9 | Boundary-condition / initial-condition penalties | — | 🔬 | **GAP** |

## Part H — Applied Case Studies *(research_notebook/projects/neural_fields)*

### H.A — Signals & images
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| H.1 | SIREN on real images / signed distance fields | P `siren_inr` (port + extend) | 🔬 | |
| H.2 | Image fitting with NeRF-PE vs SIREN vs MFN — comparison | — | 🔬 | **GAP** |
| H.3 | Audio fitting with INRs | — | 🔬 | **GAP** |

### H.B — 3D scenes
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| H.4 | NeRF on a small synthetic scene | — | 🔬 | **GAP** |
| H.5 | Gaussian Splatting on real-world capture | — | 🔬 | **GAP** |

### H.C — Scientific / geospatial
| # | Tutorial | Source | Scope | Refs / Notes |
|---|----------|--------|-------|--------------|
| H.6 | Climate / SST field reconstruction with INRs | — | 🔬 | **GAP** |
| H.7 | Spherical INRs for global atmospheric variables | — | 🔬 | **GAP** |
| H.8 | Spatiotemporal INRs — coordinates $(x, y, t)$ | — | 🔬 | **GAP** |

---

## Cross-list summary

| Item | NF ID | Other list | Suggested home |
|---|---|---|---|
| RFF / Gaussian features as PE | A.4 | BNN A.8 | pyrox (BNN canonical) |
| Slepian PE | E.1 | GP 7.19, BNN B.9 | pyrox |
| Spherical-harmonic INRs | E.2 | GP 7.18 | research_notebook |
| Neural SDE | F.4 | gaussianization | research_notebook |
| Bayesian INR / NeRF | — | BNN G.1–G.3 | research_notebook |

## Proposed final homes

- **pyrox/docs/notebooks/** → A.B (PE primitives), B.A (SIREN), B.B (MFN), D.A (conditioning), E.A (Slepian)
- **research_notebook/projects/neural_fields/** → all of C (NeRF & variants), D.B–D.D, E.B–E.C, F, G, H

## In-scope vs aspirational

- **In scope today**: B.1 (SIREN exists in pyrox), D.1 (conditioning exists in pyrox)
- **In scope with planned features**: A.3, A.4, A.6, B.2, B.3, B.4, D.2, E.1, G.1
- **Aspirational**: everything else — needs new infra / research / dataset work
