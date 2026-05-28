# Neural Fields / Implicit Neural Representations

Tutorials and applied case studies for **continuous-coordinate function approximators** — MLPs with positional encodings, SIREN, multiplicative filter networks, NeRF and successors, conditioning, multi-resolution encodings, and continuous-depth models.

The full curriculum lives in [`TUTORIAL_MASTER_LIST.md`](TUTORIAL_MASTER_LIST.md). Bayesian extensions (probabilistic SIREN, Bayesian NeRF, Bayesian Neural Fields) live in [`../bayesian_nns/`](../bayesian_nns/) **Part G**.

## Layout

- `notebooks/A_foundations/` — MLP function approximation, spectral bias, positional / Fourier encodings, init & training dynamics
- `notebooks/B_architectures/` — SIREN family, multiplicative filter networks, periodic / wavelet / Gabor INRs
- `notebooks/C_volumetric_nerf/` — vanilla NeRF, mip-NeRF, Plenoxels, Gaussian Splatting, SDF / occupancy
- `notebooks/D_conditioning/` — FiLM, Hyper-RFF, hypernetworks, meta-learning, latent-modulated INRs
- `notebooks/E_spatial_encoding/` — Slepian, spherical harmonics, hashgrid (Instant NGP), tri-plane
- `notebooks/F_continuous_depth/` — Neural ODE / CDE / SDE
- `notebooks/G_loss_constraints/` — PINN, equivariance / conservation, sparsity / TV / boundary penalties
- `notebooks/H_applied/` — signals & images, 3D scenes, scientific / geospatial fields

## Companion lists

- Bayesian variants → [`../bayesian_nns/TUTORIAL_MASTER_LIST.md`](../bayesian_nns/TUTORIAL_MASTER_LIST.md)
- Pure GPs (RFF, spectral kernels, harmonic features) → [`../gaussian_processes/TUTORIAL_MASTER_LIST.md`](../gaussian_processes/TUTORIAL_MASTER_LIST.md)
- Normalizing flows (continuous-depth + invertible) → [`../gaussianization/TUTORIAL_MASTER_LIST.md`](../gaussianization/TUTORIAL_MASTER_LIST.md)
