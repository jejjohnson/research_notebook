---
title: Gaussianization flows — pure-Keras bijectors and density estimation
---

# Gaussianization flows

A self-contained sub-project for **Gaussianization flows** — normalising flows built from two kinds of bijectors stacked in blocks:

1. **Rotation** — a trainable Householder product `Q = H_0 … H_{K-1}` or a fixed orthogonal matrix (optionally initialised from the data PCA).
2. **Marginal transform** — a per-dim mixture-CDF Gaussianization `u_i = F_i(x_i)`, `z_i = Φ⁻¹(u_i)`.

Both operators are **pure Keras 3** layers that work across TensorFlow, JAX, and PyTorch back-ends via `keras.ops`. The library also ships a conditional **coupling** variant driven by an MLP conditioner, and an **iterative-Gaussianization** warm-start (`initialize_flow_from_ig`) that greedy-fits each block to the data before gradient training.

## Sub-package

- [`gauss_keras`](src/gaussianization/gauss_keras/) — the full library:
  - `bijectors.rotation` — `Householder`, `FixedOrtho` (+ `from_pca` factory).
  - `bijectors.marginal` — `MixtureCDFGaussianization` (mixture-CDF forward + bisection inverse).
  - `bijectors.coupling` — `MixtureCDFCoupling` with user-supplied conditioner, plus `make_mlp_conditioner` / `make_shared_mlp_conditioner` builders.
  - `bijectors.flow` — `GaussianizationFlow` model (`log_prob`, `sample`, `invert`, `forward_with_intermediates`).
  - `ig_init` — `initialize_flow_from_ig(flow, X)`, the iterative-Gaussianization warm-start using sklearn PCA + GMM.
  - `training` — `base_nll_loss` + `make_gaussianization_flow` / `make_coupling_flow` factories.

## Notebooks

- [01_gaussianization_2d](notebooks/01_gaussianization_2d.ipynb) — canonical Gaussianization flow on the two-moons distribution; compares iterative and parametric formulations.
- [02_coupling_flow_2d](notebooks/02_coupling_flow_2d.ipynb) — coupling flow with a mixture-CDF bijector driven by an MLP conditioner; dissects residual / bijector layers and shapes.
- [03_iterative_gaussianization_init](notebooks/03_iterative_gaussianization_init.ipynb) — warm-start initialisation with `initialize_flow_from_ig`, compared head-to-head against random init on both diagonal and coupling flows.
- [04_coupling_equivalence](notebooks/04_coupling_equivalence.ipynb) — proof (and empirical check) that a zero-kernel-initialised coupling flow is numerically equivalent to its diagonal-marginal counterpart; training then breaks that equivalence.

## Layout

```
projects/gaussianization/
├── pyproject.toml                        # standalone package "gaussianization"
├── src/gaussianization/gauss_keras/
│   ├── _math.py                          # standard-normal CDF / quantile / log pdf
│   ├── mixtures/gaussian.py              # MixtureOfGaussians (cdf, pdf, log_pdf)
│   ├── bijectors/
│   │   ├── base.py                       # Bijector ABC + hybrid call
│   │   ├── rotation.py                   # Householder, FixedOrtho
│   │   ├── marginal.py                   # MixtureCDFGaussianization
│   │   ├── coupling.py                   # MixtureCDFCoupling + conditioner builders
│   │   ├── flow.py                       # GaussianizationFlow model
│   │   └── _householder_decomp.py        # QR decomposition helper
│   ├── ig_init.py                        # iterative-Gaussianization warm-start
│   └── training.py                       # loss + flow factories
├── notebooks/
└── tests/
```

## Running

The package is a standalone Python project under `projects/gaussianization/`. Tests default to the TensorFlow backend (set via `KERAS_BACKEND`); JAX and PyTorch back-ends also work, they just aren't exercised in CI.

```bash
cd projects/gaussianization
uv sync --all-extras          # or: pip install -e ".[dev,notebooks]"
KERAS_BACKEND=tensorflow uv run pytest tests -v
```

## Public API

```python
from gaussianization.gauss_keras import (
    # core bijectors
    Bijector,
    Householder, FixedOrtho,
    MixtureCDFGaussianization,
    MixtureCDFCoupling,
    # flow + training helpers
    GaussianizationFlow,
    base_nll_loss,
    make_gaussianization_flow, make_coupling_flow,
    # coupling conditioner builders
    default_half_mask,
    make_mlp_conditioner, make_shared_mlp_conditioner,
    tanh_log_scale_clamp, sigmoid_log_scale_clamp,
    # iterative-Gaussianization warm-start
    initialize_flow_from_ig,
    # mixture primitive
    MixtureOfGaussians,
)
```
