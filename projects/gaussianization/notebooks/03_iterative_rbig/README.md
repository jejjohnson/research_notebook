---
title: "Part 3 — Iterative Gaussianization (RBIG)"
short_title: "Part 3 · Iterative (RBIG)"
subject: gaussianization tutorial
---

# Part 3 — Iterative Gaussianization (RBIG)

The classical, non-parametric Gaussianization algorithm. Part 1 built the marginal
transforms and Part 2 the rotations; **Rotation-Based Iterative Gaussianization**
{cite}`laparra2011rbig` simply alternates them — `marginal → rotate → marginal →
rotate …` — until any distribution flows to $\mathcal{N}(0,I)$. Each block is
fit once, greedily, and never revisited (the *parametric*, end-to-end-trained
version is Part 4). This part builds the loop, proves it converges, shows the
rotation sets the speed, and adds the numerical care that makes it robust —
grounded in [`rbig`](https://github.com/jejjohnson/rbig) (the iterative algorithm
and its information measures) and [`gauss_flows`](https://github.com/jejjohnson/gauss_flows)
(the smooth, exact-log-det version used for density and sampling).

## Notebooks

| # | notebook | master list | what you take away |
|---|----------|:-----------:|--------------------|
| 00 | [The canonical RBIG loop](00_rbig_loop.ipynb) | 3.1–3.2 | marginal→rotate iteration; two-moons morph; forward density / inverse generation; smooth-marginal sampling |
| 01 | [Convergence & stopping](01_convergence_stopping.ipynb) | 3.3–3.4 | TC validated on Gaussian; `tc_per_layer_` → 0; `zero_tolerance` early-stop vs fixed depth |
| 02 | [Rotation-choice studies](02_rotation_choices.ipynb) | 3.5–3.6 | PCA/ICA/Picard converge in 1 layer, random in ~14; Picard as fast scalable ICA |
| 03 | [Boundary issues & support extension](03_boundary_support.ipynb) | 3.9–3.10 | empirical-CDF tails → ±∞; `bound_correct`, `pdf_extension`, KDE tails; dequantisation |

## The two tools, and when to use each

Part 3 deliberately uses both packages for what each does best:

- **`rbig`** (NumPy, sklearn-style) — the canonical *iterative algorithm*
  (`AnnealedRBIG`, `MarginalGaussianize`, the rotation zoo) and the
  *information-theoretic measures* (`total_correlation`, `tc_per_layer_`,
  `negentropy`, `score`, `entropy`) that drive convergence and stopping.
- **`gauss_flows`** (JAX) — the *smooth, differentiable* RBIG (`fit_rbig`) with
  mixture-CDF marginals and an exact autodiff log-det, used wherever **sample or
  density quality** matters (notebook 00 shows histogram marginals generate diffuse
  samples; the smooth version generates crisp ones).

## Threads

- The **change-of-variables** machinery (Part 0) makes the forward/inverse and
  log-det of the whole stack concrete; the **marginal** (Part 1) and **rotation**
  (Part 2) halves are the per-layer pieces.
- **Why rotation matters** (Part 2 [00](../02_rotations/00_rotation_zoo.ipynb)) —
  marginal-only stalls — is the reason RBIG interleaves a rotation; notebook 02 here
  quantifies how the rotation *choice* sets convergence speed.
- **Warm-starting** a trainable flow from a greedy RBIG fit (master-list 3.7–3.8) is
  deferred to **Part 4**, where it belongs: it only matters once there is a
  parametric flow to initialise.

## Running

Same `uv` environment as the earlier parts (`rbig` + `gauss_flows` + a Jupyter
stack):

```bash
cd projects/gaussianization
.venv-tutorials/bin/jupyter nbconvert --to notebook --execute --inplace \
  notebooks/03_iterative_rbig/0*.ipynb --ExecutePreprocessor.timeout=600
```

Notebooks are paired (`jupytext`, `py:percent`) and kept light enough to execute in
seconds (small samples, low layer counts).
