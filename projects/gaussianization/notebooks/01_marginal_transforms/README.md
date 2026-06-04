---
title: "Part 1 — 1D Marginal Transforms"
short_title: "Part 1 · Marginal transforms"
subject: gaussianization tutorial
---

# Part 1 — 1D Marginal Transforms

The atomic operation of Gaussianization: turn one coordinate's distribution into
a standard normal via $z = \Phi^{-1}(F(x))$ for some monotone CDF estimator $F$.
Every method in the curriculum stacks these 1D maps between rotations, so Part 1
builds them every way that matters — estimating $F$, computing the **Jacobian**
the flow needs, inverting it, and training it — grounded in
[`rbig`](https://github.com/jejjohnson/rbig) and
[`gauss_flows`](https://github.com/jejjohnson/gauss_flows).

Each notebook keeps the Part 0 pattern: derive the idea, then confirm it against
the packages.

## Notebooks

| # | notebook | master list | what you take away |
|---|----------|:-----------:|--------------------|
| 00 | [ECDF & histograms](00_ecdf_histograms.ipynb) | 1.1–1.3 | rank→uniform→normal; Glivenko–Cantelli; the ECDF's degenerate Jacobian |
| 01 | [KDE & Gaussian-mixture CDFs](01_kde_mixture_cdf.ipynb) | 1.4–1.6 | smooth CDFs; analytic mixture log-det; choosing $h$ / $K$ (BIC) |
| 02 | [Monotone-spline CDFs](02_spline_cdf.ipynb) | 1.7–1.8 | monotonicity (PCHIP vs overshoot); RQS with exact inverse + analytic log-det |
| 03 | [Learnable mixture-CDF](03_learnable_mixture_cdf.ipynb) | 1.9 | the marginal as a trainable layer; end-to-end MLE |
| 04 | [Inversion strategies](04_inversion_strategies.ipynb) | 1.10–1.12 | bisection vs Newton; safeguarded hybrid; differentiating the inverse (unroll/one-step/adjoint); batched `vmap` |

Each estimator notebook also carries a **Jacobian / log-determinant** section
($\mathrm{d}z/\mathrm{d}x = f(x)/\varphi(z)$, $\log|T'| = \log f(x) - \log\varphi(z)$),
since that per-coordinate gradient is the term a flow sums in `log_prob`.

## Threads from Part 0

- The **change-of-variables** log-det ([Part 0, 00](../00_foundations/00_change_of_variables.ipynb))
  becomes concrete here: each estimator's $f(x)$ *is* its log-det.
- The **forward/inverse** trade-off ([Part 0, 02](../00_foundations/02_forward_vs_inverse.ipynb))
  is realised: smooth CDFs invert by root-find, splines invert in closed form.
- **Differentiating** the root-find inverse — and the live
  [gauss_flows#111](https://github.com/jejjohnson/gauss_flows/issues/111) zero-gradient
  pitfall — is covered in notebook 04.

## Running

Same `uv` environment as
[Part 0](../00_foundations/README.md#running) (`rbig` + `gauss_flows` + a Jupyter
stack), with `interpax` added for `gauss_flows.HistogramCDF`:

```bash
cd projects/gaussianization
uv pip install --python .venv-tutorials/bin/python interpax
.venv-tutorials/bin/jupyter nbconvert --to notebook --execute --inplace \
  notebooks/01_marginal_transforms/0*.ipynb --ExecutePreprocessor.timeout=600
```

Notebooks are paired (`jupytext`, `py:percent`) and set `jax_enable_x64`.
