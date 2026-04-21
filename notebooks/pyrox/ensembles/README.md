---
title: Ensemble methods — MAP and VI via `ensemble_step`
---

# Ensemble methods — MAP and VI via `ensemble_step`

Ensemble inference represents the posterior with a *particle set* $\{\theta^{(j)}\}_{j=1}^J$ rather than a parametric distribution. Instead of sampling a Markov chain or optimizing variational parameters, you push the particles around with an update rule that mimics the effect of the posterior gradient. The family traces back to the ensemble Kalman filter {cite}`evensen2003enkf` and its "inverse problems" relatives — ensemble Kalman inversion (EKI) {cite}`iglesias2013eki`, the ensemble Kalman sampler (EKS) {cite}`garbunoInigo2020eks`, and calibrate-emulate-sample pipelines {cite}`cleary2021caliki`. Pyrox bundles these into two primitives — `ensemble_step` and the `EnsembleMAP` / `EnsembleVI` runners — that drop into Equinox + NumPyro models with no custom plumbing.

## Model

Given an unnormalized log posterior $\log \tilde p(\theta) = \log p(\theta) + \log p(\mathbf{y} \mid \theta)$, we maintain $J$ particles $\theta^{(j)}$. Let $\bar\theta = \tfrac{1}{J}\sum_j \theta^{(j)}$ be the ensemble mean and $C_\theta = \tfrac{1}{J-1}\sum_j (\theta^{(j)} - \bar\theta)(\theta^{(j)} - \bar\theta)^\top$ the ensemble covariance.

### MAP via ensemble gradient

For maximum-a-posteriori estimation, each step pre-conditions the log-posterior gradient with the ensemble covariance and applies a standard gradient step:

$$
\theta^{(j)}_{t+1} = \theta^{(j)}_t + \eta_t\, C_\theta\, \nabla_\theta \log \tilde p\bigl(\theta^{(j)}_t\bigr).
$$

The ensemble covariance acts as a data-driven metric tensor, so the update is invariant to linear reparameterizations of $\theta$ — a practical advantage over vanilla SGD on constrained parameters.

### VI via ensemble-Langevin dynamics

Adding a random-walk term turns the step into a sampler. EKS / interacting-Langevin updates {cite}`garbunoInigo2020eks`:

$$
\theta^{(j)}_{t+1} = \theta^{(j)}_t + \eta_t\, C_\theta\, \nabla_\theta \log \tilde p\bigl(\theta^{(j)}_t\bigr) + \sqrt{2\eta_t\, C_\theta}\; \xi^{(j)}_t,\qquad \xi^{(j)}_t \overset{\text{iid}}{\sim} \mathcal{N}(0, I).
$$

In the $J \to \infty$ / $\eta_t \to 0$ limit this is a Langevin diffusion targeting $p(\theta \mid \mathbf{y})$; in practice a modest $J$ (tens to a few hundred) is already competitive with mean-field VI {cite}`blei2017vi` on problems where the posterior is curved or multi-modal.

## Numerical considerations

- **Ensemble size $J$.** Rank-$J$ covariance $C_\theta$ is singular when $\dim(\theta) > J$. Two practical fixes: localization (zero out off-diagonal entries beyond a correlation cutoff — inherited from geophysical data assimilation) or Tikhonov regularization $C_\theta + \lambda I$. The pyrox primitives expose both via hooks.
- **Step-size $\eta_t$.** A constant $\eta$ is fine for MAP if the loss landscape is well-conditioned; for VI, $\eta_t \to 0$ is required for asymptotic correctness. Warmup + cosine-decay schedules are a safe default.
- **Jitter in $\sqrt{C_\theta}$.** The noise term requires a Cholesky of $C_\theta$, which is rank-deficient when $J \le \dim(\theta)$. Add $\epsilon I$ before factorizing and document $\epsilon$ as a hyperparameter.
- **Parallelism.** Each particle's gradient evaluation is independent and `jax.vmap`s trivially. The `ensemble_step` primitive already does this; `EnsembleMAP` / `EnsembleVI` compose it with an `optax` optimizer and log history.
- **When not to use it.** Ensembles shine on moderate-dim ($\le 10^3$) physics-inspired problems where gradients are cheap but the posterior is nasty. For flat, high-dimensional neural-net posteriors, NUTS + reparam or mean-field VI {cite}`kucukelbir2017advi` usually wins.

## Notebooks

- [`ensemble_primitives_tutorial`](ensemble_primitives_tutorial.ipynb) — three ways to drive the low-level `ensemble_step` primitive: as a function, as a jit-compiled loop, and inside an `optax` update.
- [`ensemble_runner_tutorial`](ensemble_runner_tutorial.ipynb) — the higher-level `EnsembleMAP` and `EnsembleVI` classes — same runtime, with history logging, config handling, and drop-in use against any NumPyro model.

## References

```{bibliography}
:filter: docname in docnames
```
