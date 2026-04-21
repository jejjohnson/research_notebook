---
title: Dynamics — ODEs, PDEs, and parameter estimation
---

# Dynamics — ODEs, PDEs, and parameter estimation

Once state lives in a `Field` and derivatives are coordinate-aware, the step
to a full dynamical-system workflow is small: wire the RHS into an ODE
solver, differentiate through it, and optimize. The three notebooks here
integrate a 1-D advection-diffusion PDE with `diffrax` {cite}`kidger2021thesis`,
then invert it for unknown parameters and initial states using
`optax` and `jax.value_and_grad`.

## Forward problem

The target PDE is the linear 1-D advection-diffusion equation

$$
\partial_t T \;=\; -\,U\,\partial_x T \;+\; \kappa\,\partial_{xx} T,
$$

discretized in space with the finite-difference operators from the
[derivatives](../derivatives/README.md) section and integrated in time
with `diffrax.Tsit5` and a PID step controller.

Writing the RHS as $\dot{\mathbf{T}} = f(\mathbf{T}; U, \kappa)$, the
`diffeqsolve` call is an end-to-end-differentiable function of the state
*and* the parameters — this is the lever neural-ODE-style frameworks
{cite}`chen2018node` pull to do gradient-based inversion without manually
assembling tangent equations.

## Inverse problems

Given noisy observations $\{y_k\}$ at times $\{t_k\}$, the two inverse
problems in this section are variations on a least-squares objective:

$$
\mathcal{L}(\theta, \mathbf{T}_0) \;=\; \sum_k \bigl\| \mathbf{T}(t_k; \theta, \mathbf{T}_0) - \mathbf{y}_k \bigr\|^2 \;+\; \mathcal{R}(\theta, \mathbf{T}_0),
$$

where $\theta = (U, \kappa)$ are the PDE parameters, $\mathbf{T}_0$ is the
(unknown) initial state, and $\mathcal{R}$ is an optional prior /
regularizer. Closed-form gradients are infeasible — the solver is implicit
in $t$ — so `jax.value_and_grad` through `diffeqsolve` + `optax.adam` is the
only reasonable workflow at this scale. The equivalence with ensemble
Kalman-type inversion {cite}`evensen2009book` is worth noting: both target
the same MAP point, but gradient optimization is cheaper when derivatives
are available and the forward model is smooth.

## Numerical considerations

- **Adjoint choice.** `diffrax.RecursiveCheckpointAdjoint` is the default
  here — it rolls forward with checkpoints and backpropagates, trading
  memory for gradient accuracy. Pure reverse-mode ("discretize-then-optimize")
  is more accurate but allocates the whole trajectory; continuous adjoint
  ("optimize-then-discretize") is memory-cheap but less accurate. Pick based
  on trajectory length vs. state size.
- **Step controller.** A stiff or near-stiff RHS (large $\kappa$ on a fine
  grid) triggers aggressive step-size shrinking under the PID controller —
  gradient quality degrades long before the forward solve fails. If loss is
  flat but $\nabla\mathcal{L}$ is noisy, tighten `rtol/atol` by an order of
  magnitude and re-check.
- **Parameter scaling.** $U$ and $\kappa$ differ by orders of magnitude at
  realistic values; unscaled `adam` spends most of its budget on the larger
  one. Reparameterize as $\log\kappa$ or use per-parameter learning rates.
- **Initial-state identifiability.** Joint $(\theta, \mathbf{T}_0)$
  estimation is identifiable only when the observation window resolves the
  relevant diffusion/advection timescales. For $t_{\text{obs}} \ll 1/\kappa$,
  the problem collapses to state estimation; for $t_{\text{obs}} \gg L/U$,
  upstream information is wiped out. The notebooks pick observation
  schedules inside the identifiable regime.
- **Validation.** Always run the "twin experiment" first — simulate with
  known parameters, recover them. If the twin fails, the real data will
  too. All three notebooks follow this pattern.

## Notebooks

- [`08_ode_integration`](08_ode_integration.ipynb) — forward solve of
  advection-diffusion with `diffrax`; wrapping state as a `Field` for
  coordinate-aware RHS evaluation; a conservation check on the integrated
  trajectory.
- [`09_ode_parameter_state_estimation`](09_ode_parameter_state_estimation.ipynb)
  — joint recovery of $(U, \kappa)$ and $\mathbf{T}_0$ from noisy
  observations via `optax.adam` through `diffeqsolve`. Twin experiment.
- [`10_pde_parameter_estimation`](10_pde_parameter_estimation.ipynb) —
  learning PDE parameters using `equinox.Module` to keep the
  parameter/state split clean and the `diffrax` solve inside `__call__`.

## References

```{bibliography}
:filter: docname in docnames
```
