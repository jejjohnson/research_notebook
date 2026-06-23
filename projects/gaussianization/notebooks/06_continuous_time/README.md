---
title: "Part 6 — Continuous-time Gaussianization"
short_title: "Part 6 · Continuous-time"
subject: gaussianization tutorial
---

# Part 6 — Continuous-time Gaussianization

Parts 3–5 built Gaussianization out of a *finite* stack of bijectors — rotate,
Gaussianize the margins, repeat. Take that stack to its infinite-depth limit and
the discrete composition becomes an **ordinary differential equation**: a learned
vector field $\dot x = f_\theta(t, x)$ whose flow transports the data distribution
to $\mathcal{N}(0, I)$ along a continuous path. This part is the **bridge** between
the explicit-Jacobian flows of Parts 4–5 and the stochastic, score-based
Gaussianizers of Part 9 (diffusion). It is built on
[`gauss_flows`](https://github.com/jejjohnson/gauss_flows) (FFJORD / neural-ODE
bijections) with `diffrax` doing the integration.

The defining move is the **instantaneous change of variables**: where a discrete
layer adds $\log|\det J|$, a continuous flow *integrates* the trace of the
Jacobian along the trajectory,

$$
\frac{\mathrm{d}\log p_t(x_t)}{\mathrm{d}t} = -\operatorname{tr}\!\big(\partial_x f_\theta(t, x_t)\big),
$$

so the log-density is a line integral and the vector field needs **no
architectural invertibility constraint** at all.

## Notebooks

| # | notebook | master list | what you take away |
|---|----------|:-----------:|--------------------|
| 00 | [FFJORD on two moons](00_ffjord_2d.ipynb) | 6.1 | the instantaneous change of variables; train a CNF; data ⇄ $\mathcal{N}(0,I)$ transport |
| 01 | [Hutchinson trace estimator](01_hutchinson_trace.ipynb) | 6.2 | $O(d)$ exact trace → $O(1)$ stochastic estimate; bias/variance; when each wins |
| 02 | [Matrix-exponential neural flow](02_matrix_exponential_flow.ipynb) | 6.3 | linear ODE $\dot x = Ax$ with closed-form $\log|\det| = T\operatorname{tr}(A)$ |
| 03 | [Latent ODE on spirals](03_latent_ode_spirals.ipynb) | 6.4 | encode → latent ODE → decode; Gaussianization on the latent state |

## The headline: the trace, not the determinant

A discrete coupling layer earns a *free* log-det by being triangular (Part 5). A
continuous flow earns it differently: the log-density change is the **trace** of
the Jacobian integrated over time, and the trace is cheap to estimate even when the
full Jacobian is not. Notebook 01 is the crux — the exact trace costs $O(d)$
Jacobian-vector products per ODE step, and **Hutchinson's estimator** trades that
for an $O(1)$ stochastic probe, which is what makes free-form continuous flows
scale past toy dimensions.

## Threads

- **Back to Parts 4–5.** A CNF is the infinite-depth limit of the stacked blocks;
  the [layer-wise pushforward](../04_parametric_flows/02_layerwise_inspection.ipynb)
  becomes a continuous trajectory here.
- **Forward to Part 9.** The probability-flow ODE of a diffusion model is exactly a
  continuous Gaussianization in this same family — the deterministic counterpart of
  the forward noising SDE. Notebook 00's transport picture is the $\sigma\to 0$
  limit of that story.
- **Latent ODEs (notebook 03)** reappear in Part 11 (irregular time-series).

## Running

Continuous flows need the FlowJax + `diffrax` stack (not the `rbig` env of the
earlier parts). Notebooks are paired (`jupytext`, `py:percent`) and set
`jax_enable_x64`:

```bash
cd projects/gaussianization
PATH="$GF_VENV/bin:$PATH" "$GF_VENV/bin/jupyter" nbconvert --to notebook \
  --execute --inplace notebooks/06_continuous_time/0*.ipynb \
  --ExecutePreprocessor.timeout=1800
```

where `$GF_VENV` is a virtualenv with `gauss_flows`, `flowjax`, `diffrax`,
`matfree`, `optax`, `interpax` (for the latent-ODE notebook's path interpolation),
and a Jupyter stack. FFJORD training solves an ODE per sample, so these are the
slowest notebooks in the curriculum — a couple of minutes each.
