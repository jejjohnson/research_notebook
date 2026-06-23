---
title: "Part 7 — Conditional Gaussianization"
short_title: "Part 7 · Conditional"
subject: gaussianization tutorial
---

# Part 7 — Conditional Gaussianization

Every flow so far learned a single density $p(x)$. Make the flow's parameters depend
on a **context** $y$ and it learns a whole family of densities $p(x \mid y)$ at once —
a *conditional Gaussianizer* $T(\cdot \mid y)$ that maps each conditional slice of the
data to the same $\mathcal{N}(0, I)$, with a tractable conditional log-likelihood and
one-pass conditional sampling. This part shows **where** to inject the context (the
base, the couplings, or both), how conditional density estimation works, and how a
conditional flow becomes an **amortised posterior** for inverse problems — the bridge
to the plug-and-play priors of Part 16. Built on
[`gauss_flows`](https://github.com/jejjohnson/gauss_flows).

The defining change from Parts 4-6: a parameter that was constant becomes a function
of $y$,

$$
p(x \mid y) = p_Z\big(T_\theta(x; y)\big)\,\big|\det J_{T_\theta(\cdot;y)}(x)\big|,
$$

and the same NLL training, sampling, and log-det machinery carries straight over.

## Notebooks

| # | notebook | master list | what you take away |
|---|----------|:-----------:|--------------------|
| 00 | [Three ways to condition](00_three_ways_to_condition.ipynb) | 7.1, 7.4 | inject $y$ at the base, the couplings, or both — and how to choose |
| 01 | [Conditional marginals & density estimation](01_conditional_density.ipynb) | 7.2, 7.3 | $y$-dependent CDF margins; calibrated $p(x\mid y)$ on a heteroscedastic benchmark |
| 02 | [Conditional flow as an amortised posterior](02_amortised_posterior.ipynb) | 7.5 | train once on $(x, y=Ax+\eta)$, sample $p(x\mid y)$ for any $y$ — feeds Part 16 |

## The headline: where to put the context

A conditional flow has three slots for $y$ — the **base** $p_Z(\cdot\mid y)$ (per-context
location/scale), the **couplings** $T_{\theta(y)}$ (per-context *shape*), and, for
transforms that cannot natively read a context (rotations, normalisations), a FiLM-style
`Conditioner` wrapper. Notebook 00 fits all four combinations side by side and reads off
the rule of thumb: condition the base for shifts, the couplings for shape changes.

## Threads

- **Back to Part 5.** The conditioner that drove coupling layers is the same machinery;
  here it simply also reads the external context $y$ (cf. 5.17).
- **Forward to Part 16.** A conditional flow trained on $(x, y=Ax+\eta)$ is an amortised
  posterior $p(x\mid y)$ — notebook 02 is the toy that the plug-and-play inverse-problem
  solvers of Part 16 scale up.

## Running

Same FlowJax stack as the earlier parts (no ODE this time, so these are fast).
Notebooks are paired (`jupytext`, `py:percent`) and set `jax_enable_x64`:

```bash
cd projects/gaussianization
PATH="$GF_VENV/bin:$PATH" "$GF_VENV/bin/jupyter" nbconvert --to notebook \
  --execute --inplace notebooks/07_conditional/0*.ipynb \
  --ExecutePreprocessor.timeout=900
```

where `$GF_VENV` is a virtualenv with `gauss_flows`, `flowjax`, `optax`, and a Jupyter
stack.
