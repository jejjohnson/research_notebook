---
title: "Part 4 — Parametric Gaussianization Flows"
short_title: "Part 4 · Parametric flows"
subject: gaussianization tutorial
---

# Part 4 — Parametric Gaussianization Flows

Part 3's RBIG fit each rotation + marginal block **greedily**, one at a time. Part 4
takes the *same* architecture and makes it **trainable**: every block's parameters
are free, and the whole stack is fit end-to-end by maximum likelihood. This part
covers the negative-log-likelihood objective and its log-det anatomy, the RBIG
**warm-start** that initialises a trainable diagonal flow from a greedy fit, and
**layer-wise inspection** to see where a flow does its work — all grounded in
[`gauss_flows`](https://github.com/jejjohnson/gauss_flows). (The coupling flow, and
its warm-start, are the subject of **Part 5**.)

Training uses a small `optax` loop (gradient clipping + a one-cycle cosine learning
rate) on the `gauss_flows` flows directly, rather than the package's convenience
trainer, so the optimisation knobs are explicit.

## Notebooks

| # | notebook | master list | what you take away |
|---|----------|:-----------:|--------------------|
| 00 | [NLL training](00_nll_training.ipynb) | 4.1, 4.3, 4.5 | $\log p = \log p_Z + \log\lvert\det J\rvert$; train `gaussianization_flow`; iterative vs parametric |
| 01 | [RBIG warm-start (diagonal)](01_rbig_warmstart.ipynb) | 3.7 | greedy `fit_rbig` seeds the flow; equal budget → better optimum than random init |
| 02 | [Layer-wise inspection](02_layerwise_inspection.ipynb) | 4.7 | per-layer pushforward + diagnostics; the rotation↔marginal push-pull; `unroll_scan` |

## The recurring hero: coupling

The flows here are **diagonal** (per-coordinate marginals between rotations); the
**coupling** flow — a bijector whose parameters are predicted by a conditioner
network from the other coordinates — is the subject of **Part 5**, which makes the
conditioner the headline. The diagonal-vs-coupling comparison and the coupling RBIG
warm-start (drafted in this part) live there, alongside the coupling↔diagonal
equivalence they lead into.

## Threads

- The **change-of-variables** log-det (Part 0) is the NLL objective here (00).
- **Greedy RBIG** (Part 3) becomes the *initialisation* of a trainable flow (01, 03)
  — this is where the master list's "iterative Gaussianization warm-start"
  (items 3.7–3.8) lives, since warm-starting only matters once there is a parametric
  flow to initialise.
- The **convergence / depth-selection** signal (Part 3, notebook 01) reappears in 04:
  layer-wise inspection shows the work is front-loaded, exactly what early-stopping
  exploits.

## Running

Same `uv` environment as the earlier parts (`rbig` + `gauss_flows` + `optax` + a
Jupyter stack):

```bash
cd projects/gaussianization
.venv-tutorials/bin/jupyter nbconvert --to notebook --execute --inplace \
  notebooks/04_parametric_flows/0*.ipynb --ExecutePreprocessor.timeout=900
```

Notebooks are paired (`jupytext`, `py:percent`) and set `jax_enable_x64`. The
training notebooks (00–03) take ~1–2 minutes each; the inspection notebook (04) is
fast (no training).
