---
title: "Fair learning with frozen Gaussianization flows"
short_title: "Fair Gaussianization"
subtitle: "Overview, reading order, and status"
description: >
  Landing page for the fair-learning sub-project. Indexes the two design
  docs and three executable notebooks that build, test, and benchmark
  Gaussianization-flow-based fairness penalties.
authors:
  - name: J. E. Johnson
    github: jejjohnson
date: 2026-05-27
keywords:
  - fairness
  - Gaussianization
  - normalising flows
  - mutual information
  - CKA
status: design
---

(sec-fair-overview)=
# Fair learning with frozen Gaussianization flows

This sub-project replaces the {abbr}`CKA` fairness penalty in
[`keras-fairkl`](https://github.com/jejjohnson/keras-fairkl) with a
family of penalties built from a **frozen Gaussianization flow**. The
flow is trained once, frozen, and reused as a differentiable
*Gaussian-space probe* inside any downstream predictor's optimisation
loop.

## The one-paragraph pitch

A Gaussianization flow $T : \mathbb{R}^d \to \mathbb{R}^d$ turns
arbitrary marginals into approximately standard normals while preserving
all statistical dependence ($T(Z) \perp T(Q) \iff Z \perp Q$).  Once
trained and frozen, $T$ acts as a fixed, scale-normalising,
differentiable preprocessor — it absorbs the kernel/bandwidth choices of
{abbr}`CKA` and {abbr}`HSIC` into its mixture-CDF parameters, and turns
"measure non-linear dependence between $z$ and $q$" into "measure
*linear* dependence between near-Gaussian variables." Three concrete
penalties exploit this: {abbr}`G-XCOV`, {abbr}`G-MI`, and {abbr}`G-TC`.

(sec-reading-order)=
## Reading order

```{list-table}
:header-rows: 1
:widths: 6 30 64

* - #
  - Page
  - What it is for
* - 1
  - [](./fair_gaussianization_experiment.md)
  - **Design doc.**  Mental model, math, hypotheses, experiment plan,
    risks, milestones.  Read first.
* - 2
  - [](../notebooks/fair_gauss/05_fair_gauss_pretrain.ipynb)
  - **Notebook 05.**  Pretrain + freeze a flow on a 2-D dataset; four
    diagnostics that prove the flow Gaussianises, freezes, and inverts.
* - 3
  - [](../notebooks/fair_gauss/06_fair_gauss_synthetic.ipynb)
  - **Notebook 06.**  Fair MLP regression on synthetic data;
    Pareto curve of $(\text{RMSE}, |\mathrm{corr}(\hat y, q)|)$ across
    {abbr}`G-XCOV`, {abbr}`G-MI`, {abbr}`G-TC`, and {abbr}`CKA`.
* - 4
  - [](../notebooks/fair_gauss/07_fair_gauss_adult.ipynb)
  - **Notebook 07.**  Same setup on UCI Adult Census; Pareto curves on
    AUC vs. {abbr}`DP`-diff and {abbr}`EO`-diff.
* - 5
  - [](./fair_gaussianization_followups.md)
  - **Follow-up doc.**  Seven *input-side* alternatives that move the
    flow from the predictor's output to its input / representation /
    data pipeline.
```

(sec-three-losses)=
## The three penalties at a glance

```{list-table} Output-side fairness penalties built on a frozen Gaussianization flow.
:header-rows: 1
:name: tbl-three-losses

* - Loss
  - Captures
  - Closed form?
  - Joint flow needed?
* - {abbr}`G-XCOV`
  - 2nd-moment dependence in Gaussianised space (linear {abbr}`CKA`)
  - yes
  - no — two marginal flows
* - {abbr}`G-MI`
  - {abbr}`MI` assuming joint-Gaussian after Gaussianisation
  - yes
  - no — two marginal flows
* - {abbr}`G-TC`
  - Full {abbr}`MI` / total correlation, no joint-Gaussian assumption
  - no — via flow {abbr}`NLL`
  - **yes** — one joint flow over $(z, q)$
```

All three are differentiable in the predictor's parameters and plug
into `FairModelWrapper` via its `fairness_loss=...` argument.  See
[`§4 of the design doc`](./fair_gaussianization_experiment.md) for the
math, and [Table 4.4](./fair_gaussianization_experiment.md) for the
property comparison.

(sec-fair-status)=
## Status

```{list-table}
:header-rows: 1
:widths: 4 60 36

* -
  - Milestone
  - Acceptance
* - ✅
  - Skeleton: `fair/{losses,freeze,pretrain,metrics}.py` + tests
  - `pytest tests/test_fair.py` green
* - ✅
  - Notebook 05: pretrain + freeze + 4 diagnostics
  - Executed and committed
* - ✅
  - Notebook 06: synthetic Pareto with {abbr}`G-XCOV` vs {abbr}`CKA`
  - Pareto curve from RMSE 0.11 → 1.35
* - ✅
  - Notebook 07: Adult Pareto with {abbr}`G-XCOV` vs {abbr}`CKA`
  - Pareto traced
* - 🟡
  - {abbr}`G-MI` + {abbr}`G-TC` losses + tests
  - In flight
* - 🟡
  - Notebooks 06/07 re-executed with {abbr}`G-MI` and {abbr}`G-TC` curves
  - Pending
* - ⏳
  - H3 quadratic-dependence experiment (`08_quadratic_dependence.ipynb`)
  - Pending
* - ⏳
  - Input-side follow-ups (see [](./fair_gaussianization_followups.md))
  - Pending
```
