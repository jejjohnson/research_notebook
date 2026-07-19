---
title: "`plumax` — Roadmap index"
short_title: "Roadmap index"
subject: "plumax — architectural overview"
authors:
  - name: J. Emmanuel Johnson
    affiliations: [UNEP, IMEO, MARS]
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: [plumax, methane, plume, dispersion, modeling cycle, emulator, amortized inference, tier I, tier II, tier III, tier IV, tier V, MARS]
---

# `plumax` — Roadmap & Architecture

> Mathematical models for plume simulation, methane retrieval, source identification, and emission estimation.

This page is the **index** for the architecture roadmap. The detail for each tier lives in its own file so they can grow independently as design decisions land. The high-level overview (philosophy, tier table, principles) stays here; each tier page expands the math, module layout, validation strategy, and open questions.

---

(cycle-overview)=
## The Data-Driven Modeling Cycle

The architecture has **two axes**:

- The **inference axis** (vertical) is a fixed *six-step loop* — the same recipe at every tier.
- The **complexity axis** (horizontal) is the *forward-model fidelity* — Tier I analytic → Tier V population.

Every column runs the **same** six steps; what changes left-to-right is how complex the model behind each step is. Step 6 — *improve* — is the move that walks you one column to the right.

```text
                    MODEL COMPLEXITY  ─────────────────────────────────────────────▶
                    Tier I         Tier II        Tier III       Tier IV        Tier V
                    Gaussian       Lagrangian     Eulerian FV    Coupled + RTM  Population
                    (analytic)     (stoch. ODE)   (adv–diff PDE) (multi-inst)   (point proc.)
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
1 │ Simple model    plume / puff   particles      adv–diff PDE   transport+RTM  λ(t), f(Q), Pd  │
2 │ Model inference MAP / MCMC     footprint inv  4D-Var         joint fusion   NUTS population │
3 │ Emulator        skip (cheap)   footprint NN   FNO/neural-ODE coupled net    flow over post. │
4 │ Emu. inference  —              EKI            PDE-free 4DVar EKI / gradient SVI             │
5 │ Amortized       Q-predictor    traj-pred.     field-pred.    overpass → Q   basin → λ, tot  │
6 │ Improve ────────┴──────────────┴──────────────┴──────────────┴──────────────┘  ▲ climb a tier
  └───────────────────────────────────────────────────────────────── same six steps ──────────┘
    ▲ INFERENCE AXIS
```

Read **down** a column for one tier's full cycle; read **across** a row to watch a single step grow in complexity tier-to-tier. The point the diagram makes: this is *not* a single linear "model → amortized predictor" pipeline. **Each of the six steps is itself swappable for a more complex model**, and "improve" (Step 6) is structured — it picks one component (a richer forward model, a higher tier, more data, a better posterior family) and re-enters the loop with the previous step (or tier) as ground truth.

- **Step 1 — Simple Model** gives you a **generative story** — a known mathematical structure you can simulate from.
- **Step 2 — Model-Based Inference** gives you **ground-truth inference** — slow but exact, used to validate everything downstream.
- **Step 3 — Model Emulator** makes Step 2 **tractable at scale** — replace the expensive forward model with a fast surrogate (skip if the model is already cheap).
- **Step 4 — Emulator-Based Inference** is Step 2 again, but now running in seconds instead of hours.
- **Step 5 — Amortized Inference (Predictor)** collapses the inference loop entirely — the predictor learns the posterior map directly.
- **Step 6 — Improve** closes the loop — every component is independently upgradable, and the complexity axis tells you *which* component to upgrade and *how* to validate it.

---

(tier-overview)=
## Tier overview

```{list-table} `plumax` tier table — forward models, complexity, and links to detail pages.
:label: tbl-tier-overview
:header-rows: 1

* - Tier
  - Forward model
  - Complexity
  - When to use
  - Detail
* - 0 (prereq)
  - Met field + AK operator
  - Data interface
  - All tiers depend on it
  - [Prerequisites](00_prerequisites.md)
* - I
  - Gaussian plume / puff
  - Analytical
  - Fast prototyping, validation
  - [Tier I — Gaussian family](01_tier1_gaussian.md)
* - II
  - Lagrangian particle / footprint
  - Stochastic ODE
  - Wind-realistic transport
  - [Tier II — Lagrangian](02_tier2_lagrangian.md)
* - III
  - Eulerian finite-volume PDE
  - PDE
  - High-fidelity spatial fields
  - [Tier III — Eulerian FV](03_tier3_eulerian.md)
* - —
  - Radiative transfer (parallel track)
  - Multi-physics
  - Connects any tier to radiances
  - [RTM stack](04_rtm_stack.md)
* - IV
  - Coupled transport + RTM
  - End-to-end
  - Operational satellite → source posterior
  - [Tier IV — Coupled E2E](05_tier4_coupled.md)
* - V
  - Population & forecasting (TMTPP)
  - Stochastic point process
  - Aggregate per-event posteriors → wait times, totals
  - [Tier V — Population](06_tier5_population.md) (and [V.A](06a_instantaneous.md), [V.B](06b_point_process.md), [V.C](06c_persistency.md), [V.D](06d_total_emission.md))
```

The build order is roughly: **Prerequisites → Tier I → RTM stack (parallel) → Tier II → Tier III → Tier IV → Tier V.** RTM is independent of transport tier, so it can be developed in parallel by a different person without coordination cost. Tier V depends on at least Tier I being usable end-to-end (per-event posteriors are the input), but does not need Tiers II–IV — it can launch with Tier I posteriors and absorb richer ones later.

For how the tiers compose into a single operational pipeline — from a satellite radiance all the way to a leak-recurrence forecast — see the [end-to-end retrieval → persistency walkthrough](../end_to_end_retrieval_to_persistency.md).

---

(architectural-principles)=
## Architectural principles

:::{important} 1. The cycle is the architecture
Don't treat emulation and amortization as afterthoughts. Design the forward-model API at Step 1 so Steps 3–5 are natural substitutions, not rewrites.
:::

:::{important} 2. Each step validates the next
The model-based posterior (Step 2) is the ground truth for the emulator posterior (Step 4), which is the ground truth for the amortized predictor (Step 5). Never skip validation; otherwise emulator bugs become posterior bugs.
:::

:::{important} 3. The forward-model interface is fixed across tiers
All four tiers implement the same shape: `forward(params, met) → observations`. Inference code (`vardaX`, `filterax`, NumPyro) is written once and reused. See [Prerequisites — fixed forward interface](00_prerequisites.md#prereqs-forward-interface) for the concrete signature.
:::

:::{important} 4. WRF is a data source, not a competitor
WRF provides met forcing and benchmark concentration fields. `plumax` learns to be **fast, differentiable, and probabilistic** — properties WRF doesn't have.
:::

:::{important} 5. Improvement is structured
Step 6 is not vague iteration. Each improvement targets a specific component — better physics, more training data, richer posterior family, tighter observation operator — and the cycle structure (which row) and the complexity axis (which column) together tell you which component to upgrade and how to validate it.
:::

---

(status-snapshot)=
## Status snapshot (2026-06-02)

Module-level status is tracked per tier. Two things move at different speeds: the **design maturity** (these roadmap pages, kept in sync with the upstream [`plumax`](https://github.com/jejjohnson/plumax) reference implementation) and the **in-repo `plume_simulation` port**, which currently mirrors Tier I + the RTM stack + the Tier III scaffolding. Quick overview:

- **Tier I — Gaussian:** ✓ plume + puff forward models, ✓ MAP/MCMC inversion (port + upstream). Emulator + amortized predictor not yet started.
- **Tier II — Lagrangian:** 🚧 upstream `plumax.lagrangian` now lands the forward model + model-based inference — Markov-1 Langevin particles, homogeneous + Hanna turbulence, forward residence-time concentration, backward footprint, and closed-form Gaussian / lognormal source inversion with a Matérn-3/2 prior. Not yet ported into this repo. Footprint emulator + predictor (Steps 3–5) not started.
- **Tier III — Eulerian FV:** 🚧 [`les_fvm`](../../src/plume_simulation/les_fvm/) advection/diffusion/dynamics implemented in-repo; upstream now wires the strong-constraint **4D-Var loop end-to-end** (differentiable emission→column-obs forward, whitened temporal Matérn-3/2 control space, L-BFGS with the exact discrete adjoint via reverse-mode AD through the diffrax FV solver), plus a Gauss–Newton Laplace **posterior covariance** around the MAP (Hessian via `gaussx`). The in-repo [`assimilation`](../../src/plume_simulation/assimilation/) cost/control/solve scaffolding tracks the earlier snapshot.
- **RTM stack:** 🚧 [`hapi_lut`](../../src/plume_simulation/hapi_lut/) LUT generator + Beer–Lambert, [`radtran`](../../src/plume_simulation/radtran/) instrument/SRF/forward modules, and [`matched_filter`](../../src/plume_simulation/matched_filter/) detection pipeline all in place. Optimal-estimation retrieval not wired; neural RTM not started.
- **Tier IV — Coupled:** 🚧 upstream `plumax.coupled` lands v1 multi-instrument fusion — the Tier I plume + averaging-kernel coupled forward kept per-instrument at native resolution (`CoupledForward`, `Instrument`), a closed-form joint posterior over `(Q, bias_inst)` across satellites (`fuse_observations`, exploiting the plume's linearity in `Q`), and an additive RTM-based observation operator (`RadianceObservationOperator`) mapping plume column enhancement → gas ΔVMR → band-integrated normalised radiance. Full Tier II/III + RTM coupling, the `Q(t)` stochastic process, and trans-dimensional source count are future work. Not yet ported into this repo.
- **Tier V — Population & forecasting:** 🚧 the standalone [`methane_pod`](../../../methane_pod/) library is feature-complete (intensity catalog, POD catalog, paradox simulator, NUTS fitter, synthetic-data validation). Upstream now also lands the in-tree `plumax.population` v1 core: the tier-agnostic cross-tier posterior catalog (`event_from_posterior` over Gaussian / lognormal / fusion posteriors), the V.A hierarchical lognormal size-distribution fit with per-event uncertainty propagation, and the V.B point-process core (closed-form Gamma–Poisson rate + log-linear inhomogeneous intensity). Still missing: the importance-corrected TMTPP mark likelihood, real-data CSV ingestion, multi-satellite POD fusion, and the LGCP intensity.

See each tier page for the module-level breakdown, and the [end-to-end retrieval → persistency walkthrough](../end_to_end_retrieval_to_persistency.md) for how the tiers compose into one operational pipeline.
