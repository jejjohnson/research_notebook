---
title: "`plumax` — Roadmap & Architecture"
short_title: "plumax roadmap"
subject: "plumax — top-level pointer"
authors:
  - name: J. Emmanuel Johnson
    affiliations: [UNEP, IMEO, MARS]
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: [plumax, methane, plume simulation, retrieval, source identification, emission estimation, roadmap]
---

# `plumax` — Roadmap & Architecture

> Mathematical models for plume simulation, methane retrieval, source identification, and emission estimation.

The detailed roadmap now lives in [`roadmap/`](roadmap/README.md), one file per tier. This page is a one-screen pointer. Start with the [roadmap index](roadmap/README.md) for the philosophy, the tier table, and the architectural principles.

---

## Quick links

- [Roadmap index — philosophy, tier table, principles](roadmap/README.md)
- [Prerequisites — Met field infrastructure](roadmap/00_prerequisites.md)
- [Tier I — Gaussian family](roadmap/01_tier1_gaussian.md)
- [Tier II — Lagrangian particle transport](roadmap/02_tier2_lagrangian.md)
- [Tier III — Eulerian finite-volume transport](roadmap/03_tier3_eulerian.md)
- [Radiative transfer (RTM) stack — parallel track](roadmap/04_rtm_stack.md)
- [Tier IV — End-to-end coupled system](roadmap/05_tier4_coupled.md)
- [Tier V — Source population & forecasting](roadmap/06_tier5_population.md)
  - [V.A — Instantaneous emission estimation](roadmap/06a_instantaneous.md)
  - [V.B — Point process model (TMTPP)](roadmap/06b_point_process.md)
  - [V.C — Persistency](roadmap/06c_persistency.md)
  - [V.D — Total emission estimation](roadmap/06d_total_emission.md)
- [**Operational attribution — unified-stack success-story plan**](https://jejjohnson.github.io/research_journal_v2/operational-attribution) — projected end-to-end pipeline (3 instruments, 3 orchestration phases) showing how MARS-style attribution rebuilds on `georeader` + `GeoCatalog` + `geotoolz` + `plumax`. *(Now lives in the [research journal](https://github.com/jejjohnson/research_journal_v2) under `notes/geotoolz/` alongside the rest of the geotoolz design notes.)*

---

(roadmap-cycle)=
## The cycle, in one diagram

The architecture has **two axes**: an **inference axis** (the six-step loop, fixed at every tier) and a **complexity axis** (the forward-model fidelity, Tier I → V).

```text
                         MODEL COMPLEXITY  ─────────────────────────▶
                  Tier I      Tier II     Tier III    Tier IV    Tier V
  inference  ┌──  Gaussian    Lagrangian  Eulerian    Coupled    Population
  axis       │    (analytic)  (stoch ODE) (PDE)       (+RTM)     (point proc.)
    │        │
    ▼   1 Simple model → 2 Model inference → 3 Emulator
        → 4 Emulator inference → 5 Amortized predictor → 6 Improve ──┐
        ▲                                                            │
        └──────────────  climb one tier (complexity axis)  ◀─────────┘
```

The cycle structure is the architecture. Each tier swaps in a richer forward model, but the inference loop, the validation tests, and the upgrade discipline are the same — and *each of the six steps* can itself become a more complex model as you move right along the complexity axis. Step 6 ("improve") is the move that climbs one tier. See the [roadmap index](roadmap/README.md) for the full two-axis diagram and the rationale for each step.
