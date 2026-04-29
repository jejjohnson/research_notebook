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

---

## The cycle, in one diagram

Every tier in `plumax` follows the same six-step loop:

```
Simple model → model-based inference → emulator
            → emulator-based inference → amortized predictor → improve
```

The cycle structure is the architecture. Each tier swaps in a richer forward model, but the inference loop, the validation tests, and the upgrade discipline are the same. See the [roadmap index](roadmap/README.md) for the full diagram and the rationale for each step.
