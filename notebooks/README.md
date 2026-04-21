# Notebooks

This directory holds the showcase notebooks rendered into the MyST docs site.

## Layout

Notebooks are organized by the source repo or topic they showcase:

```
notebooks/
├── <source-or-topic>/
│   ├── <name>.ipynb        # executed notebook, outputs embedded
│   └── <name>.md           # optional prose write-up / landing page
└── README.md               # this file
```

## Authoring workflow

1. Open JupyterLab: `pixi run -e jupyterlab lab`.
2. Author the notebook under `notebooks/<source-or-topic>/<name>.ipynb`.
3. Execute all cells end-to-end so outputs (figures, prints, tables) embed
   into the `.ipynb` — the committed file is both the source and the
   rendered output.
4. Optionally add `notebooks/<source-or-topic>/<name>.md` alongside it for
   longer prose commentary that links back to the notebook.
5. Commit. MyST picks up both `.ipynb` and `.md` via the `Notebooks`
   section pattern in `myst.yml` — no TOC edit needed.

Figures render inline via the notebook's own outputs; do not save separate
PNGs or reference a `docs/images/` directory.

## Current showcases

- [`pyrox/`](./pyrox/README.md) — equinox + NumPyro + JAX probabilistic
  programming (GP regression/classification, regression masterclass,
  ensemble methods).
- [`coordax/`](./coordax/README.md) — coordinate-aware arrays for JAX
  (foundations, finite-difference and finite-volume derivatives, ODE/PDE
  integration and parameter estimation).

Larger sub-projects with their own source tree, tests, and pixi feature live
under `projects/` rather than here:

- [`projects/methane_pod/`](../projects/methane_pod/README.md) — thinned
  marked temporal point processes + POD models for satellite methane
  retrieval. Self-contained `methane_pod` package with 4 library modules,
  80+ tests, and 8 notebooks (theory `.md` + executed `.ipynb` galleries +
  a NumPyro NUTS fit on synthetic data).
- [`projects/plume_simulation/`](../projects/plume_simulation/README.md) —
  atmospheric plume dispersion forward models. Currently ships
  `gauss_plume`, a steady-state Gaussian plume (JAX forward model +
  NumPyro Bayesian inference) with 36 tests and three notebooks
  covering forward simulation, emission-rate parameter estimation, and
  time-varying-state estimation.
