---
title: Data assimilation — Lorenz-63 benchmark
---

# Data assimilation — Lorenz-63 benchmark

A side-by-side comparison of the seven `pipekit_cycle.AnalysisStep`
methods shipped in [vardax](https://github.com/jejjohnson/vardax) on a
single shared Lorenz-63 partial-observation problem. Every notebook
loads the *same* problem (truth, observations, mask, $B$, $R$) via
`assimilation.generate_problem(key)` so the numbers stack into one
comparison table.

## Headline (one run, `PRNGKey(42)`)

| Method | RMSE total | Inference | Training | Notes |
|---|---:|---:|---:|---|
| Optimal Interpolation | 16.87 | 0.4 ms | — | Floor — no time coupling, $y, z$ stay at prior |
| 3DVar | 16.87 | 1.0 s | — | Matches OI — Decision D14 linear-Gaussian invariant |
| Weak-4DVar | 7.15 | 2.1 s | — | Allows model error; perfect-model handicap |
| Incremental-4DVar | 2.39 | 7.4 s | — | Operational fast path; tune `(n_outer, n_inner)` for accuracy |
| Strong-4DVar | 0.91 | 1.7 s | — | Dynamics constraint — the textbook 4DVar win |
| FourDVarNet | 1.55 | 0.1 s | 2.5 s | Learned 4DVar solver; ~3 s training on 32 simulated trajectories |
| **AmortizedPosterior** | **0.68** | **8 ms** | 1.9 s | Sub-ms MAP, but **predictive variance mis-calibrated** — see notebook 7 |

## Notebooks

- [`00_lorenz63_setup`](notebooks/00_lorenz63_setup.md) — problem
  derivation, observation model, harness API.
- [`01_optimal_interpolation`](notebooks/01_optimal_interpolation.ipynb)
  — closed-form BLUE.
- [`02_threedvar`](notebooks/02_threedvar.ipynb) — same cost as OI,
  iterative solver.
- [`03_strong_4dvar`](notebooks/03_strong_4dvar.ipynb) — control =
  $x_0$ + dynamics rollout.
- [`04_weak_4dvar`](notebooks/04_weak_4dvar.ipynb) — augmented
  control + model-error term.
- [`05_incremental_4dvar`](notebooks/05_incremental_4dvar.ipynb) —
  Gauss-Newton outer + CG inner.
- [`06_fourdvarnet`](notebooks/06_fourdvarnet.ipynb) — learned
  solver, simulation-based training.
- [`07_amortized_posterior`](notebooks/07_amortized_posterior.ipynb) —
  encoder + regression head, no inner solve.
- [`08_benchmark_comparison`](notebooks/08_benchmark_comparison.ipynb)
  — all seven side-by-side, table + plots.

## Running

```bash
# From repo root, set up the env (uv)
uv pip install -e projects/assimilation

# Execute every notebook end-to-end
for f in projects/assimilation/notebooks/0[1-8]_*.py; do
  uv run jupytext --to ipynb --execute "$f"
done

# Or open one in JupyterLab
uv run jupyter lab projects/assimilation/notebooks/03_strong_4dvar.ipynb
```

The whole 8-notebook run takes ~2 minutes on a laptop CPU; the two
training notebooks (06, 07) dominate the runtime.

## Code layout

```
projects/assimilation/
├── pyproject.toml                 # vardax + scientific stack
├── README.md                      # this file
├── src/assimilation/
│   ├── lorenz63.py                # Lorenz63Forward + generate_problem
│   ├── metrics.py                 # rmse, sigma_coverage, nll_gaussian
│   └── benchmark.py               # MethodResult, run_method, compare
└── notebooks/
    ├── 00_lorenz63_setup.md
    └── 0[1-8]_*.ipynb + .py        # jupytext-paired sources + executed
```

The harness is pure: every metric and the `MethodResult` dataclass
operate on JAX arrays so the notebooks compose cleanly with
`jax.jit` / `jax.vmap` if you want to push the comparison to a sweep
over `(obs_every, obs_noise, T)`.

## Open follow-ups

- **Multi-window cycling demo** using `vardax.VarDACycle` — re-run
  OI vs incremental-4DVar over 10 consecutive 40-step windows.
- **Six-step gate sweep** — wire `vardax.assert_posterior_agreement`
  and `vardax.simulation_based_calibration` against the amortized
  head to make the calibration story quantitative.
- **Structured $B$** — replace `lx.DiagonalLinearOperator` with a
  Matérn covariance via `gaussx` so OI / 3DVar can actually couple
  observed and unobserved components.
