---
title: Data assimilation benchmarks
---

# Data assimilation benchmarks

A side-by-side comparison of the seven `pipekit_cycle.AnalysisStep`
methods shipped in [vardax](https://github.com/jejjohnson/vardax) on
two shared chaotic test problems — **Lorenz-63** (3-D, sparse-time
partial obs) and **Lorenz-96** (single-level, 40-D, sparse spatial +
temporal obs). Every notebook loads the *same* problem (truth,
observations, mask, $B$, $R$) via the harness factories
`assimilation.generate_problem(key)` (L63) and
`assimilation.generate_l96_problem(key)` (L96), so the numbers stack
into one comparison table per system.

## Headline — Lorenz-63 (`PRNGKey(42)`)

| Method | RMSE total | Inference | Training | Notes |
|---|---:|---:|---:|---|
| Optimal Interpolation | 16.87 | 0.4 ms | — | Floor — no time coupling, $y, z$ stay at prior |
| 3DVar | 16.87 | 1.0 s | — | Matches OI — Decision D14 linear-Gaussian invariant |
| Weak-4DVar | 7.15 | 2.1 s | — | Allows model error; perfect-model handicap |
| Incremental-4DVar | 2.39 | 7.4 s | — | Operational fast path |
| Strong-4DVar | 0.91 | 1.7 s | — | Dynamics constraint — the textbook 4DVar win |
| FourDVarNet | 1.55 | 0.1 s | 2.5 s | Learned 4DVar solver; ~3 s training on 32 simulated trajectories |
| **AmortizedPosterior** | **0.68** | **8 ms** | 1.9 s | Sub-ms MAP, but **predictive variance mis-calibrated** — see notebook 7 |

## Headline — Lorenz-96 (`PRNGKey(0)`)

K=40 grid points, observe every 4th cell at every 4th time step
(60 obs constraining a 840-entry trajectory):

| Method | RMSE total | Inference | Training |
|---|---:|---:|---:|
| OI = 3DVar | 3.83 | 0.4 s | — |
| FourDVarNet | 3.63 | 0.2 s | ~75 s |
| Weak-4DVar | 3.21 | 3.3 s | — |
| AmortizedPosterior | 3.10 | 12 ms | ~4.5 s |
| Incremental-4DVar | 2.96 | 12 s | — |
| **Strong-4DVar** | **2.94** | 2.5 s | — |

Same shape of story as L63 — dynamics-aware methods cut the
prior-only RMSE — but compressed, because L96's typical state
magnitude (~8) is closer to the zero prior than L63's typical
magnitude (~10-25). FourDVarNet is under-trained in this notebook
(K=40 needs more sims than the K=3 L63 case); train longer for
better numbers.

## Notebooks

**Lorenz-63 (4-D state-space)**

- [`00_lorenz63_setup`](notebooks/00_lorenz63_setup.md) — derivation,
  observation model, harness API.
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
  — all seven L63 side-by-side, table + plots.

**Lorenz-96 (single-level, $K=40$)**

- [`09_lorenz96_setup`](notebooks/09_lorenz96_setup.ipynb) — model,
  simulation, sparse spatial + temporal observation design, sanity
  checks. **Start here for L96.**
- [`10_lorenz96_benchmark`](notebooks/10_lorenz96_benchmark.ipynb) —
  all seven methods on the L96 problem, Hovmöller-overlay plots and
  the accuracy-latency scatter.

**Lorenz-96 two-level ($K=8$ slow, $J=8$ fast → $D=72$)**

- [`11_lorenz96_2l_setup`](notebooks/11_lorenz96_2l_setup.ipynb) —
  Wilks 2005 sub-grid model: equations, slow-fast Hovmöllers,
  slow-only sparse obs design, prior-floor analysis. **Start here
  for the multi-scale problem.**
- [`12_lorenz96_2l_benchmark`](notebooks/12_lorenz96_2l_benchmark.ipynb) —
  six methods on the two-level problem (Incremental-4DVar diverges
  on the stiff slow-fast coupling and is documented as a known
  failure mode). The slow-vs-fast RMSE breakdown is the headline:
  most 4DVar variants improve slow but **degrade fast** (the
  classic "imbalance" failure), while `AmortizedPosterior`
  preserves the fast block near the prior floor.

## Headline — Lorenz-96 two-level (`PRNGKey(0)`)

Per-block RMSE (prior floors: slow 7.58, fast 0.39):

| Method | slow RMSE | fast RMSE | total | Inference | Training |
|---|---:|---:|---:|---:|---:|
| OI = 3DVar | 6.71 | 0.39 | 2.27 | 0.5 s | — |
| FourDVarNet | 6.16 | 1.25 | 2.37 | 0.2 s | ~140 s |
| Strong-4DVar | 4.60 | 0.88 | 1.75 | 2.5 s | — |
| **AmortizedPosterior** | 4.64 | **0.44** | 1.60 | **10 ms** | ~15 s |
| **Weak-4DVar** | **4.12** | 0.84 | **1.58** | 7.1 s | — |
| Incremental-4DVar | (diverges) | | | | |

The take-home: dynamics-aware methods improve slow but **disturb**
the unobserved fast block above its prior floor — the imbalance
failure mode. Only the amortized regression head, which learned the
slow-fast joint structure from 128 simulated pairs, recovers the
slow variables without sacrificing the fast ones.

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
