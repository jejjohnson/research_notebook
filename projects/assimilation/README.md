---
title: Data assimilation benchmarks
---

# Data assimilation benchmarks

A side-by-side comparison of the seven `pipekit_cycle.AnalysisStep`
methods shipped in [vardax](https://github.com/jejjohnson/vardax) on
three shared chaotic test problems — **Lorenz-63** (3-D), **Lorenz-96
single-level** (40-D), and **Lorenz-96 two-level** (Wilks 2005 multi-
scale, 72-D). Following Ahmed et al. 2020
([PyDA](https://github.com/Shady-Ahmed/PyDA)), every notebook runs
the analysis on a short assim window and then **free-forecasts** for
many Lyapunov times so the chaotic divergence (and how well each
method's $x_0$ suppresses it) is the headline view. RMSE(t) traces
overlay all methods on a single plot.

## Headline — Lorenz-63 (`PRNGKey(42)`)

0.5-time-unit assim + 9.5-time-unit free-forecast (~9 Lyapunov times).
Full state observed every 0.05 time units, Gaussian noise std 0.5.

| Method | rmse_assim | rmse_forecast | rmse_total | Inference | Training |
|---|---:|---:|---:|---:|---:|
| **Strong-4DVar** | **0.05** | **0.08** | **0.08** | 1.7 s | — |
| AmortizedPosterior | 0.32 | 0.43 | 0.42 | 8 ms | ~2 s |
| Weak-4DVar (T_assim=10) | 0.42 | 0.76 | 0.75 | 2.1 s | — |
| FourDVarNet | 6.45 | 0.91 | 1.71 | 0.1 s | ~3 s |
| OI = 3DVar | 15.11 | 1.09 | 3.57 | (D14) | — |
| Incremental-4DVar | 1.20 | 2.17 | 2.13 | 8 s | — |

`rmse_assim` is dominated by unobserved time slices for OI / 3DVar
(no temporal coupling in $B$), so the forecast column is the real
discriminator. **Strong-4DVar's analysis recovers $x_0$ to ~0.05 RMSE
and the perfect-model integration stays on the truth's attractor
branch for the full 10-time-unit window** — a 14× forecast-skill win
over OI / 3DVar.

## Headline — Lorenz-96 single-level (`PRNGKey(0)`)

0.5-time-unit assim + 2-time-unit free-forecast (~5 L96 Lyapunov
times). Spatial + temporal sparse obs on every 4th cell every 0.05
time units (110 of 2010 entries observed).

| Method | rmse_assim | rmse_forecast | rmse_total | Inference |
|---|---:|---:|---:|---:|
| **Strong-4DVar** | 3.74 | **4.14** | **4.06** | 2.6 s |
| OI / 3DVar | 3.93 | 4.21 | 4.16 | (D14) |
| AmortizedPosterior | 3.07 | 4.48 | 4.23 | 0.2 s |
| FourDVarNet | 3.18 | 4.73 | 4.46 | 0.4 s |
| Incremental-4DVar | 4.63 | 5.12 | 5.02 | 13 s |

L96's faster Lyapunov clock (~0.5 time units) means even Strong-4DVar's
forecast saturates near the attractor RMS within 2 time units. The
separation between methods is most visible in the **RMSE(t) trace**
in notebook 10 — methods with good $x_0$ recovery stay below the
saturation level for ~1 extra Lyapunov time.

## Headline — Lorenz-96 two-level (`PRNGKey(0)`)

0.4-time-unit assim + 0.6-time-unit free-forecast (~2 slow Lyapunov
times). Slow obs every 2nd cell every 0.05 time units; **fast
unobserved**.

Per-block RMSE (prior floors: slow 7.15, fast 0.35):

| Method | slow | fast | total | rmse_forecast | Inference |
|---|---:|---:|---:|---:|---:|
| **Strong-4DVar** | **1.66** | 0.49 | **0.72** | **0.52** | 4.6 s |
| OI / 3DVar | 6.07 | 0.38 | 2.05 | 1.97 | (D14) |
| AmortizedPosterior | 6.82 | 0.39 | 2.30 | 2.57 | 0.2 s |
| Weak-4DVar | 8.90 | 0.47 | 3.00 | 3.49 | ~7 s |
| FourDVarNet | NaN | NaN | NaN | NaN | (under-trained — see notebook) |
| Incremental-4DVar | (diverges) | | | | |

**Strong-4DVar recovers slow at 4× the prior accuracy**; OI / 3DVar
and Amortized stay near the prior on the slow block. The fast block
sits near the prior floor for everyone except weak (which diverges
on this stiff coupling). Incremental-4DVar's GN linearisation can't
handle the slow-fast stiffness; FourDVarNet's 72-D head training is
under-tuned at the budget used here.

## Notebooks

**Lorenz-63**

- [`00_lorenz63_setup`](notebooks/00_lorenz63_setup.md) — prose intro.
- [`01_optimal_interpolation`](notebooks/01_optimal_interpolation.ipynb)
  through
  [`07_amortized_posterior`](notebooks/07_amortized_posterior.ipynb) —
  seven per-method notebooks, each: trajectory plot (truth + obs +
  analysis-then-forecast), RMSE(t) trace, discussion.
- [`08_benchmark_comparison`](notebooks/08_benchmark_comparison.ipynb)
  — all seven methods overlaid + comparison table + forecast-skill
  bar chart.

**Lorenz-96 single-level**

- [`09_lorenz96_setup`](notebooks/09_lorenz96_setup.ipynb) — model,
  simulation, sparse spatial+temporal obs design, sanity checks.
- [`10_lorenz96_benchmark`](notebooks/10_lorenz96_benchmark.ipynb) —
  all methods with Hovmöllers + RMSE(t) trace.

**Lorenz-96 two-level (Wilks 2005)**

- [`11_lorenz96_2l_setup`](notebooks/11_lorenz96_2l_setup.ipynb) —
  Wilks 2005 sub-grid model, slow-only obs design.
- [`12_lorenz96_2l_benchmark`](notebooks/12_lorenz96_2l_benchmark.ipynb)
  — six methods with per-block RMSE breakdown.

## Running

```bash
# From repo root, set up the env (uv)
uv pip install -e projects/assimilation

# Execute every notebook end-to-end (skips the 00 setup .md)
for f in projects/assimilation/notebooks/{0[1-8],09,10,11,12}_*.py; do
  uv run jupytext --to ipynb --execute "$f"
done

# Or open one in JupyterLab
uv run jupyter lab projects/assimilation/notebooks/03_strong_4dvar.ipynb
```

The whole 12-notebook run takes ~10-15 minutes on a laptop CPU; the
learned-method training dominates the runtime.

## Code layout

```
projects/assimilation/
├── pyproject.toml                 # vardax + scientific stack
├── README.md
├── src/assimilation/
│   ├── lorenz63.py                # Lorenz63Forward + generate_problem
│   ├── lorenz96.py                # L96 single-level
│   ├── lorenz96_2l.py             # L96 two-level (Wilks 2005)
│   ├── metrics.py                 # rmse, sigma_coverage, nll_gaussian
│   └── benchmark.py               # assim_batch, free_forecast,
│                                  # assemble_full_trajectory, run_method, compare
├── tests/
│   └── test_lorenz96_2l.py        # vector-field regression tests
└── notebooks/
    ├── 00_lorenz63_setup.md       # prose intro for L63
    └── 0[1-8],09,10,11,12_*.ipynb + .py
```

The assim+forecast harness is in `benchmark.py`: methods run on
`assim_batch(problem)`, return either an `(T_assim+1, D)` trajectory
or just `x_0`, then `assemble_full_trajectory` extends to
`(T_total+1, D)` by free-forecasting the rest. `MethodResult` carries
`rmse_assim`, `rmse_forecast`, `rmse_total`, and a `rmse_trace(t)` for
plotting.

## Known limitations

- **Weak-4DVar on L63 with T_assim ≥ 50**: BFGS diverges due to the
  enlarged $(x_0, \eta_t)$ control space on the chaotic-rollout cost
  surface. Notebook 04 uses a shorter T_assim=10 window; notebook 08
  runs weak in this reduced regime.
- **Weak-4DVar on L96-2L**: similar divergence; reported as a
  cautionary entry in the table.
- **Incremental-4DVar on L96-2L**: GN linearisation of the stiff
  slow-fast coupling drives the Hessian near-singular; omitted with
  a documented note.
- **FourDVarNet on L96-2L (D=72)**: under-tuned at the training
  budget used (results are NaN). Longer training + LR tuning closes
  the gap; left as a follow-up.

## Open follow-ups

- **Multi-cycle DA via `vardax.VarDACycle`** — chain many consecutive
  assim windows back-to-back to demonstrate operational cycling
  behaviour.
- **Six-step gates wired into the amortized validation** —
  `vardax.assert_posterior_agreement` and
  `vardax.simulation_based_calibration` against a Strong-4DVar
  oracle.
- **Structured $B$ (gaussx Matérn)** so OI / 3DVar can couple
  observed and unobserved cells through prior cross-covariances.
