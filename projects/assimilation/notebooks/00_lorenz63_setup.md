---
title: Lorenz-63 benchmark — problem setup
subtitle: One forward model, seven assimilation methods, the same noisy observations
---

# The Lorenz-63 partial-observation problem

This project benchmarks the seven `pipekit_cycle.AnalysisStep`-compliant
methods shipped in [vardax](https://github.com/jejjohnson/vardax) on a
single shared toy problem. Every per-method notebook loads the
**identical** problem from `assimilation.generate_problem(key)` so that
RMSE / runtime numbers are apples-to-apples.

## The model

The Lorenz-63 system is the three-dimensional ODE

$$
\dot{x} = \sigma (y - x), \qquad
\dot{y} = x (\rho - z) - y, \qquad
\dot{z} = x y - \beta z,
$$

with the canonical chaotic parameters $\sigma = 10$, $\rho = 28$,
$\beta = 8/3$. It's the textbook test-bed for nonlinear filtering and
4DVar: the dynamics are chaotic but low-dimensional, the attractor
fits on a screen, and ground-truth trajectories are cheap to
re-simulate.

We integrate with a single RK4 step of $\Delta t = 0.01$ per
forecast step (see `assimilation.lorenz63.Lorenz63Forward`) and run
the benchmark over a window of $T = 40$ steps (so trajectories have
length $T + 1 = 41$).

## The observations

The whole *point* of the benchmark is to expose what each method gets
out of a hard observation regime. We use the canonical Lorenz
partial-observation setup:

| Component | Observed? |
|---|---|
| $x$ | every $4^\text{th}$ step, with Gaussian noise $\sigma_o = 1$ |
| $y$ | **never** |
| $z$ | **never** |

That's 11 noisy scalar observations to constrain a 41-step trajectory
in three components — 123 unknowns total. With identity prior
covariance (no time coupling, no cross-component coupling), the only
way to recover $y$ and $z$ is via the **dynamics**: that's the whole
point of 4DVar, and the reason OI / 3DVar with diagonal $B$ are
indistinguishable from the prior on the unobserved components.

## The shared problem object

`assimilation.generate_problem(key=jax.random.PRNGKey(42))` returns a
`LorenzProblem` dataclass with:

| Field | Shape | Used by |
|---|---|---|
| `truth` | $(T+1, 3)$ | metrics |
| `obs` | $(T+1, 3)$ | every method (zero at masked entries) |
| `mask` | $(T+1, 3)$ | every method |
| `prior_mean` | $(T+1, 3)$ | OI, 3DVar, FourDVarNet, AmortizedPosterior |
| `prior_mean_state` | $(3,)$ | strong / weak / incremental 4DVar (control = $x_0$) |
| `B_op`, `R_op` | $(T+1, 3) \to (T+1, 3)$ | OI, 3DVar, learned |
| `B_op_state`, `R_op_state` | $(3,) \to (3,)$ | 4DVar family |
| `dt`, `T`, `obs_every`, `obs_noise` | scalars | bookkeeping |

`B = \sigma_b^2 I$ with $\sigma_b = 5$ (diagonal, untagged time coupling).
`R = \sigma_o^2 I$ with $\sigma_o = 1$. Both are wrapped in a
`lineax.TaggedLinearOperator(..., positive_semidefinite_tag)` so that
the inner `lineax.CG` solver inside the 3DVar / incremental cost
accepts them.

## The benchmark harness

Each method notebook follows the same five-section pattern:

1. **Load** — call `generate_problem(key)` and build the `Batch1D`.
2. **Build** — instantiate the method (one of the seven `AnalysisStep`).
3. **Run** — wrap the analysis in `assimilation.run_method(name, fn, problem)`
   which times the call and computes RMSE.
4. **Inspect** — print per-component metrics, plot the trajectory.
5. **(Optional) Train** — only `FourDVarNet` and `AmortizedPosterior`
   need this; both train on a vmap-batch of fresh simulated problems.

The comparison notebook ([`08_benchmark_comparison`](08_benchmark_comparison.ipynb))
loads every per-method result into a `pandas` table and overlays the
trajectories. That's where the headline story lives.

## What to expect

Roughly (your numbers will vary with the random seed):

| Method | RMSE | Why |
|---|---|---|
| Optimal Interpolation | ~16 | Diagonal $B$, no time coupling. Only observed entries get the obs; the rest stays at the prior. |
| 3DVar | ~16 | Identical to OI in this linear-Gaussian limit (Decision D14 invariant). |
| Strong-4DVar | ~1 | Dynamics constraint recovers $y, z$ from $x$ obs through the chaotic mixing. |
| Weak-4DVar | ~7 | Looser fit: model error term lets the trajectory drift off the strong-constraint attractor. |
| Incremental-4DVar | ~2 | The operational fast path; close to strong-4DVar at a fraction of the inner-solve cost. |
| FourDVarNet | ~0.5 | Learned solver; outperforms strong-4DVar after a few seconds of training on simulated trajectories. |
| AmortizedPosterior | ~0.05 | Sub-millisecond MAP; matches the training distribution well but **NLL is large** — the variances are mis-calibrated. The six-step cycle (Decision D12) is the safeguard. |

The big jump from ~16 to ~1 is the value of dynamics. The drop from
~1 to ~0.5 is the value of a *learned* solver over a fixed iterative
one. The drop from ~0.5 to ~0.05 is what amortization buys — at the
cost of needing simulation-based training and accepting calibration
risk.
