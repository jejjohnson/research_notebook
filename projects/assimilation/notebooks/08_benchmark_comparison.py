# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # L63 benchmark — all seven methods, analysis + free forecast
#
# Every method assimilates the same 0.5-time-unit window, then
# free-forecasts for 9.5 more time units (~9 L63 Lyapunov times).
# The visualisation is PyDA-style: trajectory plots showing the
# truth alongside each method's analysis-plus-forecast, with the
# assim window highlighted; plus an RMSE(t) trace that shows when
# each method's forecast loses skill.

# %%
from __future__ import annotations

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import matplotlib.pyplot as plt
import optax
import vardax as vdx

from assimilation import (
    Lorenz63Forward,
    assemble_full_trajectory,
    assim_batch,
    compare,
    generate_problem,
    run_method,
)


# %% [markdown]
# ## 1. Shared problem

# %%
prob = generate_problem(key=jax.random.PRNGKey(42))
batch = assim_batch(prob)
fwd = Lorenz63Forward(dt=prob.dt)
H_full = lx.IdentityLinearOperator(
    jax.ShapeDtypeStruct(prob.prior_mean.shape, jnp.float32)
)
H_state = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((3,), jnp.float32))
t_axis = jnp.arange(prob.T_total_plus_1) * prob.dt
print(f"T_assim={prob.T_assim} ({prob.T_assim * prob.dt} time units), "
      f"T_total={prob.T_total} ({prob.T_total * prob.dt} time units)")

# %% [markdown]
# ## 2. Classical methods (no training)

# %%
oi = vdx.OptimalInterpolation(
    obs_op=vdx.LinearObs(H_mat=H_full),
    prior_mean=prob.prior_mean, prior_cov_op=prob.B_op, obs_cov_op=prob.R_op,
)
three = vdx.ThreeDVar(
    obs_op=vdx.LinearObs(H_mat=H_full),
    prior_mean=prob.prior_mean, prior_cov_op=prob.B_op, obs_cov_op=prob.R_op,
    max_steps=1000,
)
strong = vdx.StrongFourDVar(
    forward=fwd, obs_op=vdx.LinearObs(H_mat=H_state),
    prior_mean=prob.prior_mean_state, prior_cov_op=prob.B_op_state,
    obs_cov_op=prob.R_op_state, max_steps=2000,
)
inc = vdx.IncrementalFourDVar(
    forward=fwd, obs_op=vdx.LinearObs(H_mat=H_state),
    prior_mean=prob.prior_mean_state, prior_cov_op=prob.B_op_state,
    obs_cov_op=prob.R_op_state,
    config=vdx.IncrementalConfig(n_outer=4, n_inner=30),
)

results = [
    run_method("oi", lambda: assemble_full_trajectory(oi(batch)[0], prob, fwd), prob),
    run_method("3dvar", lambda: assemble_full_trajectory(three(batch)[0], prob, fwd), prob),
    run_method("strong_4dvar",
               lambda: assemble_full_trajectory(strong(batch)[0], prob, fwd), prob),
    run_method("incremental_4dvar",
               lambda: assemble_full_trajectory(inc(batch)[0], prob, fwd), prob),
]
for r in results:
    print(f"{r.name:20s} assim={r.rmse_assim:7.3f}  "
          f"forecast={r.rmse_forecast:7.3f}  total={r.rmse_total:7.3f}")

# %% [markdown]
# ### Weak-4DVar — shorter assim window for solver stability
#
# Weak-4DVar's enlarged $(x_0, \eta_t)$ control space makes BFGS
# diverge on the default 0.5-time-unit window. We use a shorter
# 0.1-time-unit assim for weak only, free-forecasting the same 10-
# time-unit total. Numbers go in the table alongside the rest.

# %%
prob_weak = generate_problem(key=jax.random.PRNGKey(42), T_assim=10, obs_every=2)
batch_weak = assim_batch(prob_weak)
weak = vdx.WeakFourDVar(
    forward=fwd, obs_op=vdx.LinearObs(H_mat=H_state),
    prior_mean=prob_weak.prior_mean_state, prior_cov_op=prob_weak.B_op_state,
    obs_cov_op=prob_weak.R_op_state, model_err_cov_op=prob_weak.B_op_state,
    max_steps=1000,
)


def weak_run():
    x0_b, etas_b = weak(batch_weak)
    return assemble_full_trajectory(x0_b[0], prob_weak, fwd, etas=etas_b[0])


results.append(run_method("weak_4dvar", weak_run, prob_weak))
print(f"{results[-1].name:20s} assim={results[-1].rmse_assim:7.3f}  "
      f"forecast={results[-1].rmse_forecast:7.3f}  total={results[-1].rmse_total:7.3f}")

# %% [markdown]
# ## 3. Learned methods (with training)

# %%
def make_pair(k):
    p = generate_problem(key=k)
    return p.obs, p.mask, p.truth[: prob.T_assim_plus_1]


# FourDVarNet
key = jax.random.PRNGKey(0)
fvn = vdx.FourDVarNet1D(
    state_dim=3, n_time=prob.T_assim_plus_1, latent_dim=8, hidden_dim=16,
    n_solver_steps=5, key=key,
)
fvn_keys = jax.random.split(jax.random.PRNGKey(1), 32)
obs_train, mask_train, truth_train = jax.vmap(make_pair)(fvn_keys)
fvn_batch = vdx.Batch1D(input=obs_train, mask=mask_train, target=truth_train)
fvn_opt = optax.adam(1e-2)
fvn_opt_state = fvn_opt.init(eqx.filter(fvn, eqx.is_array))
t0 = time.perf_counter()
for _ in range(200):
    fvn, fvn_opt_state, _ = vdx.train_step(fvn, fvn_batch, fvn_opt, fvn_opt_state)
fvn_train_time = time.perf_counter() - t0

results.append(run_method(
    "fourdvarnet",
    lambda: assemble_full_trajectory(fvn(batch)[0], prob, fwd),
    prob, train_time_s=fvn_train_time,
))
print(f"FourDVarNet train: {fvn_train_time:.1f}s, "
      f"rmse_forecast={results[-1].rmse_forecast:.3f}")

# Amortized
k_enc, k_head = jax.random.split(jax.random.PRNGKey(0))
amort = vdx.AmortizedPosterior(
    encoder=vdx.MLPObsEncoder(
        input_size=prob.T_assim_plus_1 * 3, context_dim=32,
        hidden_dim=64, depth=2, key=k_enc,
    ),
    head=vdx.RegressionHead(
        context_dim=32, state_shape=(prob.T_assim_plus_1, 3),
        hidden_dim=64, depth=2, key=k_head,
    ),
    config=vdx.AmortizedConfig(head_type="regression"),
)
am_keys = jax.random.split(jax.random.PRNGKey(2), 128)
obs_train, mask_train, truth_train = jax.vmap(make_pair)(am_keys)
am_batch = vdx.Batch1D(input=obs_train, mask=mask_train, target=truth_train)
am_opt = optax.adam(3e-3)
am_opt_state = am_opt.init(eqx.filter(amort, eqx.is_array))
t0 = time.perf_counter()
for _ in range(500):
    amort, am_opt_state, _ = vdx.amortized_train_step(
        amort, am_batch, am_opt, am_opt_state,
    )
am_train_time = time.perf_counter() - t0

results.append(run_method(
    "amortized",
    lambda: assemble_full_trajectory(amort(batch)[0], prob, fwd),
    prob, train_time_s=am_train_time,
))
print(f"Amortized train: {am_train_time:.1f}s, "
      f"rmse_forecast={results[-1].rmse_forecast:.3f}")

# %% [markdown]
# ## 4. Comparison table

# %%
table = compare(*results).sort_values("rmse_forecast")
table

# %% [markdown]
# ## 5. Trajectories overlaid (3 components, 7 methods, full 10-time-unit window)

# %%
fig, axs = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True)
colors = ["C0", "C1", "C2", "C4", "C3", "C5", "C6"]  # OI 3DV Strong Inc Weak FVN Amort
t_obs = t_axis[: prob.T_assim_plus_1][prob.mask[:, 0] > 0.5]
for i, ax in enumerate(axs):
    ax.axvspan(0.0, prob.T_assim * prob.dt, color="yellow", alpha=0.2,
               label="assim window")
    ax.plot(t_axis, prob.truth[:, i], "k-", lw=2.2, label="truth", zorder=10)
    obs_v = prob.obs[prob.mask[:, i] > 0.5, i]
    if len(obs_v) > 0:
        ax.plot(t_obs, obs_v, "kx", ms=7, label="obs", zorder=11)
    for r, c in zip(results, colors, strict=False):
        ax.plot(t_axis, r.mean[:, i], "--", color=c, lw=1.0, alpha=0.85,
                label=f"{r.name}")
    ax.set_ylabel("xyz"[i])
    if i == 0:
        ax.legend(loc="upper right", fontsize=8, ncol=3)
axs[-1].set_xlabel("time")
fig.suptitle("L63 — seven methods, analysis-then-forecast over 10 time units")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 6. RMSE(t) — the dynamics view
#
# This is the headline plot. The y-axis is log-scale instantaneous
# spatial RMSE; the x-axis spans the whole 10-time-unit window. The
# methods with good $x_0$ recovery (Strong-4DVar, FourDVarNet,
# Amortized) stay 1-3 orders of magnitude below the chaotic-error
# saturation level for many Lyapunov times.

# %%
fig, ax = plt.subplots(figsize=(11, 5))
ax.axvspan(0.0, prob.T_assim * prob.dt, color="yellow", alpha=0.2,
           label="assim window")
for r, c in zip(results, colors, strict=False):
    ax.plot(t_axis, r.rmse_trace, color=c, lw=1.5, label=r.name)
ax.set_xlabel("time")
ax.set_ylabel("instantaneous RMSE (log)")
ax.set_yscale("log")
ax.set_title("L63 — analysis-then-forecast RMSE trace, all methods")
ax.grid(True, alpha=0.3, which="both")
ax.legend(loc="best", fontsize=9, ncol=2)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Forecast-skill horizon view
#
# Bar chart of `rmse_forecast` (the average error over the 9.5-time-
# unit free-forecast window). Lower is better.

# %%
fig, ax = plt.subplots(figsize=(9, 4))
sorted_results = sorted(results, key=lambda r: r.rmse_forecast)
ax.barh([r.name for r in sorted_results],
        [r.rmse_forecast for r in sorted_results],
        color=[colors[[s.name for s in results].index(r.name)] for r in sorted_results])
ax.set_xlabel("RMSE over 9.5-time-unit free forecast")
ax.set_xscale("log")
ax.set_title("Forecast skill — sorted, lower is better")
ax.grid(True, alpha=0.3, axis="x", which="both")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Headline numbers (one run, `PRNGKey(42)`)
#
# - **OI / 3DVar** match each other in the linear-Gaussian limit
#   (Decision D14). Their analysis at unobserved time slices is the
#   prior (zero), so rmse_assim is dominated by those slices; their
#   *forecast* from the last observed state (close to truth) is
#   actually OK over a few Lyapunov times.
# - **Strong-4DVar** is the textbook win: compress the assim window
#   to 3 unknowns, recover $x_0$ to ~0.05 RMSE, and the perfect-
#   model forecast tracks the truth for the entire 9.5-time-unit
#   window.
# - **Incremental-4DVar** matches strong-4DVar's forecast skill at
#   the operational fast-path cost.
# - **Weak-4DVar** runs on a shorter assim window for solver
#   stability; its forecast quality is intermediate.
# - **FourDVarNet** and **AmortizedPosterior** come within a small
#   factor of strong-4DVar after a few seconds of simulation-based
#   training. Their advantage shows up in the inference time
#   column — sub-millisecond MAPs for the amortized head.
#
# The single plot to take away is the RMSE(t) trace in §6: the
# 3-order-of-magnitude separation between the "dynamics-aware" and
# "dynamics-blind" methods through the free-forecast window is the
# entire reason 4DVar is the operational standard for NWP.
