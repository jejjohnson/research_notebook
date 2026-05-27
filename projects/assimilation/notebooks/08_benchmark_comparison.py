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
# # Benchmark — all seven methods, head to head
#
# Every per-method notebook computes RMSE and runtime against the
# **same** `generate_problem(key=jax.random.PRNGKey(42))` problem.
# This notebook re-runs each method, stacks the results into one
# table, and overlays the seven analyses on a single plot.

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
    compare,
    generate_problem,
    run_method,
)


# %% [markdown]
# ## 1. Shared problem

# %%
prob = generate_problem(key=jax.random.PRNGKey(42))
batch = vdx.Batch1D(input=prob.obs[None], mask=prob.mask[None], target=None)
fwd = Lorenz63Forward(dt=prob.dt)
H_full = lx.IdentityLinearOperator(jax.ShapeDtypeStruct(prob.prior_mean.shape, jnp.float32))
H_state = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((3,), jnp.float32))
print(f"truth: {prob.truth.shape},  obs: {int(prob.mask.sum())} / {prob.mask.size} entries")

# %% [markdown]
# ## 2. Classical methods (no training)

# %%
oi = vdx.OptimalInterpolation(
    obs_op=vdx.LinearObs(H_mat=H_full),
    prior_mean=prob.prior_mean,
    prior_cov_op=prob.B_op,
    obs_cov_op=prob.R_op,
)
three = vdx.ThreeDVar(
    obs_op=vdx.LinearObs(H_mat=H_full),
    prior_mean=prob.prior_mean,
    prior_cov_op=prob.B_op,
    obs_cov_op=prob.R_op,
)
strong = vdx.StrongFourDVar(
    forward=fwd,
    obs_op=vdx.LinearObs(H_mat=H_state),
    prior_mean=prob.prior_mean_state,
    prior_cov_op=prob.B_op_state,
    obs_cov_op=prob.R_op_state,
)
weak = vdx.WeakFourDVar(
    forward=fwd,
    obs_op=vdx.LinearObs(H_mat=H_state),
    prior_mean=prob.prior_mean_state,
    prior_cov_op=prob.B_op_state,
    obs_cov_op=prob.R_op_state,
    model_err_cov_op=prob.B_op_state,
)
inc = vdx.IncrementalFourDVar(
    forward=fwd,
    obs_op=vdx.LinearObs(H_mat=H_state),
    prior_mean=prob.prior_mean_state,
    prior_cov_op=prob.B_op_state,
    obs_cov_op=prob.R_op_state,
    config=vdx.IncrementalConfig(n_outer=4, n_inner=20),
)


def rollout(x0):
    def step(s, _):
        new = fwd.step(s, fwd.dt)
        return new, new
    _, traj = jax.lax.scan(step, x0, None, length=prob.T)
    return jnp.concatenate([x0[None, :], traj], axis=0)


def weak_run():
    x0_b, etas_b = weak(batch)
    x0, etas = x0_b[0], etas_b[0]
    def step(s, eta):
        new = fwd.step(s, fwd.dt) + eta
        return new, new
    _, traj = jax.lax.scan(step, x0, etas)
    return jnp.concatenate([x0[None, :], traj], axis=0)


results = [
    run_method("oi", lambda: oi(batch)[0], prob),
    run_method("3dvar", lambda: three(batch)[0], prob),
    run_method("strong_4dvar", lambda: rollout(strong(batch)[0]), prob),
    run_method("weak_4dvar", weak_run, prob),
    run_method("incremental_4dvar", lambda: rollout(inc(batch)[0]), prob),
]

# %% [markdown]
# ## 3. Learned methods (with timing the training too)

# %%
def make_pair(k):
    p = generate_problem(key=k)
    return p.obs, p.mask, p.truth


# FourDVarNet
key = jax.random.PRNGKey(0)
fvn = vdx.FourDVarNet1D(
    state_dim=3, n_time=prob.T_plus_1, latent_dim=8, hidden_dim=16,
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

results.append(
    run_method("fourdvarnet", lambda: fvn(batch)[0], prob, train_time_s=fvn_train_time)
)

# Amortized
k_enc, k_head = jax.random.split(jax.random.PRNGKey(0))
amort = vdx.AmortizedPosterior(
    encoder=vdx.MLPObsEncoder(input_size=prob.T_plus_1 * 3, context_dim=32,
                              hidden_dim=64, depth=2, key=k_enc),
    head=vdx.RegressionHead(context_dim=32, state_shape=(prob.T_plus_1, 3),
                            hidden_dim=64, depth=2, key=k_head),
    config=vdx.AmortizedConfig(head_type="regression"),
)
am_keys = jax.random.split(jax.random.PRNGKey(2), 128)
obs_train, mask_train, truth_train = jax.vmap(make_pair)(am_keys)
am_batch = vdx.Batch1D(input=obs_train, mask=mask_train, target=truth_train)
am_opt = optax.adam(3e-3)
am_opt_state = am_opt.init(eqx.filter(amort, eqx.is_array))

t0 = time.perf_counter()
for _ in range(500):
    amort, am_opt_state, _ = vdx.amortized_train_step(amort, am_batch, am_opt, am_opt_state)
am_train_time = time.perf_counter() - t0

results.append(
    run_method("amortized", lambda: amort(batch)[0], prob, train_time_s=am_train_time)
)

# %% [markdown]
# ## 4. Comparison table

# %%
table = compare(*results).sort_values("rmse_total")
table

# %% [markdown]
# ## 5. Trajectories overlaid

# %%
fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
t = jnp.arange(prob.T_plus_1) * prob.dt
colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]

for i, ax in enumerate(axs):
    ax.plot(t, prob.truth[:, i], "k-", lw=2.5, label="truth", zorder=10)
    obs_t = t[prob.mask[:, i] > 0.5]
    if len(obs_t) > 0:
        ax.plot(obs_t, prob.obs[prob.mask[:, i] > 0.5, i], "kx", ms=8, label="obs", zorder=11)
    for r, c in zip(results, colors, strict=False):
        ax.plot(t, r.mean[:, i], "--", color=c, lw=1.2, alpha=0.85,
                label=f"{r.name} (rmse={r.rmse_total:.2f})")
    ax.set_ylabel("xyz"[i])
    if i == 0:
        ax.legend(loc="upper right", fontsize=8, ncol=2)
axs[-1].set_xlabel("time")
fig.suptitle("Seven assimilation methods on the same Lorenz-63 partial-obs problem")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Runtime vs accuracy

# %%
fig, ax = plt.subplots(figsize=(8, 5))
for r, c in zip(results, colors, strict=False):
    ax.scatter(r.runtime_ms, r.rmse_total, s=120, color=c, edgecolors="black", zorder=5)
    ax.annotate(r.name, (r.runtime_ms, r.rmse_total),
                xytext=(8, 5), textcoords="offset points", fontsize=9)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("inference time (ms, log)")
ax.set_ylabel("RMSE (log)")
ax.set_title("Accuracy / latency trade-off")
ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Headline numbers
#
# - **OI = 3DVar** ≈ 16.9 → linear-Gaussian invariant verified
#   (Decision D14).
# - **Strong-4DVar / Incremental-4DVar** drop RMSE by 7× thanks to
#   the dynamics constraint.
# - **FourDVarNet** edges past strong-4DVar with ~3 seconds of
#   training.
# - **AmortizedPosterior** wins the MAP race outright (RMSE ~0.7) and
#   wins by 100× on inference latency — but the predictive variances
#   are mis-calibrated, see the per-notebook NLL discussion in
#   [`07_amortized_posterior`](07_amortized_posterior.ipynb).
#
# Open follow-ups (separate notebooks, not in this PR):
#
# - **`VarDACycle` cycling demo** — re-run OI vs incremental-4DVar
#   over 10 consecutive 40-step windows to show how the analysis
#   improves over cycle index.
# - **Six-step gate sweep** — wire
#   `vardax.assert_posterior_agreement` and
#   `vardax.simulation_based_calibration` into a CI check on the
#   amortized head.
