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
# # Two-level L96 benchmark — six methods, analysis-then-forecast
#
# `IncrementalFourDVar` is omitted on this multi-scale problem (the
# Gauss-Newton linearisation of the stiff slow-fast coupling drives
# the Hessian near-singular and the analysis returns NaN; documented
# failure mode of the incremental approximation here).
#
# The story we expect:
#
# - OI / 3DVar give noisy per-time-step analyses; the slow free-
#   forecast launched from the last analysed state diverges within
#   one Lyapunov time.
# - Strong / Weak-4DVar use the slow obs + dynamics to recover a
#   reasonable joint slow-fast $x_0$; the forecast then tracks the
#   slow attractor for several Lyapunov times.
# - Learned methods can outperform if their training distribution
#   matches the deployment regime.

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
    Lorenz96TwoLevelForward,
    assemble_full_trajectory,
    assim_batch,
    compare,
    generate_l96_2l_problem,
    rmse,
    run_method,
)


# %% [markdown]
# ## 1. Shared problem

# %%
prob = generate_l96_2l_problem(key=jax.random.PRNGKey(0))
batch = assim_batch(prob)
fwd = Lorenz96TwoLevelForward(K=prob.K, J=prob.J, F=prob.F, h=prob.h,
                              c=prob.c, b=prob.b, dt=prob.dt)
H_full = lx.IdentityLinearOperator(
    jax.ShapeDtypeStruct(prob.prior_mean.shape, jnp.float32)
)
H_state = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((prob.D,), jnp.float32))
t_axis = jnp.arange(prob.T_total_plus_1) * prob.dt
print(f"K={prob.K}, J={prob.J}, D={prob.D}")
print(f"T_assim={prob.T_assim}, T_total={prob.T_total}")

# %% [markdown]
# ## 2. Classical methods

# %%
oi = vdx.OptimalInterpolation(
    obs_op=vdx.LinearObs(H_mat=H_full),
    prior_mean=prob.prior_mean, prior_cov_op=prob.B_op, obs_cov_op=prob.R_op,
)
three = vdx.ThreeDVar(
    obs_op=vdx.LinearObs(H_mat=H_full),
    prior_mean=prob.prior_mean, prior_cov_op=prob.B_op, obs_cov_op=prob.R_op,
    max_steps=500,
)
strong = vdx.StrongFourDVar(
    forward=fwd, obs_op=vdx.LinearObs(H_mat=H_state),
    prior_mean=prob.prior_mean_state, prior_cov_op=prob.B_op_state,
    obs_cov_op=prob.R_op_state, max_steps=2000,
)
weak = vdx.WeakFourDVar(
    forward=fwd, obs_op=vdx.LinearObs(H_mat=H_state),
    prior_mean=prob.prior_mean_state, prior_cov_op=prob.B_op_state,
    obs_cov_op=prob.R_op_state, model_err_cov_op=prob.B_op_state,
    max_steps=1000,
)


def weak_run():
    x0_b, etas_b = weak(batch)
    return assemble_full_trajectory(x0_b[0], prob, fwd, etas=etas_b[0])


results = [
    run_method("oi",
               lambda: assemble_full_trajectory(oi(batch)[0], prob, fwd), prob),
    run_method("3dvar",
               lambda: assemble_full_trajectory(three(batch)[0], prob, fwd), prob),
    run_method("strong_4dvar",
               lambda: assemble_full_trajectory(strong(batch)[0], prob, fwd), prob),
    run_method("weak_4dvar", weak_run, prob),
]
for r in results:
    print(f"{r.name:20s} assim={r.rmse_assim:7.3f}  "
          f"forecast={r.rmse_forecast:7.3f}  total={r.rmse_total:7.3f}")

# %% [markdown]
# **`IncrementalFourDVar` is omitted.** On the multi-scale L96-2L
# problem its Gauss-Newton linearisation drives the Hessian near-
# singular and the analysis returns NaN across every
# `(n_outer, n_inner, cg_tol)` config tried. Documented limitation
# of the incremental approximation on stiff multi-scale systems.

# %% [markdown]
# ## 3. Learned methods

# %%
def make_pair(k):
    p = generate_l96_2l_problem(key=k)
    return p.obs, p.mask, p.truth[: prob.T_assim_plus_1]


# FourDVarNet
key = jax.random.PRNGKey(0)
fvn = vdx.FourDVarNet1D(
    state_dim=prob.D, n_time=prob.T_assim_plus_1,
    latent_dim=48, hidden_dim=96, n_solver_steps=8, key=key,
)
fvn_keys = jax.random.split(jax.random.PRNGKey(1), 48)
obs_train, mask_train, truth_train = jax.vmap(make_pair)(fvn_keys)
fvn_batch = vdx.Batch1D(input=obs_train, mask=mask_train, target=truth_train)
fvn_opt = optax.adam(3e-3)
fvn_opt_state = fvn_opt.init(eqx.filter(fvn, eqx.is_array))
t0 = time.perf_counter()
for _ in range(500):
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
        input_size=prob.T_assim_plus_1 * prob.D, context_dim=128,
        hidden_dim=384, depth=2, key=k_enc,
    ),
    head=vdx.RegressionHead(
        context_dim=128, state_shape=(prob.T_assim_plus_1, prob.D),
        hidden_dim=384, depth=2, key=k_head,
    ),
    config=vdx.AmortizedConfig(head_type="regression"),
)
am_keys = jax.random.split(jax.random.PRNGKey(2), 128)
obs_train, mask_train, truth_train = jax.vmap(make_pair)(am_keys)
am_batch = vdx.Batch1D(input=obs_train, mask=mask_train, target=truth_train)
am_opt = optax.adam(2e-3)
am_opt_state = am_opt.init(eqx.filter(amort, eqx.is_array))
t0 = time.perf_counter()
for _ in range(600):
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
# ## 4. Per-block RMSE — slow vs fast

# %%
def block_rmses(mean):
    return (
        float(rmse(mean[:, : prob.K], prob.truth[:, : prob.K])),
        float(rmse(mean[:, prob.K:], prob.truth[:, prob.K:])),
    )


prior_slow = float(rmse(jnp.zeros_like(prob.truth[:, : prob.K]),
                        prob.truth[:, : prob.K]))
prior_fast = float(rmse(jnp.zeros_like(prob.truth[:, prob.K:]),
                        prob.truth[:, prob.K:]))
print(f"PRIOR FLOORS: slow = {prior_slow:.3f}, fast = {prior_fast:.3f}")
print("-" * 70)
for r in results:
    s, f = block_rmses(r.mean)
    print(f"  {r.name:20s} slow rmse={s:6.3f}  fast rmse={f:6.3f}  "
          f"total rmse={r.rmse_total:6.3f}")

# %% [markdown]
# ## 5. Comparison table

# %%
table = compare(*results).sort_values("rmse_forecast")
table

# %% [markdown]
# ## 6. Slow-block Hovmöllers (truth + 6 methods)

# %%
n = len(results)
fig, axs = plt.subplots(1 + n, 1, figsize=(11, 1.4 * (1 + n)), sharex=True)
vmax = float(jnp.max(jnp.abs(prob.truth[:, : prob.K])))
kwargs = {"aspect": "auto", "cmap": "RdBu_r", "origin": "lower",
          "vmin": -vmax, "vmax": vmax,
          "extent": (0.0, prob.T_total * prob.dt, 0, prob.K)}
axs[0].imshow(prob.truth[:, : prob.K].T, **kwargs)
axs[0].axvline(prob.T_assim * prob.dt, color="yellow", lw=2)
axs[0].set_ylabel("truth")
for ax, r in zip(axs[1:], results, strict=False):
    ax.imshow(r.mean[:, : prob.K].T, **kwargs)
    ax.axvline(prob.T_assim * prob.dt, color="yellow", lw=2)
    s, f = block_rmses(r.mean)
    ax.set_ylabel(f"{r.name}\nslow={s:.2f}\nfast={f:.2f}")
axs[-1].set_xlabel("time")
fig.suptitle("L96-2L slow-block Hovmöllers — yellow line = assim / forecast")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 7. RMSE(t) — the dynamics view

# %%
colors = ["C0", "C1", "C2", "C3", "C5", "C6"]
fig, ax = plt.subplots(figsize=(11, 5))
ax.axvspan(0.0, prob.T_assim * prob.dt, color="yellow", alpha=0.2,
           label="assim window")
for r, c in zip(results, colors, strict=False):
    ax.plot(t_axis, r.rmse_trace, color=c, lw=1.5, label=r.name)
ax.set_xlabel("time")
ax.set_ylabel("instantaneous RMSE (log)")
ax.set_yscale("log")
ax.set_title("L96-2L — analysis-then-forecast RMSE trace, all methods")
ax.grid(True, alpha=0.3, which="both")
ax.legend(loc="best", fontsize=9, ncol=2)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Discussion
#
# The slow-only observation regime with strong slow-fast coupling
# is the canonical "sub-grid" test:
#
# - **OI / 3DVar** can't propagate slow obs into fast or unobserved
#   slow grid points; their per-time-step analyses are noisy, and
#   the free-forecast diverges within a Lyapunov time.
# - **Strong-4DVar** uses the slow obs + perfect-model dynamics to
#   recover a joint $(X, Y)$ initial condition; the forecast then
#   tracks the slow attractor for several Lyapunov times. The catch:
#   the *fast* block at the end of the assim window can sit above
#   its prior floor (the imbalance failure mode).
# - **Weak-4DVar** softens the imbalance trap at the cost of a
#   looser fit.
# - **AmortizedPosterior** trained on simulated $(X, Y)$ pairs
#   internalises the slow-fast joint structure, often recovering
#   both blocks without the imbalance penalty.
#
# The RMSE(t) plot in §7 makes the dynamics story explicit: the
# moment we leave the yellow assim region, OI's and 3DVar's RMSE
# climbs to the prior-floor level; the dynamics-aware methods'
# traces stay bounded for many Lyapunov times.
