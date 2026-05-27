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
# # Two-level L96 benchmark — six methods, head to head
#
# Same script as the L63 and L96-1L comparison notebooks, restricted
# to six of the seven AnalysisStep methods (Incremental-4DVar
# diverges on the stiff slow-fast coupling — see the note at the end
# of Section 2 — and is omitted here).
#
# The benchmark runs on the multi-scale 72-D problem set up in
# [`11_lorenz96_2l_setup`](11_lorenz96_2l_setup.ipynb). The notebook
# tracks **two** RMSE columns — one for the slow block and one for
# the fast block — because the slow-only observation regime means
# the two scales are constrained very differently.
#
# What we're watching for:
#
# - **OI / 3DVar** match the prior floor on slow (~$6.5$) and
#   fast (~$0.3$) — the diagonal $B$ has no way to propagate
#   information.
# - **Strong-4DVar** drops slow RMSE dramatically (the dynamics
#   constraint propagates slow obs into unobserved slow grid
#   points), but **degrades** the (unobserved) fast block above its
#   prior floor — the classic "imbalance" failure.
# - **AmortizedPosterior** learns the slow-fast joint structure from
#   simulated pairs; it's the only method that preserves the fast
#   block at the prior floor while still improving slow recovery.

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
    compare,
    generate_l96_2l_problem,
    rmse,
    run_method,
)


# %% [markdown]
# ## 1. Shared problem

# %%
prob = generate_l96_2l_problem(key=jax.random.PRNGKey(0))
batch = vdx.Batch1D(input=prob.obs[None], mask=prob.mask[None], target=None)
fwd = Lorenz96TwoLevelForward(K=prob.K, J=prob.J, F=prob.F, h=prob.h,
                              c=prob.c, b=prob.b, dt=prob.dt)
H_full = lx.IdentityLinearOperator(jax.ShapeDtypeStruct(prob.prior_mean.shape, jnp.float32))
H_state = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((prob.D,), jnp.float32))
print(f"K={prob.K}, J={prob.J}, D={prob.D}, T+1={prob.T_plus_1}")
print(f"obs: {int(prob.mask.sum())} scalars on slow block only")

# %% [markdown]
# ## 2. Classical methods

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
    max_steps=500,
)
strong = vdx.StrongFourDVar(
    forward=fwd,
    obs_op=vdx.LinearObs(H_mat=H_state),
    prior_mean=prob.prior_mean_state,
    prior_cov_op=prob.B_op_state,
    obs_cov_op=prob.R_op_state,
    max_steps=500,
)
weak = vdx.WeakFourDVar(
    forward=fwd,
    obs_op=vdx.LinearObs(H_mat=H_state),
    prior_mean=prob.prior_mean_state,
    prior_cov_op=prob.B_op_state,
    obs_cov_op=prob.R_op_state,
    model_err_cov_op=prob.B_op_state,
    max_steps=500,
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
]
for r in results:
    print(f"{r.name:20s} rmse_total={r.rmse_total:6.3f}  runtime={r.runtime_ms:8.1f} ms")

# %% [markdown]
# **`IncrementalFourDVar` is omitted from this benchmark.** Across a
# sweep of `(n_outer, n_inner, cg_tol)` configurations the
# Gauss-Newton linearisation of the stiff slow-fast coupling drives
# the Hessian near-singular and the analysis returns NaN. This is a
# genuine limitation of the incremental approximation on multi-scale
# systems — operationally it would be addressed by a structured
# control-variable transform (Decision D11 follow-up, not yet
# shipped) or by linearising the coupling separately. On the L96
# single-level benchmark
# ([`10_lorenz96_benchmark`](10_lorenz96_benchmark.ipynb))
# incremental converges happily; the multi-scale jump is what
# breaks it.

# %% [markdown]
# ## 3. Learned methods

# %%
def make_pair(k):
    p = generate_l96_2l_problem(key=k)
    return p.obs, p.mask, p.truth


# %% [markdown]
# ### FourDVarNet
# Bigger network than L96-1L because $D = 72$ trajectory entries per
# time slice + the slow-fast joint structure means the learned
# modulator needs more capacity.

# %%
key = jax.random.PRNGKey(0)
fvn = vdx.FourDVarNet1D(
    state_dim=prob.D,
    n_time=prob.T_plus_1,
    latent_dim=48,
    hidden_dim=96,
    n_solver_steps=8,
    key=key,
)
fvn_keys = jax.random.split(jax.random.PRNGKey(1), 48)
obs_train, mask_train, truth_train = jax.vmap(make_pair)(fvn_keys)
fvn_batch = vdx.Batch1D(input=obs_train, mask=mask_train, target=truth_train)
fvn_opt = optax.adam(3e-3)
fvn_opt_state = fvn_opt.init(eqx.filter(fvn, eqx.is_array))

t0 = time.perf_counter()
for _ in range(500):
    fvn, fvn_opt_state, fvn_loss = vdx.train_step(
        fvn, fvn_batch, fvn_opt, fvn_opt_state
    )
fvn_train_time = time.perf_counter() - t0
print(f"FourDVarNet train: {fvn_train_time:.1f}s, final MSE: {float(fvn_loss):.3f}")

results.append(
    run_method("fourdvarnet", lambda: fvn(batch)[0], prob,
               train_time_s=fvn_train_time)
)
print(f"FourDVarNet rmse: {results[-1].rmse_total:.3f}")

# %% [markdown]
# ### AmortizedPosterior
# Output shape $(T+1, D) = (41, 72) = 2952$. Wider head: 128-dim
# context, 384-wide MLPs.

# %%
k_enc, k_head = jax.random.split(jax.random.PRNGKey(0))
amort = vdx.AmortizedPosterior(
    encoder=vdx.MLPObsEncoder(
        input_size=prob.T_plus_1 * prob.D, context_dim=128,
        hidden_dim=384, depth=2, key=k_enc,
    ),
    head=vdx.RegressionHead(
        context_dim=128, state_shape=(prob.T_plus_1, prob.D),
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
    amort, am_opt_state, am_loss = vdx.amortized_train_step(
        amort, am_batch, am_opt, am_opt_state
    )
am_train_time = time.perf_counter() - t0
print(f"Amortized train: {am_train_time:.1f}s, final NLL: {float(am_loss):.1f}")

results.append(
    run_method("amortized", lambda: amort(batch)[0], prob,
               train_time_s=am_train_time)
)
print(f"Amortized rmse: {results[-1].rmse_total:.3f}")

# %% [markdown]
# ## 4. Per-block RMSE — slow vs fast
#
# Total RMSE alone is misleading: ~88% of state entries are fast and
# have small magnitude, so a method that crushes the slow block can
# still look mediocre overall (and vice-versa). The slow/fast
# decomposition is the right view.

# %%
def block_rmses(mean):
    return (
        float(rmse(mean[:, :prob.K], prob.truth[:, :prob.K])),
        float(rmse(mean[:, prob.K:], prob.truth[:, prob.K:])),
    )


prior_slow = float(rmse(jnp.zeros_like(prob.truth[:, :prob.K]), prob.truth[:, :prob.K]))
prior_fast = float(rmse(jnp.zeros_like(prob.truth[:, prob.K:]), prob.truth[:, prob.K:]))
print(f"PRIOR FLOORS: slow = {prior_slow:.3f}, fast = {prior_fast:.3f}")
print("-" * 60)
for r in results:
    s, f = block_rmses(r.mean)
    print(f"  {r.name:20s} slow rmse={s:6.3f}  fast rmse={f:6.3f}  "
          f"total rmse={r.rmse_total:6.3f}")

# %% [markdown]
# ## 5. Comparison table (auto-summarised over all 72 components)

# %%
table = compare(*results).sort_values("rmse_total")
table

# %% [markdown]
# ## 6. Hovmöller plots — slow block only

# %%
n = len(results)
fig, axs = plt.subplots(2 + n, 1, figsize=(11, 1.5 * (2 + n)), sharex=True)
vmax = float(jnp.max(jnp.abs(prob.truth[:, :prob.K])))
kwargs = {"aspect": "auto", "cmap": "RdBu_r", "origin": "lower",
          "vmin": -vmax, "vmax": vmax}

axs[0].imshow(prob.truth[:, :prob.K].T, **kwargs)
axs[0].set_ylabel("truth")
axs[1].imshow(prob.obs[:, :prob.K].T, **kwargs)
axs[1].set_ylabel("obs")
for ax, r in zip(axs[2:], results, strict=False):
    ax.imshow(r.mean[:, :prob.K].T, **kwargs)
    s, f = block_rmses(r.mean)
    ax.set_ylabel(f"{r.name}\nslow={s:.2f}\nfast={f:.2f}")
axs[-1].set_xlabel("time step")
fig.suptitle("Slow-block Hovmöller — truth + obs + seven analyses")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Discussion
#
# Reading of the table (your numbers will vary slightly with the
# random seed):
#
# - **OI / 3DVar.** Both match the slow prior floor (~6.5) and leave
#   fast at the fast prior floor (~0.3). Identical by Decision D14.
# - **Strong-4DVar.** Cuts slow RMSE roughly 3× via the dynamics
#   constraint. Best slow recovery in the benchmark, but **fast RMSE
#   doubles above the prior floor** — the classic "imbalance"
#   failure mode where slow-fitting drives fast values that satisfy
#   slow residuals but diverge from truth.
# - **Weak-4DVar.** Surprisingly poor on this problem: the model-
#   error augmentation enlarges the control space substantially and
#   the BFGS solver struggles to converge in the default
#   `max_steps`. The per-grid-point `rmse_max` column shows the
#   blow-up. Increasing `max_steps` and tightening
#   `model_err_cov_op` would help.
# - **FourDVarNet.** Learned solver, trained on 48 simulated
#   trajectories. Under-trained at $D = 72$ — the modulator hasn't
#   seen enough joint slow-fast structure yet. Longer training
#   closes the gap.
# - **AmortizedPosterior.** No inner solve, sub-millisecond
#   inference. The regression head learns the slow-fast statistical
#   structure from 128 simulated pairs; this is the **only method
#   that preserves the fast block at the prior floor** while still
#   improving slow recovery — exactly the behaviour the imbalance
#   failure makes impossible for iterative methods.
#
# Take-home: **slow obs + dynamics is not enough to constrain fast
# state in a well-balanced way**. Strong-4DVar wins on slow recovery
# but pays for it on fast; only the learned method internalises
# enough joint structure to update slow without disturbing fast.
#
# Follow-ups:
#
# - **Larger $J$.** Canonical Wilks 2005 uses $J = 32$. With
#   $K = 8, J = 32$ the state dim is 264 — beyond what fits in a fast
#   notebook but a natural scaling experiment.
# - **Direct fast observations.** Add a few noisy direct observations
#   of $Y_{j,k}$; this is the operational "sub-grid in-situ" regime
#   that lets DA constrain the fast scale.
# - **Coupled prior**. Replace the diagonal $B$ with a structured
#   covariance that encodes the slow-fast coupling (a `gaussx`
#   block-Matern would be the obvious choice).
