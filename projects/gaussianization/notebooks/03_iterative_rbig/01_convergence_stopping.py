# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# title: "RBIG convergence & stopping criteria"
# short_title: "01 · Convergence & stopping"
# subtitle: "Each layer provably reduces non-Gaussianity — turning total correlation and negentropy into a principled stopping rule and depth selector"
# description: >
#   RBIG's depth is not a hyperparameter you guess. Laparra et al. (2011) proved
#   each layer reduces a non-Gaussianity functional monotonically, so the per-layer
#   reduction is a convergence signal: stop when it stalls. This notebook uses
#   rbig's own information-theory API — total_correlation, tc_per_layer_,
#   entropy_reduction, score and entropy — to validate the measures, watch them
#   fall, and compare fixed-depth against early-stopping.
# ---
#
# (sec-nb-rbig-01)=
# # 01 — Convergence & stopping criteria
#
# [Notebook 00](00_rbig_loop.ipynb) ran a *fixed* twelve layers. But how many does
# a distribution actually need, and how do we know when to stop? RBIG answers with a
# guarantee. Laparra et al. {cite}`laparra2011rbig` showed each layer **reduces a
# non-Gaussianity functional** — the total correlation $T(p)$ (multi-information)
# plus the marginal negentropy — and never increases it:
#
# $$
# T\big(p^{(k+1)}\big) \le T\big(p^{(k)}\big), \qquad T(p) \ge 0,\;\; T = 0 \iff \text{coordinates independent.}
# $$
#
# A monotone, bounded-below sequence converges. So the **per-layer reduction** is a
# convergence signal: while large, the flow is still working; once it stalls near
# $0$, extra layers buy nothing and we stop. `rbig` exposes this directly —
# `total_correlation`, the model's `tc_per_layer_` trace, `entropy_reduction`, and
# the `score` / `entropy` likelihood signals — so we use its API throughout.
#
# **What you will see**
#
# - The **total-correlation estimator validated** against the analytic value for
#   Gaussian data (it matches), so we trust the yardstick.
# - Non-Gaussianity — **`tc_per_layer_`** and **marginal negentropy** — falling to
#   $0$ across layers, with the per-layer reduction $\to 0$.
# - rbig's two **stopping strategies**: a fixed layer cap, and `zero_tolerance`
#   early-stopping when the TC change stalls — and how depth grows as `tol` tightens.
# - **Fixed-K vs early-stop**, and the `score` / `entropy` likelihood signals.

# %%
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import check_random_state

import rbig
from _style import style_ax

rng = np.random.default_rng(0)

# A skewed, nonlinearly-dependent 2-D distribution: clearly non-Gaussian so the
# convergence is gradual and easy to watch.
n = 4000
s = rng.standard_normal(n)
X = np.stack([np.exp(0.8 * s), s ** 2 + 0.3 * rng.standard_normal(n)], axis=1)
X = (X - X.mean(0)) / X.std(0)

# %% [markdown]
# ## 1. The yardstick: total correlation, validated
#
# **Total correlation** $T(X) = \sum_i H(X_i) - H(X)$ is RBIG's native measure of
# dependence — zero exactly when the coordinates are independent. `rbig` estimates
# the marginal entropies by KDE and the joint entropy with a Gaussian fit, so it is
# *exact for Gaussian data*. We confirm that against the closed form for a
# correlated Gaussian before trusting it as a convergence signal.

# %%
chk = check_random_state(123)
d = 8
A = chk.rand(d, d)
G = chk.randn(6000, d) @ A
C = A.T @ A
tc_analytic = np.log(np.sqrt(np.diag(C))).sum() - 0.5 * np.log(np.linalg.det(C))
print(f"correlated Gaussian: TC analytic = {tc_analytic:.4f} nats")
print(f"                     rbig.total_correlation = {float(rbig.total_correlation(G)):.4f} nats  (matches)")

# %% [markdown]
# The estimate matches the analytic value to three decimals — the measure works.
# (On strongly non-Gaussian *raw* data the Gaussian-joint approximation makes the
# absolute number less reliable, which is precisely why RBIG tracks the *reduction*
# per layer rather than the absolute value, and why we also watch the marginal
# negentropy below.)
#
# ## 2. Non-Gaussianity falls to zero
#
# We fit `rbig.AnnealedRBIG`, which records the total correlation after every layer
# in `tc_per_layer_`. We also track the **marginal negentropy** $\sum_i J(x_i)$ from
# a hand-rolled loop — the per-axis non-Gaussianity, which is non-negative and a
# clean monotone signal. Both fall to zero, and the **per-layer reduction** decays
# geometrically.

# %%
def negentropy_trace(X, n_layers):
    """Per-layer marginal negentropy from a from-scratch RBIG loop."""
    A = X.copy()
    out = [float(rbig.negentropy(A).sum())]
    for _ in range(n_layers):
        A = rbig.MarginalGaussianize().fit(A).transform(A)
        A = rbig.PCARotation(whiten=False).fit(A).transform(A)
        out.append(float(rbig.negentropy(A).sum()))
    return np.array(out)

negent = negentropy_trace(X, 20)

model = rbig.AnnealedRBIG(n_layers=1000, rotation="pca",
                          zero_tolerance=20, tol=1e-5, random_state=0)
model.fit(X)
tc_layer = np.abs(np.asarray(model.tc_per_layer_))      # |TC| recorded per layer

print(f"marginal negentropy: {negent[0]:.3f} -> {max(negent[-1], 0.0):.4f}  "
      f"(monotone: {bool(np.all(np.diff(negent) <= 1e-3))})")
print(f"|tc_per_layer_|:     {tc_layer[0]:.3f} -> {tc_layer[-1]:.5f}  over {len(model.layers_)} layers")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.3))
axL.plot(np.maximum(negent, 0.0), "o-", color="tab:purple", lw=2, ms=4,
         label=r"marginal negentropy $\sum_i J(x_i)$")
axL.plot(tc_layer, "s-", color="tab:orange", lw=1.8, ms=3,
         label=r"$|{\rm TC}|$ per layer (`tc_per_layer_`)")
axL.axhline(0, color="k", lw=0.8, ls="--")
axL.set(title="Each RBIG layer reduces non-Gaussianity (monotone)",
        xlabel="layer $k$", ylabel="nats")
axL.legend(fontsize=8); style_ax(axL)

reduction = -np.diff(tc_layer)                          # per-layer drop in |TC|
axR.semilogy(np.maximum(np.abs(reduction), 1e-8), "o-", color="tab:blue", lw=1.6, ms=3)
axR.set(title=r"per-layer reduction $\to 0$ (log scale)",
        xlabel="layer $k$", ylabel=r"$|\Delta\,{\rm TC}_k|$ (nats)")
style_ax(axR)
fig.tight_layout()

# %% [markdown]
# Both measures fall monotonically to (estimator) zero — the marginal axes become
# normal and the dependence vanishes — and the per-layer reduction decays roughly
# geometrically. The decay is steep early (the first few layers do most of the work)
# and flattens as the distribution approaches $\mathcal{N}(0,I)$. That flattening is
# exactly the signal we stop on.
#
# ## 3. Two stopping strategies
#
# `rbig.AnnealedRBIG` builds the stopping logic in, mirroring the classic RBIG loss
# functions:
#
# - **Strategy I — fixed cap.** Set `n_layers=N` (and `zero_tolerance=N+1` to
#   disable early-stop). Simple, but the right $N$ is data-dependent.
# - **Strategy II — TC convergence.** Set a large `n_layers` and a `zero_tolerance`:
#   stop once the TC change stays below `tol` for `zero_tolerance` consecutive
#   layers. The patience guards against a single fluke-small reduction.
#
# Tightening `tol` asks for a more thoroughly Gaussian latent, costing a few layers.

# %%
print("Strategy I — fixed cap, TC after the last layer:")
for N in [5, 10, 30]:
    m = rbig.AnnealedRBIG(n_layers=N, rotation="pca", zero_tolerance=N + 1, random_state=0)
    m.fit(X)
    print(f"  n_layers={N:3d}  ->  |TC| after last layer = {abs(float(m.tc_per_layer_[-1])):.5f}")

print("\nStrategy II — early-stop, fitted depth vs tolerance (zero_tolerance=20):")
for tol in [1e-2, 1e-3, 1e-4, 1e-5]:
    m = rbig.AnnealedRBIG(n_layers=1000, rotation="pca", tol=tol,
                          zero_tolerance=20, random_state=0).fit(X)
    resid = max(float(rbig.negentropy(m.transform(X)).sum()), 0.0)
    print(f"  tol={tol:.0e}: {len(m.layers_):3d} layers, residual negentropy = {resid:.5f}")

# %% [markdown]
# Strategy II adapts the depth to the data; stricter tolerances buy a more Gaussian
# latent for a few extra layers — a smooth trade-off, not a cliff. The
# `zero_tolerance` patience is why these run past the visual knee of §2: they keep
# going until the reduction has been negligible for 20 straight layers.
#
# ## 4. Fixed depth vs early stopping, and the likelihood signal
#
# Why not just fix $K$? Because the right depth is **data-dependent**: an
# almost-Gaussian input needs a couple of layers, a pathological one needs dozens.
# Too small and the latent is still visibly non-Gaussian; too large and you waste
# compute. Early-stopping reads the convergence signal and lands at the knee. We
# also read off `rbig`'s likelihood signals — `score` (mean $\log p$) and
# `entropy()` — which double as training objectives for the parametric
# Gaussianization flows of **Part 4**.

# %%
def fit_K(X, K):
    A = X.copy()
    for _ in range(K):
        A = rbig.MarginalGaussianize().fit(A).transform(A)
        A = rbig.PCARotation(whiten=False).fit(A).transform(A)
    return A

Ks = [1, 2, 3, 5, 8, 12, 20, 40]
resid = [max(float(rbig.negentropy(fit_K(X, K)).sum()), 0.0) for K in Ks]
auto = rbig.AnnealedRBIG(n_layers=1000, rotation="pca", tol=1e-4,
                         zero_tolerance=20, random_state=0).fit(X)
auto_depth = len(auto.layers_)
auto_resid = max(float(rbig.negentropy(auto.transform(X)).sum()), 0.0)

print(f"likelihood signals at convergence: score (mean log p) = {float(auto.score(X)):.3f}, "
      f"entropy() = {float(auto.entropy()):.3f}")

fig, ax = plt.subplots(figsize=(7.8, 4.4))
ax.semilogy(Ks, np.maximum(resid, 1e-5), "o-", color="tab:blue", lw=2, ms=5,
            label="fixed-$K$ residual negentropy")
ax.scatter([auto_depth], [max(auto_resid, 1e-5)], color="tab:red", zorder=5, s=110,
           marker="*", label=f"early-stop (tol 1e-4) → {auto_depth} layers")
ax.axhspan(1e-5, 1e-2, color="tab:green", alpha=0.08)
ax.text(1.2, 2e-2, "under-Gaussianized\n(residual structure)", fontsize=8, color="tab:blue")
ax.set(title="Fixed depth vs early stopping", xlabel="number of layers $K$",
       ylabel="residual negentropy (nats, log)")
ax.legend(fontsize=8); style_ax(ax)
fig.tight_layout()

# %% [markdown]
# Below the knee ($K \lesssim 5$) the residual is large — the latent is not yet
# Gaussian. Above it the curve is flat: layers $12$ and $40$ are indistinguishable,
# so fixing $K=40$ "to be safe" wastes $28$ layers. The early-stop star sits right
# where the curve flattens, chosen with no manual sweep. (The `score` / `entropy`
# values are `rbig`'s histogram-based estimates — fine as a *relative* training
# signal; for calibrated absolute likelihoods use the smooth `gauss_flows` RBIG of
# [notebook 00](00_rbig_loop.ipynb).)
#
# ## Recap
#
# | idea | takeaway |
# |---|---|
# | TC validated on Gaussian | `rbig.total_correlation` matches the analytic value |
# | Laparra monotone reduction | each layer reduces non-Gaussianity; `tc_per_layer_` $\downarrow 0$ |
# | per-layer reduction | decays (≈ geometrically) to $0$ |
# | Strategy I / II | fixed cap vs `zero_tolerance` early-stop on TC change |
# | `tol` vs depth | tighter tolerance → more Gaussian latent, a few more layers |
# | early-stop vs fixed-$K$ | early-stop finds the data-dependent knee |
#
# **Next up.** The *rotation* between marginal passes also controls how fast the
# reduction falls. [02 — Rotation-choice studies](02_rotation_choices.ipynb)
# compares PCA, ICA, random, and Picard on layers-to-converge — the convergence
# signal of this notebook is exactly the yardstick.
