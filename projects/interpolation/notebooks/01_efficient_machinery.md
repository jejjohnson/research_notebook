---
title: "Efficient machinery for SSH pathwise sampling — a tour of gaussx + pyrox"
---

# Efficient machinery for SSH pathwise sampling

The companion note [`00_ssh_pathwise_sampling.md`](00_ssh_pathwise_sampling.md) writes the algorithm out as if you were going to implement it from scratch — dense Gram matrices, dense Cholesky, dense cross-covariance, hand-rolled RFF prior. That description makes the math obvious but it carries three concrete costs that get unpleasant the moment you push to realistic Mediterranean grids:

| Bottleneck in the naive write-up | Why it hurts |
|---|---|
| Materialise $K_{\mathcal{X}^*\mathcal{X}} \in \mathbb{R}^{n \times m}$ | $\sim 4\,\text{GB}$ at $n=10^5$, $m=5\times 10^3$ — the dominant memory term |
| Cholesky of $C \in \mathbb{R}^{m \times m}$ | $\mathcal{O}(m^3)$ — single-threaded LAPACK, ~40 s at $m=5\text{k}$ |
| Per-sample $K_{\mathcal{X}^*\mathcal{X}}\,\beta$ matvec, repeated $S$ times | $\mathcal{O}(Snm)$, plus the same $4\,\text{GB}$ buffer reused $S$ times |

This note walks through the [gaussx](https://github.com/jejjohnson/gaussx) and [pyrox](https://github.com/jejjohnson/pyrox) primitives that fix each of those, with pseudocode close enough to real Python that you could turn it into runnable code in an afternoon. The point is not to refactor the SSH pipeline today — it is to leave a paper trail of *which existing primitive solves which step* so future-me (or future-anyone) can pick the right tool without re-deriving the algorithm.

Throughout, the four "objects" from the 00 derivation keep their names: $K_{\mathcal{X}\mathcal{X}}$, $C = K_{\mathcal{X}\mathcal{X}} + \sigma_{obs}^2 I$, $K_{\mathcal{X}^*\mathcal{X}}$, and the prior path $\tilde f$.

---

## Machinery 1 — matrix-free kernels (kills the $4\,\text{GB}$ peak)

The cross-covariance $K_{\mathcal{X}^*\mathcal{X}}$ is *only ever multiplied against vectors* in the algorithm: once for the posterior mean ($K_{\mathcal{X}^*\mathcal{X}}\,\alpha$), and once per sample for the correction ($K_{\mathcal{X}^*\mathcal{X}}\,\beta$). There is no reason to allocate the dense block.

### The primitive

[`gaussx.ImplicitCrossKernelOperator`](https://github.com/jejjohnson/gaussx/blob/main/src/gaussx/_operators/_implicit_cross_kernel.py) wraps any pure-function kernel `k(x_i, z_j) -> scalar` as a `lineax.AbstractLinearOperator` whose `.mv(v)` evaluates

$$
(K v)_i = \sum_{j=1}^{m} k(x_i, z_j)\, v_j
$$

via `jax.lax.scan` over rows of $\mathcal{X}^*$, in batches of size `batch_size`. Peak memory per matvec is $\mathcal{O}(\text{batch\_size} \cdot m)$ — set `batch_size` to fit in cache and you are done. It carries a `custom_jvp` so autodiff through hyperparameters stays cheap (vmap-vectorised JVP, transposes cleanly into a VJP).

The square sibling, [`gaussx.ImplicitKernelOperator`](https://github.com/jejjohnson/gaussx/blob/main/src/gaussx/_operators/_implicit_kernel.py), does the same trick for $K_{\mathcal{X}\mathcal{X}}$ and accepts an optional `noise_var=σ²` so the **noisy Gram** $C = K_{\mathcal{X}\mathcal{X}} + \sigma^2 I$ is one operator with no extra allocation.

### Pseudocode

```python
import gaussx as gx
import lineax as lx

def k_st(x, z, sigma2, Ls, Lt):
    """One scalar evaluation of σ²·k_s·k_t. Pure function — no batching."""
    d_space = great_circle(x[:2], z[:2])
    d_time  = jnp.abs(x[2] - z[2])
    spatial = matern32(d_space, Ls)
    temporal = jnp.exp(-d_time / Lt)
    return sigma2 * spatial * temporal

# Noisy Gram C = K_XX + σ²I — never materialised.
C_op = gx.ImplicitKernelOperator(
    kernel_fn = lambda xi, xj: k_st(xi, xj, sigma2, Ls, Lt),
    X         = X_train,                  # (m, 3)
    noise_var = sigma_obs ** 2,
    tags      = lx.positive_semidefinite_tag,
)

# Cross-covariance K_{X*, X} — never materialised.
Kxs_op = gx.ImplicitCrossKernelOperator(
    kernel_fn   = lambda xi, zj: k_st(xi, zj, sigma2, Ls, Lt),
    X_data      = X_grid,                 # (n, 3) — the prediction inputs
    X_inducing  = X_train,                # (m, 3)
    batch_size  = 4096,                   # rows of X_grid per scan step
)

mean_field = Kxs_op.mv(alpha)             # (n,) — Step 5 of the 00 algorithm
correction = Kxs_op.mv(beta)              # (n,) — Step 10, called once per sample
```

### What you save

| Object | Naive | Matrix-free |
|---|---|---|
| Peak memory for cross-cov | $8nm$ B = $4\,\text{GB}$ | $8 \cdot \text{batch\_size} \cdot m$ B ≈ $160\,\text{MB}$ at `batch_size=4096` |
| Peak memory for Gram | $8m^2$ B = $200\,\text{MB}$ | $8m$ B ≈ $40\,\text{KB}$ per row |
| Per-matvec flops | $\mathcal{O}(nm)$ | $\mathcal{O}(nm)$ — same |

The flop count does not change — but the working set goes from "blows your laptop" to "comfortably fits in L3 cache".

---

## Machinery 2 — avoiding the $m^3$ Cholesky

You have two routes, depending on whether you want an *exact* solve against the dense Gram or an *approximate* solve against a low-rank surrogate.

### Route A — preconditioned CG against the implicit Gram (exact)

[`gaussx.PreconditionedCGSolver`](https://github.com/jejjohnson/gaussx/blob/main/src/gaussx/_strategies/_precond_cg.py) builds a rank-$k$ pivoted partial Cholesky preconditioner via `matfree.low_rank.cholesky_partial_pivot` and feeds it into `lineax.CG`. For $C = K_{\mathcal{X}\mathcal{X}} + \sigma^2 I$ — the textbook target of preconditioned GP CG — convergence is essentially independent of $m$ once the preconditioner captures the leading spectrum. Each CG iteration is one $C$-matvec ⇒ one `ImplicitKernelOperator` matvec ⇒ no dense matrix ever exists.

```python
solver = gx.PreconditionedCGSolver(
    preconditioner_rank = 50,             # rank-50 partial Cholesky
    shift               = sigma_obs ** 2, # the σ²I in C — used by Woodbury inside the precond
    rtol                = 1e-6,
    max_steps           = 200,            # 50–200 iters typical
    lanczos_order       = 30,             # only used for SLQ logdet, irrelevant here
)

alpha = gx.solve(C_op, y, solver=solver)  # (m,) — Step 3 of the 00 algorithm
# Per-sample correction also reuses the same solver:
beta  = gx.solve(C_op, y - f_tilde_X, solver=solver)
```

Cost: $\mathcal{O}(\text{n\_iters} \cdot m^2)$ flops, $\mathcal{O}(\text{precond\_rank} \cdot m)$ memory. With `n_iters ≈ 100` and $m=5\text{k}$ that is $2.5 \times 10^9$ flops vs. $4 \times 10^{10}$ for dense Cholesky — about **15× faster**, and you skipped the $200\,\text{MB}$ Gram allocation along the way.

### Route B — RFF + Woodbury (approximate, but very cheap when $r \ll m$)

The 00 note is already going to compute RFF features for the prior path. Reuse them for the Gram itself: by Bochner's theorem $K_{\mathcal{X}\mathcal{X}} \approx \Phi \Phi^\top$ where $\Phi \in \mathbb{R}^{m \times r}$. Then $C \approx \sigma^2 I + \Phi \Phi^\top$, which is exactly the structure that [`gaussx.LowRankUpdate`](https://github.com/jejjohnson/gaussx/blob/main/src/gaussx/_operators/_low_rank_update.py) is built for, and `gx.solve` dispatches Woodbury on it automatically.

[`gaussx.rff_operator`](https://github.com/jejjohnson/gaussx/blob/main/src/gaussx/_kernels/_kernel_approx.py) constructs this in one call:

```python
import jax.random as jr

# Spectral frequencies for σ²·k_s·k_t — separable kernel ⇒ ω = (ω_s, ω_t)
omega = sample_spectral_frequencies(key_omega, n_features=2000)   # (r, 3)
phase = jr.uniform(key_phase, (2000,), maxval=2 * jnp.pi)

K_lr = gx.rff_operator(X_train, omega=omega, b=phase)             # K ≈ ΦΦᵀ as LowRankUpdate
C_lr = K_lr + lx.IdentityLinearOperator(in_struct, in_struct) * sigma_obs ** 2
# (or: gx.low_rank_plus_diag(...) — see __init__.py for the helper)

alpha = gx.solve(C_lr, y)                                          # Woodbury under the hood
```

Cost: $\mathcal{O}(r^3 + r^2 m)$ for the inner $r\times r$ solve, $\mathcal{O}(rm)$ memory. With $r=2\text{k}$, $m=5\text{k}$: $8 \times 10^9 + 2 \times 10^{10} = 2.8 \times 10^{10}$ flops — comparable to dense Cholesky on this size, but it scales as $rm^2$ instead of $m^3$, so it pulls ahead fast as $m$ grows.

[`gaussx.nystrom_operator`](https://github.com/jejjohnson/gaussx/blob/main/src/gaussx/_kernels/_kernel_approx.py) is the inducing-point cousin if uniformly subsampled altimeter tracks make a better basis than RFF — same `LowRankUpdate` output, same Woodbury dispatch.

---

## Machinery 3 — the whole Matheron loop is already in pyrox

[`pyrox.gp.PathwiseSampler`](https://github.com/jejjohnson/pyrox/blob/main/src/pyrox/gp/_pathwise.py) implements Steps 6–10 of the 00 algorithm verbatim. It combines:

- [`pyrox._basis._rff.draw_rff_cosine_basis`](https://github.com/jejjohnson/pyrox/blob/main/src/pyrox/_basis/_rff.py) — draws `(variance, lengthscale, ω, phase, weights)` from the kernel's spectral density, supports RBF and Matérn (any $\nu$ — including $3/2$ used here);
- `evaluate_rff_cosine_paths` — evaluates the prior path $\tilde f$ at any $X$, vectorised over `n_paths`;
- a frozen `(X1, X2) -> K(X1, X2)` callable that bakes in the *same* hyperparameter draw used for the RFF, so the correction $K(\mathcal{X}^*, \mathcal{X})\,\beta$ stays consistent.

The result is a `PathwiseFunction` you call at any $\mathcal{X}^*$ — the per-call cost is $\mathcal{O}(n_* F D + n_* m)$ per path.

### Pseudocode — drop-in replacement for Steps 6–10

```python
from pyrox.gp import GPPrior, PathwiseSampler, Matern

# Build a Matern-3/2 kernel on (lon, lat, t). Could also be a product of
# spatial-Matern × temporal-OU (Matern-1/2) — pyrox supports kernel_mul.
kernel = Matern(nu=1.5, lengthscale=Ls, variance=sigma_eta_2)

prior     = GPPrior(kernel=kernel, X=X_train)            # ConditionedGP carries C internally
posterior = prior.condition(y, noise_var=sigma_obs ** 2)

sampler   = PathwiseSampler(posterior, n_features=2000)
paths     = sampler.sample_paths(key, n_paths=100)       # PathwiseFunction
samples   = paths(X_grid)                                # (100, n) — Step 10
```

That is the entire pathwise loop in five lines.

### Caveat — swap the inner solver

`PathwiseSampler` currently uses `gaussx.cholesky` for the $C^{-1}$ inside Matheron's correction. To combine wins #2 and #3 you would either (a) use Route B (low-rank $C$) so the Cholesky is cheap, or (b) pass a different solver into the sampler. The latter is a small monkeypatch / fork — flag it if you ever need $m \gtrsim 10^4$.

---

## Machinery 4 — pointwise posterior variance without the $nm^2$ row-loop

The 00 note flags pointwise variance at $\mathcal{O}(nm^2)$ — one solve per test point. [`gaussx.love_cache`](https://github.com/jejjohnson/gaussx/blob/main/src/gaussx/_gp/_love.py) (LanczOs Variance Estimates, Pleiss et al. 2018) precomputes a rank-$k$ Lanczos factorisation of $C^{-1}$ once, after which every test-point variance costs $\mathcal{O}(mk)$ instead of $\mathcal{O}(m^2)$.

```python
cache = gx.love_cache(C_op, lanczos_order=50)            # one-time, ~50 CG-matvecs

def post_var_at(x_star):
    k_star_row = jax.vmap(lambda xj: k_st(x_star, xj, sigma2, Ls, Lt))(X_train)  # (m,)
    return sigma_eta_2 - gx.love_variance(cache, k_star_row)

post_var_map = jax.vmap(post_var_at)(X_grid)             # (n,)
```

For $n=10^5$, $m=5\text{k}$, $k=50$: cost drops from $\sim 2.5 \times 10^{12}$ flops (one solve per test point) to $\sim 2.5 \times 10^{10}$ — about **100× cheaper**. Use the empirical sample variance from $\{\eta^*_s\}$ when you only need a few samples; switch to LOVE when you specifically want a smooth, non-Monte-Carlo-noisy uncertainty map.

---

## Two structural exploits the libraries don't do for you

These are not in gaussx/pyrox as primitives but they compose with the operators above and are worth coding by hand for an SSH-shaped problem.

### Exploit A — temporal factor of $K_{\mathcal{X}^*\mathcal{X}}$ has only 3 distinct values

All prediction points share time $t$, so for any observation $j$ in time group $t-\tau$, $t$, or $t+\tau$ the temporal weight $k_t(t, t_j; L_t)$ is one of three scalars: $e^{-\tau/L_t}$, $1$, $e^{-\tau/L_t}$. Build *spatial-only* implicit cross operators per time group and combine with scalar weights at matvec time:

```python
def make_block(X_train_group, weight):
    spatial_op = gx.ImplicitCrossKernelOperator(
        kernel_fn   = lambda x, z: sigma2 * matern32_great_circle(x[:2], z[:2], Ls),
        X_data      = X_grid_xy,        # (n, 2) — drop time axis
        X_inducing  = X_train_group,    # (m_group, 2)
        batch_size  = 4096,
    )
    return weight, spatial_op

w_minus = jnp.exp(-tau / Lt)
blocks = [
    make_block(X_train_minus_xy, w_minus),
    make_block(X_train_zero_xy,  1.0),
    make_block(X_train_plus_xy,  w_minus),
]

def Kxs_mv(beta):                           # beta is partitioned to match groups
    out = 0.0
    for (w, op), b_group in zip(blocks, beta_partition):
        out = out + w * op.mv(b_group)
    return out
```

This **halves the kernel-eval count** (one trig stack per spatial pair, not one per spatiotemporal pair) and lets you cache the spatial cross-covs across re-fits with different $L_t$ — handy when you sweep temporal length scales during hyperparameter learning.

### Exploit B — the $3 \times 3$ block structure of $K_{\mathcal{X}\mathcal{X}}$

The training inputs $\mathcal{X}$ split into 3 time groups, so $K_{\mathcal{X}\mathcal{X}}$ is a $3\times 3$ block matrix whose $(a,b)$ block is

$$
\bigl[K_{\mathcal{X}\mathcal{X}}\bigr]_{ab} = \sigma_\eta^2 \cdot e^{-|t_a - t_b|/L_t} \cdot K_s\bigl(\text{group}_a,\, \text{group}_b\bigr).
$$

Six unique spatial blocks (3 diagonal + 3 off-diagonal); the temporal weights are scalars. Build the spatial blocks once with `ImplicitKernelOperator` / `ImplicitCrossKernelOperator`, wrap with a temporally-weighted block matvec, and you avoid $\sim 8/9$ of the redundant great-circle evaluations on every Gram matvec inside CG.

This composes with [`gaussx.BlockDiag`](https://github.com/jejjohnson/gaussx/blob/main/src/gaussx/_operators/_block_diag.py) for the within-group blocks, with the off-diagonal cross-blocks added back as scaled implicit operators inside a custom matvec. Worth the effort once you push to many time groups (a sliding window with $|\tau| \in \{1, 2, \ldots, 5\}$ days, say).

---

## Putting it all together — the efficient SSH algorithm

The same ten-step algorithm from 00, rewritten with the machinery above. Storage is now $\mathcal{O}(m + r + n)$ throughout — never $nm$, never $m^2$.

### Offline (once per time window)

```python
# --- Choose a structured representation of C ---
# Route A (exact, preferred when m ≲ 5e4 and you have time for ~100 CG iters):
C_op  = gx.ImplicitKernelOperator(k_st_curried, X_train, noise_var=sigma_obs ** 2,
                                  tags=lx.positive_semidefinite_tag)
solver = gx.PreconditionedCGSolver(preconditioner_rank=50, shift=sigma_obs ** 2,
                                   rtol=1e-6, max_steps=200)

# Route B (low-rank, preferred when m ≫ 1e4 and r ≪ m):
# K_lr  = gx.rff_operator(X_train, omega=omega, b=phase)
# C_lr  = K_lr + identity(m) * sigma_obs ** 2     # LowRankUpdate ⇒ Woodbury
# solver = None                                    # gx.solve auto-dispatches Woodbury

# --- Step 3: solve for the dual weights α ---
alpha = gx.solve(C_op, y, solver=solver)          # (m,)

# --- Steps 4–5: matrix-free posterior mean field ---
Kxs_op = gx.ImplicitCrossKernelOperator(
    kernel_fn  = k_st_curried,
    X_data     = X_grid,                          # (n, 3)
    X_inducing = X_train,                         # (m, 3)
    batch_size = 4096,
)
mu_field = Kxs_op.mv(alpha)                       # (n,) — never materialise (n, m)
```

### Per posterior sample

```python
# --- Steps 6–7: RFF prior path at training + grid (one consistent draw) ---
variance, lengthscale, omega, phase, weights = pyrox.rff.draw_rff_cosine_basis(
    kernel      = matern_3_2,
    key         = sample_key,
    n_paths     = 1, n_features=2000, in_features=3, dtype=jnp.float64,
)
f_tilde_X     = pyrox.rff.evaluate_rff_cosine_paths(
    X_train, variance=variance, lengthscale=lengthscale,
    omega=omega, phase=phase, weights=weights)[0]
f_tilde_grid  = pyrox.rff.evaluate_rff_cosine_paths(
    X_grid,  variance=variance, lengthscale=lengthscale,
    omega=omega, phase=phase, weights=weights)[0]

# --- Step 8–9: innovation + correction solve (reuses solver) ---
delta = y - f_tilde_X                             # (m,)
beta  = gx.solve(C_op, delta, solver=solver)      # (m,) — same CG, same precond

# --- Step 10: matrix-free correction ---
eta_sample = f_tilde_grid + Kxs_op.mv(beta)       # (n,) — one exact posterior draw
```

Or, equivalently, the four-line version using the pre-built sampler:

```python
posterior = GPPrior(kernel=matern_3_2_x_OU, X=X_train).condition(y, sigma_obs ** 2)
sampler   = PathwiseSampler(posterior, n_features=2000)
paths     = sampler.sample_paths(key, n_paths=S)
samples   = paths(X_grid)                          # (S, n)
```

### Pointwise variance map (optional, when sample-variance is too noisy)

```python
cache = gx.love_cache(C_op, lanczos_order=50)
post_var_map = jax.vmap(
    lambda x_star: sigma_eta_2 - gx.love_variance(
        cache,
        jax.vmap(lambda xj: k_st(x_star, xj, sigma2, Ls, Lt))(X_train),
    )
)(X_grid)                                          # (n,)
```

---

## Cost recap

Same complexity table as 00, side-by-side with what the machinery delivers.

| Phase | Naive (00) | With machinery (here) |
|---|---|---|
| Build $K_{\mathcal{X}\mathcal{X}}$ + Cholesky | $\mathcal{O}(m^2)$ memory, $\mathcal{O}(m^3)$ flops | $\mathcal{O}(m)$ memory, $\mathcal{O}(\text{n\_iters} \cdot m^2)$ flops via PCG (Route A); or $\mathcal{O}(rm)$ + $\mathcal{O}(r^2 m + r^3)$ via Woodbury (Route B) |
| Build $K_{\mathcal{X}^*\mathcal{X}}$ + posterior mean | $\mathcal{O}(nm)$ memory, $\mathcal{O}(nm)$ flops | $\mathcal{O}(\text{batch\_size} \cdot m)$ memory, $\mathcal{O}(nm)$ flops |
| $S$ posterior samples | $\mathcal{O}(Sn)$ memory, $S \cdot \mathcal{O}(nm + rn)$ flops | same flops, but cross-cov memory shrinks from $nm$ to $\text{batch\_size} \cdot m$ — and the $C^{-1}\delta$ solve is the cheap PCG/Woodbury solve, not a fresh Cholesky |
| Pointwise variance | $\mathcal{O}(nm^2)$ flops | $\mathcal{O}(nmk)$ flops via LOVE, $k\sim 50$ |

The **memory bound drops from $\mathcal{O}(m^2 + nm)$ to $\mathcal{O}(m + r + n + \text{batch\_size}\cdot m)$** — i.e. from 4 GB to a few hundred MB at SSH-realistic scales. The **flop bound drops from $\mathcal{O}(m^3 + Snm)$ to $\mathcal{O}(\text{n\_iters}\cdot m^2 + Snm)$** with the same constants. Both wins compound: matrix-free + PCG means you can push to $m \sim 5 \times 10^4$ (a full Mediterranean month) on a single GPU, where the naive code would OOM long before it finished its first Cholesky.

---

## Wall-clock estimates for realistic SSH reconstructions

Order-of-magnitude time budgets to reconstruct daily SSH fields over the **Mediterranean Sea**, **North Atlantic**, and **global ocean**, for both an analysis-day window $[t-\tau,\, t]$ (operational, 2 time groups) and a reanalysis-day window $[t-\tau,\, t,\, t+\tau]$ (3 time groups, lookahead allowed). All numbers below are accurate to a factor of $\sim 2$ — they exist to flag *which configurations are tractable on what hardware*, not to commit to a specific runtime.

### Daily observation counts (Copernicus Marine Service catalogue, 2024–2026)

CMEMS publishes the along-track and SWOT L3 streams as `SEALEVEL_GLO_PHY_L3_*_OBSERVATIONS_008_*` (NRT and reprocessed multi-mission), the SWOT KaRIn 7.5 km L3 as `SEALEVEL_GLO_PHY_L3_MY_008_069`, and tide-gauge SSH as `INSITU_GLO_PHY_SSH_DISCRETE_NRT_013_059`. Post-QC daily counts:

| Stream | Per-satellite raw | After QC | Active in 2024–2026 |
|---|---|---|---|
| Nadir altimeters (1 Hz) | $\sim 86\text{k}$/day/sat | $\sim 50\text{k}$/day/sat | S3A, S3B, S6-MF, SARAL, CryoSat-2, HY-2B/C — **6 sats ⇒ $\sim 3\times 10^5$/day globally** |
| SWOT KaRIn (2 km native, 7.5 km L3) | $\sim 5\times 10^6$/day | $\sim 2\times 10^6$/day | 1 mission, operational since 2023 |
| In-situ SSH (tide gauges, GLOSS) | — | $\sim 10^4$/day | Negligible vs altimetry; anchors absolute datum |
| **Combined total (post-QC)** | | **$\sim 2\times 10^6$/day globally** | SWOT dominates |

Region fractions (area-weighted):

| Region | Area | Fraction | Daily obs (with SWOT) | Daily obs (nadir-only, archived years) |
|---|---|---|---|---|
| Mediterranean Sea | $2.5\,\text{M km}^2$ | $0.7\%$ | $\sim 1.5\times 10^4$ | $\sim 2\times 10^3$ |
| North Atlantic | $40\,\text{M km}^2$ | $11\%$ | $\sim 2.5\times 10^5$ | $\sim 3\times 10^4$ |
| Global ocean | $360\,\text{M km}^2$ | $100\%$ | $\sim 2\times 10^6$ | $\sim 3\times 10^5$ |

### Per-day problem sizes

With $\tau \approx 2$–$3$ days and obs grouped by time-band:

| Region | Grid $n$ ($0.1°$ res) | $m$ analysis (2 groups, with SWOT) | $m$ reanalysis (3 groups, with SWOT) | $m$ reanalysis (nadir-only) |
|---|---|---|---|---|
| Mediterranean | $10^5$ | $3\times 10^4$ | $4.5\times 10^4$ | $6\times 10^3$ |
| North Atlantic | $6.5\times 10^5$ | $5\times 10^5$ | $7.5\times 10^5$ | $10^5$ |
| Global ocean | $6.5\times 10^6$ | $4\times 10^6$ | $6\times 10^6$ | $10^6$ |

### Per-day compute, two regimes

Two implementation paths from above, with $r=2000$ RFF features and $S=100$ posterior samples per day. The "fully-RFF" column uses RFF for *both* the prior path *and* the cross-covariance — the per-sample correction matvec then collapses from $\mathcal{O}(nm)$ to $\mathcal{O}(r(n+m))$, removing $m$ from the inner loop entirely.

| Region (reanalysis day, with SWOT) | Exact GP via PCG (Route A) — flops/day | Fully-RFF Woodbury (Route B+) — flops/day |
|---|---|---|
| Mediterranean ($m\!\sim\!4.5\text{k}$, $n\!\sim\!10^5$) | $\sim 2\times 10^{13}$ | $\sim 2\times 10^{11}$ |
| North Atlantic ($m\!\sim\!7.5\text{k}\cdot 10^2$, $n\!\sim\!6.5\times 10^5$) | $\sim 6\times 10^{15}$ | $\sim 3\times 10^{12}$ |
| Global ocean ($m\!\sim\!6\times 10^6$, $n\!\sim\!6.5\times 10^6$) | $\sim 4\times 10^{17}$ — **infeasible** | $\sim 2.5\times 10^{13}$ |

Effective sustained throughput on common targets: a modern Xeon (32 cores, MKL) hits $\sim 200\,\text{GFLOPS}$ f64 on these matvec/scan kernels; an A100 GPU hits $\sim 5\,\text{TFLOPS}$ f64 (well below peak because the workload is bandwidth-bound). Per-day wall-clock follows directly:

| Region | Route A — A100 / day | Route A — CPU / day | Fully-RFF — A100 / day | Fully-RFF — CPU / day |
|---|---|---|---|---|
| Mediterranean | $\sim 4\,\text{s}$ | $\sim 100\,\text{s}$ | $< 0.1\,\text{s}$ | $\sim 1\,\text{s}$ |
| North Atlantic | $\sim 20\,\text{min}$ | $\sim 8\,\text{h}$ | $\sim 0.6\,\text{s}$ | $\sim 15\,\text{s}$ |
| Global | $\sim 22\,\text{h}$ | weeks | $\sim 5\,\text{s}$ | $\sim 2\,\text{min}$ |

Analysis day (2 time groups instead of 3) reduces $m$ by $\sim 1/3$, so per-day cost drops by $\sim 0.5\times$ for Route A (the $m^2$ term) and $\sim 0.6\times$ for Route B+. Numbers below take this into account.

### Full-window wall-clock

Multiply per-day by 30 / 90 / 180. Reanalysis with the full SWOT-era constellation, on a single A100:

| Region | 1 month (30 d) | 3 months (90 d) | 6 months (180 d) |
|---|---|---|---|
| **Mediterranean** | Route A: $2\,\text{min}$ · RFF+: $3\,\text{s}$ | Route A: $6\,\text{min}$ · RFF+: $9\,\text{s}$ | Route A: $12\,\text{min}$ · RFF+: $18\,\text{s}$ |
| **North Atlantic** | Route A: $10\,\text{h}$ · RFF+: $20\,\text{s}$ | Route A: $30\,\text{h}$ · RFF+: $1\,\text{min}$ | Route A: $60\,\text{h}$ · RFF+: $2\,\text{min}$ |
| **Global** | Route A: **infeasible** · RFF+: $2.5\,\text{min}$ | RFF+: $7.5\,\text{min}$ | RFF+: $15\,\text{min}$ |

Same totals on a 32-core CPU (Route A column for global is dropped — even with RFF+ it would take days at exact-GP scales):

| Region | 1 month | 3 months | 6 months |
|---|---|---|---|
| **Mediterranean** | Route A: $50\,\text{min}$ · RFF+: $30\,\text{s}$ | Route A: $2.5\,\text{h}$ · RFF+: $90\,\text{s}$ | Route A: $5\,\text{h}$ · RFF+: $3\,\text{min}$ |
| **North Atlantic** | Route A: $10\,\text{d}$ · RFF+: $7.5\,\text{min}$ | Route A: $30\,\text{d}$ · RFF+: $22\,\text{min}$ | Route A: $60\,\text{d}$ · RFF+: $45\,\text{min}$ |
| **Global** | RFF+: $1\,\text{h}$ | RFF+: $3\,\text{h}$ | RFF+: $6\,\text{h}$ |

Analysis day (operational, single $[t-\tau, t]$ window per run) is roughly half these per-day numbers — Mediterranean analysis is sub-second on any hardware; global analysis with RFF+ is a few seconds per A100 day.

### Where things break

- **Route A on global SWOT-era data is hopeless on a single node.** $m \sim 6\times 10^6$ pushes the per-CG-iter cost ($\mathcal{O}(m^2)$ via implicit-kernel scan) into the $10^{13}$-flop range, with $\sim 100$ iterations per solve and $\sim S$ solves per day. You end up at $\sim 10^{17}$ flops/day, which is days-per-day on an A100. Either drop SWOT (reverts to the nadir column where Route A is viable for NA but still tight for global), reduce resolution, switch to **inducing-point sparse GPs** (`gaussx.nystrom_operator` + `LowRankUpdate`), or move to a **state-space SPDE formulation** (which gets you sparse precision matrices and $\mathcal{O}(m)$ smoothing, at the cost of requiring an isotropic Matérn kernel and more setup work).
- **RFF+ is the right default for North Atlantic and global.** The accuracy loss vs exact GP is $\mathcal{O}(1/\sqrt{r})$ in the kernel approximation; for SSH-mesoscale fields and $r=2000$ this is well below the altimeter noise floor.
- **Hyperparameter learning is not in these numbers.** Estimating $(\sigma_\eta^2, L_s, L_t, \sigma_{obs}^2)$ via marginal-likelihood gradient descent multiplies the cost by $\sim 50$–$100$ optimiser steps. Do this once on a representative window, then freeze the hyperparameters for the full reanalysis pass.
- **I/O is not in these numbers.** At $2\times 10^6$ obs/day SWOT-era, the *download* from CMEMS at typical link speeds (~$50\,\text{MB/s}$) is comparable to or larger than the compute on Mediterranean/NA. Pre-stage the data; do not re-fetch per day.
- **Posterior variance maps via LOVE add $\sim 5$–$10\%$ to the per-day budget** — cheap when needed, skip when the empirical sample variance from $\{\eta^*_s\}$ suffices.
