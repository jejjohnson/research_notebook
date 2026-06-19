# Worked Example: Reduced-Order 3D-Var Analogous to DINEOF

This is an end-to-end worked example of filling a gappy ocean field with a reduced-order 3D-Var, and showing that DINEOF is the special case you get when the basis is an EOF basis and the analysis is run in the masking limit.
It assumes the same four libraries are implemented: geonnax for the eigenfunction basis, gauss_flows for the EOF/PPCA prior, somax for the basis wrapper, and vardax for the variational analysis.
Where the MIOST example was dynamical (a 4D-Var through a `diffrax` rollout), this one is static (a single-time 3D-Var), which is exactly the setting DINEOF lives in.

The thread of the example is to start from DINEOF as people usually meet it, rewrite it as a reduced-order 3D-Var, and then run that 3D-Var with vardax so the equivalence is concrete.
The payoff is that once DINEOF is seen as a 3D-Var with a low-rank background covariance, every richer prior we have built becomes available without changing the solver.

## 1. DINEOF as people usually meet it

DINEOF (Data-Interpolating Empirical Orthogonal Functions) fills gaps in a field by iterating two steps: project the currently-filled field onto its leading EOFs, then overwrite the observed pixels with the data and leave the reconstruction at the missing pixels.
Concretely, given a gappy field $y$ with a binary mask $m$ (one where observed, zero where missing), DINEOF repeats

$$
x \leftarrow \Phi_K \Phi_K^\top x,   x \leftarrow m \odot y + (1 - m) \odot x
$$

where $\Phi_K$ holds the $K$ leading EOFs.
The first step is a rank-$K$ projection, the truncated SVD reconstruction; the second is the data-replacement that keeps the observed pixels pinned.
Iterating to convergence gives a gap-free field that lives in the span of the leading EOFs and matches the data where data exist.

This is intuitive but it hides what it is statistically: a maximum-a-posteriori estimate under a low-rank Gaussian prior, in the limit of zero observation noise.
Making that explicit is what lets us generalise it.

## 2. Rewriting DINEOF as a reduced-order 3D-Var

A 3D-Var minimises a background term plus an observation term,

$$
J(x) = \frac{1}{2} \lVert x - x_b \rVert^2_{B^{-1}} + \frac{1}{2} \lVert m \odot (y - x) \rVert^2_{R^{-1}}
$$

and the reduced-order idea is to parameterise the increment in a basis instead of solving for $x$ pixel by pixel.
Writing $x = x_b + \Phi w$ with coefficients $w \sim \mathcal{N}(0, \Lambda)$ induces the background covariance $B = \Phi \Lambda \Phi^\top$, and the background term becomes a penalty on the coefficients:

$$
J(w) = \frac{1}{2} \lVert w \rVert^2_{\Lambda^{-1}} + \frac{1}{2} \lVert m \odot (y - x_b - \Phi w) \rVert^2_{R^{-1}}
$$

Now take the EOF basis for $\Phi$, set $\Lambda$ to the EOF variances, and let the observation noise go to zero.
The minimiser of $J$ is then the field that lies in the EOF span and matches the data exactly where observed, which is precisely the DINEOF fixed point.
DINEOF’s alternating projection is the iterative solver for this 3D-Var in the noiseless, EOF-basis, low-rank limit; the replacement step is the $R \to 0$ data constraint and the projection step is the restriction to the EOF span.

Seen this way, three knobs that DINEOF fixes become free.
The basis need not be EOFs; the prior need not be the raw EOF variances; and the observation noise need not be zero.
We will keep the EOF basis to stay analogous, but route it through the same machinery that would accept any of the others.

## 3. The EOF/PPCA basis from gauss_flows and geonnax

The EOF basis is the principal subspace of a data ensemble, which is exactly what probabilistic PCA fits by SVD, and gauss_flows provides that as `GaussianPCA.from_data`.
Fitting it to a stack of historical (gap-filled or model) fields gives the loadings $W$, the leading directions, and a noise floor $\sigma^2$ — the PPCA covariance $\Sigma = W W^\top + \sigma^2 I$ whose leading eigenvectors are the EOFs.

$$
\Sigma = W W^\top + \sigma^2 I,   W = V_K (\Delta_K - \sigma^2 I)^{1/2}
$$

with $V_K$ the top-$K$ singular vectors of the centred ensemble and $\Delta_K$ their variances.
The columns of $W$ are the EOFs scaled by their standard deviations, so $W$ is directly the synthesis dictionary $\Phi \Lambda^{1/2}$ we want, and the whitening map $w \mapsto u$ is the PPCA preconditioner.

```python
from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from gauss_flows import GaussianPCA

# ensemble: (n_samples, N) stack of historical fields, gaps pre-filled or modelled.
ppca = GaussianPCA.from_data(ensemble, latent_dim=20)   # SVD/PPCA -> 20 EOFs

# The EOF dictionary scaled by sqrt-variance is exactly W; the whitening pair
# (to_base / from_base) gives the preconditioned control.
Phi_Lambda_half: Float[Array, "N K"] = ppca.W           # synthesis: x_inc = W u
```

If instead we wanted a fixed analytic basis rather than a data-driven one — say a smooth large-scale prior — geonnax would supply it through the same shape: `fourier_basis` for the dictionary and `fourier_eigenvalues` for the spectrum, with a kernel spectral density giving $\Lambda$.
The point is that `somax`’s `SpatialBasis` does not care which of these produced the dictionary; it holds $\Phi$ and the per-mode std and exposes synthesis and whitening.

```python
from somax._src.core.basis import SpatialBasis

# Wrap the PPCA EOFs as a somax SpatialBasis. Here the per-mode std is folded
# into W already, so std is ones; for an analytic basis it would be sqrt(Lambda).
basis = SpatialBasis(Phi=ppca.W, std=jnp.ones(ppca.W.shape[1]))
```

## 4. The vardax solve

vardax provides `ThreeDVar`, which minimises the background-plus-observation cost with `lineax` operators for $B$ and $R$ and an `optimistix` inner minimiser.
The reduced-order basis enters as the background covariance: we hand vardax $B = \Phi \Lambda \Phi^\top$ as a `lineax` operator, so the solver works in state space while the operator carries the low-rank EOF structure, and the analysis is automatically confined to the EOF span as the EOF variances dominate.

```python
import lineax as lx
import optimistix as optx

from vardax import ThreeDVar
from vardax import LinearObs            # masked identity observation operator

N, K = ppca.W.shape

# B as a lineax operator: B v = W (W^T v) + sigma^2 v, never densified.
sigma2 = jnp.exp(2.0 * ppca.log_sigma)
def b_mv(v: Float[Array, "T N"]) -> Float[Array, "T N"]:
    coeffs = jnp.einsum("nk,tn->tk", ppca.W, v)         # analyse: W^T v
    return jnp.einsum("nk,tk->tn", ppca.W, coeffs) + sigma2 * v   # synthesize + floor

B = lx.FunctionLinearOperator(b_mv, jax.eval_shape(lambda: jnp.zeros((1, N))))
R = lx.DiagonalLinearOperator(jnp.full((1, N), 1e-4))   # small obs noise -> DINEOF limit

analysis = ThreeDVar(
    obs_op=LinearObs(),                  # H = masked identity: samples observed pixels
    prior_mean=x_background[None, :],    # x_b, shape (T=1, N); the ensemble mean is natural
    prior_cov_op=B,                      # <-- the EOF/PPCA basis, as the B operator
    obs_cov_op=R,
    minimiser=optx.BFGS(rtol=1e-8, atol=1e-8),
)

# batch.input = m ⊙ y (gappy field), batch.mask = m. Shapes (B, T=1, N).
x_analysis = analysis(batch)             # (B, 1, N) gap-filled field
```

As $R \to 0$ this reproduces DINEOF: the analysis matches the data at observed pixels and lies in the EOF span elsewhere.
With a finite $R$ it is the better-posed thing DINEOF cannot do, a noise-aware reconstruction that does not interpolate observation error.

Equivalently, and more in the spirit of the reduced control, we can optimise the coefficients directly in the whitened space, which is the form that makes a non-Gaussian prior exact and which mirrors the MIOST example.

```python
import einx

def cost(u: Float[Array, "K"]):
    x = x_background + ppca.from_base(u)            # x = x_b + W u  (or T^{-1}(u))
    resid = batch.mask[0] * (batch.input[0] - x)    # masked misfit
    j_obs = 0.5 * einx.sum("n ->", resid**2) / 1e-4
    return j_obs + 0.5 * einx.sum("k ->", u**2)     # isotropic whitened EOF prior

u_star = optx.minimise(cost, optx.BFGS(rtol=1e-8, atol=1e-8), y0=jnp.zeros(K)).value
x_filled = x_background + ppca.from_base(u_star)
```

The whitened prior term $\frac{1}{2} \lVert u \rVert^2$ is the isotropic image of the EOF prior, so the coefficient problem is perfectly conditioned and BFGS converges in a handful of steps — the well-posed counterpart of DINEOF’s many alternating-projection sweeps.

## 5. What the example demonstrates

The reconstruction fills the gappy field from the observed pixels, confined to the EOF subspace, exactly as DINEOF would, but expressed as a 3D-Var with a low-rank background covariance.
Three points carry over from the dynamical example and one is new.

The libraries again each did one job: gauss_flows fit the EOF/PPCA basis by SVD and supplied the whitening map, somax wrapped it as a `SpatialBasis`, and vardax ran the 3D-Var with the basis as the background operator $B$.
geonnax would have supplied an analytic basis through the same `SpatialBasis` shape had we wanted one instead of the data-driven EOFs.

The basis is again a covariance: $B = \Phi \Lambda \Phi^\top$ is the EOF prior, and DINEOF is the noiseless, EOF-basis limit of the general reduced-order 3D-Var.

The reduced control is again whitened, so the coefficient problem is well-conditioned and the solver is fast and exact at the optimum rather than an alternating heuristic.

The new point is the bridge to the dynamical case.
This 3D-Var is the single-time slice of the weak-constraint 4D-Var in the MIOST example: there the basis modelled the per-step model error $Q$, here it models the static background error $B$, but the object — a reduced, whitened, basis-induced Gaussian — and the code path are the same.
Moving from DINEOF to MIOST is therefore not a change of method, only a change of which covariance the basis stands in for and whether a `diffrax` rollout sits between the control and the observations.

A caveat: DINEOF in practice also estimates the optimal number of EOFs by cross-validation, which here is the choice of `latent_dim`; the example fixes it, and a faithful DINEOF replacement would select it by holding out observed pixels and minimising the reconstruction error, which the same `ThreeDVar` call supports by sweeping `latent_dim`.
