---
title: "3D-Var — derivation, preconditioning, and adjoint methods with lineax / optimistix"
---

# 3D-Var for methane retrieval

This note derives the cost function used by `plume_simulation.assimilation`, walks through the three control-variable formulations we implement, and then maps each piece of the math onto a concrete `jax` / `lineax` / `optimistix` / `gaussx` call. The aim is that you can read this side-by-side with [solve.py](../../src/plume_simulation/assimilation/solve.py) and see exactly which line implements which equation.

The derivation borrows notation from Bouttier & Courtier's *Data assimilation concepts and methods* [^bc1999] and Rodgers' *Inverse methods for atmospheric sounding* [^rodgers2000]; the JAX-side implementation tricks are from the lineax docs [^lineaxdocs] and the optimistix paper [^optimistix].

[^bc1999]: Bouttier, F. & Courtier, P. (1999). *Data assimilation concepts and methods*. Meteorological Training Course Lecture Series, ECMWF. https://www.ecmwf.int/sites/default/files/elibrary/2003/16928-data-assimilation-concepts-and-methods.pdf

[^rodgers2000]: Rodgers, C. D. (2000). *Inverse methods for atmospheric sounding: theory and practice*. World Scientific.

[^lineaxdocs]: Rader, J., Lyons, T., & Kidger, P. (2024). *Lineax: a library for linear solvers in JAX.* https://docs.kidger.site/lineax/

[^optimistix]: Rader, J., Lyons, T., & Kidger, P. (2024). *Optimistix: modular optimisation in JAX and Equinox.* arXiv:2402.09983.

## 1. From Bayes to the 3D-Var cost

We have a state vector $x \in \mathbb{R}^N$ — for our methane retrieval, the per-pixel VMR field flattened to length $N = n_y \cdot n_x$ — and an observation vector $y \in \mathbb{R}^M$, the band-integrated radiance cube. Both come with Gaussian uncertainty:

- **Prior**: $x \sim \mathcal{N}(x_b, B)$. The background mean $x_b$ is whatever we knew before the observation arrived (typically a uniform climatological VMR), and $B \in \mathbb{R}^{N \times N}$ encodes per-pixel variance and any spatial correlations.
- **Likelihood**: $y \mid x \sim \mathcal{N}(H(x), R)$ with the (possibly nonlinear) observation operator $H$ and the observation-error covariance $R \in \mathbb{R}^{M \times M}$. For our case, $H = \text{SRF} \circ \text{GSD} \circ \text{PSF} \circ \exp(-a \cdot \Delta\text{VMR})$.

By Bayes' rule the posterior $p(x \mid y) \propto p(y \mid x)\, p(x)$ is proportional to $\exp(-J(x))$ with

$$
J(x) \;=\; \tfrac{1}{2}\,(x - x_b)^\top B^{-1} (x - x_b) \;+\; \tfrac{1}{2}\,\bigl(y - H(x)\bigr)^\top R^{-1} \bigl(y - H(x)\bigr).
$$ (eq-J)

The MAP estimate $\hat x = \arg\min_x J(x)$ is the 3D-Var analysis. The "3D" refers to the three spatial dimensions of the state — there is no time axis, distinguishing 3D-Var from 4D-Var.

### 1.1 Incremental form

In practice we minimise over the **increment** $\delta x = x - x_b$, because (i) the background term then has a clean quadratic form, and (ii) the observation operator is often linearised around $x_b$ (the *inner loop* of incremental 3D-Var). Substituting $x = x_b + \delta x$ and writing $d := y - H(x_b)$ for the *innovation*:

$$
J(\delta x) \;=\; \tfrac{1}{2}\,\delta x^\top B^{-1} \delta x \;+\; \tfrac{1}{2}\,\bigl(d - H'\, \delta x\bigr)^\top R^{-1} \bigl(d - H'\, \delta x\bigr),
$$ (eq-J-incremental)

where $H' := \partial H / \partial x \, \big|_{x_b}$ is the **tangent-linear** operator. The optimum satisfies the normal equations

$$
\bigl(B^{-1} + H'^\top R^{-1} H'\bigr)\, \delta\hat x \;=\; H'^\top R^{-1}\, d,
$$ (eq-normal)

with Hessian $A := B^{-1} + H'^\top R^{-1} H'$ — the *information matrix* of the posterior. Its inverse $A^{-1}$ is the textbook posterior covariance for the linear-Gaussian case.

> **In code.** Equation [eq-J](#eq-J) is built by [`build_cost_x`](../../src/plume_simulation/assimilation/cost.py); the prior term uses `gaussx.solve(B, δx_flat)` so any structured `B` from [background.py](../../src/plume_simulation/assimilation/background.py) — diagonal, Kronecker, low-rank — is plugged in without materialising. The obs term differentiates through the JAX `forward_fn` via reverse-mode AD, so we never write `H'^\top` by hand.

## 2. Why preconditioning matters

The catch in [eq-normal](#eq-normal) is that the conditioning of $A$ tracks the spread of eigenvalues of $B^{-1} + H'^\top R^{-1} H'$. In a typical retrieval $B^{-1}$ has a large dynamic range (some pixels are tightly constrained by the prior, others very loose), so a vanilla minimiser like LBFGS can take hundreds of iterations to make progress along the floppy directions.

The **control variable transform** (CVT) eliminates this by changing variables to $\xi := U^{-1} \delta x$, where $U$ is *any* square root of $B$:

$$
B \;=\; U U^\top.
$$ (eq-B-factor)

We pick $U = \mathrm{chol}(B)$ but other choices (eigendecomposition, balanced operators) work the same way. Substituting $\delta x = U \xi$ into [eq-J-incremental](#eq-J-incremental) gives

$$
J(\xi) \;=\; \tfrac{1}{2}\,\xi^\top \xi \;+\; \tfrac{1}{2}\,\bigl(d - (H' U)\, \xi\bigr)^\top R^{-1} \bigl(d - (H' U)\, \xi\bigr).
$$ (eq-J-whitened)

The prior term collapses to $\tfrac{1}{2}\|\xi\|^2$ — a sphere — and the Hessian becomes

$$
A_\xi \;=\; I + (H'U)^\top R^{-1} (H' U),
$$ (eq-hess-whitened)

a perturbation of the identity by a positive-semidefinite term whose rank is at most $M = \dim(y)$. Even vanilla LBFGS converges in a handful of iterations regardless of how badly conditioned $B$ was — this is, in practice, *the single biggest lever you have* in variational DA.

> **In code.** Equation [eq-J-whitened](#eq-J-whitened) is built by [`build_cost_xi`](../../src/plume_simulation/assimilation/cost.py); the operator $U$ is constructed by [`WhiteningTransform.from_background`](../../src/plume_simulation/assimilation/control.py), which calls `gaussx.cholesky(B)`. Crucially, gaussx returns a *structured* operator — a Cholesky of a `Kronecker` is itself a Kronecker of Cholesky factors — so all subsequent matvecs and inverse-solves stay in the cheap regime.

## 3. Primal vs dual: PSAS

The normal equations [eq-normal](#eq-normal) are an $N \times N$ system. When the observation count is small ($M \ll N$ — typical for sparse plumes seen by an instrument with a few bands), the **dual** form is dramatically cheaper. Apply the matrix-inversion lemma to [eq-normal](#eq-normal):

$$
\delta\hat x \;=\; B H'^\top \bigl(H' B H'^\top + R\bigr)^{-1} d.
$$ (eq-dual-solution)

The middle factor is a $M \times M$ system. Define $\lambda := (H' B H'^\top + R)^{-1} d \in \mathbb{R}^M$ — the *dual variable* — so the recipe is:

1. Solve $(H' B H'^\top + R)\, \lambda = d$ for $\lambda$ (CG-friendly: the matrix is symmetric PD).
2. Set $\delta\hat x = B H'^\top \lambda$.

This is the Physical-Space Statistical Analysis System (PSAS) [^cohn1998]. For a *linear* $H$, the dual and primal are mathematically identical; for a nonlinear $H$, the dual is what the inner loop of incremental 4D-Var solves.

[^cohn1998]: Cohn, S. E., et al. (1998). *Assessing the effects of data selection with the DAO Physical-Space Statistical Analysis System.* Mon. Wea. Rev. **126**, 2913-2926.

> **In code.** [`run_dual_psas`](../../src/plume_simulation/assimilation/solve.py) wraps the dual matrix as a `lineax.FunctionLinearOperator`, tags it symmetric+PSD, and feeds it to `lineax.linear_solve(..., solver=lineax.CG(...))`. The two `H'` and $H'^\top$ applications inside the matvec are `forward_linear_fn` and `jax.vjp` respectively — *no explicit adjoint of the obs operator anywhere in the code.*

### 3.1 The matched filter as a single PSAS step

If we set $B = \infty \cdot I$ (uninformative prior), $R = \Sigma_\text{pixel}$ (per-pixel covariance), and restrict $\delta x$ to a 1-D subspace spanned by a target signature $t = H' \cdot \mathbf{1}$, the dual solution [eq-dual-solution](#eq-dual-solution) collapses to the pixel-wise matched filter

$$
\hat\varepsilon \;=\; \frac{(x - \mu)^\top \Sigma^{-1} t}{t^\top \Sigma^{-1} t}.
$$ (eq-matched)

So matched filter, narrow-band retrieval, and full 3D-Var are *the same equation* with progressively richer prior assumptions. The matched-filter implementation in [matched_filter.py](../../src/plume_simulation/radtran/matched_filter.py) and the `gaussx`-backed variant in [gaussx_solve.py](../../src/plume_simulation/radtran/gaussx_solve.py) drop out as the simplest case of the framework here.

## 4. Adjoint methods — four flavours

Computing the gradient $\nabla J(\delta x)$ requires the *adjoint* (transpose) of the tangent-linear $H'$. Historically this was the part of variational DA you wrote and maintained by hand; with JAX we get four options out of the box, each with a different cost / memory profile.

| Method | When it applies | Cost | Memory | JAX call |
|---|---|---|---|---|
| **Discrete adjoint** (reverse-mode AD) | $H$ is a JAX program; one-shot forward pass | $\mathcal{O}(\text{forward})$ | $\mathcal{O}(\text{forward tape})$ | `jax.grad(loss)` |
| **Tangent-linear** (forward-mode AD) | Sensitivities, ensemble propagation | $\mathcal{O}(\text{forward}) \times \dim(\text{input})$ | $\mathcal{O}(\text{forward})$ | `jax.jvp(H, (x,), (v,))` |
| **Implicit-function adjoint** | Differentiating *through* a converged inner solve | $\mathcal{O}(1\,\text{linear solve})$ | $\mathcal{O}(\text{linear solve})$ | `optimistix.ImplicitAdjoint(linear_solver=...)` |
| **Checkpointed adjoint** | Memory-bounded long unrolls (ODE / iterative) | $\mathcal{O}(\text{forward} \log n)$ | $\mathcal{O}(\sqrt{n})$ | `optimistix.RecursiveCheckpointAdjoint(...)` |

### 4.1 Discrete adjoint (`jax.grad`)

This is what we use for the obs-term gradient in [eq-J](#eq-J). Concretely, [`build_cost_xi`](../../src/plume_simulation/assimilation/cost.py) wraps the cost in `jax.value_and_grad(value)`; reverse-mode AD walks back through `forward_fn`, applying each elementary VJP — including the kernel-flipped convolution adjoint inside [`PointSpreadFunction`](../../src/plume_simulation/radtran/instrument.py) and the spread-with-1/f² adjoint inside [`GroundSamplingDistance`](../../src/plume_simulation/radtran/instrument.py). The unit tests in [test_instrument.py](../../tests/radtran/test_instrument.py) verify the inner-product identity $\langle A x, u \rangle = \langle x, A^\top u \rangle$ for both operators, so the autodiff gradient is provably equal to what a hand-rolled adjoint would give.

### 4.2 Tangent-linear (`jax.jvp`)

`jax.jvp(H, (x,), (v,))` computes $H'(x) \cdot v$ in one forward pass without ever forming the Jacobian. Useful for:
- The dual matvec inside [eq-dual-solution](#eq-dual-solution): one `jvp` per CG iteration (we use this implicitly via `forward_linear_fn` in [`run_dual_psas`](../../src/plume_simulation/assimilation/solve.py)).
- Hessian-vector products via forward-over-reverse: `jax.jvp(jax.grad(J), (x,), (v,))` returns $\nabla^2 J \cdot v$ at $\mathcal{O}(\text{forward})$ cost. This is what [`Cost.hvp`](../../src/plume_simulation/assimilation/cost.py) does, and it's what feeds the diagnostics in [diagnostics.py](../../src/plume_simulation/assimilation/diagnostics.py).

### 4.3 Implicit-function adjoint

Suppose we want $\partial \hat x / \partial \theta$ — the derivative of the converged 3D-Var solution with respect to a hyperparameter $\theta$ (e.g., a regularisation weight, an obs-error variance, a prior length-scale). Naively differentiating through the LBFGS unroll would be both expensive and numerically suspect.

The implicit-function theorem gives a closed-form. The optimum satisfies $F(\hat x, \theta) = \nabla_x J(\hat x, \theta) = 0$. Differentiating implicitly,

$$
\frac{\partial \hat x}{\partial \theta} \;=\; -\bigl(\nabla_x F\bigr)^{-1}\, \nabla_\theta F,
$$ (eq-implicit-adjoint)

so we just need to solve **one** linear system in the Hessian $\nabla_x F$. This is what `optimistix.ImplicitAdjoint(linear_solver=lineax.CG(...))` does internally; you write a normal `optimistix.minimise(...)` call, and gradients with respect to anything captured by the cost flow back through this single adjoint solve. No tape, no checkpoints.

> **In code.** All of our solvers in [solve.py](../../src/plume_simulation/assimilation/solve.py) accept the optimistix default `adjoint=ImplicitAdjoint()`; you can override it with a custom `linear_solver` (e.g., `lineax.CG(rtol=...)` for matrix-free, or `lineax.Cholesky()` for small dense Hessians) when you start computing sensitivities of the analysis with respect to its inputs.

### 4.4 Checkpointed adjoint

When the forward map is itself an iterative procedure (e.g., a chemistry solver inside the obs operator, or a Kalman-smoother prior), the reverse-mode tape can blow up memory. `optimistix.RecursiveCheckpointAdjoint(checkpoints=k)` does $\sqrt{n}$-style binomial checkpointing: only $k$ forward states are stored, and the rest are recomputed on the backward pass. We don't currently need this — the methane forward map is a single fused convolution chain — but it's the right tool the moment you couple the retrieval to a dynamical model.

## 5. Mapping math → libraries

| Math object                                | Library call                                             |
|---|---|
| $B$ (Kronecker / low-rank / diagonal)      | `gaussx.Kronecker`, `gaussx.LowRankUpdate`, `lineax.DiagonalLinearOperator` |
| $B^{-1} \delta x$                          | `gaussx.solve(B, δx)` (structured dispatch)              |
| $U$ such that $B = U U^\top$               | `gaussx.cholesky(B)`                                     |
| $H$ (forward op, JAX)                      | `RadianceObservationModel.forward(...)`                  |
| $H' \cdot v$ (tangent-linear)              | `jax.jvp(forward_fn, (x,), (v,))`                        |
| $H'^\top \cdot u$ (adjoint)                | `jax.vjp(forward_fn, x)[1](u)` — no manual adjoint       |
| $\nabla J$ (discrete adjoint)              | `jax.grad(J)`                                            |
| $\nabla^2 J \cdot v$ (Hessian-vector)      | `jax.jvp(jax.grad(J), (x,), (v,))`                       |
| Outer minimiser (whitened LBFGS)           | `optimistix.LBFGS(...)` via `optimistix.minimise(...)`   |
| Gauss-Newton with custom inner solver      | `optimistix.GaussNewton(...)` + `lineax.CG(...)` for $J^\top J$ |
| Dual / PSAS solve                          | `lineax.linear_solve(dual_op, d, solver=lineax.CG(...))` |
| Adjoint *through* the converged solve      | `optimistix.ImplicitAdjoint(linear_solver=lineax.CG(...))` |
| $\chi^2_\text{red}$, DFS, posterior cov    | [`reduced_chi_squared`, `degrees_of_freedom_for_signal`, `posterior_covariance_proxy`](../../src/plume_simulation/assimilation/diagnostics.py) |

## 6. Worked example

The end-to-end demo lives in [06_3dvar_methane_retrieval.ipynb](06_3dvar_methane_retrieval.ipynb). It builds a synthetic plume from the HAPI LUT, injects it via the JAX forward operator, and runs the retrieval three ways — naive primal, whitened LBFGS, and dual PSAS — comparing iteration counts, run times, and posterior diagnostics. The same notebook also shows the matched filter from PR #24 falling out of the dual recipe with a single CG iteration.
