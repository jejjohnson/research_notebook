---
title: "Matched filter — derivation, covariance flavours, and the link to 3D-Var"
---

# Matched filter for hyperspectral methane retrieval

The matched filter (MF) is the maximum-SNR linear detector for a known target signature in additive Gaussian noise. In hyperspectral methane retrieval it is the workhorse baseline — fast, closed-form, and surprisingly competitive with full variational retrievals when the signal is weak and the scene is statistically homogeneous. This note derives the MF from scratch, lays out the four covariance flavours implemented in [plume_simulation.matched_filter](../../src/plume_simulation/matched_filter/), and shows the exact sense in which the MF is the **one-step limit of the dual / PSAS 3D-Var solver** from the companion [`00_3dvar_derivation.md`](../assimilation/00_3dvar_derivation.md).

The derivation borrows notation from Funk et al. [^funk2001], Theiler & Foy [^theiler2006], and Thompson et al. [^thompson2015]; the operator-based implementation mirrors the structured-linear-algebra view of Golub & Van Loan [^gvl2013] and the Woodbury-friendly design in [`gaussx`](https://github.com/jejjohnson/gaussx).

[^funk2001]: Funk, C. C., Theiler, J., Roberts, D. A., & Borel, C. C. (2001). Clustering to improve matched filter detection of weak gas plumes in hyperspectral thermal imagery. *IEEE TGRS*, 39(7).

[^theiler2006]: Theiler, J. & Foy, B. R. (2006). Effect of signal contamination in matched-filter detection of the signal on a cluttered background. *IEEE GRSL*, 3(1).

[^thompson2015]: Thompson, D. R., et al. (2015). Real-time remote detection and measurement for airborne imaging spectroscopy: a case study with methane. *AMT*, 8.

[^gvl2013]: Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins.

## 1. The signal model

We observe a per-pixel radiance spectrum $x \in \mathbb{R}^B$ (one vector per pixel of the scene, with $B$ spectral bands). The model is

$$
x \;=\; \mu \;+\; \alpha\, t \;+\; \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \Sigma),
$$ (eq-signal-model)

where

- $\mu \in \mathbb{R}^B$ is the **background mean spectrum** (what you would observe at a pixel free of plume enhancement),
- $t \in \mathbb{R}^B$ is the **target signature** (the spectral shape that a unit VMR enhancement would leave in $x$),
- $\alpha \ge 0$ is the **plume amplitude** (the unknown — pixels with $\alpha = 0$ are background, $\alpha > 0$ is a plume),
- $\Sigma \in \mathbb{R}^{B \times B}$ is the background clutter covariance (noise + surface heterogeneity + calibration residual), assumed Gaussian.

Under this model, the log-likelihood of a pixel is, up to additive constants,

$$
\ell(\alpha \mid x) \;=\; -\tfrac{1}{2}\,(x - \mu - \alpha t)^\top \Sigma^{-1} (x - \mu - \alpha t).
$$ (eq-loglik)

Setting $\partial \ell / \partial \alpha = 0$ gives the maximum-likelihood estimator of the amplitude:

$$
\boxed{\;\hat\alpha(x) \;=\; \frac{(x - \mu)^\top \Sigma^{-1} t}{t^\top \Sigma^{-1} t}\;}
$$ (eq-mf)

This is the **matched filter**. It is unbiased ($\mathbb{E}[\hat\alpha] = \alpha$), has the minimum variance among all unbiased linear estimators (Gauss–Markov), and maximises SNR in the Neyman–Pearson sense for detecting the alternative $\alpha > 0$ against the null $\alpha = 0$.

### 1.1 The SNR and detection threshold

Under the null ($\alpha = 0$) the score $\hat\alpha(x)$ is $\mathcal{N}(0, 1/(t^\top \Sigma^{-1} t))$. Under an alternative with amplitude $\alpha$ it is $\mathcal{N}(\alpha, 1/(t^\top \Sigma^{-1} t))$. So the **detection SNR** is

$$
\mathrm{SNR}(\alpha) \;=\; \alpha \sqrt{t^\top \Sigma^{-1} t},
$$ (eq-snr)

and the Gaussian detection threshold for a false-alarm rate (FAR) is

$$
\tau_{\mathrm{FAR}} \;=\; \frac{\Phi^{-1}(1 - \mathrm{FAR})}{\sqrt{t^\top \Sigma^{-1} t}}.
$$ (eq-threshold)

Both are implemented in [`matched_filter.core`](../../src/plume_simulation/matched_filter/core.py) as `matched_filter_snr` and `detection_threshold`.

## 2. Practical decomposition: precompute once, apply per pixel

Equation (5) is deceptively cheap. Per pixel, it's a dot product of length $B$ — except $\Sigma^{-1}$ appears. Done naively, you invert a $B \times B$ matrix once, but that discards the chance to exploit any structure in $\Sigma$.

The right decomposition is

$$
w \;:=\; \Sigma^{-1} t, \qquad z \;:=\; t^\top w \;=\; t^\top \Sigma^{-1} t.
$$ (eq-precompute)

Then per pixel

$$
\hat\alpha(x) \;=\; \frac{(x - \mu) \cdot w}{z}
$$ (eq-mf-apply)

— one scalar solve and one scalar, followed by one dot product per pixel (a BLAS call when vectorised over the image). This is exactly what `apply_image` in `core.py` does:

```python
def apply_image(cube, mean, cov_op, target):
    w = gx.solve(cov_op, target)      # ← Σ⁻¹ t, one solve
    z = jnp.dot(target, w)             # ← scalar
    residual = cube - mean             # ← broadcast
    scores = einops.einsum(residual, w, "H W B, B -> H W")
    return scores / z
```

The key move is `gx.solve(cov_op, target)`. Because `cov_op` is an abstract `lineax.AbstractLinearOperator`, the solve dispatches on structure: a dense operator triggers a Cholesky, a diagonal operator triggers a per-element divide, and a low-rank update `λI + V D V^\top` triggers Woodbury — with no change to the MF kernel above.

## 3. Four flavours of $\Sigma$

The four reference files in `jej_vc_snippets/methane_retrieval/` originally wrote out four separate implementations of equation (5), one per covariance model. In the operator-based view they collapse into four *estimation* routines that all return the same type — a `lineax.AbstractLinearOperator`.

:::{list-table} Covariance estimators and the operators they produce.
:header-rows: 1
:widths: 20 30 30 20

* - Estimator
  - Formula
  - Operator type
  - Per-pixel cost
* - `estimate_cov_empirical`
  - $\hat\Sigma = \frac{1}{N}\sum_i (x_i - \mu)(x_i - \mu)^\top$
  - `MatrixLinearOperator` (dense)
  - $\mathcal{O}(B^2)$ once, then $\mathcal{O}(B)$
* - `estimate_cov_shrunk` (Ledoit–Wolf, OAS)
  - $\hat\Sigma_\lambda = (1-\lambda)\hat\Sigma + \lambda F$
  - `MatrixLinearOperator` (dense, PD)
  - same
* - `estimate_cov_lowrank`
  - $\hat\Sigma = \tau I + V D V^\top$, $V$ top-$k$ SVD
  - `gaussx.LowRankUpdate` (Woodbury)
  - $\mathcal{O}(B \cdot k + k^3)$ once, then $\mathcal{O}(B)$
* - `gmm_cluster_background`
  - $(\mu_c, \Sigma_c)$ per GMM cluster $c$
  - List of operators above, per cluster
  - per-cluster amortised
:::

The Ledoit–Wolf and OAS estimators come from `sklearn.covariance`; the randomised low-rank SVD comes from `sklearn.decomposition.TruncatedSVD`. Because MF estimation is off the JAX gradient path — the background is treated as a fixed input — we lean on scikit-learn freely. Only the per-pixel *application* stays in JAX, vectorised with `jax.vmap` / `einsum` for GPU-friendly batching.

### 3.1 Why low-rank + Tikhonov matters

For $n_{\text{samples}} \gg B$ and a homogeneous scene, the spectral covariance is approximately low-rank: a handful of surface-reflectance PCs explain most of the variance. Writing $\Sigma = \tau I + V D V^\top$ with $V \in \mathbb{R}^{B \times k}$ and $D \in \mathbb{R}^{k \times k}$ diagonal lets us apply Woodbury:

$$
\Sigma^{-1} \;=\; \tau^{-1} I \;-\; \tau^{-1} V \bigl(D^{-1} + \tau^{-1} V^\top V\bigr)^{-1} V^\top \tau^{-1}.
$$ (eq-woodbury)

The inner inverse is $k \times k$, so $w = \Sigma^{-1} t$ costs $\mathcal{O}(B k + k^3)$ instead of $\mathcal{O}(B^3)$. For a typical hyperspectral methane setup ($B = 400$, $k = 16$), this is a ~1000× speedup on the MF precompute — visible to the user only in that [`estimate_cov_lowrank`](../../src/plume_simulation/matched_filter/background.py) returns a `gaussx.LowRankUpdate` instead of a dense operator.

## 4. Relationship to 3D-Var

3D-Var minimises (see [`00_3dvar_derivation.md`](../assimilation/00_3dvar_derivation.md), eq. 2):

$$
J(\delta x) \;=\; \tfrac{1}{2}\, \delta x^\top B^{-1} \delta x \;+\; \tfrac{1}{2}\,(d - H' \delta x)^\top R^{-1} (d - H' \delta x),
$$ (eq-3dvar)

where $d = y - H(x_b)$ is the innovation and $H'$ is the tangent-linear of the forward operator. The PSAS / dual path solves in observation space:

$$
(H' B H'^\top + R) \lambda \;=\; d, \qquad \hat{\delta x} \;=\; B H'^\top \lambda.
$$ (eq-psas)

**The MF is exactly the PSAS analysis with a flat (infinite-variance) prior and identity state-space coupling.**

Take $B = \sigma_B^2 I$ with $\sigma_B \to \infty$, $R = \Sigma$, and $H' \delta x = \alpha t$ (the tangent-linear sees a unit plume amplitude as the target signature). Then equation (13) becomes

$$
(\sigma_B^2 t t^\top + \Sigma)\, \lambda = d \;\;\Rightarrow\;\; \lambda = \Sigma^{-1} d - \frac{\Sigma^{-1} t t^\top \Sigma^{-1}}{\sigma_B^{-2} + t^\top \Sigma^{-1} t}\, d.
$$

As $\sigma_B \to \infty$ the second term saturates, and a short calculation gives

$$
\hat{\delta x}_{\mathrm{PSAS}} \;=\; \underbrace{\frac{d^\top \Sigma^{-1} t}{t^\top \Sigma^{-1} t}}_{=\,\hat\alpha_{\mathrm{MF}}(x = \mu + d)}\, t.
$$

So running a **single CG iteration** of [`run_dual_psas`](../../src/plume_simulation/assimilation/solve.py) with the flat prior and linearised $H'$ gives the MF score as a by-product. The MF is therefore *not* a heuristic: it is the exact, closed-form, flat-prior dual 3D-Var solution, and the SNR / threshold formulas follow without further work.

The corollary is that when $B$ is *not* flat — i.e. when we have real prior information about the VMR field (spatial smoothness, a climatological mean, etc.) — the full 3D-Var retrieval strictly dominates the MF. See notebook `01_mf_vs_3dvar.ipynb` for a numerical comparison on the same scene.

## 5. Where the target signature $t$ comes from

For methane, $t(\nu)$ is the spectral signature of a unit VMR enhancement. From the Beer–Lambert forward model $L(\nu) = L_0(\nu)\,\exp(-a(\nu)\,\Delta\mathrm{VMR})$, the Taylor expansion at the background state gives

$$
t(\nu) \;=\; \frac{\partial L}{\partial (\Delta \mathrm{VMR})}\Big|_{\Delta\mathrm{VMR}=0} \;=\; -L_0(\nu) \cdot a(\nu).
$$ (eq-target)

Rather than hard-coding this, we take the **JVP of the existing observation operator**:

```python
t = linear_target_from_obs(obs, vmr_background, pattern="uniform")
#   internally: jax.jvp(obs.forward, (vmr_background,), (ones,))
```

This guarantees the target is exactly consistent with whatever forward model (PSF, GSD, SRF) the retrieval uses — no drift between "target generator" and "forward model" implementations. For strong plumes where Beer–Lambert saturation matters, `nonlinear_target_from_obs(obs, x_b, amplitude=α)` computes the exact finite-amplitude response $L(x_b + \alpha\,\delta x) - L(x_b)$ instead.

## 6. Summary

The pieces of [`plume_simulation.matched_filter`](../../src/plume_simulation/matched_filter/):

:::{list-table}
:header-rows: 1
:widths: 25 75

* - Module
  - Role
* - [`core.py`](../../src/plume_simulation/matched_filter/core.py)
  - `apply_pixel`, `apply_image`, `matched_filter_snr`, `detection_threshold` — equation (5) + equations (6)–(7). Dispatches on `cov_op` structure via `gaussx.solve`.
* - [`target.py`](../../src/plume_simulation/matched_filter/target.py)
  - `linear_target_from_obs` (JVP) and `nonlinear_target_from_obs` (finite-amplitude).
* - [`background.py`](../../src/plume_simulation/matched_filter/background.py)
  - `estimate_mean` (robust variants), `estimate_cov_{empirical,shrunk,lowrank}` — sklearn in, `lineax`/`gaussx` operators out.
* - [`cluster.py`](../../src/plume_simulation/matched_filter/cluster.py)
  - `gmm_cluster_background`, `adaptive_window_background` — spatially varying $(\mu, \Sigma)$ when the scene is heterogeneous.
* - [`streaming.py`](../../src/plume_simulation/matched_filter/streaming.py)
  - `WelfordAccumulator`, `streaming_background` — for multi-scene / multi-file background aggregation.
* - [`io.py`](../../src/plume_simulation/matched_filter/io.py)
  - `apply_image_xarray`, `open_multi_scene` — xarray glue (dimensional metadata, Dask chunks, `open_mfdataset`).
:::

The two companion notebooks:
- [`01_mf_vs_3dvar.ipynb`](01_mf_vs_3dvar.ipynb) — MF versus full 3D-Var on the same HAPI methane scene used in the assimilation tutorial, so you can see where the flat-prior MF gives up information.
- [`02_covariance_flavours.ipynb`](02_covariance_flavours.ipynb) — dense vs shrunk vs low-rank vs GMM-clustered $\Sigma$ on a heterogeneous scene, comparing detection power and runtime.
