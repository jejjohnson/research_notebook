---
title: "Variational SSH reconstruction — 3D-Var, 4D-Var, DYMOST, BFN-QG, 4DVarNet"
---

# Variational reconstruction with dynamical priors

The pipeline in [00](00_ssh_pathwise_sampling.md)–[03](03_global_scaling_patches.md) is a **pathwise Gaussian process** with a static, isotropic prior covariance encoded by a kernel. That works because SSH variability has a known length scale and the inverse problem is mostly one of *interpolation* — fill in the gaps between satellite tracks under a smoothness assumption. It stops working when the dominant signal is *advection* by mesoscale eddies: the field at $t+\tau$ is not a smoothed version of the field at $t-\tau$, it is a *transported* version of it. A static kernel cannot represent that.

This is where data assimilation (DA) takes over. The DA literature has spent fifty years on exactly this problem — combining sparse observations with a dynamical model to estimate an evolving state — and the SSH community has built a tower of methods on it: classical 3D-Var (≈ what we just did), 4D-Var (the time-evolving version), DYMOST (4D-Var with a quasi-geostrophic propagator), BFN-QG (a forward/backward nudging shortcut around the adjoint), and 4DVarNet (a learned variational scheme). All of these *replace* the GP prior with something that knows about ocean physics, so they live outside the gaussx pathwise stack — but they are the right comparison targets, and one of them is likely the operational successor.

This note frames the math, situates each method in the same picture, and gives concrete pointers to the local code that implements them ([`jej_vc_snippets/quasigeostrophic_model/massh/`](file:///home/azureuser/localfiles/jej_vc_snippets/quasigeostrophic_model/massh/), [`jej_vc_snippets/4dvar/`](file:///home/azureuser/localfiles/jej_vc_snippets/4dvar/)) and to the existing 3D-Var derivation in the plume-simulation project[^pl3dvar]. The aim is *navigational* — you should be able to read this side by side with the SSH-mapping data-challenge papers and see exactly which equation each algorithm is solving.

[^pl3dvar]: A complementary 3D-Var derivation aimed at hyperspectral methane retrieval lives at [`projects/plume_simulation/notebooks/assimilation/00_3dvar_derivation.md`](../../plume_simulation/notebooks/assimilation/00_3dvar_derivation.md) — same math, different application. We borrow its preconditioning and dual-form discussion when relevant.

---

## 1. Variational data assimilation in one paragraph

The *analysis* state $\hat x \in \mathbb{R}^N$ is the MAP estimate under a Gaussian prior $x \sim \mathcal{N}(x_b, B)$ and a Gaussian likelihood $y \mid x \sim \mathcal{N}(\mathcal{H}(x), R)$:

$$
\hat x \;=\; \arg\min_{x}\;
\underbrace{\tfrac{1}{2}\,(x - x_b)^\top B^{-1} (x - x_b)}_{\mathcal{J}_{b}(x)\ \text{— background term}}
\;+\;
\underbrace{\tfrac{1}{2}\,\bigl(y - \mathcal{H}(x)\bigr)^\top R^{-1} \bigl(y - \mathcal{H}(x)\bigr)}_{\mathcal{J}_{o}(x)\ \text{— observation term}}.
$$ (eq-3dvar)

The *only* dependence on the analysis problem at hand is in the choice of $(x_b,\, B,\, \mathcal{H},\, R)$. Everything below is a pattern of choices.

| Quantity | Static-GP / 3D-Var | 4D-Var | DYMOST | BFN-QG | 4DVarNet |
|---|---|---|---|---|---|
| State $x$ | SSH field at $t$ | SSH field at $t_0$, propagated by a model | SSH field along Lagrangian particles | Trajectory across window | SSH field across window |
| Prior mean $x_b$ | 0 (anomaly) or DUACS | 0 or previous analysis | model trajectory | model trajectory | 0 or previous analysis |
| Prior cov $B$ | Matérn × OU kernel | $B$ at $t_0$, propagated by model TLM | static, in propagated frame | implicit (nudging coefficient) | learned $\\Phi_\\theta$ |
| Forward $\mathcal{H}$ | linear (sample at obs locations) | $\mathcal{H} \circ M^t$ — model-then-sample | $\mathcal{H} \circ M_{\text{QG}}$ | $\mathcal{H} \circ M_{\text{QG}}$ | $\mathcal{H}$ (sample) |
| Solver | dense / CG / Woodbury | adjoint + LBFGS | adjoint + LBFGS | forward/backward nudging — no adjoint | learned ConvLSTM grad. modulator |
| Code | gaussx 00/01/03 | massh `inversion.py` | massh `model.py` + `inversion.py` | massh `model.py` (BFN loop) | `fourdvarnet/` |

The rest of the note works through each row.

---

## 2. 3D-Var ≡ static-GP optimal interpolation (sanity check)

The pathwise-GP pipeline from 00–03 *is* 3D-Var. Setting

- $x = \eta(\mathcal{X}^*)$ — the SSH field on the prediction grid,
- $x_b = 0$ — anomaly convention,
- $B = K_{\mathcal{X}^* \mathcal{X}^*}$ — the GP prior covariance on the grid,
- $\mathcal{H}$ = linear interpolation from grid to observation locations,
- $R = \sigma_{obs}^2 I$,

the 3D-Var cost [eq-3dvar](#eq-3dvar) becomes

$$
\mathcal{J}(\eta) \;=\; \tfrac{1}{2}\,\eta^\top K_{\mathcal{X}^*\mathcal{X}^*}^{-1}\, \eta \;+\; \tfrac{1}{2\sigma_{obs}^2}\,\bigl(y - \mathcal{H}\eta\bigr)^\top \bigl(y - \mathcal{H}\eta\bigr).
$$

Setting $\nabla \mathcal{J} = 0$ yields, after the matrix-inversion lemma and a little reorganisation, the GP posterior mean from 00:

$$
\hat\eta \;=\; K_{\mathcal{X}^*\mathcal{X}}\,(K_{\mathcal{X}\mathcal{X}} + \sigma_{obs}^2 I)^{-1}\, y \;=\; \mu_{f|y}.
$$

This is the **dual** form of 3D-Var (PSAS) and matches the GP analytical posterior exactly — it is one of those "the same algorithm with two names" results that recurs in DA every five years[^bc1999] [^lorenc2000]. **The static-GP pipeline you already have is already 3D-Var.** Calling it "GP" or "OI" or "3D-Var" only changes which jargon you use to describe its components; the equations are identical.

The reason the GP framing wins for the static case is that the kernel encodes $B$ implicitly via its closed-form covariance function — you never have to *store* $B \in \mathbb{R}^{n \times n}$, you just call `gx.ImplicitKernelOperator`. The DA literature spent decades wrestling with structured representations of $B$ (digital filters[^purser2003], NMC method[^parrish1992], wavelet $B$[^berre2010]); GP kernels solve it for free.

---

## 3. 4D-Var — propagating the analysis through time

3D-Var assumes the field is approximately stationary over the analysis window. When $\tau \cdot u_{\text{eddy}}$ approaches a deformation radius — i.e. an eddy moves by its own diameter during the temporal window — that assumption fails. **4D-Var** fixes it by adding a deterministic dynamical model $M$ to the cost:

$$
\hat x_0 \;=\; \arg\min_{x_0}\;
\tfrac{1}{2}\,(x_0 - x_b)^\top B^{-1} (x_0 - x_b)
\;+\;
\tfrac{1}{2}\sum_{i=0}^{T} \bigl(y_i - \mathcal{H}(M^i x_0)\bigr)^\top R_i^{-1} \bigl(y_i - \mathcal{H}(M^i x_0)\bigr).
$$ (eq-4dvar)

The control variable is the **initial state $x_0$**. The model $M$ propagates it through the window; observations at each time $t_i$ are compared to $\mathcal{H}(M^i x_0)$. Two things change vs. 3D-Var:

- **The prior $B$ now propagates implicitly along $M$**: the effective prior at time $t_i$ is $M^i B (M^i)^\top$. So an isotropic Gaussian prior at $t_0$ becomes an anisotropic, eddy-stretched prior at $t_i$ — exactly what the static GP cannot represent.
- **The cost gradient requires the adjoint $M^\top$**: backpropagating the obs misfit through the model. Hand-coded adjoints used to be a major engineering project (see ECMWF's IFS); JAX does it for you with `jax.grad` provided $M$ is differentiable.

The minimisation is iterative — typically incremental 4D-Var: linearise $M$ and $\mathcal{H}$ around the current trajectory (the "outer loop"), solve the resulting quadratic problem (the "inner loop") with PCG-LBFGS, update the trajectory, repeat[^bc1999]. The control-variable transform from [`plume_simulation/.../00_3dvar_derivation.md` §2`](../../plume_simulation/notebooks/assimilation/00_3dvar_derivation.md) carries over directly: pre-conditioning by $U$ with $B = U U^\top$ collapses the prior to a sphere and makes LBFGS converge in ~10 iterations.

> **In code.** [`jej_vc_snippets/4dvar/src/fourdvarnet/costs.py`](file:///home/azureuser/localfiles/jej_vc_snippets/4dvar/src/fourdvarnet/costs.py) builds [eq-4dvar](#eq-4dvar) for the `(B, T, H, W)` 2-D layout; the gradient is taken via `jax.grad` so the adjoint is implicit. `solver.py` runs the inner-loop LBFGS.

The catch with 4D-Var is the model $M$. For SSH, the relevant model is some flavour of **quasi-geostrophic** dynamics — the natural mid-latitude approximation when the Rossby number is small.

---

## 4. The 1.5-layer QG model — the workhorse propagator for SSH

The simplest model that captures mesoscale eddy advection is the **1.5-layer quasi-geostrophic** equation: an active surface layer over a deep, motionless layer. The SSH $\eta$ acts as a streamfunction (up to the Coriolis factor $g/f$); potential vorticity $q$ is materially conserved:

$$
q \;=\; \nabla^2 \psi \;-\; \frac{1}{L_R^2}\,\psi, \qquad \psi = g \eta / f,
$$ (eq-pv)

$$
\partial_t q \;+\; J(\psi, q) \;=\; 0, \qquad J(\psi, q) = \partial_x \psi \cdot \partial_y q - \partial_y \psi \cdot \partial_x q.
$$ (eq-qg)

$L_R$ is the first baroclinic Rossby radius (~50 km mid-latitude, ~10 km at high latitudes). The Jacobian $J(\psi, q)$ is the **eddy advection** operator — this is what the static GP cannot express. Numerically: solve the elliptic [eq-pv](#eq-pv) with a Helmholtz solver to get $\psi$ from $q$, advect $q$ along the velocity $(-\partial_y \psi, \partial_x \psi)$ with an upwind or Arakawa scheme.

> **In code.** The user's local [`jej_vc_snippets/quasigeostrophic_model/mqg/qgm.py`](file:///home/azureuser/localfiles/jej_vc_snippets/quasigeostrophic_model/mqg/qgm.py) and [`mqg/helmholtz.py`](file:///home/azureuser/localfiles/jej_vc_snippets/quasigeostrophic_model/mqg/helmholtz.py) implement the elliptic solve and Arakawa-Jacobian advection in JAX (differentiable). [`mqg/example_natl.py`](file:///home/azureuser/localfiles/jej_vc_snippets/quasigeostrophic_model/mqg/example_natl.py) runs it on a North Atlantic configuration. This is the same numerical core as the MASSH propagator, just packaged separately.

---

## 5. DYMOST — 4D-Var in particle-following coordinates

DYMOST (Ubelmann, Klein, Fu)[^ubelmann2015] [^ubelmann2020] is a **4D-Var with the 1.5-layer QG model as $M$**, but with a clever simplification: instead of running a full incremental-4D-Var loop, propagate the analysed SSH backward and forward along [eq-qg](#eq-qg) and perform OI in a Lagrangian, *particle-following* frame. In that frame the field looks closer to stationary — the QG flow has done most of the work of "straightening out" the eddies — so a single 3D-Var step on the propagated residual gives a near-optimal answer.

Schematically:

```
For each obs time t_i in the window:
  η_propagated_i  =  M^{i - t_target}  η_target            # propagate analysis to obs time
  innovation_i    =  y_i - H(η_propagated_i)               # residual in observation space
Combine innovations with a static OI to update η_target.
```

The whole window thus collapses to one 3D-Var solve (the static-GP machinery from 00/01) on the QG-propagated residuals. **Cost: ~1× a QG forward + 1× backward integration over the window, plus one GP solve.** No adjoint, because the OI step is linear and the QG propagation only enters as a forward operator.

Performance: 30–60% more SSH variance recovered in the western boundary currents (Gulf Stream, Kuroshio) over static OI[^ubelmann2020] — exactly where eddy advection dominates.

---

## 6. BFN-QG — the adjoint-free 4D-Var

Even DYMOST simplifies the adjoint problem; full 4D-Var still needs $M^\top$. Le Guillou et al.[^leguillou2021] [^leguillou2023] propose **back-and-forth nudging**: integrate the QG model forward over the window with a Newtonian relaxation toward observations, then integrate backward with the same relaxation, iterate until both passes agree. The relaxation replaces the gradient descent on the cost — no adjoint, no covariance, no LBFGS.

Forward pass:
$$
\partial_t \psi \;=\; -J(\psi, q) \;-\; \kappa \cdot \mathbb{1}_{\text{obs}} \cdot (\mathcal{H}\psi - y),
$$
where $\kappa$ is the nudging coefficient (the only hyperparameter) and $\mathbb{1}_{\text{obs}}$ is the indicator at observation locations + times. Backward pass: same equation integrated from $t_0 + T$ down to $t_0$ with reversed time-derivative sign.

Iterate the forward/backward pair $\sim 5$–$10$ times. The final analysis is the forward trajectory at the iteration where forward/backward states agree to within a tolerance.

Pros: **no adjoint, no $B^{-1}$, embarrassingly simple to code** (three loops). Cons: the implicit error covariance is whatever fixed point the nudging settles to — no posterior uncertainty out of the box, no principled way to tune $\kappa$ except by cross-validation. Despite that, BFN-QG matches incremental 4D-Var on the SWOT data challenges[^leguillou2023] in energetic regions, making it the highest-bang-per-buck dynamical method on the menu.

> **In code.** [`jej_vc_snippets/quasigeostrophic_model/massh/`](file:///home/azureuser/localfiles/jej_vc_snippets/quasigeostrophic_model/massh/) is the reference open implementation by Le Guillou. `massh/model.py` runs the QG dynamics; `massh/4dvar.py` and `massh/inversion.py` provide both the BFN loop and a true incremental 4D-Var path. Upstream is [github.com/leguillf/MASSH](https://github.com/leguillf/MASSH).

---

## 7. 4DVarNet — learn the prior, learn the solver

The catch with all of §3–§6 is that the QG model is *only* a model — it captures geostrophic eddy advection, but misses internal tides, ageostrophic motions, submesoscale frontogenesis, and a long tail of finer physics. Hand-extending QG with each missing piece is open-ended. **4DVarNet** (Fablet, Beauchamp et al.)[^beauchamp2023] [^febvre2024] takes a different route: keep the variational *form* of the cost, but replace the hand-crafted prior with a learned regulariser, and replace the gradient solver itself with a learned one.

Cost:
$$
\hat x \;=\; \arg\min_{x}\;
\underbrace{\bigl\|\mathcal{H} x - y\bigr\|^2_{R^{-1}}}_{\mathcal{J}_o\,\text{— observation}}
\;+\;
\underbrace{\bigl\|x - \Phi_\theta(x)\bigr\|^2}_{\mathcal{J}_p\,\text{— learned prior}}.
$$ (eq-4dvarnet-cost)

$\Phi_\theta$ is a neural autoencoder (typically a bilinear convolutional AE on the spatiotemporal patch) trained as a fixed point on plausible SSH fields: $\Phi_\theta(x_{\text{plausible}}) \approx x_{\text{plausible}}$. The prior cost $\|x - \Phi_\theta(x)\|^2$ is then small whenever $x$ is in-distribution and large otherwise — a *learned* characterisation of "plausible SSH."

Learned solver:
$$
x_{k+1} \;=\; x_k \;-\; \alpha_k \cdot \Psi_\omega\bigl(\nabla \mathcal{J}(x_k)\bigr) \;-\; \beta_k \cdot \nabla \mathcal{J}(x_k),
$$ (eq-4dvarnet-solver)

with $\Psi_\omega$ a ConvLSTM "gradient modulator" that learns a state-dependent preconditioner from training data. Training is end-to-end: backpropagate through $K$ unrolled solver iterations to update $(\theta, \omega)$, with implicit differentiation for memory efficiency.

> **In code.** [`jej_vc_snippets/4dvar/src/fourdvarnet/`](file:///home/azureuser/localfiles/jej_vc_snippets/4dvar/src/fourdvarnet/): `priors.py` (autoencoder $\Phi_\theta$), `grad_mod.py` (ConvLSTM $\Psi_\omega$), `solver.py` (the unrolled solver [eq-4dvarnet-solver](#eq-4dvarnet-solver)), `model.py` (`FourDVarNet1D`/`2D` putting it all together), `training.py` (end-to-end fit). The full math reference is in `jej_vc_snippets/4dvar/docs/`.

Performance: 4DVarNet trained on simulation (NATL60, eNATL60) and applied to real SWOT observations beats DUACS / MIOST / BFN-QG by another 5–15% RMSE on the open ocean[^febvre2024], with no further tuning. The price is a training stage that requires a high-resolution simulation ground truth — which is fine for SSH (multiple GCMs available) but expensive to repeat for each new region.

---

## 8. Where this leaves the GP pathwise pipeline

These methods don't *replace* the GP pipeline so much as **sit alongside it at different points on the cost-vs-physics curve**:

| Method | Captures eddy advection? | Cost vs static GP | Posterior uncertainty? | Code |
|---|---|---|---|---|
| Static-GP 3D-Var (00/01/03) | No | 1× | Yes (pathwise samples) | gaussx pathwise stack |
| DYMOST | Yes (linearised) | 3–5× | Yes (around the QG trajectory) | `mqg/`, `massh/` |
| BFN-QG | Yes | 5–10× | No (nudging fixed point) | `massh/` |
| Incremental 4D-Var (full QG) | Yes | 10–20× | Yes (Hessian-based) | `massh/`, `4dvar/` |
| 4DVarNet | Yes (learned) | 1–2× *after training* (training is one-shot) | No (point estimate) | `4dvar/` |

Two practical hybrid recipes:

- **GP residual after a dynamical first guess.** Run BFN-QG or DYMOST to get a $\hat\eta_{\text{dyn}}$; subtract it from the observations; feed the residual through the static GP from 00/01 to model whatever physics the QG missed (internal tides, ageostrophic motions, submesoscale fronts). Final analysis = $\hat\eta_{\text{dyn}} + \hat\eta_{\text{GP residual}}$. This is the cleanest split and keeps both stacks intact.
- **GP warm-start for 4DVarNet.** Initialise the unrolled solver iteration $x_0$ with the GP posterior mean from 00/01 instead of a zero / climatology. Has been shown to roughly halve the iteration count[^febvre2024]; the GP solve is essentially free relative to the 4DVarNet inference cost.

For the SSH-mapping benchmarks (Ocean Data Challenges, MEOM/IGE), the operational hierarchy as of 2025 is roughly: **DUACS < static GP ≈ MIOST < DYMOST < BFN-QG ≈ incremental-4D-Var < 4DVarNet**, with each step ~5–15% RMSE improvement and ~3–5× compute. The right choice for your project depends on which end of that curve you are willing to live on.

---

[^bc1999]: Bouttier, F., Courtier, P. (1999). *Data assimilation concepts and methods*. Meteorological Training Course Lecture Series, ECMWF.

[^lorenc2000]: Lorenc, A. C., Ballard, S. P., Bell, R. S., Ingleby, N. B., Andrews, P. L. F., Barker, D. M., Bray, J. R., Clayton, A. M., Dalby, T., Li, D., Payne, T. J., Saunders, F. W. (2000). "The Met. Office global three-dimensional variational data assimilation scheme." *Q. J. R. Meteorol. Soc.* 126, 2991–3012.

[^purser2003]: Purser, R. J., Wu, W.-S., Parrish, D. F., Roberts, N. M. (2003). "Numerical aspects of the application of recursive filters to variational statistical analysis. Part I: Spatially homogeneous and isotropic Gaussian covariances." *Mon. Weather Rev.* 131, 1524–1535.

[^parrish1992]: Parrish, D. F., Derber, J. C. (1992). "The National Meteorological Center's spectral statistical-interpolation analysis system." *Mon. Weather Rev.* 120, 1747–1763.

[^berre2010]: Berre, L., Pannekoucke, O., Desroziers, G., Stefanescu, S. E., Chapnik, B., Raynaud, L. (2010). "A variational assimilation ensemble and the spatial filtering of its error covariances: increase of sample size by local spatial averaging." *ECMWF Workshop on Diagnostics of Data Assimilation System Performance*.

[^ubelmann2015]: Ubelmann, C., Klein, P., Fu, L.-L. (2015). "Dynamic interpolation of sea surface height and potential applications for future high-resolution altimetry mapping." *J. Atmos. Ocean. Technol.* 32, 177–184.

[^ubelmann2020]: Ubelmann, C., Cornuelle, B., Fu, L.-L. (2020). "Dynamic mapping of along-track ocean altimetry: performance from real observations." *J. Atmos. Ocean. Technol.* 37, 1691–1707.

[^leguillou2021]: Le Guillou, F., Metref, S., Cosme, E., Ubelmann, C., Ballarotta, M., Le Sommer, J., Verron, J. (2021). "Mapping altimetry in the forthcoming SWOT era by back-and-forth nudging a one-layer quasi-geostrophic model." *J. Atmos. Ocean. Technol.* 38, 697–710.

[^leguillou2023]: Le Guillou, F., Metref, S., Cosme, E., Le Sommer, J., Ubelmann, C., Verron, J., Ballarotta, M. (2023). "Regional mapping of energetic short mesoscale ocean dynamics from altimetry." *Ocean Science* 19, 1517–1530.

[^beauchamp2023]: Beauchamp, M., Febvre, Q., Georgenthum, H., Fablet, R. (2023). "4DVarNet-SSH: end-to-end learning of variational interpolation schemes for nadir and wide-swath satellite altimetry." *Geosci. Model Dev.* 16, 2119–2147.

[^febvre2024]: Febvre, Q., Le Sommer, J., Ubelmann, C., Fablet, R. (2024). "Training neural mapping schemes for satellite altimetry with simulation data." *J. Adv. Model. Earth Syst.* 16, e2023MS003959.
