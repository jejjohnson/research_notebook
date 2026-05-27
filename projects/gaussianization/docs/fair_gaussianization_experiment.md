---
title: "Fair learning with frozen Gaussianization flows"
short_title: "Fair Gaussianization — engineering doc"
description: >
  Replace the CKA fairness penalty in keras-fairkl with a fairness loss
  built from a frozen Gaussianization flow. Three penalties are proposed:
  G-XCOV (linear CKA in Gaussianised space), G-MI (closed-form Gaussian
  mutual-information after Gaussianisation), and G-TC (total correlation
  under a frozen joint flow).
status: design
---

# Fair learning with frozen Gaussianization flows

**Project:** `projects/gaussianization`
**Author:** J. E. Johnson
**Last updated:** 2026-05-27

## 1. TL;DR

Replace the **CKA fairness penalty** in [`keras-fairkl`](https://github.com/jejjohnson/keras-fairkl)'s
`FairModelWrapper` with a family of fairness losses built from a
**frozen Gaussianization flow**.  The flow is trained once on an
auxiliary dataset, its weights are frozen, and it is then used as a
differentiable *Gaussian-space probe* inside the downstream task's
optimisation loop.  Gaussianisation lets us replace bandwidth-tuned
RBF kernels (CKA, HSIC) with closed-form, parametric, scale-invariant
penalties — the flow absorbs the kernel choice.

Three penalties, in order of strictness:

| Loss   | Captures                                                        | Closed form? | Joint flow needed? | Class |
|--------|-----------------------------------------------------------------|-------------|--------------------|---|
| G-XCOV | 2nd-moment dependence in Gaussianised space (= linear CKA there) | yes         | no — two marginal flows | `GaussianizedXCovLoss` |
| G-MI   | Mutual information assuming joint-Gaussian after Gaussianisation | yes         | no — two marginal flows | `GaussianizedMutualInfoLoss` |
| G-TC   | Full mutual information / total correlation, no Gaussian-joint assumption | no — via flow NLL | **yes** — one joint flow over $(z, q)$ | `GaussianizedTotalCorrelationLoss` |

All three are differentiable w.r.t. the downstream model parameters and
plug into `FairModelWrapper` via its `fairness_loss=...` argument.

---

## 2. The mental model — three pictures

### 2.1 Stage 1 (one-time): pretrain the probes

```
       D_pre  (auxiliary data, fixed sample)
         │
         ├─► T_z  ──► fit by MLE on N(0,I) base ──► freeze ─────────┐
         │   (marginal flow on z)                                   │
         │                                                          │
         ├─► T_q  ──► fit by MLE on N(0,I) base ──► freeze ─────────┤  used by
         │   (marginal flow on q)                                   │  G-XCOV, G-MI
         │                                                          ▼
         │                                                  ┌──────────────┐
         │                                                  │  frozen      │
         └─► T_zq ─► fit on (z, π(q))  ──► freeze ──────────┤  probes      │
             (joint flow on shuffled       independent pairs│              │
              product distribution)        only             │  used by G-TC│
                                                            └──────────────┘
```

### 2.2 Stage 2 (every step): the fair training loop

The trick — and the load-bearing claim of this whole experiment — is
that the **flow's weights are frozen** but the **flow's input is the
predictor's output**, so gradients still propagate from the loss back
through $T_z$ to $\theta$.

```
       X ───► f_θ ───────────► z ──┐                                ┌──► ∂L/∂θ
        ▲                          │                                │     ▲
        │                          ├──► task_loss(z, y) ───┐        │     │ updates θ
        │                          │                       │        │     │
        │                          │   ┌── T_z(z) ─────┐   │        │     │
        │                          ├──►│               ├──►│        │     │
       (data)                      │   │  L_fair(·, ·) │   ├──► L ──┤     │
                                   │   │               │   │ = task │     │
       q ──────────────────────────┼──►│   T_q(q)  ────┘   │ + μ·L_fair    │
                                   │   └───────────────┘   │        │     │
                                   │   frozen, but grads   │        │     │
                                   │   flow through  z     │        │     │
                                   └───────────────────────┘        │     │
                                                                    ▼     │
                                                          backprop ──────►┘
                                                          (no updates to T_z, T_q)
```

### 2.3 Where each loss penalises

```
   predictor output z              sensitive attribute q
         │                                  │
         ▼                                  ▼
      ┌──────┐                          ┌──────┐
      │  T_z │  (frozen)                │  T_q │  (frozen)
      └───┬──┘                          └───┬──┘
          │                                 │
          ▼                                 ▼
   Z = T_z(z) ~ N(0, I_dz) marginally  Q = T_q(q) ~ N(0, I_dq) marginally
          │                                 │
          └──────────────┬──────────────────┘
                         ▼
              ┌─────────────────────┐
              │ C = sample          │   ← the (d_z × d_q) cross-cov
              │  cross-cov (Z, Q)   │     is the *only* dependence
              └─────┬───────────────┘     signal that survives
                    │                     Gaussianisation under
                    │                     the joint-Gaussian
       ┌────────────┼──────────────┐      assumption
       ▼            ▼              ▼
    ‖C‖²_F     -½ logdet      − log p_N(0,I)( T_zq(z, q) )
    /‖Sz‖‖Sq‖  (I − C C^T)         ▲
       │           │                │  ← uses a full joint flow
       ▼           ▼                │     trained on shuffled
   ┌──────┐    ┌──────┐         ┌─────────┐  (independent) pairs;
   │G-XCOV│    │ G-MI │         │  G-TC   │  catches higher-order
   └──────┘    └──────┘         └─────────┘  dependence
```

---

## 3. Why Gaussianisation helps (the one-paragraph theory)

A Gaussianization flow $T: \mathbb{R}^d \to \mathbb{R}^d$ is a smooth
diffeomorphism with smooth inverse, so it preserves all statistical
dependence: $T(Z) \perp T(Q) \iff Z \perp Q$.  What it changes is the
*shape* of the marginals.  After training, each marginal of $T(X)$ is
approximately $\mathcal{N}(0, 1)$.  Three consequences:

1. **Bandwidth-free dependence measures.**  CKA and HSIC need a kernel
   bandwidth; the "right" bandwidth depends on the scale of the data.
   In Gaussianised space the scale is fixed at 1, so a *linear* kernel
   (or a unit-bandwidth RBF) suffices.  The flow absorbs the bandwidth
   choice into its mixture-CDF parameters during pretraining.

2. **Gaussian-joint assumption becomes nearly free.**  Closed-form MI
   ($-\tfrac{1}{2}\log\det(I - CC^\top)$) requires assuming the joint
   $(Z, Q)$ is Gaussian.  On raw data this is wildly wrong.  After
   marginal Gaussianisation it is much closer to true — the marginals
   are exact, only the copula remains non-Gaussian — so the closed-form
   MI estimate becomes a usable surrogate.

3. **Compatibility with frozen-flow autodiff.**  All flow components
   (`MixtureCDFGaussianization`, `Householder`) are smooth in their
   inputs.  Stopping gradient on the flow's *weights* does not stop
   gradient flow through the flow's *inputs*, so the predictor's
   parameters $\theta$ still receive a gradient signal from
   $\nabla_z \mathcal{L}_{\text{fair}}(T_z(z), T_q(q))$ via the chain
   rule.

The flow is therefore exactly the right thing to freeze: a fixed,
smooth, scale-normalising, differentiable preprocessor that turns
"measure non-linear dependence in the data" into "measure linear
dependence between near-Gaussian variables."

---

## 4. Mathematical formulation

Let $z = f_\theta(X) \in \mathbb{R}^{d_z}$ be the predictor output and
$q \in \mathbb{R}^{d_q}$ the sensitive attribute.  Define
$Z = T_z(z)$, $Q = T_q(q)$ in Gaussianised space, and let
$C = \widehat{\mathrm{Cov}}(Z, Q)$,
$S_z = \widehat{\mathrm{Cov}}(Z, Z)$,
$S_q = \widehat{\mathrm{Cov}}(Q, Q)$ denote sample (cross-)covariances
on a batch of size $n$.

### 4.1 G-XCOV — linear-CKA in Gaussianised space

$$
\mathcal{L}_{\text{G-XCOV}}
\;=\;
\frac{\lVert C \rVert_F^2}
     {\lVert S_z \rVert_F \, \lVert S_q \rVert_F + \varepsilon}
\quad\in\; [0, 1].
$$

This is exact **linear CKA** (Cortes et al. 2012; Kornblith et al. 2019)
applied to the Gaussianised features.  In $d_z = d_q = 1$ it collapses
to $\rho^2$ where $\rho$ is the Gaussianised cross-correlation.  The
un-normalised numerator $\lVert C \rVert_F^2$ is identically **HSIC with
linear kernels** on the Gaussianised features — so this single loss
covers both "linear CKA in Gaussianised space" and "HSIC with linear
kernels in Gaussianised space" depending on whether you toggle
`normalize`.

Bounded, smooth, second-moment only.  Gradient $\partial \rho^2 / \partial \rho = 2\rho$
is bounded — the loss is gentle near perfect dependence.

### 4.2 G-MI — closed-form Gaussian mutual information

If $(Z, Q)$ were *jointly* Gaussian with standardised marginals,
mutual information has the Gel'fand–Yaglom closed form:

$$
I(Z; Q) \;=\; -\tfrac{1}{2}\log\det\bigl(I_{d_z} - C\, S_q^{-1}\, C^\top\, S_z^{-1}\bigr).
$$

After Gaussianisation $S_z \approx I_{d_z}$ and $S_q \approx I_{d_q}$,
so

$$
\mathcal{L}_{\text{G-MI}}
\;=\; -\tfrac{1}{2}\log\det(I_{d_z} - C\, C^\top)
\quad\in\; [0, +\infty).
$$

In $d_z = d_q = 1$ this is $-\tfrac{1}{2}\log(1 - \rho^2)$.  The gradient
$\partial \mathcal{L} / \partial \rho = \rho / (1 - \rho^2)$ **diverges**
as $\rho \to 1$, so G-MI is much sharper than G-XCOV at high dependence.
We clip the eigenvalues of $I - CC^\top$ at a small $\varepsilon$ for
numerical safety; this caps the loss at
$-\tfrac{d}{2}\log\varepsilon$.

The closed-form requires the joint to be *Gaussian after Gaussianisation*.
Marginal Gaussianisation gets us most of the way there, but the
**residual copula** is still arbitrary — so G-MI underestimates true MI
whenever the dependence has structure beyond second-order correlation
(quadratic-in-$Z$, XOR-style, multi-modal).  That gap is exactly what
G-TC closes.

### 4.3 G-TC — total correlation under a frozen joint flow

Pretrain a **joint** flow $T_{zq}: \mathbb{R}^{d_z + d_q} \to \mathbb{R}^{d_z + d_q}$
on the empirical **product distribution** of the baseline data:
draw $(z^{(0)}_i, q_{\pi(i)})$ where $\pi$ is a random permutation.  By
construction these pairs are independent, so a well-fit $T_{zq}$
Gaussianises independent draws to $\mathcal{N}(0, I_{d_z + d_q})$.  Freeze
$T_{zq}$.

At downstream training time, evaluate the **same frozen** $T_{zq}$ on
the *actual* (potentially dependent) pair $(z, q)$:

$$
\mathcal{L}_{\text{G-TC}}
\;=\;
-\frac{1}{n}\sum_{i=1}^{n} \log p_{\mathcal{N}(0, I)}\!\bigl(T_{zq}(z_i, q_i)\bigr).
$$

When $(z, q)$ is independent like the baseline, $T_{zq}(z, q) \sim \mathcal{N}(0, I)$
and the loss equals the entropy of $\mathcal{N}(0, I_{d_z + d_q})$
(a constant in $\theta$).  When $(z, q)$ carries dependence, $T_{zq}$
no longer Gaussianises the joint and the NLL is strictly larger.

By a change-of-variables argument, this NLL difference is exactly the
KL divergence between $p(z, q)$ and $p(z)\,p(q)$ — i.e. the mutual
information.  Unlike G-MI it does **not** assume the joint is Gaussian:
the flow itself learns the copula during pretraining.  The price is
needing a richer pretraining stage.

### 4.4 Comparison table

| Property            | G-XCOV          | G-MI                 | G-TC                          |
|---------------------|-----------------|----------------------|-------------------------------|
| Order of dependence | 2nd moment      | All (joint-Gaussian) | All (no joint assumption)     |
| Range               | $[0, 1]$        | $[0, -\tfrac{d}{2}\log\varepsilon]$ | $[H_{\mathcal{N}(0,I)}, +\infty)$ |
| Gradient at high dep| Bounded         | **Diverges**         | Bounded if flow well-fit      |
| Closed form         | yes             | yes                  | no — needs flow forward pass  |
| Pretraining         | 2 marginal flows| 2 marginal flows     | 2 marginal *or* 1 joint flow  |
| Compute / batch     | one matmul      | one matmul + eigh    | full joint-flow forward       |
| Sensitive to copula structure beyond 2nd order | no | no | **yes** |

### 4.5 Deferred (stretch)

* **G-HSIC-RBF.**  HSIC with a unit-bandwidth RBF kernel in Gaussianised
  space.  Identical to existing CKA code but with $T_z(x)$ in place of
  $x$.  Useful as an ablation to separate "flow as preprocessor" from
  "linear vs RBF after the flow".
* **DR-MI.**  True mutual information via Monte-Carlo marginalisation
  through the joint flow.  Expensive; G-TC already captures the same
  signal at lower cost.

---

## 5. Hypotheses we're testing

Each loss family makes a falsifiable prediction about what kind of
dependence it can suppress, and at what cost in accuracy.

```{admonition} H1 — G-XCOV plateaus where 2nd-moment dependence saturates.
:class: hypothesis

If the predictor's residual dependence on $q$ is purely linear in
Gaussianised space (e.g. binary $q$, scalar $z$), G-XCOV at large $\mu$
will drive both Pareto axes toward zero monotonically.

**Failure prediction.** When the predictor encodes $q$ through a
non-monotone function (think: $|q|$, $q^2$ where $q$ is centred), G-XCOV
sees $C \approx 0$ even though dependence is large.  CKA-RBF and G-MI
should both still see it; G-TC definitely will.
```

```{admonition} H2 — G-MI matches CKA's terminal fairness at lower μ.
:class: hypothesis

Because G-MI's gradient diverges as $\rho^2 \to 1$, the optimiser feels
a sharply rising cost in the high-dependence regime and pushes harder.
On Adult (where G-XCOV at μ = 200 still leaves DP-diff ≈ 0.14), G-MI at
moderate μ should reach DP-diff ≈ 0.02 — comparable to CKA at μ = 50.

**Failure prediction.** If the predictor's dependence on $q$ is so weak
that $\rho$ never enters the steep regime ($\rho^2 \lesssim 0.3$), G-MI
behaves like a slightly noisier G-XCOV and offers no advantage.
```

```{admonition} H3 — G-TC catches structure G-MI misses on engineered data.
:class: hypothesis

Construct a synthetic predictor that satisfies $\rho(z, q) \approx 0$ but
$z$ is determined by $q$ through a quadratic relationship — the "XOR
analogue" of the fair_adult_census sidebar.  G-MI's value is near zero,
its gradient near zero, and the unfair predictor is undisturbed.  G-TC,
because its joint flow learnt that *independent* pairs Gaussianise to
$\mathcal{N}(0, I)$ and quadratic-dependent pairs do not, sees a finite
NLL gap and pushes back.

**Failure prediction.** If the joint flow's effective capacity is too
small (`num_blocks` too few), it cannot represent the quadratic copula
and G-TC reduces to a noisy G-MI.
```

```{admonition} H4 — Magnitude scales differ; μ must be re-tuned per loss.
:class: hypothesis

The natural magnitudes of the three losses span ~3 orders of magnitude
at the unconstrained baseline.  G-XCOV ≈ ρ² is $O(1)$; G-MI is also
$O(1)$ but with a different curvature; G-TC differs from its baseline
NLL by a small KL value.  Comparing Pareto fronts at matched μ is
unfair.  The Pareto plots should therefore parameterise by μ on
**separate** colour-coded curves, and the only fair comparison is at
matched fairness (vertical slice).
```

---

## 6. Experiment design

### 6.1 Two-stage pipeline (reprise of the ASCII diagrams)

**Stage 1 — pretrain + freeze.**

* Train $T_z$ on the predictor-output distribution of an unconstrained
  baseline (e.g. for classification, sigmoid probabilities of the
  baseline MLP — *not* the raw binary labels, otherwise the flow lives
  on a 2-point support that is off-support of the actual predictions).
* Train $T_q$ on the marginal of $q$.
* Train $T_{zq}$ on the **shuffled product distribution** of the same
  baseline data — independent pairs by construction.
* Freeze all weights.

**Stage 2 — fair downstream training.**

* Drop in `FairModelWrapper(base, mu=μ, fairness_loss=...)` with any of
  the three new losses.
* `compile(..., loss="...")` as usual.  The wrapper handles the dict
  packing `{"x": X, "q": q}`.

### 6.2 What gets penalised, where the gradient goes

A short walk through one optimiser step:

```
forward:
    z = f_θ(X)               # (n, 1)
    L_task = mse(z, y)        # standard Keras loss
    L_fair = G-MI(T_z(z),     # passes through frozen T_z,
                  T_q(q))     #   T_q -- their weights are not
                              #   in trainable_variables
    L = L_task + μ · L_fair

backward (autodiff):
    ∂L_task/∂z   →  ∂L_task/∂θ
    ∂L_fair/∂z   = ∂L_fair / ∂T_z(z) · ∂T_z(z) / ∂z
                   ↑                    ↑
                   eigh of (I − CC^T)   smooth mixture-CDF jacobian
                                        — frozen but still differentiable
    ∂L_fair/∂z   →  ∂L_fair/∂θ
    optimiser update:  θ ← θ − η ∂L/∂θ
    (T_z, T_q, T_zq receive no gradient — they have zero trainable_weights)
```

The key chain-rule fact: `stop_gradient` (or `trainable=False`) blocks
gradients into the **parameters** of the flow, but does **not** block
gradients into its **inputs**.  Without this property the whole scheme
collapses.

### 6.3 File layout

```
projects/gaussianization/
├── src/gaussianization/fair/
│   ├── __init__.py          # public API
│   ├── losses.py            # GaussianizedXCovLoss / MutualInfoLoss / TotalCorrelationLoss
│   ├── pretrain.py          # fit_and_freeze, fit_and_freeze_joint
│   ├── freeze.py            # freeze_flow helper
│   └── metrics.py           # numpy fairness eval metrics
├── tests/test_fair.py       # 15 tests including closed-form checks
├── notebooks/fair_gauss/
│   ├── 05_fair_gauss_pretrain.ipynb
│   ├── 06_fair_gauss_synthetic.ipynb   # G-XCOV + G-MI + G-TC + CKA
│   ├── 07_fair_gauss_adult.ipynb       # same on UCI Adult
│   └── _style.py
└── docs/fair_gaussianization_experiment.md   # this file
```

---

## 7. Datasets

| Dataset                                                                              | Sensitive $q$ | Task                 | Use                                                                                                |
|--------------------------------------------------------------------------------------|--------------|----------------------|----------------------------------------------------------------------------------------------------|
| Synthetic regression: $y = \tanh(x_1) + 0.5 x_2 + 3 q + \varepsilon$, $q\sim\mathrm{Bern}(0.5)$ | $q$          | regression           | Notebook 06; the structure is exactly the fairkl `fair_model_wrapper` benchmark.                  |
| UCI Adult Census (OpenML id `adult` v2)                                              | gender       | binary classification| Notebook 07; ~49k rows, 5 numeric features + gender as a feature.                                  |
| (Future) Engineered quadratic-dependence dataset                                     | $q$          | regression           | H3 test: $z$ determined by $q^2$ so $\rho \approx 0$ but MI > 0.  Distinguishes G-MI from G-TC.    |
| (Future) COMPAS                                                                      | race         | classification       | Second real-data check.                                                                            |

---

## 8. Evaluation plan

For every (dataset, loss, $\mu$, seed) combination report:

* **Predictive metrics.**  RMSE / R² (regression).  Accuracy, ROC-AUC,
  log-loss (classification).
* **Fairness metrics (numpy-side, neutral judge).**  Demographic-parity
  difference, equalized-odds difference, |Pearson($\hat y, q$)|.  These
  are computed at evaluation time, *not* used as the training loss, so
  they are independent of the training penalty.
* **Diagnostic: training-time loss value.**  Each loss's own value
  through training (so we can see G-MI saturate at the eps clip if it
  does, etc.).

**Comparison axes**

1. Loss family: `cka | g_xcov | g_mi | g_tc` × $\mu \in \{0, 0.1, 0.5, 2, 10, 50, 200\}$.
2. Flow depth (`num_blocks ∈ {2, 4, 8}`) — does a deeper joint flow
   improve G-TC on the quadratic-dependence test (H3)?
3. Pretraining set — same-as-task vs held-out vs i.i.d. Gaussian
   (the latter collapses G-XCOV to vanilla cross-cov; control).
4. Batch size — G-MI and G-TC may be more batch-sensitive than G-XCOV
   (eigh + flow forward).

**Statistical rigour.**  3 seeds for the headline figures, 5 for any
"X beats Y at matched fairness" claim, paired-bootstrap CIs.

**Magnitude calibration follow-up.**  Each loss has a different natural
scale.  To make the loss-family axis a *fair* comparison, scale $\mu$
per-family by the loss value at the unconstrained baseline, so that
`effective μ` puts everyone at the same relative pressure.  Report
both raw-μ and calibrated-μ Pareto fronts.

---

## 9. Risks & open questions

```{admonition} Risk — flow off-support drift
:class: warning

`flow_z` is pretrained on the baseline predictor's outputs.  During fair
training the predictor's output distribution shifts.  If it drifts far
off-support of $D_{\text{pre}}$ the flow's Gaussianisation breaks down
and the fairness loss becomes unreliable.

**Mitigation.**  (a) Pretrain with a wider noise-augmented dataset.
(b) Periodically refresh `flow_z` on a fresh batch of predictor outputs.
(c) Monitor the marginal NLL of the predictor's outputs under
`flow_z.log_prob` during training and warn when it climbs.
```

```{admonition} Risk — joint-flow capacity for G-TC
:class: warning

If $T_{zq}$ doesn't have enough capacity to represent the empirical
product distribution, it will not Gaussianise independent pairs cleanly
— the baseline NLL is then noisy and G-TC has high-variance gradients.

**Mitigation.**  Use `make_coupling_flow` rather than the diagonal stack
for $T_{zq}$ when $d_z + d_q > 2$; pretrain with more `n_shuffles` so
the product distribution is well-sampled.
```

| Other risk | Mitigation |
|---|---|
| Eigendecomposition (`ops.linalg.eigh`) is slow for large $d_z$ | We expect $d_z = 1$ in practice (regression / single-logit classification); scalar fast path is taken. |
| Frozen-flow gradient is silently zeroed | Existing `test_loss_gradients_flow_to_predictor` parametrised over G-XCOV and G-MI asserts non-zero gradients to $\theta$. |
| Comparing μ's across losses is misleading | The doc warns about it; notebooks plot each loss as its own Pareto curve, no point-to-point comparison at matched μ. |

**Open questions for follow-ups.**

* Does *fine-tuning* the marginal flows during downstream training (a
  light EMA update) actually help, or does it leak into the predictor's
  gradient and corrupt the fairness signal?
* G-TC requires a joint flow.  Can a single $T_{zq}$ pretrained once
  serve every downstream task on the same data, or does it need to be
  re-trained per architecture?
* For multi-class sensitive attributes (race in COMPAS), does the
  one-hot encoding of $q$ work, or do we need a categorical flow head?

---

## 10. Milestones

| # | Milestone | Acceptance |
|---|---|---|
| ✅ M1 | Skeleton: `fair/{losses,freeze,pretrain,metrics}.py` + tests pass on synthetic data | `pytest tests/test_fair.py` green; lint + typecheck clean |
| ✅ M2 | Notebook 05: pretrain + freeze + 4 diagnostics | Executed, committed with figures |
| ✅ M3 | Notebook 06: synthetic Pareto with G-XCOV vs CKA | Pareto curve from RMSE 0.11 to 1.35 |
| ✅ M4 | Notebook 07: Adult Pareto with G-XCOV vs CKA | Pareto traced; CKA beats G-XCOV terminal fairness as expected |
| 🟡 M5 | **G-MI loss + G-TC loss + extended tests** | This commit |
| 🟡 M6 | Notebooks 06 and 07 re-executed with G-MI and G-TC trajectories | Pareto plots show 4 curves (CKA / G-XCOV / G-MI / G-TC) |
| ⏳ M7 | H3 quadratic-dependence experiment isolating G-TC's advantage | New notebook `08_quadratic_dependence.ipynb` |
| ⏳ M8 | Hydra config + DVC stage for reproducibility | `pixi run dvc repro` regenerates all figures |
| ⏳ M9 | Magnitude-calibration follow-up | Calibrated-μ Pareto plots alongside raw-μ |

---

## 11. References

* Cortes, Mohri, Rostamizadeh (2012). *Algorithms for Learning Kernels Based on Centered Alignment.* JMLR 13.
* Kornblith, Norouzi, Lee, Hinton (2019). *Similarity of Neural Network Representations Revisited.* ICML.
* Gretton, Bousquet, Smola, Schölkopf (2005). *Measuring Statistical Dependence with Hilbert–Schmidt Norms.* ALT.
* Gel'fand & Yaglom (1957). *Computation of the amount of information about a random function contained in another such function.* AMS Translations 12.
* Chen, Gopinath (2000). *Gaussianization.* NeurIPS.
* Laparra, Camps-Valls, Malo (2011). *Iterative Gaussianization: From ICA to Random Rotations* (RBIG).
* Meng, Song, Song, Ermon (2020). *Gaussianization Flows.* AISTATS.
* Watanabe (1960). *Information theoretical analysis of multivariate correlation.* (Total correlation.)
* Pérez-Suay et al. (2017). *Fair Kernel Learning.* ECML PKDD.
* Belghazi et al. (2018). *MINE: Mutual Information Neural Estimation.* ICML.

---

## Appendix A — Minimum working example

```python
import os
os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
import numpy as np

from fairkl.models import FairModelWrapper
from gaussianization.fair import (
    GaussianizedMutualInfoLoss,
    GaussianizedXCovLoss,
    GaussianizedTotalCorrelationLoss,
    fit_and_freeze,
    fit_and_freeze_joint,
)

# Synthetic data with q as a feature
rng = np.random.default_rng(0)
n = 4000
q = rng.binomial(1, 0.5, n).astype("float32")
x = rng.standard_normal((n, 3)).astype("float32")
y = (np.tanh(x[:, 0]) + 0.5 * x[:, 1] + 3 * q
     + 0.1 * rng.standard_normal(n)).astype("float32")
X = np.concatenate([x, q.reshape(-1, 1)], axis=1)

# Stage 1: pretrain + freeze probes
flow_y, _  = fit_and_freeze(y.reshape(-1, 1), num_blocks=4, epochs=40, seed=0)
flow_q, _  = fit_and_freeze(q.reshape(-1, 1), num_blocks=2, epochs=40, seed=0)
flow_yq, _ = fit_and_freeze_joint(y, q, num_blocks=4, epochs=40, seed=0)

# Stage 2: pick a loss and train
mlp = keras.Sequential([keras.layers.Dense(32, "relu"),
                        keras.layers.Dense(32, "relu"),
                        keras.layers.Dense(1)])
fair = FairModelWrapper(
    mlp, mu=2.0,
    fairness_loss=GaussianizedMutualInfoLoss(flow_z=flow_y, flow_q=flow_q),
    # or: GaussianizedXCovLoss(flow_z=flow_y, flow_q=flow_q),
    # or: GaussianizedTotalCorrelationLoss(joint_flow=flow_yq),
)
fair.compile(optimizer="adam", loss="mse")
fair.fit(X, y, q=q, epochs=40, batch_size=256)
```
