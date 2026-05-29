---
title: "Fair learning with frozen Gaussianization flows — design doc"
short_title: "Design doc"
subtitle: "Three fairness penalties built from a frozen Gaussianization flow"
description: >
  Replace the CKA fairness penalty in keras-fairkl with a fairness loss
  built from a frozen Gaussianization flow. Three penalties are proposed:
  G-XCOV (linear CKA in Gaussianised space), G-MI (closed-form Gaussian
  mutual-information after Gaussianisation), and G-TC (total correlation
  under a frozen joint flow).
authors:
  - name: J. E. Johnson
    github: jejjohnson
date: 2026-05-27
status: design
---

(sec-fair-design)=
# Fair learning with frozen Gaussianization flows

:::{seealso} Companion pages
[](./fair_overview.md) · [](./fair_gaussianization_followups.md)
:::

(sec-tldr)=
## 1. TL;DR

Replace the **{abbr}`CKA` fairness penalty** in [`keras-fairkl`](https://github.com/jejjohnson/keras-fairkl)'s
`FairModelWrapper` — a Keras port of the fair-kernel-learning idea of
{cite:t}`perezSuay2017fairkernel` — with a family of fairness losses
built from a **frozen Gaussianization flow**.  The flow is trained once on an
auxiliary dataset, its weights are frozen, and it is then used as a
differentiable *Gaussian-space probe* inside the downstream task's
optimisation loop.  Gaussianisation lets us replace bandwidth-tuned
RBF kernels ({abbr}`CKA`, {abbr}`HSIC`) with closed-form, parametric,
scale-invariant penalties — the flow absorbs the kernel choice.

Three penalties, in order of strictness:

:::{table} Output-side fairness losses; see [](#sec-formulation) for the math.
:name: tbl-tldr-losses
:align: left

| Loss   | Captures                                                          | Closed form?     | Joint flow needed?      | Class                              |
| ------ | ----------------------------------------------------------------- | ---------------- | ----------------------- | ---------------------------------- |
| G-XCOV | 2nd-moment dependence in Gaussianised space (linear CKA there)    | yes              | no — two marginal flows | `GaussianizedXCovLoss`             |
| G-MI   | MI assuming joint-Gaussian after Gaussianisation                  | yes              | no — two marginal flows | `GaussianizedMutualInfoLoss`       |
| G-TC   | Full MI / total correlation, no joint-Gaussian assumption         | no — via flow NLL | **yes** — one joint flow over $(z, q)$ | `GaussianizedTotalCorrelationLoss` |
:::

All three are differentiable w.r.t. the downstream model parameters and
plug into `FairModelWrapper` via its `fairness_loss=...` argument.

---

(sec-mental-model)=
## 2. The mental model — three pictures

(sec-stage1)=
### 2.1 Stage 1 (one-time): pretrain the probes

```{mermaid}
flowchart LR
    D[("D<sub>pre</sub> · auxiliary data")]:::data

    D --> Tz["T<sub>z</sub><br/><i>marginal flow on z</i>"]:::frozen
    D --> Tq["T<sub>q</sub><br/><i>marginal flow on q</i>"]:::frozen
    D --> Tzq["T<sub>zq</sub><br/><i>joint flow on shuffled (z, π(q))<br/>— independent pairs only</i>"]:::frozen

    Tz -- "MLE fit → freeze" --> Pm[["frozen probes<br/><b>used by G-XCOV, G-MI</b>"]]:::output
    Tq -- "MLE fit → freeze" --> Pm
    Tzq -- "MLE fit → freeze" --> Pj[["frozen probe<br/><b>used by G-TC</b>"]]:::output

    classDef data fill:#f1f5f9,stroke:#475569,stroke-width:1.5px,color:#0f172a;
    classDef frozen fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,stroke-dasharray:5 3,color:#1e1b4b;
    classDef output fill:#dcfce7,stroke:#15803d,stroke-width:1.5px,color:#052e16;
```

(sec-stage2)=
### 2.2 Stage 2 (every step): the fair training loop

The trick — and the load-bearing claim of this whole experiment — is
that the **flow's weights are frozen** but the **flow's input is the
predictor's output**, so gradients still propagate from the loss back
through $T_z$ to $\theta$.

```{mermaid}
flowchart LR
    X[("X")]:::data
    q[("q")]:::data
    y[("y")]:::data

    X --> Ftheta["f<sub>θ</sub><br/><i>predictor (trainable)</i>"]:::trainable
    Ftheta --> z(["z = f<sub>θ</sub>(X)"]):::output

    z --> Ltask{{"L<sub>task</sub><br/>mse(z, y)"}}:::loss
    y --> Ltask

    z --> Tz["T<sub>z</sub>(z)<br/><i>frozen</i>"]:::frozen
    q --> Tq["T<sub>q</sub>(q)<br/><i>frozen</i>"]:::frozen
    Tz --> Lfair{{"μ · L<sub>fair</sub>"}}:::loss
    Tq --> Lfair

    Ltask --> L((("L = L<sub>task</sub> + μ·L<sub>fair</sub>"))):::loss
    Lfair --> L

    L -. "∂L/∂θ — gradient flows through frozen T<sub>z</sub> back to θ" .-> Ftheta

    classDef data fill:#f1f5f9,stroke:#475569,stroke-width:1.5px,color:#0f172a;
    classDef trainable fill:#fef3c7,stroke:#b45309,stroke-width:2px,color:#451a03;
    classDef frozen fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,stroke-dasharray:5 3,color:#1e1b4b;
    classDef loss fill:#fce7f3,stroke:#be185d,stroke-width:2px,color:#500724;
    classDef output fill:#dcfce7,stroke:#15803d,stroke-width:1.5px,color:#052e16;
```

(sec-where-penalise)=
### 2.3 Where each loss penalises

```{mermaid}
flowchart TB
    z[("predictor output z")]:::data
    q[("sensitive attribute q")]:::data

    z --> Tz["T<sub>z</sub><br/><i>frozen</i>"]:::frozen
    q --> Tq["T<sub>q</sub><br/><i>frozen</i>"]:::frozen

    Tz --> Z(["Z = T<sub>z</sub>(z)<br/>~ N(0, I<sub>d_z</sub>) marginally"]):::compute
    Tq --> Q(["Q = T<sub>q</sub>(q)<br/>~ N(0, I<sub>d_q</sub>) marginally"]):::compute

    Z --> C[["C = sample cross-cov(Z, Q)<br/><i>only 2nd-order signal that<br/>survives Gaussianisation</i>"]]:::compute
    Q --> C

    z --> Tzq["T<sub>zq</sub><br/><i>joint flow, frozen</i>"]:::frozen
    q --> Tzq

    C --> GXCOV{{"G-XCOV<br/>‖C‖²<sub>F</sub> / (‖S<sub>z</sub>‖·‖S<sub>q</sub>‖)<br/><i>2nd moment only</i>"}}:::loss
    C --> GMI{{"G-MI<br/>−½ log det(I − C Cᵀ)<br/><i>joint-Gaussian assumed</i>"}}:::loss
    Tzq --> GTC{{"G-TC<br/>−log p<sub>N(0,I)</sub>(T<sub>zq</sub>(z, q))<br/><i>full copula, no Gaussian joint</i>"}}:::loss

    classDef data fill:#f1f5f9,stroke:#475569,stroke-width:1.5px,color:#0f172a;
    classDef frozen fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,stroke-dasharray:5 3,color:#1e1b4b;
    classDef compute fill:#f5f3ff,stroke:#7c3aed,stroke-width:1.5px,color:#2e1065;
    classDef loss fill:#fce7f3,stroke:#be185d,stroke-width:2px,color:#500724;
```

---

(sec-why-gauss)=
## 3. Why Gaussianisation helps (the one-paragraph theory)

A Gaussianization flow $T: \mathbb{R}^d \to \mathbb{R}^d$ — in the
lineage of {cite:t}`chenGopinath2000gauss`, {cite:t}`laparra2011rbig`,
and {cite:t}`meng2020gaussflow` — is a smooth diffeomorphism with smooth
inverse, so it preserves all statistical dependence: $T(Z) \perp T(Q) \iff Z \perp Q$.  What it changes is the
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

(sec-formulation)=
## 4. Mathematical formulation

Let $z = f_\theta(X) \in \mathbb{R}^{d_z}$ be the predictor output and
$q \in \mathbb{R}^{d_q}$ the sensitive attribute.  Define
$Z = T_z(z)$, $Q = T_q(q)$ in Gaussianised space, and let
$C = \widehat{\mathrm{Cov}}(Z, Q)$,
$S_z = \widehat{\mathrm{Cov}}(Z, Z)$,
$S_q = \widehat{\mathrm{Cov}}(Q, Q)$ denote sample (cross-)covariances
on a batch of size $n$.

(sec-gxcov)=
### 4.1 G-XCOV — linear-CKA in Gaussianised space

$$
\mathcal{L}_{\text{G-XCOV}}
\;=\;
\frac{\lVert C \rVert_F^2}
     {\lVert S_z \rVert_F \, \lVert S_q \rVert_F + \varepsilon}
\quad\in\; [0, 1].
$$ (eq-gxcov)

Equation {eq}`eq-gxcov` is exact **linear {abbr}`CKA`**
{cite:p}`cortes2012cka,kornblith2019cka` applied to the Gaussianised
features.  In $d_z = d_q = 1$ it collapses to $\rho^2$ where $\rho$ is
the Gaussianised cross-correlation.  The un-normalised numerator
$\lVert C \rVert_F^2$ is identically **{abbr}`HSIC` with linear
kernels** {cite:p}`gretton2005hsic` on the Gaussianised features — so
this single loss covers both "linear {abbr}`CKA` in Gaussianised space"
and "{abbr}`HSIC` with linear kernels in Gaussianised space" depending
on whether you toggle `normalize`.

Bounded, smooth, second-moment only.  Gradient $\partial \rho^2 / \partial \rho = 2\rho$
is bounded — the loss is gentle near perfect dependence.

(sec-gmi)=
### 4.2 G-MI — closed-form Gaussian mutual information

If $(Z, Q)$ were *jointly* Gaussian with standardised marginals,
mutual information has the Gel'fand–Yaglom closed form
{cite:p}`gelfandYaglom1957`:

$$
I(Z; Q) \;=\; -\tfrac{1}{2}\log\det\bigl(I_{d_z} - C\, S_q^{-1}\, C^\top\, S_z^{-1}\bigr).
$$ (eq-gy-mi)

After Gaussianisation $S_z \approx I_{d_z}$ and $S_q \approx I_{d_q}$,
so {eq}`eq-gy-mi` simplifies to

$$
\mathcal{L}_{\text{G-MI}}
\;=\; -\tfrac{1}{2}\log\det(I_{d_z} - C\, C^\top)
\quad\in\; [0, +\infty).
$$ (eq-gmi)

In $d_z = d_q = 1$ {eq}`eq-gmi` is $-\tfrac{1}{2}\log(1 - \rho^2)$.  The
gradient $\partial \mathcal{L} / \partial \rho = \rho / (1 - \rho^2)$
**diverges** as $\rho \to 1$, so {abbr}`G-MI` is much sharper than
{abbr}`G-XCOV` at high dependence.  We clip the eigenvalues of
$I - CC^\top$ at a small $\varepsilon$ for numerical safety; this caps
the loss at $-\tfrac{d}{2}\log\varepsilon$.

The closed-form requires the joint to be *Gaussian after Gaussianisation*.
Marginal Gaussianisation gets us most of the way there, but the
**residual copula** is still arbitrary — so {abbr}`G-MI` underestimates
true {abbr}`MI` whenever the dependence has structure beyond
second-order correlation (quadratic-in-$Z$, XOR-style, multi-modal).
That gap is exactly what {abbr}`G-TC` closes.

(sec-gtc)=
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
$$ (eq-gtc)

When $(z, q)$ is independent like the baseline, $T_{zq}(z, q) \sim \mathcal{N}(0, I)$
and {eq}`eq-gtc` equals the entropy of $\mathcal{N}(0, I_{d_z + d_q})$
(a constant in $\theta$).  When $(z, q)$ carries dependence, $T_{zq}$
no longer Gaussianises the joint and the {abbr}`NLL` is strictly larger.

By a change-of-variables argument, this {abbr}`NLL` difference is
exactly the KL divergence between $p(z, q)$ and $p(z)\,p(q)$ — i.e. the
mutual information; see {cite:t}`watanabe1960tc` for the
total-correlation framing.  Unlike {abbr}`G-MI` it does **not** assume
the joint is Gaussian: the flow itself learns the copula during
pretraining.  The price is needing a richer pretraining stage.

(sec-comparison)=
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

(sec-deferred)=
### 4.5 Deferred (stretch)

* **{abbr}`G-HSIC`-RBF.**  {abbr}`HSIC` with a unit-bandwidth RBF kernel
  in Gaussianised space.  Identical to existing {abbr}`CKA` code but
  with $T_z(x)$ in place of $x$.  Useful as an ablation to separate
  "flow as preprocessor" from "linear vs RBF after the flow".
* **DR-{abbr}`MI`.**  True mutual information via Monte-Carlo
  marginalisation through the joint flow.  Compare with the neural
  estimator of {cite:t}`belghazi2018mine`.  Expensive; {abbr}`G-TC`
  already captures the same signal at lower cost.

---

(sec-hypotheses)=
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

(sec-exp-design)=
## 6. Experiment design

(sec-two-stage)=
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

(sec-grad-flow)=
### 6.2 What gets penalised, where the gradient goes

A short walk through one optimiser step. Solid arrows are the forward
pass; dashed arrows are the backward (autodiff) pass, each labelled with
the chain-rule factor it carries.

```{mermaid}
flowchart LR
    X[("X")]:::data
    q[("q")]:::data
    y[("y")]:::data

    X --> Ftheta["f<sub>θ</sub><br/><i>trainable</i>"]:::trainable
    Ftheta --> z(["z = f<sub>θ</sub>(X)"]):::output
    z --> Ltask{{"L<sub>task</sub><br/>mse(z, y)"}}:::loss
    y --> Ltask
    z --> Tz["T<sub>z</sub>(z)<br/><i>frozen, differentiable</i>"]:::frozen
    q --> Tq["T<sub>q</sub>(q)<br/><i>frozen</i>"]:::frozen
    Tz --> Lfair{{"μ · L<sub>fair</sub><br/>− ½ log det(I − CCᵀ)"}}:::loss
    Tq --> Lfair
    Ltask --> L((("L = L<sub>task</sub> + μ·L<sub>fair</sub>"))):::loss
    Lfair --> L

    L -. "∂L/∂L<sub>fair</sub> · ∂L<sub>fair</sub>/∂T<sub>z</sub>(z)<br/>via eigh(I − CCᵀ)" .-> Tz
    Tz -. "× ∂T<sub>z</sub>(z)/∂z<br/>mixture-CDF Jacobian (frozen weights, live input)" .-> z
    z -. "× ∂z/∂θ → ∂L/∂θ<br/>θ ← θ − η · ∂L/∂θ" .-> Ftheta

    classDef data fill:#f1f5f9,stroke:#475569,stroke-width:1.5px,color:#0f172a;
    classDef trainable fill:#fef3c7,stroke:#b45309,stroke-width:2px,color:#451a03;
    classDef frozen fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,stroke-dasharray:5 3,color:#1e1b4b;
    classDef loss fill:#fce7f3,stroke:#be185d,stroke-width:2px,color:#500724;
    classDef output fill:#dcfce7,stroke:#15803d,stroke-width:1.5px,color:#052e16;
```

The key chain-rule fact, read off the dashed path above: `stop_gradient`
(or `trainable=False`) blocks gradients into the **parameters** of the
flow, but does **not** block gradients into its **inputs** — the
mixture-CDF Jacobian $\partial T_z(z)/\partial z$ is still smooth and
non-zero, so the predictor still receives a fairness signal. Without
this property the whole scheme collapses. $T_z$, $T_q$, and $T_{zq}$
themselves receive no gradient: they have zero `trainable_weights`.

(sec-file-layout)=
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

(sec-datasets)=
## 7. Datasets

| Dataset                                                                              | Sensitive $q$ | Task                 | Use                                                                                                |
|--------------------------------------------------------------------------------------|--------------|----------------------|----------------------------------------------------------------------------------------------------|
| Synthetic regression: $y = \tanh(x_1) + 0.5 x_2 + 3 q + \varepsilon$, $q\sim\mathrm{Bern}(0.5)$ | $q$          | regression           | Notebook 06; the structure is exactly the fairkl `fair_model_wrapper` benchmark.                  |
| UCI Adult Census (OpenML id `adult` v2)                                              | gender       | binary classification| Notebook 07; ~49k rows, 5 numeric features + gender as a feature.                                  |
| (Future) Engineered quadratic-dependence dataset                                     | $q$          | regression           | H3 test: $z$ determined by $q^2$ so $\rho \approx 0$ but MI > 0.  Distinguishes G-MI from G-TC.    |
| (Future) COMPAS                                                                      | race         | classification       | Second real-data check.                                                                            |

---

(sec-eval-plan)=
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

(sec-risks)=
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

(sec-milestones)=
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

(sec-fair-design-mwe)=
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
