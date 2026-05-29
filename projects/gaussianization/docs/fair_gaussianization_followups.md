---
title: "Fair Gaussianization — input-side follow-up experiments"
short_title: "Follow-ups"
subtitle: "Seven alternatives that put the flow on inputs instead of outputs"
description: >
  Seven follow-up experiments that move the Gaussianization flow from
  the predictor's *output* (current setup) to the inputs / representations
  / per-sample reweighting / counterfactual generation pipeline. Math,
  engineering, pseudocode, asks, and tradeoffs for each.
authors:
  - name: J. E. Johnson
    github: jejjohnson
date: 2026-05-27
status: design
---

(sec-fair-followups)=
# Fair Gaussianization — follow-up experiments

:::{seealso} Companion pages
[](./fair_overview.md) · [](./fair_gaussianization_experiment.md)
:::

(sec-followup-why)=
## 0. Why a follow-up

The original experiment puts the Gaussianization flow on the **predictor's
output side** — $T_z(f_\theta(X))$ and $T_q(q)$ — and measures their
dependence as a training-time fairness penalty. That works, but it has
two structural problems we surfaced empirically in notebooks 06–07:

1. **Moving target on $T_z$.** The predictor's output distribution
   shifts during training, so $T_z$ (pretrained on a baseline) goes
   off-support. Gradients keep flowing, but they encode "distance from
   the baseline distribution" rather than "distance from
   independence." G-TC's constant-predictor collapse is the symptom.
2. **Inputs are untouched.** $X$ never moves during predictor
   training, so a flow on $X$ is *always* in-support. We're not using
   that.

This doc proposes **seven follow-up experiments** that move the flow's
role around the pipeline. Three of them are pure preprocessing (the
flow runs once, offline); two are training-time but on stable
quantities; one is a counterfactual data-augmentation. Each comes with
math, pseudocode, an explicit "ask" of what new infrastructure is
needed, honest tradeoffs, and a falsifiable hypothesis.

(sec-followup-tldr)=
## 1. TL;DR — the seven approaches at a glance

```{list-table} The seven follow-up approaches.  Each row is a separate downstream-training recipe; the flow's job changes from row to row.
:header-rows: 1
:name: tbl-seven-approaches

* - Approach
  - Flow's job
  - When it runs
  - Predictor sees
  - Fairness mechanism
* - **A. Input whitening**
  - $T_X$ Gaussianises features
  - Offline
  - $T_X(X)$
  - indirect (better conditioning)
* - **B. Fair feature selection**
  - Per-feature dependence score
  - Offline
  - $X_{:, S_K}$ (top-$K$ independent)
  - hard subset selection
* - **C. Subspace projection**
  - Joint $T_X$ + q-orthogonal projection $P$
  - Offline
  - $P^\top T_X(X)$
  - hard linear projection
* - **D. Conditional flow $T_{X\|q}$**
  - Gaussianises $X$ given $q$
  - Offline
  - $T_{X\|q}(X, q)$
  - structural ($Z \perp q$ by construction)
* - **E. Counterfactual augmentation**
  - Generates $\tilde X$ with $q$ flipped
  - Offline (data prep)
  - $X \cup \tilde X$
  - data-level + consistency loss
* - **F. Density-ratio reweighting**
  - Estimates $p(X\|q)$
  - Offline (weight prep)
  - $X$ with weights $w_i$
  - classical importance weighting
* - **G. Representation bottleneck** *(stretch)*
  - $T_R$ on encoder output
  - Training-time (with refresh)
  - encoded $R$ then head
  - soft penalty on intermediate representation
```

All seven leave the predictor's training-loop architecture **unchanged**
(or nearly so) — the work happens before training, and the predictor
sees a tweaked feature space or a weighted task loss. Compare to the
*original* experiment, where the fairness logic was inside the
optimisation loop. The follow-ups are easier to compose, easier to
debug, and don't carry the moving-target risk.

:::{admonition} Design-space map — where the flow operates
:class: tip

```{mermaid}
flowchart LR
    subgraph IN["<b>flow on inputs X</b>"]
        direction TB
        AA["<b>A</b> · whiten<br/><i>feature stream</i>"]:::input
        CC["<b>C</b> · subspace project<br/><i>feature stream</i>"]:::input
        DD["<b>D</b> · conditional T(X∣q)<br/><i>feature stream</i>"]:::input
        BB["<b>B</b> · rank features<br/><i>scoring</i>"]:::input
        EE["<b>E</b> · counterfactual aug.<br/><i>data stream</i>"]:::input
        FF["<b>F</b> · density-ratio IPW<br/><i>data stream</i>"]:::input
    end
    subgraph OUT["<b>flow on outputs z</b> · original experiment"]
        direction TB
        OO[["G-XCOV · G-MI · G-TC"]]:::output
    end
    subgraph REP["<b>flow on representations R</b>"]
        direction TB
        GG["<b>G</b> · bottleneck<br/><i>stretch</i>"]:::stretch
    end

    classDef input fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,color:#1e1b4b;
    classDef output fill:#fef3c7,stroke:#b45309,stroke-width:2px,color:#451a03;
    classDef stretch fill:#f5f3ff,stroke:#7c3aed,stroke-width:1.5px,stroke-dasharray:5 3,color:#2e1065;
```

The original experiment lives in the *outputs* column; every
follow-up here moves it to the inputs (or, for **G**, to an intermediate
representation).
:::

Notation throughout: $X \in \mathbb{R}^{n \times d}$ inputs,
$y \in \mathbb{R}^n$ targets, $q \in \mathbb{R}^{n \times d_q}$
sensitive attribute, $f_\theta$ the trainable predictor.

---

(sec-approach-a)=
## 2. Approach A — Input whitening

The simplest and most boring of the six. Frozen Gaussianization flow as
*preprocessor* — a direct descendant of {abbr}`RBIG`-style
whitening {cite:p}`laparra2011rbig`; no fairness machinery added on top.

### 2.1 Math

Pretrain a joint Gaussianization flow

$$
T_X : \mathbb{R}^d \to \mathbb{R}^d, \qquad T_X(X) \approx \mathcal{N}(0, I_d) \text{ marginally and jointly.}
$$ (eq-tx-joint)

Freeze. The predictor sees Gaussianised inputs:

$$
f_\theta\bigl(T_X(X)\bigr) \approx y.
$$ (eq-predict-whitened)

This is **information-preserving** ($T_X$ is a diffeomorphism, so
$T_X(X)$ carries the same information as $X$), but the predictor
operates on features with controlled marginals and a closer-to-isotropic
joint.

### 2.2 Pipeline

```{mermaid}
flowchart LR
    subgraph OFF["Offline"]
        direction LR
        Xp[("X<sub>pre</sub>")]:::data --> Tx["T<sub>X</sub><br/><i>fit MLE → freeze</i>"]:::frozen
    end
    subgraph TRN["Training loop"]
        direction LR
        X[("X")]:::data --> Tx2["T<sub>X</sub>(X)"]:::frozen --> F["f<sub>θ</sub>"]:::trainable --> L{{"task loss<br/><i>no fairness penalty</i>"}}:::loss
        L -.->|θ updates| F
    end
    Tx -. reused frozen .-> Tx2

    classDef data fill:#f1f5f9,stroke:#475569,stroke-width:1.5px,color:#0f172a;
    classDef frozen fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,stroke-dasharray:5 3,color:#1e1b4b;
    classDef trainable fill:#fef3c7,stroke:#b45309,stroke-width:2px,color:#451a03;
    classDef loss fill:#fce7f3,stroke:#be185d,stroke-width:2px,color:#500724;
```

### 2.3 Pseudocode

```python
from gaussianization.fair import fit_and_freeze

# Stage 1: pretrain once
T_X, _ = fit_and_freeze(
    X_train, num_blocks=8, num_components=12, epochs=200, seed=0,
)

# Stage 2: standard Keras training on Gaussianised inputs
X_train_whitened = T_X(X_train)
X_test_whitened  = T_X(X_test)

mlp = keras.Sequential([
    keras.Input(shape=(d,)),
    keras.layers.Dense(32, "relu"),
    keras.layers.Dense(1),
])
mlp.compile(optimizer="adam", loss="mse")
mlp.fit(X_train_whitened, y_train, ...)
```

Or, if you want $T_X$ inside the predictor graph (so saliency
explanations live in original-X space), wrap it as a frozen layer:

```python
inputs = keras.Input(shape=(d,))
whitened = GaussianizationLayer(T_X)(inputs)   # NEW: thin wrapper
out = keras.layers.Dense(32, "relu")(whitened)
out = keras.layers.Dense(1)(out)
mlp = keras.Model(inputs, out)
```

### 2.4 Asks (new infrastructure)

| Item | Effort | Notes |
|---|---|---|
| `gauss_keras.GaussianizationLayer(flow, trainable=False)` | **S** (one wrapper) | A `keras.layers.Layer` that forwards through the flow and refuses gradient updates on its params. |
| Notebook `08_input_whitening_baseline.ipynb` | **M** | Adult + synthetic; ablation of whitening on/off. |

No new losses. The existing `fit_and_freeze` handles the offline step.

### 2.5 Tradeoffs

**Plus**

- *Pure preprocessing* — composes with any predictor, any task loss,
  any other fairness method.
- *Better-conditioned input* tends to speed up training and removes
  the need for per-feature scale tuning.
- Helps when features have heavy tails or multi-modal marginals.

**Minus**

- *Doesn't reduce fairness gap on its own.* If the bias was already in
  the joint of $(X, q)$, $T_X$ preserves it.
- One extra forward pass per minibatch. Tiny on CPU; negligible on GPU.
- Interpretability cost: a "feature value" is now in Gaussianised space,
  so domain-meaningful values (`age = 45`) need an inverse pass to
  recover.

### 2.6 Hypothesis

```{admonition} H-A — Whitening is a fairness multiplier, not a fairness source.
:class: hypothesis

Whitening inputs improves the predictor's task loss (by 1–3% on Adult)
and **flattens the Pareto curve** of any downstream fairness method
(G-XCOV, G-MI, CKA) — i.e. the accuracy cost per unit of fairness
shrinks — but does not on its own reduce the achievable DP-diff floor.

**Failure prediction.** If the input distribution is already close to
Gaussian (e.g. after standard z-scoring of mostly-normal numeric
features), the whitening Pareto curve overlaps the no-whitening curve
within seed noise and adds zero value.
```

---

(sec-approach-b)=
## 3. Approach B — Fair feature selection

Use per-feature Gaussianised dependence to rank features by their
"q-leakage" and select the most-independent subset — a non-linear
generalisation of the linear-{abbr}`CKA` filter
{cite:p}`cortes2012cka`. The flow's job is to compute a **score per
feature**, not to enter the predictor's forward pass.

### 3.1 Math

For each feature dimension $i \in \{1, \ldots, d\}$:

$$
\rho_i^{\text{G}} \;=\; \widehat{\text{Corr}}\bigl(T_{X_i}(X_i), T_q(q)\bigr) \;\in\; [-1, 1],
$$ (eq-rho-g)

where $T_{X_i}$ is a per-feature 1-D Gaussianization flow (or, more
efficiently, the $i$-th marginal of a joint $T_X$). Rank features by
$|\rho_i^{\text{G}}|$ ascending; select the top-$K$ smallest:

$$
S_K \;=\; \operatorname*{arg\,min}_{|S| = K} \sum_{i \in S} |\rho_i^{\text{G}}|.
$$ (eq-topk-subset)

Train the predictor on $X_{:, S_K}$.

**Soft variant.** Replace the hard top-$K$ with a learnable sigmoid mask
$m \in [0, 1]^d$:

$$
\mathcal{L}(\theta, m) \;=\; \mathcal{L}_{\text{task}}\bigl(f_\theta(m \odot X), y\bigr)
                       \;+\; \lambda \sum_i m_i\, |\rho_i^{\text{G}}|
                       \;+\; \gamma \|m\|_1,
$$ (eq-soft-mask)

with $\rho_i^{\text{G}}$ pre-computed (frozen).  This is end-to-end
differentiable in $m$ and $\theta$, with $\rho^{\text{G}}$ as a fixed
weight vector.

**Higher-order variant.** Replace $\rho_i^{\text{G}}$ with {abbr}`G-MI`
or {abbr}`G-TC` per feature — picks up non-monotone dependence that the Pearson-corr
analog (`|cor(X_i, q)|`) misses.

### 3.2 Pipeline

```{mermaid}
flowchart LR
    subgraph OFF["Offline · one pass"]
        direction TB
        TXi["T<sub>X<sub>i</sub></sub><br/><i>per-feature MLE</i>"]:::frozen
        Tq2["T<sub>q</sub><br/><i>MLE</i>"]:::frozen
        Score["ρ<sub>i</sub><sup>G</sup> = Corr(T<sub>X<sub>i</sub></sub>(X<sub>i</sub>), T<sub>q</sub>(q))"]:::compute
        TXi --> Score
        Tq2 --> Score
        Score --> Sel[["rank by |ρ<sup>G</sup>|<br/><b>select top-K</b>"]]:::output
    end
    subgraph TRN["Training loop"]
        direction LR
        Xk[("X[:, S<sub>K</sub>]")]:::data --> F["f<sub>θ</sub>"]:::trainable --> L{{"task loss<br/><i>no fairness penalty</i><br/><i>(or soft mask: ρ<sup>G</sup> as fixed regulariser)</i>"}}:::loss
    end
    Sel -. selected features .-> Xk

    classDef data fill:#f1f5f9,stroke:#475569,stroke-width:1.5px,color:#0f172a;
    classDef frozen fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,stroke-dasharray:5 3,color:#1e1b4b;
    classDef compute fill:#f5f3ff,stroke:#7c3aed,stroke-width:1.5px,color:#2e1065;
    classDef trainable fill:#fef3c7,stroke:#b45309,stroke-width:2px,color:#451a03;
    classDef loss fill:#fce7f3,stroke:#be185d,stroke-width:2px,color:#500724;
    classDef output fill:#dcfce7,stroke:#15803d,stroke-width:1.5px,color:#052e16;
```

### 3.3 Pseudocode

**Hard selection:**

```python
from gaussianization.fair import (
    fit_and_freeze,
    score_features_g,      # NEW
)

# Stage 1: per-feature flows + dependence scores
flows = [fit_and_freeze(X_train[:, i:i+1], ...)[0] for i in range(d)]
T_q, _ = fit_and_freeze(q_train.reshape(-1, 1), ...)
rho_g = score_features_g(X_train, q_train, flows, T_q, metric="g_xcov")
# rho_g : np.ndarray of shape (d,), values in [0, 1]

# Stage 2: select top-K least dependent features
S_K = np.argsort(np.abs(rho_g))[:K]

# Stage 3: standard training on the selected subset
mlp = build_mlp(input_dim=K)
mlp.fit(X_train[:, S_K], y_train, ...)
```

**Soft selection:**

```python
# rho_g pre-computed as above; freeze it as a non-trainable constant
class FeatureMaskedMLP(keras.Model):
    def __init__(self, d, rho_g, lam=1.0, gamma=0.01):
        super().__init__()
        # Trainable logits for the mask; sigmoid pushes to [0, 1]
        self.mask_logits = self.add_weight(shape=(d,), initializer="zeros")
        self.rho_g = ops.convert_to_tensor(rho_g, dtype="float32")
        self.mlp = build_mlp(d)
        self.lam, self.gamma = lam, gamma

    def call(self, x, training=False):
        m = ops.sigmoid(self.mask_logits)
        if training:
            self.add_loss(self.lam * ops.sum(m * ops.abs(self.rho_g)))
            self.add_loss(self.gamma * ops.sum(m))   # sparsity
        return self.mlp(x * m)
```

### 3.4 Asks (new infrastructure)

| Item | Effort | Notes |
|---|---|---|
| `gaussianization.fair.score_features_g(X, q, flows, T_q, metric)` | **S** | Vectorised per-feature scoring; returns shape `(d,)`. |
| `gaussianization.fair.fit_marginals(X, ...)` | **S** | Convenience: fit one 1-D flow per feature in parallel. |
| Notebook `08_fair_feature_selection.ipynb` | **L** | Bar chart of `|ρ^G|` vs `|Pearson|`, top-K predictor sweep, soft-mask sweep, Pareto curves. |

No new losses; the soft-variant uses standard `add_loss` plumbing.

### 3.5 Tradeoffs

**Plus**

- Once features are selected, **the predictor's training loop has no
  fairness logic** — composes with any architecture, any optimiser.
- Catches non-monotone proxies that the linear `|Pearson|` baseline
  misses. (E.g. a feature with a U-shaped relationship to gender: linear
  corr ≈ 0, Gaussianised corr huge.)
- Interpretable: one score per feature, easy to communicate to
  stakeholders.
- Naturally extends to *soft* selection with a sparsity-regularised
  mask, which is end-to-end differentiable.

**Minus**

- Hard selection loses information — some unfair features are also
  predictive, and dropping them costs accuracy.
- Per-feature flows ignore correlations across features. A bivariate
  proxy (`(X_5, X_8)` together leak $q$ but neither alone does) is
  invisible. Use a joint flow or HSIC-over-feature-blocks to catch this.
- Static — the dependence ranking is computed once, doesn't adapt to
  which features the predictor actually uses.

### 3.6 Hypothesis

```{admonition} H-B — Gaussianised feature ranking dominates Pearson-based ranking.
:class: hypothesis

On UCI Adult Census, selecting the top-$K$ least-`|ρ^G|` features
produces a strictly Pareto-better (AUC, DP-diff) curve than selecting
the top-$K$ least-`|Pearson|` features, at every $K \in \{2, 3, 4, 5\}$,
because at least one Adult feature has a non-monotone dependence on
gender that the linear measure misses.

**Failure prediction.** If every Adult feature's joint distribution
with gender is well-approximated by its first two moments, the two
rankings coincide and there's no advantage. (Possible — Adult's
numeric features are mostly low-cardinality.)
```

---

(sec-approach-c)=
## 4. Approach C — Gaussianised subspace projection

Generalise B from "drop features" to "project out the q-direction in
Gaussianised space" — a Gaussianisation analogue of fair PCA
{cite:p}`olfat2018fairpca`. Same flow, more powerful selection.

### 4.1 Math

Pretrain a *joint* flow $T_X : X \to Z \in \mathbb{R}^d$ with
$Z \sim \mathcal{N}(0, I)$. The flow's rotation layers have implicitly
chosen a basis for the Gaussianised latent space. In that basis, the
"q-direction" is the cross-covariance:

$$
u_q \;=\; \widehat{\mathrm{Cov}}(Z, T_q(q)) \;\in\; \mathbb{R}^{d \times d_q}.
$$ (eq-uq-direction)

**Hard projection (single sensitive attribute, $d_q = 1$).** The
unit-length q-direction in $Z$-space is
$\hat u_q = u_q / \|u_q\|_2$. The orthogonal projection is

$$
P \;=\; I_d - \hat u_q \hat u_q^\top, \qquad Z' = Z P.
$$ (eq-orth-proj)

By construction $\widehat{\mathrm{Cov}}(Z', T_q(q)) = 0$ — the linear
component of dependence is zero. Because $Z, T_q(q)$ are marginally
Gaussian, this is most of the dependence.

**Hard projection (multi-class, $d_q > 1$).** SVD of $u_q$, project
onto the orthogonal complement of its $k$ largest singular vectors.
Strips the top-$k$ q-correlated directions.

**Soft projection (learnable basis).** Parameterise a linear map
$P \in \mathbb{R}^{d \times k}$ with orthogonality constraint, train
end-to-end with task loss and a G-XCOV penalty on $P^\top Z$:

$$
\min_{\theta, P : P^\top P = I_k}
   \mathcal{L}_{\text{task}}\bigl(f_\theta(P^\top Z), y\bigr)
   + \mu\, \mathcal{L}_{\text{G-XCOV}}\bigl(P^\top Z, T_q(q)\bigr).
$$ (eq-soft-stiefel)

The orthogonality constraint can be enforced via Stiefel-manifold
optimisation or a soft penalty $\|P^\top P - I_k\|_F^2$.

### 4.2 Pipeline

```{mermaid}
flowchart LR
    subgraph OFF["Offline"]
        direction TB
        Tx["T<sub>X</sub><br/><i>joint flow, frozen</i>"]:::frozen
        Tq3["T<sub>q</sub><br/><i>frozen</i>"]:::frozen
        Uq["u<sub>q</sub> = Cov(T<sub>X</sub>(X), T<sub>q</sub>(q))"]:::compute
        P["P = I − û<sub>q</sub> û<sub>q</sub><sup>T</sup><br/><i>frozen projection</i>"]:::frozen
        Tx --> Uq
        Tq3 --> Uq
        Uq --> P
    end
    subgraph TRN["Training loop"]
        direction LR
        X[("X")]:::data --> Tx2["T<sub>X</sub>"]:::frozen --> Z["Z"]:::compute --> Pp["P"]:::frozen --> Zp(["Z'"]):::output --> F["f<sub>θ</sub>"]:::trainable --> L{{"task loss<br/><i>no penalty for hard variant;<br/>add G-XCOV on P<sup>T</sup>Z for soft</i>"}}:::loss
    end
    Tx -. reused .-> Tx2
    P -. reused .-> Pp

    classDef data fill:#f1f5f9,stroke:#475569,stroke-width:1.5px,color:#0f172a;
    classDef frozen fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,stroke-dasharray:5 3,color:#1e1b4b;
    classDef compute fill:#f5f3ff,stroke:#7c3aed,stroke-width:1.5px,color:#2e1065;
    classDef trainable fill:#fef3c7,stroke:#b45309,stroke-width:2px,color:#451a03;
    classDef loss fill:#fce7f3,stroke:#be185d,stroke-width:2px,color:#500724;
    classDef output fill:#dcfce7,stroke:#15803d,stroke-width:1.5px,color:#052e16;
```

### 4.3 Pseudocode

**Hard projection:**

```python
from gaussianization.fair import fit_and_freeze, q_orthogonal_projection

T_X, _ = fit_and_freeze(X_train, num_blocks=8, ...)        # joint flow
T_q, _ = fit_and_freeze(q_train.reshape(-1, 1), ...)

Z_train = np.asarray(T_X(X_train))
Q_train = np.asarray(T_q(q_train.reshape(-1, 1)))
P = q_orthogonal_projection(Z_train, Q_train)             # NEW: (d, d) matrix

# Predictor sees the projected representation
mlp = build_mlp(input_dim=d)
def features(X):
    return ops.matmul(T_X(X), P)
mlp.fit(features(X_train), y_train, ...)
```

**Soft variant** (Stiefel-soft):

`GaussianizedXCovLoss` would re-apply `T_X` to its `z_pred` input and
also shape-mismatch when `k != d`, so we compute the cross-covariance
penalty directly on the already-projected $Z_p$ against $T_q(q)$:

```python
class FairProjMLP(keras.Model):
    def __init__(self, d, k, T_X, T_q, mu=1.0, ortho_lam=10.0):
        super().__init__()
        self.T_X, self.T_q = T_X, T_q
        self.P = self.add_weight(
            shape=(d, k), initializer="orthogonal"
        )
        self.mlp = build_mlp(input_dim=k)
        self.mu, self.ortho_lam = mu, ortho_lam

    def _xcov_penalty(self, Zp, q):
        # ||Cov(Zp, T_q(q))||_F^2  / (||Cov(Zp)||_F · ||Cov(T_q(q))||_F)
        qg = self.T_q(q)
        Zp_c = Zp - ops.mean(Zp, axis=0, keepdims=True)
        qg_c = qg - ops.mean(qg, axis=0, keepdims=True)
        n = ops.cast(ops.shape(Zp_c)[0], Zp_c.dtype)
        denom = ops.maximum(n - 1.0, 1.0)
        C    = ops.matmul(ops.transpose(Zp_c), qg_c) / denom
        S_z  = ops.matmul(ops.transpose(Zp_c), Zp_c) / denom
        S_q  = ops.matmul(ops.transpose(qg_c), qg_c) / denom
        fz = ops.sqrt(ops.sum(S_z * S_z)); fq = ops.sqrt(ops.sum(S_q * S_q))
        return ops.sum(C * C) / (fz * fq + 1e-12)

    def call(self, inputs, training=False):
        x, q = inputs["x"], inputs["q"]
        Z = self.T_X(x)
        Zp = ops.matmul(Z, self.P)
        if training:
            # Fairness penalty on the projected latent (k-dim, not d-dim)
            self.add_loss(self.mu * self._xcov_penalty(Zp, q))
            # Orthogonality constraint as soft penalty
            PtP = ops.matmul(ops.transpose(self.P), self.P)
            self.add_loss(self.ortho_lam *
                          ops.sum((PtP - ops.eye(self.P.shape[1])) ** 2))
        return self.mlp(Zp)
```

(For the eventual implementation, factor out `_xcov_penalty` as a free
function — it's the same linear-CKA computation used in
`GaussianizedXCovLoss`, just without the leading $T_z$ pass.)

### 4.4 Asks

| Item | Effort | Notes |
|---|---|---|
| `gaussianization.fair.q_orthogonal_projection(Z, Q, rank=1)` | **S** | SVD-based; handles multi-dim $q$. |
| `gaussianization.fair.FairProjModel` (soft variant) | **M** | Composes `GaussianizationLayer` (Approach A) + Stiefel-soft trainable $P$. |
| Notebook `09_subspace_projection.ipynb` | **L** | Hard vs soft on Adult; orthogonality monitoring. |

### 4.5 Tradeoffs

**Plus**

- Hard variant has *closed form* — no extra training, just one SVD.
- Captures the *direction* of dependence in $Z$-space, not merely
  dimensional selection. Stronger than B in two ways: (i) handles
  bivariate proxies (info leakage that lives across two original
  features), (ii) removes only the $q$-correlated component, keeping the
  rest of each feature.
- Composes with any of our existing losses (project first, then add
  G-XCOV on top — defence in depth).

**Minus**

- The projection is linear *in $Z$-space*, which is non-linear in
  $X$-space (the flow is non-linear). So the "fair subspace" doesn't
  have a clean interpretation in original feature units. Interpretation
  requires composing with $T_X^{-1}$.
- One direction at a time; multi-q (race in COMPAS) needs SVD or
  iterative.
- Information loss — removes $k$ dimensions of variance entirely.
  Predictive signal that happened to align with $u_q$ is gone.

### 4.6 Hypothesis

```{admonition} H-C — Hard projection matches the strongest soft-penalty floor at zero training cost.
:class: hypothesis

On UCI Adult, hard q-orthogonal projection of $T_X(X)$ achieves the
same DP-diff floor as G-MI at maximum $\mu$, but the predictor never
sees the fairness penalty — it's a *pure preprocessing fix*. AUC of
the projected-predictor matches the AUC of the strongest
soft-fairness predictor at matched DP-diff.

**Failure prediction.** If Adult's $u_q$ has more than 2 strong
singular directions (race + gender + something else), projecting out
just rank-1 is insufficient and the floor is reached only at rank-3+
projection — at which point the soft variants may be cheaper.
```

---

(sec-approach-d)=
## 5. Approach D — Conditional flow $T_{X \mid q}$

The most ambitious — and close in spirit to the conditional normalising
flows of {cite:t}`winkler2019cnf` and the invariant-representation
objective of {cite:t}`moyer2018invariant`. A flow whose parameters
depend on $q$, Gaussianising $X$ *given* $q$. The residual is
*structurally* independent of $q$.

### 5.1 Math

Train a conditional Gaussianization flow

$$
T_{X \mid q}: \mathbb{R}^d \times \mathbb{R}^{d_q} \to \mathbb{R}^d
$$ (eq-txq-type)

such that for every value of $q$,

$$
T_{X \mid q}(X, q) \;\sim\; \mathcal{N}(0, I_d) \quad \text{when } X \sim p(X \mid q).
$$ (eq-txq-objective)

Freeze. Train a predictor on the residual $Z = T_{X \mid q}(X, q)$:

$$
f_\theta(Z) \approx y \qquad \text{with} \qquad Z \perp q \text{ by construction.}
$$ (eq-residual-predictor)

**Mechanism.** Coupling-layer Gaussianization with FiLM conditioning:
each coupling layer's conditioner MLP takes both the active half of $X$
*and* the conditioning $q$, and produces shift/scale parameters that
depend on both. The marginal Gaussianization layers' mixture-CDF
parameters also become $q$-dependent (a small MLP from $q$ to mixture
parameters).

**Why it's "by construction".** $Z = T_{X \mid q}(X, q)$ has the same
distribution $\mathcal{N}(0, I)$ for *every* value of $q$ — that's the
training objective. So $p(Z \mid q) = p(Z)$, i.e. $Z \perp q$.

### 5.2 Pipeline

```{mermaid}
flowchart LR
    subgraph OFF["Offline · expensive"]
        direction TB
        X1[("X")]:::data
        q1[("q")]:::data
        Txq["T<sub>X|q</sub><br/><i>conditional flow<br/>MLE on N(0, I) for every q<br/>→ freeze</i>"]:::frozen
        X1 --> Txq
        q1 --> Txq
    end
    subgraph TRN["Training loop"]
        direction LR
        X2[("X")]:::data
        q2[("q")]:::data
        Txq2["T<sub>X|q</sub>"]:::frozen
        Z(["Z = T<sub>X|q</sub>(X, q)<br/><b>Z ⊥ q by construction</b>"]):::output
        F["f<sub>θ</sub>"]:::trainable
        L{{"task loss<br/><i>no fairness penalty</i>"}}:::loss
        X2 --> Txq2
        q2 --> Txq2
        Txq2 --> Z --> F --> L
    end
    Txq -. reused frozen .-> Txq2

    classDef data fill:#f1f5f9,stroke:#475569,stroke-width:1.5px,color:#0f172a;
    classDef frozen fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,stroke-dasharray:5 3,color:#1e1b4b;
    classDef trainable fill:#fef3c7,stroke:#b45309,stroke-width:2px,color:#451a03;
    classDef loss fill:#fce7f3,stroke:#be185d,stroke-width:2px,color:#500724;
    classDef output fill:#dcfce7,stroke:#15803d,stroke-width:1.5px,color:#052e16;
```

### 5.3 Pseudocode

```python
from gaussianization.gauss_keras.conditional import (
    ConditionalGaussianizationFlow,     # NEW class
)
from gaussianization.fair import freeze_flow

# Stage 1: pretrain conditional flow
T_xq = ConditionalGaussianizationFlow(
    input_dim=d,
    cond_dim=d_q,
    num_blocks=8,
    num_components=12,
)
T_xq.compile(optimizer=keras.optimizers.Adam(1e-3), loss=base_nll_loss)
T_xq.fit(
    [X_train, q_train],   # input is a (data, condition) pair
    X_train,              # NLL target
    epochs=200,
    batch_size=256,
)
freeze_flow(T_xq)

# Stage 2: standard predictor on the residual
def residual(X, q):
    return T_xq([X, q])

mlp = build_mlp(input_dim=d)
mlp.compile(optimizer="adam", loss="mse")
mlp.fit(residual(X_train, q_train), y_train, ...)
```

### 5.4 Asks

| Item | Effort | Notes |
|---|---|---|
| `gauss_keras.bijectors.MixtureCDFGaussianization` accepts a `condition` input | **M** | FiLM-style conditioning on mixture params via a small head MLP. |
| `gauss_keras.bijectors.MixtureCDFCoupling` already takes a conditioner — extend its conditioner to accept `q` alongside the active half. | **S** | One arg change. |
| `gauss_keras.flows.ConditionalGaussianizationFlow` | **M** | Threads `q` through every layer; subclasses or wraps existing `GaussianizationFlow`. |
| `fair.fit_and_freeze_conditional(X, q, ...)` | **S** | Convenience helper. |
| Notebook `10_conditional_flow.ipynb` | **L** | Comparison against A–C on Adult; sanity check $Z \perp q$ after freezing. |

This is the **most invasive** change: it touches the core `gauss_keras`
library, not just `fair/`. Worth doing because conditional flows are
broadly useful (density estimation conditional on covariates).

### 5.5 Tradeoffs

**Plus**

- *Structurally* enforces $Z \perp q$ — strongest fairness guarantee.
- The predictor cannot reverse-engineer $q$ from $Z$ no matter how hard
  it tries (the information is gone).
- Naturally handles continuous $q$ (the conditional flow's parameters
  interpolate). Binary, multi-class, real-valued sensitive attributes
  all the same.

**Minus**

- *Information loss is unbounded.* Removes everything about $X$ that
  varies with $q$ — including useful predictive signal. The "fair"
  representation is structurally pure but may be predictively poor.
- Most expensive to pretrain: per-$q$ Gaussianisation. The flow has to
  model $p(X \mid q)$ everywhere, not just $p(X)$.
- Architectural lift to `gauss_keras` is non-trivial.
- "Frozen" is now more fragile: at inference time we need to apply
  $T_{X \mid q}$ with the *test* $q$, and if test $q$ has a value never
  seen during pretraining (continuous $q$, distribution shift), the flow
  is off-support.

### 5.6 Hypothesis

```{admonition} H-D — Conditional flow gives exact $Z \perp q$ (DP-diff = 0); EO-diff is *not* guaranteed; biggest AUC cost.
:class: hypothesis

A predictor on $T_{X \mid q}(X, q)$ achieves **demographic-parity
difference = 0** by construction (within numerical noise), because
$Z \perp q$ implies $f_\theta(Z) \perp q$ for any deterministic
$f_\theta$.

**Equalized-odds difference is *not* guaranteed to be zero**: EO
conditions on the true label $y$, and when $y$ is correlated with $q$
(which is the case in any non-trivial fairness benchmark), conditioning
on $y$ can reintroduce dependence between $f_\theta(Z)$ and $q$ even
when the marginal $Z \perp q$ holds. EO-diff = 0 would additionally
require $y \perp q$ in the data, which Adult Census clearly violates.

In practice we expect a substantial AUC cost compared to A–C — the
conditional flow strips all $q$-conditioned predictive signal,
including the legitimate component (e.g. real education effects
across groups).

**Failure prediction.** If $q$ has very little information about
predictively-useful aspects of $X$, the conditional flow's removal
costs nothing and D matches the AUC of A–C at perfect fairness — i.e.
the AUC penalty for "exact fairness" was illusory all along.
```

---

(sec-approach-e)=
## 6. Approach E — Counterfactual sample augmentation

Use the (conditional) flow's *inverse* pass to generate counterfactual
$\tilde X$ with $q$ flipped, then train a predictor to make the same
decision on both. Targets *individual* counterfactual fairness in the
sense of {cite:t}`kusner2017cf`, not just population-level statistics.

### 6.1 Math

For each training example $(X_i, q_i, y_i)$, define the counterfactual

$$
\tilde X_i \;=\; T_{X \mid 1 - q_i}^{-1}\bigl(T_{X \mid q_i}(X_i, q_i),\, 1 - q_i\bigr),
$$ (eq-cf-sample)

i.e. Gaussianise $X_i$ given $q_i$, then *invert* the Gaussianisation
under the opposite $q$-value. The result has the same "position in the
Gaussianised latent" but the marginal of the opposite group.

(For continuous or multi-class $q$, swap $1 - q_i$ for a chosen
reference value or sample of values.)

Augmented dataset: $D' = \{(X_i, q_i, y_i), (\tilde X_i, 1 - q_i, y_i)\}$.
Train with a **consistency loss**:

$$
\mathcal{L}(\theta) \;=\; \mathcal{L}_{\text{task}}(f_\theta(X), y)
   \;+\; \lambda \, \mathbb{E}_i \bigl\|f_\theta(X_i) - f_\theta(\tilde X_i)\bigr\|^2.
$$ (eq-cf-consistency)

The consistency term explicitly says: an individual's prediction must
not change if you flip their sensitive attribute (Kusner et al. 2017,
"Counterfactual Fairness").

### 6.2 Pipeline

```{mermaid}
flowchart LR
    subgraph OFF["Offline · one pass over training set"]
        direction TB
        Pair[("(X<sub>i</sub>, q<sub>i</sub>)")]:::data
        FwdInv["X̃<sub>i</sub> = T<sub>X|1−q<sub>i</sub></sub><sup>−1</sup>(<br/>&nbsp;&nbsp;T<sub>X|q<sub>i</sub></sub>(X<sub>i</sub>, q<sub>i</sub>),<br/>&nbsp;&nbsp;1 − q<sub>i</sub>)"]:::compute
        Pair --> FwdInv
        FwdInv --> Aug[["augmented (X ∪ X̃, y ∪ y)"]]:::output
    end
    subgraph TRN["Training loop · shared θ"]
        direction LR
        Xi[("X<sub>i</sub>")]:::data --> F1["f<sub>θ</sub>"]:::trainable --> Ltask{{"task loss"}}:::loss
        Xt[("X̃<sub>i</sub>")]:::data --> F2["f<sub>θ</sub>"]:::trainable --> Lcons{{"consistency loss<br/>λ ‖f<sub>θ</sub>(X<sub>i</sub>) − f<sub>θ</sub>(X̃<sub>i</sub>)‖<sup>2</sup>"}}:::loss
        F1 -. shared weights .- F2
    end
    Aug -. paired samples .-> Xi
    Aug -. paired samples .-> Xt

    classDef data fill:#f1f5f9,stroke:#475569,stroke-width:1.5px,color:#0f172a;
    classDef compute fill:#f5f3ff,stroke:#7c3aed,stroke-width:1.5px,color:#2e1065;
    classDef trainable fill:#fef3c7,stroke:#b45309,stroke-width:2px,color:#451a03;
    classDef loss fill:#fce7f3,stroke:#be185d,stroke-width:2px,color:#500724;
    classDef output fill:#dcfce7,stroke:#15803d,stroke-width:1.5px,color:#052e16;
```

### 6.3 Pseudocode

```python
from gaussianization.fair import (
    generate_counterfactuals,      # NEW
    CounterfactualConsistencyLoss, # NEW
)

# Stage 1: pretrain a conditional flow (Approach D's machinery)
T_xq, _ = fit_and_freeze_conditional(X_train, q_train, ...)

# Stage 2: generate counterfactuals for the training set
X_tilde = generate_counterfactuals(T_xq, X_train, q_train)
# X_tilde[i] is the counterfactual of X_train[i] with q flipped

# Stage 3: train predictor with consistency loss
class FairCFMLP(keras.Model):
    def __init__(self, d, lam=1.0):
        super().__init__()
        self.mlp = build_mlp(d)
        self.lam = lam

    def call(self, inputs, training=False):
        x, x_tilde = inputs["x"], inputs["x_tilde"]
        y_hat = self.mlp(x)
        if training:
            y_hat_tilde = self.mlp(x_tilde)
            self.add_loss(self.lam * ops.mean((y_hat - y_hat_tilde) ** 2))
        return y_hat

model = FairCFMLP(d, lam=1.0)
model.compile(optimizer="adam", loss="mse")
model.fit({"x": X_train, "x_tilde": X_tilde}, y_train, ...)
```

### 6.4 Asks

| Item | Effort | Notes |
|---|---|---|
| Approach D's `ConditionalGaussianizationFlow` and its inverse | **L** | Big — see Approach D. |
| `generate_counterfactuals(flow, X, q)` | **S** | One forward + one inverse pass per batch; precomputable. |
| `CounterfactualConsistencyLoss` (or use plain `add_loss` as above) | **S** | Just an MSE between two predictor calls. |
| Notebook `11_counterfactual_fairness.ipynb` | **L** | Visualise counterfactual quality (a few $X_i, \tilde X_i$ pairs); individual fairness metrics. |

Building blocks: needs Approach D first.

### 6.5 Tradeoffs

**Plus**

- Targets **individual** counterfactual fairness — "Would the same
  applicant get the same prediction if their gender were flipped?" —
  which population-level DP/EO doesn't capture.
- Composes with any predictor architecture.
- Naturally handles continuous $q$: average consistency loss over
  multiple counterfactual draws from $p(q)$.

**Minus**

- Needs an **invertible** flow with a faithful conditional. Counterfactuals
  are only as good as $T_{X \mid q}$'s coverage.
- Doubles dataset size (or doubles forward passes per step if
  precomputed). Manageable.
- "Counterfactual" is a fantasy when the actual $(X, q)$ joint has no
  support at $(\tilde X, 1 - q_i)$. Adult example: if a feature like
  "occupation = mining engineer" essentially never co-occurs with
  female in the data, the counterfactual is extrapolating. Honest
  warning needed.

### 6.6 Hypothesis

```{admonition} H-E — Counterfactual augmentation reduces individual flip-rate even when DP-diff is already low.
:class: hypothesis

For a predictor trained with G-XCOV penalty (notebook 07) at the
strongest $\mu$ that still keeps usable AUC, the *individual flip rate*
(% of test samples whose predicted label changes if $q$ is flipped via
the conditional flow) is still nontrivial (say, 5–8% on Adult).
Adding the counterfactual consistency loss drives this to ≤ 1%
without further DP-diff improvement, demonstrating that
individual-level fairness is a *strictly separate* axis.

**Failure prediction.** If the conditional flow's counterfactual
quality is poor (e.g. the inverse pass produces out-of-distribution
$\tilde X$), the consistency loss optimises a phantom and the flip
rate doesn't drop.
```

---

(sec-approach-f)=
## 7. Approach F — Density-ratio reweighting

Use the (conditional) flow's log-density to estimate per-sample
weights that **rebalance the training set**. Classical importance
weighting in the style of {cite:t}`caldersVerwer2010nbfair`, with
flow-based densities replacing the usual kernel-density or
logistic-regression-style propensity estimates.

### 7.1 Math

Two equivalent formulations:

**Group-balanced reweighting.** Estimate the group-conditional density
ratio:

$$
w_i \;=\; \frac{p(X_i \mid q = 0)}{p(X_i \mid q = q_i)}
\quad\Longrightarrow\quad
\mathbb{E}_{w}\bigl[L_{\text{task}}(f_\theta(X), y) \mid q = q_i\bigr]
\;=\;
\mathbb{E}\bigl[L_{\text{task}} \mid q = 0\bigr]
\quad \forall q_i.
$$ (eq-ipw-balance)

That is, the weighted task loss is the same across groups — closing
the population-level disparity without a fairness penalty.

**Inverse-propensity reweighting.** Use the flow to estimate
$p(q \mid X)$ via Bayes, then $w_i = 1 / p(q_i \mid X_i)$.

Both forms come from the *same* set of densities; the choice is just
which factorisation is more numerically stable.

The flow estimates $p(X \mid q)$ directly via $\log p(X \mid q) = \log
\mathcal{N}(T_{X \mid q}(X, q); 0, I) + \log |\det \nabla T_{X \mid q}|$.

### 7.2 Pipeline

```{mermaid}
flowchart LR
    subgraph OFF["Offline · weight estimation"]
        direction TB
        Txq3["T<sub>X|q</sub><br/><i>conditional flow, frozen</i>"]:::frozen
        LP1["log p(X<sub>i</sub> | q<sub>i</sub>)"]:::compute
        LP0["log p(X<sub>i</sub> | q = 0)"]:::compute
        Txq3 --> LP1
        Txq3 --> LP0
        W[["w<sub>i</sub> = exp(log p<sub>0</sub> − log p<sub>q<sub>i</sub></sub>)<br/><i>clip to bound variance</i>"]]:::output
        LP1 --> W
        LP0 --> W
    end
    subgraph TRN["Training loop"]
        direction LR
        X[("X")]:::data --> F["f<sub>θ</sub>"]:::trainable --> L{{"<b>weighted</b> task loss<br/><i>no penalty; effective sample<br/>size shrinks with clip</i>"}}:::loss
    end
    W -. per-sample weights .-> L

    classDef data fill:#f1f5f9,stroke:#475569,stroke-width:1.5px,color:#0f172a;
    classDef frozen fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,stroke-dasharray:5 3,color:#1e1b4b;
    classDef compute fill:#f5f3ff,stroke:#7c3aed,stroke-width:1.5px,color:#2e1065;
    classDef trainable fill:#fef3c7,stroke:#b45309,stroke-width:2px,color:#451a03;
    classDef loss fill:#fce7f3,stroke:#be185d,stroke-width:2px,color:#500724;
    classDef output fill:#dcfce7,stroke:#15803d,stroke-width:1.5px,color:#052e16;
```

### 7.3 Pseudocode

```python
from gaussianization.fair import (
    fit_and_freeze_conditional,
    density_ratio_weights,         # NEW
)

# Stage 1: conditional flow gives log p(X | q)
T_xq, _ = fit_and_freeze_conditional(X_train, q_train, ...)

# Stage 2: per-sample weights w_i = p(X_i | q=0) / p(X_i | q=q_i)
w_train = density_ratio_weights(
    T_xq, X_train, q_train, target_q=0, clip=10.0,
)
# Returns shape (n,), positive, normalised to mean 1.

# Stage 3: standard weighted training
mlp = build_mlp(d)
mlp.compile(optimizer="adam", loss="mse")
mlp.fit(X_train, y_train, sample_weight=w_train, ...)
```

### 7.4 Asks

| Item | Effort | Notes |
|---|---|---|
| Conditional flow log-density (Approach D's machinery) | **L** | Needed first. |
| `density_ratio_weights(flow, X, q, target_q, clip)` | **S** | One-liner once log-density is available; clipping handles tail. |
| Notebook `12_density_ratio_reweighting.ipynb` | **M** | Pareto: G-XCOV penalty vs IPW weighting. |

### 7.5 Tradeoffs

**Plus**

- *Classical* importance-weighting machinery — well-understood
  theoretically. Pearl & co. would approve.
- Composes with any task loss; just multiply by weights.
- One-time per-sample weight computation — training is otherwise
  vanilla.
- Direct attack on population-level fairness; can be cleanly combined
  with E (CF augmentation) for individual + population fairness.

**Minus**

- Per-sample weights can be high-variance for tails of $p(X \mid q)$.
  Need clipping (`w_i = min(w_i, 10)`) which biases the estimator.
- Effective sample size shrinks. If one group is very different from
  the target, you're effectively training on the overlap.
- Needs robust flow log-density estimates. Mixture-CDF
  Gaussianisation gives them cleanly (analytic Jacobian), but a
  miscalibrated flow translates 1:1 into miscalibrated weights.

### 7.6 Hypothesis

```{admonition} H-F — IPW matches G-XCOV's mid-curve with simpler training.
:class: hypothesis

On UCI Adult, density-ratio reweighting (clipped at $w \in [0.1, 10]$)
achieves the (AUC, DP-diff) pair of G-XCOV at $\mu \approx 50$ without
adding any term to the training loss — just `sample_weight=w_train`.
At the extreme end of the curve, however, IPW saturates (effective
sample size collapses) while G-XCOV/G-MI can still trade more AUC for
fairness.

**Failure prediction.** If the two group-conditional densities have
limited overlap, IPW weights blow up, clipping kicks in heavily, and
the weighted estimator becomes badly biased — i.e. the Pareto curve
of IPW *crosses* G-XCOV's at low DP-diff.
```

---

(sec-approach-g)=
## 8. Approach G — Information-bottleneck on representations *(stretch)*

A seventh idea worth recording, even if it's the most speculative:
apply the fairness penalty to an *intermediate* layer of the
predictor, not to its output.

### 8.1 Sketch

Most fair-representation literature does exactly this — see e.g. the
{abbr}`VFAE` of {cite:t}`louizos2016vfae`. An encoder
$e_\phi : X \to R$, a head $h_\psi : R \to y$, and a penalty
$\mu \cdot \text{Dep}(R, q)$ on the bottleneck representation.  The
encoder learns a representation that is task-useful but
$q$-uninformative.

Drop in any of our existing losses on $R$ and $T_q(q)$:

$$
\min_{\phi, \psi}\ \mathcal{L}_{\text{task}}(h_\psi(e_\phi(X)), y)
   \;+\; \mu \, \mathcal{L}_{\text{G-MI}}\bigl(e_\phi(X),\, q\bigr).
$$ (eq-bottleneck-loss)

For G-MI we need a flow on $R$ — but $R$ is high-dim and *moves*
during training, same problem as the original output-side experiment.
Two workarounds:

1. **Periodic refresh:** refit $T_R$ on $\{e_\phi(X)\}$ every $N$
   epochs. Costs a few extra minutes; gives a fresh dependence probe.
2. **VAE-style structural fix:** make $R$ Gaussian by construction
   (e.g. with a KL-to-$\mathcal{N}(0, I)$ regulariser on $e_\phi$, like
   a VAE encoder). Now $T_R = \text{id}$ and G-XCOV reduces to plain
   linear cross-covariance on the bottleneck — cheap and exact.

### 8.2 Tradeoffs (briefly)

**Plus**: composes with any downstream architecture; lets a small
classifier head sit on top of a strongly-fair representation; the
representation itself can be reused for multiple downstream tasks.

**Minus**: moving target on $T_R$ (same risk as the original
experiment); requires architectural surgery on the predictor.

### 8.3 Hypothesis

```{admonition} H-G — Representation-level fairness transfers; output-level doesn't.
:class: hypothesis

A representation $e_\phi(X)$ trained with G-MI penalty against $q$ is
"q-uninformative" in a way that **transfers** to a second downstream
task $y'$ (different label, same sensitive attribute) — i.e. you can
freeze $e_\phi$ and train a new head on a new task and still get DP-diff
near the headline. Output-level fairness from the original experiment
does not transfer.

**Failure prediction.** If the encoder is over-regularised and loses
*all* signal correlated with $q$ (including the legitimate signal),
the second task fails too — the representation is too narrow.
```

---

(sec-followup-matrix)=
## 9. Cross-cutting comparison matrix

| | A (whiten) | B (select) | C (project) | D (cond. flow) | E (CF aug.) | F (IPW) | G (bottleneck) |
|---|---|---|---|---|---|---|---|
| Flow on inputs? | ✅ joint | ✅ marginals | ✅ joint | ✅ conditional | ✅ conditional | ✅ conditional | ❌ representations |
| Flow frozen? | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ refresh |
| Predictor sees fairness penalty during training? | ❌ | ❌ (soft: ✅) | ❌ (soft: ✅) | ❌ | ✅ (consistency) | ❌ | ✅ |
| Pretraining cost | low | medium (d marginals) | medium | **high** | high | high | medium |
| Information loss | none | hard | rank-$k$ | total q-conditioned | none | importance-weighted | task-driven |
| Granularity of fairness | n/a | population | population | structural | individual | population | representation |
| Needs new core (`gauss_keras`) infra? | thin layer | n/a | n/a | **yes** | yes | yes | maybe |
| Composes with G-XCOV / G-MI? | ✅ | ✅ | ✅ | redundant | ✅ | ✅ | ✅ |
| Effort estimate (S/M/L) | S | M | M | **L** | L | M | L |

---

(sec-followup-sequencing)=
## 10. Recommended sequencing

```{mermaid}
flowchart TB
    subgraph R1["Round 1 · cheap, learn fast<br/><i>only needs fit_and_freeze (shipped)</i>"]
        direction LR
        A1["<b>A</b> · input whitening"]:::r1
        B1["<b>B</b> · fair feature selection"]:::r1
        C1["<b>C</b> · subspace projection (hard SVD)"]:::r1
    end

    subgraph R2["Round 2 · infrastructure lift<br/><i>build ConditionalGaussianizationFlow in gauss_keras</i>"]
        direction LR
        Build["Approach D's prereq"]:::r2
    end

    subgraph R3["Round 3 · uses the conditional flow"]
        direction LR
        D3["<b>D</b> · conditional flow predictor"]:::r3
        E3["<b>E</b> · counterfactual augmentation"]:::r3
        F3["<b>F</b> · density-ratio reweighting"]:::r3
    end

    subgraph R4["Round 4 · representation-level<br/><i>no urgent dependency</i>"]
        direction LR
        G4["<b>G</b> · information bottleneck (stretch)"]:::r4
    end

    R1 --> R2 --> R3 --> R4

    classDef r1 fill:#dcfce7,stroke:#15803d,stroke-width:1.5px,color:#052e16;
    classDef r2 fill:#fef3c7,stroke:#b45309,stroke-width:2px,color:#451a03;
    classDef r3 fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,color:#1e1b4b;
    classDef r4 fill:#f5f3ff,stroke:#7c3aed,stroke-width:1.5px,stroke-dasharray:5 3,color:#2e1065;
```

Round 1 alone is plausibly *the strongest paper* of the set — A+B+C
give three preprocessing-only fairness baselines that the original
output-side losses can be benchmarked against.  If Round 1 is sufficient
in practice, Rounds 2–4 become "did you really need to add a fairness
loss at all?" — a sharp result either way.

---

(sec-followup-api)=
## 11. New library additions, summarised

If we eventually ship Rounds 1–3, the public API of
`gaussianization.fair` grows by:

```
from gaussianization.fair import (
    # existing
    GaussianizedXCovLoss,
    GaussianizedMutualInfoLoss,
    GaussianizedTotalCorrelationLoss,
    fit_and_freeze,
    fit_and_freeze_joint,
    freeze_flow,
    is_fully_frozen,
    demographic_parity_difference,
    equalized_odds_difference,
    pearson_corr,

    # Round 1
    score_features_g,                  # B
    q_orthogonal_projection,           # C
    fit_marginals,                     # B convenience

    # Round 2 (infrastructure)
    fit_and_freeze_conditional,        # D

    # Round 3
    generate_counterfactuals,          # E
    density_ratio_weights,             # F
)
```

And `gauss_keras` grows:

```
from gaussianization.gauss_keras import (
    # existing
    GaussianizationFlow,
    make_gaussianization_flow,
    make_coupling_flow,
    ...

    # Round 1
    GaussianizationLayer,              # A: frozen-flow wrapper as keras.Layer

    # Round 2
    ConditionalGaussianizationFlow,    # D
)
```

Six new public symbols in `gaussianization.fair`
(`score_features_g`, `q_orthogonal_projection`, `fit_marginals`,
`fit_and_freeze_conditional`, `generate_counterfactuals`,
`density_ratio_weights`), two new classes in `gauss_keras`
(`GaussianizationLayer` — a thin frozen-flow wrapper for Approach A;
`ConditionalGaussianizationFlow` — the conditional flow for D/E/F),
and one extended bijector
(`MixtureCDFGaussianization` gains an optional `condition` input).
None of the existing API breaks.

---

(sec-followup-open)=
## 12. Open questions

1. **Joint flow vs $d$ marginal flows for Approach B.** Per-feature flows
   are independent and parallelisable, but they ignore cross-feature
   structure.  A joint flow scores features via its marginal projections
   but is harder to fit.  Worth a small ablation in the B notebook.

2. **Pre-vs-post Gaussianisation for $q$.** All approaches assume $q$
   has its own Gaussianisation flow $T_q$.  For binary $q$ this is
   overkill — $T_q$ is essentially a sign-flip + scale.  Is there an
   ablation showing $T_q$ matters?  Or can we use raw $q$ for the
   sensitive side when $d_q = 1$?

3. **Continuous $q$ semantics.**  For age-as-sensitive-attribute (a
   continuous variable), what does "DP-diff" even mean?  Approaches D
   and E need to specify a counterfactual policy: do we flip a 25-year-old
   to a 45-year-old, or to the average, or to the marginal distribution?

4. **Flow capacity vs predictor capacity.**  All approaches assume the
   flow has "enough" capacity to faithfully model $p(X)$, $p(X \mid q)$,
   etc.  An under-capacity flow gives bad scores / bad projections /
   bad counterfactuals — and the failure mode is silent (the predictor
   just inherits the flow's blind spots).  Possible mitigation:
   diagnostic that monitors per-feature $T_X$ log-likelihood on held-out
   data.

5. **Combining approaches.**  Nothing prevents stacking — e.g. whiten
   inputs (A), select features (B), project out residual q-direction
   (C), and *then* add a small G-XCOV penalty for defence in depth.
   Does that compound, or does each subsequent step add nothing?

---
