# Fair Learning with Gaussianization Flows — Engineering Doc

**Status:** Design / proposal
**Project:** `projects/gaussianization`
**Author:** J. E. Johnson
**Last updated:** 2026-05-27

---

## 1. TL;DR

Replace the **CKA fairness penalty** in the [`keras-fairkl`](https://github.com/jejjohnson/keras-fairkl)
`FairModelWrapper` with a **Gaussianization-Flow-based independence loss**.
The flow is trained once on an auxiliary dataset, its weights are frozen, and it
is then used as a differentiable "Gaussian-space probe" inside the downstream
task's optimisation loop. Because Gaussianization makes marginals (approximately)
standard Normal, **linear measures of dependence in flow space approximate
non-linear measures of dependence in data space** — which gives us a principled,
batch-friendly, fully differentiable replacement for CKA.

The experiment lives under `projects/gaussianization/` (Keras 3, JAX backend) and
reuses three existing assets:

| Asset | Role |
|---|---|
| `projects/gaussianization/.../gauss_keras` | Keras 3 Gaussianization Flow (pretrained then frozen) |
| `keras-fairkl` | `FairModelWrapper`, baseline `CKALoss`, datasets, training scaffold |
| `gauss_flows` (JAX) | Reference implementation + scientific validation harness |

---

## 2. Background & Motivation

### 2.1 Fair learning via dependence penalties

Following the kernel-fairness line (Pérez-Suay et al. 2017, Cortes et al. 2012),
`keras-fairkl` minimises

$$
\mathcal{L}(\theta) \;=\; \underbrace{\mathcal{L}_{\text{task}}(f_\theta(X), y)}_{\text{predictive}}
                       + \mu \underbrace{\mathrm{Dep}(f_\theta(X), q)}_{\text{fairness}}
$$

where $q$ is a sensitive attribute and `Dep` is a kernel statistic — currently
HSIC, MMD, or **CKA** (Centred Kernel Alignment, `fairkl.metrics.cka.CKALoss`).
CKA is bounded in $[0, 1]$, scale-invariant for linear kernels, and works as a
per-batch statistic via the `add_loss` mechanism inside
`FairModelWrapper.call` (`src/fairkl/models/fair_wrapper.py:155`).

### 2.2 Why try Gaussianization?

Gaussianization Flows (Chen & Gopinath 2000; Laparra et al. 2011 — "RBIG";
Meng et al. 2020 — "Gaussianization Flows") learn a diffeomorphism
$T: \mathbb{R}^d \to \mathbb{R}^d$ such that $T(X) \approx \mathcal{N}(0, I)$.
Three properties make them attractive for fairness:

1. **Bijectivity** — $T$ preserves information; independence in flow space is
   exactly equivalent to independence in data space.
2. **Marginal Gaussianity** — after $T$, marginal distributions are
   (approximately) standard Normal, so the *linear* covariance fully captures
   marginal-Gaussian dependence. Cross-correlation Frobenius norm in flow space
   is therefore a tight surrogate for non-linear dependence in data space.
3. **Closed-form differentiable density** — `log_prob` and `log_det_jacobian`
   are smooth in both inputs and parameters, so frozen flows compose naturally
   into downstream gradient-based optimisation.

Crucially, freezing $T$ collapses the bilevel "min-task while $T$ models
distribution" problem to a single-level problem with a fixed surrogate metric,
which is exactly the regime where `keras-fairkl`'s `add_loss` machinery thrives.

### 2.3 What's wrong with CKA that this fixes?

* **Bandwidth sensitivity.** RBF-CKA in `fairkl.metrics.cka.cka_rbf` requires
  picking `sigma_f, sigma_q`; the experiment author currently sets these by
  hand or tunes them with Keras Tuner. A pretrained flow absorbs the bandwidth
  choice into its learned components (mixture-of-CDF parameters).
* **Batch-size sensitivity.** CKA is a U/V-statistic that is noisy at small
  batches (see `CKALoss` docstring's warning at batch < 128). A flow-based
  test that operates on individual samples is less batch-bound.
* **Linear-in-RKHS only.** CKA is a Hilbert-Schmidt cross-norm — it captures
  second-order dependence in feature space. Flows model the full density.

---

## 3. Three-Repo Landscape

### 3.1 `keras-fairkl` — the fairness training surface

| Asset | Path | Notes |
|---|---|---|
| `FairModelWrapper` | `src/fairkl/models/fair_wrapper.py:26` | Wraps any `keras.Model`; injects penalty via `add_loss` in `call()` (line 155). Input protocol: dict `{"x": X, "q": q}`. |
| `CKALoss` | `src/fairkl/metrics/cka.py` | Baseline to beat. Constructor takes `sigma_f, sigma_q, kernel, debiased`. |
| `HSICLoss`, `MMDLoss` | `src/fairkl/metrics/{hsic,mmd}.py` | Alternative baselines. |
| `FairLinear`, `FairKernelRidge`, `FairPCA`, `FairKernelPCA` | `src/fairkl/models/` | Hard-wired predictors — useful for closed-form sanity checks. |
| Adult Census example | `docs/notebooks/fair_adult_census.py` | Drop-in reference experiment. |
| Backend | Keras 3, JAX backend tested | Reuse `KERAS_BACKEND=jax` to share infra with `gauss_keras`. |

The single extension point we need: **a `keras.losses.Loss` subclass with the
signature `__call__(q, f_pred) -> scalar`**, instantiable as
`fairness_loss=…` on the `FairModelWrapper`. Everything else is free.

### 3.2 `projects/gaussianization` — the Keras flow library

| Asset | Path | Notes |
|---|---|---|
| `GaussianizationFlow` | `src/gaussianization/gauss_keras/bijectors/flow.py:11` | `log_prob(x) -> (batch,)`, `sample(n, seed)`, `invert(z)`, `forward_with_intermediates(x)`. |
| `make_gaussianization_flow` | `src/gaussianization/gauss_keras/training.py:33` | Builds `(FixedOrtho → (Householder → MixtureCDFGaussianization) × N)`. Supports PCA + quantile init. |
| `make_coupling_flow` | `training.py:71` | Coupling-layer variant with MLP conditioners. |
| `base_nll_loss` | `training.py:22` | Use as Keras loss on the *latent* output during pretraining. |
| Backend | Pure Keras 3 (TF / JAX / PyTorch all work) | Same as fairkl — clean integration. |

### 3.3 `gauss_flows` — JAX reference

JAX/Equinox/FlowJax implementation. **Not used at runtime** in this
experiment (we stay in Keras for direct fairkl interop), but used to:

* Cross-validate density/log-prob numbers on shared test fixtures.
* Compare optimisation behaviour (Keras vs. FlowJax `fit_to_data`).
* Provide the `entropy(...)` helper as a sanity check on a pretrained flow.

Key files: `src/gauss_flows/_src/flows/gaussianization.py`,
`src/gauss_flows/_src/inference/train.py`.

---

## 4. Mathematical Formulation

Let $z = f_\theta(X) \in \mathbb{R}^{d_z}$ be the predictor output (or
embedding) and $q \in \mathbb{R}^{d_q}$ the sensitive attribute.

### 4.1 Approach A (recommended) — Gaussianised Cross-Covariance ("G-XCOV")

Pretrain two flows (or a single joint flow whose marginals we project):

$$
T_z : \mathbb{R}^{d_z} \to \mathbb{R}^{d_z}, \qquad
T_q : \mathbb{R}^{d_q} \to \mathbb{R}^{d_q}
$$

each by maximum likelihood under a $\mathcal{N}(0, I)$ base. After pretraining,
**freeze all parameters of $T_z, T_q$**. The fairness loss for a batch
$\{(z_i, q_i)\}_{i=1}^n$ is

$$
\mathcal{L}_{\text{fair}}^{\text{G-XCOV}}
\;=\;
\bigl\lVert \widehat{\mathrm{Cov}}\bigl(T_z(z), T_q(q)\bigr) \bigr\rVert_F^2,
$$

i.e. the Frobenius norm of the empirical cross-covariance computed *after*
Gaussianisation. Because $T_z, T_q$ are diffeomorphisms,
$T_z(z) \perp T_q(q) \iff z \perp q$. Because marginals are (approximately)
standard Normal, the cross-covariance is a tight low-order surrogate for full
non-linear dependence.

Equivalent reading: this is **CKA with a linear kernel applied in
Gaussianised space**. The flow replaces the RBF bandwidth.

Differentiability: $T_z$ is composed of `MixtureCDFGaussianization` (smooth via
`ndtri` / mixture-CDF) and `Householder` (linear). All ops are JAX-traceable
and `jax.lax.stop_gradient` on the flow's variables freezes them.

### 4.2 Approach B — Density-Ratio Mutual Information (DR-MI)

Train a **joint** flow $p_\phi(z, q)$ on baseline data. Freeze it. Mutual
information is

$$
I(z; q) = \mathbb{E}_{(z,q)}\Bigl[\log p_\phi(z, q) - \log p_\phi(z) - \log p_\phi(q)\Bigr].
$$

The marginals $p_\phi(z), p_\phi(q)$ are obtained by Monte-Carlo
marginalisation through the joint flow (sample $q' \sim p_\phi(q)$ then
$\log p_\phi(z) \approx \log \tfrac{1}{M}\sum_{m} p_\phi(z, q'_m)$).

Pros: principled MI estimate. Cons: expensive marginalisation per batch;
joint flow harder to pretrain. **Defer to a stretch goal.**

### 4.3 Approach C — Total-Correlation under a Joint Flow ("TC-loss")

Train one joint flow $T$ on baseline $(z_0, q)$. Freeze. For a new batch
$(f_\theta(X), q)$, the total correlation is

$$
\mathrm{TC} = D_{\mathrm{KL}}\bigl(p(T(z, q)) \,\big\|\, \prod_i \mathcal{N}(0, 1)\bigr),
$$

estimated as

$$
\widehat{\mathrm{TC}}
= -\tfrac{1}{n}\sum_i \log p_{\mathcal{N}(0,I)}\bigl(T(z_i, q_i)\bigr)
  - H_{\text{marginal-Gauss}}.
$$

The marginal-Gaussian entropy is a constant in $\theta$. This is essentially
"penalise non-Gaussianity of the joint after Gaussianisation". Cheap, but the
pretraining baseline distribution matters.

### 4.4 Approach D — HSIC with Gaussianised Kernels ("G-HSIC")

Plug Gaussianised features into HSIC: replace each `cka_rbf` kernel matrix in
`fairkl.metrics.cka` with $K_{ij} = \langle T(x_i), T(x_j) \rangle$ (linear in
Gaussianised space) or $K_{ij} = \exp(-\|T(x_i) - T(x_j)\|^2 / 2)$ (RBF in
Gaussianised space with fixed unit bandwidth — the flow absorbs the bandwidth).
This is the **smallest possible code delta** from current CKA: literally swap
the kernel call.

### 4.5 Recommended primary loss

**Use Approach A (G-XCOV) as the headline,** with Approach D (G-HSIC) as a
control to disentangle "flow as preprocessor" from "linear vs. RBF after
flow". Approach C is a useful sanity check. Approach B is a stretch goal.

---

## 5. Experiment Design

### 5.1 Two-stage pipeline

```
Stage 1 — pretrain & freeze
  data D_pre  ─►  fit T_z, T_q  by MLE (NLL on standard-Normal base)
                     │
                     └─►  save weights; mark non-trainable

Stage 2 — fair downstream training
  data D_task ─►  f_θ  ─►  z = f_θ(X)
                            │
                            ├─► task_loss(z, y)        ◄── user-supplied
                            └─► μ · L_fair(T_z(z), T_q(q))
                                        ▲
                                        └── frozen flows, autograd flows through
```

`D_pre` and `D_task` can be:

* **Same data**: warm-start the flow on the training set. Simple. Risk: flow
  over-specialises to training distribution; doesn't see test marginals.
* **Different splits**: pretrain on a held-out auxiliary set; safer.
* **Synthetic ablation**: pretrain on i.i.d. Gaussian (so $T \approx \text{id}$) —
  this collapses G-XCOV to vanilla cross-covariance and is the strongest
  baseline to beat.

### 5.2 Freezing the flow inside Keras

```python
from gaussianization.gauss_keras.training import make_gaussianization_flow

T_z = make_gaussianization_flow(input_dim=d_z, num_blocks=6)
T_z(keras.ops.zeros((1, d_z)))      # force build
# ... fit T_z on D_pre with base_nll_loss ...
for w in T_z.weights:               # critical: stop optimiser updates
    w._trainable = False
T_z.trainable = False
```

When the frozen `T_z` is called inside `FairGaussLoss.__call__`, Keras 3 with
the JAX backend traces it as part of the gradient w.r.t. `z` (and therefore
w.r.t. $\theta$), but emits no updates to the flow's variables. Under the
JAX backend, an extra safety belt is `keras.ops.stop_gradient` on the flow's
parameter tensors if any optimiser path could otherwise see them.

### 5.3 The new loss class

```python
# projects/gaussianization/src/gaussianization/fair/losses.py
import keras
from keras import ops

class GaussianizedXCovLoss(keras.losses.Loss):
    """Fairness loss: Frobenius norm of cross-cov in Gaussianised space.

    L = || E[T_z(z) T_q(q)^T] - E[T_z(z)] E[T_q(q)]^T ||_F^2
    """
    def __init__(self, flow_z, flow_q, normalize=True, name="g_xcov", **kw):
        super().__init__(name=name, **kw)
        self.flow_z = flow_z
        self.flow_q = flow_q
        self.normalize = normalize

    def call(self, q_true, z_pred):
        zg = self.flow_z(z_pred)                            # (n, d_z)
        qg = self.flow_q(ops.cast(q_true, z_pred.dtype))    # (n, d_q)
        zg = zg - ops.mean(zg, axis=0, keepdims=True)
        qg = qg - ops.mean(qg, axis=0, keepdims=True)
        n  = ops.cast(ops.shape(zg)[0], zg.dtype)
        C  = ops.matmul(ops.transpose(zg), qg) / (n - 1.0)  # (d_z, d_q)
        loss = ops.sum(C * C)
        if self.normalize:
            sz = ops.sum(zg * zg) / (n - 1.0)
            sq = ops.sum(qg * qg) / (n - 1.0)
            loss = loss / (sz * sq + 1e-12)
        return loss
```

Wire it in exactly like CKA:

```python
from fairkl.models import FairModelWrapper

mlp = keras.Sequential([keras.layers.Dense(64, "relu"),
                        keras.layers.Dense(1)])
fair = FairModelWrapper(
    mlp, mu=0.5,
    fairness_loss=GaussianizedXCovLoss(flow_z=T_z, flow_q=T_q),
)
fair.compile(optimizer="adam", loss="mse")
fair.fit(X_train, y_train, q=q_train, epochs=50, batch_size=256)
```

### 5.4 Pretraining the flow — concrete recipe

```python
KERAS_BACKEND=jax  # set before any keras import
T = make_gaussianization_flow(
        input_dim=d, num_blocks=8, num_components=12,
        pca_init_data=D_pre, mixture_init_data=D_pre,
    )
T.compile(optimizer=keras.optimizers.Adam(1e-3),
          loss=base_nll_loss)
T.fit(D_pre, D_pre, epochs=300, batch_size=512, validation_split=0.1,
      callbacks=[keras.callbacks.EarlyStopping(patience=15)])
T.save("artefacts/T_z.keras")
```

Validate the flow has actually Gaussianised the data before trusting any
fairness number on top:

* **Marginal QQ-plot** of `T(D_pre)` vs `N(0,1)`.
* **Negentropy** ≈ 0 on each marginal.
* **Cross-validated NLL** stable across epochs.
* **`gauss_flows`-side cross-check**: re-fit a FlowJax `gaussianization_flow`
  on the same `D_pre` and compare per-sample log-probs (Pearson > 0.99
  expected up to numerical noise).

### 5.5 Architecture / file layout

A self-contained sub-tree inside the existing `projects/gaussianization`
project (do **not** spawn a new top-level project — this is an *experiment
on* the existing library, not a new library):

```
projects/gaussianization/
├── src/gaussianization/
│   └── fair/                       # NEW subpackage
│       ├── __init__.py
│       ├── losses.py               # GaussianizedXCovLoss, GHSICLoss, TCLoss
│       ├── freeze.py               # freeze_flow(model) helper
│       └── pretrain.py             # fit_and_freeze(D, **kw) -> frozen flow
├── notebooks/
│   ├── 05_fair_gauss_synthetic.ipynb     # toy: y = tanh(x) + 3q + ε
│   ├── 06_fair_gauss_adult.ipynb         # UCI Adult Census
│   └── 07_fair_gauss_vs_cka.ipynb        # head-to-head ablation
├── tests/
│   ├── test_fair_losses.py         # shape/gradient/freeze tests
│   └── test_fair_independence.py   # known-independent inputs ⇒ loss ≈ 0
└── docs/
    └── fair_gaussianization_experiment.md   # this file
```

The notebooks follow the repo convention: executed `.ipynb` with embedded
outputs, optionally paired with a short prose `.md` (see `CLAUDE.md`).

### 5.6 Hydra config sketch

Adopt the existing `configs/` hierarchy:

```yaml
# configs/experiment/fair_gauss_adult.yaml
defaults:
  - /data: adult
  - /model: mlp_small
  - _self_

flow:
  num_blocks: 8
  num_components: 12
  epochs_pretrain: 300
  artefact_path: artefacts/T_q_adult.keras

fairness:
  loss: g_xcov              # one of: cka | hsic | mmd | g_xcov | g_hsic | tc
  mu: 0.5

train:
  batch_size: 256
  epochs: 100
  optimizer: adam
  lr: 1e-3
```

Sweeps come for free: `pixi run train experiment=fair_gauss_adult fairness.mu=0.0,0.1,0.5,1.0,5.0 fairness.loss=cka,g_xcov`.

---

## 6. Datasets

| Dataset | Sensitive $q$ | Task | Use |
|---|---|---|---|
| Synthetic: $y = \tanh(x_1) + 0.5 x_2 + 3 q + \varepsilon$, $q \sim \text{Bern}(0.5)$ | $q$ | regression | Sanity / unit test (mirrors `fairkl/docs/notebooks/fair_model_wrapper.py`). |
| **UCI Adult Census** | gender | binary classif. (income > 50K) | Headline benchmark; already wired in `fairkl/docs/notebooks/fair_adult_census.py`. ≈32k/6.5k. |
| **COMPAS** | race | recidivism classif. | Standard fairness benchmark; add later. |
| **German Credit** | age / gender | credit-default classif. | Small (1000 rows) → useful for over-fitting studies. |
| **CelebA (subset)** | gender | smile classif. | Stretch: representation-level fairness (z = penultimate-layer embedding). |

All non-synthetic datasets are downloaded via a new
`scripts/preprocess_fairness.py` and DVC-tracked (`.dvc` pointers committed,
raw CSVs excluded). Follow the existing pattern in `scripts/preprocess.py`.

---

## 7. Evaluation Plan

For every (dataset, fairness loss, $\mu$) combination report:

**Predictive metrics**
* Regression: RMSE, R².
* Classification: accuracy, ROC-AUC, log-loss.

**Fairness metrics** (compute on test set with the trained downstream model;
implement in a new `gaussianization.fair.metrics` module — these are
*evaluation* statistics, not training losses):

* **Demographic Parity Difference / Ratio** — `|P(ŷ=1|q=1) − P(ŷ=1|q=0)|`.
* **Equalized Odds Difference** — max gap in TPR and FPR across groups.
* **Statistical-parity HSIC** — kernel statistic on $(ŷ, q)$ with fixed RBF
  bandwidth (median heuristic) for a *neutral* judge.
* **Predictive-MI estimate** — separately fit a small MINE / KNN-based
  MI estimator on $(ŷ, q)$.

**Sanity / diagnostic**
* **G-XCOV value itself** on train + val + test, plotted vs. $\mu$.
* Marginal QQ-plots of $T_z(f_\theta(X))$ before / after fair training.

**Comparison axes**
1. Loss family: `cka | g_xcov | g_hsic | tc` × $\mu \in \{0, 0.1, 0.5, 1, 5\}$.
2. Flow capacity: `num_blocks ∈ {2, 4, 8, 16}`.
3. Flow training set: same-as-task vs. held-out vs. i.i.d. Normal (control).
4. Batch size: 32, 128, 512 (test the batch-sensitivity claim).

**Statistical rigour**: 5 seeds, report mean ± std; paired bootstrap for
"G-XCOV vs CKA" deltas at fixed $\mu$.

---

## 8. Risks & Open Questions

| Risk | Mitigation |
|---|---|
| **Flow over-fits $D_{\text{pre}}$ and gives wrong gradients off-support.** | Validate on held-out NLL; consider $D_{\text{pre}} \neq D_{\text{task}}$; add light noise injection at flow training time. |
| **`add_loss` + JAX backend funkiness with sub-models.** | Smoke test early: build a 2-layer MLP, wrap, run one `fit` step. The `FairModelWrapper` already uses this exact pattern with CKA. |
| **Freezing doesn't really freeze.** | Assert in `test_fair_losses.py` that `len(model.trainable_variables)` is unchanged after wrapping vs. before. Diff a frozen flow's weights before/after a training step. |
| **`MixtureCDFGaussianization` inverse uses bisection** (`mixture_cdf.py`) — non-smooth gradients at root. | We only call the *forward* direction in fairness loss, which is smooth. Avoid `invert` in the loss path. |
| **G-XCOV is only second-order in flow space.** | Approach B/C are higher-order; can swap in later. Report G-HSIC alongside for higher-order check. |
| **Per-sample flow eval slows training.** | Profile. Flow forward is ~`num_blocks × (d² + dK)` per sample — for `d=8, K=12, blocks=8` and `bs=256` this is sub-millisecond on a GPU. |
| **CKA baseline tuned with Keras Tuner but G-XCOV uses fixed flow — unfair comparison.** | Tune the flow's `num_blocks, num_components` with the same Keras Tuner budget as CKA's $\sigma$. |

**Open questions for follow-ups**

* Does pretraining $T_z$ on the *predictor's outputs* (a moving target) work
  better than on the raw features? (Bilevel iteration vs. fixed flow.)
* Is one *joint* flow $T(z, q)$ strictly better than two marginal flows
  $T_z, T_q$ for capturing dependence? Empirically test.
* Can the flow's `forward_with_intermediates` give us a *per-layer*
  fairness signal — i.e. anneal the depth of $T$ during training?

---

## 9. Stretch Goals

1. **Approach B (DR-MI)** with a joint flow + MC marginalisation.
2. **Fair representation learning** — use the encoder of an autoencoder
   as $f_\theta$, push G-XCOV between bottleneck and $q$.
3. **Fair clustering** — combine `FairPCA` / `FairKernelPCA` from `fairkl`
   with a G-XCOV penalty on cluster assignments.
4. **JAX/Equinox port** — re-implement `GaussianizedXCovLoss` against
   `gauss_flows` FlowJax flows; benchmark against the Keras version on a
   shared CI fixture.
5. **Theoretical note** — write up the equivalence
   "linear cross-cov in Gaussianised space ≡ HSIC with marginal-Gaussian
   kernel in data space" as a section of `survae_flows_proof.md` (or a
   sibling proof doc).

---

## 10. Milestones

| # | Milestone | Acceptance |
|---|---|---|
| M1 | Skeleton: `src/gaussianization/fair/{losses,freeze,pretrain}.py` + tests pass with synthetic data | `pytest tests/test_fair_losses.py` green; `make lint typecheck` clean |
| M2 | Synthetic notebook `05_fair_gauss_synthetic.ipynb` reproduces $\mu \uparrow \Rightarrow$ fairness $\uparrow$, accuracy slowly $\downarrow$ | Notebook executed, committed with outputs |
| M3 | UCI Adult notebook `06_fair_gauss_adult.ipynb`; beat or match CKA on demographic parity at matched accuracy | Comparison table + plots |
| M4 | Ablation notebook `07_fair_gauss_vs_cka.ipynb` (5-seed bars, $\mu$ sweep, loss-family sweep) | All four axes of §7 covered |
| M5 | Hydra config + DVC stage so `pixi run dvc repro` regenerates results from a clean clone | `dvc repro` succeeds; results committed under `results/fair_gauss/` |
| M6 | MyST docs entry under "Gaussianization flows" toc in `myst.yml`; cross-link to this engineering doc | Built docs render the new notebooks |
| M7 (stretch) | Approach C (TC-loss) implementation + comparison row in the ablation table | Added to notebook 07 |
| M8 (stretch) | JAX/`gauss_flows` cross-validation: same data, two flow implementations, log-probs agree to 1e-5 | Cross-check script in `scripts/` |

---

## 11. Conventions Checklist

Per `research_notebook/CLAUDE.md` and `keras-fairkl/CLAUDE.md`:

* `from __future__ import annotations` at the top of every new `.py`.
* Type hints on every public function.
* Google-style docstrings (match `fairkl` style — `FairModelWrapper` is the template).
* `pathlib.Path` for filesystem.
* Notebooks committed *executed*; figures inline via `plt.show()`, no
  separate PNGs.
* Conventional commits: `feat(fair_gauss): …`, `test(fair_gauss): …`,
  `docs(fair_gauss): …`.
* Pre-commit gates: `pytest`, `ruff check .`, `ruff format --check .`,
  `ty check src/gaussianization`.
* Plans/scratch in `.plans/` (gitignored); this engineering doc is *committed*
  because it's the spec, not a scratchpad.

---

## 12. References

* Cortes, Mohri, Rostamizadeh (2012). *Algorithms for Learning Kernels Based
  on Centered Alignment.* JMLR.
* Gretton, Bousquet, Smola, Schölkopf (2005). *Measuring Statistical
  Dependence with Hilbert–Schmidt Norms.* ALT.
* Pérez-Suay et al. (2017). *Fair Kernel Learning.* ECML PKDD.
* Chen, Gopinath (2000). *Gaussianization.* NeurIPS.
* Laparra, Camps-Valls, Malo (2011). *Iterative Gaussianization: From
  ICA to Random Rotations* (RBIG).
* Meng, Song, Song, Ermon (2020). *Gaussianization Flows.* AISTATS.
* Belghazi et al. (2018). *MINE: Mutual Information Neural Estimation.* ICML.

---

## Appendix A — Minimum working example

```python
# scripts/exp_fair_gauss_demo.py
from __future__ import annotations
import os
os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import keras
from keras import ops

from gaussianization.gauss_keras.training import (
    make_gaussianization_flow, base_nll_loss,
)
from gaussianization.fair.losses import GaussianizedXCovLoss
from fairkl.models import FairModelWrapper

rng = np.random.default_rng(0)
n, d = 4000, 4
q = rng.binomial(1, 0.5, size=n).astype(np.float32)
X = rng.normal(size=(n, d)).astype(np.float32)
y = np.tanh(X[:, 0]) + 0.5 * X[:, 1] + 3.0 * q + 0.1 * rng.normal(size=n)
y = y.astype(np.float32)

# -- Stage 1: pretrain & freeze the two flows -----------------------------
def fit_freeze(data_1d_or_kd):
    a = np.atleast_2d(data_1d_or_kd.T).T if data_1d_or_kd.ndim == 1 else data_1d_or_kd
    T = make_gaussianization_flow(input_dim=a.shape[1], num_blocks=4,
                                  num_components=8, mixture_init_data=a)
    T.compile(optimizer=keras.optimizers.Adam(1e-3), loss=base_nll_loss)
    T.fit(a, a, epochs=80, batch_size=256, verbose=0)
    T.trainable = False
    return T

T_z = fit_freeze(y.reshape(-1, 1))     # downstream output is 1-d for regression
T_q = fit_freeze(q.reshape(-1, 1))

# -- Stage 2: fair downstream training ------------------------------------
mlp = keras.Sequential([keras.layers.Dense(32, "relu"),
                        keras.layers.Dense(1)])
fair = FairModelWrapper(
    mlp, mu=0.5,
    fairness_loss=GaussianizedXCovLoss(flow_z=T_z, flow_q=T_q),
)
fair.compile(optimizer=keras.optimizers.Adam(3e-3), loss="mse")
fair.fit(X, y, q=q, epochs=40, batch_size=256, verbose=2)

# -- Diagnostics ----------------------------------------------------------
yhat = fair.predict(X).ravel()
print("RMSE:", np.sqrt(np.mean((yhat - y) ** 2)))
print("|corr(yhat, q)|:", abs(np.corrcoef(yhat, q)[0, 1]))
```

Running this end-to-end is the M1 acceptance criterion.
