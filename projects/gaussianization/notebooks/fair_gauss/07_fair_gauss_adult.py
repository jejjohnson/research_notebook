# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 07 — Fair classification on UCI Adult Census (G-XCOV vs CKA)
#
# Real-data analogue of notebook 06. We classify whether annual income
# exceeds 50K USD, with **gender** as the protected attribute $q$.
#
# Setup:
#
# - Features: a small hand-picked numeric subset of UCI Adult.
# - Target $y$: binary `income > 50K`.
# - Sensitive $q$: binary gender.
# - Predictor: small MLP returning a sigmoid probability.
# - Fairness losses compared on the **same** wrapper:
#   `G-XCOV` (this work) vs. `CKA` (fairkl baseline).
# - We sweep `μ ∈ {0, 0.1, 0.5, 2, 10, 50}` × 3 seeds, report a
#   Pareto curve of `(ROC-AUC, demographic-parity difference)`.

# %%
import os

os.environ.setdefault("KERAS_BACKEND", "jax")

# %%
import keras  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import fetch_openml  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from fairkl.metrics.cka import CKALoss  # noqa: E402
from fairkl.models import FairModelWrapper  # noqa: E402
from gaussianization.fair import (  # noqa: E402
    GaussianizedXCovLoss,
    demographic_parity_difference,
    equalized_odds_difference,
    fit_and_freeze,
)

keras.utils.set_random_seed(0)
print("keras backend:", keras.config.backend())

# %% [markdown]
# ## 1. Load and preprocess UCI Adult
#
# Features: age, education-num, hours-per-week, capital-gain, capital-loss,
# **and gender itself**. Including gender as a feature is the harder
# setting for fair learning — the model can exploit it directly, so the
# fairness penalty must actively suppress it.

# %%
ds = fetch_openml("adult", version=2, as_frame=True, parser="liac-arff")
df = ds.frame.dropna(subset=["sex", "class"]).reset_index(drop=True)

numeric_cols = ["age", "education-num", "hours-per-week", "capital-gain", "capital-loss"]
q_all = (df["sex"].astype(str).str.strip() == "Male").to_numpy(dtype="float32")
X_num = df[numeric_cols].to_numpy(dtype="float32")
X_all = np.concatenate([X_num, q_all.reshape(-1, 1)], axis=1).astype("float32")
y_all = (df["class"].astype(str).str.strip() == ">50K").to_numpy(dtype="float32")

print(f"n = {len(df)}, positive rate = {y_all.mean():.3f}, "
      f"male rate = {q_all.mean():.3f}")
print("base-rate gap (P(y=1|q=1) - P(y=1|q=0)):",
      f"{y_all[q_all == 1].mean() - y_all[q_all == 0].mean():.3f}")

X_train, X_test, y_train, y_test, q_train, q_test = train_test_split(
    X_all, y_all, q_all, test_size=0.25, random_state=0, stratify=y_all
)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train).astype("float32")
X_test = scaler.transform(X_test).astype("float32")
print("train:", X_train.shape, "test:", X_test.shape)

# %% [markdown]
# ## 2. Pretrain & freeze flows
#
# `T_z` is fit on the **probabilities produced by an unconstrained baseline
# predictor**, not on the raw binary labels. This is important: the
# downstream model emits sigmoid probabilities in $(0, 1)$, and we want
# `T_z` to see the same support distribution at inference time. (Fitting
# on binary `{0, 1}` would leave the central probability region
# off-support and wash out the fairness gradient.)
#
# `T_q` is fit on the marginal distribution of $q$. Both flows are small.

# %%
def fresh_mlp(input_dim: int) -> keras.Model:
    return keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )


keras.utils.set_random_seed(0)
baseline_mlp = fresh_mlp(X_train.shape[1])
baseline_mlp.compile(
    optimizer=keras.optimizers.Adam(3e-3), loss="binary_crossentropy"
)
baseline_mlp.fit(
    X_train, y_train, epochs=25, batch_size=512, verbose=0
)
baseline_proba = (
    np.asarray(baseline_mlp.predict(X_train, verbose=0))
    .ravel()
    .astype("float32")
)
print(
    f"baseline proba — min={baseline_proba.min():.3f}, "
    f"max={baseline_proba.max():.3f}, mean={baseline_proba.mean():.3f}"
)

flow_z, _ = fit_and_freeze(
    baseline_proba.reshape(-1, 1),
    num_blocks=4,
    num_components=8,
    epochs=80,
    batch_size=512,
    lr=2e-3,
    seed=0,
    verbose=0,
)
flow_q, _ = fit_and_freeze(
    q_train.reshape(-1, 1).astype("float32"),
    num_blocks=2,
    num_components=4,
    epochs=40,
    batch_size=512,
    lr=2e-3,
    seed=0,
    verbose=0,
)

# %% [markdown]
# ## 3. Train + evaluate one configuration

# %%
def train_eval(
    fairness_loss,
    mu: float,
    seed: int = 0,
    epochs: int = 25,
    batch_size: int = 512,
) -> dict:
    keras.utils.set_random_seed(seed)
    mlp = fresh_mlp(X_train.shape[1])
    model = FairModelWrapper(mlp, mu=mu, fairness_loss=fairness_loss)
    model.compile(
        optimizer=keras.optimizers.Adam(3e-3), loss="binary_crossentropy"
    )
    model.fit(
        X_train,
        y_train,
        q=q_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )
    proba = np.asarray(model.predict(X_test, verbose=0)).ravel()
    return {
        "mu": mu,
        "auc": float(roc_auc_score(y_test, proba)),
        "acc": float(np.mean((proba >= 0.5).astype(int) == y_test.astype(int))),
        "dp_diff": demographic_parity_difference(proba, q_test, threshold=0.5),
        "eo_diff": equalized_odds_difference(
            y_test, proba, q_test, threshold=0.5
        ),
    }


# %% [markdown]
# ## 4. Sweep `μ` for both fairness losses

# %%
mus = [0.0, 0.5, 2.0, 10.0, 50.0, 200.0]
seeds = [0, 1, 2]

g_loss = GaussianizedXCovLoss(flow_z=flow_z, flow_q=flow_q)
cka_loss = CKALoss(sigma_f=1.0, sigma_q=1.0, kernel="rbf", debiased=False)

records = []
for fam, loss in [("g_xcov", g_loss), ("cka", cka_loss)]:
    for mu in mus:
        for s in seeds:
            r = train_eval(loss, mu=mu, seed=s)
            r["family"] = fam
            r["seed"] = s
            records.append(r)
            print(
                f"  {fam:>7s}  mu={mu:6.2f}  seed={s}  "
                f"auc={r['auc']:.3f}  acc={r['acc']:.3f}  "
                f"dp={r['dp_diff']:.3f}  eo={r['eo_diff']:.3f}"
            )

# %% [markdown]
# ## 5. Aggregate across seeds

# %%
def agg(records, fam):
    out = []
    for mu in mus:
        rows = [r for r in records if r["family"] == fam and r["mu"] == mu]
        auc = np.array([r["auc"] for r in rows])
        acc = np.array([r["acc"] for r in rows])
        dp = np.array([r["dp_diff"] for r in rows])
        eo = np.array([r["eo_diff"] for r in rows])
        out.append(
            {
                "mu": mu,
                "auc_m": auc.mean(), "auc_s": auc.std(),
                "acc_m": acc.mean(), "acc_s": acc.std(),
                "dp_m": dp.mean(), "dp_s": dp.std(),
                "eo_m": eo.mean(), "eo_s": eo.std(),
            }
        )
    return out


agg_g = agg(records, "g_xcov")
agg_c = agg(records, "cka")

print(f"{'family':>7s} {'mu':>6s} {'AUC':>14s} {'ACC':>14s} {'DP-diff':>14s} {'EO-diff':>14s}")
for fam, rows in [("g_xcov", agg_g), ("cka", agg_c)]:
    for r in rows:
        print(
            f"{fam:>7s} {r['mu']:>6.2f} "
            f"{r['auc_m']:6.3f}±{r['auc_s']:5.3f} "
            f"{r['acc_m']:6.3f}±{r['acc_s']:5.3f} "
            f"{r['dp_m']:6.3f}±{r['dp_s']:5.3f} "
            f"{r['eo_m']:6.3f}±{r['eo_s']:5.3f}"
        )

# %% [markdown]
# ## 6. Pareto curves
#
# Left: ROC-AUC vs. demographic-parity difference.
# Right: ROC-AUC vs. equalized-odds difference.
# Top-left of each plot is the ideal corner (high AUC, low gap).

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for fam, rows, marker, color in [
    ("g_xcov", agg_g, "o", "C0"),
    ("cka", agg_c, "s", "C3"),
]:
    auc_m = np.array([r["auc_m"] for r in rows])
    auc_s = np.array([r["auc_s"] for r in rows])
    dp_m = np.array([r["dp_m"] for r in rows])
    dp_s = np.array([r["dp_s"] for r in rows])
    eo_m = np.array([r["eo_m"] for r in rows])
    eo_s = np.array([r["eo_s"] for r in rows])

    axes[0].errorbar(
        dp_m, auc_m, xerr=dp_s, yerr=auc_s,
        marker=marker, color=color, label=fam, capsize=3, lw=1.2,
    )
    axes[1].errorbar(
        eo_m, auc_m, xerr=eo_s, yerr=auc_s,
        marker=marker, color=color, label=fam, capsize=3, lw=1.2,
    )

    if fam == "g_xcov":
        for mu, x, y in zip(mus, dp_m, auc_m, strict=True):
            axes[0].annotate(f"μ={mu:g}", (x, y), fontsize=8,
                             xytext=(4, 4), textcoords="offset points")
        for mu, x, y in zip(mus, eo_m, auc_m, strict=True):
            axes[1].annotate(f"μ={mu:g}", (x, y), fontsize=8,
                             xytext=(4, 4), textcoords="offset points")

axes[0].set_xlabel("Demographic-parity difference  (lower = fairer)")
axes[0].set_ylabel("ROC-AUC  (higher = more accurate)")
axes[0].set_title("Pareto: AUC vs. DP")
axes[0].legend()

axes[1].set_xlabel("Equalized-odds difference")
axes[1].set_ylabel("ROC-AUC")
axes[1].set_title("Pareto: AUC vs. EO")
axes[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Takeaways
#
# - The unconstrained model (`μ = 0`) shows the inherited base-rate gap
#   between groups — substantial DP-diff with the highest AUC.
# - Both G-XCOV and CKA shrink DP-diff and EO-diff as `μ` grows; AUC
#   degrades gracefully.
# - With the kernel-side bandwidths **un-tuned** (CKA at `σ = 1` and a
#   plain RBF), the flow-based loss is competitive at matched fairness
#   — the flow has absorbed the bandwidth choice into its mixture-CDF
#   parameters.
# - The Pareto fronts are seed-stable (small error bars).
#
# This is a small experiment on a deliberately tiny feature subset; the
# main claim isn't "G-XCOV beats CKA" but "G-XCOV is a drop-in,
# bandwidth-free fairness penalty that competes with CKA on the standard
# Pareto axes". A larger benchmark (more features, more seeds, joint
# `μ` × bandwidth tuning for CKA) is the natural follow-up.
