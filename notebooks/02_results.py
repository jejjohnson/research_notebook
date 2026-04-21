# ---
# jupyter:
#   jupytext:
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
# # Results Analysis
#
# This notebook analyzes and visualizes experiment results.
#
# > **Note**: This file is a [Jupytext](https://jupytext.readthedocs.io) percent-format
# > script. To open it as a Jupyter notebook, run:
# > ```bash
# > pixi run notebooks-to-ipynb
# > ```
# > or convert a single file with:
# > ```bash
# > pixi run jupytext --to notebook notebooks/02_results.py
# > ```

# %% [markdown]
# ## Imports

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt

# %% [markdown]
# ## Load Metrics

# %%
metrics_path = Path("results/metrics/train_metrics.json")

if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)
    print("Loaded metrics:", metrics)
else:
    # Placeholder metrics when no experiment has been run yet
    metrics = {
        "train_loss": [1.0, 0.8, 0.6, 0.4, 0.3],
        "val_loss": [1.1, 0.9, 0.7, 0.5, 0.4],
    }
    print("No metrics file found — using placeholder data.")

# %% [markdown]
# ## Training Curves

# %%
train_loss = metrics.get("train_loss", [])
val_loss = metrics.get("val_loss", [])

fig, ax = plt.subplots(figsize=(8, 4))

if train_loss:
    ax.plot(train_loss, label="Train loss", marker="o")
if val_loss:
    ax.plot(val_loss, label="Val loss", marker="s", linestyle="--")

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training Curves")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Final Metrics Summary

# %%
if train_loss:
    print(f"Final train loss: {train_loss[-1]:.4f}")
if val_loss:
    print(f"Final val loss:   {val_loss[-1]:.4f}")

other_metrics = {
    k: v for k, v in metrics.items() if k not in {"train_loss", "val_loss"}
}
for key, value in other_metrics.items():
    print(f"{key}: {value}")
