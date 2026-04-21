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
# # Exploratory Data Analysis
#
# This notebook provides an initial exploration of the dataset.
#
# > **Note**: This file is a [Jupytext](https://jupytext.readthedocs.io) percent-format
# > script. To open it as a Jupyter notebook, run:
# > ```bash
# > pixi run notebooks-to-ipynb
# > ```
# > or convert a single file with:
# > ```bash
# > pixi run jupytext --to notebook notebooks/01_eda.py
# > ```

# %% [markdown]
# ## Imports

# %%
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ## Load Data

# %%
# Placeholder: replace with actual data loading
rng = np.random.default_rng(42)
data = rng.normal(loc=0, scale=1, size=500)
labels = (data > 0).astype(int)

print(f"Loaded {len(data)} samples")
print(f"Class balance: {labels.mean():.2%} positive")

# %% [markdown]
# ## Distribution Overview

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(data, bins=30, edgecolor="white")
axes[0].set_title("Feature Distribution")
axes[0].set_xlabel("Value")
axes[0].set_ylabel("Count")

axes[1].bar(["Negative", "Positive"], [(labels == 0).sum(), (labels == 1).sum()])
axes[1].set_title("Class Balance")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary Statistics

# %%
print(f"Mean:   {np.mean(data):.4f}")
print(f"Std:    {np.std(data):.4f}")
print(f"Min:    {np.min(data):.4f}")
print(f"Max:    {np.max(data):.4f}")
print(f"Median: {np.median(data):.4f}")
