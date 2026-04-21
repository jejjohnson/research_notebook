# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "marimo",
#   "numpy",
#   "matplotlib",
#   "altair",
#   "polars",
# ]
# ///
"""Marimo reactive exploration notebook.

Marimo notebooks are stored as .py files, making them:
- Git diff-friendly (no JSON blobs)
- Importable as regular Python modules
- Runnable as scripts: `python 01_exploration.py`
- Editable in the browser: `marimo edit 01_exploration.py`

Reactive execution: when a cell changes, all dependent cells
automatically re-execute, like a spreadsheet.
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell
def _(mo):
    # Reactive UI: changing this slider automatically updates downstream cells
    n_samples = mo.ui.slider(10, 1000, value=100, label="Number of samples")
    n_samples
    return (n_samples,)


@app.cell
def _(mo, n_samples, np, plt):
    # This cell re-runs automatically when n_samples changes
    rng = np.random.default_rng(42)
    data = rng.normal(size=n_samples.value)

    fig, ax = plt.subplots()
    ax.hist(data, bins=30, edgecolor="white")
    ax.set_title(f"Normal distribution (n={n_samples.value})")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")

    mo.mpl.interactive(fig)
    return ax, data, fig, rng


@app.cell
def _(data, mo, np):
    mo.md(f"""
    ## Summary Statistics

    | Statistic | Value |
    |-----------|-------|
    | Mean      | {np.mean(data):.4f} |
    | Std       | {np.std(data):.4f} |
    | Min       | {np.min(data):.4f} |
    | Max       | {np.max(data):.4f} |
    """)
    return


if __name__ == "__main__":
    app.run()
