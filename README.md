# Research Notebook

[![CI](https://github.com/jejjohnson/research_notebook/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/research_notebook/actions/workflows/ci.yml)
[![Docs](https://github.com/jejjohnson/research_notebook/actions/workflows/pages.yml/badge.svg)](https://jejjohnson.github.io/research_notebook)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Pixi](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![DVC](https://img.shields.io/badge/data%20versioned%20with-DVC-945DD6)](https://dvc.org)

> A batteries-included scientific research project template for reproducible experiments.

## Features

- 🐍 **Python 3.12+** with modern type annotations
- 📦 **[Pixi](https://pixi.sh)** for reproducible conda-forge environments across platforms
- 📊 **[DVC](https://dvc.org)** for data versioning and ML pipeline management
- ⚙️ **[Hydra](https://hydra.cc)** + **[hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/)** for type-safe configuration management
- 📓 **[JupyterLab](https://jupyterlab.readthedocs.io)** with LSP, Git integration, and MyST rendering
- 📝 **[Jupytext](https://jupytext.readthedocs.io)** — notebooks stored as diff-friendly `.py` files, convert to `.ipynb` on demand
- 🌊 **[Marimo](https://marimo.io)** reactive notebook environment
- 📚 **[MyST-MD](https://mystmd.org)** for publication-quality documentation
- 🔍 **[Ruff](https://docs.astral.sh/ruff/)** for fast linting and formatting
- 🔒 **Pre-commit hooks** for code quality enforcement
- 🚀 **14 GitHub Actions workflows** for CI/CD, reproducibility, and automation
- 🖥️ **Cross-platform** support (Linux x86-64, macOS ARM64)
- 📝 **[CITATION.cff](https://citation-file-format.github.io/)** for academic citations
- 🔬 **Hydra multirun** sweeps for hyperparameter optimization

## Project Structure

```
research_template/
├── .github/workflows/    # 14 GitHub Actions workflows
├── .devcontainer/        # VS Code / Codespaces dev container
├── configs/              # Hydra configuration hierarchy
│   ├── train.yaml        # Main config with defaults
│   ├── model/            # Model configs (baseline, transformer)
│   └── data/             # Data configs (small, full)
├── data/                 # Data directories (DVC-managed)
│   ├── raw/
│   ├── processed/
│   └── external/
├── docs/                 # MyST documentation
├── marimo_notebooks/     # Marimo reactive notebooks
├── notebooks/            # Jupyter notebooks
├── results/              # Experiment results (DVC-managed)
├── scripts/              # Entry point scripts
├── src/research_notebook/        # Source package
│   ├── data/             # Data loading utilities
│   ├── models/           # Model implementations
│   ├── trainers/         # Training loops
│   └── utils/            # Utility functions
└── tests/                # Test suite
```

## Quick Start

### Prerequisites

Install [Pixi](https://pixi.sh):

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### Installation

```bash
git clone https://github.com/jejjohnson/research_notebook
cd research_notebook
pixi install
```

### Run an Experiment

```bash
# Preprocess data
pixi run preprocess

# Train with default config
pixi run train

# Train with parameter overrides
pixi run train training.lr=0.001 model=transformer

# Evaluate
pixi run evaluate
```

## Environments

| Environment | Features | Use Case |
|-------------|----------|----------|
| `default` | dev | Testing, linting, training |
| `docs` | docs | Building MyST documentation |
| `jupyterlab` | dev + jupyterlab | Interactive notebooks |
| `marimo` | dev + marimo | Reactive notebooks |

```bash
# Activate specific environment
pixi run -e jupyterlab lab
pixi run -e marimo marimo-edit
pixi run -e docs docs-build
```

## Configuration Management

This template uses both [Hydra](https://hydra.cc) and [hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/) for configuration management.

### Classic Hydra (YAML-based)

```bash
# Use default config
pixi run train

# Override parameters
pixi run train training.lr=0.001 training.epochs=50

# Use different model config
pixi run train model=transformer

# Hyperparameter sweep
pixi run train -m training.lr=0.001,0.01,0.1
```

### hydra-zen (Python-based, type-safe)

```bash
pixi run train-zen
```

hydra-zen eliminates YAML boilerplate by defining configs as Python dataclasses:

```python
from hydra_zen import builds, make_config, launch

ModelConfig = builds(BaselineModel, hidden_size=64, num_layers=2)
ExperimentConfig = make_config(model=ModelConfig, seed=42)
```

## Experiment Tracking

This template uses [DVC](https://dvc.org) for data and experiment tracking.

```bash
# Add data to DVC
dvc add data/raw/dataset.csv

# Run the full pipeline
dvc repro

# Check pipeline status
dvc status

# View pipeline DAG
dvc dag

# Compare metrics across experiments
dvc metrics diff
```

The DVC pipeline is defined in `dvc.yaml`:
1. **preprocess**: Processes raw data → `data/processed/`
2. **train**: Trains model → `results/metrics/train_metrics.json`

## Documentation

Documentation is built with [MyST-MD](https://mystmd.org) using the book theme.

```bash
# Serve locally with live reload
pixi run -e docs docs-serve

# Build static HTML
pixi run -e docs docs-build
```

Docs are automatically deployed to GitHub Pages on every push to `main`.

## Notebook Environments

Notebooks are stored as [Jupytext](https://jupytext.readthedocs.io) percent-format
`.py` scripts — not `.ipynb` files. This keeps the repository clean, diff-friendly,
and free of committed cell outputs. Generated `.ipynb` files are gitignored.

### Converting to .ipynb

```bash
# Convert all notebooks/ scripts to .ipynb
pixi run notebooks-to-ipynb

# Convert a single file
pixi run jupytext --to notebook notebooks/01_eda.py

# Keep .py and .ipynb in sync while editing (paired mode)
pixi run -e jupyterlab jupytext --set-formats py:percent,ipynb notebooks/01_eda.py
```

### JupyterLab

Full-featured JupyterLab with LSP, Git integration, MyST rendering, and spell checking:

```bash
pixi run -e jupyterlab lab
```

### Marimo

Reactive, reproducible notebooks in pure Python:

```bash
pixi run -e marimo marimo-edit
```

Marimo notebooks in `marimo_notebooks/` are also stored as `.py` files,
making them diff-friendly and importable as regular Python modules.

## CI/CD Workflows

| Workflow | Trigger | Description |
|----------|---------|-------------|
| `ci.yml` | push/PR | pytest on ubuntu + macos |
| `lint.yml` | push/PR | Ruff linting |
| `typecheck.yml` | push/PR | ty type checking |
| `pages.yml` | push to main | Build + deploy MyST docs |
| `dvc-check.yml` | DVC file changes | Validate DVC pipeline |
| `notebooks.yml` | notebook changes | Validate Jupytext .py notebooks |
| `reproducibility.yml` | weekly schedule | Full `dvc repro` |
| `experiment-report.yml` | PR | DVC metrics diff comment |
| `citation.yml` | CITATION.cff changes | Validate citation file |
| `pixi-update.yml` | monthly schedule | Update pixi lockfile |
| `codeql.yml` | push/PR/schedule | Security scanning |
| `conventional-commits.yml` | PR | Validate PR title |
| `label-pr.yml` | PR | Auto-label PRs |
| `pre-commit-autoupdate.yml` | weekly schedule | Update pre-commit hooks |

## Academic

### Citation

If you use this template, please cite it using the metadata in `CITATION.cff`:

```bibtex
@software{johnson2026researchnotebook,
  author = {Johnson, Juan Emmanuel},
  title  = {Research Notebook},
  year   = {2024},
  url    = {https://github.com/jejjohnson/research_notebook},
}
```

### References

Add BibTeX references to `references.bib`. They are automatically available
in MyST documentation and Jupyter notebooks with the `jupyterlab-myst` extension.

### Zenodo

Connect your GitHub repository to [Zenodo](https://zenodo.org) for automatic
DOI assignment on releases.

## Acknowledgments

This template was inspired by:
- [jejjohnson/pypackage_template](https://github.com/jejjohnson/pypackage_template) — library-focused Python package template
- [DrivenData Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- [Hydra documentation](https://hydra.cc/docs/intro/)
- [DVC documentation](https://dvc.org/doc)
