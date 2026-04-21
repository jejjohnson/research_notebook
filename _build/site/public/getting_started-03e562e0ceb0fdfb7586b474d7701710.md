# Getting Started

## Prerequisites

Install [Pixi](https://pixi.sh) for environment management:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

## Installation

Clone the repository and install the environment:

```bash
git clone https://github.com/jejjohnson/research_notebook
cd research_notebook
pixi install
```

## Running Experiments

### Data Preprocessing

```bash
pixi run preprocess
```

### Training

```bash
# Using Hydra (classic)
pixi run train

# Using hydra-zen (type-safe)
pixi run train-zen

# Override hyperparameters
pixi run train training.lr=0.01 model=transformer
```

### Evaluation

```bash
pixi run evaluate
```

## Notebook Environments

### JupyterLab

```bash
pixi run -e jupyterlab lab
```

### Marimo

```bash
pixi run -e marimo marimo-edit
```
