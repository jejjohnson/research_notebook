# pyrox notebooks

Showcase notebooks ported from [jejjohnson/pyrox](https://github.com/jejjohnson/pyrox) —
an equinox + numpyro + JAX probabilistic-programming library.

Each notebook is executed end-to-end with outputs embedded, so everything
(figures, prints, tables) renders inline in the MyST docs site without
re-execution.

## GP regression & classification

| Notebook | Topic |
|---|---|
| [`exact_gp_regression.ipynb`](./exact_gp_regression.ipynb) | Exact GP regression — the three patterns |
| [`latent_gp_classification.ipynb`](./latent_gp_classification.ipynb) | Latent GP classification — the three patterns |

## Regression masterclass — three patterns for parameter handling

The masterclass trio walks through three ways to wire the same regression
problem, each leaning on a different level of abstraction:

| Notebook | Pattern |
|---|---|
| [`regression_masterclass_treeat.ipynb`](./regression_masterclass_treeat.ipynb) | Pattern 1: `eqx.tree_at` + raw NumPyro |
| [`regression_masterclass_pyrox_sample.ipynb`](./regression_masterclass_pyrox_sample.ipynb) | Pattern 2: `PyroxModule` + `pyrox_sample` |
| [`regression_masterclass_parameterized.ipynb`](./regression_masterclass_parameterized.ipynb) | Pattern 3: `Parameterized` + `PyroxParam` + native `pyrox.gp` |

## Ensemble methods

| Notebook | Topic |
|---|---|
| [`ensemble_primitives_tutorial.ipynb`](./ensemble_primitives_tutorial.ipynb) | Three ways to drive `ensemble_step` |
| [`ensemble_runner_tutorial.ipynb`](./ensemble_runner_tutorial.ipynb) | `EnsembleMAP` and `EnsembleVI` runners |

## Running locally

These notebooks depend on pyrox and its ML stack (JAX, NumPyro, equinox,
matplotlib). The copies here carry their outputs, so MyST renders them
without needing the kernel installed. To re-execute locally, install pyrox
and the showcase deps separately:

```bash
pip install "pyrox @ git+https://github.com/jejjohnson/pyrox"
# or, in a fresh env:
pixi run -e jupyterlab lab
# then install pyrox in the kernel
```
