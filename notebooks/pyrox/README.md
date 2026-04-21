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
matplotlib). The committed `.ipynb` files carry their cell outputs, so
MyST renders them without needing the kernel installed.

A dedicated pixi environment bundles everything needed to re-execute them:

```bash
pixi install -e pyrox                    # install pyrox + colab + optax extras
pixi run -e pyrox execute-pyrox          # nbconvert --execute --inplace on all 7
# or interactively:
pixi run -e pyrox jupyter lab
```

### JAX / numpyro version note

The feature pins `jax>=0.7,<0.9` because numpyro 0.20.1 still imports
`xla_pmap_p` from `jax.extend.core.primitives` — a symbol JAX removed at the
0.9 line. Once numpyro ≥0.21 lands with the updated import, the upper bound
can come off.
