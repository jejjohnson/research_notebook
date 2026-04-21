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

### Upstream caveat (as of Apr 2026)

`pixi install -e pyrox` succeeds but `execute-pyrox` currently fails at
import time with:

```
ImportError: cannot import name 'xla_pmap_p' from 'jax.extend.core.primitives'
```

Root cause is an upstream mismatch: the latest numpyro on PyPI (0.20.1) still
imports `xla_pmap_p`, which JAX removed in 0.9. gaussx (a transitive pyrox
dep) pins `jaxlib>=0.9.2`, so pip can't resolve jax down to the
numpyro-compatible range. Resolution is pending one of:

- numpyro ≥0.21 (drops the removed-primitive import)
- gaussx loosening its `jaxlib>=0.9.2` pin

Until then the committed outputs — as last re-executed in pyrox's own CI —
are the source of truth for what you see rendered.
