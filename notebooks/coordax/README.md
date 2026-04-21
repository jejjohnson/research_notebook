# coordax notebooks

Showcase notebooks ported from [jej_vc_snippets/jax/coordax](https://github.com/jejjohnson/jej_vc_snippets/tree/main/jax/coordax) —
pedagogical tutorials for [coordax](https://github.com/neuralgcm/coordax), a
coordinate-aware array library for JAX that sits between raw `jax.numpy`
arrays and the full `xarray` {cite}`hoyer2017xarray` stack.

Each notebook is executed end-to-end with outputs embedded, so everything
(prints, tables, numeric checks) renders inline in the MyST docs site without
re-execution.

Each sub-section is a curated landing page that leads with the math,
numerics, and references before pointing at the notebooks themselves.

## [Foundations](./foundations/README.md)

| Notebook | Topic |
|---|---|
| [`foundations/01_create_datasets.ipynb`](./foundations/01_create_datasets.ipynb) | Wrapping arrays as `Field` objects with `LabeledAxis` / `SizedAxis` |
| [`foundations/02_ops_unary_binary.ipynb`](./foundations/02_ops_unary_binary.ipynb) | Arithmetic, comparison, and unary ops on `Field`; broadcasting rules |
| [`foundations/03_ops_coordinates.ipynb`](./foundations/03_ops_coordinates.ipynb) | `isel`, `sel`, reindexing, `CartesianProduct`, coordinate composition |
| [`foundations/04_reductions.ipynb`](./foundations/04_reductions.ipynb) | Coordinate-aware reductions: sum, mean, max over named dims |

## [Derivatives](./derivatives/README.md)

| Notebook | Topic |
|---|---|
| [`derivatives/05_finite_difference.ipynb`](./derivatives/05_finite_difference.ipynb) | Periodic + non-uniform finite-difference derivatives; `cmap` pattern |
| [`derivatives/06_finite_difference_spherical.ipynb`](./derivatives/06_finite_difference_spherical.ipynb) | Lat-lon grid derivatives, variable $\mathrm{d}x$, vorticity, divergence |
| [`derivatives/07_finite_volume.ipynb`](./derivatives/07_finite_volume.ipynb) | Cell-centred FV operators; flux divergence; conservative schemes |

## [Dynamics](./dynamics/README.md)

| Notebook | Topic |
|---|---|
| [`dynamics/08_ode_integration.ipynb`](./dynamics/08_ode_integration.ipynb) | Integrating ODEs (advection-diffusion) with `diffrax`; state as `Field` |
| [`dynamics/09_ode_parameter_state_estimation.ipynb`](./dynamics/09_ode_parameter_state_estimation.ipynb) | Joint parameter/state estimation via `optax` + `jax.value_and_grad` |
| [`dynamics/10_pde_parameter_estimation.ipynb`](./dynamics/10_pde_parameter_estimation.ipynb) | Learning PDE parameters from data; coordinate-aware residuals |

## Running locally

These notebooks depend on `coordax` and its ML stack (JAX, diffrax, optax,
equinox). The committed `.ipynb` files carry their cell outputs, so MyST
renders them without needing the kernel installed.

A dedicated pixi environment bundles everything needed to re-execute them:

```bash
pixi install -e coordax                  # install coordax + JAX stack
pixi run -e coordax execute-coordax      # nbconvert --execute --inplace on all 10
# or interactively:
pixi run -e coordax jupyter lab
```

### chex dependency note

The `coordax` feature pins `chex` explicitly because coordax unconditionally
imports `chex` at package init (`coordax/testing.py`) without declaring it as
a runtime dependency. Remove the pin once coordax either declares the dep or
guards the import.
