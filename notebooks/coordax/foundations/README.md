---
title: Foundations — Field, axes, and coordinate-aware operations
---

# Foundations — `Field`, axes, and coordinate-aware operations

The basic unit in coordax is a `Field` — a thin, JAX-native wrapper that pairs
a `jax.Array` with a tuple of *named* axes. Each axis is either a
`LabeledAxis` (numeric ticks, e.g. latitude values) or a `SizedAxis` (a name
with a size but no ticks, e.g. RGB channels). This design is explicitly modeled
after `xarray.DataArray` {cite}`hoyer2017xarray`, but with one hard
constraint dropped and one gained: labels must be *numeric* (so everything
stays inside JAX's typed array world), and the whole object is a JAX pytree —
so `jit`, `vmap`, `grad`, and `scan` work without custom registrations.

## Data model

A `Field` is the pair

$$
\mathbf{F} = (\mathbf{X},\ \{a_1, a_2, \ldots, a_n\}),\qquad \mathbf{X} \in \mathbb{R}^{N_1 \times N_2 \times \cdots \times N_n},
$$

where each axis $a_i$ carries a *name* and, optionally, a *coordinate vector*
$\mathbf{c}_i \in \mathbb{R}^{N_i}$. Positional axes (no name) are also
allowed and behave like raw NumPy dimensions — useful for batch / channel
dims that shouldn't participate in coordinate-aware dispatch.

### Why named axes at all?

Two reasons, one practical and one mathematical.

- **Practical**: it turns "axis 0" bugs into compile-time errors. A velocity
  field on a $(\text{time}, \text{lat}, \text{lon})$ grid can be reduced
  along `'lat'` by name; no more counting axes after a `vmap`.
- **Mathematical**: many numerical operators (derivatives, reductions,
  broadcasts) are naturally defined on a *coordinate*, not an index. A
  centered difference $\partial_x u$ needs the ticks $\mathbf{c}_x$; a mean
  over latitude needs the metric $\cos\phi$. Carrying the axis around means
  the operator can dispatch on it.

## Broadcasting rules

Binary ops $\mathbf{F}_1 \odot \mathbf{F}_2$ follow a simple rule: axes are
matched by *name*, not by position. Unnamed (positional) axes follow the
usual NumPy rules. This gives `xarray`-style broadcasting
{cite}`hoyer2017xarray` while keeping the fast-path tensor semantics that
JAX relies on.

## Numerical considerations

- **Tick dtype.** `LabeledAxis` ticks must be floating — monotone integer
  indices should use `SizedAxis` instead. This trips people coming from
  `xarray`, which happily labels with strings.
- **Alignment cost.** Binary ops between fields with different tick vectors
  do *not* interpolate; they raise. Reindex (`sel` / `isel`) or
  `reindex_like` before combining, so mismatched grids fail loudly instead
  of silently broadcasting zeros.
- **Positional vs named mixing.** A `Field` with a positional axis behaves
  like NumPy along that axis: coordinate-aware ops (reductions by name,
  `sel`) simply skip it. This is the intended "escape hatch" for batch
  dimensions.
- **JIT.** Axis names are static metadata (compile-time); axis sizes may be
  traced (runtime). Don't put Python-level axis manipulation inside a
  `jit`-compiled function if the axis *names* depend on data.

## Notebooks

- [`01_create_datasets`](01_create_datasets.ipynb) — building `Field` objects
  for RGB images, time-series, and spatio-temporal lat-lon data. Covers
  `cx.field()`, `LabeledAxis`, `SizedAxis`, and the `wrap` / `untag`
  round-trip.
- [`02_ops_unary_binary`](02_ops_unary_binary.ipynb) — how `+`, `*`,
  `jnp.where`, and the rest of the NumPy API lift onto `Field`; the
  name-matching broadcasting rule in anger.
- [`03_ops_coordinates`](03_ops_coordinates.ipynb) — positional slicing
  (`isel`), label slicing (`sel`), reindexing, and
  `CartesianProduct` for stacking independent axes into one.
- [`04_reductions`](04_reductions.ipynb) — `sum`, `mean`, `max`, `std`
  dispatched by axis *name*; the pattern that enables zonal means and
  time averages without index arithmetic.

## References

```{bibliography}
:filter: docname in docnames
```
