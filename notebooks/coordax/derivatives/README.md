---
title: Derivatives — finite differences and finite volumes on labeled grids
---

# Derivatives — finite differences, spherical harmonics, and finite volumes

Once state lives in a `Field` with named axes, numerical derivatives become
coordinate-aware: the operator uses the axis ticks to pick a stencil, apply
boundary conditions, and emit a `Field` on the correct (possibly shifted)
grid. The three notebooks in this section build up from 1-D finite differences
{cite}`durran2010`, through **spherical-harmonic derivatives** on a lat-lon
grid (where FD runs into metric singularities at the poles), to finite-volume
flux divergence {cite}`leveque2002fv`.

## Finite differences

For a uniform grid with spacing $\Delta x$ and periodic boundaries, the
standard 2nd-order centered difference is

$$
(\partial_x u)_i \;=\; \frac{u_{i+1} - u_{i-1}}{2\,\Delta x},\qquad i \in \{0, \ldots, N-1\},
$$

with indices taken modulo $N$. On a non-uniform grid with cell centres
$x_i$ and variable spacing $\Delta x_i = x_{i+1} - x_{i-1}$, the same stencil
generalizes to

$$
(\partial_x u)_i \;=\; \frac{u_{i+1} - u_{i-1}}{x_{i+1} - x_{i-1}} \;+\; \mathcal{O}(\bar{\Delta x}\,\Delta x'),
$$

i.e., it stays second-order only when spacing varies slowly. Coordax
implements this by reading the tick vector from the axis rather than
assuming $\Delta x$ is constant.

## Spherical harmonics on a lat-lon grid

On a lat-lon grid finite differences are not the right tool — not because
they can be made to work (they can, with care), but because the
alternative is dramatically better. The standard spectral approach
{cite}`durran2010,williamson1992stswe` expands a field in
*spherical harmonics*:

$$
f(\phi, \lambda) \;=\; \sum_{\ell=0}^{\ell_{\max}} \sum_{m=-\ell}^{\ell} a_\ell^m\, \bar P_\ell^m(\sin\phi)\, e^{im\lambda},
$$

where $\bar P_\ell^m$ are the fully-normalized associated Legendre
functions and $a_\ell^m$ are spectral coefficients. In this basis every
interesting operator becomes algebra on coefficients:

$$
\partial_\lambda \;\longleftrightarrow\; im, \qquad
\nabla^2 \;\longleftrightarrow\; -\frac{\ell(\ell+1)}{a^2}, \qquad
\cos\phi\,\partial_\phi \;\longleftrightarrow\; \text{three-term }\ell\text{-recurrence},
$$

with no $1/\cos\phi$ anywhere inside the spectral flow — the pole
singularity lives entirely in the physical-space metric, not in the
coefficients. The transform uses a **Gauss–Legendre latitude grid** so
Legendre quadrature is exact up to degree $2N_\phi - 1$.

For reference, the physical-space identities that motivate the SH form
are {cite}`durran2010`:

$$
\nabla\!\cdot\!\mathbf{v} \;=\; \frac{1}{a\cos\phi}\,\partial_\lambda u \;+\; \frac{1}{a\cos\phi}\,\partial_\phi(v\cos\phi),\qquad
\zeta \;=\; \frac{1}{a\cos\phi}\,\partial_\lambda v \;-\; \frac{1}{a\cos\phi}\,\partial_\phi(u\cos\phi).
$$

The notebook demonstrates forward + inverse SHT on a Gauss–Legendre grid,
verifies machine-precision round-trip on an exact spherical harmonic, and
compares spectral vs finite-difference accuracy on a bandlimited test
field — typically 10+ orders of magnitude in favor of the spectral method.

## Finite volumes

Finite-volume (FV) methods {cite}`leveque2002fv` work with cell averages
$\bar u_i$ and fluxes $F_{i+1/2}$ at cell interfaces. The semidiscrete
conservation law

$$
\frac{\mathrm{d}\bar u_i}{\mathrm{d}t} \;=\; -\,\frac{F_{i+1/2} - F_{i-1/2}}{\Delta x_i}
$$

is conservative by construction — the interior flux at $i+1/2$ is exactly
cancelled by the adjacent cell's $i-1/2$ flux, so the discrete sum
$\sum_i \bar u_i \Delta x_i$ is preserved up to boundary contributions. This
is the property you want for tracer transport, mass balance, and any
quantity that must globally conserve.

## Numerical considerations

- **Stencil shifts.** Centered differences live on the same grid; FV fluxes
  live on the staggered grid. Coordax tracks this: a 1st-order derivative
  returns a `Field` on a shifted axis, and you `reindex_like` back to the
  cell centres before combining with the original field.
- **Boundary conditions.** Periodic BCs are "free" — just roll the array.
  Dirichlet / Neumann require padding; the examples use `jnp.pad` and
  document the resulting stencil accuracy drop at the boundary.
- **Pole singularity.** The $1/\cos\phi$ factor blows up at $\phi = \pm\pi/2$.
  Real GCMs use pole filters, polar caps, or non-singular grids (cubed
  sphere, Yin-Yang). The notebooks clamp $\phi$ away from the pole and
  document the limit.
- **CFL for advection.** When these derivatives feed into a time stepper, the
  advective stability bound is $\Delta t \le \Delta x / |u|$; diffusive
  stability needs $\Delta t \le \Delta x^2 / (2\kappa)$. Pick the smaller.
- **`cmap` pattern.** For operators that naturally act on one axis at a time
  (most of the ones here), coordax provides a `cmap`-style decorator that
  vmaps the rest — equivalent to `jax.vmap(..., in_axes=...)` but dispatched
  by axis name.

## Notebooks

- [`05_finite_difference`](05_finite_difference.ipynb) — periodic and
  non-uniform 1-D derivatives; the coordinate-aware `cmap` dispatch pattern.
- [`06_finite_difference_spherical`](06_finite_difference_spherical.ipynb) —
  lat-lon derivatives with metric factors; vorticity and divergence on the
  sphere; pole handling.
- [`07_finite_volume`](07_finite_volume.ipynb) — cell-centred FV operators
  and flux divergence; a conservation check that verifies exact tracer mass
  preservation to machine precision.

## References

```{bibliography}
:filter: docname in docnames
```
