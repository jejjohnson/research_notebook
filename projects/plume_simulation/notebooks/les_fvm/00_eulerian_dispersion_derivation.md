---
title: "Eulerian 3-D dispersion (L2) — derivation and numerical scheme"
---

# Eulerian 3-D dispersion: derivation and finite-volume discretisation

This note derives the equations solved by `plume_simulation.les_fvm` and explains how the terms map onto the `finitevolX` Arakawa C-grid operators. The L2 model sits between the analytic Gaussian puff (L1, `plume_simulation.gauss_puff`) and a full resolved-flow LES (L3, future port): it evolves a passive methane tracer on a 3-D Cartesian grid under a **prescribed** wind field with a **K-theory** (gradient-diffusion) closure. No turbulence is resolved; the model pays for full 3-D transport with spatially-varying wind and physically realistic boundaries (ground, inlet, outlet) that the Gaussian puff cannot represent.

## 1. Governing equation

We solve the conservative advection-diffusion equation for a passive scalar $C(\vec x, t)$ [kg/m³]:

$$
\frac{\partial C}{\partial t} + \nabla \cdot (\vec u\, C) \;=\; \nabla \cdot (\mathbf K\, \nabla C) + S(\vec x, t),
$$ (eq-ad)

with prescribed 3-D wind $\vec u(\vec x, t) = (u, v, w)$, diagonal eddy diffusivity $\mathbf K = \operatorname{diag}(K_h, K_h, K_z)$ (horizontally isotropic), and source $S$. Equation {eq}`eq-ad` is Stockie equation (2.2) [^stockie2011] without the A3 / A4 / A5 simplifications of the Gaussian plume — we keep every transport term and impose physically meaningful boundaries instead.

## 2. Closure assumptions

- **(E1) Prescribed flow.** $\vec u(\vec x, t)$ is an external input (no momentum equation solved). The wind need not be divergence-free — incompressibility is *not* enforced, since the model treats $\vec u$ as a kinematic input; breaking $\nabla \cdot \vec u = 0$ only affects the tracer continuity slightly and is typically $< 10\%$ for realistic atmospheric inputs.
- **(E2) K-theory.** Turbulent fluxes are modelled by the gradient closure $\overline{u'C'} = -\mathbf K \nabla C$, with $\mathbf K$ specified directly (as a scalar, anisotropic pair $(K_h, K_z)$, or field) or derived from Pasquill-Gifford σ curves via the Taylor identity $K = \sigma^2 / (2t)$ — see `les_fvm.diffusion.pg_eddy_diffusivity`.
- **(E3) Boundary conditions.** Unlike the Gaussian puff (unbounded, ground reflection by method of images), L2 lives in a bounded box with user-chosen BCs per face. The defaults are ``Dirichlet-clean`` on the inlet, ``outflow`` on the downstream face, ``periodic`` laterally, and ``Neumann`` (no flux) on the ground and top.
- **(E4) Passive scalar.** No buoyancy, decay, or chemistry; $S$ is a Gaussian-regularised point source with user-specified constant or time-varying emission rate.

## 3. Spatial discretisation

### 3.1 Arakawa C-grid

Fields live on a Cartesian 3-D Arakawa C-grid (`finitevolx.CartesianGrid3D`). The horizontal stagger is classical:

- $C, w, K$ at **T-points** (cell centres);
- $u$ at **U-points** (east faces);
- $v$ at **V-points** (north faces).

The vertical is collocated at T-points — finitevolX's 3-D operators treat $z$ as a batch axis, so there is no vertical staggering inside a field array. Arrays have shape ``[Nz, Ny, Nx]`` with one ghost cell per axis for boundary conditions; the interior spans `[1:-1, 1:-1, 1:-1]`.

### 3.2 Advection term

The divergence of the advective flux splits cleanly into horizontal and vertical parts:

$$
\nabla \cdot (\vec u C) \;=\; \partial_x(u C) + \partial_y(v C) + \partial_z(w C).
$$

The horizontal part — **two terms per z-level**, always acting in the (y, x) plane — is delegated to `finitevolx.Advection3D`, which applies a WENO5 (default) or user-chosen reconstruction at U- and V-faces for every z-level. The vertical part is built by `les_fvm._vertical_ops.vertical_advection_tendency` as a first-order upwind flux at k-faces:

$$
\widehat{F}_{k+1/2} \;=\; \begin{cases} w_{k+1/2}\, C_k, & w_{k+1/2} > 0, \\ w_{k+1/2}\, C_{k+1}, & w_{k+1/2} \leq 0, \end{cases}
$$

with $w_{k+1/2} = \tfrac12(w_k + w_{k+1})$. The tendency at interior T-point $k$ is then $-(\widehat{F}_{k+1/2} - \widehat{F}_{k-1/2}) / \Delta z$. First-order upwind is monotone, which matters for a non-negative tracer; vertical resolution in a plume simulation is typically too coarse for a higher-order scheme to pay off anyway.

### 3.3 Diffusion term

The K-theory flux $-\mathbf K \nabla C$ similarly splits:

$$
\nabla \cdot (\mathbf K \nabla C) \;=\; \partial_x(K_h \partial_x C) + \partial_y(K_h \partial_y C) + \partial_z(K_z \partial_z C).
$$

The horizontal Laplacian (first two terms) is computed per z-level by `finitevolx.Diffusion3D`. The vertical part is assembled by `_vertical_ops.vertical_diffusion_tendency` with face-averaged diffusivity and central finite differences on a uniform grid:

$$
\partial_z(K_z \partial_z C)\big|_k \approx \frac{\tfrac12(K_{z,k} + K_{z,k+1})\,(C_{k+1} - C_k)/\Delta z \;-\; \tfrac12(K_{z,k-1} + K_{z,k})\,(C_k - C_{k-1})/\Delta z}{\Delta z}.
$$

For uniform $K_z$ this collapses to the standard three-point stencil $(C_{k+1} - 2 C_k + C_{k-1})/\Delta z^2$.

### 3.4 Source term

The regularised point source writes

$$
S(\vec x, t) = q(t) \cdot \rho_s(\vec x), \qquad \rho_s(\vec x) = \frac{1}{Z}\exp\!\left(-\frac{|\vec x - \vec x_s|^2}{2 r_s^2}\right),
$$

with $Z$ chosen so that $\int \rho_s\, \mathrm d V = 1$ over the interior (discrete cell-sum $\times \Delta x \Delta y \Delta z$). The default radius is $r_s = 2 \max(\Delta x, \Delta y, \Delta z)$ — wide enough to avoid single-cell aliasing on coarse grids. A time-varying rate $q(t)$ is supplied as a JAX-compatible callable; a scalar rate is treated as a closure that discards $t$. See `les_fvm.source.make_gaussian_source`.

## 4. Boundary conditions

finitevolX ships 2-D BC atoms (`Dirichlet1D`, `Neumann1D`, `Outflow1D`, `Periodic1D`, ...) that update a ghost ring on a `[Ny, Nx]` field. Applying them to a 3-D field needs two composable helpers that live in `les_fvm.boundary`:

1. `HorizontalBC` — wraps a `finitevolx.BoundaryConditionSet` and vectorises it with `eqx.filter_vmap` over the z-axis, so every z-slice receives the same per-face BC update.
2. `VerticalBC` — directly updates the `[0, :, :]` and `[-1, :, :]` ghost slices with `{Dirichlet, Neumann, Outflow, Periodic}` flavours.

The default concentration BCs are:

- **x-axis**: `Dirichlet` with value 0 upstream (clean inlet) / `Outflow` downstream (zero-gradient).
- **y-axis**: `Periodic` laterally (standard assumption for plume experiments in a homogeneous transect).
- **z-axis**: zero-`Neumann` at the ground and top (no flux through solid walls).

These are all overridable at the `simulate_eulerian_dispersion` call site.

## 5. Time integration

The tendency is assembled in `les_fvm.dynamics.EulerianDispersionRHS` and wrapped in a `diffrax.ODETerm`. The default solver is `diffrax.Tsit5` (adaptive 5th-order Runge-Kutta) with PID step control; fixed-step alternatives (`diffrax.Heun`, `finitevolx.RK3SSP`) are available for benchmarking. Because the RHS is pure, monolithically JIT-compiled, and operates entirely on JAX arrays, the entire simulation — grid construction, wind evaluation, advection-diffusion tendencies, BC application, and source injection — runs inside a single compiled graph.

## 6. CFL and stability

The explicit scheme is CFL-limited by the fastest wave in the system:

$$
\Delta t \;<\; \min\!\left( \frac{\Delta x}{|u|}, \frac{\Delta y}{|v|}, \frac{\Delta z}{|w|}, \frac{\Delta x^2}{2 K_h}, \frac{\Delta y^2}{2 K_h}, \frac{\Delta z^2}{2 K_z} \right),
$$

with a safety factor around 0.4 for WENO5. Tsit5's adaptive controller in practice selects $\Delta t$ that stays well inside this bound, but users running fixed-step (`Heun` / `RK3SSP`) should set `dt0` explicitly.

## 7. Cross-check against `gauss_puff`

Under spatially-uniform wind + zero vertical velocity + PG-calibrated diffusivity at a reference distance matching the observation point, the L2 solution should agree with the Gaussian puff centreline concentration to within discretisation error. Notebook [03_puff_vs_eulerian.ipynb](03_puff_vs_eulerian.ipynb) runs this cross-check.

[^stockie2011]: Stockie, J. M. (2011). *The Mathematics of Atmospheric Dispersion Modelling*. SIAM Review, **53**(2), 349-372. https://www.sfu.ca/~jstockie/atmos/paper.pdf.

[^seinfeld2016]: Seinfeld, J. H. & Pandis, S. N. (2016). *Atmospheric Chemistry and Physics*. Wiley. Eulerian transport and K-theory in §18.
