---
title: "Gaussian puff — derivation from the 3-D advection-diffusion equation"
---

# Derivation of the Gaussian puff solution

This note derives the Gaussian-puff solution used by the forward model in `plume_simulation.gauss_puff.puff.puff_concentration`. The setup mirrors the steady Gaussian plume derivation in [00_gaussian_plume_derivation.md](../00_gaussian_plume_derivation.md), but with a different simplification strategy: instead of dropping the time derivative (A3), we keep it and work with an instantaneous release — the continuous-source plume and the repeated-release puff are two projections of the same 3-D advection-diffusion equation. Stockie [^stockie2011] covers the puff model in §3.5.6 and time-varying wind in §§4.3-4.4; his approach derives the puff by the same Laplace-transform / r-coordinate route he uses for the plume (see Stockie eq. 3.19-3.20). In this note we follow a complementary path — a Galilean transformation into the puff-centre frame — which reduces the PDE to the free-space 3-D heat equation and makes the time-varying-wind extension especially natural. The two routes are equivalent for constant wind; for time-varying wind the Galilean path is the one we implement with `diffrax` in [wind.py](../../src/plume_simulation/gauss_puff/wind.py).

[^stockie2011]: Stockie, J. M. (2011). *The Mathematics of Atmospheric Dispersion Modelling*. SIAM Review, **53**(2), 349-372. https://www.sfu.ca/~jstockie/atmos/paper.pdf. Puff solution and its plume-superposition relation: §3.5.6 (Eqs. 3.19-3.20 and Exercise 8). Time-varying wind via sub-interval quasi-steady plumes: §§4.3-4.4.

[^sp2016]: Seinfeld, J. H. & Pandis, S. N. (2016). *Atmospheric Chemistry and Physics: From Air Pollution to Climate Change* (3rd ed.). Wiley. Gaussian-puff treatment in §18.3.

## 1. Governing equation

As in the plume case, mass conservation for a single contaminant $C(\vec x, t)$ [kg/m³] at position $\vec x = (x, y, z)$ [m] and time $t$ [s] reads (Stockie 2.2)

$$
\frac{\partial C}{\partial t} + \nabla \cdot (C \vec u) \;=\; \nabla \cdot (\mathbf K\, \nabla C) + S,
$$ (eq-ad)

with wind $\vec u$ [m/s], eddy diffusivity $\mathbf K = \operatorname{diag}(K_x, K_y, K_z)$ [m²/s], and volumetric source $S$ [kg/m³·s].

## 2. Assumptions for the puff model

We impose a subset of Stockie's A1-A7 — different from those chosen for the plume.

- **(P1)** The source is **instantaneous**: a single burst of mass $m$ [kg] released at time $t_0$ at position $\vec x_0 = (x_0, y_0, H)$. Formally, $S(\vec x, t) = m\, \delta(t - t_0)\, \delta(\vec x - \vec x_0)$. A continuous source at rate $Q$ [kg/s] is then a superposition of instantaneous puffs of mass $m = Q\, \Delta t$ released every $\Delta t$ seconds.
- **(P2)** The **wind is uniform in space** but may vary in time: $\vec u(t) = (u(t), v(t), 0)$. Vertical wind is neglected.
- **(P3)** The eddy diffusivities are **constant and isotropic horizontally**: $K_x = K_y =: K_h$, $K_z$ possibly different. They may be functions of $t$ (or equivalently, of the puff age $\tau = t - t_0$).
- **(P4)** **Ground reflection** (same as plume A7): no penetration of the contaminant through $z = 0$, closed by a zero-flux Neumann BC, $K_z\, \partial C / \partial z (x, y, 0, t) = 0$.
- **(P5)** No decay or chemistry.

Crucially, we **keep** $\partial C / \partial t$ (unlike A3 in the plume) and we **keep** the along-wind diffusion term $K_x\, \partial^2 C / \partial x^2$ (unlike A5). A puff released from a point source has non-zero spread in *every* direction, not just crosswind; dropping $K_x$ would force an unphysical zero-width along-wind profile at all times.

Applying (P1)-(P3) to [equation](#eq-ad) gives (ignoring the ground reflection for now — we'll add it by method of images in §5)

$$
\frac{\partial C}{\partial t} + u(t)\, \frac{\partial C}{\partial x} + v(t)\, \frac{\partial C}{\partial y} \;=\; K_h\, \left(\frac{\partial^2 C}{\partial x^2} + \frac{\partial^2 C}{\partial y^2}\right) + K_z\, \frac{\partial^2 C}{\partial z^2} + m\, \delta(t - t_0)\, \delta(\vec x - \vec x_0),
$$ (eq-puff)

on $\mathbb{R}^3 \times \mathbb{R}$, with $C(\vec x, t) \to 0$ as $|\vec x| \to \infty$ and $C(\vec x, t < t_0) = 0$.

## 3. Galilean transformation into the puff-centred frame

The advection term can be absorbed by moving into a frame that follows the puff centre. Define the puff-centre trajectory

$$
x_c(t) \;=\; x_0 + \int_{t_0}^{t} u(\tau)\, \mathrm d\tau, \qquad y_c(t) \;=\; y_0 + \int_{t_0}^{t} v(\tau)\, \mathrm d\tau,
$$ (eq-centre)

i.e. the trajectory of a passive tracer at $\vec x_0$ under the wind $(u, v)$. In the moving coordinates $\tilde x = x - x_c(t)$, $\tilde y = y - y_c(t)$, $\tilde z = z$, $\tilde t = t - t_0$, the chain rule gives

$$
\frac{\partial C}{\partial t} \bigg|_{x, y, z} \;=\; \frac{\partial C}{\partial \tilde t} - \dot x_c\, \frac{\partial C}{\partial \tilde x} - \dot y_c\, \frac{\partial C}{\partial \tilde y}, \qquad \frac{\partial C}{\partial x} \bigg|_{t} = \frac{\partial C}{\partial \tilde x},
$$

and since $\dot x_c = u(t)$ and $\dot y_c = v(t)$ by construction, [equation](#eq-puff) becomes

$$
\frac{\partial C}{\partial \tilde t} \;=\; K_h\, \left(\frac{\partial^2 C}{\partial \tilde x^2} + \frac{\partial^2 C}{\partial \tilde y^2}\right) + K_z\, \frac{\partial^2 C}{\partial \tilde z^2} + m\, \delta(\tilde t)\, \delta(\tilde x)\, \delta(\tilde y)\, \delta(\tilde z - H),
$$ (eq-moving)

the **3-D heat equation with a delta source at $\tilde t = 0$**. The Galilean transform has removed the advection completely — this is the same trick that makes plume theory tractable under A2.

## 4. Fundamental solution (unbounded domain)

Equation [moving](#eq-moving) on $\mathbb R^3$ admits the separable fundamental solution

$$
C(\tilde x, \tilde y, \tilde z, \tilde t) \;=\; m\, G_{2 K_h \tilde t}(\tilde x)\, G_{2 K_h \tilde t}(\tilde y)\, G_{2 K_z \tilde t}(\tilde z - H),
$$ (eq-fundamental)

where

$$
G_{\sigma^2}(s) \;=\; \frac{1}{\sqrt{2 \pi \sigma^2}}\, \exp\!\left(- \frac{s^2}{2 \sigma^2}\right)
$$

is the 1-D normal PDF with variance $\sigma^2$. Direct substitution confirms this. The identification

$$
\sigma_x^2(\tilde t) = \sigma_y^2(\tilde t) = 2 K_h\, \tilde t, \qquad \sigma_z^2(\tilde t) = 2 K_z\, \tilde t
$$ (eq-sigmasq)

is the Fickian-diffusion scaling: variances grow linearly in puff age.

## 5. Ground reflection (method of images)

The Neumann boundary $K_z\, \partial C/\partial \tilde z|_{\tilde z = 0} = 0$ can be enforced by superposing the free-space solution with an **image source** at $\tilde z = -H$ (reflected about $z = 0$). The full solution on $\{\tilde z \geq 0\}$ is then

$$
C(\tilde x, \tilde y, \tilde z, \tilde t) \;=\; m\, G_{\sigma_x^2}(\tilde x)\, G_{\sigma_y^2}(\tilde y)\, \Big[\, G_{\sigma_z^2}(\tilde z - H) \;+\; G_{\sigma_z^2}(\tilde z + H)\, \Big],
$$ (eq-puff-solution)

which is the expression implemented in `puff_concentration`. The image source has the same mass as the real source, so the total contaminant mass in the half-space is preserved: $\int_0^\infty (G_{\sigma_z^2}(\tilde z - H) + G_{\sigma_z^2}(\tilde z + H))\, \mathrm d \tilde z = 1$ by construction.

## 6. The σ-parameterisation (Pasquill-Gifford)

As in the plume case, the linear-in-$\tilde t$ scaling [σ²](#eq-sigmasq) is a consequence of *constant* eddy diffusivity — an idealisation. In practice the dispersion is driven by turbulent eddies whose statistics depend on atmospheric stability, surface roughness, and the scale of the eddy relative to the puff age. Empirical fits (Pasquill, 1961; Gifford, 1961) to field data give σ as functions of **puff travel distance** $s = |\vec x_c(t) - \vec x_0|$ rather than age $\tilde t$, because $s$ correlates more directly with the turbulent spectrum encountered by the puff.

The classic PG correlations are tabulated curves per stability class A (very unstable) through F (very stable). Convenient **log-quadratic fits** (Beychok 2005; used in `plume_simulation.gauss_puff.dispersion.calculate_pg_dispersion`) are

$$
\sigma_y(s) = \exp\!\big(a_y + b_y\, \ln s + c_y\, (\ln s)^2\big), \qquad \sigma_z(s) = \exp\!\big(a_z + b_z\, \ln s + c_z\, (\ln s)^2\big),
$$ (eq-pg)

with stability-class-specific coefficients $(a_y, b_y, c_y, a_z, b_z, c_z)$. Horizontal spread is taken isotropic: $\sigma_x = \sigma_y$.

Using [σ²](#eq-sigmasq) we could define an effective time-dependent diffusivity $K_h(\tilde t) = \sigma_y^2(s(\tilde t)) / (2 \tilde t)$, but this is a *fit*, not a first-principles quantity — exactly the eddy-diffusivity / σ-parameterisation inconsistency discussed in Stockie §3.4. The σ-form is more robust empirically, so we adopt it, with the understanding that the "diffusion equation" interpretation is a useful fiction.

`plume_simulation.gauss_puff.dispersion` also exposes the Briggs power-law parameterisation (`calculate_briggs_dispersion_xyz`) shared with the plume model, so the puff and plume can be run with the *same* $\sigma(s)$ curves when a consistent comparison is needed.

## 7. Time-varying wind via cumulative integrals

The wind-dependent part of the solution lives entirely in the puff-centre trajectory [centre](#eq-centre): once we know $\vec x_c(t)$ and the travel distance

$$
s(t) \;=\; \int_{t_0}^{t} \big|\vec u(\tau)\big|\, \mathrm d\tau,
$$ (eq-travel)

we have everything needed to evaluate [puff solution](#eq-puff-solution) with σ from [PG](#eq-pg). The three integrals $I_u(t) = \int_{t_0}^t u\, \mathrm d\tau$, $I_v(t) = \int_{t_0}^t v\, \mathrm d\tau$, $s(t)$ satisfy the trivial ODE system

$$
\frac{\mathrm d}{\mathrm d t} \begin{pmatrix} I_u \\ I_v \\ s \end{pmatrix} \;=\; \begin{pmatrix} u(t) \\ v(t) \\ \sqrt{u(t)^2 + v(t)^2} \end{pmatrix}, \qquad \begin{pmatrix} I_u(t_0) \\ I_v(t_0) \\ s(t_0) \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix},
$$ (eq-wind-ode)

which is solved once with `diffrax.diffeqsolve` in `plume_simulation.gauss_puff.wind.cumulative_wind_integrals`, evaluating at all release times and the current evaluation time in a single adaptive sweep. Puff-$i$ positions are then elementwise differences: $x_c^{(i)}(t) = x_0 + I_u(t) - I_u(t_0^{(i)})$, and likewise for $y_c^{(i)}, s^{(i)}$. This avoids any Python-level per-puff loop and keeps the forward model differentiable for NUTS.

## 8. Continuous source as a superposition of puffs

A continuous source at rate $Q(t)$ [kg/s] releases $m_i = Q(t_i) \Delta t$ every $\Delta t = 1 / f_{\text{release}}$ seconds, starting at $t_0 = 0$. By linearity of [equation](#eq-ad), the total concentration is the superposition

$$
C(\vec x, t) \;=\; \sum_{i : t_i \leq t} C^{(i)}(\vec x, t), \qquad C^{(i)}(\vec x, t) \;=\; [\text{equation}~\href{#eq-puff-solution}{(5)}~\text{with}~m_i, \vec x_0 = \vec x_c(t_i), \tilde t = t - t_i].
$$ (eq-superposition)

`simulate_puff_field` implements this sum via `jax.vmap`. In the limit $f_{\text{release}} \to \infty$ at fixed $Q$, $\Delta t \to 0$ and the sum becomes an integral — this is the formal link to the steady plume. Stockie asks the reader to verify this in his Exercise 8: integrating his puff formula (Stockie Eq. 3.20) over $t \in [0, \infty)$ recovers the plume expression (Stockie Eq. 3.8). For finite $\Delta t$, the puff sum has a jittery structure near the source that the steady plume lacks; smooth-looking fields require $\Delta t$ small compared with the transit time from source to receptor.

### Comparison with Stockie's sub-interval approach (§§4.3-4.4)

For time-varying wind, Stockie §4.3 divides the total window into sub-intervals $\Delta t \sim 10$ min, assumes the wind is steady within each sub-interval, and applies the **steady plume** solution in a rotated coordinate frame aligned with each sub-interval's wind vector. His §4.4 then sums deposition contributions over the sub-intervals. This is a *quasi-steady plume* treatment — fast and accurate for slowly-varying winds, but fundamentally limited to timescales long enough for each sub-interval plume to reach steady state.

The puff formulation in this note sidesteps that limitation: because we keep the time derivative in [equation](#eq-puff), we can handle winds that vary on timescales *shorter* than the plume transit time — a buffeting wind, a gust, a sudden directional shift. The price is computational: instead of one plume per sub-interval we carry an $N_\text{puffs}$-particle ensemble and integrate [wind ODE](#eq-wind-ode) through it. `diffrax` makes this ensemble evolution a single jittable pass.

## 9. Operational pitfalls

Three recurring gotchas in implementing the puff model:

1. **Release cadence vs. receptor distance.** The field within the first few puff-widths of the source is *dominated* by a small number of individual puffs and does not look Gaussian — it looks bumpy. If you are evaluating close to the source, pick $\Delta t$ small enough that several puffs overlap within one $\sigma_y$. A rule of thumb: $\Delta t \lesssim \sigma_y(s) / |\vec u|$ for the shortest relevant $s$.
2. **σ evaluation at $s = 0$.** PG gives $\sigma \to 0$ at the moment of release, so a freshly-released puff has infinite centreline concentration. `calculate_pg_dispersion` clamps $s$ to 1 m to avoid this — physically the puff has a nonzero initial size $\sigma_0$ set by the source geometry, which is ignored by PG but rarely matters at receptor distances $\gtrsim$ tens of metres.
3. **Calm winds.** When $|\vec u| \to 0$, the puff trajectory freezes and puffs pile up at the source. The model in this limit is unphysical — a buoyant source should transition to a *puff-rise* regime, not a static pile. `MIN_WIND_SPEED = 0.5` m/s is a numerical floor documented in `puff.py`; for inference, calms should be filtered out upstream.

These are not bugs in the PG puff model — they are the edges of its domain of validity, inherited from the same idealisations that make it tractable.
