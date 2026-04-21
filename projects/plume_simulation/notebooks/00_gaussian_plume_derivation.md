---
title: "Gaussian plume — derivation from the advection-diffusion equation"
---

# Derivation of the Gaussian plume solution

This note derives the steady-state Gaussian plume solution used by the forward model in `plume_simulation.gauss_plume.plume.plume_concentration`. The derivation follows Stockie [^stockie2011] closely; Stockie's paper is an excellent pedagogical reference that starts from the full advection-diffusion PDE, states the simplifying assumptions explicitly, and walks through the Laplace-transform and Green's-function solutions in detail. The structure, equation numbering, and notation of this note mirror Stockie §§2-3; any reader who wants more depth — in particular the proofs behind the equivalence theorem (2.4) ≡ (2.5), the Green's-function route, or the discussion of the eddy-diffusivity / σ-parameterisation inconsistency — should consult the paper itself.

[^stockie2011]: Stockie, J. M. (2011). *The Mathematics of Atmospheric Dispersion Modelling*. SIAM Review, **53**(2), 349-372. Preprint: https://www.sfu.ca/~jstockie/atmos/paper.pdf. Referenced here as "Stockie (2011)". All equation numbers cited with the form "(2.1)" refer to that paper; equation numbers without a Stockie prefix are introduced in this note.

## 1. Governing equation (Stockie §2)

Let $C(\vec{x}, t)$ [kg/m³] denote the mass concentration of a single contaminant at position $\vec{x} = (x, y, z) \in \mathbb{R}^3$ [m] and time $t \in \mathbb{R}$ [s]. Conservation of mass (Stockie eq. 2.1) reads

$$
\frac{\partial C}{\partial t} + \nabla \cdot \vec{J} \;=\; S,
$$

where $S(\vec{x}, t)$ [kg/m³·s] is a volumetric source term and $\vec{J}$ [kg/m²·s] is the total contaminant mass flux. Two physical mechanisms contribute to $\vec{J}$. **Advection** by the wind field $\vec{u}$ [m/s] contributes $\vec{J}_A = C \vec{u}$. **Turbulent diffusion** contributes a Fickian flux $\vec{J}_D = -\mathbf{K}\, \nabla C$, where $\mathbf{K}(\vec{x}) = \operatorname{diag}(K_x, K_y, K_z)$ [m²/s] is the eddy diffusivity — an effective, position-dependent diffusion tensor that parameterises turbulent mixing at scales well below the resolution of interest. Summing the two contributions and substituting into the conservation law yields the **three-dimensional advection-diffusion equation** (Stockie eq. 2.2):

$$
\frac{\partial C}{\partial t} + \nabla \cdot (C \vec{u}) \;=\; \nabla \cdot (\mathbf{K}\, \nabla C) + S.
$$ (eq-ad)

This PDE is the starting point. It is exact under the Fickian-closure hypothesis for the turbulent flux; everything that follows is a chain of simplifications to reduce it to a form with a closed-form solution.

## 2. The seven classical assumptions (Stockie §2)

To recover the Gaussian plume, Stockie makes seven assumptions (A1-A7). Each is a physically-motivated simplification; together they replace the full PDE by a linear, constant-coefficient problem that admits a closed-form solution.

- **(A1)** The source is a **continuous point source** at height $H$ above the ground, releasing at constant rate $Q$ [kg/s]: $S(\vec{x}) = Q\, \delta(x) \delta(y) \delta(z - H)$. $H$ is the *effective* stack height $h + \delta h$, the sum of the physical stack height and the plume rise $\delta h$ due to buoyancy.
- **(A2)** The **wind is constant and aligned with $+x$**: $\vec{u} = (u, 0, 0)$ with $u \geq 0$. This is both a physical assumption ("no crosswind, no vertical mean motion") and a coordinate choice — in practice we solve the PDE in a wind-aligned frame and rotate back.
- **(A3)** The solution is **steady state**: $\partial C / \partial t = 0$. Valid when $u$ and $\mathbf{K}$ are time-independent on the timescale of interest (a single satellite overpass, say).
- **(A4)** The eddy diffusivities depend only on downwind distance: $K_x(x) = K_y(x) = K_z(x) =: K(x)$. Diffusion is assumed *isotropic* — a strong simplification that we will revisit when introducing the Briggs σ-parameterisation below.
- **(A5)** Along-wind diffusion is small compared with advection, so $K_x\, \partial_x^2 C$ is dropped. Stockie's Exercise 2 non-dimensionalises the full PDE to justify this: the Péclet number $u H / K$ is large for realistic $u \sim 1\text{-}10$ m/s, $H \sim 10^1\text{-}10^2$ m, $K \sim 1\text{-}10$ m²/s.
- **(A6)** The ground surface is flat: $z = 0$, no topography.
- **(A7)** The contaminant **does not penetrate the ground**. This closes the problem with a zero-flux Neumann BC at $z = 0$.

Applying (A1)-(A6) to [equation](#eq-ad) gives the scalar, elliptic boundary-value problem (Stockie 2.4a-c):

$$
u\, \frac{\partial C}{\partial x} \;=\; K\, \frac{\partial^2 C}{\partial y^2} + K\, \frac{\partial^2 C}{\partial z^2} + Q\, \delta(x)\, \delta(y)\, \delta(z - H),
$$

on the half-space $x \geq 0$, $z \geq 0$, $y \in (-\infty, \infty)$, with boundary conditions $C(0, y, z) = 0$ (no contaminant upwind), $C(\infty, y, z) = C(x, \pm\infty, z) = C(x, y, \infty) = 0$ (finite total mass), and the ground-reflection Neumann condition $K\, \partial C / \partial z (x, y, 0) = 0$ from (A7).

## 3. Boundary-source equivalence (Stockie §2, eqs. 2.5)

Stockie notes that the source term can be absorbed into the boundary condition at $x = 0$: the volumetric source $Q\, \delta(x)\, \delta(y)\, \delta(z - H)$ is equivalent to a surface source $C(0, y, z) = (Q/u)\, \delta(y)\, \delta(z - H)$ on the inflow plane, with no interior source. This is a standard result (Stakgold's theorem; Stockie's Exercise 1 asks the reader to prove it by integrating the PDE over $x \in [-d, d]$ and taking $d \to 0^+$). We will use the boundary-source form — it simplifies the Laplace-transform manipulation.

## 4. Change of variable: the r-coordinate (Stockie §3, eq. 3.1)

Because $K$ can depend on downwind distance $x$ (Assumption A4), the PDE has a *variable* coefficient in the y- and z-Laplacians. Stockie eliminates this by introducing

$$
r \;=\; \frac{1}{u} \int_0^{x} K(\xi)\, \mathrm{d}\xi \qquad [\text{m}^2],
$$ (eq-r)

so that $\partial r / \partial x = K(x) / u$ and $u\, \partial / \partial x = K(x)\, \partial / \partial r$. Substituting, the reduced PDE becomes (Stockie eq. 3.2)

$$
\frac{\partial c}{\partial r} \;=\; \frac{\partial^2 c}{\partial y^2} + \frac{\partial^2 c}{\partial z^2},
$$

a standard two-dimensional heat equation in which $r$ plays the role of time. The boundary conditions carry over: $c$ vanishes at $\infty$ in $y$ and $z$, $K\, \partial c / \partial z (r, y, 0) = 0$, and the "initial condition" at $r = 0$ is the delta-source boundary from §3.

## 5. Separation of variables (Stockie §3, eq. 3.3)

The delta source factorises in $y$ and $z$, suggesting the ansatz

$$
c(r, y, z) \;=\; \frac{Q}{u}\, a(r, y)\, b(r, z).
$$

Substituting and separating yields two 1-D diffusion problems:

- **Crosswind** (Stockie eqs. 3.4):

$$
\frac{\partial a}{\partial r} = \frac{\partial^2 a}{\partial y^2}, \qquad a(0, y) = \delta(y), \qquad a(r, \pm\infty) = 0.
$$

- **Vertical with ground reflection** (Stockie eqs. 3.5):

$$
\frac{\partial b}{\partial r} = \frac{\partial^2 b}{\partial z^2}, \qquad b(0, z) = \delta(z - H), \qquad b(r, \infty) = 0, \qquad \frac{\partial b}{\partial z}(r, 0) = 0.
$$

Each is a 1-D heat equation with a delta initial condition — the first on $\mathbb{R}$, the second on the half-line $[0, \infty)$ with a zero-flux BC at the origin.

## 6. Solving the two subproblems (Stockie §3.1)

Stockie solves both subproblems by Laplace transforms in $r$ and (for the crosswind problem) also in $y$; §3.2 gives an equivalent Green's-function derivation. Since we only need the final solutions, I'll quote them directly and flag the physical content.

### 6.1 Crosswind Gaussian

The $y$-problem is a 1-D heat equation on $\mathbb{R}$ with $a(0, y) = \delta(y)$. Its fundamental solution is the heat kernel (Stockie eq. 3.6):

$$
a(r, y) \;=\; \frac{1}{\sqrt{4 \pi r}}\, \exp\!\left( -\frac{y^2}{4 r} \right).
$$

### 6.2 Vertical Gaussian with ground reflection

The $z$-problem lives on $[0, \infty)$ with a zero-flux boundary at $z = 0$. The **method of images** handles this cleanly: reflect the source at height $H$ to an image source at $-H$ and solve on all of $\mathbb{R}$. The superposition of the two fundamental solutions automatically satisfies $\partial b / \partial z \big|_{z=0} = 0$ because the reflected contribution exactly cancels the vertical derivative of the direct contribution at the ground. The result is (Stockie eq. 3.7):

$$
b(r, z) \;=\; \frac{1}{\sqrt{4 \pi r}}\, \left[\, \exp\!\left( -\frac{(z - H)^2}{4 r} \right) + \exp\!\left( -\frac{(z + H)^2}{4 r} \right) \,\right].
$$

The two exponentials correspond to the **direct** plume path from the physical source at $(0, 0, H)$ and the **reflected** path from the image source at $(0, 0, -H)$ respectively.

## 7. Assembly and σ-rewrite (Stockie §3, eqs. 3.8, 3.9)

Substituting $a$ and $b$ into the separable ansatz gives the **Gaussian plume solution in r-coordinates** (Stockie eq. 3.8):

$$
C(x, y, z) \;=\; \frac{Q}{4 \pi u\, r}\, \exp\!\left( -\frac{y^2}{4 r} \right) \left[\, \exp\!\left( -\frac{(z - H)^2}{4 r} \right) + \exp\!\left( -\frac{(z + H)^2}{4 r} \right) \,\right].
$$

It is standard in the atmospheric-science literature to replace $r$ with the **standard deviations of the Gaussian plume**:

$$
\sigma^2(x) \;=\; \frac{2}{u} \int_0^x K(\xi)\, \mathrm{d}\xi \;=\; 2 r.
$$ (eq-sigma)

With this substitution and distributing the factor of 2 in the exponentials, we recover the canonical form used in every textbook and in `plume_concentration`:

$$
C(x, y, z) \;=\; \frac{Q}{2 \pi u\, \sigma_y\, \sigma_z}\, \exp\!\left( -\frac{y^2}{2 \sigma_y^2} \right) \left[\, \exp\!\left( -\frac{(z - H)^2}{2 \sigma_z^2} \right) + \exp\!\left( -\frac{(z + H)^2}{2 \sigma_z^2} \right) \,\right].
$$ (eq-plume)

This is exactly the expression evaluated in [plume.py](../src/plume_simulation/gauss_plume/plume.py) at the points where `plume_concentration` computes `normalization`, `exp_y`, `exp_z_direct`, and `exp_z_reflected`.

### A subtlety: distinct σ_y and σ_z

Stockie derives the solution under assumption (A4) that $K_x = K_y = K_z = K$ — the eddy diffusivity is isotropic. Under that assumption $\sigma_y = \sigma_z$ follows identically. In practice, atmospheric turbulence is **anisotropic**: vertical mixing is suppressed relative to horizontal mixing by density stratification, and the empirical dispersion parameterisations (Pasquill-Gifford, Briggs-McElroy-Pooler) give different $\sigma_y(x)$ and $\sigma_z(x)$ in every stability regime. The standard practice — which we follow — is to carry [equation](#eq-plume) over verbatim and separately parameterise $\sigma_y$ and $\sigma_z$ from empirical data. This is *technically* an inconsistency with the isotropic-$K$ starting point; Stockie §3.3 discusses it candidly, citing Llewellyn's critique. The error turns out to be small in the regime where the Gaussian plume is used operationally.

## 8. Dispersion-coefficient parameterisations (Stockie §3.3)

Stockie §3.3 notes that $\sigma^2(x)$ is determined by the eddy diffusivity via [equation](#eq-sigma), but that in practice one parameterises $\sigma$ directly from field experiments. He mentions the simple power-law form

$$
\sigma^2(x) \;=\; a x^b,
$$ (eq-sigma-power)

and notes that experimentally $b > 0.70$ in most conditions — larger than the $b = 1/2$ predicted by constant $K$, reflecting the fact that real-atmosphere diffusivities grow with downwind distance.

Our library uses the slightly richer **Briggs-McElroy-Pooler** parameterisation:

$$
\sigma_i(x) \;=\; a_i\, x \, (1 + b_i\, x)^{c_i}, \qquad i \in \{y, z\},
$$ (eq-briggs)

with one coefficient triple $(a_i, b_i, c_i)$ per Pasquill-Gifford stability class A-F. For $b_i = 0$ this reduces to Stockie's form [equation](#eq-sigma-power) with $a = a_i^2$ and $b = 2$; for $b_i > 0$ and $c_i < 0$ the extra factor produces a gentle rollover at large $x$ that matches measurements better than the pure power law. The six stability classes — from A (strongly unstable, sunny convective day) through D (neutral) to F (strongly stable, clear calm night) — give six different $(\sigma_y, \sigma_z)$ curves. Class A produces the largest $\sigma$'s and hence the most diluted plume; class F produces long, thin plumes with high peak concentration near the centerline. [dispersion.py](../src/plume_simulation/gauss_plume/dispersion.py) stores the six parameter triples in `BRIGGS_DISPERSION_PARAMS` and evaluates [equation](#eq-briggs) in `calculate_briggs_dispersion`.

## 9. Mass conservation

Integrating [equation](#eq-plume) across a plane $x = \text{const}$ (that is, integrating in $y \in (-\infty, \infty)$ and in $z \in [0, \infty)$) gives

$$
\int_0^{\infty} \int_{-\infty}^{\infty} C(x, y, z) \, u \, \mathrm{d}y \, \mathrm{d}z \;=\; Q,
$$

which is mass-flux conservation: the downwind flux of contaminant mass through any cross-section equals the source rate. The factor of 2 in the denominator of [equation](#eq-plume) (i.e. $2 \pi u\, \sigma_y\, \sigma_z$ rather than $4 \pi u\, \sigma_y\, \sigma_z$) is exactly what's needed for this — the ground reflection puts the full vertical Gaussian on the physical half-space $z \geq 0$, and the $y$-Gaussian is symmetric around zero.

## 10. Operational conventions and pitfalls

Three small items that are easy to trip over:

- **Wind-direction convention**. In meteorology `wind_direction` is the direction the wind is *from*: `wind_direction = 270°` means a westerly wind (blowing *from* the west, flowing *toward* the east). Our `simulate_plume` adopts this convention; the conversion to Cartesian velocity components is $u = -U \sin(\theta)$, $v = -U \cos(\theta)$ with $\theta$ in radians.
- **Upwind masking**. [equation](#eq-plume) is only valid for $x' > 0$ in the wind-aligned frame. Numerically evaluating the formula at $x' \leq 0$ gives a nonsensical (and very negative) exponent inside $\sigma^2 \propto x'^{1+}$ and we mask those points to zero. `plume_concentration` uses `jnp.where(x_downwind > 0.0, concentration, 0.0)` for this.
- **Calm-wind limit**. [equation](#eq-plume) has a $1/u$ prefactor that diverges as $u \to 0$, but this divergence is spurious — Stockie eq. 3.11 shows that if one carries $r$ rather than $u\sigma^2$, the true $u \to 0$ limit is finite. Our implementation clamps $u$ below a floor `MIN_WIND_SPEED = 0.5 m/s` rather than handling the calm regime rigorously; the rigorous fix is to switch to a Gaussian-puff model (Stockie §3.5.6) when wind is very low, which is on the future-work list for this sub-project.

## What to read next

- **Notebook [01_gaussian_plume_forward](01_gaussian_plume_forward.ipynb)** — numerical evaluation of [equation](#eq-plume) on a 3-D grid, visual checks of the Gaussian cross-sections, and a stability-class sweep illustrating how σ_y, σ_z scale with class.
- **Notebook [02_emission_rate_parameter_estimation](02_emission_rate_parameter_estimation.ipynb)** — Bayesian inversion for $Q$ given a downwind transect, using NumPyro NUTS against [equation](#eq-plume) as the forward operator.
- **Notebook [03_plume_state_estimation](03_plume_state_estimation.ipynb)** — a state-space extension where $Q_t$ evolves as a random walk; the forward model at each $t$ is still [equation](#eq-plume).
- **Stockie §§3.4-3.6** — multi-source superposition (linear in $Q$), the "menagerie" of Gaussian-plume variants (cross-wind-integrated, settling, deposition), and the Gaussian-puff solution for transient / low-wind cases.
- **Stockie §§4-6** — the *inverse* source-identification problem via constrained linear least squares and its application to the Inco Superstack dataset. The Bayesian view in our notebooks is a probabilistic cousin of Stockie's least-squares treatment.
