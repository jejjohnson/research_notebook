# Elliptical PDE Solvers

## Motivation

In many PDEs, we have the famous elliptical system of the form:

$$
\begin{aligned}
\nabla^2u &= f \\
\mathbf{Lu} &=\mathbf{F}
\end{aligned}
$$

where $\nabla^2$ is the Laplacian operator, $f$ is a source term, $u$ is the unknown, $\mathbf{L}$ is a matrix representation of the Laplacian operator, and $\mathbf{F}$ is a matrix representation of the source term. So we need to solve this equation for the unknown $\mathbf{u}$. Fortunately, many PDEs have a simple linear operator, $\nabla^2$, so we can simply solve this with an inversion method.

$$
\mathbf{u} = \mathbf{L}^{-1}\mathbf{F}
$$

Typically we have to use some sort of linear solver in order to do the inverse.
There are traditional inversion schemes, , e.g. Jacobi, Successive OverRelaxation (SOR), Optimized SOR, etc, but these methods can be extremely expensive because they can take a long time to converge.
There are other iterative schemes which have magnitudes of order of speed up, e.g. Steepest Descent, Conjugate Gradient (CG), Preconditioned CG, Multi-Grid schemes, etc.
However, there is an alternative method to solving simple elliptical PDEs whereby we use the *Discrete Sine Transform*.
These are my notes about the formulation, the use cases and also my imagined pseudo-code.

## Formulation

Let's take a field, $u$, which describes some quantity of interest

$$
\begin{aligned}
u =\boldsymbol{u}(\vec{\mathbf{x}}) && && \vec{\mathbf{x}}\in\Omega\sub\mathbb{R}^D
\end{aligned}
$$ (eq:field)

and derive a time-independent solution which takes the form of a generic Elliptical PDE.

$$
\begin{aligned}
\nabla^2u &=
\boldsymbol{f}(\vec{\mathbf{x}})
&& && \vec{\mathbf{x}}\in\Omega \\
\boldsymbol{b}(\vec{\mathbf{x}})
&= b_u
&& && \vec{\mathbf{x}}\in\partial\Omega \\
\end{aligned}
$$ (eq:PDE)

Technically, we can decompose the field, $u$, into two fields. For example, we could have

$$
\begin{aligned}
\begin{cases}
u = u_1 + u_2 && && \vec{\mathbf{x}}\in\Omega \\
u = u_1 + u_2  && && \vec{\mathbf{x}}\in\partial\Omega
\end{cases}
\end{aligned}
$$ (eq:PDE_decomposed)

We will propose one such decomposition here. We observe that all known boundary conditions can appear on the right hand side (RHS) of the PDE {eq}`eq:PDE` because they are known. So we can write the decomposition of the PDE as an interior PDE and an exterior PDE:

$$
\begin{aligned}
\begin{cases}
u = u_\mathrm{I} + u_\mathrm{B} && && \vec{\mathbf{x}}\in\Omega \\
u = u_\mathrm{I} + u_\mathrm{B}  && && \vec{\mathbf{x}}\in\partial\Omega
\end{cases}
\end{aligned}
$$ (eq:PDE_decomposed_bv)

We can say that the interior PDE, $u_\mathrm{I}$, is zero along the boundaries and the exterior PDE, $u_\mathrm{B}$, is zero everywhere except the boundaries.
Notice: The only nonzero values of the $u_\mathrm{B}$ term are along the boundaries.
So we can write the PDE constraint as:

$$
\begin{aligned}
\nabla^2u_\mathrm{B} &= \boldsymbol{f}_\mathrm{B}
&& && \vec{\mathbf{x}}\in\Omega \\
u_\mathrm{B} &= b_u
&& && \vec{\mathbf{x}}\in\partial\Omega \\
\end{aligned}
$$ (eq:PDE_exterior)

where the values along the boundary $\partial\Omega$ are what we set them as and the values within the domain $\Omega$ are "unknown" and the values.
So we can rewrite the solution to the PDE in {eq}`eq:PDE` taking into account the decomposition.

$$
\begin{aligned}
\nabla^2u_\mathrm{I} &= -\nabla^2 u_\mathrm{B} + \boldsymbol{f}
&& && \vec{\mathbf{x}}\in\Omega \\
u_\mathrm{I} &= 0
&& && \vec{\mathbf{x}}\in\partial\Omega \\
\end{aligned}
$$ (eq:PDE_interior)

where we see that we're left with Dirichlet boundary conditions (BC) for the interior PDE. The extra term $\nabla^2 u_\mathrm{B}$ can be thought of as removing the contaminated values from the interior points.
Now we can see that the solution to this PDE will satisfy the following decomposition

$$
\begin{aligned}
\begin{cases}
u = u_\mathrm{I} && && \vec{\mathbf{x}}\in\Omega \\
u = u_\mathrm{B}  && && \vec{\mathbf{x}}\in\partial\Omega
\end{cases}
\end{aligned}
$$ (eq:PDE_decomposed)

whereby we simply need to solve the interior PDE given by equation {eq}`eq:PDE_interior` and then add the boundary values from our original PDE {eq}`eq:PDE`.

### Discrete Sine Transformation

Now we have an elliptical PDE which is zero on all of the boundaries, i.e. Dirichlet boundary conditions.
This means we can use ultra-fast methods to solve this like the *Discrete Sine Transform* (DST).
This particular solver does work on *linear* Elliptical PDEs with Dirichlet boundary conditions.

The Laplacian operator in DST space is:

$$
\hat{\nabla}_H^2 =
2\left[
    \cos\left(\frac{2\pi n}{M-1}-1\right)\Delta x^{-2} +
    \cos\left(\frac{2\pi m}{N-1}-1\right)\Delta y^{-2}
\right]
$$

There are two DST types that we can use. The first one is the DST-I transform given by this equation

$$
\begin{aligned}
\text{DST-I}[x]_k &= \sum_{n=0}^{N-1} x_n \sin\left[ \frac{\pi}{N-1}(n+1)(k+1)\right] && && k=0,\ldots, N-1
\end{aligned}
$$

This transformation is useful when we have an unknown, $u$, a RHS, $r$, and Dirichlet BCs all on the same grid.
The other type of DST transform, i.e. DST-II, and it is given by

$$
\begin{aligned}
\text{DST-I}[x]_k &=
\sum_{n=0}^{N-1} x_n
\sin\left[
    \frac{\pi}{N}(n+\frac{1}{2})(k+1)\right] && && k=0,\ldots, N-1
\end{aligned}
$$

This transformation is useful with $u,r$ are defined at the cell center but the Dirichlet BCs are on the cell corners.

So both the interior points and the boundary values are both satisfied.
Now, let's say that the interior points, $\vec{\mathbf{x}}$,

$$
\begin{aligned}
\begin{cases}
u_1 = u && && \vec{\mathbf{x}}\in\Omega \\
u_1 = 0  && && \vec{\mathbf{x}}\in\partial\Omega
\end{cases} \\
\begin{cases}
u_2 = 0 && && \vec{\mathbf{x}}\in\Omega \\
u_2 = u  && && \vec{\mathbf{x}}\in\partial\Omega
\end{cases}
\end{aligned}
$$

To rename them, let's call PDE 1, $u_1$, the solution of the PDE at the interior points with dirichlet boundaries, $u_I$, and let's call PDE 2, $u_2$, the solution of the PDE at the boundaries with all interior points being zero, $u_B$.

So going back to the original PDE {eq}`eq:PDE`, we can incorporate these constraints into the solutions:


## Use Cases

There are some PDEs where we define a PDE in terms of a derived variable however, we have access to some measurements of the original variable. For example, let's say that we observe some variable, $\xi$, that has a relationship with a PDE field, $u$.

$$
\begin{aligned}
\frac{\partial u}{\partial t} &= \mathcal{N}[u,\xi](\mathbf{x}) \\
\end{aligned}
$$

$$
(\nabla^2 - k^2)u = \xi
$$


````{admonition} Example: Poisson Equation
:class: dropdown, info

````


````{admonition} Example: Quasi-Geostrophic Equations
:class: dropdown, info

We have many approximate models for sea surface height (SSH), $\eta$. One such model is called the Quasi-Geostrophic (QG) Equations.
In this model, we can directly relate SSH to the streamfunction, $\psi$, and subsequently the potential vorticity, $q$.
Let's define these as scalar fields as

$$
\begin{aligned}
u=\boldsymbol{u}(\vec{\mathbf{x}}) && &&
\psi=\boldsymbol{\psi}(\vec{\mathbf{x}})&& &&
q=\boldsymbol{q}(\vec{\mathbf{x}})
\end{aligned}
$$

where the input vector $\vec{\mathbf{x}}=[x,y]^\top$ defined on a bounded domain, $\vec{\mathbf{x}}\in\Omega\sub\mathbb{R}^2$.
Both the PV and streamfunction are derived variables of SSH given by the following relationship:

$$
\begin{aligned}
q &= \nabla^2\psi - b\psi \\
\psi &= a\eta
\end{aligned}
$$

where $a=\frac{g}{f_0},b=\frac{f_0^2}{c_1^2}$.
The QG equations relate the variables via the following PDE:

$$
\begin{aligned}
\frac{\partial q}{\partial t} +
\det\boldsymbol{J}(\psi,q) &= 0 && &&
\vec{\mathbf{x}}\in\Omega \\
\end{aligned}
$$

To simplify the PDE, we can write everything in terms of SSH, $\eta$.
So substituting $\eta$ into each of the derived variables $\psi,q$ gives us

$$
\begin{aligned}
q &= a\nabla^2\eta - ab\eta \\
\psi &= a\eta
\end{aligned}
$$

Now, we can substitute these values in terms of SSH into the QG equation.

$$
\begin{aligned}
\partial_t \left(\nabla^2 - b \right)\eta &+
a\det J(\eta, \nabla^2\eta) = 0 && && \vec{\mathbf{x}}\in\Omega\\
\mathcal{BC}[\eta](\vec{\mathbf{x}}) &=
\boldsymbol{b}(\vec{\mathbf{x}}) && &&
\vec{\mathbf{x}} \in \partial\Omega
\end{aligned}
$$

However, the QG equations describe a PDE which time steps through a derived quantity of SSH, i.e. the Helmholtz operator, $(\alpha\nabla^2 - \beta^2)$.
So this means, that we can will have boundary values of SSH but not boundary values of the Helmholtz of SSH.

Imagine we have to time step with $\eta^n$, we need to isolate the right-hand side of the PDE, which gives us:

$$
\partial_t \left(\nabla^2 - b\right)\eta = -a\boldsymbol{J}(\eta,\nabla^2\eta)
$$

but of course, now we have to solve the inverse Helmholtz operator to isolate $\eta$.
Let $\mathbf{L}_\mathrm{H}=(\nabla^2 -\beta)$ be the linear operator we wish to invert and $\mathbf{F}=-a\det\boldsymbol{J}(\eta,\nabla^2\eta)$ be the RHS of the equation.
So we would do a linear solve:

$$
\begin{aligned}
\mathbf{L}_\mathrm{H}\boldsymbol{\eta} &= \mathbf{F} \\
\boldsymbol{\eta} &= \mathbf{L}_\mathrm{H}^{-1}\mathbf{F}
\end{aligned}
$$

How we actually solve this equation is where we can use the generic algorithm that we outlined below.
Now we can step forward in time with our generic time stepper, $\boldsymbol{g}$, to give us the next time step $t+1$ for $\eta$.

$$
\eta^{t+1} = \eta^t + \boldsymbol{g}(\Delta t, \eta^t, \mathbf{L}_\mathrm{H}^{-1}\mathbf{F})
$$

````


````{admonition} Example: Navier-Stokes
:class: dropdown, info

````



## Practical Algorithm

**Define field of interest**

$$
\begin{aligned}
u &= u(\mathbf{x}) && &&
\mathbf{x}= [x, y]^{\top}
&&\mathbf{x}\in\Omega
\end{aligned}
$$

```python
domain: Domain = ...
u: Field = init_field(domain)
f: Field = init_source(domain)
```


**Solve PDE and BCs on the Full Field**

$$
\begin{aligned}
\mathcal{N}[u](\mathbf{x}) &= f && && \mathbf{x}\in\Omega \\
\mathcal{BC}[u](\mathbf{x}) &= b && && \mathbf{x}\in\partial\Omega
\end{aligned}
$$

```python
u = PDESolver(f, rhs_fn, bcs_fn)
```

**Parse the field into a zero field but maintaining the boundaries**

$$
\begin{aligned}
u_\mathrm{B} &= 0 && && \mathbf{x}\in\Omega \\
u_\mathrm{B} &= b && && \mathbf{x}\in\partial\Omega
\end{aligned}
$$

```python
u_b = zero_interior(u)
u_b = bcs_fn(u)
```

**Apply the same PDE and Boundary Conditions on the empty Field**

$$
\begin{aligned}
\mathcal{N}[u_\mathrm{B}](\mathbf{x}) &= f && && \mathbf{x}\in\Omega \\
\mathcal{BC}[u_\mathrm{B}](\mathbf{x}) &= b && && \mathbf{x}\in\partial\Omega
\end{aligned}
$$

```python
f_b = PDESolver(u_b, rhs_fn, bcs_fn)
```

**Remove the Boundary Effects on the Solution**

$$
\begin{aligned}
u_\mathrm{I} &= u - u_\mathrm{B} && && \mathbf{x}\in\Omega
\end{aligned}
$$

```python
f_I = f[1:-1,1:-1] - f_b[1:-1,1:-1]
```

**Do Inversion for Elliptical PDE**

$$
\begin{aligned}
\mathbf{Lu}_\mathrm{I} &= \mathbf{F} \\
\mathbf{u}_\mathrm{I} &= \mathbf{L}^{-1}\mathbf{F}
\end{aligned}
$$

```python
matvec_Ax: Callable = ...
u_I = LinearSolve(matvec_Ax, b=f_I)
```

**Add the boundaries to the field**


$$
\begin{aligned}
u &= u_\mathrm{I} && && \mathbf{x}\in\Omega \\
u &= u_\mathrm{B} && && \mathbf{x}\in\partial\Omega
\end{aligned}
$$

```python
u = zeros_like(u)
u[1:-1,1:-1] = u_I
u[bc_indices] = u_B
```


## PseudoCde


### High Level


```python
# get derived variable
q, u_bc = load_state(...)
# define callable function
matvec_Ax: Callable = ...
# solve for variable
u: Field = LinearSolve(q, matvec_Ax)
# add boundary conditions
u: Field = boundary_conditions(u, u_bv)
# calculate RHS
rhs: Field = RHS(u, q)
# step once
u_new: Field = TimeStepper(dt, u, rhs)
```
