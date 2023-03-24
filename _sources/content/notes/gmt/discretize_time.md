# Temporal Discretization

## Motivation

The world is continuous but our operations are discrete.

From the modeling world, this is easily one of the most important decisions one needs to make.

It is also responsible for one of the greatest sources of errors.

---

## Formulation

Let’s take the general form of a PDE.

$$
\begin{aligned}
\dot{\boldsymbol{u}}(\mathbf{x}_s,t) &= \mathcal{N}[\boldsymbol{u},\boldsymbol{\theta}](\mathbf{x}_s,t) && && && \mathbf{x}_s\in\Omega\sub\mathbb{R}^{D_s} && t \in \mathcal{T} \sub \mathbb{R}^+ \\
\boldsymbol{b}(\mathbf{x}_s, t) &= \mathcal{B}[\boldsymbol{u},\boldsymbol{\theta}](\mathbf{x}_s,t) && && &&  \mathbf{x}_s\in\partial\Omega && t \in \mathcal{T} \\

\end{aligned}
$$

where:

- $\boldsymbol{u}:\Omega\times\mathcal{T}\rightarrow \Omega$ - is the unknown vector field
-

---

Let’s take the general form of an ODE

$$
\dot{x}(t) = \boldsymbol{F}\left( x(t),t \right) \hspace{10mm} t \in \mathcal{T} \sub\mathbb{R}^+
$$

where:

- $x:\mathcal{T} \rightarrow\mathbb{R}^D$ - is an unknown function
- $\boldsymbol{F}:\mathbb{R}^D\times \mathbb{R} \rightarrow \mathbb{R}^D$ - is a vector field
- $\mathcal{T}$ - is the time domain; typically $\mathcal{T}=[0,\mathcal{T}]$

The art of differential equations is *knowing the form of solution and then finding tricks to arrive there*. So we can write this down using the *fundamental theorem of calculus*. It has the form:

$$
x(t) = x(0) + \int_0^t\boldsymbol{F}(x(\tau),\tau)d\tau
$$

where the solution depends upon the initial value $x(0)=x_0$. So knowing this, we can actually rewrite the general form of the ODE in the above equation to be:

$$
\dot{x}(t) = \boldsymbol{F}\left( x(t),t \right) \hspace{10mm} t \in \mathcal{T} \sub\mathbb{R}^+ \hspace{5mm} x(0) = x_0
$$

where:

- $x:\mathcal{T} \rightarrow\mathbb{R}^D$ - is an unknown function
- $\boldsymbol{F}:\mathbb{R}^D\times \mathbb{R} \rightarrow \mathbb{R}^D$ - is a vector field
- $\mathcal{T}$ - is the time domain; typically $\mathcal{T}=[0,\mathcal{T}]$
- $x_0$ - is the initial value

However, the equation for the solution is difficult because of the integral term. Integrals are hard…as it is impossible to take the expectation of the space with all time. In general, our *numerical solvers* find the solution by stepping forward in time. So instead of finding the complete trajectory from $x(0)$ to $x(t)$, we take a sequence of sub-steps, i.e. $x(t)$ to $x(t+\delta t)$. So we can rewrite the solution as

$$
x(t + \delta t) = x(0) + \int_0^{t+\delta t}\boldsymbol{F}(x(\tau),\tau)d\tau
$$

---

### Taylor Expansion Methods

We can use the Taylor series expansion around:

$$
g(\tau) = \sum_{n=0}^\infty \frac{g^{(n)}(t_0)}{n!}(\tau - t_0)^n
$$

Having access to all of the derivatives would fully describe the dynamical system

$$
x(t+\delta t) = \sum_{k=0}^q\frac{h^k}{k!}x^{(k)}(t) + \mathcal{O}(h^{q+1})
$$

---

### Quadrature Methods

We can use numerical quadrature!

$$
\int_l^r g(\tau)d\tau \approx \sum_{i=1}^Nw_ig(t_i)
$$

which motivates the *explicit Runge-Kutta* algorithm.

$$
\int_t^{t+\delta t} \boldsymbol{F}(x(\tau),\tau)d\tau \approx h \sum_{i=1}^N w_i \boldsymbol{F}(\hat{x}(\tau_i),\tau_i)
$$

---

## Generalized Form

$$
x(t+\delta t) = x(t) + \mathbf{g}_t
$$

$$
[\mathbf{g}_t, \mathbf{h}_{t+\delta t}] = \boldsymbol{g}\left(\boldsymbol{\nabla}\boldsymbol{f}(x(t)),\mathbf{h}_t\right)
$$

Source: *[Learning to Learn with JAX - Teddy Kroker](https://teddykoker.com/2022/04/learning-to-learn-jax/)*

---

### Surrogate Methods

We can even use some

$$
x(t+\delta t) = x(t) + \mathbf{g}
$$

---

# Temporal Discretization

> Parameters are artefacts of discretizations…*
>

Time Steppers - dealing with the integral

`X(t)=x(0) + int f(x(t),t)dt`

$$
u(\mathbf{x}_s, t) = u(\mathbf{x}_s, 0) + \int_{0}^{t} \mathcal{N}[u](\mathbf{x}_s,\tau)d\tau
$$

This integral is a hard problem so we instead try and do a time-stepping way.

$$
u(t+\delta t) = u(t) + \int_{t}^{t+\delta t} \mathcal{N}[u](\mathbf{x}_s, \tau)d\tau
$$

where we can apply this recursively by marching along.

$$
u(t+\delta t) = u(t) + \boldsymbol{g}(u(t), h)
$$

- Euler - correction
- RK - correction, basis functions
- Explicit vs implicit



---

# ML Models

- Discrete Space - Convolutions + Finite Difference
    - Fourier, Real
- Discrete Time - RNNs, LSTMs
    - Basis Function
    - Hidden State
- Both - ConvLSTM
