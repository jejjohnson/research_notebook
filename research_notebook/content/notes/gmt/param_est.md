# Parameter Estimation



Given some data

$$
\mathcal{D}= \left\{ \mathbf{y}_{n}, \mathbf{u}_n  \right\}_{n=1}^N
$$

We can estimate the parameters, $\boldsymbol{\theta}$, of the model, $\mathcal{M}$, from the data, $\mathcal{D}$.

$$
p(\boldsymbol{\theta}|\mathcal{D}) = \frac{p(\mathcal{D}|\boldsymbol{\theta})p(\boldsymbol{\theta})}{p(\mathcal{D})}
$$

So the name of the game is to approximate the true state, $\mathbf{u}$, using a model, $\mathbf{u}_{\boldsymbol{\theta}}$, and then minimize the data likelihood

$$
\begin{aligned}
 \mathbf{u} &\approx \mathbf{u}_{\boldsymbol{\theta}} \\
\boldsymbol{\theta}^* &=
\underset{\boldsymbol{\theta}}{\text{argmax}} \hspace{2mm}
\mathcal{L}(\boldsymbol{\theta};\mathbf{u}_{\boldsymbol{\theta}})
\end{aligned}
$$

Some example parameterizations of the state, $\mathbf{u}_{\boldsymbol \theta}$:

* ODE/PDE
* Neural Field
* Hybrid Model (Parameterization)


Most of the prior information that can be embedded within the problem:

* Data, e.g. Historical observations
* State Representation / Architectures, e.g. coords --> NerFs, discretized --> CNN
* Loss, e.g. physics-informed, gradient-based



---

## Examples


### Partial Differential Equations

We could describe the state via a PDE with some constraints on variants of the derivative of the field wrt space and time.

Let's assume we approximate the true state, $\boldsymbol{u}$, with an approximation

$$
\boldsymbol{u}(\vec{\mathbf{x}},t) \approx \boldsymbol{u_\theta}(\vec{\mathbf{x}},t)
$$

Now, we can add a constraint on the field s.t. it respects some physical laws.
These laws are prescribed by constraining the derivative wrt space and time.
So we can explicitly constrain the state with the following set of equations:

$$
\begin{aligned}
\text{Equation of Motion}: && \boldsymbol{R}(\boldsymbol{u};\boldsymbol{\theta} )&=
\partial_t\boldsymbol{u} -
\boldsymbol{F}[\boldsymbol{u};\boldsymbol{\theta}](\vec{\mathbf{x}},t),
\hspace{10mm} &t\in\mathcal{T}, \hspace{2mm} \vec{\mathbf{x}}\in\Omega  \\
\text{Boundary Conditions}: && \boldsymbol{R_{bc}}(\boldsymbol{u};\boldsymbol{\theta} ) &=
\boldsymbol{u}(\vec{\mathbf{x}},t),
\hspace{10mm} &t\in\mathcal{T}, \hspace{2mm}  \partial\vec{\mathbf{x}}\in\Omega \\
\text{Initial Conditions}: &&
\boldsymbol{R_{ic}}(\boldsymbol{u};\boldsymbol{\theta} ) &=
\boldsymbol{u}_0(\vec{\mathbf{x}},0),
\hspace{10mm} &\vec{\mathbf{x}}\in\Omega \\
\end{aligned}
$$

where $\boldsymbol{R}(\cdot)$ is the constraint on the state itself, $\boldsymbol{R_{bc}}(\cdot)$ is the constraint on the boundaries, and $\boldsymbol{R_{ic}}(\cdot)$ is the constraint on the initial condition.
To obtain the solution of this PDE, we have the fundamental theorem of calculus which tells us the solution is of the form:

$$
\boldsymbol{u_\theta}:=\boldsymbol{u}(t) = \boldsymbol{u}(0) +
\int_{0}^T \boldsymbol{F}[\boldsymbol{u};\boldsymbol{\theta}]
(\vec{\mathbf{x}},\tau)\tau,
$$

So assuming we can plug-in-play to find the solution, we can then plug in this to get a solution to the state.
From this solution, we can try to minimize the likelihood function for the data, $p(\mathcal{D}|\boldsymbol{\theta})$.
An example likelihood function is to use a Gaussian assumption with a constant noise value.

$$
p(\boldsymbol{y}|\boldsymbol{u},\boldsymbol{\theta}) =
\mathcal{N}\left(\boldsymbol{y}|\boldsymbol{H_\theta}, \sigma^2\right)
$$

Now, we can plug this into the full formulation for the loss function:

$$
\begin{aligned}
\boldsymbol{\theta}^* &=
\underset{\boldsymbol{\theta}}{\text{argmax}} \hspace{2mm}
\mathcal{L}(\boldsymbol{\theta};\mathbf{u}_{\boldsymbol{\theta}}) \\
&= \underset{\boldsymbol{\theta}}{\text{argmax}} \hspace{2mm}
\frac{1}{2\sigma^2}||\boldsymbol{y} -\boldsymbol{H_\theta}(\boldsymbol{u_\theta})||^2_2
\end{aligned}
$$




````{admonition} Code Formulation
:class: dropdown idea

```python
# initialize the domain
domain: Domain = Grid(N=(100,100), dx=(0.01, 0.01))

# initialize the BC and IC functions
bc_fn: Callable = ...
ic_fn: Callable = ...

# initialize the field and spatial discretization
state_init: Discretization = FiniteDiffDiscretization(domain, ic_fn)

# describe the equation of motion (+ params)
params: Params = ...

def equation_of_motion(t, u, params):
    # parse state and params
    u = state.u
    diffusivity = params.nu

    # apply BCs
    u = bc_fn(u)

    # equation of motion
    u_rhs = diffusivity * laplacian(u)

    return u_rhs

# initialize the time stepper (solver)
solver: TimeStepper = Euler()

# solver ODE solution
u = ODESolve(fn, solver, u_state, params, bc_fn)
```


````

---

### Neural Fields

Neural fields (NerFs) are a coordinate-based model that parameterizes the field based on the spatio-temporal locations.

$$
\boldsymbol{u}(\vec{\boldsymbol{x}}, t) \approx \boldsymbol{u}_{\boldsymbol{\theta}}(\vec{\mathbf{x}},t)
$$

The parameters, $\boldsymbol{\theta}$, could be a function or it could be parameters described by a PDE.
Now, if we assume that we don't have

$$
\boldsymbol{y} = \boldsymbol{H}[\boldsymbol{u_{\boldsymbol{\theta}}};\boldsymbol{\theta}](\vec{\mathbf{x}}, t) + \varepsilon_n, \hspace{5mm} \varepsilon\sim\mathcal{N}(0,\sigma^2)
$$

**Note**: In this example, we assumed that we simply observe a corrupted version of the state. However, we could also not directly observe the state and we would need an additional operator. For example, we could have a composite operator that first parameterizes the state and then transforms the state quantity to the observed quantity, i.e. $y = \boldsymbol{H}_q \circ \boldsymbol{H}_u (\mathbf{x},t)$


---

### Conditional Generative Models

First, let's assume that we have a likelihood term which describes the posterior directly.

$$
p(\boldsymbol{u}|\boldsymbol{y})
$$

Now, we want to estimate the distribution of

$$
\begin{aligned}
\text{Conditional Prior}: &&
\boldsymbol{z} &\sim p_{\boldsymbol\theta}(\boldsymbol{z}|\boldsymbol{y})\\
\text{Conditional Transform}: &&
\boldsymbol{u} &= \boldsymbol{T_\theta}(\boldsymbol{z}|\boldsymbol{y})
\end{aligned}
$$

$$
\begin{aligned}
\text{Conditional Prior}: &&
\boldsymbol{z} &\sim p_{\boldsymbol\theta}(\boldsymbol{z}|\boldsymbol{y})\\
\text{Conditional Transform}: &&
\boldsymbol{u} &\sim p_{\boldsymbol\theta}(\boldsymbol{u}|\boldsymbol{z},\boldsymbol{y})
\end{aligned}
$$

We can use the change of variables formulation to estimate the conditional distribution

$$
p(\boldsymbol{u}|\boldsymbol{y}) = p(\boldsymbol{z}|\boldsymbol{y})|\det\boldsymbol{\nabla_u T_\theta}^{-1}(\boldsymbol{u}|\boldsymbol{y})|
$$


$$
\boldsymbol{\theta}^* =
\underset{\boldsymbol \theta}{\text{argmin}}
\hspace{2mm}
\mathcal{L}(\boldsymbol{\theta})
$$

where

$$
\begin{aligned}
\mathcal{L}(\boldsymbol{\theta})  &=
\log p(\boldsymbol{u}|\boldsymbol{y}) \\
&= \log p(\boldsymbol{z}|\boldsymbol{y}) +
\log|\det\boldsymbol{\nabla_u T_\theta}^{-1}(\boldsymbol{u}|\boldsymbol{y})|
\end{aligned}
$$
