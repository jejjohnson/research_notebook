# 4DVarNet


---

**Observations**

We assume that we have the following representation:

$$
\mathbf{y}_{\text{obs}} = \mathbf{x}_{\text{gt}} + \epsilon
$$

We assume they are the true state, $\mathbf{x}_\text{gt}$, corrupted by some noise, $\epsilon$.


---
**Prior**

We have some way to generate samples. 

$$
\mathbf{x}_\text{init} \sim P_\text{init}
$$

e.g. we have an *inexpensive* physical model where we can generate samples from. We could also use a *cheap* emulator of the physical model to draw samples.

---
**Prior** vs. **Observations**

We need to relate the **prior**, $\mathbf{x}_\text{gt}$, and the observations, $\mathbf{y}_\text{gt}$.

We assume that there is some function, $\boldsymbol{f}$, that maps the observations, $\mathbf{y}_\text{obs}$, and prior, $\mathbf{x}_\text{init}$, to the true state, $\mathbf{x}_\text{gt}$.

$$
\mathbf{x}_\text{gt} = \boldsymbol{f}(\mathbf{x}_\text{init}, \mathbf{y}_\text{obs}; \boldsymbol{\theta})
$$

**Note**:


---
**Data** (Unsupervised)

The first case is when we do not have any realization of the ground truth.

$$
\mathcal{D}_{\text{unsupervised}} = \left\{ \mathbf{x}_{\text{init}^(i)}, \mathbf{y}_{\text{obs}^(i)}\right\}
$$

In this scenario, we do not have

---
**Data** (Supervised)

The second case assumes that we do not have any realizations of the true state, we only have our prior and posterior.

$$
\mathcal{D}_{\text{supervised}} = \left\{ \mathbf{x}_{\text{init}^(i)},\; \mathbf{x}_{\text{gt}^(i)},\; \mathbf{x}_{\text{obs}^(i)}\right\}
$$

**Note**: This is very useful in the case of pre-training a model on emulated data.

---




**Minimization Problem**

$$
\boldsymbol{g}(\mathbf{x}_\text{init}, \mathbf{y}_\text{obs}, \mathbf{x}_\text{gt}; \boldsymbol{\theta}) = \boldsymbol{f}(\mathbf{x}_\text{init}, \mathbf{y}_\text{obs}; \boldsymbol{\theta}) - \mathbf{x}_\text{gt}
$$


---
**Minimization Strategy**

We are interested in finding the fixed-point solution such that

$$
\mathbf{x}^{(k+1)} = \boldsymbol{f}(\mathbf{x}^{(k)}, \mathbf{y}_{\text{obs}} )
$$


In this task, we are looking to minimize the above function wrt to the inputs, $\mathbf{x}_\text{init}$, but also with the best parameters, $\boldsymbol{\theta}$.

$$
\mathbf{x}^*(\boldsymbol{\theta}) = \argmin_{\mathbf{x}} \; \boldsymbol{g}(\mathbf{x}_\text{init}, \mathbf{y}_\text{obs}, \mathbf{x}_\text{gt}; \boldsymbol{\theta})
$$

---

**Loss Function** (Generic)

Given $N$ samples of data pairs, $\mathcal{D} = \left\{ \mathbf{x}_{\text{init}^{(i)}}, \mathbf{y}_{\text{obs}^{(i)}}\right\}_{i=1}^N$, we have a energy function, $\boldsymbol{U}$ which represents the generic inverse problem.

$$
\boldsymbol{U}(\mathbf{x}_{\text{init}^{(i)}}, \mathbf{y}_{\text{obs}^{(i)}}) = \mathcal{L}_\text{Data}(\mathbf{x}_{\text{init}^{(i)}}, \mathbf{y}_{\text{obs}^{(i)}}) + \lambda \mathcal{R}(\mathbf{x}_{\text{init}^{(i)}})
$$

**Loss Function** (4D Variational)

---
## Problem Setting


<!-- $$
\mathcal{L}(\theta) = \argmin_\theta \lambda_1 || \mathbf{x} - \mathbf{y}||_{\Omega}^2 + \lambda_2 ||\mathbf{x} - \boldsymbol{\phi}(\mathbf{x})||_2^2
$$

where $\theta = \{ x,\}$ -->

Let's take some observations, $\mathbf{y}$, which are sparse and incomplete. We are interested in finding a state, $\mathbf{x}$, that best matches the observations **and** fills in the missing observations.

This is a learning problem of the reconstruction error for the observed data. It is given by:

$$
\theta^* = \argmin_{\theta}\sum_i^N || \mathbf{y}_i - \boldsymbol{I}(\boldsymbol{U}(\mathbf{x};\theta), \mathbf{y}_i, \Omega_i)||_{\Omega_i}^2
$$

where:
* $||\cdot||_\Omega^2$ - L2 norm evaluated on the subdomain $\Omega$.
* $\boldsymbol{U}(\cdot)$ - the energy function
* $\boldsymbol{I}(\cdot)$ - the solution to the interpolation problem


#### Interpolation Problem, $\boldsymbol{I}$

This tries to solve the interpolation problem of finding the best state, $\mathbf{x}$, given the observations, $\mathbf{y}$, on a subdomain, $\boldsymbol{\Omega}$. This involves minimizing some energy function, $\boldsymbol{U}$.


$$
\mathbf{x}^* = \argmin_{\mathbf{x}} \boldsymbol{U}(\mathbf{x},\mathbf{y},\boldsymbol{\theta}, \boldsymbol{\Omega}) \coloneqq \boldsymbol{I}()
$$

They use a fixed point method to solve this problem.

---
## Domain

The first term in the loss function is the observation term defined as:

$$
||\cdot ||_{\Omega}^2
$$(loss_domain)

This is the evaluation of the quadratic norm restricted to the domain, $\Omega$.

**Pseudo-Code**

```python

# do some computation
x = ... 

# fill the solution with observations
x[mask] = y[mask]

```

---
## Operators

### ODE/PDE


### Constrained

$$
\boldsymbol{\psi}(\mathbf{x};\boldsymbol{\theta}) = 
$$


###


---

## Optimization


### Fixed Point Algorithm

We are interested in minimizing this function.

$$
\mathbf{x}^* = \argmin_{\mathbf{x}} \boldsymbol{U}(\mathbf{x},\mathbf{y}, \Omega, \boldsymbol{\theta}) \text{  s.t. } \mathbf{y}_{} =
$$



---
**Psuedo-Code**

```python
# initialize x
x = x_init

# loop through number of iterations
for k in range(n_iterations):

    # update sigma point method.
    x = fn(x)

    # update via known observations
    x[mask] = y[mask]
```


---
#### Project-Based Iterative Update

* DINEOF, Alvera-Azcarate et. al. (2016)
* DINCAE, Barth et. al. (2020)

**Projection**

We will use our function, $\boldsymbol{\phi}$, to map the data one iteration, $k$.

$$
\tilde{\mathbf{x}}^{(k+1)} = \boldsymbol{\psi}(\mathbf{x}^{(k)}; \boldsymbol{\theta})
$$

**Update Observed Domain**

We will update the true state, $\mathbf{x}$, where we have observations, $\mathbf{y}$. This is given by the $\Omega$ function (i.e. a mask).

$$
\mathbf{x}^{(k+1)}(\boldsymbol{\Omega}) = \mathbf{y}(\boldsymbol{\Omega}) 
$$

**Update Unobserved Domain**

$$
\mathbf{x}^{(k+1)}(\bar{\boldsymbol \Omega}) = \tilde{\mathbf{x}}^{(k+1)}(\bar{\boldsymbol \Omega})
$$

---
**Pseudo-code**


```python
def update(x: Array, y: Array, mask: Array, params: pytree, phi_fn: Callable):

    x = mask * y + phi_fn(x, params) * (1 - mask)

    return x
```


---
#### Gradient-Based Iterative Update

Let $\boldsymbol{U}(\mathbf{x}, \mathbf{y},\boldsymbol{\Omega},\boldsymbol{\theta}) : \mathbb{R}^D \rightarrow \mathbb{R}$ be the energy function.



**Gradient Step**

$$
\tilde{\mathbf{x}}^{(k+1)} = \mathbf{x}^{(k)} - \lambda \boldsymbol{\nabla}_{\mathbf{x}^{(k)}}\boldsymbol{U}(\mathbf{x}^{(k)}, \mathbf{y}, \boldsymbol{\Omega}, \boldsymbol{\theta})
$$

**Update Observed Domain**

$$
\mathbf{x}^{(k+1)}(\boldsymbol{\Omega}) = \mathbf{y}(\boldsymbol{\Omega}) 
$$

**Update Unobserved Domain**

$$
\mathbf{x}^{(k+1)}(\bar{\boldsymbol \Omega}) = \tilde{\mathbf{x}}^{(k+1)}(\bar{\boldsymbol \Omega})
$$



---
**Pseudo-Code**

```python
def energy_fn(
    x, Array[Batch, Dims], 
    y: Array[Batch, Dims], 
    mask: Array[Batch, Dims], 
    params: pytree,
    alpha_prior: float=0.01,
    alpha_obs: float=0.99
    ) -> float:

    loss_obs = np.mean(mask * (x - y) ** 2)

    loss_prior = np.mean((phi(x, params) - x))

    total_loss = alpha_obs * loss_obs + alpha_prior * loss_prior

    return total_loss
```

```python
def update(
    x: Array[Batch, Dims], 
    y: Array[Batch, Dims], 
    mask: Array[Batch, Dims], 
    params: pytree, 
    energy_fn: Callable,
    alpha: float
    ) -> Array[Batch, Dims]:
    

    x = x - alpha * jax.grad(energy_fn)(x, y, mask, params)

    return x
```

---
#### NN-Interpolator Iterative Update

Let $\boldsymbol{NN}(\mathbf{x}, \mathbf{y};\boldsymbol{\theta})$ be an arbitrary NN function.

**Projection**

$$
\dot{\mathbf{x}}^{(k)} = \lambda \boldsymbol{\nabla}_{\mathbf{x}^{(k)}}\boldsymbol{U}(\mathbf{x}^{(k)}, \mathbf{y}, \boldsymbol{\Omega}, \boldsymbol{\theta})
$$



**Gradient Step**

$$
\tilde{\mathbf{x}}^{(k)} = \mathbf{x}^{(k)} - \boldsymbol{NN} \left( \mathbf{x}^{(k)},  \dot{\mathbf{x}}^{(k)}; \boldsymbol{\theta}\right)
$$

**Update Observed Domain**

$$
\mathbf{x}^{(k+1)}(\boldsymbol{\Omega}) = \mathbf{y}(\boldsymbol{\Omega}) 
$$

**Update Unobserved Domain**

$$
\mathbf{x}^{(k+1)}(\bar{\boldsymbol \Omega}) = \tilde{\mathbf{x}}^{(k+1)}(\bar{\boldsymbol \Omega})
$$

Example:

$$
\boldsymbol{NN} \left( \mathbf{x}^{(k); \boldsymbol{\theta}},  \tilde{\mathbf{x}}^{(k+1)}\right) = \tilde{\boldsymbol{NN}} \left( \mathbf{x}^{(k)} - \tilde{\mathbf{x}}^{(k+1)}; \boldsymbol{\theta}\right)
$$

---
##### LSTM

**Pseudo-Code**


```python
def update(
    x: Array[Batch, Dims], 
    y: Array[Batch, Dims], 
    mask: Array[Batch, Dims], 
    params: pytree, 
    hidden_params: Tuple[Array[Dims]],
    alpha: float,
    energy_fn: Callable,
    rnn_fn: Callable,
    activation_fn: Callable
    ) -> Array[Batch, Dims]:

    # gradient - variational cost
    x_g = alpha * jax.grad(energy_fn)(x, y, mask, params)

    # NN Gradient update
    g = rnn_fn(x_g, hidden_params)

    x = x - activation_fn(g)

    return x
```

##### CNN

```python

activation_fn = lambda x: tanh(x)


def update(
    x: Array[Batch, Dims], 
    y: Array[Batch, Dims], 
    mask: Array[Batch, Dims], 
    x_g: Array[Batch, Dims],
    params: pytree, 
    alpha: float,
    energy_fn: Callable,
    nn_fn: Callable,
    activation_fn: Callable = lambda x: tanh(x)
    ) -> Array[Batch, Dims]:

    # gradient - variational cost
    x_g_new = alpha * jax.grad(energy_fn)(x, y, mask, params)

    # NN Gradient update
    x_g = jnp.concatenate([x_g_new, x_g])

    g = nn_fn(x_g)

    x = x - activation_fn(g)

    return x

```

---
## Stochastic Transformation


#### Deterministic

$$
\mathbf{x}^{(k+1)} = \boldsymbol{f}(\mathbf{x}^{(k)}, \mathbf{y}_\text{obs})
$$

**Loss**

$$
\mathcal{L} = ||\mathbf{x} - \boldsymbol{f}(\mathbf{x},\mathbf{y})||_2^2
$$

#### Probabilistic

$$
p(\mathbf{x}^{(k+1)}|\mathbf{x}^{(k)}) = \mathcal{N}(\mathbf{x}^{(k)}| \boldsymbol{\mu}_{\boldsymbol \theta}(\mathbf{x}^{(k)}), \boldsymbol{\sigma}^2_{\boldsymbol \theta}(\mathbf{x}))
$$

**Loss**

$$
-\log p(y|\mathbf{x}) = \frac{1}{2}\log \boldsymbol{\sigma}^2_{\boldsymbol \theta}(\mathbf{x}) + \frac{(y - \boldsymbol{\mu}_{\boldsymbol \theta}(\mathbf{x}))}{2\boldsymbol{\sigma}^2_{\boldsymbol \theta}(\mathbf{x})} + \text{constant}
$$

where:

* $y \in \mathbb{R}$
* $\mathbf{x} \in \mathbb{R}^{D}$

---
$$
-\log p(\mathbf{y}|\mathbf{x}) = \frac{1}{2} \log|\det \boldsymbol{\Sigma}_{\boldsymbol \theta}(\mathbf{x}) |  + ||\mathbf{y} - \boldsymbol{\mu}_{\boldsymbol \theta}(\mathbf{x})||^2_{\boldsymbol{\Sigma}_{\boldsymbol \theta}(\mathbf{x})} + \text{constant}
$$

where:

* $\mathbf{y} \in \mathbb{R}^{D_y}$
* $\mathbf{x} \in \mathbb{R}^{D_x}$
* $\boldsymbol{\mu}_{\boldsymbol \theta}:\mathbb{R}^{D_x} \rightarrow \mathbb{R}^{D_y}$
* $\boldsymbol{\Sigma}_{\boldsymbol \theta}:\mathbb{R}^{D_x} \rightarrow \mathbb{R}^{D_y \times D_y}$


---
## Other Perspectives

### Explicit vs Implicit Model

#### Explicit Model

$$
\mathbf{x}_\text{gt} = \boldsymbol{f}(\mathbf{x}_\text{init})
$$

**Penalize the Loss**

$$
\mathcal{L} = \mathcal{L}_\text{model}(\mathbf{x}_\text{gt}, \tilde{\mathbf{x}}_\text{init}) + \mathcal{L}_\text{data}(\mathbf{x}_\text{gt}(\boldsymbol{\Omega}), \mathbf{y}_\text{obs}(\boldsymbol{\Omega}))
$$

#### Conditional Explicit Model

We assume some function maps both the observations and the initial condition to the solution.

$$
\mathbf{x}_\text{gt} = \boldsymbol{f}(\mathbf{x}_\text{init}, \mathbf{y}_\text{obs})
$$

The difference here is that we do not add any extra terms into the loss because it is explicit.

#### Implicit Model

We assume
$$
\begin{aligned}
\mathbf{x}_\text{gt}\boldsymbol{g}(\mathbf{x}_\text{init}, \mathbf{y}_\text{obs}) = 0
\end{aligned}
$$

where:

$$
\boldsymbol{g}(\mathbf{x}_\text{init}, \mathbf{y}_\text{obs}) = \boldsymbol{f}(\mathbf{x}_\text{init}, \mathbf{y}_\text{obs}) - \mathbf{x}_\text{init}
$$

This is the same as a fixed-point solution.

$$
\begin{aligned}
\mathbf{x}_\text{gt} &= \mathbf{x}_\text{init}\\
\mathbf{x}_\text{gt} &= \boldsymbol{f}(\mathbf{x}_\text{gt}, \mathbf{y}_\text{obs})
\end{aligned}
$$

---
## Literature

* Learning Latent Dynamics for Partially Observed Chaotic Systems - []() 
* Joint Interpolation and Representation Learning for Irregularly Sampled Satellite-Derived Geophysics - []()
* Learning Variational Data Assimilation Models and Solvers - []()
* Intercomparison of Data-Driven and Learning-Based Interpolations of Along-Track Nadir and Wide-Swath SWOT Altimetry Observations - []()
* Variational Deep Learning for the Identification and Reconstruction of Chaotic and Stochastic Dynamical Systems from Noisy and Partial Observations - Nguyen et. al. (2020) - [Paper](https://arxiv.org/abs/2009.02296) | [Code](https://github.com/CIA-Oceanix/DAODEN)