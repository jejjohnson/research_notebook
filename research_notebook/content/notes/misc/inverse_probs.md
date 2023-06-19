# Inverse Problems

## Problem Definition

$$
\boldsymbol{y}_{obs}(\vec{\mathbf{x}},t) =
\boldsymbol{H}\left[ \mathbf{u}; \boldsymbol{\theta} \right]
(\vec{\mathbf{x}},t)
+ \boldsymbol{\varepsilon}
$$ (eq:inv_prob_continuous)

**Discretized**

$$
\boldsymbol{y}_{obs}(\boldsymbol{\Omega}_{obs}(\tau_{obs}),\tau_{obs}) =
\boldsymbol{H}
\left[ \mathbf{u}; \boldsymbol{\theta} \right]
\left(\boldsymbol{\Omega}_{state}(\tau_{state}),\tau_{state}\right)
+ \boldsymbol{\varepsilon}
$$ (eq:inv_prob_discretized)


---

## Terms

### Likelihood (Data Fidelity)

This term models the mapping between the state space and the observation space.

$$
\boldsymbol{L}(\mathbf{u}) = - \log \nu \left(\mathbf{y}_{obs} - \boldsymbol{H}(\mathbf{u})\right)
$$ (eq:inv_prob_likelihood)

### Prior (Regularization)

This term models the *shape* of the state.

$$
\boldsymbol{R}(\mathbf{u}) = - \log p(\mathbf{u})
$$ (eq:inv_prob_prior)

### Objective

$$
\boldsymbol{J}(\mathbf{u}) = \boldsymbol{L}(\mathbf{u}) + \boldsymbol{R}(\mathbf{u})
$$ (eq:inv_prob_objective)


### Posterior

This is the term we are actually interested in learning.

$$
p(\mathbf{u}|\mathbf{y}) = \frac{1}{Z}\nu
\left(\mathbf{y}_{obs} -
\boldsymbol{H}(\mathbf{u})\right)
p(\mathbf{u}) \propto \exp \left(- \boldsymbol{J}(\mathbf{u}) \right)
$$ (eq:inv_prob_posterior)


### Minimization Problem

$$
\begin{aligned}
\mathbf{u}^* &= \underset{\mathbf{u}}{\text{argmax}} \hspace{2mm}
p(\mathbf{u}|\mathbf{y}_{obs}) \\
&= \underset{\mathbf{u}}{\text{argmax}} \hspace{2mm}
p(\mathbf{y}_{obs}|\mathbf{u})p(\mathbf{u}) \\
&= \underset{\mathbf{u}}{\text{argmax}} \hspace{2mm}
\boldsymbol{J}(\mathbf{u})
\end{aligned}
$$

**Note**: This means that maximizing the objective function, $\boldsymbol{J}(\cdot)$, is equivalent to maximizing the posterior, $p(u|y)$.


---

## Likelihood Options

The likelihood encodes the data type we expect to observe.
For example, if we are dealing with continuous data, then we need to encode this.
If we're dealing with discrete data, the we need to encode this.
Another example is if we only expect positive values, then we need to encode this.


---

#### IID Gaussian Obs. Noise (Constant)

We define the noise term as:

$$
\boldsymbol{\varepsilon} = \mathcal{N}(0,\sigma^2\mathbf{I})
$$

This means that the likelihood distribution will be

$$
\begin{aligned}
p(\mathbf{y}_{obs}|\mathbf{u})
&= \nu\left( \mathbf{y}_{obs} - \boldsymbol{H}(\mathbf{u}) \right) \\
&= \mathcal{N}(\mathbf{y}_{obs}|\boldsymbol{H}(\mathbf{u}), \sigma^2\mathbf{I}) \\
&\propto \exp \left( - \frac{1}{2\sigma^2}||\mathbf{y}_{obs} - \boldsymbol{H}(\mathbf{u}) ||^2_2\right)
\end{aligned}
$$

Therefore the likelihood term will be

$$
\boldsymbol{L}(\mathbf{u}) = - \frac{1}{2\sigma^2}||\mathbf{y}_{obs} - \boldsymbol{H}(\mathbf{u}) ||^2_2
$$

---

We have some other standard options that we can do:


**Full Covariance**: $\mathcal{N}(0,\mathbf{\Sigma})$

This would result in the likelihood term:

$$
\boldsymbol{L}(\mathbf{u}) = - \frac{1}{2\sigma^2}||\mathbf{y}_{obs} - \boldsymbol{H}(\mathbf{u}) ||^2_{\mathbf{\Sigma}}
$$

**T-Student**:


**Cauchy**:

**Heterogeneous**:


---

## Priors

Most of the time is spent here!


### Gaussian Assumption

We can use the canonical Gaussian assumption. Let's use the *full Gaussian* assumption with a mean vector and full covariance.
This will promote smooth solutions.
This gives us the distribution as:

$$
p(\mathbf{u}) = \mathcal{N}(\boldsymbol{\mu},\mathbf{\Sigma}) \propto
\exp \left( - \frac{1}{2} || \mathbf{u} - \boldsymbol{\mu} ||_\mathbf{\Sigma}^2 \right)
$$

which results in the

$$
\boldsymbol{R}(\mathbf{u}) = \frac{1}{2} ||\mathbf{u} - \boldsymbol{\mu}||_\mathbf{\Sigma}^2
$$

We can also make an assumption of a Gaussian with a constant covariance.
This gives us the distribution

$$
p(\mathbf{u}) = \mathcal{N}(\mathbf{0},\lambda^{-1}\mathbf{I}) \propto
\exp \left( - \frac{\lambda}{2} || \mathbf{u}||_2^2 \right)
$$

which results in the

$$
\boldsymbol{R}(\mathbf{u}) = \frac{\lambda}{2} || \mathbf{u}||_2^2
$$

**Note**: This is equivalent to the L$_2$-regularizer.

### Laplace Assumption

We can also make an assumption of a Laplacian with a constant covariance.
This will promote sparse solutions.
This gives us the distribution

$$
p(\mathbf{u}) = \mathcal{L}(\mathbf{0},\lambda^{-1}\mathbf{I}) \propto
\exp \left( - \frac{\lambda}{2} || \mathbf{u}||_1 \right)
$$

which results in the

$$
\boldsymbol{R}(\mathbf{u}) = \frac{\lambda}{2} || \mathbf{u}||_1
$$

**Note**: This is equivalent to the L$_1$-regularizer.

---

## Minimization Problem


---

### State Estimation

$$
p(\mathbf{u}|\mathbf{y}_{obs}) = \frac{p(\mathbf{y}_{obs}|\mathbf{u})p(\mathbf{u})}{p(\mathbf{y}_{obs})}
$$

So the name of the game is to estimate the state, $\mathbf{u}$, given some objective function which may depend upon some external parameters, $\boldsymbol{\theta}$.


$$
\begin{aligned}
\mathbf{u}^* &=
\underset{\mathbf{u}}{\text{argmax}} \hspace{2mm}
\mathcal{J}(\mathbf{u};\boldsymbol{\theta})
\end{aligned}
$$

---

### Parameter Estimation

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

### Bi-Level Optimization


$$
\begin{aligned}
\boldsymbol{\theta}^* &=
\underset{\boldsymbol{\theta}}{\text{argmax}} \hspace{2mm}
\mathcal{L}(\boldsymbol{\theta}) \\
\mathbf{u}^*(\boldsymbol{\theta}) &=
\underset{\mathbf{u}}{\text{argmax}} \hspace{2mm}
\mathcal{J}(\mathbf{u};\boldsymbol{\theta})
\end{aligned}
$$
