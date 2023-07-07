# State Estimation


$$
p(\mathbf{u}|\mathbf{y}) \propto p(\mathbf{y}|\mathbf{u})p(\mathbf{u}) = \exp\left( - \boldsymbol{J}(\mathbf{u}) \right)
$$

## Likelihood (Data Fidelity)

This term models the mapping between the state space and the observation space.

$$
\boldsymbol{L}(\mathbf{u}) = - \log \nu \left(\mathbf{y}_{obs} - \boldsymbol{H}(\mathbf{u})\right)
$$ (eq:inv_prob_likelihood)

## Prior (Regularization)

This term models the *shape* of the state.

$$
\boldsymbol{R}(\mathbf{u}) = - \log p(\mathbf{u})
$$ (eq:inv_prob_prior)

## Objective

$$
\boldsymbol{J}(\mathbf{u}) = \boldsymbol{L}(\mathbf{u}) + \boldsymbol{R}(\mathbf{u})
$$ (eq:inv_prob_objective)


## Posterior

This is the term we are actually interested in learning.

$$
p(\mathbf{u}|\mathbf{y}) = \frac{1}{Z}\nu
\left(\mathbf{y}_{obs} -
\boldsymbol{H}(\mathbf{u})\right)
p(\mathbf{u}) \propto \exp \left(- \boldsymbol{J}(\mathbf{u}) \right)
$$ (eq:inv_prob_posterior)


## Minimization Problem

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
## Example Priors

Below we will take a look at some priors that we can place on the

| Prior |  Equation|
|:------|:-------|
| Gaussian | $\|\|\boldsymbol{u}\|\|_2^2$ |
| Laplacian | $\|\|\boldsymbol{u}\|\|_1$ |
| Gradient-Based Prior | $\|\|\boldsymbol{\nabla u}\|\|_2^2$ |
| Transform Distribution | $p(\boldsymbol{u})=p(\boldsymbol{z})\|\det\boldsymbol{\nabla_u T_\theta}^{-1}(\boldsymbol{u})$ |
| Neural Field | $\|\|\boldsymbol{u} - \boldsymbol{u_\theta}\|\|_2^2$ |
| Partial Differential Equation | $\|\|\boldsymbol{u} - (\partial_t\boldsymbol{u} - \mathcal{N}[\boldsymbol{u};\theta])\|\|_2^2$ |

---

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

---

### Laplace Assumption

We can also make an assumption of a Laplacian with a constant covariance.
This will promote sparse solutions.
This gives us the distribution

$$
p(\mathbf{u}) = \text{Laplace}(\mathbf{0},\lambda^{-1}\mathbf{I}) \propto
\exp \left( - \frac{\lambda}{2} || \mathbf{u}||_1 \right)
$$

which results in the

$$
\boldsymbol{R}(\mathbf{u}) = \frac{\lambda}{2} || \mathbf{u}||_1
$$

**Note**: This is equivalent to the L$_1$-regularizer.

---

### Transform Distribution

We may not have the most expressive prior with just a single set of parameters.
For example, perhaps the prior field is too complex to be captured by a mixture of Gaussians.
In this case, we can use some parameterized transformation, $\boldsymbol{T_\theta}$, which maps the state to a latent vector, $\boldsymbol{z}$, in a transform domain whereby the distribution is much simpler.
For example, we could have the following relationship:

$$
\begin{aligned}
\boldsymbol{z} &\sim P_\theta \\
\boldsymbol{u} &= \boldsymbol{T_\theta}(\boldsymbol{z})
\end{aligned}
$$

If we assume that the transformation is a bijection, then we can calculate the likelihood of the state, $\boldsymbol{u}$, by using the change of variables formula resulting in:

$$
p_{\boldsymbol\theta}(\boldsymbol{u}) = p_{\boldsymbol\theta}(\boldsymbol{z})|\det\boldsymbol{\nabla_u T_\theta}^{-1}(\boldsymbol{u})|
$$

So, we can plug this into equation {eq}`eq:inv_prob_prior` for the prior distribution to give

$$
\begin{aligned}
\boldsymbol{R}(\boldsymbol{u};\boldsymbol{\theta})
&= -\log p(\boldsymbol{u}) \\
&= - \log p_{\boldsymbol\theta}(\boldsymbol{z}) - \log |\det\boldsymbol{\nabla_u T_\theta}^{-1}(\boldsymbol{u})|
\end{aligned}
$$

So now, we need

**Note**: in the above example we use a bijective transformation. However, we can use surjective or stochastic transformations which could be more appropriate for the dataset.



---

## Example Likelihood

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
