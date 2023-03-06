# Inference Schemes



[**Source**](https://www.cs.ubc.ca/~schmidtm/MLRG/GaussianProcesses.pdf) | Deisenroth - [Sampling](https://drive.google.com/file/d/1Ryb1zDzndnv1kOe8nT0Iu4OD6m0KC8ry/view)

**Advances in VI** - [Notebook](https://github.com/magister-informatica-uach/INFO320/blob/master/6_advances_in_VI.ipynb)

* Numerical Integration (low dimension)
* Bayesian Quadrature
* Expectation Propagation
* Conjugate Priors (Gaussian Likelihood w/ GP Prior)
* Subset Methods (Nystrom)
* Fast Linear Algebra (Krylov, Fast Transforms, KD-Trees)
* Variational Methods (Laplace, Mean-Field, Expectation Propagation)
* Monte Carlo Methods (Gibbs, Metropolis-Hashings, Particle Filter)


**Local Methods**


**Sampling Methods**


---
## Local Methods


#### Mean Squared Error (MSE)

In the case of regression, we can use the MSE as a loss function. This will exactly solve for the negative log-likelihood term above.


````{admonition} Proof
:class: info dropdown

The likelihood of our model is:

$$\log p(y|\mathbf{X,w}) = \sum_{i=1}^N \log p(y_i|x_i,\theta)$$

And for simplicity, we assume the noise $\epsilon$ comes from a Gaussian distribution and that it is constant. So we can rewrite our likelihood as

$$\log p(y|\mathbf{X,w}) = \sum_{i=1}^N \log \mathcal{N}(y_i | \mathbf{x}_i\mathbf{w}, \sigma^2)$$

Plugging in the full formula for the Gaussian distribution with some simplifications gives us:

$$
\log p(y|\mathbf{X,w}) =
\sum_{i=1}^N
\log \frac{1}{\sqrt{2 \pi \sigma_e^2}}
\exp\left( - \frac{(y_i - \mathbf{x}_i\mathbf{w})^2}{2\sigma_e^2} \right)
$$

We can use the log rule $\log ab = \log a + \log b$ to rewrite this expression to separate the constant term from the exponential. Also, $\log e^x = x$.

$$
\log p(y|\mathbf{X,w}) = - \frac{N}{2} \log 2 \pi \sigma_e^2 - \sum_{i=1}^N \frac{(y_i - \mathbf{x_iw})^2}{2\sigma_e^2}
$$

So, the first term is constant so that we can ignore that in our loss function. We can do the same for the denominator for the second term. Let's simplify it to make our life easier.

$$
\log p(y|\mathbf{X,w}) = - \sum_{i=1}^N (y_i - \mathbf{x}_i\mathbf{w})^2
$$

So we want to maximize this quantity: in other words, I want to find the parameter $\mathbf{w}$ s.t. this equation is maximum.

$$
\mathbf{w}_{MLE} = \operatorname*{argmin}_{\mathbf{w}} - \sum_{i=1}^N (y_i - \mathbf{x}_i\mathbf{w})^2
$$

We can rewrite this expression because the maximum of a negative quantity is the same as minimizing a positive quantity.

$$
\mathbf{w}_{MLE} = \operatorname*{argmin}_{\mathbf{w}} \frac{1}{N} \sum_{i=1}^N (y_i - \mathbf{x}_i\mathbf{w})^2
$$

This is the same as the MSE error expression; with the edition of a scalar value $1/N$.

$$
\begin{aligned}
\mathbf{w}_{MLE} &= \operatorname*{argmin}_{\mathbf{w}} \frac{1}{N} \sum_{i=1}^N (y_i - \mathbf{x}_i\mathbf{w})^2 \\
&= \operatorname*{argmin}_{\mathbf{w}} \text{MSE}
\end{aligned}
$$

**Note**: If we did not know $\sigma_y^2$ then we would have to optimize this as well.

````






**Sources**:

* [Intro to Quantitative Econ w. Python](https://python-intro.quantecon.org/mle.html)



---
#### Maximum A Priori (MAP)


##### Loss Function

$$
\boldsymbol{\theta}_{\text{MAP}} = \operatorname*{argmax}_{\boldsymbol{\theta}} - \frac{1}{N}\sum_n^N\log p\left(y_n|f(x_n; \theta)\right) + \log p(\theta)
$$(map_loss)

````{admonition} Proof
:class: dropdown info

$$
\boldsymbol{\theta}_{\text{MAP}} = \operatorname*{argmax}_{\boldsymbol{\theta}} \log p(\boldsymbol{\theta}|\mathcal{D})
$$

We can plug in the base Bayesian formulation

$$
\boldsymbol{\theta}_{\text{MAP}} = \operatorname*{argmax}_{\boldsymbol{\theta}} \log \left[ \frac{p(\mathcal{D}|\boldsymbol{\theta})p(\boldsymbol{\theta})}{p(\mathcal{D})} \right]
$$

We can expand this term using the log rules

$$
\theta_{map} = \operatorname*{argmax}_\theta \left[ \log p(D|\theta) + \log p(\theta) - \log p(D) \right]
$$

Notice that $\log p(D)$ is a constant as the distribution of the data won't change. It also does not depend on the parameters, $\theta$. So we can cancel that term out.

$$
\theta_{map} = \operatorname*{argmax}_\theta \left[ \log p(D|\theta) + \log p(\theta) \right]$$

We will change this problem into a minimization problem instead of maximization

$$
\theta_{map} = \operatorname*{argmin}_{\theta} - \log p(D|\theta)
$$

We cannot find the probability distribution of $p(D|\theta)$ irregardless of what it is conditioned on. So we need to take some sort of expectations over the entire data.

$$
\theta_{map} = \operatorname*{argmin}_{\theta} - \mathbb{E}_{\mathbf{x}\sim P_X} \left[ \log p(D|\theta)\right] + \log p(\theta)
$$

We can approximate this using Monte carlo samples. This is given by:

$$
\mathbb{E}_{x}[\log p(D|\theta)] \approx \frac{1}{N}\sum_n^N p(y_n | f(x_n; \theta))
$$

and we assume that with enough samples, we will capture the essence of our data.

$$
\theta_{map} = \operatorname*{argmin}_{\theta} - \frac{1}{N}\sum_n^N \log p(y_n| f(x_n;\theta))+ \log p(\theta)
$$


````



---

#### Maximum Likelihood Estimation (MLE)


##### Loss Function

$$
\theta_{map} = \operatorname*{argmin}_{\theta} - \frac{1}{N}\sum_n^N \log p(y_n| f(x_n;\theta))
$$


````{admonition} Proof
:class: info dropdown

This is straightforward to derive because we can pick up from the proof of the MAP loss function, eq:{eq}`map_loss`.

$$
\theta_{map} = \operatorname*{argmin}_{\theta} - \frac{1}{N}\sum_n^N \log p(y_n| f(x_n;\theta))+ \log p(\theta)
$$

In this case, we will assume a uniform prior on our parameters, $\theta$. This means that any parameter value would work to solve the problem. The uniform distribution has a constant probability of 1. As a result, the $\log$ of $p(\theta)=1$ is equal to 0. So we can simply remove the log prior on our parameters in the above equation.

$$
\theta_{map} = \operatorname*{argmin}_{\theta} - \frac{1}{N}\sum_n^N \log p(y_n| f(x_n;\theta))
$$

````




```{prf:remark}

You can get an intuition that this will lead to local minimum as there are many possible solutions that would minimize this equation. Or even worse, there are many possible local minimum that we could get stuck in when trying to optimize for this.

```

#### KL-Divergence (Forward)

$$
\text{D}_{\text{KL}}\left[ p_*(x) || p(x;\theta) \right] = \mathbb{E}_{x\sim p_*}\left[ \log \frac{p_*(x)}{p(x;\theta)}\right]
$$

This is the distance between the best distribution, $p_*(x)$, for the data and the parameterized version, $p(x;\theta)$.

There is an equivalence between the (Forward) KL-Divergence and the Maximum Likelihood Estimation. Maximizing the likelihood expresses it as maximizing the likelihood of the data given our estimated distribution. Whereas the KL-divergence is a distance measure between the parameterized distribution and the "true" or "best" distribution of the real data. They are equivalent formulations but the MLE equations shows how this is a proxy for fitting the "real" data distribution to the estimated distribution function.

````{admonition} Proof
:class: info dropdown

$$
\text{D}_{\text{KL}}\left[ p_*(x) || p(x;\theta) \right] = \mathbb{E}_{x\sim p_*}\left[ \log \frac{p_*(x)}{p(x;\theta)}\right]
$$

We can expand this term via logs

$$
\mathbb{E}_{x\sim p_*}\left[ \log \frac{p_*(x)}{p(x;\theta)}\right] = \mathbb{E}_{x\sim p_*}\left[ \log p_*(x) - \log p(x;\theta)  \right]
$$

The first expectation, $\mathbb{E}_{x\sim p_*}[p_*(x)]$, is the *entropy* term (i.e. the expected uncertainty in the data). This is a constant term because no matter how well we estimate this distribution via our parameterized representation, $p(x;\theta)$, this term will not change. So we can ignore this term in our loss function.

$$
\mathbb{E}_{x\sim p_*}\left[ \log \frac{p_*(x)}{p(x;\theta)}\right] = -\mathbb{E}_{x\sim p_*}\left[ \log p(x;\theta)\right]
$$

We can rewrite this in its integral form:

$$
-\mathbb{E}_{x\sim p_*}\left[ \log p(x;\theta)\right] = - \int \log p(x;\theta) p_*(x)dx
$$

We will assume that the data distribution is a delta function, $p_*(x) = \delta (x - x_i)$. This means that each data point is represented equally. If we plug that into our model, we see that it is


$$
-\int \log p(x;\theta) p_*(x)dx = - \int \log p(x;\theta) \delta (x - x_i)dx
$$

We will do the same approximation of the integral with samples from our delta distribution.

$$
-\int \log p(x;\theta) \delta (x - x_i)dx = - \frac{1}{N}\sum_n^N \log p(x_n;\theta)
$$

So we have:

$$
\text{D}_{\text{KL}}\left[ p_*(x) || p(x;\theta) \right] = - \frac{1}{N}\sum_n^N \log p(x_n;\theta) = \mathcal{L}_{NLL}(\theta)
$$

which exactly the function for the NLL Loss



````




---
#### Laplace Approximation


This is where we approximate the posterior with a Gaussian distribution $\mathcal{N}(\mu, A^{-1})$.

* $w=w_{map}$, finds a mode (local max) of $p(w|D)$
* $A = \nabla\nabla \log p(D|w) p(w)$ - very expensive calculation
* Only captures a single mode and discards the probability mass
  * similar to the KLD in one direction.


**Sources**

* [Modern Arts of Laplace Approximation](https://agustinus.kristia.de/techblog/2021/10/27/laplace/) - Agustinus - Blog

---
#### Variational Inference


**Definition**: We can find the best approximation within a given family w.r.t. KL-Divergence.
$$
\text{KLD}[q||p] = \int_w q(w) \log \frac{q(w)}{p(w|D)}dw
$$
Let $q(w)=\mathcal{N}(\mu, S)$ and then we minimize KLD$(q||p)$ to find the parameters $\mu, S$.

> "Approximate the posterior, not the model" - James Hensman.

We write out the marginal log-likelihood term for our observations, $y$.

$$
\log p(y;\theta) = \mathbb{E}_{x \sim p(x|y;\theta)}\left[  \log p(y|\theta) \right]
$$

We can expand this term using Bayes rule: $p(y) = p(x,y)p(x|y)$.

$$
\log p(y;\theta) = \mathbb{E}_{x \sim p(x|y;\theta)}\left[ \log \underbrace{p(x,y;\theta)}_{prior} - \log \underbrace{p(x|y;\theta)}_{posterior}\right]
$$

where $p(x,y;\theta)$ is the joint distribution function and $p(x|y;\theta)$ is the posterior distribution function.

We can use a variational distribution, $q(x|y;\phi)$ which will approximate the

$$
\log p(y;\theta) \geq \mathcal{L}_{ELBO}(\theta,\phi)
$$

where $\mathcal{L}_{ELBO}$ is the Evidence Lower Bound (ELBO) term. This serves as an upper bound to the true marginal likelihood.


$$
\mathcal{L}_{ELBO}(\theta,\phi) = \mathbb{E}_{q(x|y;\phi)}\left[ \log p(x,y;\theta) - \log q(x|y;\phi) \right]
$$

we can rewrite this to single out the expectations. This will result in two important quantities.

$$
\mathcal{L}_{ELBO}(\theta,\phi) = \underbrace{\mathbb{E}_{q(x|y;\phi)}\left[ \log p(x,y;\theta)\right]}_{\text{Reconstruction}} - \underbrace{\text{D}_{\text{KL}}\left[ \log q(x|y;\phi) || p(x;\theta)\right]}_{\text{Regularization}}
$$





---
## Sampling Methods

#### Monte Carlo

We can produce samples from the exact posterior by defining a specific Monte Carlo chain.

We actually do this in practice with NNs because of the stochastic training regimes. We modify the SGD algorithm to define a scalable MCMC sampler.

[Here](https://chi-feng.github.io/mcmc-demo/) is a visual demonstration of some popular MCMC samplers.



#### Hamiltonian Monte Carlo


#### Stochastic Langevin Dynamics
