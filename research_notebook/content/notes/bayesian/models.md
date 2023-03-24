# Models


**Model**

$$
\mathbf{y} = f(\mathbf{x}; \boldsymbol{\theta}) + \boldsymbol{\epsilon}
$$

**Measurement Model**

$$
p(\mathbf{y}|\mathbf{x}; \boldsymbol{\theta}) \sim \mathcal{N}(\mathbf{y}|\boldsymbol{f}(\mathbf{x};\boldsymbol{\theta}), \sigma^2)
$$

**Likelihood Loss Function**

$$
\log p(\mathbf{y}|\mathbf{x}; \boldsymbol{\theta})
$$


**Loss Function**

$$
\mathcal{L}(\boldsymbol{\theta}) = - \frac{1}{2\sigma^2}||\mathbf{y} - f(\mathbf{x};\boldsymbol{\theta})||_2^2 - \log p(\mathbf{x};\boldsymbol{\theta})
$$

---
## Error Minimization

We choose an error measure or loss function, $\mathcal{L}$, to minimize wrt the parameters, $\theta$.

$$
\mathcal{L}(\theta) = \sum_{i=1}^N \left[ f(x_i;\theta) - y   \right]^2
$$

We typically add some sort of regularization in order to constrain the solution

$$
\mathcal{L}(\theta) = \sum_{i=1}^N \left[ f(x_i;\theta) - y   \right]^2 + \lambda \mathcal{R}(\theta)
$$

---
## Probabilistic Approach

We explicitly account for noise in our model.

$$
y = f(x;\theta) + \epsilon(x)
$$

where $\epsilon$ is the noise. The simplest noise assumption we see in many approaches is the iid Gaussian noise.

$$
\epsilon(x) \sim \mathcal{N}(0, \sigma^2)
$$

So given our standard Bayesian formulation for the posterior

$$
p(\theta|\mathcal{D}) \propto p(y|x,\theta)p(\theta)
$$

we assume a Gaussian observation model

$$
p(y|x,\theta) = \mathcal{N}(y;f(x;\theta), \sigma^2)
$$

and in turn a likelihood model

$$
p(y|x;\theta) = \prod_{i=1}^N \mathcal{N}(y_i; f(x_i; \theta), \sigma^2)
$$

**Objective**: maximize the likelihood of the data, $\mathcal{D}$ wrt the parameters, $\theta$.

**Note**: For a Gaussian noise model (what we have assumed above), this approach will use the same predictions as the MSE loss function (that we saw above).

$$
\log p(y|x,\theta) \propto - \frac{1}{2\sigma^2}\sum_{i=1}^N \left[ y_i - f(x_i;\theta)\right]
$$

We can simplify the notion a bit to make it more compact. This essentially puts all of the observations together so that we can use vectorized representations, i.e. $\mathcal{D} = \{ x_i, y_i\}_{i=1}^N$

$$
\begin{aligned}
\log p(\mathbf{y}|\mathbf{x},\theta)
&= - \frac{1}{2\sigma^2} \left(\mathbf{y} - \boldsymbol{f}(\mathbf{x};\theta)\right)^\top\left(\mathbf{y} - \boldsymbol{f}(\mathbf{x};\theta) \right) \\
&= - \frac{1}{2\sigma^2} ||\mathbf{y} - \boldsymbol{f}(\mathbf{x};\theta)||_2^2
\end{aligned}
$$

where $||\cdot ||_2^2$ is the Maholanobis Distance.


**Note**: we often see this notation in many papers and books.

#### Priors



---

## Different Parameterizations


|   Model    |  Equation |
|:--------|:---------|
|  Identity  | $ \mathbf{x}$ |
|   Linear   | $\mathbf{wx}+\mathbf{b}$ |
|   Basis    |  $\mathbf{w}\boldsymbol{\phi}(\mathbf{x}) + \mathbf{b}$  |
| Non-Linear | $\sigma\left( \mathbf{wx} + \mathbf{b}\right)$ |
| Neural Network | $\boldsymbol{f}_{L}\circ \boldsymbol{f}_{L-1}\circ\ldots\circ\boldsymbol{f}_1$ |
| Functional | $\boldsymbol{f} \sim \mathcal{GP}\left(\boldsymbol{\mu}_{\boldsymbol \alpha}(\mathbf{x}),\boldsymbol{\sigma}^2_{\boldsymbol \alpha}(\mathbf{x})\right)$ |

---
#### Identity


$$
f(x;\theta) = x
$$

$$
p(y|x,\theta) \sim \mathcal{N}(y|x, \sigma^2)
$$


$$
\mathcal{L}(\theta) = - \frac{1}{2\sigma^2}||y - x||_2^2
$$

---
#### Linear

A linear function of $\mathbf{w}$ wrt $\mathbf{x}$.

$$
f(x;\theta) = w^\top x
$$

$$
p(y|x,\theta) \sim \mathcal{N}(y|w^\top x, \sigma^2)
$$

$$
\mathcal{L}(\theta) = - \frac{1}{2\sigma^2}|| y - w^\top x||_2^2
$$

---
#### Basis Function

A linear function of $\mathbf{w}$ wrt to the basis function $\phi(x)$.

$$
f(x;\theta) = w^\top \phi(x;\theta)
$$

**Examples**

* $\phi(x) = (1, x, x^2, \ldots)$
* $\phi(x) = \tanh(x + \gamma)^\alpha$
* $\phi(x) = \exp(- \gamma||x-y||_2^2)$
* $\phi(x) = \left[\sin(2\pi\boldsymbol{\omega}\mathbf{x}),\cos(2\pi\boldsymbol{\omega}\mathbf{x}) \right]^\top$

**Prob Formulation**

$$
p(y|x,\theta) \sim \mathcal{N}(y|w^\top \phi(x), \sigma^2)
$$


**Likelihood Loss**

$$
\mathcal{L}(\theta) = - \frac{1}{2 \sigma^2} ||y - w^\top \phi(x; \theta) ||_2^2
$$


---
#### Non-Linear Function

A non-linear function in $\mathbf{x}$ and $\mathbf{w}$.

$$
f(x; \theta) = g (w^\top \phi (x; \theta_{\phi}))
$$

**Examples**

* Random Forests
* Neural Networks
* Gradient Boosting

**Prob Formulation**

$$
p(y|x,\theta) \sim \mathcal{N}(y|w^\top \phi(x), \sigma^2)
$$


**Likelihood Loss**

$$
\mathcal{L}(\theta) = - \frac{1}{2\sigma^2}||y - g(w^\top \phi(x))||_2^2
$$

---
#### Generic

A non-linear function in $\mathbf{x}$ and $\mathbf{w}$.

$$
y = f(x; \theta)
$$

**Examples**

* Random Forests
* Neural Networks
* Gradient Boosting

**Prob Formulation**

$$
p(y|x,\theta) \sim \mathcal{N}(y|f(x; \theta), \sigma^2)
$$


**Likelihood Loss**

$$
\mathcal{L}(\theta) = - \frac{1}{2\sigma^2}||y - f(x; \theta)||_2^2
$$

---
#### Generic (Heteroscedastic)

A non-linear function in $\mathbf{x}$ and $\mathbf{w}$.

$$
y = \boldsymbol{\mu}(x; \theta) + \boldsymbol{\sigma}^2(x;\theta)
$$

**Examples**

* Random Forests
* Neural Networks
* Gradient Boosting

**Prob Formulation**

$$
p(y|x,\theta) \sim \mathcal{N}(y|\boldsymbol{\mu}(x; \theta), \boldsymbol{\sigma}^2(x; \theta))
$$


**Likelihood Loss**

$$
-\log p(y|x,\theta) = \frac{1}{2}\log \boldsymbol{\sigma}^2(\mathbf{x};\boldsymbol{\theta}) + \frac{1}{2}||\mathbf{y} - \boldsymbol{\mu}(\mathbf{x};\boldsymbol{\theta})||^2_{\boldsymbol{\sigma}^2(\mathbf{x};\boldsymbol{\theta})} + \text{C}
$$
