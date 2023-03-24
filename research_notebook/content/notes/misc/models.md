# Models


**Model**

$$
\mathbf{y} = f(\mathbf{x}; \boldsymbol{\theta}) + \boldsymbol{\epsilon}
$$


---
## Error Minimization

We choose an error measure or loss function, $\mathcal{L}$, to minimize wrt the parameters, $\theta$.

$$
\mathcal{L}(\theta) = \sum_{i=1}^N \left[ f(x_i;\theta) - y   \right]^2
$$

We typically add some sort of regularization in order to constrain the solution

$$
\mathcal{L}(\theta) = \frac{1}{|\mathcal{D}|}\sum_{i=1}^N \mathcal{L_D} \left( \mathbf{x},\mathbf{y}\right)^2 + \lambda \mathcal{R}(\theta)
$$

<!-- $$
\mathcal{L}(\theta) = \sum_{i=1}^N \left[ f(x_i;\theta) - y   \right]^2 + \lambda \mathcal{R}(\theta)
$$ -->


---

## Different Parameterizations


|   Model    |  Equation |
|:--------|:---------|
|  Identity  | $ \mathbf{x}$ |
|   Linear   | $\mathbf{wx}+\mathbf{b}$ |
|   Basis    |  $\mathbf{w}\boldsymbol{\phi}(\mathbf{x}) + \mathbf{b}$  |
| Non-Linear | $\sigma\left( \mathbf{wx} + \mathbf{b}\right)$ |
| Neural Network | $\boldsymbol{f}_{L}\circ \boldsymbol{f}_{L-1}\circ\ldots\circ\boldsymbol{f}_1(\mathbf{x})$ |
| Functional | $\boldsymbol{f} \sim \mathcal{GP}\left(\boldsymbol{\mu}_{\boldsymbol \alpha}(\mathbf{x}),\boldsymbol{\sigma}^2_{\boldsymbol \alpha}(\mathbf{x})\right)$ |

---
#### Identity


$$
f(x;\theta) = x
$$


---
#### Linear

A linear function of $\mathbf{w}$ wrt $\mathbf{x}$.

$$
f(x;\theta) = \mathbf{w}^\top x
$$


---
#### Basis Function

A linear function of $\mathbf{w}$ wrt to the basis function $\phi(x)$.

$$
\boldsymbol{f}(\mathbf{x};\boldsymbol{\theta}) = \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x};\boldsymbol{\theta})
$$

**Examples**

* $\phi(x) = (1, x, x^2, \ldots)$
* $\phi(x) = \tanh(x + \gamma)^\alpha$
* $\phi(x) = \exp(- \gamma||x-y||_2^2)$
* $\phi(x) = \left[\sin(2\pi\boldsymbol{\omega}\mathbf{x}),\cos(2\pi\boldsymbol{\omega}\mathbf{x}) \right]^\top$



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
