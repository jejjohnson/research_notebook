# Minimization Problems


Recall the bilevel optimization problem.

$$
\begin{aligned}
\boldsymbol{\theta}^* &= \underset{\boldsymbol{\theta}}{\text{argmin  }}  \mathcal{L}(\boldsymbol{\theta},\mathbf{x}^*(\boldsymbol{\theta})) \\
\mathbf{x}^*(\boldsymbol{\theta}) &= \underset{\mathbf{x}}{\text{argmin  }} \mathcal{U}(\mathbf{x},\boldsymbol{\theta})
\end{aligned}
$$


We can define a simple minimization problem as

$$
x^* = \underset{\mathbf{x}}{\text{argmin  }} \mathcal{U}(\mathbf{x},\boldsymbol{\theta})
$$

To be more explicit, let's have

$$
x^* = \underset{\mathbf{x}}{\text{argmin  }} ||y-Ax||^2 - \log p_\theta(x)
$$

---

We let the normalizing flow be the prior!

$$
\log p_\theta(x) = \log p_z(T(x)) + \log\det|\nabla_x T_\theta(x)|
$$

or equivalently

$$
\log p_\theta(x) = \log p_z(z) + \log\det|\nabla_z T_\theta^{-1}(z)|
$$

---

We define the same minimization function in the transform domain, $z$.

$$
\mathcal{L}_z(x,\theta) = \underset{\mathbf{z}}{\text{argmin  }}
$$

$$
\mathcal{L}_z(z,\theta) = ||y-AT_\theta(z)||^2 - \log\mathcal{G}(z)
$$

**Proof**:

$$
p(x|y) \propto \frac{1}{Z}\exp(-||y-Ax||^2)p_\theta(x)
$$

We can compute expectations of the posterior

$$
\mathbb{E}_{x\sim p(x|y)}[f(x)] = \int_{} f(x)\exp(-||y-Ax||^2)p_\theta(x)dx
$$

Let's describe the function

$$
F(x):=f(x)\exp(-||y-Ax||^2)
$$

We can do the change of variables here:

$$
F(T_\theta(z)):=f(T_\theta(z))\exp(-||y-AT_\theta(z)||^2)
$$

Now we can plug this back into the integral

$$
\mathbb{E}_{z\sim p(z|y)}[f(z)] = \int_{} f(T_\theta(z))\exp(-||y-AT_\theta(z)||^2)\eta(z)dz
$$

So we have our loss function again

$$
\mathcal{L}_z(z,\theta) = ||y-AT_\theta(z)||^2 - \log\mathcal{G}(z)
$$


**Proof**:
