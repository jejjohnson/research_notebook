# Physics-Informed Loss


---
## Function

The learned function, $\boldsymbol{f_\theta}$, will map the spatial coordinates, $\mathbf{x}_\phi \in \mathbb{R}^{D\phi}$, and time coordinate, $t \in \mathbb{R}$, to sea surface height, $u \in \mathbb{R}$.

$$
u = \boldsymbol{f_\theta}(\mathbf{x}_\phi, t)
$$

---
## Loss

The standard loss term is data-driven

$$
\mathcal{L}_{data} = \text{MSE}(u, \hat{u}) = \frac{1}{N} \sum_{n=1}^N  \left(u - \boldsymbol{f_\phi}(\mathbf{x}_\phi, t) \right)^2
$$

However, there is no penalization to make the field behave the way we would expect. We also want a regularization which makes the field, $u$, behave how we would expect. This can be achieved by adding a physics-informed loss regularization term to the total loss.

$$
\mathcal{L} = \mathcal{L}_{data} + \lambda \mathcal{L}_{phy}
$$

This loss term can be minimized by effectively minimizing a PDE function. For example:

$$
\boldsymbol{f}_{phy}(\mathbf{x},t):= \partial_t u(\mathbf{x},t) + \mathcal{N}[u(\mathbf{x},t)] = 0
$$

where $\partial_t$ is the derivative of the field, $u$, wrt to time and $\mathcal{N}[\cdot]$ are some partial differential equations. We are interested in minimizing the full PDE, which we denote $\boldsymbol{f}_{phy}$, st it is 0. So the standard loss function applies.

---
## Examples

Below are some examples of how I have used the PINNs loss/regularization formulation in my own research.

---
### QG Equation

We have the following PDE for the QG dynamics:

$$
\partial_t q + \det J(\psi, q) = 0
$$

where $q(x,t) \in \mathbb{R}^2 \times \mathbb{R} \rightarrow \mathbb{R}$ is the potential vorticity (PV), $\psi(x,t) \in \mathbb{R}^2 \times \mathbb{R} \rightarrow \mathbb{R}$ is the stream function, $\partial_t$ is the partial derivative wrt $t$, $\boldsymbol{J}$, is the Jacobian operator and $\det \boldsymbol{J}(\cdot,\cdot)$ is the *determinant* of the Jacobian.

**Objective**: We want to convert this PDE in terms of sea surface height (SSH) instead of PV and the stream function.

---
#### QG Equation 4 SSH (TLDR)


**Note**: For the ease of notation, let's denote $u$ as the SSH. The above PDE can be written in terms of $u$

$$
\partial_t \nabla^2 u + c_2 \partial_t u + c_1\det \boldsymbol{J}(u, \nabla^2 u) +  c_3 \det \boldsymbol{J}(u, u) = 0
$$

See [the following page](./qg.md) for more and how this equation was derived.
