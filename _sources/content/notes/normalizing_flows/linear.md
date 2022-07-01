# Linear Layers


---
## Overview






---
## Permutations

$$
\boldsymbol{f}(\mathbf{x}) = \mathbf{Px}
$$

where $\mathbf{P}$ is a permutation matrix.


---
## Free


$$
\boldsymbol{f} = \mathbf{Ax}
$$

### Orthogonal

$$
\boldsymbol{f}(\mathbf{x}) = \mathbf{Ax}
$$

where $\mathbf{AA}\top = \mathbf{I}$. There are two ways to accomplish this:

* Random sample st it is 
* QR decomposition
* HouseHolder Parameterization

### HouseHolder Parameterization

---
## Sylvester


---
## 1x1 Convolution



---
## Convolutional Exponential


**Forward**

$$
\boldsymbol{f}(\mathbf{x}) = \exp(\mathbf{M})\mathbf{x}
$$

**Inverse**

$$
\boldsymbol{f}^{-1}(\mathbf{x}) = \exp(-\mathbf{M})\mathbf{x}
$$

This is because $\exp(x)^{-1} = \exp(-x)$ (Golinski et al, 2019).

**Log Determinant Jacobian**

$$
\boldsymbol{\nabla}_{\mathbf{x}}\boldsymbol{f}(\mathbf{x}) =\log |\exp(\mathbf{M})| = \text{trace}(\mathbf{M})
$$

We can apply the same strategy for convolutions.

$$
\mathbf{m}_e^* \mathbf{x} = \mathbf{x} + \frac{\mathbf{mx}}{1!} + \frac{\mathbf{m}(\mathbf{mx})}{2!} + \ldots
$$

All higher order terms need the same consecutive convolutions.


