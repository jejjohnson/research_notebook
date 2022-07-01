# Conjugate Gradients


Consider the following optimal solution to the GP given the best optimal hyper-parameters, $\boldsymbol{\theta} = \{ \boldsymbol{\psi, \phi}, \sigma^2 \}$:

$$
\boldsymbol{\alpha} = \mathbf{K}_{\boldsymbol{\phi}}^{-1} \bar{\mathbf{Y}}_{\boldsymbol{\psi}}
$$

We can rewrite this as a linear system.

$$
\mathbf{K}_{\boldsymbol{\phi}}\boldsymbol{\alpha} - \bar{\mathbf{Y}}_{\boldsymbol{\psi}} = \mathbf{0}
$$

We can reformulate this as a quadratic optimization problem:

$$
\boldsymbol{\alpha}^* = \argmin_{\boldsymbol{\alpha}} \boldsymbol{\alpha}^\top \mathbf{K}_{\boldsymbol{\phi}}\boldsymbol{\alpha} - \boldsymbol{\alpha}^\top \bar{\mathbf{Y}}_{\boldsymbol{\psi}}
$$

There are many efficient ways to solve this problem where one of them is the *conjugate gradient* operation. This is an iterative algorithm that recovers the exact solution after $k$ iterations. But we can recover an approximate solution after $\tilde{k}$ iterations where $\tilde{k} << k$. Each iteration is $\mathcal{O}(N^2)$.

**GPyTorch: BlackBox Matrix-Matrix Gaussian Process Inference with GPU Acceleration** - Gardner et al (2018) 