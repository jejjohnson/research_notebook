# Learning


## Parameter Learning Problem

$$
\boldsymbol{\theta}^* = \underset{\boldsymbol{\theta}}{\text{argmin  }} \mathcal{L}(\boldsymbol{\theta})
$$ (eqn:learn-param)

## Minimization Problem

$$
\mathbf{x}^* = \underset{\mathbf{x}}{\text{argmin  }} \mathcal{U}(\mathbf{x})
$$ (eqn:learn-minimize)

## Bi-Level Optimization Scheme

$$
\begin{aligned}
\boldsymbol{\theta}^* &= \underset{\boldsymbol{\theta}}{\text{argmin  }}  \mathcal{L}(\boldsymbol{\theta}) \\
\mathbf{x}^*(\boldsymbol{\theta}) &= \underset{\mathbf{x}}{\text{argmin  }} \mathcal{U}(\mathbf{x})
\end{aligned}
$$ (eqn:learn-bilevel)
