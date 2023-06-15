# Modulations


$$
\phi_\ell = \sigma\left( \omega_\ell\left(\mathbf{wx} + \mathbf{b} \right) \right)
$$ (eq:nerf_mod)

where $\sigma$ is an arbitrary activation function.


$$
\boldsymbol{h}(\mathbf{x};\boldsymbol{\theta}(\mathbf{x}))
$$


For example we have the hypernetwork:

$$
\begin{aligned}
\phi_\ell(\mathbf{x};\theta) &= \mathbf{w}_\theta\mathbf{x} + \mathbf{b}_\theta \\
[\mathbf{w}_\theta,\mathbf{b}_\theta] &= \boldsymbol{h}(\mathbf{z};\boldsymbol{\theta})
\end{aligned}
$$

We have the additive transformation

$$
\begin{aligned}
\phi_\ell(\mathbf{x};\theta) &= \mathbf{w}_\ell\mathbf{x} + \mathbf{b}_\ell + \mathbf{s}_\theta \\
\mathbf{s}_\theta &= \boldsymbol{h}(\mathbf{z};\boldsymbol{\theta})
\end{aligned}
$$


We have the affine transformation

$$
\begin{aligned}
\phi_\ell(\mathbf{x};\theta) &=
\left(\mathbf{w}_\ell\mathbf{x} + \mathbf{b}_\ell\right)\mathbf{s}_\theta + \mathbf{a}_\theta \\
[\mathbf{s}_\theta, \mathbf{a}_\theta] &= \boldsymbol{h}(\mathbf{z};\boldsymbol{\theta})
\end{aligned}
$$


## Example: 2D Sea Surface Height

In this example, we will train the same SIREN model from before but we will add a modulation extension.
