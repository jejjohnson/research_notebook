# Structured Kernel Interpolation

$$
\tilde{\boldsymbol{k}}(\mathbf{x}, \mathbf{x}') = \mathbf{w}_\mathbf{x}\mathbf{K}_{\mathbf{XX}}\mathbf{w}_{\mathbf{x}'}
$$

$$
\tilde{\mathbf{K}}_{\mathbf{XX}} = \mathbf{W} \mathbf{K}_{\mathbf{XX}}\mathbf{W}^\top
$$

$$
\tilde{\mathbf{K}}_{\mathbf{XU}} \approx \boldsymbol{w}_{U}(\mathbf{x})\mathbf{K_{UU}}
$$

where $\mathbf{W_U} \in \mathbb{R}^{N \times M}$ is matrix of interpolation weights.

Here we have the standard decomposition of the inverse

$$
(\mathbf{K} + \sigma^2 \mathbf{I})^{-1}\mathbf{y} = (\mathbf{QVQ}^\top + \sigma^2\mathbf{I})^{-1}\mathbf{y}
$$
