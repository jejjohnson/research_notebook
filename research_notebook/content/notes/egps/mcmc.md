# Monte Carlo Sampling

$$
p(\mathbf{f}_*|\mathbf{\mu}_\mathbf{x},\mathbf{\Sigma}_\mathbf{x},\mathcal{D}) = \int \mathcal{N}\left(\mathbf{f}_*|\mathbf{\mu}_\mathcal{GP}(\mathbf{x}_*),\mathbf{\Sigma}^2_\mathcal{GP}(\mathbf{x}_*) \right) \; \mathcal{N}(\mathbf{x}_*|\mathbf{\mu}_\mathbf{x}, \mathbf{\Sigma}_\mathbf{x})\; d\mathbf{x}_*
$$