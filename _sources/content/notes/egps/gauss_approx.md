# Gaussian Approximation




$$
\begin{aligned}
m(\mathbf{\mu}_\mathbf{x}, \mathbf{\Sigma}_\mathbf{x}) &= \mathbb{E}_\mathbf{x_*}\left[ \mathbb{E}_{f_*} \left[ f_* |\mathbf{x}_* \right] \right] = \mathbb{E}_\mathbf{x_*}\left[ \mathbf{\mu}(\mathbf{x}) \right] \\ 
\nu(\mathbf{\mu}_\mathbf{x}, \mathbf{\Sigma}_\mathbf{x}) &= \mathbb{E}_\mathbf{x_*} \left[ \mathbb{V}_{f_*} \left[ f_* | \mathbf{x_*} \right]  \right] + \mathbb{V}_\mathbf{x_*}\left[ \mathbb{E}_{f_*} \left[  f_*|\mathbf{x}_* \right] \right] \\ 
&= \mathbb{E}_\mathbf{x_*}\left[ \mathbf{\Sigma}^2(\mathbf{x_*}) \right] + \mathbb{V}\left[ \mathbf{\mu}(\mathbf{x_*}) \right] \\ 
&= \mathbb{E}_\mathbf{x_*}\left[ \mathbf{\Sigma}^2(\mathbf{x_*}) \right] + \mathbb{E}_\mathbf{x_*}\left[ \mathbf{\mu}^2(\mathbf{x_*}) \right] - \mathbb{E}^2_\mathbf{x_*}\left[ \mathbf{\mu}(\mathbf{x_*}) \right]
\end{aligned}
$$


## Proof


#### Mean Function

$$
\begin{aligned}
m(\mu_\mathbf{x}, \Sigma_\mathbf{x})
&=
\mathbb{E}_\mathbf{f_*}
\left[ f_* \, \mathbb{E}_\mathbf{x_*} \left[ p(f_*|\mathbf{x}_*) \right] \right] \\
&=
\mathbb{E}_\mathbf{x_*}
\left[ \mathbb{E}_{f_*} \left[ f_* \,p(f_* | \mathbf{x_*}) \right]\right]\\
&=
\mathbb{E}_{x_*}\left[ \mu_\text{GP}(\mathbf{x_*}) \right]
\end{aligned}
$$

#### Variance Function

The variance term is a bit more complex.

$$
\begin{aligned}
v(\mu_\mathbf{x}, \Sigma_\mathbf{x})
&=
\mathbb{E}_\mathbf{f_*}
\left[ f_*^2 \, \mathbb{E}_\mathbf{x_*} \left[ p(f_*|\mathbf{x}_*) \right] \right] -
\left(\mathbb{E}_\mathbf{f_*}
\left[ f_* \, \mathbb{E}_\mathbf{x_*} \left[ p(f_*|\mathbf{x}_*) \right] \right]\right)^2\\
&=
\mathbb{E}_\mathbf{x_*}
\left[ \mathbb{E}_\mathbf{x_*} \left[ f_*^2 \, p(f_*|\mathbf{x}_*) \right] \right] -
\left(\mathbb{E}_\mathbf{x_*}
\left[ \mathbb{E}_\mathbf{x_*} \left[ f_* \, p(f_*|\mathbf{x}_*) \right] \right]\right)^2\\
&=
\mathbb{E}_\mathbf{x_*}
\left[  \sigma_\text{GP}^2(\mathbf{x}_*) + \mu_\text{GP}^2(\mathbf{x}_*) \right] -
\mathbb{E}_{x_*}\left[ \mu_\text{GP}(\mathbf{x_*}) \right]^2 \\
&=
\mathbb{E}_\mathbf{x_*}
\left[  \sigma_\text{GP}^2(\mathbf{x}_*) \right] + \mathbb{E}_\mathbf{x_*} \left[ \mu_\text{GP}^2(\mathbf{x}_*) \right] -
\mathbb{E}_{x_*}\left[ \mu_\text{GP}(\mathbf{x_*}) \right]^2\\
&=
\mathbb{E}_\mathbf{x_*} \left[  \sigma_\text{GP}^2(\mathbf{x}_*) \right] +
\mathbb{V}_\mathbf{x_*} \left[\mu_\text{GP}(\mathbf{x}_*) \right]
\end{aligned}
$$
