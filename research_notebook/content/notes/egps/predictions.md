# Uncertain Predictions


## Numerical


### Monte Carlo Methods


$$
\begin{aligned}
p(\mathbf{f}_*|\boldsymbol{\mu}_\mathbf{x},\mathbf{\Sigma}_\mathbf{x},\mathcal{D}) 
&= \int \mathcal{N}\left(\mathbf{f}_*|\boldsymbol{\mu}_\mathcal{GP}(\mathbf{x}_*),\boldsymbol{\sigma}^2_\mathcal{GP}(\mathbf{x}_*) \right) \; \mathcal{N}(\mathbf{x}_*|\boldsymbol{\mu}_\mathbf{x}, \mathbf{\Sigma}_\mathbf{x})\; d\mathbf{x}_* \\
&= \frac{1}{T} \sum_{t=1}^T p(\mathbf{f}_*|\mathcal{D}, \mathbf{x}_*^t)\\
&= \frac{1}{T} \sum_{t=1}^T  \mathcal{N}\left(\boldsymbol{\mu}_\mathcal{GP}(\mathbf{x}_*^t),\boldsymbol{\sigma}^2_\mathcal{GP}(\mathbf{x}_*^t) \right)
\end{aligned}
$$

---

### Sigma Point Methods

**TODO**

---

## Approximate Moments


$$
\begin{aligned}
m(\boldsymbol{\mu}_\mathbf{x}, \mathbf{\Sigma}_\mathbf{x}) &= \mathbb{E}_\mathbf{x_*}\left[ \mathbb{E}_{f_*} \left[ f_* |\mathbf{x}_* \right] \right] = \mathbb{E}_\mathbf{x_*}\left[ \boldsymbol{\mu}(\mathbf{x}) \right] \\ 
\nu(\boldsymbol{\mu}_\mathbf{x}, \mathbf{\Sigma}_\mathbf{x}) &= \mathbb{E}_\mathbf{x_*} \left[ \mathbb{V}_{f_*} \left[ f_* | \mathbf{x_*} \right]  \right] + \mathbb{V}_\mathbf{x_*}\left[ \mathbb{E}_{f_*} \left[  f_*|\mathbf{x}_* \right] \right] \\ 
&= \mathbb{E}_\mathbf{x_*}\left[ \boldsymbol{\sigma}^2(\mathbf{x_*}) \right] + \mathbb{V}\left[ \boldsymbol{\mu}(\mathbf{x_*}) \right] \\ 
&= \mathbb{E}_\mathbf{x_*}\left[ \boldsymbol{\sigma}^2(\mathbf{x_*}) \right] + \mathbb{E}_\mathbf{x_*}\left[ \boldsymbol{\mu}^2(\mathbf{x_*}) \right] - \mathbb{E}^2_\mathbf{x_*}\left[ \boldsymbol{\mu}(\mathbf{x_*}) \right]
\end{aligned}
$$


* Deterministic Methods
  * Linearization
  * Sigma Points
* Stochastic Methods
  * Monte Carlo Sampling


---

### Linearization

This involves the Taylor Series expansion of the function $f$.


$$
\begin{aligned}
\tilde{\boldsymbol{\mu}}_\text{LinGP}(\mathbf{x_*}) &= \boldsymbol{\mu}_\text{GP}(\boldsymbol{\mu}_\mathbf{x_*}) +
\underbrace{\frac{1}{2} \text{Tr}\left\{ \frac{\partial^2 \boldsymbol{\mu}_\text{GP}(\boldsymbol{\mu}_\mathbf{x_*})}{\partial \mathbf{x_*} \partial \mathbf{x_*}^\top}  \mathbf{\Sigma}_\mathbf{x_*}\right\}}_\text{second Order}\\
\tilde{\mathbf{\Sigma}}^2_\text{LinGP} (\mathbf{x_*}) &= 
\boldsymbol{\sigma}^2_\text{GP}(\boldsymbol{\mu}_\mathbf{x_*}) + 
\underbrace{\frac{\partial \boldsymbol{\mu}_\text{GP}(\boldsymbol{\mu}_\mathbf{x_*})}{\partial \mathbf{x_*}}^\top
\mathbf{\Sigma}_\mathbf{x_*}
\frac{\partial \boldsymbol{\mu}_\text{GP}(\boldsymbol{\mu}_\mathbf{x_*})}{\partial \mathbf{x_*}}}_\text{1st Order} +
\underbrace{\frac{1}{2} \text{Tr}\left\{ \frac{\partial^2 \boldsymbol{\sigma}^2_\text{GP}(\boldsymbol{\mu}_\mathbf{x_*})}{\partial \mathbf{x_*} \partial \mathbf{x_*}^\top}  \mathbf{\Sigma}_\mathbf{x_*}\right\}}_\text{second Order}
\end{aligned}
$$


---

### Sigma Points 

**TODO**

---

## (More) Exact Moments


$$
\begin{aligned}
\tilde{\boldsymbol{\mu}}_{MMGP}(\mathbf{x_*}) &= \int \boldsymbol{\mu}_{GP}(\mathbf{x_*}) p(\mathbf{x_*})d\mathbf{x_*} \\ 
\tilde{\mathbf{\Sigma}}^2_{mmGP}(\mathbf{x}_*) &= \int \boldsymbol{\sigma}^2_{GP}(\mathbf{x_*}) p(\mathbf{x_*}) d\mathbf{x}_* + \int  \boldsymbol{\mu}_{GP}^2(\mathbf{x_*})p(\mathbf{x_*})d\mathbf{x_*}  - \left[ \int \boldsymbol{\mu}_{GP}(\mathbf{x_*}) p(\mathbf{x_*})d\mathbf{x_*}\right]^2
\end{aligned}
$$

After some manipulation, this results in the follow equations for the predictive mean and variance:

$$
\begin{aligned}
\tilde{\boldsymbol{\mu}}_{MMGP}(\mathbf{x_*}) &= \Psi_1^\top\alpha \\
\tilde{\mathbf{\Sigma}}^2_{MMGP}(\mathbf{x}_*)
&=
\psi_0 - \text{Tr}\left( \left(\mathbf{K}_{GP}^{-1}  - \alpha\alpha^\top\right) \Psi_2\right) - \text{Tr}\left( \Psi_1\Psi_1^\top\alpha\alpha^\top \right),
\end{aligned}
$$

where we have $\boldsymbol{\Psi_i}$ quantities called kernel expectations denoted by:

$$
\begin{aligned}
[\psi_0]_{i}  &= \int k(\mathbf{x}_i, \mathbf{x}_i)p(\mathbf{x}_i)d\mathbf{x}_i \\
[\Psi_1]_{ij} &= \int k(\mathbf{x}_i, \mathbf{y}_j)p(\mathbf{x}_i)d\mathbf{x}_i \\
[\Psi_2]_{ijk} &= \int k(\mathbf{x}_i, \mathbf{y}_j)k(\mathbf{x}_i, \mathbf{z}_k) d\mathbf{x}_i.
\end{aligned}
$$


---

### Analytical

* RBF Kernel
* Linear Kernel
* Spectral Mixture Kernel

**TODO**

---

### Linearization


#### Mean Function


$$
\begin{aligned}
\tilde{\boldsymbol{\mu}}_\text{LinGP}(\mathbf{x_*}) &= \boldsymbol{\mu}_\text{GP}(\boldsymbol{\mu}_\mathbf{x_*}) +
\frac{1}{2} \sum_{n=1}^N\alpha_n \text{Tr}\left\{ k''(\boldsymbol{\mu}_\mathbf{x_*},\mathbf{x}_n)  \mathbf{\Sigma}_\mathbf{x_*}\right\}
\end{aligned}
$$

#### Variance Function

$$
\begin{aligned}
\tilde{\boldsymbol{\sigma}}^2_\text{LinGP} (\mathbf{x_*}) &= \boldsymbol{\sigma}^2_\text{GP}(\boldsymbol{\mu}_\mathbf{x_*}) +
\frac{1}{2} \sum_{n=1}^N\alpha_n \text{Tr}\left\{ k''(\boldsymbol{\mu}_\mathbf{x_*},\mathbf{x}_n)  \mathbf{\Sigma}_\mathbf{x_*}\right\} \\
&- \sum_{i,j}^{N}(K_{ij}^{-1}-\alpha_i\alpha_j)\text{Tr}\left\{
    k'(\boldsymbol{\mu}_\mathbf{x}, \mathbf{x}_i)k'(\boldsymbol{\mu}_\mathbf{x}, \mathbf{x}_j)\boldsymbol{\Sigma}_\mathbf{x} \right\}\\
&- \frac{1}{2} \sum_{i,j=1}^NK_{ij}^{-1}
\left( k(\boldsymbol{\mu}_\mathbf{x}, \mathbf{x}_i) \text{Tr}\left\{k''(\boldsymbol{\mu}_\mathbf{x}, \mathbf{x}_j)\mathbf{\Sigma}_\mathbf{x} \right\} + k(\boldsymbol{\mu}_\mathbf{x}, \mathbf{x}_j)\text{Tr}\left\{k''(\boldsymbol{\mu}_\mathbf{x}, \mathbf{x}_j)\mathbf{\Sigma}_\mathbf{x} \right\}\right)
\end{aligned}
$$

---

### Sigma Points 

**TODO**


---

### Monte Carlo Sampling


$$
\begin{aligned}
[ \psi_0 ]  = \int k(\mathbf{x}_*, \mathbf{x}_*)p(\mathbf{x}_*)d\mathbf{x}_* 
&\approx \frac{1}{T} \sum_{t=1}^T   k(\mathbf{x}_*^t, \mathbf{x}_*^t)\\
[\Psi_1]_{j} = \int k(\mathbf{x}_*, \mathbf{y}_j)p(\mathbf{x}_*)d\mathbf{x}_* 
&\approx \frac{1}{T} \sum_{t=1}^T   k(\mathbf{x}^t_*, \mathbf{y}_j) \\
[\Psi_2]_{jk} = \int k(\mathbf{x}_*, \mathbf{y}_j)k(\mathbf{x}_*, \mathbf{z}_k) d\mathbf{x}_*  
&\approx \frac{1}{T} \sum_{t=1}^T   k(\mathbf{x}^t_*, \mathbf{y}_j) k(\mathbf{x}^t_*, \mathbf{z}_k)
\end{aligned}
$$