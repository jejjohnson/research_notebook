# Moment Matching



$$
\begin{aligned}
\tilde{\mathbf{\mu}}_{MMGP}(\mathbf{x_*}) &= \int \mathbf{\mu}_{GP}(\mathbf{x_*}) p(\mathbf{x_*})d\mathbf{x_*} \\ 
\tilde{\mathbf{\Sigma}}^2_{mmGP}(\mathbf{x}_*) &= \int \mathbf{\Sigma}^2_{GP}(\mathbf{x_*}) p(\mathbf{x_*}) d\mathbf{x}_* + \int  \mathbf{\mu}_{GP}^2(\mathbf{x_*})p(\mathbf{x_*})d\mathbf{x_*}  - \left[ \int \mathbf{\mu}_{GP}(\mathbf{x_*}) p(\mathbf{x_*})d\mathbf{x_*}\right]^2
\end{aligned}
$$

After some manipulation, this results in the follow equations for the predictive mean and variance:

$$
\begin{aligned}
\tilde{\mathbf{\mu}}_{MMGP}(\mathbf{x_*}) &= \Psi_1^\top\alpha \\
\tilde{\mathbf{\Sigma}}^2_{MMGP}(\mathbf{x}_*)
&=
\psi_0 - \text{Tr}\left( \left(\mathbf{K}_{GP}^{-1}  - \alpha\alpha^\top\right) \Psi_2\right) - \text{Tr}\left( \Psi_1\Psi_1^\top\alpha\alpha^\top \right),
\end{aligned}
$$

where we have $\boldsymbol{\Psi_i}$ quantities called kernel expectations denoted by:

$$
\begin{aligned}
[ \psi_0 ]_{i}  &= \int k(\mathbf{x}_i, \mathbf{x}_i)p(\mathbf{x}_i)d\mathbf{x}_i \\
[\Psi_1]_{ij} &= \int k(\mathbf{x}_i, \mathbf{y}_j)p(\mathbf{x}_i)d\mathbf{x}_i \\
[\Psi_2]_{ijk} &= \int k(\mathbf{x}_i, \mathbf{y}_j)k(\mathbf{x}_i, \mathbf{z}_k) d\mathbf{x}_i.
\end{aligned}
$$

---

## Proof


### Mean Function


$$
\begin{aligned}
\tilde{\mu}_{GP}(\mu_{\mathbf{x}}, \Sigma_\mathbf{x})
    &= \mathbb{E}_{\mathbf{x}_*}\left[
    \mu_{GP}(\mathbf{x}_*) \right]
    \\
    &= \int_\mathcal{X} \left[m_{GP} (\mathbf{x}_*) + k(\mathbf{X},\mathbf{x}_*)^\top \mathbf{K}_{GP}^{-1}(\mathbf{y}-m_{GP}(\mathbf{x}_*))\right]\; p(\mathbf{x}_*) d\mathbf x_*\\
    &= \int_\mathcal{X} m_{GP} (\mathbf{x}_*) \; p(\mathbf{x}_*) \; d\mathbf{x}_*
    +
    \int_\mathcal{X} k(\mathbf{X}, \mathbf{x}_*) \mathbf{K}_{\mathcal{GP}}^{-1}(\mathbf{y}-m_{\mathcal{GP}}(\mathbf{x}_*)) \; p(\mathbf{x}_*)d\mathbf{x}_*\\
    &= 
    \int_\mathcal{X} k(\mathbf{X}, \mathbf{x}_*) \underbrace{\mathbf{K}_{\mathcal{GP}}^{-1}\mathbf{y}}_{\alpha} \; p(\mathbf{x}_*)d\mathbf{x}_*\\
    &= 
     \int_\mathcal{X} k(\mathbf{X,x}_*)^{\top}\alpha  \; p(\mathbf{x}_*)\; d\mathbf{x}_*\\
    &= \alpha^\top \underbrace{\int_{X} k(\mathbf{X,x}_*) \cdot p(\mathbf{x}_*)d\mathbf{x}_*}_{\Psi_1}\\
\tilde{\mu}_{GP}(\mu_{\mathbf{x}}, \Sigma_\mathbf{x})
&= \Psi_1^\top\alpha \\
\end{aligned}
$$

### Predictive Variance


$$
\begin{aligned}
    \tilde{\sigma}^2_{GP}(\mathbf{x}_*) &=  \underbrace{\mathbb{E}_{\mathbf{x}_*} [\sigma^2_{GP}(\mathbf{x}_*)]}_{\text{Term I}} + \underbrace{\mathbb{E}_{\mathbf{x}_*}[\mu_{GP}^2(\mathbf{x}_*)]}_{\text{Term II}} - \underbrace{\mathbb{E}_{\mathbf{x}_*}[\mu_{GP}(\mathbf{x}_*)]^2}_{\text{Term III}}
\end{aligned}
$$

#### Term I

$$
\begin{aligned}
    \mathbb{E}_{\mathbf{x}_*} [\sigma_{GP}^2(\mathbf{x}_*)] 
    &= \int_{\mathcal{X}} \left[ k(\mathbf{x}_*, \mathbf{x}_*) - k(\mathbf{X,x}_*) \mathbf{K}_{GP}^{-1}k(\mathbf{X,x}_*)^{\top} \right] \; p(\mathbf{x}_*)d\mathbf{x}_*\\
    &= \int_{\mathcal{X}} k(\mathbf{x}_*, \mathbf{x}_*) \; p(\mathbf{x}_*)d\mathbf{x}_* - \int_{\mathcal{X}} k(\mathbf{X,x}_*) \mathbf{K}_{GP}^{-1}k(\mathbf{X,x}_*)^{\top} \; p(\mathbf{x}_*)d\mathbf{x}_* \\
    &=  \int_{\mathcal{X}} k(\mathbf{x}_*, \mathbf{x}_*) \; p(\mathbf{x}_*)d\mathbf{x}_* - \sum_{i,j}\mathbf{K}_{GP(i,j)}^{-1} k(\mathbf{x_i,x}_*) k(\mathbf{x_j,x}_*) \; p(\mathbf{x}_*)\\
    &=  \int_{\mathcal{X}} k(\mathbf{x}_*, \mathbf{x}_*) \; p(\mathbf{x}_*)d\mathbf{x}_* - \text{Tr}\left( \mathbf{K}_{GP}^{-1} \int_{\mathcal{X}}k(\mathbf{X,x}_*) k(\mathbf{X,x}_*) \; p(\mathbf{x}_*) d\mathbf{x}_*\right)\\
    &=  \psi_0 - \text{Tr}\left( \mathbf{K}_{GP}^{-1} \Psi_2\right)\\
\end{aligned}
$$


#### Term II

$$
\begin{aligned}
    \mathbb{E}_{\mathbf{x}_*}[\mu_{GP}^2(\mathbf{x}_*)] &= 
    \int_{\mathcal{X}} k(\mathbf{X,x}_*)^{\top}\alpha\alpha^\top k(\mathbf{X,x}_*) \; p(\mathbf{x}_*)d\mathbf{x}_* \\
    &=  \sum_{i,j}\alpha_i\alpha_j k(\mathbf{x_i,x}_*) k(\mathbf{x_j,x}_*) \; p(\mathbf{x}_*)\\
    &=  \text{Tr}\left( \alpha\alpha^\top \int_{\mathcal{X}}k(\mathbf{X,x}_*) \;k(\mathbf{X,x}_*) \; p(\mathbf{x}_*) \; d\mathbf{x}_*\right)\\
    &=  \text{Tr}\left( \alpha\alpha^\top \Psi_2\right)\\
\end{aligned}
$$

#### Term III

This is the squared expected value of the GP mean w.r.t. the noisy inputs $\mathbf{x}_*$. We've already calculated this above so we can just substitute this expression and square it:

$$
\begin{aligned}
\mathbb{E}_{\mathbf{x}_*}[\mu_{GP}(\mathbf{x}_*)]^2 
&= 
\left[ \tilde{\mu}_{GP}(\mu_{\mathbf{x}}, \Sigma_\mathbf{x})\right]^2 \\
&= [\Psi_1^\top\alpha]^2  \\
&= \text{Tr}\left( \Psi_1\Psi_1^\top\alpha\alpha^\top \right)  \\
\end{aligned}
$$


#### Final Solution

$$
\begin{aligned}
\tilde{\sigma}^2_{GP}(\mathbf{x}_*) 
&=
\psi_0 - \text{Tr}\left( \mathbf{K}_{GP}^{-1} \Psi_2\right) + \text{Tr}\left( \alpha\alpha^\top \Psi_2\right) - \text{Tr}\left(  \Psi_1\Psi_1^\top\alpha\alpha^\top \right)  \\
&=
\psi_0 - \text{Tr}\left( \left(\mathbf{K}_{GP}^{-1}  - \alpha\alpha^\top\right) \Psi_2\right) - \text{Tr}\left( \Psi_1\Psi_1^\top\alpha\alpha^\top \right) \\
&=
\psi_0 - \text{Tr}\left( \left(\mathbf{K}_{GP}^{-1}  - \alpha\alpha^\top\right) \Psi_2 - \Psi_1\Psi_1^\top\alpha\alpha^\top \right) \\
&=
\psi_0 - \text{Tr}\left( \left(\mathbf{K}_{GP}^{-1}  - \alpha\alpha^\top\right) \Psi_2 - (\Psi_1^\top\alpha)^2 \right)
\end{aligned}
$$