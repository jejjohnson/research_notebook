# Taylor Approximation


$$
\begin{aligned}
\tilde{\mathbf{\mu}}_\text{LinGP}(\mathbf{x_*}) &= \mathbf{\mu}_\text{GP}(\mathbf{\mu}_\mathbf{x_*}) +
\underbrace{\frac{1}{2} \text{Tr}\left\{ \frac{\partial^2 \mathbf{\mu}_\text{GP}(\mathbf{\mu}_\mathbf{x_*})}{\partial \mathbf{x_*} \partial \mathbf{x_*}^\top}  \mathbf{\Sigma}_\mathbf{x_*}\right\}}_\text{second Order}\\
\tilde{\mathbf{\Sigma}}^2_\text{LinGP} (\mathbf{x_*}) &= 
\mathbf{\Sigma}^2_\text{GP}(\mathbf{\mu}_\mathbf{x_*}) + 
\underbrace{\frac{\partial \mathbf{\mu}_\text{GP}(\mathbf{\mu}_\mathbf{x_*})}{\partial \mathbf{x_*}}^\top
\mathbf{\Sigma}_\mathbf{x_*}
\frac{\partial \mathbf{\mu}_\text{GP}(\mathbf{\mu}_\mathbf{x_*})}{\partial \mathbf{x_*}}}_\text{1st Order} +
\underbrace{\frac{1}{2} \text{Tr}\left\{ \frac{\partial^2 \mathbf{\Sigma}^2_\text{GP}(\mathbf{\mu}_\mathbf{x_*})}{\partial \mathbf{x_*} \partial \mathbf{x_*}^\top}  \mathbf{\Sigma}_\mathbf{x_*}\right\}}_\text{second Order}
\end{aligned}
$$

---

## Proof

We will approximate our mean and variance function via a Taylor Expansion. First let's take a step back and look at the taylor expansion of a single function $f$ w.r.t. to $\mathbf{x}_*$ which is characterized by its mean $\mu_\mathbf{x_*}$ and variance function $\Sigma_\mathbf{x_*}$. Taking the first two orders, we get

$$\begin{aligned}
\mathbf{z}_\mu = f(\mathbf{x_*}) &\approx
\underbrace{f(\mu_\mathbf{x_*})}_\text{Zeroth Order} +
\underbrace{\nabla_\mathbf{x_*} f\bigg\vert_{\mathbf{x}_* = \mu_\mathbf{x}}
(\mathbf{x}_* - \mu_\mathbf{x_*})}_\text{1st Order} \\
&+ \underbrace{\nabla^2_\mathbf{x_*} f \bigg\vert_{\mathbf{x}_* = \mu_\mathbf{x}}
(\mathbf{x}_* - \mu_\mathbf{x_*})^\top (\mathbf{x}_* - \mu_\mathbf{x_*})}_\text{2nd Order} +
\underbrace{\mathcal{O} (\mathbf{x_*}^3)}_\text{Higher Order}
\end{aligned}$$


### Mean Function


Now we need to take the expectation of our approximation, $\mathbb{E}[\mathbf{z}_\mu]$. We tackle each of the terms individually below.

#### Zeroth Term

For the first term, we take the expectation. 

$$
 \mathbb{E}_\mathbf{x_*}\left[ f(\mu_\mathbf{x_*})\right] =  f(\mu_\mathbf{x_*}) 
$$

This is the same because the expectation of the mean of a function $f$ is simply the function $f$ evaluated at the mean.

#### 1st Order Term

The 1st order term is given by:

$$\begin{aligned}
 \mathbb{E}_\mathbf{x_*}\left[ \nabla_\mathbf{x_*} f \bigg\vert_{\mathbf{x}_* = \mu_\mathbf{x}}
(\mathbf{x}_* - \mu_\mathbf{x_*})\right] 
&=  \nabla_\mathbf{x_*} f(\mu_\mathbf{x_*}) \mathbb{E}_\mathbf{x_*}\left[ (\mathbf{x}_* - \mu_\mathbf{x_*}) \right] = 0
\end{aligned}$$

$\mathbb{E}[(\mathbf{x}_* - \mu_\mathbf{x_*})]=0$ because the terms cancel each other out.

#### 2nd Order Term

The 2nd order term is given by:

$$\begin{aligned}
\mathbb{E}_\mathbf{x_*} \left[ \nabla^2_\mathbf{x_*} f \bigg\vert_{\mathbf{x}_* = \mu_\mathbf{x}}
(\mathbf{x}_* - \mu_\mathbf{x_*})^\top (\mathbf{x}_* - \mu_\mathbf{x_*}) \right] &= \nabla_\mathbf{x_*}^2 f(\mu_\mathbf{x_*})  \mathbb{E}_\mathbf{x_*}\left[ (\mathbf{x}_* - \mu_\mathbf{x_*})^\top (\mathbf{x}_* - \mu_\mathbf{x_*}) \right]
\end{aligned}$$

We have the covariance term so we can simplify this:

$$
\mathbb{E}_\mathbf{x_*} \left[ \nabla^2_\mathbf{x_*} f \bigg\vert_{\mathbf{x}_* = \mu_\mathbf{x}}
(\mathbf{x}_* - \mu_\mathbf{x_*})^\top (\mathbf{x}_* - \mu_\mathbf{x_*}) \right] = \nabla_\mathbf{x_*}^2 f(\mu_\mathbf{x_*}) \Sigma_\mathbf{x_*}
$$

So we're left with:

$$
 \mathbb{E}_\mathbf{x_*}\left[ f(\mathbf{x_*})\right]  \approx  f(\mu_\mathbf{x_*}) + \nabla^2_\mathbf{x_*} f(\mu_\mathbf{x})^\top\;\Sigma_{\mathbf{x_*}} + \mathcal{O} (\mathbf{x_*}^3) 
$$

which we can simplify to:

$$
f(\mathbf{x_*}) \approx f(\mu_\mathbf{x_*}) +
\underbrace{\frac{1}{2} \text{Tr}\left\{ \nabla^2_\mathbf{x_*} f(\mu_\mathbf{x_*})^\top\;\Sigma_\mathbf{x_*}\right\}}_\text{2nd Order}
$$


#### Linearized GP Mean

So now instead of a simple function $f$, we have our GP predictive mean equation $\mu_\text{GP}$ which we can simply plug into the approximation.
%
$$
\tilde{\mu}_\text{LinGP}(\mathbf{x_*}) = \mu_\text{GP}(\mu_\mathbf{x_*}) +
\underbrace{\frac{1}{2} \text{Tr}\left\{ \nabla^2_\mathbf{x_*} \mu_\text{GP}(\mu_\mathbf{x_*})^\top\;\Sigma_\mathbf{x_*}\right\}}_\text{2nd Order}
$$


### Variance Function

So the variance term is a bit more difficult to calculate due to the $\mathbb{V}[\cdot]$ operator.

$$
\tilde{\sigma}^2_\text{LinGP}(\mathbf{x_*}) =\mathbb{E}_\mathbf{x_*} \left[  \sigma_\text{GP}^2(\mathbf{x}_*) \right] +
\mathbb{V}_\mathbf{x_*} \left[\mu_\text{GP}(\mathbf{x}_*) \right]
$$

**Term I**: The formulation for the expectation of the Taylor expanded predictive variance function is similar to the equation above. So we can replace $\mu_\text{GP}$ with $\sigma^2_\text{GP}$.

$$
\mathbb{E}_\mathbf{x_*} \left[  \sigma_\text{GP}^2(\mathbf{x}_*) \right] = \sigma^2_\text{GP}(\mu_\mathbf{x_*}) +
\underbrace{\frac{1}{2} \text{Tr}\left\{ \nabla^2_\mathbf{x_*} \sigma^2_\text{GP}(\mu_\mathbf{x_*})^\top\;\Sigma_\mathbf{x_*}\right\}}_\text{2nd Order}
$$

**Term II**: This term is more difficult to calculate. Again, we'll take a step back and see how this is for a function $f$. If we want a first order approximation, we will have the following:

$$
\mathbb{V}_\mathbf{x_*}\left[f(\mathbf{x_*}) \right] = \nabla_\mathbf{x_*}f(\mu_\mathbf{x_*})^\top \Sigma_\mathbf{x_*}\nabla_\mathbf{x_*}f(\mu_\mathbf{x_*})
$$

This is a sufficient approximation when $f(\mathbf{x})$ is approximately linear and/or when $\Sigma_\mathbf{x_*}$ is relatively small compared to $f(\mu_\mathbf{x_*})$. Alternatively, we can add a second order approximation which would add the following terms:

$$\begin{aligned}
\mathbb{V}_\mathbf{x_*}\left[f(\mathbf{x_*}) \right] &\approx
\underbrace{\left(\nabla_\mathbf{x_*}f(\mu_\mathbf{x_*})\right)^2 \Sigma_\mathbf{x_*}}_\text{1st Order} \\
&- \underbrace{\frac{1}{4}\left(\nabla^2_\mathbf{x_*}f(\mu_\mathbf{x_*})\right)^2\Sigma_\mathbf{x_*} + \mathbb{E}_\mathbf{x_*}\left[ \mathbf{x_*}-\mu_\mathbf{x_*} \right]\nabla^3_\mathbf{x_*}f(\mu_\mathbf{x_*}) + \frac{1}{4}\mathbb{E}_\mathbf{x_*}\left[\mathbf{x_*}-\mu_\mathbf{x_*} \right](\nabla^2_\mathbf{x_*}f(\mu_\mathbf{x_*}))^2}_\text{2nd Order}
\end{aligned}$$

This expression has 3rd and 4th central moments with respect to the mean. These terms are often negligible according to the conditions mentioned above. In addition, it is a very expensive calculation for some functions. So a practical compromise is to use the 2nd order approximation for the mean and the first order approximation for the variance. This is the approach given here as 3rd and 4th central moments of kernel functions is very expensive. So combining the terms together, we get:

$$\begin{aligned}
\mathbb{V}_\mathbf{x_*}\left[f(\mathbf{x_*}) \right] &\approx f(\mu_\mathbf{x_*}) + \frac{1}{2} \text{Tr}\left\{ \nabla^2_\mathbf{x_*} f(\mu_\mathbf{x_*})^\top\;\Sigma_\mathbf{x_*}\right\} +
\left(\nabla_\mathbf{x_*}f(\mu_\mathbf{x_*})\right)^2 \Sigma_\mathbf{x_*}
\end{aligned}$$

So then subsituting all of the portions for the appropriate GP function, we get the following:

$$\begin{aligned}
\tilde{\sigma}^2_\text{LinGP}(\mathbf{x_*}) = \sigma^2_\text{GP}(\mu_\mathbf{x_*})+
\left(\nabla_\mathbf{x_*}\mu_\text{GP}(\mu_\mathbf{x_*})\right)^2 \Sigma_\mathbf{x_*}+
\frac{1}{2} \text{Tr}\left\{ \nabla^2_\mathbf{x_*} \sigma^2_\text{GP}(\mu_\mathbf{x_*})^\top\;\Sigma_\mathbf{x_*}\right\} 
\end{aligned}$$

### Linearized Predictive Mean and Variance

$$\begin{aligned}
\tilde{\mu}_\text{LinGP}(\mathbf{x_*}) &= \underbrace{\mu_\text{GP}(\mu_\mathbf{x_*})}_\text{1st Order} +
\underbrace{\frac{1}{2} \text{Tr}\left\{ \nabla^2_\mathbf{x_*} \mu_\text{GP}(\mu_\mathbf{x_*})^\top\;\Sigma_\mathbf{x_*}\right\}}_\text{2nd Order}\\
\tilde{\sigma}^2_\text{LinGP}(\mathbf{x_*}) &= \underbrace{\sigma^2_\text{GP}(\mu_\mathbf{x_*})+
\nabla_\mathbf{x_*}\mu_\text{GP}(\mu_\mathbf{x_*})^\top \Sigma_\mathbf{x_*} \nabla_\mathbf{x_*}\mu_\text{GP}(\mu_\mathbf{x_*})}_\text{1st Order} \\
&+
\underbrace{\frac{1}{2} \text{Tr}\left\{ \nabla^2_\mathbf{x_*} \sigma^2_\text{GP}(\mu_\mathbf{x_*})^\top\;\Sigma_\mathbf{x_*}\right\}}_\text{2nd Order}
\end{aligned}$$

where $\nabla_\mathbf{x_*}$ is the gradient of the function w.r.t. $\mathbf{x}$ and $\nabla_\mathbf{x_*}^2 $ is the second derivative (the Hessian) of the function w.r.t. $\mathbf{x_*}$. This is a second-order approximation which has that expensive Hessian term. There have have been studies that have shown that that term tends to be neglible in practice and a first-order approximation is typically enough.

Practically speaking, this leaves us with the following predictive mean and variance functions:

$$\begin{aligned}
\mu_\text{GP}(\mathbf{x_*}) &= k(\mathbf{x_*}) \, \mathbf{K}_{GP}^{-1}y=k(\mathbf{x_*}) \, \alpha  \\
\nu_{GP}^2(\mathbf{x_*}) &= \sigma_y^2 + {\color{red}{\nabla_{\mu_\text{GP}}\,\Sigma_\mathbf{x_*} \,\nabla_{\mu_\text{GP}}^\top} }+ k_{**}- {\mathbf k}_* ({\mathbf K}+\sigma_y^2 \mathbf{I}_N )^{-1} {\mathbf k}_{*}^{\top}
\end{aligned}$$

As seen above, the only extra term we need to include is the derivative of the mean function that is present in the predictive variance term.



## Sparse GPs

We can extend this method to other GP algorithms including sparse GP models. The only thing that changes are the original $\mu_{GP}$ and $\nu^2_{GP}$ equations. In a sparse GP we have the following predictive functions

$$\begin{aligned}
    \mu_{SGP} &= K_{*z}K_{zz}^{-1}m \\
    \nu^2_{SGP} &= K_{**} 
    - K_{*z}\left[ K_{zz}^{-1} - K_{zz}^{-1}SK_{zz}^{-1} \right]K_{*z}^{\top}
\end{aligned}$$

So the new predictive functions will be:

$$\begin{aligned}
    \mu_{SGP} &= k_{*z}K_{zz}^{-1}m \\
    \nu^2_{SGP} &= K_{**} 
    - K_{*z}\left[ K_{zz}^{-1} - K_{zz}^{-1}SK_{zz}^{-1} \right]K_{*z}^{\top} 
    + \tilde{\Sigma}_x
\end{aligned}$$

As shown above, this is a fairly extensible method that offers a cheap improved predictive variance estimates on an already trained GP model. Some future work could be evaluating how other GP models, e.g. Sparse Spectrum GP, Multi-Output GPs, e.t.c.