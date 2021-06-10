# Linearization (Taylor Expansions)

## TLDR


$$
\begin{aligned}
\tilde{\mathbf{\mu}}_\text{LinGP}(\mathbf{x_*}) &= \mathbf{\mu}_\text{GP}(\mathbf{\mu}_\mathbf{x_*}) +
\underbrace{\frac{1}{2} \text{Tr}\left\{ \frac{\partial^2 \mathbf{\mu}_\text{GP}(\mathbf{\mu}_\mathbf{x_*})}{\partial \mathbf{x_*} \partial \mathbf{x_*}^\top}  \mathbf{\Sigma}_\mathbf{x_*}\right\}}_\text{second Order}\\
\tilde{\mathbf{\sigma}}^2_\text{LinGP} (\mathbf{x_*}) &= 
\mathbf{\sigma}^2_\text{GP}(\mathbf{\mu}_\mathbf{x_*}) + 
\underbrace{\frac{\partial \mathbf{\mu}_\text{GP}(\mathbf{\mu}_\mathbf{x_*})}{\partial \mathbf{x_*}}^\top
\mathbf{\Sigma}_\mathbf{x_*}
\frac{\partial \mathbf{\mu}_\text{GP}(\mathbf{\mu}_\mathbf{x_*})}{\partial \mathbf{x_*}}}_\text{1st Order} +
\underbrace{\frac{1}{2} \text{Tr}\left\{ \frac{\partial^2 \mathbf{\Sigma}^2_\text{GP}(\mathbf{\mu}_\mathbf{x_*})}{\partial \mathbf{x_*} \partial \mathbf{x_*}^\top}  \mathbf{\Sigma}_\mathbf{x_*}\right\}}_\text{2nd Order}
\end{aligned}
$$

---

### Analytical Moments

The posterior of this distribution is non-Gaussian because we have to propagate a probability distribution through a non-linear kernel function. So this integral becomes intractable. We can compute the analytical Gaussian approximation by only computing the mean and the variance of the 

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

---

### Taylor Approximation

We will approximate our mean and variance function via a Taylor Expansion. First the mean function:

$$
\begin{aligned}
\mathbf{z}_\mu =
\mu_\text{GP}(\mathbf{x_*})=
\mu_\text{GP}(\mu_\mathbf{x_*}) +
\nabla \mu_\text{GP}\bigg\vert_{\mathbf{x}_* = \mu_\mathbf{x}}
(\mathbf{x}_* - \mu_\mathbf{x_*})
+ \mathcal{O} (\mathbf{x_*}^2)
\end{aligned}
$$

and then the variance function:

$$
\begin{aligned}
\mathbf{z}_\sigma =
\nu^2_\text{GP}(\mathbf{x_*})=
\nu^2_\text{GP}(\mu_\mathbf{x_*}) +
\nabla \nu^2_\text{GP}\bigg\vert_{\mathbf{x}_* = \mu_\mathbf{x}}
(\mathbf{x}_* - \mu_\mathbf{x_*})
+ \mathcal{O} (\mathbf{x_*}^2)
\end{aligned}
$$

---

#### Linearized Predictive Mean and Variance

$$
\begin{aligned}
m(\mu_\mathbf{x_*}, \Sigma_\mathbf{x_*})
&=
\mu_\text{GP}(\mu_\mathbf{x_*})\\
v(\mu_\mathbf{x_*}, \Sigma_\mathbf{x_*})
&= \nu^2_\text{GP}(\mu_{x_*}) +
\nabla_\mathbf{x_*} \mu_\text{GP}(\mu_{x_*})^\top
\Sigma_{x_*}
\nabla_\mathbf{x_*} \mu_\text{GP}(\mu_{x_*}) +
\frac{1}{2} \text{Tr}\left\{ \frac{\partial^2 \nu^2(\mu_{x_*})}{\partial x_* \partial x_*^\top}  \Sigma_{x_*}\right\}
\end{aligned}
$$

where $\nabla_x$ is the gradient of the function $f(\mu_x)$ w.r.t. $x$ and $\nabla_x^2 f(\mu_x)$ is the second derivative (the Hessian) of the function $f(\mu_x)$ w.r.t. $x$. This is a second-order approximation which has that expensive Hessian term. There have have been studies that have shown that that term tends to be neglible in practice and a first-order approximation is typically enough. 

Practically speaking, this leaves us with the following predictive mean and variance functions:

$$
\begin{aligned}
\mu_\text{GP}(\mathbf{x_*}) &= k(\mathbf{x_*}) \, \mathbf{K}_{GP}^{-1}y=k(\mathbf{x_*}) \, \alpha  \\
\nu_{GP}^2(\mathbf{x_*}) &= \sigma_y^2 + {\color{red}{\nabla_{\mu_\text{GP}}\,\Sigma_\mathbf{x_*} \,\nabla_{\mu_\text{GP}}^\top} }+ k_{**}- {\bf k}_* ({\bf K}+\sigma_y^2 \mathbf{I}_N )^{-1} {\bf k}_{*}^{\top}
\end{aligned}
$$

As seen above, the only extra term we need to include is the derivative of the mean function that is present in the predictive variance term.


## Conditional Gaussian Distributions

### I: Additive Noise Model ($x,f$)

This is the noise
$$
\begin{bmatrix}
    x \\
    y
    \end{bmatrix}
    \sim \mathcal{N} \left( 
    \begin{bmatrix}
    \mu_{x} \\ 
    \mu_{y} 
    \end{bmatrix}, 
    \begin{bmatrix}
    \Sigma_x & C \\
    C^\top & \Pi
    \end{bmatrix}
    \right)
$$
where
$$
\begin{aligned}
\mu_y &= f(\mu_x) \\
\Pi &= \nabla_x f(\mu_x) \: \Sigma_x \: \nabla_x f(\mu_x)^\top + \nu^2(x) \\
C &= \Sigma_x \: \nabla_x^\top f(\mu_x)
\end{aligned}
$$
So if we want to make predictions with our new model, we will have the final equation as:
$$
\begin{aligned}
f &\sim \mathcal{N}(f|\mu_{GP}, \nu^2_{GP}) \\
    \mu_{GP} &= K_{*} K_{GP}^{-1}y=K_{*} \alpha \\
    \nu^2_{GP} &= K_{**} - K_{*} K_{GP}^{-1}K_{*}^{\top} + \tilde{\Sigma}_x
\end{aligned}
$$
where $\tilde{\Sigma}_x = \nabla_x \mu_{GP} \Sigma_x \nabla \mu_{GP}^\top$.



##### Other GP Methods

We can extend this method to other GP algorithms including sparse GP models. The only thing that changes are the original $\mu_{GP}$ and $\nu^2_{GP}$ equations. In a sparse GP we have the following predictive functions
$$
\begin{aligned}
    \mu_{SGP} &= K_{*z}K_{zz}^{-1}m \\
    \nu^2_{SGP} &= K_{**} 
    - K_{*z}\left[ K_{zz}^{-1} - K_{zz}^{-1}SK_{zz}^{-1} \right]K_{*z}^{\top}
\end{aligned}
$$
So the new predictive functions will be:
$$
\begin{aligned}
    \mu_{SGP} &= K_{*z}K_{zz}^{-1}m \\
    \nu^2_{SGP} &= K_{**} 
    - K_{*z}\left[ K_{zz}^{-1} - K_{zz}^{-1}SK_{zz}^{-1} \right]K_{*z}^{\top} 
    + \tilde{\Sigma}_x
\end{aligned}
$$
As shown above, this is a fairly extensible method that offers a cheap improved predictive variance estimates on an already trained GP model. Some future work could be evaluating how other GP models, e.g. Sparse Spectrum GP, Multi-Output GPs, e.t.c.


---

## Literature

* Gaussian Process Priors with Uncertain Inputs: Multiple-Step-Ahead Prediction - Girard et. al. (2002) - Technical Report
  > Does the derivation for taking the expectation and variance for the  Taylor series expansion of the predictive mean and variance. 
* Expectation Propagation in Gaussian Process Dynamical Systems: Extended Version - Deisenroth & Mohamed (2012) - NeuRIPS
  > First time the moment matching **and** linearized version appears in the GP literature.
* Learning with Uncertainty-Gaussian Processes and Relevance Vector Machines - Candela (2004) - Thesis
  > Full law of iterated expectations and conditional variance.
* Gaussian Process Training with Input Noise - McHutchon & Rasmussen et. al. (2012) - NeuRIPS 
  > Used the same logic but instead of just approximated the posterior, they also applied this to the model which resulted in an iterative procedure.
* Multi-class Gaussian Process Classification with Noisy Inputs - Villacampa-Calvo et. al. (2020) - [axriv]()
  > Applied the first order approximation using the Taylor expansion for a classification problem. Compared this to the variational inference.

---

## Supplementary


---



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

---


### Error Propagation


#### Taylor Series Expansion

> A Taylor series is representation of a function as an infinite sum of terms that are calculated from the values of the functions derivatives at a single point - Wiki

Often times we come across functions that are very difficult to compute analytically. Below we have the simple first-order Taylor series approximation.

Let's take some function $f(\mathbf x)$ where $\mathbf{x} \sim \mathcal{N}(\mu_\mathbf{x}, \Sigma_\mathbf{x})$ described by a mean $\mu_\mathbf{x}$ and covariance $\Sigma_\mathbf{x}$. The Taylor series expansion around the function $f(\mathbf x)$ is:

$$\mathbf z = f(\mathbf x) \approx f(\mu_{\mathbf x}) +   \frac{\partial f}{\partial \mathbf x} \bigg\vert_{\mathbf{x} = \mu_\mathbf{x}}\left(  \mathbf x - \mu_{\mathbf x} \right) $$

## Law of Error Propagation


This results in a mean and error covariance of the new distribution $\mathbf z$ defined by:

$$\mu_{\mathbf z} = f(\mu_{\mathbf x})$$
$$\Sigma_\mathbf{z} = \nabla_\mathbf{x} f(\mu_{\mathbf x}) \; \Sigma_\mathbf{x} \; \nabla_\mathbf{x} f(\mu_{\mathbf x})^{\top}$$


#### <font color="red">Proof:</font> Mean Function

Given the mean function:

$$\mathbb{E}[\mathbf{x}] = \frac{1}{N} \sum_{i=1} x_i$$

We can simply apply this to the first-order Taylor series function.

$$
\begin{aligned}
\mu_\mathbf{z} &= 
\mathbb{E}_{\mathbf{x}} \left[  f(\mu_{\mathbf x}) +   \frac{\partial f}{\partial \mathbf x} \bigg\vert_{\mathbf{x} = \mu_\mathbf{x}}\left(  \mathbf x - \mu_{\mathbf x} \right) \right] \\
&= \mathbb{E}_{\mathbf{x}} \left[  f(\mu_{\mathbf x}) \right] +   \mathbb{E}_{\mathbf{x}} \left[  \frac{\partial f}{\partial \mathbf x} \bigg\vert_{\mathbf{x} = \mu_\mathbf{x}}\left(  \mathbf x - \mu_{\mathbf x} \right) \right] \\
&= f(\mu_{\mathbf x}) + 
\mathbb{E}_{\mathbf{x}} \left[  \frac{\partial f}{\partial \mathbf x} \bigg\vert_{\mathbf{x} = \mu_\mathbf{x}}  \mathbf x  \right]- \mathbb{E}_{\mathbf{x}} \left[ \frac{\partial f}{\partial \mathbf x} \bigg\vert_{\mathbf{x} = \mu_\mathbf{x}}\mu_{\mathbf x} \right] \\
&= f(\mu_{\mathbf x}) +
 \frac{\partial f}{\partial \mathbf x} \bigg\vert_{\mathbf{x} = \mu_\mathbf{x}}  \mu_\mathbf{x} -  \frac{\partial f}{\partial \mathbf x} \bigg\vert_{\mathbf{x} = \mu_\mathbf{x}}\mu_{\mathbf x}  \\
&= f(\mu_{\mathbf x}) \\
\end{aligned}
$$


#### <font color="red">Proof:</font> Variance Function

Given the variance function 

$$\mathbb{V}[\mathbf{x}] = \mathbb{E}\left[ \mathbf{x} - \mu_\mathbf{x} \right]^2$$

$$
\begin{aligned}
\sigma_\mathbf{z}^2
&=
\mathbb{E} \left[ f(\mu_\mathbf{x}) - \frac{\partial f}{\partial \mathbf{x}} \bigg\vert_{\mathbf{x}=\mu_\mathbf{x}} (\mathbf{x} - \mu_\mathbf{x}) - \mu_\mathbf{x} \right] \\
&=
\mathbb{E} \left[ \frac{\partial f}{\partial \mathbf{x}} \bigg\vert_{\mathbf{x}=\mu_\mathbf{x}}  (\mathbf{x} - \mu_\mathbf{x})\right]^2 \\
&=
\left( \frac{\partial f}{\partial \mathbf{x}} \bigg\vert_{\mathbf{x}=\mu_\mathbf{x}} \right)^2 \mathbb{E}\left[  \mathbf{x} - \mu_\mathbf{x}\right]^2\\
&= \left( \frac{\partial f}{\partial \mathbf{x}} \bigg\vert_{\mathbf{x}=\mu_\mathbf{x}} \right)^2 \Sigma_\mathbf{x}
\end{aligned}
$$

I've linked a nice tutorial for propagating variances below if you would like to go through the derivations yourself.

---

#### Resources

* Essence of Calculus, Chapter 11 | Taylor Series - 3Blue1Brown - [youtube](https://youtu.be/3d6DsjIBzJ4)
* Introduction to Error Propagation: Derivation, Meaning and Examples - [PDF](http://srl.informatik.uni-freiburg.de/papers/arrasTR98.pdf)
* Statistical uncertainty and error propagation - Vermeer - [PDF](https://users.aalto.fi/~mvermeer/uncertainty.pdf)
