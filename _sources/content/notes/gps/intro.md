# Gaussian Processes

---
## Objective

We have a set of sparsely distributed observations from satellite altimetry data. For each observation, there is an associated latitude and longitude spatial coordinate as well as a temporal coordinate. The objective is to interpolate the missing observations for the remaining spatio-temporal coordinates. 

---
## Quantities of Interest

**Inputs**: We assume that the inputs are a vector of coordinates: latitude, longitude and time, i.e. $\mathbf{x} = [\text{lat, lon, time}]$. So a single data point is a 3-dimensional vector, $\mathbf{x}\in \mathbb{R}^{D_\phi}$. It is important to note that we are free to do any coordinate transform that we want in order to better represent the data. For example, the temporal coordinates can be converted to spherical coordinates to represent the curvature of the earth. Another example is to convert the temporal coordinates into cyclic coordinates where each hour, day, month, year, etc are converted into a sine and cosine. We will see later that this can be encoded within the kernel function for the Gaussian process which will allow us to capture the assumed dynamics. However, many times physical knowledge of your system can be encoded a priori which can lead to better results.

**Outputs**: The outputs are a vector of quantities of interest. For example, we could have a variable which describes the state of the ocean such as sea surface height (SSH) and or sea surface temperature (SST). These variables are then stacked together which gives us a p-dimensional vector, $\mathbf{y} \in \mathbb{R}^{D_p}$.  




---
## Data 

We assume that we have some set of input and output data points, $\mathcal{D} = \left\{ \mathbf{x}_n, \mathbf{y}_n \right\}_{n=1}^N$. This dataset, $\mathcal{D}$, will be used for training to find the parameters of interest, $\boldsymbol{\theta}$.



---
## Gaussian Processes





---

## Problem Setting

We are interested in the regression problem where we have some quantity of interest, $\mathbf{y}$, given some inputs, $\mathbf{x}$. We also assume that there exists some function, $\boldsymbol{f}$, parameterized by, $\boldsymbol{\theta}$, that provides us with a mapping between the inputs, $\mathbf{x}$, and the outputs, $\mathbf{y}$. And lastly, we assume that it is corrupted by some identically independently distributed noise, i.e. Gaussian noise. This can be written as:

$$
\mathbf{y}_n = \boldsymbol{f}(\mathbf{x}_n; \boldsymbol{\theta}) + \boldsymbol{\epsilon}_n, \hspace{10mm} \boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2)
$$

We are interested in finding a distribution over the functions, $\boldsymbol{f}$, that can explain the data, $\{\mathbf{x},\mathbf{y}\}$. In other words, we are not interested in the best set of parameters, $\boldsymbol{\theta}$ that give us the best fit. Instead we want the best set of parameters, $\boldsymbol{\theta}$, which will give us a distribution of functions, $\boldsymbol{f} \sim P$ that could possibly describe the data.




---
## Model Assumptions


We follow the Bayesian formulation but in functional space. Bayes formula can be written as follows:

$$
p(\boldsymbol{f}(\cdot) | \mathbf{X,Y}) = \frac{p(\mathbf{y}|\boldsymbol{f}(\cdot))\;p(\boldsymbol{f}(\cdot))}{p(\mathbf{Y}|\mathbf{X})}
$$

**Prior**

The prior, $p(\boldsymbol{f}(\cdot))$ is a Gaussian process prior which is specified by its mean function, $\boldsymbol{m}$, and covariance function, $\boldsymbol{k}$. So we have

$$
\boldsymbol{f}(\mathbf{x}) \sim \mathcal{GP}\left(\boldsymbol{f} | \boldsymbol{m}_\psi (\mathbf{x}), \boldsymbol{k}_\phi(\mathbf{x},\mathbf{x}')\right)
$$

**Prior Parameters**

The mean function, $\boldsymbol{m}$, is a mapping from the coordinates, $\mathbf{x}$, to the coordinates of interest, $\mathbf{y}$. It represents our prior knowledge of the relationship. This is often assumed to be zero if we don't have any prior knowledge. However, it can be a parameterized by some hyper-parameters, $\boldsymbol{\psi}$, which can also be learned through the Gaussian process regression algorithm. The kernel function, $\boldsymbol{k} : \mathbb{R}^{D_\phi} \times \mathbb{R}^{D_\phi} \rightarrow \mathbb{R}^{}$ is a mapping representing the correlations between all of the inputs. This is also has some set of parameters, $\boldsymbol{\phi}$, which are very important. It represents the correlations is the mean function and $\mathbf{K}$ is the kernel matrix.


**Likelihood**

This is the noise model which is assumed to be a Gaussian. 

$$
p(\mathbf{y}|\boldsymbol{f}(\mathbf{X})) = \mathcal{N}(\boldsymbol{f}(\mathbf{X}), \sigma^2 \mathbf{I})
$$

The advantage of this assumption is that it allows us to have keep everything Gaussian because the both the prior and the likelihood are Gaussian distributed. However, many times there is no reason to believe that the likelihood is Gaussian. For example we could have a noise model which is dependent upon the observations or we could have a classification or Log-Cox process scenario whereby we would need a Bernoulli or Poisson distribution respectively. Nevertheless, non-Gaussian likelihoods are out of scope for this work.


**Marginal Likelihood**

The marginal likelihood is typically the most difficult quantity to calculate.

$$
p(\mathbf{y}|\mathbf{X},\boldsymbol{\theta}) = \int_{\boldsymbol{f}}p(\mathbf{y}|\boldsymbol{f})p(\boldsymbol{f}|\mathbf{X},\boldsymbol{\theta}) d\boldsymbol{f}
$$


Because the prior, likelihood are Gaussian, we can utilize the conjugacy property which ensures that the marginal likelihood is also Gaussian distributed. So this integral becomes simple because we can calculate this quantity analytically. The formula is given by:

$$
p(\mathbf{y}|\mathbf{X},\boldsymbol{\theta}) = \mathcal{N}\left(\mathbf{y}|\boldsymbol{m}(\mathbf{X}), \mathbf{K}_{\mathbf{XX}} + \sigma^2\mathbf{I}\right) 
$$

**Posterior**

The posterior of a Gaussian Process is also a Gaussian process which is normally distributed given predictive mean and predictive covariance.

$$
p(\boldsymbol{f}(\mathbf{x})|\mathcal{D}) = \mathcal{N}\left(\boldsymbol{\mu}_{\mathcal{GP}}(\mathbf{x}), \boldsymbol{\sigma}^2_\mathcal{GP}(\mathbf{x}, \mathbf{x}')\right)
$$

where we have the analytical formulas for the predictive mean and covariance:

$$
\begin{aligned}
\boldsymbol{\mu}_{\mathcal{GP}}(\mathbf{x}) &= \boldsymbol{m}(\mathbf{x}) + \boldsymbol{k}_\mathbf{X}(\mathbf{x}) \boldsymbol{\alpha} \\
\boldsymbol{\sigma}^2_\mathcal{GP}(\mathbf{x}, \mathbf{x}') &= \boldsymbol{k}(\mathbf{x}, \mathbf{x}') + \boldsymbol{k}_\mathbf{X}(\mathbf{x})\left( \mathbf{K}_{\mathbf{XX}} + \sigma^2\mathbf{I} \right)^{-1}\boldsymbol{k}_\mathbf{X}(\mathbf{x})^\top
\end{aligned}
$$


---
**Predictive Density**

$$
p(\boldsymbol{f}_*|\boldsymbol{f})
$$

Because the prior, likelihood, and the marginal likelihood are all Gaussian, we have a predictive density that is characterized as a multivariate Gaussian distribution.

$$
p(\boldsymbol{f}(\mathbf{x}_*)|\mathcal{D}) = \mathcal{N}\left(\boldsymbol{\mu}_{\mathcal{GP}}(\mathbf{x}_*), \boldsymbol{\sigma}^2_\mathcal{GP}(\mathbf{x}_*, \mathbf{x}_{*}')\right)
$$

where we have the predictive mean and covariance:

$$
\begin{aligned}
\boldsymbol{\mu}_{\mathcal{GP}}(\mathbf{x}_*) &= \boldsymbol{m}(\mathbf{x}_*) + \boldsymbol{k}_*(\mathbf{x}_*) \boldsymbol{\alpha} \\
\boldsymbol{\sigma}^2_\mathcal{GP}(\mathbf{x}_*, \mathbf{x}_{*}') &= \boldsymbol{k}(\mathbf{x}_*, \mathbf{x}_{*}') + \boldsymbol{k}_\mathbf{X}(\mathbf{x}_*)\left( \mathbf{K}_{\mathbf{XX}} + \sigma^2\mathbf{I} \right)^{-1}\boldsymbol{k}_\mathbf{X}(\mathbf{x}_*)^\top
\end{aligned}
$$

where $\boldsymbol{\alpha} = \left( \mathbf{K}_{\mathbf{XX}} + \sigma^2\mathbf{I} \right)^{-1}(\mathbf{y} - \boldsymbol{m}(\mathbf{X}))$ is a fixed parameter that can be trained via some inference method, e.g. Maximum Likelihood, Maximum A Posteriori, Variational Inference, MCMC, etc.

---
## Training

As is very typically in the Bayesian formulation, we can maximize the m
$$
\boldsymbol{\theta}^* = \argmin_{\boldsymbol{\theta}} - \mathcal{L}(\boldsymbol{\theta})
$$

$$
\mathcal{L}(\boldsymbol{\theta}) := \log p(\mathbf{y}|\mathbf{X}, \boldsymbol{\theta})
$$


### Exact GP Inference

$$
\log p(\mathbf{y}|\mathbf{X}, \boldsymbol{\theta}) = - \frac{N}{2} \log 2 \pi - \frac{1}{2} \log |\mathbf{K}_{\mathbf{XX}} + \sigma^2\mathbf{I}| - \frac{1}{2} (\mathbf{y} - \boldsymbol{m}(\mathbf{X}))^\top\left( \mathbf{K}_{\mathbf{XX}} + \sigma^2\mathbf{I} \right)^{-1}(\mathbf{y} - \boldsymbol{m}(\mathbf{X}))
$$

We introduce some new notation to simplify the equations a little bit.

$$
\mathbf{K}_{\boldsymbol{\phi}} := \mathbf{K_{XX}} + \sigma^2\mathbf{I}
$$

$$
\bar{\mathbf{Y}}_{\boldsymbol{\psi}} := \mathbf{Y} - \boldsymbol{m}(\mathbf{X};\boldsymbol{\psi})
$$

We can rewrite the cost function to reflect the new notation.

$$
\log p(\mathbf{y}|\mathbf{X}, \boldsymbol{\theta}) = - \frac{N}{2} \log 2 \pi - \frac{1}{2} \log |\mathbf{K}_{\boldsymbol{\phi}}| - \frac{1}{2} \bar{\mathbf{Y}}_{\boldsymbol{\psi}}^\top\;\mathbf{K}_{\boldsymbol{\phi}}^{-1}\;\bar{\mathbf{Y}}_{\boldsymbol{\psi}}
$$

where the parameters are the hyper-parameters of the mean function, the noise likelihood and the kernel function respectively. $\boldsymbol{\theta} = \left\{ \boldsymbol{\psi}, \sigma, \boldsymbol{\phi} \right\}$.


### Optimal Solution


$$
\boldsymbol{\alpha} := (\mathbf{K_{XX}} + \sigma^2\mathbf{I})^{-1}(\mathbf{Y} - \boldsymbol{m}(\mathbf{X}))
$$

---
## Bottleneck

**Training**

$$
\mathcal{O}(N^3)
$$

**Testing**

$$
\mathcal{O}(N^2)
$$

---
### Predictive Uncertainty

---
### Conditional Sampling





---
## Scaling






### SKI Inference

$$
\log p(\mathbf{y}|\mathbf{X},\boldsymbol{\theta}) \approx - \frac{1}{2} \log \left|\det \tilde{\mathbf{K}}_{\mathbf{XX}} + \sigma^2 \mathbf{I}\right| - \frac{1}{2}(\mathbf{y} - \boldsymbol{m}(\mathbf{X}))\left( \tilde{\mathbf{K}}_{\mathbf{XX}} + \sigma^2 \mathbf{I} \right)^{-1} -\frac{N}{2} \log 2\pi
$$


### Derivations

#### Predictive Mean

$$
\begin{aligned}
\boldsymbol{\mu}_{\text{KISS-GP}}(\mathbf{x}_*) &= \boldsymbol{w}(\mathbf{x}_*)^\top \mathbf{K_{UU}} \mathbf{}
\end{aligned}
$$





---
**Training**

* (Naive) Exact GP - $\mathcal{O}(N^3)$
* Conjugate Gradient - $\mathcal{O}(N^2)$
* KISS-GP - $\approx\mathcal{O}(N)$
* Inducing Points - $\mathcal{O}(NM^3)$
* Variational Stochastic - $\mathcal{O}(M^3)$


**Predictions**

* (Naive) Exact GP - $\mathcal{O}(N^2)$
* Conjugate Gradient - $\mathcal{O}(N)$
* KISS-GP - $\approx\mathcal{O}(N)$
* Inducing Points - $\mathcal{O}(NM^2)$
* Variational Stochastic - $\mathcal{O}(M^2)$




## Regression

Let's assume we have $N$ input data points with $D$ dimensions, $\mathbf{X} = \{\mathbf{x}_i\}^N_{i=1} \in \mathbb{R}^{N \times D}$ and noisy outputs, $\mathbf{y} = \{ y_i \}^{N}_{i=1} \in \mathbb{R}^{N}$. We want to compute the predictive distribution of $y_*$ at a test location $\mathbf{x}_*$. So:

$$
y_i = f(\mathbf{x}_i) + \epsilon_i
$$

where $f$ is an unknown latent function that is corrupted by Gaussian observation noise $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$. 


### Gaussian Process

A Gaussian process is a probability distribution over functions. It places a non-parametric Bayesian model which places a GP prior over a latent function $f$ as

$$
f(\mathbf{x}) \sim \mathcal{GP}\left( m(\mathbf{x}), k(\mathbf{x},\mathbf{x}') \right).
$$

where we see it is characterized by a mean function, $m(\mathbf{x})$ and kernel function $k(\mathbf{x,x}')$:

$$
\begin{aligned}
m(\mathbf{x}) &= \mathbb{E}[f(\mathbf{x})] \\
k(\mathbf{x,x}') &= \mathbb{E}\left[ (f(\mathbf{x}) - m(\mathbf{x}))(f(\mathbf{x}') - m(\mathbf{x}))  \right]
\end{aligned}
$$


 The mean function $m(\mathbf{x})$ is typically zero (for easier computations) and the kernel function typically characterizes the smoothness, scale and ... of the GP. Given the training data $\mathcal{D}=\{\mathbf{X},\mathbf{y}\}$,



mean function GP prior, $m_\mathcal{GP}$, (typically zero) and a covariance function $k(\mathbf{x}, \mathbf{x}')$ on $f$. 

**Prior**

**Likelihood** A GP regression model assumes that the outputs can be modeled as 

$$
p(\mathbf{y}|f, \mathbf{X}) \sim \mathcal{N}(y|f, \sigma_y^2\mathbf{I})
$$

**Posterior**




---

We can obtain a marginal likelihood (model evidence) since any finite set of GPs follows a multivariate Gaussian distribution

$$
p(\mathbf{y}|\theta) = \int p(\mathbf{y}|f)p(f)df = \mathcal{N}(\mathbf{y}|m_\phi, \mathbf{K}_{GP})
$$

In the simple model (conjugate case due to the Gaussian likelihood), the posterior over $f, p(f|y)$ can be computed analytically. 

$$
p(\mathbf{y}|\theta) = \mathcal{N}(\mathbf{y}|m_\phi, \mathbf{K}_{GP})
$$

where $\mathbf{K}_{GP}=\mathbf{K}_{ff}+ \sigma^2 \mathbf{I}$ and $\theta$ comprises of the hyperparameters found on the mean function and covariance function. We would then maximize the hyperparameters $\theta$ via, log marginal-likelihood (see below).


The prediction of $y_*$ for a test point $\mathbf{x}_*$ is what we're really interested in. We can consider the joint distribution of our training data, $\mathbf{y}$ and test data, $y_*$:

$$
\begin{bmatrix}
\mathbf{y} \\
y_*
\end{bmatrix} \sim
\mathcal{N}\left(
\begin{bmatrix} 
    \mathbf{K}_{GP} & k_*^\top\\
    k_* & k_{**}+\sigma^2
\end{bmatrix}   \right)
$$

$k_{*}=\left[ k(\mathbf{x}_1, \mathbf{x}_*), \ldots, k(\mathbf{x}_N, \mathbf{x}_*)  \right]^\top$ is the projection kernel and $k_{**}=k(\mathbf{x}_*, \mathbf{x}_*)$ is the test kernel. By way of normally distributed variables, we can obtain the predictive distribution of $y_*$ conditioned on the training data $\mathbf{y}$:

$$
p(y_*|\mathbf{y}) = \mathcal{N}(\mu_\mathcal{GP}, \sigma^2_\mathcal{GP})
$$


and subsequently closed-form predictive mean and variance equations:

$$
\begin{aligned}
\mu_\mathcal{GP}(\mathbf{x}_*) &= k_{*}^\top \mathbf{K}_{GP}^{-1} \mathbf{y} = k_{*}^\top \alpha  \\
\sigma^2_\mathcal{GP}(\mathbf{x}_*) &= \sigma^2 + k_{**} - k_{*}^\top \mathbf{K}_{GP}^{-1}k_{*}
\end{aligned}
$$


### Inference

The covariance function $k(\mathbf{x}, \mathbf{x}
)$ depends on hyperparameters are usually learned by maximizing the log marginal likelihood.

$$
\theta^* = \underset{\theta}{\text{argmax}} \log p(\mathbf{y}|\mathbf{X},\theta) = \log \mathcal{N}(\mathbf{y}; 0, \mathbf{K}_\mathcal{GP})
$$

Using the log pdf of a multivariate distribution, we have

$$
\begin{aligned}
\log \mathcal{N}(\mathbf{y}; 0, \mathbf{K}_\mathcal{GP}) &= - \frac{1}{2} \mathbf{y}^\top \mathbf{K}_\mathcal{GP}^{-1}\mathbf{y} - \frac{1}{2} \log \left| \mathbf{K}_\mathcal{GP} \right| - \frac{N}{2} \log 2 \pi
\end{aligned}
$$

This optimization is done by differentiating the above equation with respect to the hyperparameters. This maximization automatically embodies Occam's razor as it is a trade-off between the model complexity and overfitting. Typically we do the Cholesky decomposition to efficiently calculate the inversion and determinant measures.

$$
\begin{aligned}
\log p(\mathbf{y}) &= - \frac{1}{2} \mathbf{y}^\top \mathbf{K}_\mathcal{GP}^{-1}\mathbf{y} - \frac{1}{2} \log \left| \mathbf{K}_\mathcal{GP} \right| - \frac{N}{2} \log 2 \pi \\
&= -\frac{1}{2} \||\mathbf{L}^{-1}\mathbf{y}||^2  - \sum_i\log \mathbf{L}_{ii} - \frac{N}{2} \log 2 \pi
\end{aligned}
$$

where the $\mathbf{L}$ is the Cholesky decomposition $\mathbf{L}= \text{chol}(\mathbf{K}_\mathcal{GP})$. This gives us a computational complexity of $\mathcal{O}(N^3 + N^2 + N)$ which is overal $\mathcal{O}(N^3)$. So this GPR method is really only suited for \~2K-5K problems maximum.

---
## Kernel Functions


**Composition Kernels**. Kernels can be combined using sums and products to obtain more expressive formations. Additive kernels (**cite**: Duvenaud) 

**Input Warping**. This is also the choice for many kernel methods where another model is used instead.


---
## Strengths and Limitations


### Limitations

```{epigraph}
“It is important to keep in mind that Gaussian processes are not appropriate priors for all problems”.

--Neal, 1998
```

It is important to note that although the GP algorithm is one of the most trusted and reliable algorithms, it is not always the best algorithm to use for all problems. Below we mention a few drawbacks that the standard GP algorithm has along with some of the standard approaches to overcoming these drawbacks.

**Gaussian Marginals**: GPs have problems modeling heavy-tailed, asymmetric or multi-modal marginal distributions. There are some methods that change the likelihood so that it is heavy tailed~\citep{GPTSTUDENT2011,GPTSTUDENT2014} but this would remove the conjugacy of the likelihood term which would incur difficulties during fitting. Deep GPs and latent covariate models are an improvement to this limitation. A very popular approach is to construct a fully Bayesian model. This entails hyperpriors over the kernel parameters and Monte carlo sampling methods such as Gibbs sampling~\citep{GPGIBBS08}, slice sampling~\citep{GPSLICE2010}, Hamiltonian Monte Carlo~\citep{GPHMC2018}, and Sequential Monte Carlo~\citep{GPSMC15}. These techniques will capture more complex distributions. With the advent of better software~\citep{PYMC16,NUMPYRO2019} and more advanced sampling techniques like a differentiable iterative NUTS implementation~\citep{NUMPYRO2019}, the usefulness of MC schemes is resurfacing.

**Limited Number of Moments**. This is related to the previous limitation: the idea that an entire function can be captured in terms of two moments: a mean and a covariance. There are some relationships which are difficult to capture without an adequate description, e.g. discontinuities~\citep{Neal96} and non-stationary processes, and thus is a limitation of the GP priors we choose. The advent of warping the inputs or outputs of a GP has becoming a very popular technique to deal with the limited expressivity of kernels. Input warping is popular in methods such as deep kernel learning whereby a Neural network is used to capture the features and are used as inputs to the kernel function output warping is common in chained~\citep{GPCHAINED2016} and heteroscedastic methods where the function output is warped by another GP to capture the noise model of the data. Deep Gaussian processes~\citep{Damianou2015} can be thought of input and output warping methods due the multi-layer composition of function inputs and outputs.

**Linearity of Predictive Mean**. The predictive mean of a GP is linear to the observations, i.e. $\mu_{GP}=\mathbf{K}\alpha$. This essentially is a smoother which can be very powerful but also will miss key features. If there is some complex structured embedded within the dataset, then a GP model can never really capture this irregardless of the covariance function found.

**Predictive Covariance**. The GP predictive variance is a function of the training inputs and it is independent of the observed inputs. This is important if the input data has some information which could be used to help determine the regions of uncertainty, e.g. the gradient. An example would be data on a spatial grid whereby some regions points would have more certainty than others which could be obtained by knowing the input location and not necessarily the expected output.



## Conjugate Case


---
### Inference

After specifying the prior distributions, we need to infer the posterior distributions of the parameters $\theta,\phi,f$.


We represent this as the 


We can optimize this function using the negative log-likelihood of $\mathbf{y}$:

$$
\mathcal{L}(\Theta) = \frac{1}{2} \left| \mathbf{K}_\mathcal{GP}+ \sigma^2 \mathbf{I} \right| + \frac{1}{2}\mathbf{y}^\top
$$


## Non-Conjugate Case


