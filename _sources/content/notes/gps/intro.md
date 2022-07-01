
---
## Objective

We have a set of sparsely distributed observations from satellite altimetry data. For each observation, there is an associated latitude and longitude spatial coordinate as well as a temporal coordinate. The objective is to interpolate the missing observations for the remaining spatio-temporal coordinates. 

---
## Quantities of Interest

**Inputs**: We assume that the inputs are a vector of coordinates: latitude, longitude and time, i.e. $\mathbf{x} = [\text{lat, lon, time}]$. So a single data point is a 3-dimensional vector, $\mathbf{x}\in \mathbb{R}^{D_\phi}$. It is important to note that we are free to do any coordinate transform that we want in order to better represent the data. For example, the temporal coordinates can be converted to spherical coordinates to represent the curvature of the earth. Another example is to convert the temporal coordinates into cyclic coordinates where each hour, day, month, year, etc are converted into a sine and cosine. We will see later that this can be encoded within the kernel function for the Gaussian process which will allow us to capture the assumed dynamics. However, many times physical knowledge of your system can be encoded a priori which can lead to better results.

**Outputs**: The outputs are a vector of quantities of interest. For example, we could have a variable which describes the state of the ocean such as sea surface height (SSH) and or sea surface temperature (SST). These variables are then stacked together which gives us a p-dimensional vector, $\mathbf{y} \in \mathbb{R}^{D_p}$.  

---
## Representation

### Field Representation


We basically present a 'raveled' version of the state whereby we measure the entire field. This can be shown as:


$$
\mathbf{x} = \left[\text{lon}_1, \ldots, \text{lon}_{D_\text{lat}}, \text{lat}_1, \ldots, \text{lat}_{D_\text{lon}},, \text{time}_1, \ldots, \text{time}_{D_\text{time}} \right] \in \mathbb{R}^{D_\text{lat} \times D_\text{lon} \times D_\text{time}}
$$

**Example**: if we have a *full* spatial lat-lon grid of `30x30` points and `30` time steps, then the vector is `30x30x30` which is `27,000`-dimensional vector! This is compounded if we wish to calculate correlations between each of the grid points which would result in a matrix of size `27,000 x 27,000` points. As we see below, this is a very high dimensional problem.


$$
D_\mathbf{x} = [\text{lat}_1, \ldots, \text{lat}_D, \text{lon}_1, \ldots, \text{lon}_D, \text{time}_1, \ldots, \text{time}_D]
$$

And the final vector, $\mathbf{x}$, can be massive for this unrolled spatio-temporal vector. So stacking all of these together gives us a very large vector of $\mathbf{X} \in \mathbb{R}^{D_\mathbf{x}}$. Estimating the covariance between each of coordinates would results in a massive matrix, $\mathbf{C}_{XX} \in \mathbb{R}^{D_x \times D_x}$. In the above algorithm, we need to do a matrix inversion in conjunction which is very expensive. Below you have the computational complexity when considering the state, $\mathbf{x}$:

State $\mathbf{x}$: 

- computational complexity - $\mathcal{O}(D_{\mathbf{x}}^3)$
- memory $\mathcal{O}(D_{\mathbf{x}}^2)$

---
### Coordinate Representation

The coordinate representation assumes the input vector, $\boldsymbol{x}$, is a single set of coordinates.

$$
D_\phi = [\text{lat,lon,time}]
$$

If we assume we have a large number of sparsely distributed coordinate values. This gives us a large number of samples, $N$. Stacked together, we get a matrix of samples and features (coordinates), $\boldsymbol{X} \in \mathbb{R}^{N \times D_\phi}$. 

**Example**: Take the full grid from the field representation, i.e. `30x30x30=27,000`-dimensional vector. Under this representation, we would have a vector which is three times the size, i.e. `N x D = 27,000 x 3` because for every grid point, we have a lat, lon, time coordinate. However, in our specific application, we have very sparse observations which means that we will never have to have a full grid of that size in memory (or during compute). If we assume there to be only 20% of the grid observed, then we have: `0.20 * 27,000 = 5,400` and a covariance of `5,400x5,400`. This is significantly less data that the full grid. This allows us to push the upper limit of the amount of observations where we can learn the parameters. For example, if we have a budget of `20,000` points for our memory (including the covariance), then we could potentially have a grid size of `46x46x46` if we wanted an evenly distribued spatio-temporal grid, `54x54x30` for a spatially dense grid and `30x30x74` for a temporal dense grid. This would allow the methods to capture processes at a finer scale without sacrificing computability.


This operation is still very expensive however, we assume that the observations are 

State $\boldsymbol{x}$:
- computational complexity $\mathcal{O}(N^3)$
- memory $\mathcal{O}(N^2)$

As you can see, the covariance matrix is still very expensive to calculate. However, the observations are very sparse compared to the full coordinate grid, i.e. $N << D_\mathbf{x}$. 

#### Pros and Cons

**Coordinate Transformations**: We have direct access to the lat-lon-time coordinates. This gives us the flexibility to perform transformations such that this is reflected within the input representation. 3 coordinates might lack information and it might be useful to transform these coordinates into a high representation. For example, if we transform the spatial coordinates from lat-lon to spherical coordinates, it goes from 2 to 3 dimensions. If we transform the time coordinate to a cyclic coordinate that encodes the minute, hour, day, month, year, then we go from a 1D vector to a 5D vector which could potentially encode some multi-scale dynamics.

**Large Upper Limit**: The data is sparse so that enables us to see more observations in space and time. The more data seen, the better the interpolation will be.

<span style="color:green">**Complete Space**</span>: We seen the complete space (all of the grid points). In many state-space methods, it is not possibly to put the entire dataset in memory. In addition, we can query observations which are not in a fine grid.



<span style="color:red">**No Physical Models**</span>: We are mapping a set of coordinates to a physical quantity of interest. This is different than the field representation which removes the physical sense. In this case, we are using a pure interpolation / smoothing setting. Many of the methods are based in proximity, e.g. nearest neighbour calculation. There is no direct physical interpretation between the space of coordinates to the physical quantity. When we consider the state-space representation, this becomes more feasible because we assume the 


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

---
### Conjugate Gradients

Consider the following optimal solution to the GP given the best optimal hyper-parameters, $\boldsymbol{\theta} = \{ \boldsymbol{\psi, \phi}, \sigma^2 \}$:

$$
\boldsymbol{\alpha} = \mathbf{K}_{\boldsymbol{\phi}}^{-1} \bar{\mathbf{Y}}_{\boldsymbol{\psi}}
$$

We can rewrite this as a linear system.

$$
\mathbf{K}_{\boldsymbol{\phi}}\boldsymbol{\alpha} - \bar{\mathbf{Y}}_{\boldsymbol{\psi}} = \mathbf{0}
$$

We can reformulate this as a quadratic optimization problem:

$$
\boldsymbol{\alpha}^* = \argmin_{\boldsymbol{\alpha}} \boldsymbol{\alpha}^\top \mathbf{K}_{\boldsymbol{\phi}}\boldsymbol{\alpha} - \boldsymbol{\alpha}^\top \bar{\mathbf{Y}}_{\boldsymbol{\psi}}
$$

There are many efficient ways to solve this problem where one of them is the *conjugate gradient* operation. This is an iterative algorithm that recovers the exact solution after $k$ iterations. But we can recover an approximate solution after $\tilde{k}$ iterations where $\tilde{k} << k$. Each iteration is $\mathcal{O}(N^2)$.

**GPyTorch: BlackBox Matrix-Matrix Gaussian Process Inference with GPU Acceleration** - Gardner et al (2018) 

---
## Structured Kernel Interpolation

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