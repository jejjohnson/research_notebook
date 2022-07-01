# Optimal Interpolation


This aims at finding the Best Linear Unbiased Estimator (BLUE).

## Problem Setting

$$
\mathbf{x} = \left[\text{lon}_1, \ldots, \text{lon}_{D_\text{lat}}, \text{lat}_1, \ldots, \text{lat}_{D_\text{lon}},, \text{time}_1, \ldots, \text{time}_{D_\text{time}} \right] \in \mathbb{R}^{D_\text{lat} \times D_\text{lon} \times D_\text{time}}
$$

For example, if we have a spatial lat-lon grid of `30x30` points and `30` time steps, then the vector is `30x30x30` which is `27,000`-dimensional vector!


---

Optimal Interpolation (OI) seems to be the golden standard for gap-filling SSH fields with missing data. Something that is very prevalent for Satellite sensors. However, the standard OI method

$$
\begin{aligned}
\tilde{\boldsymbol{\mu}}_{\mathbf{x}_{t}} &= \mathbf{x}_t + \mathbf{K}(\mathbf{y}_t - \mathbf{H}\mathbf{x}_t) \\
\tilde{\boldsymbol{\Sigma}}_{\mathbf{x}_{t}} &= (\mathbf{I} - \mathbf{KH})\boldsymbol{\Sigma}_\mathbf{x}  \\
\mathbf{K} &= \mathbf{P} \mathbf{H}\left(\mathbf{H} \mathbf{P} \mathbf{H}^\top  + \mathbf{R} \right)^{-1} \\
\end{aligned}
$$

The $\mathbf{K}$ is known as the Kalman gain. The *optimal* $\mathbf{K}$ is given by:

$$
\mathbf{K} = \boldsymbol{\Sigma}_{\mathbf{xy}} \boldsymbol{\Sigma}_{\mathbf{yy}}^{-1}
$$

where $\boldsymbol{\Sigma}_{\mathbf{yy}}$ is the covariance between the observations and $\boldsymbol{\Sigma}_{\mathbf{xy}}$ is the cross-covariance between the state and the observations.

Notice how this assumes that interpolation can be done via a linear operator, $\mathbf{H}$. The Kalman gain, $\mathbf{K}$, represents the best interpolation via this linear problem. However, this is often insufficient in many scenarios where the data does not exhibit linear dynamics. 


## Data-Driven OI

So in practice, it's often not possible

### Notation

|                                                Symbol                                                 | Meaning                                                                                                             |
| :---------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------ |
|                                          $x \in \mathbb{R}$                                           | scalar                                                                                                              |
|                                    $\mathbf{x} \in \mathbb{R}^{D}$                                    | vector                                                                                                              |
|                               $\mathbf{x} \in \mathbb{R}^{N \times D}$                                | a matrix                                                                                                            |
|                      $\mathbf{K}_{\mathbf{XX}} := \mathbf{K} \in \mathbb{R}^{D}$                      | the kernel matrix                                                                                                   |
| $\boldsymbol{k}(\mathbf{X}, \mathbf{x}) = \boldsymbol{k}_{\mathbf{X}}(\mathbf{x}) \in \mathbb{R}^{D}$ | the cross kernel matrix for some data, $\mathbf{X} \in \mathbf{R}^{N \times D}$, and some new vector, $\mathbf{x}$. |
|                                    $\mathbf{x} \in \mathbb{R}^{D}$                                    | vector                                                                                                              |

### Setting

$$
\begin{aligned}
\boldsymbol{x}(t) &= \mathcal{M}\left( \boldsymbol{x}(t-1)  \right) + \eta(t) \\
\boldsymbol{y}(t) &= \mathcal{H}(\boldsymbol{x}(t)) + \boldsymbol{\epsilon}(t)
\end{aligned}
$$


**Discrete Setting**

$$
\begin{aligned}
\mathbf{x}_{k} &= \mathbf{x}_{k-1} + \boldsymbol{\epsilon}_{\mathbf{x}}\\
\mathbf{y}_{k} &= \mathbf{Hx}_{k} + \boldsymbol{\epsilon}_{\mathbf{y}} 
\end{aligned}
$$

where:

* $\boldsymbol{\epsilon}_{\mathbf{x}} \sim \mathcal{N}(\mathbf{0}, \mathbf{P})$
* $\boldsymbol{\epsilon}_{\mathbf{y}} \sim \mathcal{N}(\mathbf{0}, \mathbf{R})$

---
### O.I. Equations

$$
\begin{aligned}
\mathbf{K} &= \mathbf{P} \mathbf{H}\left(\mathbf{H} \mathbf{P} \mathbf{H}^\top  + \mathbf{R} \right)^{-1} \\
\tilde{\boldsymbol{\mu}}_{\mathbf{x}_{t}} &= \mathbf{x}_t + \mathbf{K}(\mathbf{y}_t - \mathbf{H}\mathbf{x}_t) \\
\tilde{\boldsymbol{\Sigma}}_{\mathbf{x}_{t}} &= (\mathbf{I} - \mathbf{KH})\boldsymbol{\Sigma}_\mathbf{x}  
\end{aligned}
$$

where:

* $\mathbf{y}$ - observations
* $\mathbf{x}$ - prior input
* 
* $\mathbf{K}$ - Kalman gain
* $\mathbf{B}$ - noise covariance matrix from the 
* $\mathbf{R}$ - noise covariance matrix from the observations.

### Cost Function

Lorenc (1986) showed that OI is closely related to the 3D-Var variational data assimilation

$$
\mathcal{L}(\boldsymbol{\theta}) = (\mathbf{x} - \mathbf{x}^b)^\top \mathbf{B}^{-1} (\mathbf{x} - \mathbf{x}^b) + (\mathbf{y} - \mathbf{Hx})^\top \mathbf{R}^{-1} (\mathbf{y} - \mathbf{Hx})
$$(oi_loss)

where $\boldsymbol{\theta} = \{ \mathbf{B}, \mathbf{R}\}$

---
## In Practice

It doesn't make sense to assume that we have an affine linear transformation if indeed our model cannot be described by a linear transformation. Instead, we can try to directly learn this from data. We could also directly estimate this from the data. Let's take some coordinates, $\{\mathbf{X}_{obs}, \mathbf{Y}_{obs} \} = \{\mathbf{x}_{obs}, \mathbf{y}_{obs} \}_{i=1}^N$ which are observed. We will learn a model that best fits the observations. And then we will find some interpolation method to fill in the gaps.

**Historical Literature**:

* Bretherton et al. 1976
* McIntosh, 1990
* Le Traon et al. 1998

**Modern Literature**:

* Kernel Methods
* Kernel Interpolation


---

We can write an exact equation for the best least squares linear estimator, $\boldsymbol{f}(\mathbf{x})$:

$$
\boldsymbol{f}(\mathbf{x}^*)= \boldsymbol{\mu}_\mathbf{x} + \sum_{i=1}^{D_\mathbf{x}} \sum_{j=1}^{D_\mathbf{y}} \mathbf{A}_{ij}^{-1}\mathbf{C}_{\mathbf{x}j}(\mathbf{y}_i - \boldsymbol{\mu}_{\mathbf{x}j})
$$

where $\mathbf{A}_{ij}$ is the covariance between the observation locations:

$$
\mathbf{A}_{ij}= \langle \mathbf{x}_{obs}, \mathbf{x'}_{obs} \rangle + \langle \boldsymbol{\epsilon}, \boldsymbol{\epsilon}' \rangle
$$

and $\mathbf{C}_{\mathbf{x}j}$ is the cross covariance for the locations to be estimated and the observation locations:

$$
\mathbf{C}_{\mathbf{x}j} = \langle \mathbf{x}^*, \mathbf{x}_{obs} \rangle
$$


---
### Reformulation

We can write the full equation (in vector form) as follows:

$$
\boldsymbol{f}(\mathbf{x}^*) = \mathbf{k_{X}}(\mathbf{x}^*) (\mathbf{K_{XX}} + \sigma^2\mathbf{I})^{-1}\mathbf{Y}_{obs}
$$

where:

* $\mathbf{K_{XX}}$ - the coordinates for where we have observations, $\mathbf{Y}_{obs} \in \mathbb{R}^{N \times D}$. 
* $\mathbf{k_{X}}(\cdot)$  -  the cross-kernel for the new coordinates, $\mathbf{x}^* \in \mathbb{R}^{D}$, and the old coordinates, $\mathbf{X} \in \mathbb{R}^{N\times D}$.



We have a kernel function which is the correlation between the observation locations, $\mathbf{x}$.

$$
\mathbf{K}_{ij}= \langle \mathbf{x}_{obs}, \mathbf{x'}_{obs} \rangle
$$

where $\mathbf{X}_{obs} \in \mathbb{R}^{N \times D}$.



If we only look at the elements dependent upon the observations, $\mathbf{Y}_{obs}$, we can call this, $\boldsymbol{\alpha}$.

$$
\boldsymbol{\alpha} = (\mathbf{K_{xx}} + \sigma^2\mathbf{I})^{-1}\mathbf{Y}_{obs}
$$




where $\boldsymbol{\alpha} \in \mathbb{R}^{N \times D}$. We can solve for $\boldsymbol{\alpha}$ exactly. To find the best free parameters, $\boldsymbol{\theta} = \{ \sigma^2, \boldsymbol{\alpha} \}$, we can use cross-validation.

**Note**: This comes from kernel methods! 

---
And now we can make predictions

$$
\begin{aligned}
\boldsymbol{m}(\mathbf{x}^*) &= \boldsymbol{k}_{\mathbf{X}}(\mathbf{x}^*) \boldsymbol{\alpha} \\
\end{aligned}
$$

and we can also estimate the covariance.


$$
\begin{aligned}
\boldsymbol{\Sigma}^2(\mathbf{x}^*, \mathbf{x'}^*) &= \sigma^2 + \boldsymbol{k}(\mathbf{x}^*, \mathbf{x}^*) + \boldsymbol{k}_{\mathbf{X}}(\mathbf{x}^*) (\mathbf{K_{XX}} + \sigma^2\mathbf{I}) \boldsymbol{k}_{\mathbf{X}}(\mathbf{x}^*)^\top
\end{aligned}
$$


---
## Notation

* $\boldsymbol{k}(\mathbf{x}, \mathbf{y}): \mathcal{X}\times \mathcal{Y} \rightarrow \mathbb{R}$ - kernel function that takes in two vectors and returns a scalar value.
* $\mathbf{k}(\mathbf{x}_*): \mathcal{X} \times : \rightarrow \mathbb{R}^{D}$
* $\mathbf{K_{XX}} \in \mathbb{R}^{D_{\mathbf{x}} \times D_\mathbf{x}}$ - kernel matrix (gram matrix) which is the result of 
* $\mathbf{K_{XY}} \in \mathbb{R}^{D_{\mathbf{x}} \times D_\mathbf{y}}$ - cross kernel matrix


---
### Kernel Functions

**Linear Kernel**

$$
\boldsymbol{k}(\mathbf{x},\mathbf{y}) = \mathbf{x}^\top \mathbf{y}
$$

**Gaussian Kernel** (aka Radial Basis Function, Matern 32)

$$
\boldsymbol{k}(\mathbf{x},\mathbf{y}) = \exp \left( - \gamma||\mathbf{x} - \mathbf{y}||_2^2 \right)
$$

**Note**: This is known as the Gaussian kernel if $\gamma = \frac{1}{\sigma^2}$.

**Laplacian Kernel**

$$
\boldsymbol{k}(\mathbf{x},\mathbf{y}) = \exp \left( - \gamma||\mathbf{x} - \mathbf{y}||_1 \right)
$$

**Polynomial Kernel**

$$
\boldsymbol{k}(\mathbf{x},\mathbf{y}) = \left( \gamma\mathbf{x}^\top \mathbf{y} + c_0 \right)^d
$$

---
## Scaling

The bottleneck in the above methods is the inversion of the $\mathbf{K_{XX}}^{-1}$ matrix which is order $\mathcal{O}(N^3)$.

We can assume that our kernel matrix $\mathbf{K_{XX}}$ can be approximated via a low-rank matrix:

$$
\mathbf{K_{XX}} \approx \mathbf{K_{XZ}}\mathbf{K_{XZ}}^\top
$$

where $\mathbf{K_{XZ}} \in \mathbb{R}^{N \times r}$

Similarly, we can also write:

$$
\mathbf{K_{XX}} \approx \mathbf{K_{XZ}}\mathbf{K_{ZZ}}^{-1}\mathbf{K_{XZ}}^\top
$$

$$
\mathbf{k}_r (\mathbf{X}) = 
$$

So then we can rewrite our solution to be:

$$
\boldsymbol{\alpha} = \left( \mathbf{K_{XZ}}^\top \mathbf{K_{XZ}} + \sigma^2 \mathbf{K_{ZZ}} \right)^{\dagger} \mathbf{K_{XZ}}^\top \mathbf{Y}_{obs}
$$

Now to make predictions, we can simply write:

$$
\boldsymbol{m}(\mathbf{x}^*) = \boldsymbol{k}_{r}(\mathbf{x}^*)\boldsymbol{\alpha}
$$

This is **much** cheaper:

$$
\mathcal{O}(nm^3 + m^3) < \mathcal{O}(n^3)
$$

**Predictive Mean**

$$
\boldsymbol{m}(\mathbf{x}^*) = \boldsymbol{k}_{\mathbf{z}}\boldsymbol{\alpha}
$$

**Predictive Covariance**

$$
\boldsymbol{\Sigma}^2(\mathbf{x}^*, \mathbf{x'}^*) = \boldsymbol{k}(\mathbf{x}^*, \mathbf{x'}^*) - \boldsymbol{k}_{\mathbf{z}}(\mathbf{x}^*)^\top \mathbf{K_{ZZ}}^{-1}\boldsymbol{k}_{\mathbf{z}}(\mathbf{x}^*)^\top + \boldsymbol{k}_{\mathbf{z}}(\mathbf{x}^*) \left( \mathbf{K_{ZZ}} + \sigma^{-2} \mathbf{K_{ZX}K_{XZ}} \right) \boldsymbol{k}_{\mathbf{z}}(\mathbf{x}^*)^\top
$$

where:

* $\mathbf{K_{\mathbf{XX}}} \in \mathbb{R}^{N \times N}$
* $\mathbf{K_{\mathbf{ZZ}}} \in \mathbb{R}^{r \times r}$
* $\mathbf{K_{\mathbf{ZX}}} \in \mathbb{R}^{r \times N}$
* $\mathbf{K_{\mathbf{XZ}}} \in \mathbb{R}^{N \times r}$
* $\boldsymbol{k}_{\mathbf{Z}}(\mathbf{x}) \in \mathbb{R}^{r}$

**Resources**:

* Less is More: Nystr ̈om Computational Regularization
* Using Nystrom to Speed Up Kernel Machines

---


* Linear -> Nystrom Approximation
* Polynomial -> Nystrom / TensorSketech
* Gaussian -> Nystrom / RFF



---
## Potential Projects

* Kernel Methods - solve exactly with cross validation for the best parameters
* Gaussian Processes - solve with maximum likelihood
* Kernel Functions - explore different kernel functions and see how they work (e.g. linear, Gaussian, Laplacian, Haversine, Matern Family)
* Sparse Methods (Kernel) - use kernel approximation schemes (e.g. Nystrom, RFF)
* Sparse GP Methods - use inducing points to see how they work.


---
## Model

We have a data model that is non-parametric. It means that we fit a model that has many local fits where the number of parameters may exceed the number of data points.

$$
\mathbf{y}_i = \boldsymbol{f}(\mathbf{x}_i) + \boldsymbol{\epsilon}_i, \hspace{10mm} i=1,2, \ldots, N
$$

where:

* $\boldsymbol{f}(\cdot) $ - the regression function
* $\mathbf{x}_i$ - the sampling position, e.g. latitude, longitude, and/or time
* $\mathbf{y}_i$ - the sample
* $\boldsymbol{\epsilon}_i$ - zero mean, iid noise






---
### Kernel Methods

* Connections and Equivalences between the Nyström Method and Sparse Variational Gaussian Processes - [Arxiv](https://arxiv.org/abs/2106.01121)
* Gaussian Processes for Machine Learning - [PDF](http://www.gaussianprocess.org/gpml/chapters/RW8.pdf)
* [PyKrige](https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/examples/02_kriging3D.html)
* Spatial Analysis Made Easy with Linear Regression and Kernels - [PDF](https://arxiv.org/abs/1902.08679)
* RFFs - [blog](https://gregorygundersen.com/blog/2019/12/23/random-fourier-features/)
* GP Regression - [blog](https://gregorygundersen.com/blog/2019/06/27/gp-regression/) | [blog](https://gregorygundersen.com/blog/2019/09/12/practical-gp-regression/)
* An exact kernel framework for spatio-temporal dynamics - [arxiv](https://arxiv.org/abs/2011.06848)
* 4DVarNet - [Broadcast](https://github.com/CIA-Oceanix/4dvarnet-core/blob/james/scratches/torch_oi.py#L252-L406) | [Loops](https://github.com/ocean-data-challenges/2020a_SSH_mapping_NATL60/blob/master/src/mod_oi.py) | [Batch](https://github.com/CIA-Oceanix/4dvarnet-core/blob/james/scratches/torch_oi.py#L207)
* Spatio-Temporal VGPs - [arxiv](Spatio-Temporal Variational Gaussian Processes)
* Space-Time KErnels - [prexi](https://www.isprs.org/proceedings/xxxviii/part2/presentations/S2-3/WangJ.pdf)
* Falkon - [python](https://falkonml.github.io/falkon/examples/falkon_regression_tutorial.html)
* Kernel methods & sparse methods for computer vision - bach - [prezi](https://www.di.ens.fr/~fbach/INRIA_summer_school_2011_fbach.pdf)
* Kernel methods through the roof: handling billions of points efficiently - [arxiv](https://arxiv.org/abs/2006.10350)
* KRR Demo - [sklearn](https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_compare_gpr_krr.html#sphx-glr-auto-examples-gaussian-process-plot-compare-gpr-krr-py)


---
**Theory**

* Machine Learning with Kernel Methods - [Slides](https://members.cbio.mines-paristech.fr/~jvert/svn/kernelcourse/slides/master2017/master2017.pdf)
* Kernel Ridge Regression Course - [Class](https://github.com/djsutherland/ds3-kernels-21)


---
**References**

* Gaussian Process Machine Learning and Kriging for Groundwater Salinity Interpolation - Cui et al (2021) - [Paper](https://www.sciencedirect.com/science/article/pii/S1364815221002139)
> They use different kernels with different model outputs to try and interpolate. They found that more expressive kernels with combinations of inputs provided better models.


**Learning**

* How to Scale Up Kernel Methods to Be As Good As Deep Neural Nets - Lu et al (2015)
* On the Benefits of Large Learning Rates for Kernel Methods - Beugnot et al (2022)
* Learning Rate Annealing Can Provably Help Generalization, Even for Convex Problems - Nakkiran et al (2020)
---
**Motivation**

* Demo for Scalable Kernel Features w/ Sklearn - [Blog](https://maelfabien.github.io/machinelearning/largescale/#)

---
**Software Planning**

* [TinyGP](https://tinygp.readthedocs.io/en/stable/tutorials/quickstart.html)
* [kernellib](https://github.com/jejjohnson/kernellib/blob/master/kernellib/regression/large_scale.py)
* [jaxkern](https://github.com/IPL-UV/jaxkern)
* GPFlow Sparse GP for Spatio-Temporal Data - [Demo](https://github.com/AaltoML/spatio-temporal-GPs/blob/main/experiments/air_quality/models/m_gpflow.py)
* SKIP Example - [Demo](https://github.com/AaltoML/spatio-temporal-GPs/blob/main/experiments/air_quality/models/m_ski.py)
* LOO GPyTorch - [Code](https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/mlls/leave_one_out_pseudo_likelihood.py)
* LOO Sklearn - [Code](https://scikit-learn.org/stable/modules/cross_validation.html#leave-one-out)

---
**RFF**

* Sampling RFF Efficiently - [Blog](http://random-walks.org/content/misc/rff/rff.html#implementation) | [VFE GP - Blog](http://random-walks.org/content/gp/sparse/gp-sampling.html#training-and-posterior-sampling)


---
**Scalable Computation**

* Kernel methods through the roof: handling billions of points efficiently - Meanti et al (2020) - [arxiv](https://arxiv.org/abs/2006.10350) | [video](https://crossminds.ai/video/kernel-methods-through-the-roof-handling-billions-of-points-efficiently-606fe09af43a7f2f827bfd93/)
* Efficient Hyperparameter Tuning for Large Scale Kernel Ridge Regression - Meanti et al (2022) - [arxiv](https://arxiv.org/abs/2201.06314)
* Scalable Kernel Methods via Doubly Stochastic Gradients - Dai et al (2015)
* Making Large-Scale Nystr¨om Approximation Possible - Li et al (2010) | [Blog](https://gregorygundersen.com/blog/2019/01/17/randomized-svd/)
* Large-Scale Nyström Kernel Matrix Approximation Using Randomized SVD - Li et al (2014)
* Practical Algorithms for Latent Variable Models - Gundersen (Thesis) (2021) - [thesis](http://gregorygundersen.com/publications/gundersen2021thesis.pdf)



---
* Kernel Ridge Regression - [Falkon](https://github.com/FalkonML/falkon) | [Regression](https://falkonml.github.io/falkon/examples/falkon_regression_tutorial.html) | [Hyperparam](https://falkonml.github.io/falkon/examples/falkon_cv.html) | [Auto Hyperparam](https://falkonml.github.io/falkon/examples/hyperopt.html) | [Custom Kernel](https://falkonml.github.io/falkon/examples/custom_kernels.html) | [Custom Losses](https://falkonml.github.io/falkon/api_reference/hopt.html)
* Kernel Interpolation - [KeOps Demo](https://www.kernel-operations.io/keops/_auto_tutorials/index.html#interpolation-splines) (Numpy/PyTorch)
* Kernel Solve - [KeOps Demo](https://www.kernel-operations.io/keops/_auto_examples/index.html#numpy-api) (Numpy/PyTorch)
* Block-Matrices Reductions - [KeOps Demo](https://www.kernel-operations.io/keops/_auto_examples/pytorch/plot_grid_cluster_pytorch.html#sphx-glr-auto-examples-pytorch-plot-grid-cluster-pytorch-py) (Numpy/PyTorch) 


---
**Spatio-Temporal Kernels**

* Kernel Regression for Image Processing and Reconstruction - Takeda et al (2007) - [pdf](https://people.duke.edu/~sf59/KernelRegression_Final.pdf)
* Composite Kernels for HSI classification - Gustau (2006) - [Paper](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=XPs4yHUAAAAJ&citation_for_view=XPs4yHUAAAAJ:d1gkVwhDpl0C)


---
**Connections**

* Deep Gaussian Markov Random Fields - Siden & Lindsten (2020) - [arxiv](https://arxiv.org/abs/2002.07467)

---
**Markovian Gaussian Processes**

* Spatio-Temporal Variational Gaussian Processes - Hamelijnick et al (2021) - [arxiv](https://arxiv.org/abs/2111.01732) (JAX)
* Variational Gaussian Process State-Space Models - Frigola et al (2014) - [arxiv]()


---
**Periodic Activation Functions**

* NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis - Mildenhall et al (2020)
* Implicit Neural Representations with Periodic Activation Functions - Sitzmann et al (2020) - [arxiv](https://arxiv.org/abs/2006.09661) | [page](https://www.vincentsitzmann.com/siren/) | [eccv vid](https://www.matthewtancik.com/nerf) | [blog](https://dellaert.github.io/NeRF/)
  *  [PyTorch](https://github.com/vsitzmann/siren) | [JAX](https://github.com/KeunwooPark/siren-jax) | [Lucidrains](https://github.com/lucidrains/deep-daze)


---
## Kernel Functions