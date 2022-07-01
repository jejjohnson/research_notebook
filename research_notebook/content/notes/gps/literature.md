# Literature Review

###### tags: `interpolation`

---
## TLDR - Table of Contents

#### Objective

> We want to use modern Optimal Interpolators that are as accurate and more scalable than the standard baseline DUACs algorithm. 

---
#### Optimal Interpolation vs Gaussian Processes

> **Optimal Analysis of In Situ Data in the Western Mediterranean Using Statistics and Cross-Validation** - Brankart and Brasseur (1995)
> The authors showcase how to do OI for in-situ data via empirically calculated covariance metrics of the grid-based data. They also showcase how one could use cross-validation and sampling methods for finding the optimal parameters.


> **A Numerically Efficient Data Analysis Method with Error Map Generation** - Rixen et al (2000)
> These authors showcase the similarities between the grid-based OI method and the coordinate-based OI method. The message is that the coordinate-based method is more scalable than the standard method due to the sparsity.

> **Spatial Optimal Interpolation of Aquarius Sea Surface Salinity: Algorithms and Implementation in the North Atlantic** - Melnichenko et al (2014) 
> Probably the most modern take on OI that I could find in the related field. The equations are the same as the original formulations but I think the notation was a bit clearer to understand.

> **A Case Study Competition among Methods for Analyzing Large Spatial Data** - Heaton et al (2018)
> A very decent summary of the different methods (related more to Kriging) of how one can use scalable GPs to interpolate spatial data. They did a very good experiment section on simulated data where different groups did independent implementations.


---
#### Fancy GP Kernels

> **Automatic Model Construction with GPs** - Duvenaud et al (2014) 
> A thesis by David which has a lot of great details of how one can construct kernel functions as combinations of other kernel functions. Online version: [kernel cookbook](https://www.cs.toronto.edu/~duvenaud/cookbook/).

> **AutoGP: Exploring the Capabilities and Limitations of GP Models** - Krauth et al (2017)
> Argue that the GP is limited by the kernel function and that we should use more "non-linear" kernel functions to capture more "non-linear" processes. They investigate the use of the arc-cosine kernel function which is able to mimic the non-linearities that we see in Deep NNs. 
> **Supp**: Kernel Methods for Deep Learning - Cho et al (2009); 

> **Gaussian Process Kernels for Pattern Discovery and Extrapolation** - Wilson et al (2013)
> Something called a Spectral Mixture Kernel. 
> **Supp.**: 
> * GPatt: Fast Multidimensional Pattern Extrapolation with GPs - Wilson et al (2013); 

> **Deep Kernel Learning (DKL)** - Wilson et al (2015)
> Uses NNs for kernel functions and then this is Augmented with a GP. **Supp**: Stochastic Variational DKL - Wilson et al (2016); The promises and Pitfalls of DKL - Ober et al (2021); On Feature Collapse and DKL - Amersfoort et al (2021); Physics Informed DKL - Wang et al (2020)


---
#### Scalable Gaussian Processes

**Note**: I would recommend my guide [here](https://jejjohnson.github.io/gp_model_zoo/) for all things Gaussian process. **Note**: I haven't updated it in a while but I will soon.

> **Exact Gaussian Processes on a Million Data Points** - Wang et al (2020)
> The authors demonstrate how one can use the exact GP



---
**Structured Kernel Interpolation (SKI)**

> **Kernel Interpolation for Scalable Structured Gaussian Processses (KISS-GP)** - Wilson et al (2015)
> The original paper showcasing how one can exploit the structure of Kronecker products to computationally scale matrix calculations with a large number of inputs. This is more focused on interpolation by putting everything on a regular grid and then mapping any arbitrary coordinate to that grid.

> **Product Kernel Interpolation for Scalable Gaussian Processes (SKIP)** - Gardner et al (2018)
> This improves the above method to be able to handle a large number of dimensions.

> **Kernel Interpolation for Scalable Online Gaussian Processes (WISKI)** - Stanton et al (2021)
> The same as the above two papers exact this method allows for online learning, i.e. updating the parameters as new data is seen. They mostly applied this to the Bayesian Optimization setting.


---
**Sparse Gaussian Processes**

> **A Tutorial on Sparse Gaussian Processes and Variational Inference** - Leibfried et al (2021)
> The most up-to-date review of the methods for Sparse Variational Gaussian Processes. **Note**: We won't be using this in the paper because I feel like it is out of scope. But it is probably the most scalable method available to date. $\mathcal{O}(M^3)$

> **When Gaussian Process Meets Big Data: A Review of Scalable GPs** - Lui et al (2019)
> Probably the most complete review for all of the scalable GP methods. But it is all algorithm and no actual applications or analysis.


---
**Randomized (Fourier Features) Gaussian Processes** (<span style="color:red">**Outside the Scope**</span>)

---
**Markovian Gaussian Processes** (<span style="color:red">**Outside the Scope**</span>)

*These papers are related to the Markovian GP method. This can be thought of as a Kalman filter but applied to the Kalmn*

> **Spatio-Temporal Variational Gaussian Processes** - Hamelijnck et al (2021)
> This is the Markovian GP method scaled to spatio-temporal data. **Note**: They show in the paper that they are able to perform better than the Sparse GP and SKI methods and apply it to very long time series.
> **Supp.**: 
> * **Bayes-Newton Models for Approximate Bayesian Inference with PSD Guarantees** - Wilkinson et al (2021) - show how we can scale this to big data.
> * **Spatiotemporal Learning via Infinite-Dimensional Bayesian Filtering and Smoothing** - Sarkka et al (2013) - demonstrate how to use the standard Markovian GP on spatio-temporal data.

> **Doubly Sparse Variational Gaussian Processes** - Adam et al (2020)
> They introduce the Markovian Gaussian process method which



#### Datasets
