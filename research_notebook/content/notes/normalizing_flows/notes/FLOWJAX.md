# FlowJAX

> A package that implements some foundational normalizing flows in the JAX language. It will focus on models that are within the Gaussianization framework


---
## Demos

* 2D Plane Demo
* Tabular Data (Standard datasets)
* Information Theory Metrics (I, H, MI, TC, KLD)
* Better Sampling (HSI, BigEarthNet, QG)
* Generalized Copula Example (Privacy)


---
## Flows

### Element-Wise Flows

* [ ] Mixture CDF (Gaussian, Logistic)
* [ ] Parameterized Kernel
* [ ] Neural Spline Flows (Rational Quadratic, Linear Rational)
* [ ] Inverse CDF (Normal, Logit, Tanh)

---
### Coupling Flows

**Coupling Bijector**

* [ ] Mixture CDF (Gaussian, Logistic)
* [ ] Neural Spline

**Conditioners**

* [ ] FC
* [ ] ConvNet
* [ ] ResNet (FC, Conv)
* [ ] Self-Attention
* [ ] Nystromer
* [ ] Neural Operator (Fourier, Wavelet)

---
### Linear Transforms

**Tabular**

* [ ] Linear Orthogonal (Fixed)
* [ ] Householder

**Multi-Dim**

* [ ] 1x1 Convolutional
* [ ] Convolutional Exponential

### Multiscale

* [ ] Reshape
* [ ] iRevNet
* [ ] Wavelet (Haar)

### Dimensional Reduction

> Useful very high-dimensional problems 

* [ ] Slicing
* [ ] Stochastic (VAE)

### Augmentation

> Useful for strange topologies and insufficient dimensions.

* [ ] Augmentation

---
### Spatio-Temporal

* Neural Flow

---
## Applications

* Sampling