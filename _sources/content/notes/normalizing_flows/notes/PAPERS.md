# Papers

* All You Need is Gaussianization
* Conditional Gaussianization for Emulation
* Gaussianized Markovian Generative Models for Spatio-Temporal Data


---
## Elevator Pitch Ideas

**All You Need is Gaussianization**

> Use Gaussianization Flows for everything! A viable option for simple, effective normalizing flows.



**Gaussianized Kalman Filters**

> Combines iterative Gaussianization and traditional kalman filters for iterative filtering and smoothing solutions.

**Deep Gaussianized Kalman Filters**

> Combines Gaussianization Flows and Deep Kalman Filters for time series models. Applications include 1D, 2D and 3D spatio-temporal datasets.

**Conditional Gaussianization Flows**

> Trainable conditional Gaussianization flows for emulation.

**Deep Gaussianized Spatio-Temporal Features**

> Feature extractors + Gaussianization for complex nD Spatio-Temporal data. nD Spatio-Temporal data is very complex to model. Perhaps a feature extractor, e.g. slow-fast or convLSTM, will capture the complex elements on top of a normalizing flow. Applications: regression, classification, density estimation, anomaly detection, etc.

**Self-Normalizing Gaussianization**

> Couple the GDN with orthogonal rotations to get data-dependent invertible methods.


**Gaussianized Plug-In-Play Priors**

> Using Gaussianized Flows for plug-in-play priors.

---
## Papers

### All You Need is Gaussinization

> We want to show that  

**Algorithm Highlights**

* Simple Design
* Theoretical Guarantees
* Low Number of Parameters
* Data-Dependent Initialization

**Algorithm Improvements**

* Convolutions - showcase the orthogonal parameterization works well (1x1, Exponential)
* Flexible Mixture Class - Gaussian, Logistic
* Coupling - data-dependent element-wise layers (MixtureCDF + InverseCDF)
* Multiscale - Wavelets, Reshaping
* Dimension Reduction - Slicing

**New Datasets**

* BigEarthNet
* Hyperspectral Images
* Emulation Data

**Updated Theoretical Formulation**

* Mixtures Approximate Anything
* Uniformization goes to any distribution
* Marginal Gaussianization moves towards non-Gaussianity
* Linear layers don't make the convergence worse


---

### Gaussianized Kalman Filters


* Iterative + Kalman Filters


---
### Deep Gaussianized Kalman Filters

**Methods**

* Gaussianization Flows (GF)
* Kalman Filter (KF)
* Ensemble Kalman Filter (EnsKF)


---

### Conditional Gaussianization Flows


**Methods**

* Prior - Mixture Conditional Prior
* Coupling Layers - Data-Dependent Layers
* Augmentation - Latent Variable (Dim. Reduction / Augmentation)