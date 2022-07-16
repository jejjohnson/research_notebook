# Numpyro Tutorial

## Objective

This will be a series of tutorials that showcase how we can make use of the JAX autodifferentiation framework along with the Numpyro probabilistic programming framework to do Kalman Filters. This will demonstrate how to build a KF as well as do inference. There



---
## Structure

> Building a Kalman Filter with Numpyro

* [ ] Simple Kalman Filter - loops + known params
* [ ] Scan Function - faster, prior samples, better labels
* [ ] Conditioning - posterior samples
* [ ] Object Tracking


---


* [ ] `numpyro.param` - +ve constraints, diag/full covariance, MLE est.
* [ ] Object Tracking


---


* [ ] `numpyro.sample` | `numpyro.dist` - +ve constraint, cholesky decomposition, MAP/MCMC Estimation
* [ ] Object Tracking


---


* [ ] masking - missing values, faster inference
* [ ] reparam - augmentation, marginal Gaussianization, Gaussianization
* [ ] associative scan - faster inference


---


* [ ] ensemble - vmap, sample


---
## Inference

* [ ] Maximum Likelihood Estimation (MLE)
* [ ] Maximum A Priori (MAP)
* [ ] Sampling:
    * [ ] Markov Chain Monte Carlo (MCMC)
    * [ ] Hamiltonian Monte Carlo (HMC)
* [ ] Approximate Inference (AutoGuide)
    * [ ] Delta (MAP)
    * [ ] Laplace Approximation
    * [ ] Normal (Diag)
    * [ ] Normal (Full)
    * [ ] Normalizing Flow (NF)
* [ ] Hybrid Methods
    * [ ] NeuTraReparam
    * [ ] Differentiable Annealed Importance Sampling (DAIS)

## Examples

* [ ] Object Tracking
* [ ] Lorenz63
* [ ] Lorenz96
* [ ] Sea Surface Temperature (SST)
* [ ] Sea Surface Height (SSH)


## Kalman Filter

* [ ] Linear (Discrete)
* [ ] Linear (Continuous)
* [ ] Ensemble (Stochastic)
* [ ] Ensemble (Deterministic)
* [ ] Dynamical System
