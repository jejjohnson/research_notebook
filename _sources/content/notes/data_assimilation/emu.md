# Emulation


### Linear Gaussian State Space Model 

LGSSM

$$
\begin{aligned}
\mathbf{z}_0 &\sim \mathcal{N}( \boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0) \\
\mathbf{z}_t &= \mathbf{Fz}_{t-1} + \boldsymbol{\epsilon}_{\mathbf z} \\
\mathbf{x}_t &= \mathbf{Hz}_{t} + \boldsymbol{\epsilon}_{\mathbf x} \\
\end{aligned}
$$

**Loss Function**

We can train this using maximum likelihood estimation (MLE) like so: c

$$
\mathcal{L}(\boldsymbol{\theta};\mathbf{x}_{1:T}) = p(\mathbf{x}_{1:T}) = \sum_{t=1}^T p(\mathbf{x}_{t}|\mathbf{x}_{1:t-1})
$$

Because we have linear operations, we can minimize this exactly

$$
p(\mathbf{x}_{t}|\mathbf{x}_{1:t-1}) = \mathcal{N}(\mathbf{x}_t|\mathbf{Hz}_t, \mathbf{R}_t)
$$


---
### Transformed LGSSM

$$
\begin{aligned}
\mathbf{z}_t &= \mathbf{Fz}_{t-1} + \boldsymbol{\epsilon}_{\mathbf z} \\
\tilde{\mathbf{x}}_t &= \mathbf{Hz}_{t} + \boldsymbol{\epsilon}_{\mathbf x} \\
\mathbf{x}_t &= \boldsymbol{T}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}_t)
\end{aligned}
$$

$$
\mathcal{L} = 
$$

Again, we can solve for this exactly because we have a simple transformation function, $\boldsymbol{T}$:

$$
\mathcal{L}(\boldsymbol{\theta};\mathbf{x}_{1:T}) = p(\mathbf{x}_{1:T}) = \sum_{t=1}^T \log p(\tilde{\mathbf{x}}_{t}|\tilde{\mathbf{x}}_{1:t-1}) \; + \log \left|\det \nabla_{\mathbf{x}_t} \boldsymbol{T}^{-1}(\mathbf{x}_t) \right|
$$

Note that this is the same loss function as above, however, we see that

**Motivation**

We take our inspiration from [Koopman theory](). We assume that the perfect dynamics to describe the transition of our state from $\mathbf{x}_t$ to $\mathbf{x}_{t+1}$ can be described by a non-linear function, $\boldsymbol{f}$. 

$$
\frac{d}{dt}\mathbf{x}(t) = \boldsymbol{f}(\mathbf{x}(t), \boldsymbol{\theta})
$$

However, this is often unknown and we have no way to solve this. One way to think However, we postulate that there exists some transformation (possibly invertible) that allows us to 

$$
\begin{aligned}
\mathbf{z} = \boldsymbol{T}(\mathbf{x}; \boldsymbol{\theta}) \\
\mathbf{x} = \boldsymbol{T}^{-1} (\mathbf{z}; \boldsymbol{\theta})
\end{aligned}
$$

$$
\frac{d}{dt}\mathbf{z}(t) = \mathbf{Lz}(t)
$$

A practical example is the case of the coordinate system for the planetary motion without our solar system. An Earth-centric coordinate system shows very non-linear structure when describing the motion for other planets and the sun. However, a sun-centric coordinate system showcase very simple dynamics that can be described using simpler equations.

We are free to choose this transformation and we get some very interesting properties depending upon the restrictions we choose. Some examples of transformations, $\mathbf{T}$, include FFT, Wavelets, PCA, AE, and NFs.


**Transformations**

$$
\log p(\mathbf{x}) \approxeq \log p(\mathbf{z}) + \mathcal{V}(\mathbf{x}, \mathbf{z}) + \mathcal{E}(\mathbf{x}, \mathbf{z})
$$

where $\mathcal{V}(\mathbf{x}, \mathbf{z})$ is the *likelihood contribution* and $\mathcal{E}(\mathbf{x}, \mathbf{z})$ is the bound looseness.


#### Bijective

These are known as **Normalizing Flows** or **Invertible Neural Networks**. 


$$
\begin{aligned}
\mathbf{z} &= \boldsymbol{T}(\mathbf{x}) \\
\mathbf{x} &= \boldsymbol{T}^{-1}(\mathbf{z})
\end{aligned}
$$

The good thing is that these transformations give you *exact* likelihoods. However, they are limited because you have *limited expressivity* due to the diffeomorphic constraint. In addition, it requires this to be

In terms of the log-likelihood function, these are:

$$
\begin{aligned}
\mathcal{V}(\mathbf{x},\mathbf{z}) &= \log |\det \nabla_{\mathbf x} \boldsymbol{T}(\mathbf{x})| \\
\mathcal{E}(\mathbf{x}, \mathbf{z}) &= 0
\end{aligned}
$$

#### Stochastic

These are known as **Variational AutoEncoders**.

$$
\begin{aligned}
\mathbf{z} &= \boldsymbol{T}_e(\mathbf{x}) \\
\mathbf{x} &= \boldsymbol{T}_d(\mathbf{z})
\end{aligned}
$$

where $\mathbf{x} = \boldsymbol{T}_d \circ \boldsymbol{T}_e(\mathbf{x})$.

$$
\begin{aligned}
\mathcal{V}(\mathbf{x},\mathbf{z}) &= \frac{p(\mathbf{x}|\mathbf{z})}{q(\mathbf{z}|\mathbf{x})} = \frac{\text{encoder}}{\text{decoder}} \\
\mathcal{E}(\mathbf{x}, \mathbf{z}) &= \frac{p(\mathbf{z}|\mathbf{x})}{q(\mathbf{z}|\mathbf{x})}
\end{aligned}
$$

**Cons**:

**Approximate Inference**:

**Pros**:

**Dimensionality Reduction**: This allows us to reduce (or increase) the dimensionality 

The downside to this method is that it offers *approximate* inference. However, on the upside, there will be **higher expressivity** because you are not limited to diffeomorphic transformations. And in addition, it offers dimensionality reduction to reduce the complexity of the problem. All of the flexibility comes at a price because it becomes quite difficult to train and it is a relatively unexplored territory for EO-data.

---
### Ensemble State Space Model

$$
\begin{aligned}
\mathbf{z}_t &= \boldsymbol{f}(\mathbf{z}_{t-1}) + \boldsymbol{\epsilon}_{\mathbf z} \\
\mathbf{x}_t &= \boldsymbol{h}(\mathbf{z}_{t}) + \boldsymbol{\epsilon}_{\mathbf x} \\
\end{aligned}
$$

This is difficult. However, if we restrict ourselves to Gaussian distributions, this allows us to have

$$
\begin{aligned}
p(\mathbf{z}_t|\mathbf{z}_{t-1}) &= \mathcal{N}(\boldsymbol{f}(\mathbf{z}_{t-1}), \mathbf{Q}) \\
p(\mathbf{x}_t|\mathbf{z}_t) &= \mathcal{N}(\boldsymbol{h}(\mathbf{z}_{t}), \mathbf{R}) \\
\end{aligned}
$$


**Log-Likelihood**

Given we have samples, we will need to 
$$
\begin{aligned}
\bar{\mathbf{x}}_t &= \frac{1}{N_e} \sum_{i=1}^{N_e} \mathbf{x}_{(i),t} \\
\bar{\mathbf{P}}_t &= \frac{1}{N_e - 1} \sum_{i=1}^{N_e} (\mathbf{x}_{(i),t} - \bar{\mathbf{x}}_t)(\mathbf{x}_{(i),t} - \bar{\mathbf{x}}_t) 
\end{aligned}
$$

$$
\mathcal{L}(\boldsymbol{\theta}) = \sum_{t=1}^T \log \mathcal{N}(\mathbf{H}\bar{\mathbf{x}}_t, \mathbf{H}\bar{\mathbf{P}}\mathbf{H}^\top + \mathbf{R})
$$





---
### Deep State Space Model (DSSM)

$$
\begin{aligned}
p(\mathbf{z}_0) &= \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)\\
p(\mathbf{z}_t|\mathbf{z}_{t-1}) &= \mathcal{N}(\text{NN}_{\boldsymbol{\mu}_t}(\mathbf{z}_{t-1}); \text{NN}_{\boldsymbol{\nu}_t}(\mathbf{z}_{t-1})) \\
p(\mathbf{x}_t|\mathbf{z}_t) &= \mathcal{N}(\text{NN}_{\boldsymbol{\mu}_e}(\mathbf{z}_{t}), \text{NN}_{\boldsymbol{\nu}_e}(\mathbf{z}_{t})) \\
\end{aligned}
$$


**Loss Function**

$$
\mathcal{L}(\boldsymbol{\theta}, \boldsymbol{\phi}; \mathbf{x}_{1:T}) = \sum_{t=1}^T \mathbb{E}_{q_{\boldsymbol \phi}}\left[ \log p (\mathbf{z}_t | \mathbf{z}_{t-1}) + \log p(\mathbf{x}_t | \mathbf{z}_t) - \log q_{\boldsymbol \phi}(\mathbf{z}_t | \mathbf{z}_{t-1})\right] 
$$

**Pros**

* More Flexible
* More Scalable

**Cons**

* Approximate Inference
* More Flexible (harder to train)



---
## Tutorials


**Generative Models**

* Iterative Gaussianization
* Gaussianization Flows
* GFs for Spatial Data
* Stochastic Transformations (Slicing, Augmentation)


**Kalman Filters**

* Linear Gaussian State Space Model (LGSSM)
* Ensemble LGSSM (EnsLGSSM)
* Gaussianized State Space Model (GaussSSM)
* Deep Gaussian State Space Model (DGSSM)

