# Inverse Problems


## Problem Formulation

$$
\mathbf{y} = \boldsymbol{h}(\mathbf{x};\boldsymbol{\theta}) + \boldsymbol{\eta}
$$

where:

* $\mathbf{y} \in \mathbb{R}^{D_\mathbf{y}}$ is a noisy measurement
* $\mathbf{x} \in \mathbb{R}^{D_\mathbf{x}}$ is the original signal
* $\boldsymbol{h}(\:\cdot \:;\boldsymbol{\theta}): \mathbb{R}^{D_\mathbf{x}} \rightarrow  \in \mathbb{R}^{D_\mathbf{y}}$ is a measurement function, parameterized by $\boldsymbol{\theta}$.

---
### Data

$$
\mathbf{x} \in \mathbb{R}^{D_{\mathbf{x}}}
$$

* **1D Signal**: $\mathbf{x} \in \mathbb{R}^{D}$, $\mathbf{x} \in \mathbb{R}^{T}$
* **2D Image**: $\mathbf{x} \in \mathbb{R}^{H\times W}$, $\mathbf{x} \in \mathbb{R}^{u\times v}$
* **3D Volume**: $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$, $\mathbf{x} \in \mathbb{R}^{u \times v \times depth}$, $\mathbf{x} \in \mathbb{R}^{u \times v \times time}$
* **2D Volume Sequence**: $\mathbf{x} \in \mathbb{R}^{H \times W \times C \times time}$, $\mathbf{x} \in \mathbb{R}^{u \times v \times depth \times time}$

---
### Forward Direction

The first direction is the forward direction. 

$$
\mathbf{y} = \boldsymbol{h}(\mathbf{x};\boldsymbol{\theta})
$$

We can try to find a solution for the forward problem which is to use the state, $\mathbf{x}$, to help predict the observations, $\mathbf{y}$. This is typically the easier of the two directions. Many times we simply need to approximate $\mathbf{y}$ with some function $\boldsymbol{f}_{\boldsymbol{\theta}}(\mathbf{x})$. We can use point estimates, i.e. $\hat{\mathbf{y}}\approx \mathbf{y}^* = \operatorname*{argmax} p(\mathbf{y|x})$, or we can try to obtain posterior distributions, i.e. $\hat{p}_{\boldsymbol{\theta}}(\mathbf{y|x}) \approx p(\mathbf{y|x})$. We often call this **discriminative** or **transductive** machine learning. However, discriminative models are hard to interpret, explain and validate.

**Note**: This is quite different than traditional sciences because we don't "model the real world".

---
### Inverse Direction

The other direction is the inverse direction. 

$$
\mathbf{x} = \boldsymbol{h}^{-1}(\mathbf{y};\boldsymbol{\theta})
$$



We can also try to find a solution to the **inverse problem** whereby we have some observations, $\mathbf{y}$, and we want to them help us predict some state, $\mathbf{x}$. So we are more interested in the data generation likelihood process, $p(\mathbf{x}|\mathbf{y})$. It is a more difficult problem because we need to make assumptions about our system in order to formulate a problem. Fortunately, once the problem has been formulated, we can use Bayes theorem and the standard tricks within to solve the problem. This is often called **generative** modeling.

**Example Problem**:

Let's take some hidden system parameters, $\mathbf{x}$, and let's take some observations of the system behaviour, $\mathbf{y}$. Our objective is the determine the posterior $p(\mathbf{x}|\mathbf{y=\hat{y}})$ to estimate the parameters, $\mathbf{x}$, from some measured $\hat{\mathbf{y}}$. We could learn a $p(\mathbf{x|y})$ using synthetic data from a simulation $\mathbf{y}=\boldsymbol{g}(\mathbf{x}; \boldsymbol{\epsilon})$ of the forward process.

---
### Noise


---

## Ill-Posed

Our problem is ill-posed. This means that the space of possible solutions is very large, many of which are nonsensical. So we need to constrain our model such that the space is manageable. If we think about our unknowns, first we have an unknown input, $\mathbf{x}$. 

$$
\mathcal{L}(\theta,\mathbf{x}) = \argmin_{\mathbf{x},\theta} ||\mathbf{y} - \boldsymbol{f}(\mathbf{x};\theta)||_2^2 + \lambda \mathcal{R}(\mathbf{x})
$$

In the case of $\boldsymbol{f}$, we also have unknown parameters, $\theta$.

$$
\mathcal{L}(\theta,\mathbf{x}) = \argmin_{\mathbf{x},\theta} ||\mathbf{y} - \boldsymbol{f}(\mathbf{x};\theta)||_2^2 + \lambda_1 \mathcal{R}(\mathbf{x}) + \lambda_2 \mathcal{R}(\theta)
$$

---
## Solutions

### Empirical Minimization

This is the case when we want to learn a solver for deterministic inverse problems. Given a set of inputs, $\mathcal{D}=\{ \mathbf{y}_i, \mathbf{x}_i \}_{i=1}^N$. We also choose the functional form of our model, $\boldsymbol{f}$, e.g. a neural network. In this case, our function is parameterized by $\theta$. 

$$
\mathcal{L}_{\text{MSE}}(\theta) = \frac{1}{2}||y - f(x;\theta)||_2^2
$$

In order to restrict the space of solutions, we penalize the parameters, $\theta$, of our function, $\boldsymbol{f}$. 

$$
\mathcal{L}_{\text{MSE}}(\theta) = \frac{1}{N}\sum_{i=1}^N||\mathbf{y}_i - f(\mathbf{x}_i;\theta)||_2^2 + \lambda \mathcal{R}(\theta)
$$

We can do an offline training regime for the best parameters, $\theta$, given our training data, $\mathcal{D}$. Then


---
### Bayesian Inversion

We take the posterior density

$$
p(\theta|x, y) \propto p(y|x,\theta)p(\theta)
$$

**Minimization (MLE)**

$$
\boldsymbol{\theta}^* = \argmax_{\mathbf{x}} p(\mathbf{y}|\mathbf{x}) 
$$

**MAP**

$$
\mathcal{L}(\boldsymbol{\theta}) = \argmax_{\mathbf{x}} p(\mathbf{x}|\mathbf{y}) \propto \argmax_{\mathbf{x}} p(\mathbf{y}|\mathbf{x})p(\mathbf{x};\boldsymbol{\theta})
$$

We assume a Gaussian measurement model

$$
p(\mathbf{y}|\mathbf{x}) \sim \mathcal{N}(\mathbf{y};\boldsymbol{h}(\mathbf{x};\boldsymbol{\theta}), \sigma^2)
$$

**Loss Function (Explicit)**

$$
\mathcal{L}(\boldsymbol{\theta},\mathbf{x}) = \argmin_{\boldsymbol{\theta}, \mathbf{x}} \frac{1}{2\sigma^2}||\mathbf{y} - \boldsymbol{h}(\mathbf{x};\boldsymbol{\theta})||_2^2 - \log p(\mathbf{x};\boldsymbol{\theta})
$$

**Energy-Based Representation**

$$
U() \propto \exp(-\mathcal{L}_{\text{MSE}}(\theta))
$$


---
## Generic Formulation


$$
\mathbf{x} = \argmin_{\mathbf{x}} \lambda_1||\mathbf{y} - \boldsymbol{h}(\mathbf{x})||_2^2 + \lambda_2 \boldsymbol{R}(\mathbf{x})
$$

---
## Example: Denoising

$$
\mathbf{y} = \mathbf{x} + \epsilon, \;\; \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

---
## Probabilistic Priors

In this case, we assume some probabilistic prior

$$
\mathbf{x} \sim P_X
$$

Now, we can formulate this as a 

$$
\mathbf{x} = \argmin_\mathbf{x} \lambda ||\mathbf{y} - \mathbf{x}||_2^2 - \log p_X(\mathbf{x})
$$

Examples: Variational AutoEncoders, AutoRegressive Models, Normalizing Flows

* Deep Unfolding with Normalizing Flow Priors for Inverse Problems - Wei et al (2021)

---
#### Normalizing Flow Prior

**Normalizing Direction**

$$
\begin{aligned}
\mathbf{z} &= \boldsymbol{f}(\mathbf{x};\boldsymbol{\theta}) \\
&= \boldsymbol{f}_L\circ \boldsymbol{f}_{L-1} \circ \cdots \circ \boldsymbol{f}_1 (\mathbf{x})\\
\end{aligned}
$$

**Generative Direction**

$$
\begin{aligned}
\mathbf{x} &= \boldsymbol{g}(\mathbf{z};\boldsymbol{\theta}) \\
&= \boldsymbol{f}^{-1}(\mathbf{z};\boldsymbol{\theta}) \\
&= \boldsymbol{f}_1^{-1}\circ \boldsymbol{f}_{2}^{-1} \circ \cdots \circ \boldsymbol{f}_L^{-1} (\mathbf{x})\\
\end{aligned}
$$

**Density Evaluation**

$$
\log p(\mathbf{x};\boldsymbol{\theta}) = \log p(\mathbf{z}) + \log \left| \det \boldsymbol{\nabla}_\mathbf{x} \boldsymbol{f}(\mathbf{x};\boldsymbol{\theta})  \right|
$$

**Cost Function (Explicit)**

$$
\begin{aligned}
\mathcal{L}(\mathbf{x}, \mathbf{z}, \boldsymbol{\theta}) 
&= \argmin_{\mathbf{\theta},\mathbf{z}} \frac{1}{2\sigma^2} ||\mathbf{y} - \mathbf{A}\boldsymbol{g}(\mathbf{z}; \boldsymbol{\theta})||_2^2 - \log p(\mathbf{z}) \\
&= \argmin_{\mathbf{\theta},\mathbf{z}}  ||\mathbf{y} - \mathbf{A}\boldsymbol{g}(\mathbf{z}; \boldsymbol{\theta})||_2^2 - \lambda ||\mathbf{z}||_2^2 \\
\end{aligned}
$$

where $\lambda$ is a regularization term to trade-off the prior versus the measurement.

**Cost Function (Implicit)**

$$
\begin{aligned}
\hat{\mathbf{z}} 
&= \argmin_{\mathbf{z}} ||\mathbf{y} - \mathbf{A}\boldsymbol{g}(\mathbf{z};\boldsymbol{\theta})||_2^2 - \log p(\mathbf{z}) \\
&= \argmin_{\mathbf{\theta},\mathbf{z}}  ||\mathbf{y} - \mathbf{A}\boldsymbol{g}(\mathbf{z}; \boldsymbol{\theta})||_2^2 - \lambda ||\mathbf{z}||_2^2 \\
\end{aligned}
$$

---
### Dictionary Prior

Examples: PCA (aka EOF, POD, SVD)


$$
\mathbf{x} = \mathbf{D}\boldsymbol{\alpha}
$$

$$
\mathbf{x} = \argmin_{\boldsymbol{\alpha}}||\mathbf{y} - \mathbf{D}\boldsymbol{\alpha}||_2^2
$$


### Norm-Based Prior

$$
\mathbf{x} = \argmin_\mathbf{x} \lambda ||\mathbf{y} - \mathbf{x}||_2^2 - \lambda_2 ||\nabla \mathbf{x}||_2^2
$$

