# Gauss-Markov Models

> In the previous [section](./markov.md), we looked at Markov models and showed how we can assume some conditional, local dependence on time. This simplified the graphical model for dependences which allowed us to get very fast inference. An additional simplification is to add the Gaussian assumption on the probability distributions and limiting the functions to linear functions with Gaussian additive noise. This results analytical forms for all of the quantities of interest for the inference, i.e. the Kalman filter and RTS smoothing algorithms. In this section, we will go over these assumptions and showcase the equations that result from this simplification.


---

## TLDR

We can assume linear dynamics to describe the transition and emission models:

$$
\begin{aligned}
\mathbf{z}_{t} &= \mathbf{A}\mathbf{z}_{t-1} + \boldsymbol{\delta} \\
\mathbf{x}_t &= \mathbf{Hz}_t + \boldsymbol{\epsilon}
\end{aligned}
$$

We can also assume that the prior, transition and emission distributions are all Gaussian distributed:

$$
\begin{aligned}
p(\mathbf{z}_0) &= \mathcal{N}(\mathbf{z}_0;\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0) \\
p(\mathbf{z}_{t}|\mathbf{z}_{t-1}) &= \mathcal{N}(\mathbf{z}_{t};\mathbf{Az}_{t-1}, \mathbf{Q}) \\
p(\mathbf{x}_t|\mathbf{z}_t) &= \mathcal{N}(\mathbf{x}_t;\mathbf{Hz}_t, \mathbf{R})
\end{aligned}
$$

We are interested in the posterior $p(\mathbf{z}_t|\mathbf{x}_t)$. This can be calculated with the filtering, $p(\mathbf{z}_t|\mathbf{x}_{1:t})$, and smoothing, $p(\mathbf{z}_t|\mathbf{x}_{1:T})$, operations. Because of the nature of Gaussians, we have analytical forms for all of these equations. The filtering operation can be solved exactly using the *Kalman Filter* (KF) and the smoothing operation can be solved exactly using the Rauch-T-Striebel (RTS) smoother. Both operations can be computed in linear time, $\mathcal{O}(T)$.

### Kalman Filter 

Recall, we need to do a filtering operation which consists of an alternating predict-update step through all of the time steps. Each of these operations has an analytical form which is known as the Kalman filter algorithm. The equations are outlined below.


#### Predict Step

First, we need the predict step.

$$
p(\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_t; \boldsymbol{\mu}_{t|t-1},\boldsymbol{\Sigma}_{t|t-1})
$$(kalman_filter)

This equation is analytical but it is very involved. Below are each of the terms.

**Predictive Mean**

$$
\boldsymbol{\mu}_{t|t-1} = \mathbf{A}\boldsymbol{\mu}_{t-1}
$$(kf_pred_mean)

**Predictive Covariance**

$$
\boldsymbol{\Sigma}_{t|t-1} = \mathbf{A}\boldsymbol{\Sigma}_{t-1}\mathbf{A}^\top + \mathbf{Q}
$$(kf_pred_cov)

#### Update Step

$$
p(\mathbf{z}_t|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_t; \boldsymbol{\mu}_t, \boldsymbol{\Sigma}_t)
$$(kf_update)

**Innvoation Residual**

$$
\boldsymbol{r}_t = \mathbf{x}_t - \mathbf{H}\boldsymbol{\mu}_{t|t-1}
$$(kf_innovation_res)

**Innovation Covariance**

$$
\mathbf{S}_t = \mathbf{H}\boldsymbol{\Sigma}_{t|t-1}\mathbf{H}^\top + \mathbf{R}
$$(kf_innovation_cov)

**Kalman Gain**

$$
\mathbf{K}_t = \boldsymbol{\Sigma}_{t|t-1}\mathbf{H}^\top \mathbf{S}^{-1}
$$(kf_kalman_gain)

**Estimation Mean**

$$
\boldsymbol{\mu}_t = \boldsymbol{\mu}_{t|t-1} + \mathbf{K}_t\mathbf{x}_t
$$(kf_est_mean)

**Estimation Covariance**

$$
\boldsymbol{\Sigma}_t = (\mathbf{I} - \mathbf{KH})\boldsymbol{\Sigma}_{t|t-1}
$$(kf_est_cov)

### Kalman Smoothing 

This is the smoothing operation which predicts the states at time $t$ given all of the measurements, $1:T$. This is given by:

$$
p(\mathbf{z}_t|\mathbf{x}_{1:T}) = \mathcal{N}(\mathbf{z}_t; \boldsymbol{\mu}_{1:T}, \boldsymbol{\Sigma}_{1:T})
$$(kf_smooth)

The terms within this equation are outlined below:

**RTS Gain**

$$
\mathbf{G}_t = \boldsymbol{\Sigma}_t \mathbf{A}^\top(\boldsymbol{\Sigma}_{t+1|t})^{-1}
$$(kf_rts_gain)

**Smoothed Mean**

$$
\boldsymbol{\mu}_{t:T} = \boldsymbol{\mu}_t + \mathbf{G}_t(\boldsymbol{\mu}_{t+1|T} - \boldsymbol{\mu}_{t+1|t})
$$(kf_smooth_mean)

**Smoothed Covariance**

$$
\boldsymbol{\Sigma}_{t:T} = \boldsymbol{\Sigma}_t + \mathbf{G}_t (\boldsymbol{\Sigma}_{t+1|T} - \boldsymbol{\Sigma}_{t+1|t})\mathbf{G}^\top
$$(kf_smooth_cov)

Note: These equations are very involved. In addition, these are the naive equations. There are many more matrix reformulations and manipulations that increase the stability or speedup. So it might be worth it to try and code it from scratch the first time, but it is worth using well-tested implementations.


---
## Setting

Recall from the previous section that we were interested in Markov models due to their simplification of high-dimensional, high-correlated time series data. We envoke a few Markov properties like local memory and conditional independence of measures to get a very simplified graphical model. We also showcased all of the resulting functions that can be computed for the quantities of interest such as filtering, smoothing and posterior predictions. However, we did not mention anything about the functional form of these distributions. In principal, they can be anything they want. 

Recall the joint distribution of interest:

$$
p(z_{1:T}, x_{1:T}) = \underbrace{p(z_0)}_{\text{Prior}}\prod_{t=2}^T\underbrace{p(z_t|z_{t-1})}_{\text{Transition}}\prod_{t=1}^T \underbrace{p(x_t|z_t)}_{\text{Emission}}
$$(markov_joint)

We see with have 3 terms that we need to define: 

* the **prior** specifies the initial condition of the latent variable, $\mathbf{z}$
* the **transition model** specifies the distribution of the latent variable between time steps, $p(\mathbf{z}_t|\mathbf{z}_{t-1})$
* the **emission model** specifies the likelihood function of the measurement, $\mathbf{x}$ given the state vector, $\mathbf{z}$.

We are going to put Gaussian assumptions on all of our distributions (i.e. prior, transition and emission)


---

### Prior

We assume a Gaussian distribution for the prior of our state.

$$
p(\mathbf{z}_0) = \mathcal{N}(\mathbf{z}_0|\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)
$$(gm_prior)

where $\boldsymbol{\mu}_0 \in \mathbb{R}^{D_z}$ is the mean and $\boldsymbol{\Sigma}_0\in \mathbb{R}^{D_z \times D_z}$ is the covariance parameterizes the prior Gaussian distribution. This is a rather uninformative prior, but this ends up not mattering too much as the filter and smoothing solution tends to dominate after just a few time steps.

### Transition Distribution

We also assume the transition function is a linear function with some additive Gaussian noise on the state.

$$
\mathbf{z}_{t} = \mathbf{A}\mathbf{z}_{t-1} + \boldsymbol{\delta}
$$(gm_trans_f)

where $\mathbf{A} \in \mathbb{R}^{D_z \times D_z}$ and $\boldsymbol{\delta} \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})$. We can explicitly write the distribution for the transition model like so:

$$
p(\mathbf{z}_{t}|\mathbf{z}_{t-1}) = \mathcal{N}(\mathbf{z}_{t}|\mathbf{Az}_{t-1}, \mathbf{Q})
$$(gm_trans_p)

Note: the linear transformation gives us extra flexibility with sacrificing the easiness of manipulating Gaussian distributions. Any other non-linear function without any restrictions would give us problems later during the inference steps.

### Emission Distribution

We also assume the emission function is a linear function with some additive Gaussian noise on the measurements.

$$
\mathbf{x}_t = \mathbf{Hz}_t + \boldsymbol{\epsilon}
$$(gm_em_f)

where $\mathbf{H} \in \mathbb{R}^{D_x \times D_z}$ is a matrix and $\boldsymbol{\delta} \sim \mathcal{N}(\mathbf{0}, \mathbf{R})$ is the additive Gaussian noise. Again, like the transition function, we can write the transition distribution as a Gaussian distribution like so: 

$$
p(\mathbf{x}_t|\mathbf{z}_t) = \mathcal{N}(\mathbf{x}_t|\mathbf{Hz}_t, \mathbf{R})
$$(gm_em_p)




---

## Proofs

### Important Equations

Assumption: if we assume that the stochastic process involves a Markov Chain (i.e. a latent state) that evolves in a manner st 

Markov Chains formalize the notion of a stochastic process with a local *finite memory*.

Inference over Markov Chains separates into three operations, that can be performed in *linear* time, i.e.$\mathcal{O}(T)$.


#### Filtering


##### Predict
**Filtering - Predict**

This is the **Chapman-Kolmogorov** equation!

$$p(\mathbf{x}_t|\mathbf{y}_{0:t-1}) = \int p(\mathbf{x}_t | \mathbf{x}_{t-1}) \; p(\mathbf{x}_{t-1}|\mathbf{y}_{0:t-1})\; d\mathbf{x}_{t-1}$$

We predict using all of the past data points... 

---

Let's take the standard integral equation:

$$p(\mathbf{z}_t|\mathbf{x}_{1:t+1}) = \int p(\mathbf{z}_t|\mathbf{z}_{t-1})p(\mathbf{z}_{t-1}|\mathbf{x}_{1:t-1})d\mathbf{z}_{t-1}$$

We can substitute our Gaussian assumptions for each of the above equations.

**Proof (Term I)**

We can directly take this term from our assumption for equation 2.

$$p(\mathbf{z}_{t+1}|\mathbf{z}_{t}) = \mathcal{N}(\mathbf{z}_{t+1}|\mathbf{Az}_t, \mathbf{Q})$$

The only difference is the change of the index, $t$. We want $p(\mathbf{z}_{t}|\mathbf{z}_{t-1})$ instead of $p(\mathbf{z}_{t+1}|\mathbf{z}_t)$.

$$p(\mathbf{z}_{t}|\mathbf{z}_{t-1})= \mathcal{N}(\mathbf{z}_{t}|\mathbf{Az}_{t-1}, \mathbf{Q})$$

QED.

**Proof (Term II)**

This term is actually much simpler than it seems. It comes from the assumption that our 

$$p(\mathbf{z}_t|\mathbf{x}_{1:t+1}) = \int \mathcal{N}(\mathbf{z}_{t}|\mathbf{Az}_{t-1}, \mathbf{Q})\;p(\mathbf{z}_{t-1}|\mathbf{x}_{1:t-1}) \; d\mathbf{z}_{t-1}$$


##### Update

**Filtering - Update**

$$p(\mathbf{x}_t|\mathbf{y}_{0:t}) = \frac{p(\mathbf{y}_t|\mathbf{x}_t)p(\mathbf{x}_t|\mathbf{y}_{0:t-1})}{p(\mathbf{y}_t)}$$

**Smoothing**

$$p(\mathbf{x}_t|\mathbf{y}) = p(\mathbf{x}_t|\mathbf{y}_{0:t}) \int p(\mathbf{x}_{t+1}|\mathbf{x}_t)\; \frac{p(\mathbf{x}_{t+1}|\mathbf{y})}{p(\mathbf{x}_{t+1}|\mathbf{y}_{1:t})} \; d\mathbf{x}_{t+1}$$

**Take-Home**: we can use any structure we want for the probability density functions are, e.g. $p(\mathbf{x|y})$, the inference in this model will be *linear* cost (instead of cubic cost like for Gaussian processes).

**Ojo**: The integrals are the tricky part... 


#### Likelihood

$$
p
$$

---

## Gauss-Markov Models

**Assumptions**:

**Predictive Distribution**: is a Gaussian distribution

$$p(\mathbf{x}_{t_{i+1}}|\mathbf{x}_{1:i}) = \mathcal{N}(\mathbf{x}_{i+1}; \mathbf{Ax}_i, \mathbf{Q})$$

This will provide a linear relationship between the previous state, $\mathbf{x}_t$, and a subsequent state, $\mathbf{x}_{t+\tau}$, with some Gaussian additive noise $\boldsymbol{\epsilon}_{\mathbf{x}}$.

We have an initial believe an initial state.

$$p(\mathbf{x}_0)= \mathcal{N}(\mathbf{x}_0; \mathbf{m}_0, \mathbf{P}_0)$$

**Observation Likelihood**:

$$p(\mathbf{y}_i|\mathbf{x})=\mathcal{N}(\mathbf{y}_i; \mathbf{Hx},\mathbf{R})$$

**Easy Solution**: Assume Gaussianity



---


One easy assumption is that we can assume linear dynamics

> By assuming linear transition and emission dynamics, we can put Gaussian likelihoods. This results in filtering and smoothing posteriors which are exact and easy to calculate via the Kalman filter and smoothing algorithms.
> 

---

## Transition Dynamics

We can assume that the transition dynamics are linear transformation with an additive noise term.

$$
\boldsymbol{f}_{\boldsymbol \theta} (\mathbf{z}_{t}) \colon= \mathbf{F}_{t} \mathbf{z}_{t-1} + \boldsymbol{\eta}_t
$$

- $\mathbf{F}_{t} \in \mathbb{R}^{N_z \times N_z}$ is the transition matrix. This measures the physics of the process
- $\boldsymbol{\eta}_t \sim \mathcal{N}(\mathbf{0},\boldsymbol{\Sigma}_t)$ is the additive noise model.

---

Because we've assumed linear dynamics, we can easily incorporate Gaussian assumptions on the mean and noise distribution, i.e.

$$p(\mathbf{z}_{t}|\mathbf{z}_{t-1}) \sim \mathcal{N}(\mathbf{z}_t| \mathbf{F}_{t} \mathbf{z}_{t-1}, \boldsymbol{\Sigma}_t)
$$

We can of course solve for these terms **exactly** using a defined set of equations because we can propagate linear transformations through Gaussian distributions in closed form.

**Note**: we can assume the initial state is Gaussian distributed as well with or without transition dynamics, $\mathbf{z}_0 \sim \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)$.

---

## Observation Model

Again, we assume the observation model can be defined by a linear operator,

$$
\mathbf{x}_t = \mathbf{A}_t^\top \mathbf{z}_t + \boldsymbol{\epsilon}_t
$$

- $\mathbf{A}_{t} \in \mathbb{R}^{N_z \times N_x}$ is the emission matrix
- $\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0},\boldsymbol{\Gamma}_t)$ is the additive noise.

So, again, we can assume a Gaussian likelihood so this distribution is straightforward:

$$
p(\mathbf{x}_t|\mathbf{z}_t) \sim \mathcal{N}(\mathbf{x}_t| \mathbf{A}_t^\top \mathbf{z}_t, \boldsymbol{\epsilon}_t)
$$

because we can propagate linear transformations through Gaussian distributions in closed form.

---

### Note about Dimensionality

The observations, $\mathbf{x}_t \in \mathbb{R}^{N_x}$, will have less features than the latent space, $\mathbf{z} \in \mathbb{R}^{N_z}$. 

$$
N_x << N_y
$$

This is not always true but we can assume that the dynamics of the state space are a higher dimension than the observation space.

---

## Learnable Parameters

So for this model, we have the following parameters, $\boldsymbol{\theta}$ to find:

- Initial State - $\{ \boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0 \}$
- Transition Matices + Noise - $\{ \mathbf{F}_t, \boldsymbol{\Sigma}_t \}_{t \geq 1}$
- Emission Matrices + Noise - $\{ \mathbf{A}_t, \boldsymbol{\Gamma}t \}{t \geq 1}$

---

### Summary Equations

$$\begin{aligned}
p(\mathbf{z}_0) &= \mathcal{N}(\mathbf{z}_0 | \boldsymbol{\mu}_0, \mathbf{P}_0) \\
p(\mathbf{z}_{t}|\mathbf{z}_{t-1}) &= \mathcal{N}(\mathbf{z}_{t}|\mathbf{F}\mathbf{z}_{t-1}, \mathbf{Q}) \\
p(\mathbf{x}_t|\mathbf{z}_t) &= \mathcal{N}(\mathbf{x}_t|\mathbf{H}\mathbf{z}_t, \mathbf{R})
\end{aligned}$$

[Untitled](https://www.notion.so/c840cb4b20624bbda3f73ac383844fed)

---

## Pros and Cons

It is important to reflect on the assumptions we made about this model. This will give us some intuition on what we can expect and if there are underlying problems we can diagnosis more easily.

### Pros


#### Simple

We have assumed linear functions and Gaussian distributions. These are easily interpretable because they are models we can understand and characterize completely.

#### Efficient

As mentioned above, all of the quantities can be calculated in linear time. There are only matrix multiplications.

### Closed-Form / Exact

Inference is straightforward for these models because the joint distribution of the observation and the latent variable can be factorized:

$$
p(\mathbf{x}_{1:T}|\mathbf{z}_{1:T}) = p(\mathbf{x}_{1:T}|\mathbf{z}_{1:T})p(\mathbf{z}_{1:T})
$$

Furthermore, with the Markovian principal of the dependency only on the previous state, we can factorize this even more.

$$
p(\mathbf{x}_{1:T}|\mathbf{z}_{1:T})p(\mathbf{z}_{1:T}) = p(\mathbf{z}_0)\prod_{t=1}^T p(\mathbf{x}_{t}|\mathbf{z}_{t}) \prod_{t=2}^{T} p(\mathbf{z}_{t}|\mathbf{z}_{t-1})$$

All of the terms within this equation are known and have exact equations (the Kalman filter equations).

### Cons

### Linear Assumptions

We linear assumptions for both the transition model and the observation. This can fail spectacularly for non-linear cases.

### High-Dimensionality

The input observation data, $\mathbf{x}$, can be very high dimensional and has different characteristics. For example we can have the following data types:

- Tabular Data: `Features = 50`
- Time Series: `Time x Features = 100 x 50 = 5_000`
- Images: `Channels x Height x Width = 3 x 32 x 32 = 3_072`
- Video: `Time x Channels x Height x Width = 100 x 3 x 32 x 32 = 30_720`

Fortunately, given the Markovian assumption, we can factorize out the temporal dimensions. However, we're still left with a lot of samples to calculate correlations.

- Tabular Data: `Features = 50`
- Images: `Channels x Height x Width = 3 x 32 x 32 = 3_072`

In addition, we have a large parameter space because we don't utilize any architectures to capture any non-linear pairwise correlations between the data points in space (or time). For example, a convolutional operator would be able to capture local correlations wrt the channels in space.

---

### Non-Linearity

As mentioned above, one downside is the assumption that the observation model can be captured in a linear projection of the state, $\mathbf{z}_t$. It is well-known that the dynamics are in fact non-linear. We augment the observation model with Normalizing Flows.

$$
\begin{aligned}
\mathbf{x}_t &= \mathbf{A}_t^\top \mathbf{z}_t + \boldsymbol{\epsilon}_t \\
\mathbf{y}_t &= \mathbf{g}_{\boldsymbol \theta}(\mathbf{x}_t)
\end{aligned}
$$

- $\mathbf{F}_{t} \in \mathbb{R}^{N_z \times N_z}$ is the transition matrix
- $\boldsymbol{\eta}_t \sim \mathcal{N}(\mathbf{0},\boldsymbol{\Sigma}_t)$ is the additive noise model.
- $\mathbf{g}_{\boldsymbol \theta}: \mathcal{Y}\in\mathbb{R}^{N_y} \rightarrow \mathcal{X}\in\mathbb{R}^{N_x}$ - is an invertible, diffeomorphic transformation

**Assumption**: by augmented the observations $\mathbf{y}\in \mathbb{R}^{N_y}$ with an invertible transformation, $\boldsymbol{g}_{\boldsymbol \theta}$, we can obtain a latent representation s.t. the observation model is a linear transformation.

The only term that it affects is the likelihood term, $p(\mathbf{x}_t|\mathbf{z}_t)$. By definition, it is an invertible transformation, so we can calculate the likelihood term exactly.

$$
p(\mathbf{y}_t|\mathbf{z}_t) = p_\mathcal{X}(\mathbf{x}_t|\mathbf{z}_t)\left| \det \boldsymbol{\nabla}_{\mathbf{y}_t} \boldsymbol{g}_{\boldsymbol \theta}(\mathbf{y}_t) \right|
$$

So we can continue to take advantage of the closed-form solutions only with the edition of a non-linear transformation.

**Note**: The bottleneck of this term

---

### Dimensionality Reduction

Another aspect mentioned above is the high-dimensional data, $\mathbf{x}$. The input could be a very long feature-vector or perhaps an image. A linear transformation would have trouble capturing and generalizing over a complex input space.

We can use a Variational AutoEncoder (VAE) to embed the input $\mathbf{y} \in \mathbb{R}^{N_y}$ to a lower dimensional representation, $\mathbf{x} \in \mathbb{R}^{N_x}$. This makes use of a encoder-decoder structure. The