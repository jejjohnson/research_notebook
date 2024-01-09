---
title: Problem Structure
subject: Modern 4DVar
subtitle: How to think about the problem structure.
short_title: Problem Structure
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - CNRS
      - MEOM
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: data-assimilation, open-science
abbreviations:
    GP: Gaussian Process
---



```{mermaid}
graph TD
    State --> Observations
    State --> Quantity-of-Interest
```

This document provides an overview of the components needed to understand the modern 4DVar formulation. We introduce the three main components: 1) the quantity of interest, 2) the observations and 3) the state. Afterwards, we describe how they are related to one another through a factor graph. We also give a brief overview of an amortized approach which simply relates the observations and the quantity of interest.


---
## Quantity of Interest



```{mermaid}
graph TD
    Quantity-of-Interest
```

Let's start with the quantity of interest (QOI). This is the thing that we are interested in estimating. We define our quantity of interest as a field which, given some coordinates within a domain, we get scalar or vector values of our QOI. Mathematically, we can write this as

$$
\begin{aligned}
\boldsymbol{u} = \boldsymbol{u}(\vec{\mathbf{x}},t)
&& \boldsymbol{u}: \boldsymbol{\Omega}_u \times \mathcal{T}_u \rightarrow \mathbb{R}^{D_u}
&& \vec{\mathbf{x}} \in \Omega_u \subseteq \mathbb{R}^{D_s}
&& t \in \mathcal{T}_u \subseteq \mathbb{R}^{+}
\end{aligned}
$$

where $(\vec{\mathbf{x}},t)$ are spatiotemporal coordinates defined within some QOI domain $(\boldsymbol{\Omega}_u,\mathcal{T}_u)$ and $\boldsymbol{u}$ is a scaler or vector-valued field defined for said domain. This QOI is basically the minimum amount of variables that we need to infer in order to accomplish some tasks. For example, we could be interested in calculating sea surface currents to predict potential trajectories for ships to follow.


---

## Observations

```{mermaid}
graph LR
    Observations

```


We define *observations* as quantities that we can measure which could be useful to help us infer the QOI we wish to estimate. If we're lucky, the observations are the exact QOI we wish to infer, perhaps with just some small noise corruption. However, in most real world problems, we often do not have access to the exact quantity of interest we are interested in estimating. So we need to use auxilary variables in the form of observations which we believe are useful and can help us infer the state.

Mathematically, we can write this as

$$
\begin{aligned}
\boldsymbol{y} = \boldsymbol{y}(\vec{\mathbf{x}},t)
&& \boldsymbol{y}: \boldsymbol{\Omega}_y \times \mathcal{T}_y \rightarrow \mathbb{R}^{D_y}
&& \vec{\mathbf{x}}\in\Omega_y\subseteq\mathbb{R}^{D_s}
&& t\in\mathcal{T}_y\subseteq\mathbb{R}^{+}
\end{aligned}
$$

where $(\vec{\mathbf{x}},t)$ are spatiotemporal coordinates defined within some observation domain $(\boldsymbol{\Omega}_y,\mathcal{T}_y)$ and $\boldsymbol{y}$ is a scaler or vector-valued field defined for said domain.

In most cases, the observations are corrupted or auxillary variables that we believe are related to the QOI. For example, our QOI maybe SSH but we only have access to SST observations. So we can use SST observates to help us infer the state which subsequently helps us infer our QOI, SSH.



---
## State

```{mermaid}
graph LR
    State
```

The state acts as a latent variable which encapsulate all of the necessary information to generate the quantity of interest. It can be exactly the same as our quantity of interest or it can be some latent embedding. We define our state as another field which, given some coordinates within a domain, we get a scalar or vector value of our state. Mathematically, we can write this as

$$
\begin{aligned}
\boldsymbol{z} = \boldsymbol{z}(\vec{\mathbf{x}},t),
&& \boldsymbol{z}: \boldsymbol{\Omega}_z \times \mathcal{T}_z \rightarrow \mathbb{R}^{D_z}
&& \vec{\mathbf{x}}\in\Omega_q\subseteq\mathbb{R}^{D_s}
&& t\in\mathcal{T}_z\subseteq\mathbb{R}^{+}
\end{aligned}
$$

where $(\vec{\mathbf{x}},t)$ are spatiotemporal coordinates defined within some state domain $(\boldsymbol{\Omega}_z,\mathcal{T}_z)$ and $\boldsymbol{z}$ is a scaler or vector-valued field defined for said domain.


**Machine Learning World**

In the ML world, we don't really care about what this state looks like. We often refer to this as a latent variable where we try to learn through various methodologies.  We've seen various instantiations of this in the literature. A simple example is the autoencoder which we use to learn a latent variable to represent different datasets, e.g. [MAE](https://arxiv.org/abs/2111.06377), [SatMAE](https://arxiv.org/abs/2207.08051), [VideoMAE](https://arxiv.org/abs/2203.12602), or [SpatioTemporal MAE](https://arxiv.org/abs/2205.09113).  Another example is the [generative modeling literature](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/), e.g. [score-based generative models](https://nvlabs.github.io/LSGM/), which projects the data to a probabilistic latent variable through a parameterized transformation. A more elaborate example is in the paper [ImageBind: One Embedding Space to Bind Them All](https://github.com/facebookresearch/ImageBind) where they use a joint embedding across 6 different data modalities: images, text, audio, depth, thermal and IMU data.




**Geoscience World**

In the geoscience world, we often do care what the state looks like but we recognize that the true full state is often impossible to define. Even with a plethora of physical knowledge about the underlying processes, we still recognize that we are formulating incomplete representations of the true state of a process. For example, in a model of the ocean, we often define the "full state" as temperature, salinity and pressure [[Wright et al., 1997](https://journals.ametsoc.org/view/journals/atot/14/3/1520-0426_1997_014_0735_aeosfu_2_0_co_2.xml)]. However, we could argue that these are only the minimum QOIs needed to calculate all necessary derived QOIs for some alterior motive. A dynamical model might define the "full state" as temperature, salinity, sea surface height and velocities because these are the minimum amount of variables needed to calculate all derived QOIs. Therefore, we argue that these "full states" are simply QOIs. The true full state of the ocean is an unknown quantity that may be jointly distributed with the QOIs but there is still missing physical quantities that we are not aware of or cannot measure.

$$
\begin{aligned}
\text{Approx. State} &= [\text{Temperature}, \text{Salinity}, \text{Pressure}] \\
\text{Dyn. State} &= [\text{Temperature}, \text{Salinity}, \text{SSH}, \text{Velocities}] \\
\text{True State} &= [\text{Temperature}, \text{Salinity}, \text{Pressure}, \text{SSH}, \text{Velocities}, \text{Missing Physics}]
\end{aligned}
$$

This is important to mention because we often get confused when talking about things like parameterizations where we want to fill in the missing physics from our numerical schemes to recover the ocean state. However, we want to acknowledge that this is not the ocean state, theses are simply quantities of interest.




---
## Relationships

In the above section, we outlined all three components within this system. It is important to be clear about which category the datasets fall into when solving problems. This makes the assumptions clear and the subsequent decisions easier to follow. Now, we will outline how each of the components interact with each other.

---
### State & Quantity of Interest

```{mermaid}
graph LR
    State --> Quantity-of-Interest
```

We can assume that the state and the QOI are jointly distributed in some fashion. Looking at the factor graph above, we explicitly state that the QOI comes from the state. This is an intuitive assumption because we acknowledge that our QOI is a subset of the true underlying process of the system. So it makes sense that the QOI is conditionally dependent upon the state and not the other way around. Using the Bayesian formulation, we can write the conditional distribution and prior for both quantities.

$$
p(\boldsymbol{u},\boldsymbol{z}) =
p(\boldsymbol{u}|\boldsymbol{z})p(\boldsymbol{z})
$$

This states that we have a prior distribution on our state and a conditional distribution of our QOI which depends on the state.

<!-- Notice how we impose the conditional distribution of the QOI given the state as shown in the diagram.  -->

<!-- We can also further assume that the QOI can determined by some parameterized function, $\boldsymbol{T}(\cdot; \boldsymbol{\theta})$, which maps the state to the QOI.

$$
\boldsymbol{u} \sim
p(\boldsymbol{u}|\boldsymbol{T}(\boldsymbol{z};\boldsymbol{\theta}))
$$
 -->

---
### State & Observations


```{mermaid}
graph LR
    State --> Observations
```

We can assume that the state and the observations are jointly distributed in some fashion. Looking at the factor graph above, we explicitly state that the observations is dependent upon the state. This is an intuitive assumption because we acknowledge that our observations are a subset of the true underlying process of the system; otherwise there is no reason to believe that the observations would help infer the QOI. So it makes sense that the observations is conditionally dependent upon the state and not the other way around. Using the Bayesian formulation, we can write the conditional distribution and prior for both quantities.

$$
p(\boldsymbol{y},\boldsymbol{z}) =
p(\boldsymbol{y}|\boldsymbol{z})p(\boldsymbol{z})
$$

This states that we have a prior distribution on our state and a conditional distribution of our observations which depends on the state. For example, in a denoising problem, we could say that the prior of the state is Gaussian but the conditional distribution is Normal with a linear transformation.


---
### State, QOI & Observations


```{mermaid}
graph LR
    State --> Quantity-of-Interest
    State --> Observations
```

Like the QOI and the observations, we can also assume that the QOI, the state and the observations are jointly distributed in some fashion. The above factor graph combines the two factor graphs whereby the QOI and observations are dependent upon the state but have no dependence on each other directly. Using the Bayesian formulation, we can write the conditional distributions and priors for all quantities.

$$
p(\boldsymbol{u},\boldsymbol{y},\boldsymbol{z}) =
p(\boldsymbol{u}|\boldsymbol{z})
p(\boldsymbol{y}|\boldsymbol{z})
p(\boldsymbol{z})
$$

This states that we have a prior distribution on our state and two conditional distributions: one on our QOI and one on our observations. So given some conditional generative process for the QOI and observations, we are interested in finding the best state.


---
### Direct Learning


```{mermaid}
flowchart LR
    Observations --> Quantity-of-Interest
```


We can ask ourselves why do we bother with this intermediate step of defining a state space. Learning the joint distribution of $p(\boldsymbol{u},\boldsymbol{y},\boldsymbol{z})$ can be philosophically easy to justify why this is necessary. But it could be hard to justify it from a practical perspective when we ultimately want to just use the observations to infer a QOI directly, i.e. $p(\boldsymbol{u}|\boldsymbol{y})$. Given a pairwise dataset of QOI and observations, $\mathcal{D}=\{\boldsymbol{u}_n, \boldsymbol{y}\}_{n=1}^{N}$, we can just *learn* some parameterized model that seeks  predict the QOI given some observations by minimizing an objective function that measures the closeness to our predictions on historical data.

$$
\begin{aligned}
\text{Objective Based}: &&
p(\boldsymbol{\theta}|\mathcal{D}) &\propto
p(\boldsymbol{u}|\boldsymbol{y}, \boldsymbol{\theta})p(\boldsymbol{\theta}) \\
\text{Objective Based}: &&
p(\boldsymbol{\theta}|\mathcal{D}) &\propto
p(\boldsymbol{u}|\boldsymbol{z},\boldsymbol{y}, \boldsymbol{\theta})
p(\boldsymbol{z}|\boldsymbol{y},\boldsymbol{\theta})
p(\boldsymbol{\theta}|\boldsymbol{y})
\end{aligned}
$$

Another way to think about this is called "discriminative learning" whereby we simply learn a deterministic mapping from one quantity to another. We can also call this "amortization" because we bypass the state space intermediate step and directly approximate the solution. This is a fair critique which has been said by other individuals like Vapnik.

> "...one should solve the problem directly and never solve a more general problem as an intermediate step..." - Vapnik (1998)

However, we argue that there is one benefit and one necessity to model the joint distribution. For example, if we model the joint distribution, we can:

* Compute arbitrary conditionals and marginals
* Compare the probabilities of different examples
* Reduce the dimensionality of the data
* Identify interpretable latent structures
* Generate/Fantasize completely new data

This offers us many opportunities for more expressive and transferable models to other domains.

The necessity stems from the fact that geosciences present unique challenges. In general, often the domain of our observations is often much smaller than the domain of our quantity of interest, $\mathbb{R}^{D_y}<<\mathbb{R}^{D_u}$. This forms an ill-posed problem whereby there are many solutions that could learn. Furthermore, we recognized that our QOI is complete and uncertain so we don't want to rely 100% on the solutions generated from minimizing our QOI.
