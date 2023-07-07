# Learning

Recall, we are interested in the following problem

$$
\boldsymbol{y} =
\boldsymbol{H}\left(\boldsymbol{u}; \boldsymbol{\theta} \right)
+ \boldsymbol{\varepsilon},
$$ (eq:inv_prob_continuous)

where $\boldsymbol{y}$ are the observations and $\boldsymbol{u}$ is the quantity of interested, i.e. the state.
We can assume that each of the fields (state, obs.) live on some domain, i.e. the state, $\boldsymbol{u}:\Omega_{state}\times\mathcal{T}_{state}\rightarrow \mathbb{R}^{D_u}$, and the $\boldsymbol{y}:\Omega_{obs}\times\mathcal{T}_{obs}\rightarrow \mathbb{R}^{D_y}$
For the purposes of this section, we don't need to specify

---

## State Estimation

The first way to approach this problem is to tackle by directly trying to estimate the state.
We are interested in the state, $\mathbf{u}$, given the observations, $\mathbf{y}$.
In Bayesian speak, this is known as the posterior, $p(\mathbf{u}|\mathbf{y})$.
So we can write this using the Bayesian formulation:

$$
p(\boldsymbol{u}|\boldsymbol{y}) = \frac{p(\boldsymbol{y}|\boldsymbol{u})p(\boldsymbol{u})}{p(\boldsymbol{y})}
\propto p(\boldsymbol{y}|\boldsymbol{u})p(\boldsymbol{u})
$$

We see that the posterior depends upon a conditional distribution which describes the likelihood of the observations, $\boldsymbol{y}$, given some state, $\boldsymbol{u}$.
This mapping can be described by the observation operator that we prescribed in equation {eq}`eq:inv_prob_continuous`.
We also have a prior distribution for the state, $\boldsymbol{u}$.
The problem is ill-posed so this prior is arguably the most important quantity within the Bayesian formulation so that we get sensible solutions.
There is a lot of subsequent research that focuses on this portion of the Bayesian formulation.
Typically, we can simplify this full Bayesian formulation to ignore the denominator which should be constant under certain conditions.
This is shown on the far RHS of the equation.

So the name of the game is to estimate the state, $\boldsymbol{u}$, given some objective function which may depend upon some external parameters, $\boldsymbol{\theta}$.
We can express this as a minimization problem where we find some state, $\mathbf{u}^*$, given an objective function, $\boldsymbol{J}(\mathbf{u};\boldsymbol{\theta})$.
This is given by

$$
\begin{aligned}
\boldsymbol{u}^* &=
\underset{\boldsymbol{u}}{\text{argmin}}
\hspace{2mm}
\boldsymbol{L}(\boldsymbol{u};\boldsymbol{\theta}) +
\boldsymbol{R}(\boldsymbol{u};\boldsymbol{\theta}) \\
&= \underset{\boldsymbol{u}}{\text{argmin}} \hspace{2mm}\boldsymbol{J}(\boldsymbol{u};\boldsymbol{\theta})
\end{aligned}
$$

where $\boldsymbol{J}(\mathbf{u};\boldsymbol{\theta})$ is objective function which is the additive combination of the likelihood term $\boldsymbol{L}(\boldsymbol{u};\boldsymbol{\theta})$ and the prior term $\boldsymbol{R}(\boldsymbol{u};\boldsymbol{\theta})$ seen in equation {eq}`eq:inv_prob_objective`.
So now, most of the decisions about how we embed prior information into our formulation will be by including priot

For more details, please see the following section about [state estimation](./state_est.md).

---

## Parameter Estimation

This is more of a *model-driven* approach.
The thinking is to have a (parameterized) model that is able to generate an approximate state that is similar to what we can observe.
First we imagine that we can estimate the state, $\boldsymbol{u}$, using some parameterized function.

$$
\boldsymbol{u}(\vec{\mathbf{x}}, t) \approx \boldsymbol{u}_{\boldsymbol{\theta}}(\vec{\mathbf{x}},t)
$$ (eq:field_parameterized)

The parameters, $\boldsymbol{\theta}$, are generic from some model, $\mathcal{M}$, could be a function or it could be parameters described by a PDE.
Secondly, we assume that the estimated state can be used as an input to the operational operator, $\boldsymbol{H}$.
Now, we can modify equation {eq}`eq:inv_prob_continuous` to include this parameterized field to be:

$$
\boldsymbol{y} = \boldsymbol{H}(\boldsymbol{u_{\boldsymbol{\theta}}};\boldsymbol{\theta}) + \varepsilon_n, \hspace{10mm} \varepsilon\sim\mathcal{N}(0,\sigma^2)
$$ (eq:inv_prob_continuous_parameterized)


Noticed that the only difference between equation {eq}`eq:inv_prob_continuous_parameterized` and {eq}`eq:inv_prob_continuous` is that the state is generated from a parameterized model, $\mathcal{M}$. We have essentially "amortized" the problem which removes the need for the inversion and directly estimate the state by having a function that directly matches the observations given some input parameters, $\boldsymbol{\theta}$.
<!-- Another way to look at it as

$$
\boldsymbol{q}(\boldsymbol{y}|\boldsymbol{\theta})
$$ -->

So given some dataset

$$
\mathcal{D}= \left\{ \boldsymbol{y}_{n}, \boldsymbol{u}_n  \right\}_{n=1}^N
$$

either historical or sufficient for the problem setting, we can estimate the parameters, $\boldsymbol{\theta}$, of the model, $\mathcal{M}$, from the data, $\mathcal{D}$.

$$
p(\boldsymbol{\theta}|\mathcal{D}) = \frac{p(\mathcal{D}|\boldsymbol{\theta})p(\boldsymbol{\theta})}{p(\mathcal{D})}
$$

So the name of the game is to approximate the true state, $\mathbf{u}$, using a model, $\mathbf{u}_{\boldsymbol{\theta}}$, and then minimize the data likelihood

$$
\begin{aligned}
\boldsymbol{\theta}^* &=
\underset{\boldsymbol{\theta}}{\text{argmax}} \hspace{2mm}
\mathcal{L}(\boldsymbol{\theta};\boldsymbol{u}_{\boldsymbol{\theta}})
\end{aligned}
$$

Some example parameterizations of the state, $\mathbf{u}_{\boldsymbol \theta}$:

* ODE/PDE
* Neural Field
* Hybrid Model (Parameterization)


Most of the prior information that can be embedded within the problem:

* Data, e.g. Historical observations
* State Representation / Architectures, e.g. coords --> NerFs, discretized --> CNN
* Loss, e.g. physics-informed, gradient-based





---

## Bi-Level Optimization

Notice that we keep the parameters, $\boldsymbol{\theta}$, within the minimization problem in equation.

A big question is how does one define the best parameters given the minimization problem.


$$
\begin{aligned}
\boldsymbol{\theta}^* &=
\underset{\boldsymbol{\theta}}{\text{argmax}} \hspace{2mm}
\mathcal{L}(\boldsymbol{\theta}) \\
\mathbf{u}^*(\boldsymbol{\theta}) &=
\underset{\mathbf{u}}{\text{argmax}} \hspace{2mm}
\mathcal{J}(\mathbf{u};\boldsymbol{\theta})
\end{aligned}
$$
