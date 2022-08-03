# Variational Inference

---
## Motivations

Variational inference is the most scalable inference method the machine learning community has (as of 2019).

Ultimately, we are interested in approximating the marginal distribution of our data, $\mathcal{X}$.

$$
\mathbf{x} \in \mathcal{X}\sim \mathbb{P}_*
$$

We write some sort of approximation of the *true* (or best) underlying distribution via some parameterized form like so

$$
p_*(\mathbf{x}) \approx p_{\boldsymbol \theta}(\mathbf{x}).
$$

However, in order to obtain this, we need to assume some latent variable, $\mathbf{z}$, plays a role in estimating the underlying density. In the simplest form, we assume a generative model for the joint distribution can be written as

$$
p_\theta(z, x) = p(x|z)p(z)
$$

When fitting a model, we are interested in maximizing the marginal likelihood

$$
p_\theta(x) = \int p_\theta(x|z)p_\theta(z)dz
$$

However, this quantity is intractable because we have a non-linear function thats within an integral.  So we use an variational distribution, $q_\phi(z|x)$, (sometimes called an encoder).

$$
\log p_\theta(x) = \mathbb{E}_{q_\phi(z|x)}\left[ \log p_\theta(x) \right]
$$

---
 ## Pros and Cons

 > These were taken from the slides of Shakir Mohamed (prob methods MLSS 2019)

 ### Why Variational Inference?

 * Applicable to all probabilistic models
 * Transforms a problem from integration to one of optimization
 * Convergence assessment
 * Principled and Scalable approach to model selection
 * Compact representation of posterior distribution
 * Faster to converge
 * Numerically stable
 * Modern Computing Architectures (GPUs)
 * **There is a LOT of research already**!

### Why Not Variational Inference?

* Approximate posterior only
* Difficulty in optimization due to local minima
* Under-estimates the variance of posterior
* Limited theory and guarantees for variational mehtods


---
## Variational Distribution

We defined the variationa distribution as $q(z|x)$. However, we have many types of variational distributions we can impose. For example, we have some of the following:

* *Gaussian*, $q(z)$
* *Mixture Distribution*, $\sum_{k}^{K}\pi_k \mathbb{P}$
* *Bijective Transform* (Flow), $q(z|\tilde{z})$
* *Stochastic Transform* (Encoder, Amortized), $q(z|x)$
* *Conditional*, $q(z|x,y)$

Below we will go through each of them and outline some potential strengths and weaknesses of each of the methods.

---
### Simple, $q(z)$

This is the simplest case where we often assume a very simple distribution can describe the distribution.

$$
q(z) = \mathcal{N}(z|\boldsymbol{\mu_\theta},\boldsymbol{\Sigma_\theta})
$$

If we take each of the Gaussian parameters as full matrices, we end up with:

$$
\boldsymbol{\mu_\theta}:=\boldsymbol{\mu} \in \mathbb{R}^D, \hspace{5mm} \boldsymbol{\Sigma_\theta}:=\boldsymbol{\Sigma} \in \mathbb{R}^{D\times D};
$$

For very high dimensional problems, these are a lot of parameters to learn. Now, we can have various simplifications (or complications) with this. For example, we can simplify the mean, $\boldsymbol{\mu}$, to be zero. The majority of the changes will come from the covariance. Here are a few modifications.


**Full Covariance**

This is when we parameterize our covariance to be a full covariance matrix. $\boldsymbol{\Sigma_\theta} := \boldsymbol{\Sigma}$. This is easily the most expensive and the most complex of the Gaussian types.

**Lower Cholesky**

We can also parameterize our covariance to be a lower triangular matrix, i.e. $\boldsymbol{\Sigma_\theta} := \mathbf{L}$, that satisfies the cholesky decomposition, i.e. $\mathbf{LL}^\top = \boldsymbol{\Sigma}$. This reduces the number of parameters of the full covariance by a factor. It also has desireable properties when parameterizing covariance matrices that are computationally attractive, e.g. positive definite.

**Diagonal Covariance**

We can parameterize our covariance matrix to be a diagonal, i.e. $\boldsymbol{\Sigma_\theta} := \text{diag}(\boldsymbol{\sigma})$. This is a very drastic simplification of our model which limits the expressivity. However, there are immense computational benefits For example, a d-dimensional multivariate Gaussian rv with a mean and a diagonal covariance is the same as the product of $d$ univeriate Gaussians.

$$
q(z) = \mathcal{N}\left(\boldsymbol{\mu_\theta}, \text{diag}(\boldsymbol{\sigma_\theta})\right) = \prod_{d}^D \mathcal{N}(\mu_d, \sigma_d )
$$

This is also known as the **mean-field** approximation and it is a very common starting point in practical VI algorithms.

**Low Rank Multivariate Normal**

Another parameterization is a low rank matrix with a diagonal matrix, i.e. $\boldsymbol{\Sigma_\theta} := \mathbf{W}\mathbf{W}^\top + \mathbf{D}$ where $\mathbf{W} \in \mathbb{R}^{D\times d}, \mathbf{D} \in \mathbb{R}^{D\times D}$. We assume that our parameterization can be low dimensional which might be appropriate for some applications. This allows for some computationally efficient schemes that make use of the Woodbury Identity and the matrix determinant lemma.

**Orthogonal Decoupled**

One interesting approach is to map the variational parameters via a subspace parameterization. For exaple, we can define the mean and variance like so:

$$
\begin{aligned}
\boldsymbol{\mu_\theta} &= \boldsymbol{\Psi}_{\boldsymbol{\mu}} \mathbf{a} \\
\boldsymbol{\Sigma_\theta} &= \boldsymbol{\Psi}_{\boldsymbol{\Sigma}} \mathbf{A} \boldsymbol{\Psi}_{\boldsymbol{\Sigma}}^\top + \mathbf{I}
\end{aligned}
$$

This is a bit of a spin off of the Low-Rank Multivariate Normal approach. However, this method takes care and provides a low-rank method for both the mean and the covariance. They argue that we would be able to put more computational effort in the mean function (computationally easy) and less computational effort for the covariance (computationally intensive).

*Source*: [Orthogonally Decoupled Variational Gaussian Process]() - Salimbeni et al (2018)



**Delta Distribution**

This is probably the distribution with the least amount of parameters. We set the covariance matrix to $0$, i.e. $\boldsymbol{\Sigma_\theta}:=\mathbf{0}$, and we let all of the mass rest on mean points, $\boldsymbol{\mu_\theta}:=\boldsymbol{\mu}=\mathbf{u}$.

$$
q(z) = \delta(z - \hat{z})
$$

---
### Mixture Distribution

The principal behind this is that a simple base distribution, e.g. Gaussian, is not expressive enough. However, a mixture of simple distributions, e.g. Mixture of Gaussians, will be more expressive. So the idea is to choose simple base distribution and replicate it $k$ times. Then, we then do a normalized weighted summation of each component to produce our mixture distribution.

$$
q(z) = \sum_{k}^K\pi_k \mathbb{P}_k
$$

where $0 \leq \pi_k \leq 1$ and $\sum_{k}^K\pi_k=1$. For example, we can use a Gaussian distribution

$$
p_\theta(z) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})
$$

where $\theta = \{\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k \}_k^K$ are potentially learned parameters.. And the mixture distribution will be

$$
q_{\boldsymbol \theta}(\mathbf{z}) = \sum_{k}^K \pi_k \mathcal{N}(\mathbf{z} |\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

Again, we are free to parameterize the covariances as flexible or restrictive as possible. For example we can have full, cholesky, low-rank or diagonal. In addition we can *tie* some of the parameters together. For example, we can have the same covariance matrix for every $k^\text{th}$ component, e.g. $\boldsymbol{\Sigma}_k=\boldsymbol{\Sigma}$. Even for VAEs, this becomes a prior distribution which has noticable improvement over the standard Gaussian prior.

**Note**: in principal, a mixture distribution is very powerful and has the ability to estimate any distribution, e.g. univariate with enough components. However, like with most problems, the issue is estimating the best parameters just from observations.


---
### Bijective Transformation (Flow)

It may be that the variational distribution, $q$, is not sufficiently expressive enough even with the complex Gaussian parameterization and/or the mixture distribution. So another option is to use a bijective transformation to map the data from a simple base distribution, e.g. Gaussian, to a more complex distribution for our variational parameter, $z$.

$$
\mathbf{z} = \boldsymbol{T_\phi}(\tilde{\mathbf{z}})
$$

We hope that the resulting variational distribution, $q(z)$, acts a better approximation to the data. Because our transformation is bijective, we can

variational parameter, $z$, to a simple base distribution st we ha
$$
q(z) = p_e(\tilde{z})|\boldsymbol{\nabla}_\mathbf{z}\boldsymbol{T_\phi}^{-1}(\mathbf{z})|
$$

where $|\boldsymbol{\nabla}_\mathbf{z} \cdot|$ is the determinant Jacobian of the transformation, $\boldsymbol{T_\phi}$.

---
### Stochastic Transformation (Encoder, Amortization)

Another type of transformation is a stochastic transformation. This is given by $q(z|x)$. In this case, we assume some non-linear. For example, a Gaussian distribution with a parameterized mean and variance via neural networks

$$
q(\mathbf{z}|\mathbf{x}) = \mathcal{N}\left(\boldsymbol{\mu_\phi}(\mathbf{x}), \boldsymbol{\sigma_\phi}(\mathbf{x})\right)
$$

or more appropriately

$$
q(\mathbf{z}|\mathbf{x}) = \mathcal{N}\left(\boldsymbol{\mu}, \text{diag}(\exp (\boldsymbol{\sigma}^2_{\log}) )\right), \hspace{4mm} (\boldsymbol{\mu}, \boldsymbol{\sigma}^2_{\log}) = \text{NN}_{\boldsymbol \theta}(\mathbf{x})
$$

It can be very difficult to try and have a variational distribution that is complicated enough to cover the whole posterior. So often, we use a variational distribution that is conditioned on the observations, i.e. $q(z|x)$. This is known as an encoder because we encode the observations to obey th




---
## ELBO (Encoder) - Derivation

> This derivation comes from the book by Probabilistic Machine Learning by Kevin Murphy. I find it to be a much better and intuitive derivation.

**Note**: I put the *encoder* tag in the title. This is because there are other ELBOs that have different purposes, for example, variational distributions without an encoder and also an encoder for conditional likelihoods. In this first one, we will like at the ELBO derivation


As mentioned above, we are interested in expanding the expectation of the marginal likelihood wrt the encoder variational distribution

$$
\log p_\theta(x) = \mathbb{E}_{q_\phi(z|x)}\left[ \log p_\theta(x) \right]
$$

We will do a bit of mathematical manipulation to expand this expectation. Firstly, we will start with Bayes rule:

$$
p_\theta(x) = \frac{p_\theta(z,x)}{p_\theta(z|x)}
$$

Plugging this into our expectation gives us:

$$
\log p_\theta(x) = \mathbb{E}_{q_\phi(z|x)}\left[ \log \frac{p_\theta(z,x)}{p_\theta(z|x)} \right]
$$

Now we will do the identity trick (multiply by $\frac{1}{1}$ :) ) within the log term to incorporate the variational distribution, $q_\phi$.

$$
\log p_\theta(x) = \mathbb{E}_{q_\phi(z|x)}\left[ \log \frac{p_\theta(z,x)q_\phi(z|x)}{p_\theta(z|x)q_\phi(z|x)} \right] = \mathbb{E}_{q_\phi(z|x)}\left[ \log \frac{p_\theta(z,x)q_\phi(z|x)}{q_\phi(z|x)p_\theta(z|x)} \right]
$$

Using the log rules, we can split this fraction into two fractions;

$$
\log p_\theta(x) =  \mathbb{E}_{q_\phi(z|x)}\left[ \log \frac{p_\theta(z,x)}{q_\phi(z|x)} + \log \frac{q_\phi(z|x)}{p_\theta(z|x)} \right]
$$

Now, we can expand the expectation term across the additive operator

$$
\log p_\theta(x) =  \mathbb{E}_{q_\phi(z|x)}\left[ \log \frac{p_\theta(z,x)}{q_\phi(z|x)} \right] + \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{q_\phi(z|x)}{p_\theta(z|x)} \right]
$$

Here, we notice that the second term is actually the Kullback-Leibler divergence term.

$$
\text{D}_{\text{KL}} [Q||P] = \mathbb{E}_Q\left[\log \frac{Q}{P} \right] = - \mathbb{E}_Q\left[\log \frac{P}{Q} \right]
$$

so we can replace this with the more compact form.

$$
\log p_\theta(x) =  \mathbb{E}_{q_\phi(z|x)}\left[ \log \frac{p_\theta(z,x)}{q_\phi(z|x)} \right] + \text{D}_{\text{KL}} \left[q_\phi(z|x)||p_\theta(z|x) \right]
$$

We know from theory that the KL divergence term is always zero or positive. So this means that we can draw a bound on the first term in terms of the marginal log-likelihood.

$$
 \mathcal{L}_{\text{ELBO}}:=\mathbb{E}_{q_\phi(z|x)}\left[ \log \frac{p_\theta(z,x)}{q_\phi(z|x)} \right] \leq \log p_\theta(x)
$$

This term is called the Evidence Lower Bound (ELBO). So the objective is to *maximize* this term which will also minimize the KLD.

$$
 \mathcal{L}_{\text{ELBO}}=\mathbb{E}_{q_\phi(z|x)}\left[ \log \frac{p_\theta(z,x)}{q_\phi(z|x)} \right]
$$

So now, we can expand the joint distribution using Bayes rule, i.e. $p(z,x)=p(x|z)p(z)$, to give us.

$$
 \mathcal{L}_{\text{ELBO}}=\mathbb{E}_{q_\phi(z|x)}\left[ \log \frac{p(x|z)p(z)}{q_\phi(z|x)} \right]
$$

We can also expand this fraction using the log rules,

$$
 \mathcal{L}_{\text{ELBO}}=\mathbb{E}_{q_\phi(z|x)}\left[ \log p(x|z) + \log p(z) - \log q_\phi(z|x) \right].
$$

where:

* $q_\phi(z|x)$ - encoder network
* $p_\theta(x|z)$ - decoder network
* $p_\theta(z)$ - prior network

Now, we have some options on how we can group the likelihood, the prior and the variational distribution together and each of them will offer a slightly different interpretation and application.

---
## Reconstruction Loss

If we group the prior probability and the variational distribution together, we get:

$$
 \mathcal{L}_{\text{ELBO}}=\mathbb{E}_{q_\phi(z|x)}\left[ \log p(x|z) \right] + \mathbb{E}_{q_\phi(z|x)}\left[ \log \frac{p(z)}{q_\phi(z|x)} \right].
$$

This is the same KLD term as before but in the reverse order. So with a slight of hand in terms of the signs, we can rearrange the term to be

$$
 \mathcal{L}_{\text{ELBO}}= \mathbb{E}_{q_\phi(z|x)}\left[ \log p(x|z) \right] - \mathbb{E}_{q_\phi(z|x)}\left[ \log \frac{q_\phi(z|x)} {p(z)}\right].
$$

---
**Proof**:

$$
\mathbb{E}_q[ \log p - \log q] = - \mathbb{E}_q[\log q - \log p] = - \mathbb{E}_q[\log\frac{q}{p}]
$$

**QED**.

---

So now, we have the exact same KLD term as before. So let's use the simplified form.

$$
 \mathcal{L}_{\text{ELBO}}={\color{blue}\mathbb{E}_{q_\phi(z|x)}\left[ \log p(x|z) \right]} - {\color{green}\text{D}_\text{KL}\left[q_\phi(z|x)||p(z)\right]}.
$$


where:
* ${\color{blue}\mathbb{E}_{q_\phi(z|x)}\left[ \log p(x|z) \right]}$ - is the $\color{blue}\text{reconstruction loss}$.
* ${\color{green}\text{D}_\text{KL}\left[q_\phi(z|x)||p(z)\right]}$ - is the complexity, i.e. the $\color{green}\text{KL divergence}$ (a distance metric) between the prior and the variational distribution.

This is easily the most common ELBO term especially with Variational AutoEncoders (VAEs). The first term is the expectation of the likelihood term wrt the variational distribution. The second term is the KLD between the prior and the variational distribution.


---
## Volume Correction

Another approach is more along the lines of the transform distribution. Assume we have our original data domain $\mathcal{X}$ and we have some stochastic transformation, p(z|x), which transforms the data from our original domain to a transform domain, $\mathcal{Z}$.

$$
z \sim p(z|x)
$$

To acquire this, let's look at the equation again

$$
 \mathcal{L}_{\text{ELBO}}=\mathbb{E}_{q_\phi(z|x)}\left[ \log p(x|z) + \log p(z) - \log q_\phi(z|x) \right].
$$

except this time we will isolate the prior and combine the likelihood and the variational distribution.

$$
 \mathcal{L}_{\text{ELBO}}={\color{blue}\mathbb{E}_{q_\phi(z|x)}\left[ \log p(z) \right]} + {\color{green}\mathbb{E}_{q_\phi(z|x)}\left[ \log \frac{p(x|z)}{q_\phi(z|x)} \right]}.
$$


where:

* ${\color{blue}\mathbb{E}_{q_\phi(z|x)}\left[ \log p(z) \right]}$ - is the expectation of the transformed distribution, aka the ${\color{blue}\text{reparameterized probability}}$.
* ${\color{green}\mathbb{E}_{q_\phi(z|x)}\left[ \log \frac{p(x|z)}{q_\phi(z|x)} \right]}$ - is the ratio between the inverse transform and the forward transform , i.e. ${\color{green}\text{Volume Correction Factor}}$ or *likelihood contribution*.


**Source**: I first saw this approach in the SurVAE Flows paper.

---
## Variational Free Energy (VFE)

There is one more main derivation that remains (that's often seen in the literature). Looking at the equation again

$$
 \mathcal{L}_{\text{ELBO}}=\mathbb{E}_{q_\phi(z|x)}\left[ \log p(x|z) + \log p(z) - \log q_\phi(z|x) \right],
$$

we now isolate the likelihood *and* the prior under the variational expectation. This gives us:

$$
 \mathcal{L}_{\text{ELBO}}={\color{blue}\mathbb{E}_{q_\phi(z|x)}\left[ \log p(x|z) p(z)\right]} - {\color{green} \mathbb{E}_{q_\phi(z|x)}\left[ \log q_\phi(z|x) \right]}.
$$

where:

* ${\color{blue}\mathbb{E}_{q_\phi(z|x)}\left[ \log p(x|z) p(z)\right]}$ - is the ${\color{blue}\text{energy}}$ function
* ${\color{green} \mathbb{E}_{q_\phi(z|x)}\left[ \log q_\phi(z|x) \right]}$ - is the ${\color{green}\text{entropy}}$


**Source**: I see this approach a lot in the Gaussian process literature when they are deriving the Sparse Gaussian Process from Titsias.

---
## ELBO (Non-Encoder) - Derivation

In all of these formulas, we have an *encoder* as our variational distribution, i.e. $q(z|x)$, which seeks to amortize the inference. Sometimes this is not necessary and we can find a complicated enough variational distribution, i.e. $q(z)$. This often happens in very simple models, e.g. $y = \mathbf{Wx} + \mathbf{b} + \epsilon$

So this will be a similar derivation as the above, however we will


$$
\log p_\theta(x) = \mathbb{E}_{q_\phi(z)}\left[ \log p_\theta(x) \right]
$$

I am going to make some grandiose assumptions and skip ahead of the derivation. But I think it might be useful to think ahead and then work my backwards.

---
**Reconstruction Loss**

This is the easiest term to show because it shows up in many simpler applications when we have very simple models and we believe that

$$
 \mathcal{L}_{\text{ELBO}}={\color{blue}\mathbb{E}_{q_\phi(z)}\left[ \log p(x|z) \right]} - {\color{green}\text{D}_\text{KL}\left[q_\phi(z)||p(z)\right]}.
$$

---
**Volume Correction**

This doesn't actually work because **we need** a transformation from $x$ to $z$.


---
**Variational Free Energy**

$$
 \mathcal{L}_{\text{ELBO}}={\color{blue}\mathbb{E}_{q_\phi(z)}\left[ \log p(x|z) p(z)\right]} - {\color{green} \mathbb{E}_{q_\phi(z)}\left[ \log q_\phi(z) \right]}.
$$


---
## ELBO - Derivation (Old)

> This is my own derivation from a few years ago. I have improved it (see above) but I keep it here for reference. :)

Let's start with the marginal likelihood function.

$$\mathcal{P}(y| \theta)=\int_\mathcal{X} \mathcal{P}(y|\mathbf x, \theta) \cdot \mathcal{P}(\mathbf x) \cdot d\mathbf{x}$$

where we have effectively marginalized out the $f$'s. We already know that it's difficult to propagate the $\mathbf x$'s through the nonlinear functions $\mathbf K^{-1}$ and $|$det $\mathbf K|$ (see previous doc for examples). So using the VI strategy, we introduce a new variational distribution $q(\mathbf x)$ to approximate the posterior distribution $\mathcal{P}(\mathbf x| y)$. The distribution is normally chosen to be Gaussian:

$$q(\mathbf x) = \prod_{i=1}^{N}\mathcal{N}(\mathbf x|\mathbf \mu_z, \mathbf \Sigma_z)$$

So at this point, we aree interested in trying to find a way to measure the difference between the approximate distribution $q(\mathbf x)$ and the true posterior distribution $\mathcal{P} (\mathbf x)$. Using some algebra, let's take the log of the marginal likelihood (evidence):

$$\log \mathcal{P}(y|\theta) = \log \int_\mathcal{X} \mathcal{P}(y|\mathbf x, \theta) \cdot \mathcal{P}(\mathbf x) \cdot d\mathbf x$$

So now we are going to use the some tricks that you see within almost every derivation of the VI framework. The first one consists of using the Identity trick. This allows us to change the expectation to incorporate the new variational distribution $q(\mathbf x)$. We get the following equation:

$$\log \mathcal{P}(y|\theta) = \log \int_\mathcal{X} \mathcal{P}(y|\mathbf x, \theta) \cdot \mathcal{P}(\mathbf x) \cdot \frac{q(\mathbf x)}{q(\mathbf x)} \cdot d\mathbf x$$

Now that we have introduced our new variational distribution, we can regroup and reweight our expectation. Because I know what I want, I get the following:

$$\log \mathcal{P}(y|\theta) = \log \int_\mathcal{X} \mathcal{P}(y|\mathbf x, \theta) \cdot  q(\mathbf x) \cdot \frac{\mathcal{P}(\mathbf x)}{q(\mathbf x)} \cdot d\mathbf x$$

Now with Jensen's inequality, we have the relationship $f(\mathbb{E}[x]) \leq \mathbb{E} [f(x)]$. We would like to put the $\log$ function inside of the integral. Jensen's inequality allows us to do this. If we let $f(\cdot)= \log(\cdot)$ then we get the Jensen's equality for a concave function, $f(\mathbb{E}[x]) \geq \mathbb{E} [f(x)]$. In this case if we match the terms to each component to the inequality, we have

$$\log \cdot \mathbb{E}_\mathcal{q(\mathbf x)} \left[ \mathcal{P}(y|\mathbf x, \theta) \cdot  \frac{\mathcal{P}(\mathbf x)}{q(\mathbf x)} \right]
\geq
\mathbb{E}_\mathcal{q(\mathbf x)} \left[\log  \mathcal{P}(y|\mathbf x, \theta) \cdot \frac{\mathcal{P}(\mathbf x)}{q(\mathbf x)} \right]$$

So now finally we have both terms in the inequality. Summarizing everything we have the following relationship:

$$log \mathcal{P}(y|\theta) = \log \int_\mathcal{X} \mathcal{P}(y|\mathbf x, \theta) \cdot  q(\mathbf x) \cdot \frac{\mathcal{P}(\mathbf x)}{q(\mathbf x)} \cdot d\mathbf x $$

$$
\log \mathcal{P}(y|\theta) \geq  \int_\mathcal{X} \left[\log  \mathcal{P}(y|\mathbf x, \theta) \cdot \frac{\mathcal{P}(\mathbf x)}{q(\mathbf x)} \right] q(\mathbf x) \cdot d\mathbf x
$$

I'm going to switch up the terminology just to make it easier aesthetically. I'm going to let $\mathcal{L}(\theta)$ be $\log \mathcal{P}(y|\theta)$ and $\mathcal{F}(q, \theta) \leq \mathcal{L}(\theta)$. So basically:

$$
\mathcal{L}(\theta) =\log \mathcal{P}(y|\theta) \geq  \int_\mathcal{X} \left[\log  \mathcal{P}(y|\mathbf x, \theta) \cdot \frac{\mathcal{P}(\mathbf x)}{q(\mathbf x)} \right] q(\mathbf x) \cdot d\mathbf x = \mathcal{F}(q, \theta)
$$

With this simple change I can talk about each of the parts individually. Now using log rules we can break apart the likelihood and the quotient. The quotient will be needed for the KL divergence.

$$
\mathcal{F}(q) =
\underbrace{\int_\mathcal{X} q(\mathbf x) \cdot \log  \mathcal{P}(y|\mathbf x, \theta) \cdot d\mathbf x}_{\mathbb{E}_{q(\mathbf{x})}} +
\underbrace{\int_\mathcal{X} q(\mathbf x) \log  \frac{\mathcal{P}(\mathbf x)}{q(\mathbf x)}   \cdot d\mathbf x}_{\text{KL}}
$$



The punchline of this (after many calculated manipulations), is that we obtain an optimization equation $\mathcal{F}(\theta)$:

$$\mathcal{F}(q)=\mathbb{E}_{q(\mathbf x)}\left[ \log \mathcal{P}(y|\mathbf x, \theta) \right] - \text{D}_\text{KL}\left[ q(\mathbf x) || \mathcal{P}(\mathbf x) \right]$$

where:

* Approximate posterior distribution: $q(x)$
  * The best match to the true posterior $\mathcal{P}(y|\mathbf x, \theta)$. This is what we want to calculate.
* Reconstruction Cost: $\mathbb{E}_{q(\mathbf x)}\left[ \log \mathcal{P}(y|\mathbf x, \theta) \right]$
  * The expected log-likelihood measure of how well the samples from $q(x)$ are able to explain the data $y$.
* Penalty: $\text{D}_\text{KL}\left[ q(\mathbf x) || \mathcal{P}(\mathbf x) \right]$
  * Ensures that the explanation of the data $q(x)$ doesn't deviate too far from your beliefs $\mathcal{P}(x)$. (Okham's razor constraint)

**Source**: [VI Tutorial](https://www.shakirm.com/papers/VITutorial.pdf) - Shakir Mohamed

If we optimize $\mathcal{F}$ with respect to $q(\mathbf x)$, the KL is minimized and we just get the likelihood. As we've seen before, the likelihood term is still problematic as it still has the nonlinear portion to propagate the $\mathbf x$'s through. So that's nothing new and we've done nothing useful. If we introduce some special structure in $q(f)$ by introducing sparsity, then we can achieve something useful with this formulation.
 But through augmentation of the variable space with $\mathbf u$ and $\mathbf Z$ we can bypass this problem. The second term is simple to calculate because they're both chosen to be Gaussian.

 ### Comments on $q(x)$

* We have now transformed our problem from an integration problem to an optimization problem where we optimize for $q(x)$ directly.
* Many people tend to simplify $q$ but we could easily write some dependencies on the data for example $q(x|\mathcal{D})$.
* We can easily see the convergence as we just have to wait until the loss (free energy) reaches convergence.
* Typically $q(x)$ is a Gaussian whereby the variational parameters are the mean and the variance. Practically speaking, we could freeze or unfreeze any of these parameters if we have some prior knowledge about our problem.
* Many people say 'tighten the bound' but they really just mean optimization: modifying the hyperparameters so that we get as close as possible to the true marginal likelihood.



---
## Resources

* Tutorial Series - [Why?](https://chrisorm.github.io/VI-Why.html) | [ELBO](https://chrisorm.github.io/VI-ELBO.html) | [MC ELBO](https://chrisorm.github.io/VI-MC.html) | [Reparameterization](https://chrisorm.github.io/VI-reparam.html) | [MC ELBO unBias](https://chrisorm.github.io/VI-ELBO-MC-approx.html) | [MC ELBO PyTorch](https://chrisorm.github.io/VI-MC-PYT.html) | [Talk](https://chrisorm.github.io/pydata-2018.html)
* Blog Posts: Neural Variational Inference
  * [Classical Theory](http://artem.sobolev.name/posts/2016-07-01-neural-variational-inference-classical-theory.html)
  * [Scaling Up](http://artem.sobolev.name/posts/2016-07-04-neural-variational-inference-stochastic-variational-inference.html)
  * [BlackBox Mode](http://artem.sobolev.name/posts/2016-07-05-neural-variational-inference-blackbox.html)
  * [VAEs and Helmholtz Machines](http://artem.sobolev.name/posts/2016-07-11-neural-variational-inference-variational-autoencoders-and-Helmholtz-machines.html)
  * [Importance Weighted AEs](http://artem.sobolev.name/posts/2016-07-14-neural-variational-importance-weighted-autoencoders.html)
  * [Neural Samplers and Hierarchical VI](http://artem.sobolev.name/posts/2019-04-26-neural-samplers-and-hierarchical-variational-inference.html)
  * [Importance Weighted Hierarchical VI](http://artem.sobolev.name/posts/2019-05-10-importance-weighted-hierarchical-variational-inference.html) | [Video](https://www.youtube.com/watch?v=pdSu7XfGhHw&feature=youtu.be)

* Normal Approximation to the Posterior Distribution - [blog](http://bjlkeng.github.io/posts/normal-approximations-to-the-posterior-distribution/)

**Lower Bound**

* [Understaing the Variational Lower Bound](http://legacydirs.umiacs.umd.edu/~xyang35/files/understanding-variational-lower.pdf)
* [Deriving the Variational Lower Bound](http://paulrubenstein.co.uk/deriving-the-variational-lower-bound/)

**Summaries**

* [Advances in Variational Inference](https://arxiv.org/pdf/1711.05597.pdf)

**Presentations**

* [VI Shakir](https://www.shakirm.com/papers/VITutorial.pdf)
* Deisenroth - [VI](https://drive.google.com/file/d/1sAIF0rqgNbVbp7ZbuiS7kh96Yns04k1i/view) | [IT](https://drive.google.com/open?id=14WOcbwn011rJbFFsSbeoeuSxY4sMG4KY)
* [Bayesian Non-Parametrics and Priors over functions](https://www.doc.ic.ac.uk/~mpd37/teaching/ml_tutorials/2017-11-22-Ek-BNP-and-priors-over-functions.pdf)
* [here](https://filebox.ece.vt.edu/~s14ece6504/slides/Moran_I_ECE_6504_VB.pdf)


**Reviews**
* [From EM to SVI](http://krasserm.github.io/2018/04/03/variational-inference/)
* [Variational Inference](https://ermongroup.github.io/cs228-notes/inference/variational/)
* [VI- Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf)
* [Tutorial on VI](http://www.robots.ox.ac.uk/~sjrob/Pubs/vbTutorialFinal.pdf)
* [VI w/ Code](https://zhiyzuo.github.io/VI/)
* [VI - Mean Field](https://blog.evjang.com/2016/08/variational-bayes.html)
* [VI Tutorial](https://github.com/philschulz/VITutorial)
* GMM
  * [VI in GMM](https://github.com/bertini36/GMM)
  * [GMM Pyro](https://mattdickenson.com/2018/11/18/gmm-python-pyro/) | [Pyro](http://pyro.ai/examples/gmm.html)
  * [GMM PyTorch](https://github.com/ldeecke/gmm-torch) | [PyTorch](https://angusturner.github.io/generative_models/2017/11/03/pytorch-gaussian-mixture-model.html) | [PyTorchy](https://github.com/RomainSabathe/dagmm/blob/master/gmm.py)

**Code**


**Extensions**

* [Neural Samplers and Hierarchical Variational Inference](http://artem.sobolev.name/posts/2019-04-26-neural-samplers-and-hierarchical-variational-inference.html)


#### From Scratch

* Programming a Neural Network from Scratch - Ritchie Vink (2017) - [blog](https://www.ritchievink.com/blog/2017/07/10/programming-a-neural-network-from-scratch/)
* An Introduction to Probability and Computational Bayesian Statistcs - Eric Ma 0[Blog](https://ericmjl.github.io/essays-on-data-science/machine-learning/computational-bayesian-stats/)
* Variational Inference from Scratch - Ritchie Vink (2019) - [blog](https://www.ritchievink.com/blog/2019/09/16/variational-inference-from-scratch/)
* Bayesian inference; How we are able to chase the Posterior - Ritchie Vink (2019) - [blog](Bayesian inference; How we are able to chase the Posterior)
* Algorithm Breakdown: Expectation Maximization - [blog](https://www.ritchievink.com/blog/2019/05/24/algorithm-breakdown-expectation-maximization/)
## Variational Inference

* Variational Bayes and The Mean-Field Approximation - Keng (2017) - [blog](http://bjlkeng.github.io/posts/variational-bayes-and-the-mean-field-approximation/)
* https://www.ritchievink.com/blog/2019/06/10/bayesian-inference-how-we-are-able-to-chase-the-posterior/
* https://www.ritchievink.com/blog/2019/09/16/variational-inference-from-scratch/
